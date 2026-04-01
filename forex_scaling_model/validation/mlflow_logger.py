"""
validation/mlflow_logger.py
============================
MLflow model logger for the Forex Scaling Model.

Logs the following on every promotion event:
  • All model hyperparameters and training config
  • Promotion gate metrics (Sharpe, profit factor, drawdown, etc.)
  • Git commit hash (code version) — embedded in every run
  • Walk-forward fold Sharpe distribution
  • The promotion gate JSON report as an MLflow artifact
  • HTML walk-forward report (if path provided)
  • Model checkpoint .pt file (if path provided)

MLflow server: http://localhost:5000 (configurable via MLFLOW_TRACKING_URI env var)

Falls back to filesystem-only logging (mlruns/) when the server is unreachable.
Falls back to rich print output when mlflow is not installed.
"""

import json
import os
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT", "forex-scaling-model")
FALLBACK_LOG_DIR = Path(os.getenv("MLFLOW_FALLBACK_DIR",
    str(Path(__file__).resolve().parent.parent / "logs" / "mlflow_fallback")))


# ── git helpers ──────────────────────────────────────────────────────────────

def _git_hash(short: bool = True) -> str:
    """Return the current git commit hash, or 'untracked' if not in a repo."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "", "HEAD"]
        cmd = [c for c in cmd if c]   # remove empty string
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "untracked"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """True if there are uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return False


# ── MLflow integration ───────────────────────────────────────────────────────

def _mlflow_available() -> bool:
    try:
        import mlflow
        return True
    except ImportError:
        return False


def _mlflow_server_reachable(uri: str = MLFLOW_URI, timeout: float = 3.0) -> bool:
    if not uri.startswith("http"):
        return True   # local file URI always OK
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(uri + "/health", timeout=timeout)
        return True
    except Exception:
        return False


class MLflowModelLogger:
    """
    Logs model metadata, metrics, and artifacts to MLflow on every promotion.

    If MLflow is not installed or the server is unreachable,
    falls back to saving a JSON file in logs/mlflow_fallback/.
    """

    def __init__(
        self,
        tracking_uri: str  = MLFLOW_URI,
        experiment:   str  = MLFLOW_EXP,
        verbose:      bool = True,
    ):
        self._uri        = tracking_uri
        self._exp        = experiment
        self._verbose    = verbose
        self._use_mlflow = _mlflow_available() and _mlflow_server_reachable(tracking_uri)
        if self._verbose:
            mode = "MLflow" if self._use_mlflow else "filesystem fallback"
            print(f"[MLflowLogger] Mode: {mode}  URI: {self._uri}")

    # ── public API ──────────────────────────────────────────────────────────

    def log_promotion(
        self,
        model_name:        str,
        gate_result:       Dict[str, Any],
        training_config:   Optional[Dict]  = None,
        fold_sharpes:      Optional[List[float]] = None,
        report_html_path:  Optional[str]   = None,
        checkpoint_path:   Optional[str]   = None,
        extra_tags:        Optional[Dict]  = None,
    ) -> str:
        """
        Log a model promotion event.

        Parameters
        ----------
        model_name      : e.g. "haelt_v4"
        gate_result     : Output dict from PromotionGate.evaluate()
        training_config : Dict of hyperparameters (from config/settings.py)
        fold_sharpes    : Sharpe per walk-forward fold
        report_html_path: Path to walk-forward HTML report
        checkpoint_path : Path to .pt checkpoint to log as artifact
        extra_tags      : Additional key-value tags

        Returns
        -------
        run_id : str  (MLflow run ID, or local file path in fallback mode)
        """
        git_hash   = _git_hash()
        git_branch = _git_branch()
        git_dirty  = _git_dirty()
        timestamp  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "model_name":      model_name,
            "git_hash":        git_hash,
            "git_branch":      git_branch,
            "git_dirty":       str(git_dirty),
            "promoted":        str(gate_result.get("promoted", False)),
            "emergency_mode":  str(gate_result.get("emergency_mode", False)),
            "timestamp":       timestamp,
        }
        if training_config:
            for k, v in training_config.items():
                if isinstance(v, (str, int, float, bool)):
                    params[f"cfg_{k}"] = str(v)

        metrics = {k: float(v) for k, v in gate_result.get("details", {}).items()
                   if isinstance(v, (int, float))}
        if fold_sharpes:
            import math
            metrics["fold_sharpe_mean"] = sum(fold_sharpes) / len(fold_sharpes)
            metrics["fold_sharpe_std"]  = math.sqrt(
                sum((s - metrics["fold_sharpe_mean"])**2 for s in fold_sharpes)
                / max(len(fold_sharpes) - 1, 1))
            metrics["fold_sharpe_min"]  = min(fold_sharpes)

        tags = {"git_hash": git_hash, "source": "promotion_gate"}
        if extra_tags:
            tags.update(extra_tags)

        if self._use_mlflow:
            return self._log_to_mlflow(
                params, metrics, tags, gate_result,
                report_html_path, checkpoint_path)
        else:
            return self._log_to_file(
                params, metrics, tags, gate_result,
                report_html_path)

    # ── internal MLflow logging ──────────────────────────────────────────────

    def _log_to_mlflow(
        self, params, metrics, tags, gate_result,
        report_html_path, checkpoint_path,
    ) -> str:
        import mlflow

        mlflow.set_tracking_uri(self._uri)
        mlflow.set_experiment(self._exp)

        with mlflow.start_run(tags=tags) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Gate report as JSON artifact
            import tempfile, json as _json
            with tempfile.NamedTemporaryFile("w", suffix=".json",
                                             delete=False) as f:
                _json.dump(gate_result, f, indent=2, default=str)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="promotion_gate")

            if report_html_path and Path(report_html_path).exists():
                mlflow.log_artifact(report_html_path, artifact_path="reports")

            if checkpoint_path and Path(checkpoint_path).exists():
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

            run_id = run.info.run_id
            if self._verbose:
                promoted = gate_result.get("promoted", False)
                print(f"[MLflowLogger] Run {run_id[:8]} | "
                      f"{'PROMOTED ✅' if promoted else 'LOGGED (not promoted) ❌'} | "
                      f"git:{params['git_hash']}")
            return run_id

    # ── filesystem fallback ──────────────────────────────────────────────────

    def _log_to_file(
        self, params, metrics, tags, gate_result, report_html_path,
    ) -> str:
        FALLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = {
            "timestamp": ts,
            "params":    params,
            "metrics":   metrics,
            "tags":      tags,
            "gate":      gate_result,
        }
        path = FALLBACK_LOG_DIR / f"promotion_{ts}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        if self._verbose:
            promoted = gate_result.get("promoted", False)
            print(f"[MLflowLogger] Fallback log → {path} | "
                  f"{'PROMOTED ✅' if promoted else 'REJECT ❌'} | "
                  f"git:{params['git_hash']}")
        return str(path)

    def patch_html_report(self, html_path: str, git_hash: str = None) -> str:
        """
        Inject the git hash into an existing HTML walk-forward report.
        Returns the (modified) HTML path.
        """
        git_hash = git_hash or _git_hash()
        if not Path(html_path).exists():
            return html_path
        html = Path(html_path).read_text(encoding="utf-8")
        badge = (
            f'<div style="font-family:monospace;font-size:11px;color:#888;'
            f'margin-top:8px">git: {git_hash} | branch: {_git_branch()}</div>'
        )
        html = html.replace("</body>", f"{badge}</body>")
        Path(html_path).write_text(html, encoding="utf-8")
        return html_path


# ── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from validation.promotion_gate import PromotionGate
    gate = PromotionGate()
    result = gate.evaluate(
        sharpe=1.8, profit_factor=1.6, max_drawdown=0.12, n_trades=700,
        regime_pnl={"trending": 0.5, "neutral": 0.3, "mean_rev": 0.2},
        gross_pnl=10_000.0, transaction_costs=1_500.0, n_obs=700,
    )

    logger = MLflowModelLogger()
    run_id = logger.log_promotion(
        model_name="haelt_v4_smoke",
        gate_result=result,
        training_config={"epochs": 100, "lr": 1e-4, "seq_len": 60},
        fold_sharpes=[1.6, 1.8, 1.9, 1.7, 2.0],
    )
    print(f"\n  Run/file: {run_id}")
    print(f"  Git hash: {_git_hash()}")
    print("OK ✓")
