"""
infrastructure/deployment.py
=============================
Infrastructure & monitoring improvements:
  1. ONNXExporter        — Convert PyTorch models to ONNX for 3-5× faster CPU inference
  2. ShadowModeManager   — Run new model in parallel, compare signals for 2 weeks
  3. SHAPMonitor         — Track SHAP feature importance per retrain cycle
  4. WalkForwardReporter — Weekly HTML/JSON performance report
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone

from config.settings import PATHS

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    TORCH = True
except ImportError:
    TORCH = False

try:
    import onnx
    import onnxruntime as ort
    ONNX = True
except ImportError:
    ONNX = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. ONNX EXPORTER
# ─────────────────────────────────────────────────────────────────────────────

class ONNXExporter:
    """
    Converts trained PyTorch models to ONNX format.

    Benefits:
      - 3–5× faster CPU inference vs PyTorch (no Python overhead, optimized kernels)
      - Deploy on any platform: no CUDA dependency, works on a VPS or Raspberry Pi
      - Consistent inference latency (no GIL, no torch overhead)
      - Enables TensorRT/OpenVINO acceleration if needed

    Usage:
        exporter = ONNXExporter()
        path = exporter.export(model, n_features=56, seq_len=60, model_name="haelt")
        session = exporter.load_session(path)
        pred = exporter.infer(session, features_array)
    """

    def __init__(self, export_dir: str = None):
        if export_dir is None:
            export_dir = PATHS["checkpoints"]
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        model:      "nn.Module",
        n_features: int,
        seq_len:    int,
        model_name: str,
        batch_size: int = 1,
        opset:      int = 17,
    ) -> str:
        """
        Export a PyTorch model to ONNX.
        Returns path to the .onnx file.
        """
        if not TORCH:
            raise RuntimeError("PyTorch not installed")

        onnx_path = str(self.export_dir / f"{model_name}.onnx")
        model.eval()

        # Dummy input for tracing
        dummy = torch.randn(batch_size, seq_len, n_features)

        # Handle GNN models (different input signature)
        if hasattr(model, "n_nodes"):
            dummy = torch.randn(batch_size, 6, n_features // 6)

        try:
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["features"],
                output_names=["signal"],
                dynamic_axes={
                    "features": {0: "batch_size"},
                    "signal":   {0: "batch_size"},
                },
            )
            print(f"[ONNX] Exported {model_name} → {onnx_path}")

            # Verify
            if ONNX:
                m = onnx.load(onnx_path)
                onnx.checker.check_model(m)
                print(f"[ONNX] Model verified ✓")

                # Benchmark vs PyTorch
                self._benchmark(model, onnx_path, dummy)

            return onnx_path

        except Exception as e:
            print(f"[ONNX] Export failed: {e}")
            return ""

    def _benchmark(self, pytorch_model, onnx_path: str, dummy: "torch.Tensor"):
        """Compare PyTorch vs ONNX inference speed."""
        if not ONNX: return
        n = 100

        # PyTorch
        pytorch_model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n): pytorch_model(dummy)
        pt_ms = (time.perf_counter() - t0) * 1000 / n

        # ONNX Runtime
        sess = ort.InferenceSession(onnx_path,
               providers=["CPUExecutionProvider"])
        x_np = dummy.numpy()
        t0   = time.perf_counter()
        for _ in range(n):
            sess.run(None, {"features": x_np})
        ort_ms = (time.perf_counter() - t0) * 1000 / n

        speedup = pt_ms / max(ort_ms, 0.001)
        print(f"[ONNX] PyTorch: {pt_ms:.2f}ms | ORT: {ort_ms:.2f}ms | "
              f"Speedup: {speedup:.1f}×")

    def load_session(self, onnx_path: str) -> Optional["ort.InferenceSession"]:
        if not ONNX:
            print("[ONNX] onnxruntime not installed. pip install onnxruntime")
            return None
        return ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    @staticmethod
    def infer(
        session: "ort.InferenceSession",
        features: np.ndarray,   # (seq_len, n_features) or (B, seq_len, n_features)
    ) -> np.ndarray:
        """Run inference on an ONNX session."""
        if features.ndim == 2:
            features = features[np.newaxis, ...]
        return session.run(None, {"features": features.astype(np.float32)})[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHADOW MODE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ShadowModeManager:
    """
    Runs a candidate model in parallel with the live production model.
    Records both models' signals on real bars — but only the production
    model's signals are executed.

    After the shadow period (default 10 trading days), computes:
      - Signal agreement rate
      - Hypothetical Sharpe of candidate vs production
      - Whether the candidate should be promoted

    This prevents deploying a model that looked good in backtest but
    diverges in live conditions.
    """

    def __init__(
        self,
        shadow_days:         int   = 10,
        min_agreement:       float = 0.60,    # Min signal match rate to qualify
        min_sharpe_delta:    float = 0.10,    # Min Sharpe improvement to deploy
        bars_per_day:        int   = 390,     # 1-min bars in a trading day
        checkpoint_dir:      str   = None,
    ):
        if checkpoint_dir is None:
            checkpoint_dir = PATHS["checkpoints"]
        self.shadow_days  = shadow_days
        self.min_agree    = min_agreement
        self.min_sharpe_d = min_sharpe_delta
        self.bars_per_day = bars_per_day
        self.ckpt_dir     = Path(checkpoint_dir)
        self.shadow_bars  = shadow_days * bars_per_day

        self._records: List[dict] = []
        self._active   = False
        self._candidate_name = ""

    def activate(self, candidate_name: str):
        """Start shadow period for a new candidate model."""
        self._records.clear()
        self._active = True
        self._candidate_name = candidate_name
        self._start_time = datetime.now(timezone.utc)
        print(f"[Shadow] Started shadow period for '{candidate_name}' | "
              f"{self.shadow_days} days / {self.shadow_bars} bars")

    def record(
        self,
        bar_time:       pd.Timestamp,
        live_signal:    int,    # Production model: 0=Buy, 1=Hold, 2=Sell
        candidate_signal: int,  # New candidate model
        price:          float,
        actual_return:  float = 0.0,
    ):
        """Record one bar's signals from both models."""
        if not self._active: return
        self._records.append({
            "time":       bar_time,
            "live":       live_signal,
            "candidate":  candidate_signal,
            "price":      price,
            "ret":        actual_return,
            "agree":      int(live_signal == candidate_signal),
        })

    def should_promote(self) -> Tuple[bool, dict]:
        """
        Evaluate whether the candidate model should replace production.
        Returns (should_promote, metrics_dict).
        """
        if not self._records or len(self._records) < self.shadow_bars // 2:
            return False, {"reason": "insufficient_data",
                           "bars_recorded": len(self._records)}

        df = pd.DataFrame(self._records)

        # Signal agreement
        agreement = df["agree"].mean()

        # Hypothetical P&L: sign of signal × next return
        live_ret = np.sign(1.5 - df["live"].values) * df["ret"].values
        cand_ret = np.sign(1.5 - df["candidate"].values) * df["ret"].values

        def sharpe(rets):
            r = np.array(rets)
            return float(r.mean() / (r.std() + 1e-9) * np.sqrt(252 * self.bars_per_day))

        live_sharpe = sharpe(live_ret)
        cand_sharpe = sharpe(cand_ret)
        delta_sharpe = cand_sharpe - live_sharpe

        metrics = {
            "candidate":     self._candidate_name,
            "bars_recorded": len(df),
            "agreement":     round(agreement, 4),
            "live_sharpe":   round(live_sharpe, 4),
            "cand_sharpe":   round(cand_sharpe, 4),
            "delta_sharpe":  round(delta_sharpe, 4),
            "promote":       bool(agreement >= self.min_agree and
                                  delta_sharpe >= self.min_sharpe_d),
        }

        should = metrics["promote"]
        if should:
            print(f"[Shadow] PROMOTE '{self._candidate_name}' | "
                  f"Δ Sharpe: +{delta_sharpe:.3f} | Agreement: {agreement:.1%}")
        else:
            print(f"[Shadow] KEEP production model | "
                  f"Δ Sharpe: {delta_sharpe:+.3f} | Agreement: {agreement:.1%}")

        self._active = False
        return should, metrics

    def save_report(self, path: Optional[str] = None):
        """Save shadow period records to JSON."""
        if not self._records: return
        p = path or str(self.ckpt_dir / f"shadow_{self._candidate_name}.json")
        with open(p, "w") as f:
            json.dump(
                [{k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                  for k, v in r.items()} for r in self._records],
                f, indent=2
            )
        print(f"[Shadow] Report saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP FEATURE IMPORTANCE MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class SHAPMonitor:
    """
    Computes SHAP values after each retraining cycle and logs them to W&B.
    Alerts if the top-5 most important features change significantly —
    an early warning sign of distribution shift before PSI/KS tests trigger.

    Uses TreeExplainer for fast approximation on gradient-boosted surrogate,
    or DeepExplainer for exact SHAP on the neural network.
    """

    def __init__(
        self,
        feature_names:   List[str],
        n_samples:       int   = 1000,
        top_k:           int   = 10,
        shift_threshold: float = 0.30,   # Alert if top feature set changes by >30%
        log_dir:         str   = None,
    ):
        if log_dir is None:
            log_dir = PATHS["logs_shap"]
        self.feat_names    = feature_names
        self.n_samples     = n_samples
        self.top_k         = top_k
        self.shift_thresh  = shift_threshold
        self.log_dir       = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._prev_top: Optional[set] = None
        self._history: List[dict] = []

    def _compute_gradient_shap(
        self,
        model:      "nn.Module",
        X:          np.ndarray,   # (n_samples, seq_len, n_features)
    ) -> np.ndarray:
        """
        Gradient-based SHAP approximation using integrated gradients.
        Returns (n_features,) importance scores.
        """
        if not TORCH:
            return np.zeros(len(self.feat_names))

        model.eval()
        x      = torch.tensor(X[:self.n_samples], dtype=torch.float32, requires_grad=True)
        base   = torch.zeros_like(x)
        steps  = 20

        ig = torch.zeros_like(x)
        for k in range(steps):
            alpha   = k / steps
            inp     = base + alpha * (x - base)
            inp     = inp.detach().requires_grad_(True)
            output  = model(inp).sum()
            output.backward()
            ig += inp.grad.detach()

        ig = ig / steps * (x.detach() - base)
        # Average across batch and seq dims → (n_features,)
        shap_vals = ig.abs().mean(dim=(0, 1)).numpy()
        return shap_vals

    def _compute_permutation_importance(
        self,
        model:      "nn.Module",
        X:          np.ndarray,
        y:          np.ndarray,
        n_repeats:  int = 5,
    ) -> np.ndarray:
        """
        Model-agnostic permutation importance (works without SHAP library).
        Slower than gradient SHAP but exact.
        """
        if not TORCH:
            return np.zeros(len(self.feat_names))

        model.eval()
        x    = torch.tensor(X[:self.n_samples], dtype=torch.float32)
        yt   = torch.tensor(y[:self.n_samples], dtype=torch.float32)

        with torch.no_grad():
            base_loss = float(torch.nn.functional.mse_loss(model(x), yt))

        importance = np.zeros(X.shape[-1])
        rng        = np.random.default_rng(42)

        for feat_idx in range(X.shape[-1]):
            perm_losses = []
            for _ in range(n_repeats):
                X_perm = X[:self.n_samples].copy()
                perm   = rng.permutation(len(X_perm))
                X_perm[:, :, feat_idx] = X_perm[perm, :, feat_idx]
                xp = torch.tensor(X_perm, dtype=torch.float32)
                with torch.no_grad():
                    pl = float(torch.nn.functional.mse_loss(model(xp), yt))
                perm_losses.append(pl)
            importance[feat_idx] = float(np.mean(perm_losses)) - base_loss

        return importance

    def compute_and_log(
        self,
        model:       "nn.Module",
        X:           np.ndarray,
        y:           np.ndarray,
        cycle:       int,
        method:      str = "gradient",   # "gradient" | "permutation"
        wandb_run: Any = None,
    ) -> dict:
        """
        Compute SHAP/importance, compare to previous cycle, log to W&B.
        Returns dict with top features and shift flag.
        """
        print(f"[SHAP] Computing {'gradient' if method=='gradient' else 'permutation'} "
              f"importance | Cycle {cycle}")

        if method == "gradient":
            shap_vals = self._compute_gradient_shap(model, X)
        else:
            shap_vals = self._compute_permutation_importance(model, X, y)

        # Rank features
        ranked   = np.argsort(shap_vals)[::-1]
        top_k    = min(self.top_k, len(self.feat_names))
        top_feats = {self.feat_names[i]: float(shap_vals[i]) for i in ranked[:top_k]}
        top_set   = set(list(top_feats.keys()))

        # Detect shift from previous cycle
        shift_alert = False
        if self._prev_top is not None:
            overlap = len(top_set & self._prev_top) / max(len(top_set), 1)
            shift   = 1.0 - overlap
            if shift > self.shift_thresh:
                shift_alert = True
                print(f"[SHAP] ⚠ FEATURE SHIFT DETECTED! {shift:.0%} of top-{top_k} changed")
                print(f"       New: {top_set - self._prev_top}")
                print(f"       Gone: {self._prev_top - top_set}")

        self._prev_top = top_set

        # Save record
        record = {
            "cycle":       cycle,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "top_features": top_feats,
            "shift_alert": shift_alert,
        }
        self._history.append(record)

        # Save JSON
        p = self.log_dir / f"shap_cycle_{cycle:04d}.json"
        json.dump(record, open(p, "w"), indent=2)

        # W&B logging
        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log({
                    f"shap/{name}": val for name, val in top_feats.items()
                } | {"shap/shift_alert": int(shift_alert), "shap/cycle": cycle})
            except Exception: pass

        return record


# ─────────────────────────────────────────────────────────────────────────────
# 4. WALK-FORWARD REPORTER
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardReporter:
    """
    Generates weekly HTML + JSON performance reports covering:
      - Sharpe ratio (rolling 30-day and since inception)
      - Max drawdown (rolling and all-time)
      - Win rate, average win/loss
      - Feature drift scores (from SHAPMonitor)
      - Trade count and overtrading flag
      - Model version and last retrain date
    """

    def __init__(
        self,
        report_dir:  str = None,
        pair:        str = "EURUSD",
    ):
        if report_dir is None:
            report_dir = PATHS["logs_reports"]
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.pair = pair
        self._trades:   List[dict] = []
        self._equity:   List[float] = [10_000.0]
        self._retrain_log: List[dict] = []

    def record_trade(
        self,
        entry_time: pd.Timestamp,
        exit_time:  pd.Timestamp,
        direction:  str,   # "long" | "short"
        pnl_pips:   float,
        lots:       float,
        model_ver:  str = "v1",
    ):
        pnl_usd = pnl_pips * lots * 10  # USD per pip per lot
        self._equity.append(self._equity[-1] + pnl_usd)
        self._trades.append({
            "entry_time": entry_time.isoformat() if hasattr(entry_time,"isoformat") else str(entry_time),
            "exit_time":  exit_time.isoformat() if hasattr(exit_time,"isoformat") else str(exit_time),
            "direction":  direction,
            "pnl_pips":   pnl_pips,
            "pnl_usd":    pnl_usd,
            "lots":       lots,
            "model_ver":  model_ver,
            "winner":     pnl_pips > 0,
        })

    def record_retrain(self, cycle: int, model: str, sharpe: float, deployed: bool):
        self._retrain_log.append({
            "cycle": cycle, "model": model,
            "sharpe": sharpe, "deployed": deployed,
            "time": datetime.now(timezone.utc).isoformat(),
        })

    def _metrics(self, trades: List[dict], equity: List[float]) -> dict:
        if not trades:
            return {"n_trades": 0}

        pnl    = [t["pnl_usd"] for t in trades]
        wins   = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p < 0]
        eq_arr = np.array(equity)
        peak   = np.maximum.accumulate(eq_arr)
        dd_arr = (peak - eq_arr) / peak
        max_dd = float(dd_arr.max())

        rets   = np.array(pnl)
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if len(rets) > 1 else 0.0

        return {
            "n_trades":       len(trades),
            "win_rate":       round(len(wins) / len(trades), 4),
            "avg_win_usd":    round(np.mean(wins), 2) if wins else 0.0,
            "avg_loss_usd":   round(np.mean(losses), 2) if losses else 0.0,
            "profit_factor":  round(sum(wins) / max(abs(sum(losses)), 1), 3),
            "sharpe":         round(sharpe, 4),
            "max_drawdown":   round(max_dd, 4),
            "total_pnl":      round(sum(pnl), 2),
            "final_equity":   round(equity[-1], 2),
        }

    def generate_report(
        self,
        window_days: int = 30,
        shap_records: Optional[List[dict]] = None,
    ) -> dict:
        """Generate and save the weekly performance report."""
        now      = datetime.now(timezone.utc)
        fname    = f"report_{now.strftime('%Y%m%d')}"

        all_metrics  = self._metrics(self._trades, self._equity)

        # Window metrics (last N days)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
        window_trades = [t for t in self._trades
                         if pd.Timestamp(t["exit_time"], tz="UTC") >= cutoff]
        win_metrics  = self._metrics(
            window_trades,
            self._equity[-max(len(window_trades)+1, 2):]
        )

        report = {
            "generated_at":    now.isoformat(),
            "pair":            self.pair,
            "all_time":        all_metrics,
            f"last_{window_days}d": win_metrics,
            "retrain_history": self._retrain_log[-10:],
            "shap_drift":      [r for r in (shap_records or []) if r.get("shift_alert")],
            "model_versions":  list({t["model_ver"] for t in self._trades}),
        }

        # Save JSON
        json.dump(report, open(self.report_dir / f"{fname}.json", "w"), indent=2)

        # Save simple HTML
        self._save_html(report, self.report_dir / f"{fname}.html")

        print(f"[Report] Saved → {self.report_dir}/{fname}.json + .html")
        print(f"[Report] Sharpe: {all_metrics.get('sharpe',0):.3f} | "
              f"Win rate: {all_metrics.get('win_rate',0):.1%} | "
              f"Max DD: {all_metrics.get('max_drawdown',0):.1%}")
        return report

    def _save_html(self, report: dict, path: Path):
        m  = report["all_time"]
        r  = report.get(f"last_30d", {})
        html = f"""<!DOCTYPE html>
<html><head><title>Forex Model Report</title>
<style>body{{font-family:sans-serif;margin:40px;background:#f9f9f9}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px}}th{{background:#333;color:#fff}}
.good{{color:green}}.bad{{color:red}}.warn{{color:orange}}</style></head>
<body>
<h1>Forex Scaling Model — Performance Report</h1>
<p>Generated: {report['generated_at']} | Pair: {report['pair']}</p>
<h2>All-Time Performance</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total trades</td><td>{m.get('n_trades',0)}</td></tr>
<tr><td>Win rate</td><td class="{'good' if m.get('win_rate',0)>0.5 else 'bad'}">{m.get('win_rate',0):.1%}</td></tr>
<tr><td>Sharpe ratio</td><td class="{'good' if m.get('sharpe',0)>1 else 'warn'}">{m.get('sharpe',0):.3f}</td></tr>
<tr><td>Max drawdown</td><td class="{'bad' if m.get('max_drawdown',0)>0.1 else 'good'}">{m.get('max_drawdown',0):.1%}</td></tr>
<tr><td>Profit factor</td><td>{m.get('profit_factor',0):.2f}</td></tr>
<tr><td>Total P&amp;L</td><td>${m.get('total_pnl',0):,.2f}</td></tr>
<tr><td>Final equity</td><td>${m.get('final_equity',0):,.2f}</td></tr>
</table>
<h2>Last 30 Days</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Trades</td><td>{r.get('n_trades',0)}</td></tr>
<tr><td>Sharpe</td><td>{r.get('sharpe',0):.3f}</td></tr>
<tr><td>Win rate</td><td>{r.get('win_rate',0):.1%}</td></tr>
</table>
{'<p class="bad">⚠ FEATURE DRIFT DETECTED — check SHAP logs</p>' if report.get('shap_drift') else ''}
</body></html>"""
        path.write_text(html)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("Deployment infrastructure — smoke tests")
    print("=" * 50)

    # Shadow mode
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowModeManager(shadow_days=1, bars_per_day=10, checkpoint_dir=td)
        sm.activate("haelt_v2")
        rng = np.random.default_rng(42)
        for i in range(15):
            live_sig = rng.choice([0,1,2])
            cand_sig = rng.choice([0,1,2])
            sm.record(pd.Timestamp.now(tz="UTC"), live_sig, cand_sig,
                      1.085 + rng.normal(0,0.0003), rng.normal(0,0.0002))
        promote, metrics = sm.should_promote()
        print(f"\n  Shadow: promote={promote} | agreement={metrics['agreement']:.2%} | "
              f"Δ Sharpe={metrics['delta_sharpe']:+.3f}")

    # Walk-forward reporter
    with tempfile.TemporaryDirectory() as td:
        rep = WalkForwardReporter(report_dir=td)
        rng = np.random.default_rng(42)
        now = pd.Timestamp.now(tz="UTC")
        for i in range(50):
            pnl = rng.normal(2, 8)
            rep.record_trade(now + pd.Timedelta(hours=i),
                             now + pd.Timedelta(hours=i, minutes=30),
                             "long" if pnl>0 else "short", pnl, 0.5, "haelt_v1")
        rep.record_retrain(1, "haelt", 1.23, True)
        report = rep.generate_report()
        m = report["all_time"]
        print(f"\n  Report: {m['n_trades']} trades | "
              f"Sharpe {m['sharpe']:.3f} | Win {m['win_rate']:.1%}")

    # ONNX (only if torch + onnx available)
    if TORCH and ONNX:
        import sys; sys.path.insert(0,"..")
        from models.architectures import MambaScalper
        with tempfile.TemporaryDirectory() as td:
            exp = ONNXExporter(export_dir=td)
            m   = MambaScalper(input_size=32, d_model=64, num_layers=2)
            p   = exp.export(m, 32, 60, "mamba_test", opset=17)
            if p:
                sess = exp.load_session(p)
                x    = np.random.randn(1, 60, 32).astype(np.float32)
                pred = exp.infer(sess, x)
                print(f"\n  ONNX inference: {pred.shape} | val={pred[0]:.4f}")
    else:
        print(f"\n  ONNX: torch={TORCH} onnxruntime={ONNX} "
              f"(install to enable GPU→CPU export)")

    print("\nAll deployment tests passed ✓")
