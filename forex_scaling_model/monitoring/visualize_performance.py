"""
monitoring/visualize_performance.py
=====================================
Training performance dashboard for the Forex Scaling Model.

Reads logs/{run}_{model}_cv.json and checkpoints/{model}_fold{N}_config.json,
then generates a multi-panel PNG dashboard and prints a summary table.

Usage
-----
  # Show all logs found in logs/
  python monitoring/visualize_performance.py

  # Filter to one model
  python monitoring/visualize_performance.py --model haelt

  # Filter to one run prefix
  python monitoring/visualize_performance.py --run haelt_0331_1206

  # Custom dirs / output
  python monitoring/visualize_performance.py --log-dir logs/ --out reports/dashboard.png

  # Open an interactive window instead of saving
  python monitoring/visualize_performance.py --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Matplotlib setup ────────────────────────────────────────────────────────
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    matplotlib.rcParams.update({
        "figure.facecolor":  "#0e1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#c9d1d9",
        "axes.titlecolor":   "#e6edf3",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "grid.color":        "#21262d",
        "grid.linewidth":    0.6,
        "text.color":        "#c9d1d9",
        "legend.facecolor":  "#161b22",
        "legend.edgecolor":  "#30363d",
        "font.family":       "monospace",
        "font.size":         9,
    })
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[ERROR] matplotlib not installed.  pip install matplotlib")
    sys.exit(1)

# ── Colour palette (one per fold, cycling) ──────────────────────────────────
FOLD_COLOURS = [
    "#58a6ff",   # blue
    "#3fb950",   # green
    "#f78166",   # red-orange
    "#d2a8ff",   # purple
    "#ffa657",   # amber
    "#79c0ff",   # light blue
    "#56d364",   # light green
    "#ff7b72",   # coral
]

MODEL_COLOURS = {
    "haelt":       "#58a6ff",
    "tft":         "#3fb950",
    "transformer": "#ffa657",
    "mamba":       "#d2a8ff",
    "gnn":         "#f78166",
    "expert":      "#79c0ff",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _model_from_filename(path: Path) -> str:
    """
    Extract model name from log filename.
    Pattern:  {run_prefix}_{model_name}_cv.json
    The model name is the token immediately before '_cv'.
    """
    stem = path.stem                        # e.g. haelt_0331_1206_tft_cv
    parts = stem.split("_")
    if parts[-1] == "cv" and len(parts) >= 2:
        return parts[-2]                    # e.g. "tft"
    return stem                             # fallback: whole stem


def _run_from_filename(path: Path) -> str:
    """Return everything before the model+_cv suffix."""
    stem  = path.stem
    parts = stem.split("_")
    if parts[-1] == "cv" and len(parts) >= 3:
        return "_".join(parts[:-2])         # e.g. "haelt_0331_1206"
    return stem


def load_cv_logs(
    log_dir: Path,
    model_filter: Optional[str] = None,
    run_filter:   Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Scan log_dir for *_cv.json files.
    Returns  {"{run}|{model}": {"run": str, "model": str, "folds": list}}
    """
    results: Dict[str, Dict] = {}
    for p in sorted(log_dir.glob("*_cv.json")):
        model = _model_from_filename(p)
        run   = _run_from_filename(p)
        if model_filter and model.lower() != model_filter.lower():
            continue
        if run_filter and not run.startswith(run_filter):
            continue
        try:
            folds = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Could not read {p}: {exc}")
            continue
        key = f"{run}|{model}"
        results[key] = {"run": run, "model": model, "folds": folds, "path": str(p)}
    return results


def load_checkpoint_configs(
    ckpt_dir: Path,
    model_filter: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """
    Scan checkpoints/ for {model}_fold{N}_config.json files.
    Returns  {"haelt": [cfg_fold0, cfg_fold1, ...], ...}
    """
    results: Dict[str, List[dict]] = {}
    for p in sorted(ckpt_dir.glob("*_fold*_config.json")):
        parts = p.stem.split("_fold")          # ["haelt", "0_config"]
        if len(parts) < 2:
            continue
        model = parts[0]
        if model_filter and model.lower() != model_filter.lower():
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Could not read {p}: {exc}")
            continue
        results.setdefault(model, []).append(cfg)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _epoch_x(history_list: list) -> np.ndarray:
    return np.arange(1, len(history_list) + 1)


def _plot_metric_per_fold(
    ax: "plt.Axes",
    folds:  list,
    key:    str,
    title:  str,
    ylabel: str,
    smooth: bool = False,
    fmt:    str  = ".4f",
) -> None:
    """Plot one metric (one line per fold) on ax."""
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.4)

    for fold_data in folds:
        fi     = fold_data["fold"]
        hist   = fold_data["history"]
        vals   = hist.get(key)
        if not vals:
            continue
        colour = FOLD_COLOURS[fi % len(FOLD_COLOURS)]
        x      = _epoch_x(vals)
        y      = np.array(vals, dtype=float)

        if smooth and len(y) > 3:
            kernel = np.ones(3) / 3
            y_sm   = np.convolve(y, kernel, mode="same")
            ax.plot(x, y_sm, color=colour, lw=1.5, label=f"Fold {fi}")
            ax.plot(x, y, color=colour, lw=0.5, alpha=0.3)
        else:
            ax.plot(x, y, color=colour, lw=1.5, label=f"Fold {fi}")

        best = fold_data.get("best_metric")
        if best is not None and key in ("val_sharpe", "dir_acc"):
            ax.axhline(best, color=colour, lw=0.8, ls="--", alpha=0.5)

    if len(folds) <= 8:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.5)


def _plot_loss_overlay(ax: "plt.Axes", folds: list) -> None:
    """Training loss (solid) + validation loss (dashed) per fold."""
    ax.set_title("Training vs Validation Loss", fontsize=10, pad=6)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)
    ax.grid(True, alpha=0.4)

    for fold_data in folds:
        fi     = fold_data["fold"]
        hist   = fold_data["history"]
        colour = FOLD_COLOURS[fi % len(FOLD_COLOURS)]

        tr = hist.get("train_loss", [])
        vl = hist.get("val_loss",   [])
        if tr:
            ax.plot(_epoch_x(tr), tr, color=colour, lw=1.5,
                    label=f"Fold {fi} train")
        if vl:
            ax.plot(_epoch_x(vl), vl, color=colour, lw=1.5,
                    ls="--", alpha=0.7, label=f"Fold {fi} val")

    # Custom legend: just show train vs val line style
    legend_els = [
        Line2D([0], [0], color="#8b949e", lw=1.5,       label="Train loss"),
        Line2D([0], [0], color="#8b949e", lw=1.5, ls="--", label="Val loss"),
    ]
    ax.legend(handles=legend_els, fontsize=7, framealpha=0.5)


def _plot_fold_bars(
    ax:     "plt.Axes",
    folds:  list,
    model:  str,
    metric: str = "best_metric",
    label:  str = "Best Sharpe",
) -> None:
    """Bar chart of best metric per fold."""
    ax.set_title(f"Best {label} per Fold — {model.upper()}", fontsize=10, pad=6)
    ax.set_xlabel("Fold", fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    ax.grid(True, alpha=0.4, axis="y")

    xs     = [f["fold"] for f in folds]
    ys     = [f.get(metric, 0.0) for f in folds]
    cols   = [FOLD_COLOURS[x % len(FOLD_COLOURS)] for x in xs]
    bars   = ax.bar(xs, ys, color=cols, width=0.6, edgecolor="#30363d", linewidth=0.5)

    mean_y = float(np.mean(ys))
    ax.axhline(mean_y, color="#ffa657", lw=1.2, ls="--",
               label=f"Mean {mean_y:.4f}")
    ax.legend(fontsize=7, framealpha=0.5)

    for bar, y in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{y:.3f}", ha="center", va="bottom", fontsize=7, color="#c9d1d9",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"F{x}" for x in xs], fontsize=8)


def _plot_model_comparison(
    ax:       "plt.Axes",
    all_logs: Dict[str, Dict],
    ckpt_cfgs: Dict[str, List[dict]],
) -> None:
    """
    Grouped bar chart: best val Sharpe per model.
    Left bars = from log files (best across folds).
    Right bars = from checkpoint configs (best_val_sharpe_proxy).
    """
    ax.set_title("Model Comparison — Best Validation Sharpe", fontsize=10, pad=6)
    ax.set_xlabel("Model", fontsize=8)
    ax.set_ylabel("Best Val Sharpe", fontsize=8)
    ax.grid(True, alpha=0.4, axis="y")

    # Gather per-model best from logs
    model_best_log:  Dict[str, float] = {}
    model_best_ckpt: Dict[str, float] = {}

    for key, entry in all_logs.items():
        model  = entry["model"]
        folds  = entry["folds"]
        best   = max((f.get("best_metric", -999) for f in folds), default=-999)
        if model not in model_best_log or best > model_best_log[model]:
            model_best_log[model] = best

    for model, cfgs in ckpt_cfgs.items():
        sharpes = [c.get("best_val_sharpe_proxy", -999) for c in cfgs]
        model_best_ckpt[model] = max(sharpes)

    all_models = sorted(set(list(model_best_log.keys()) + list(model_best_ckpt.keys())))
    if not all_models:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#8b949e")
        return

    x      = np.arange(len(all_models))
    width  = 0.35

    log_vals  = [model_best_log.get(m,  None) for m in all_models]
    ckpt_vals = [model_best_ckpt.get(m, None) for m in all_models]

    bar_log  = ax.bar(x - width/2,
                      [v if v is not None else 0 for v in log_vals],
                      width, color="#58a6ff", label="Best fold (log)", edgecolor="#30363d")
    bar_ckpt = ax.bar(x + width/2,
                      [v if v is not None else 0 for v in ckpt_vals],
                      width, color="#3fb950", label="Best ckpt Sharpe", edgecolor="#30363d")

    for bars, vals in [(bar_log, log_vals), (bar_ckpt, ckpt_vals)]:
        for bar, v in zip(bars, vals):
            if v is not None and v > -900:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, color="#c9d1d9")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in all_models], fontsize=8)
    ax.legend(fontsize=7, framealpha=0.5)


def _plot_sharpe_heatmap(
    ax:    "plt.Axes",
    folds: list,
    model: str,
) -> None:
    """Colour-coded epoch × fold Sharpe heatmap."""
    ax.set_title(f"Sharpe Heatmap — {model.upper()} (epoch × fold)", fontsize=10, pad=6)

    # Build matrix: rows = folds, cols = epochs (pad shorter folds with NaN)
    rows    = sorted(folds, key=lambda f: f["fold"])
    lengths = [len(f["history"].get("val_sharpe", [])) for f in rows]
    max_ep  = max(lengths) if lengths else 1
    matrix  = np.full((len(rows), max_ep), np.nan)
    for r, fold_data in enumerate(rows):
        vals = fold_data["history"].get("val_sharpe", [])
        matrix[r, :len(vals)] = vals

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.1)
    im   = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Sharpe", fontsize=7)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"F{f['fold']}" for f in rows], fontsize=8)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Fold", fontsize=8)


def _plot_summary_table(
    ax:        "plt.Axes",
    all_logs:  Dict[str, Dict],
    ckpt_cfgs: Dict[str, List[dict]],
) -> None:
    """Render a text summary table inside an axes."""
    ax.axis("off")
    ax.set_title("Performance Summary", fontsize=10, pad=6)

    headers = ["Model", "Run", "Folds", "Epochs (avg)", "Best Sharpe",
               "Mean Sharpe", "Best Acc", "Ckpt Sharpe"]
    col_w   = [0.10, 0.22, 0.06, 0.10, 0.10, 0.10, 0.10, 0.10]

    # Header row
    x_pos = 0.01
    for h, w in zip(headers, col_w):
        ax.text(x_pos, 0.92, h, transform=ax.transAxes,
                fontsize=7.5, color="#58a6ff", fontweight="bold", va="top")
        x_pos += w

    # Divider
    ax.axhline(0.88, color="#30363d", lw=0.8, transform=ax.transAxes,
               xmin=0.01, xmax=0.99)

    rows_data: list = []
    for key, entry in sorted(all_logs.items()):
        model  = entry["model"]
        run    = entry["run"]
        folds  = entry["folds"]
        n_fold = len(folds)
        epochs = [len(f["history"].get("train_loss", [])) for f in folds]
        avg_ep = float(np.mean(epochs)) if epochs else 0

        sharpes  = [f.get("best_metric", -999) for f in folds]
        best_sh  = max(sharpes)
        mean_sh  = float(np.mean(sharpes))

        accs = [max(f["history"].get("dir_acc", [0])) for f in folds]
        best_acc = max(accs)

        ckpt_sh  = model_best_ckpt = None
        if model in ckpt_cfgs:
            ckpt_vals = [c.get("best_val_sharpe_proxy", -999) for c in ckpt_cfgs[model]]
            ckpt_sh = max(ckpt_vals)

        rows_data.append((model, run, n_fold, avg_ep, best_sh, mean_sh, best_acc, ckpt_sh))

    for ri, (model, run, n_fold, avg_ep, best_sh, mean_sh, best_acc, ckpt_sh) in \
            enumerate(rows_data):
        y     = 0.84 - ri * 0.09
        if y < 0.05:
            break
        col   = MODEL_COLOURS.get(model, "#c9d1d9")
        vals  = [
            model.upper(),
            run[:24],
            str(n_fold),
            f"{avg_ep:.0f}",
            f"{best_sh:.4f}",
            f"{mean_sh:.4f}",
            f"{best_acc:.2%}",
            f"{ckpt_sh:.4f}" if ckpt_sh is not None else "—",
        ]
        x_pos = 0.01
        for vi, (v, w) in enumerate(zip(vals, col_w)):
            c = col if vi == 0 else "#c9d1d9"
            if vi in (4, 5) and float(vals[4].replace("—", "-999")) < 0:
                c = "#f78166"
            ax.text(x_pos, y, v, transform=ax.transAxes,
                    fontsize=7.5, color=c, va="top", fontfamily="monospace")
            x_pos += w


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(
    all_logs:  Dict[str, Dict],
    ckpt_cfgs: Dict[str, List[dict]],
    out_path:  Optional[Path] = None,
    show:      bool           = False,
) -> None:
    if not all_logs and not ckpt_cfgs:
        print("[WARN] No log or checkpoint data found. Nothing to plot.")
        return

    # ── Figure layout ────────────────────────────────────────────────────────
    # One "section" per (run, model) combo, plus a global comparison panel.
    n_combos  = len(all_logs)
    n_rows    = max(n_combos * 3 + 3, 6)   # 3 rows per combo + comparison + summary
    fig_h     = max(n_rows * 3.0, 14)

    fig = plt.figure(figsize=(18, fig_h), dpi=110)
    fig.patch.set_facecolor("#0e1117")

    # Title
    first_model = next(iter(all_logs.values()))["model"].upper() if all_logs else "—"
    fig.suptitle(
        f"Forex Scaling Model — Performance Dashboard\n"
        f"{len(all_logs)} run(s)  ·  "
        f"{sum(len(e['folds']) for e in all_logs.values())} total folds",
        fontsize=13, color="#e6edf3", y=0.995, fontweight="bold",
    )

    gs = gridspec.GridSpec(
        n_rows, 4,
        figure=fig,
        hspace=0.52,
        wspace=0.32,
        top=0.97,
        bottom=0.02,
        left=0.05,
        right=0.97,
    )

    row = 0

    # ── Per-combo panels ─────────────────────────────────────────────────────
    for key, entry in sorted(all_logs.items()):
        model = entry["model"]
        run   = entry["run"]
        folds = entry["folds"]

        # Row label
        fig.text(
            0.01, 1 - (row / n_rows) - 0.005,
            f"▶  {model.upper()}  ·  {run}  ·  {len(folds)} folds",
            fontsize=9, color=MODEL_COLOURS.get(model, "#c9d1d9"),
            fontweight="bold", transform=fig.transFigure,
        )

        # Row 1: Loss | Directional accuracy
        ax_loss = fig.add_subplot(gs[row, :2])
        ax_acc  = fig.add_subplot(gs[row, 2:])
        _plot_loss_overlay(ax_loss, folds)
        _plot_metric_per_fold(ax_acc, folds, "dir_acc",
                              "Directional Accuracy", "Accuracy", smooth=True)
        row += 1

        # Row 2: Sharpe | LR
        ax_sh = fig.add_subplot(gs[row, :2])
        ax_lr = fig.add_subplot(gs[row, 2:])
        _plot_metric_per_fold(ax_sh, folds, "val_sharpe",
                              "Validation Sharpe (proxy)", "Sharpe", smooth=True)
        _plot_metric_per_fold(ax_lr, folds, "lr",
                              "Learning Rate Schedule", "LR")
        row += 1

        # Row 3: Fold bar | Heatmap
        ax_bar  = fig.add_subplot(gs[row, :2])
        ax_heat = fig.add_subplot(gs[row, 2:])
        _plot_fold_bars(ax_bar, folds, model)
        _plot_sharpe_heatmap(ax_heat, folds, model)
        row += 1

    # ── Global comparison (only if >1 model or checkpoint data) ─────────────
    if len(all_logs) > 0 or ckpt_cfgs:
        ax_cmp = fig.add_subplot(gs[row, :])
        _plot_model_comparison(ax_cmp, all_logs, ckpt_cfgs)
        row += 1

    # ── Summary table ────────────────────────────────────────────────────────
    if row < n_rows and all_logs:
        ax_tbl = fig.add_subplot(gs[row:row + 2, :])
        _plot_summary_table(ax_tbl, all_logs, ckpt_cfgs)

    # ── Output ───────────────────────────────────────────────────────────────
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Dashboard] Saved → {out_path}")

    if show:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    all_logs:  Dict[str, Dict],
    ckpt_cfgs: Dict[str, List[dict]],
) -> None:
    if not all_logs and not ckpt_cfgs:
        print("No data found.")
        return

    sep = "─" * 82
    print(f"\n{sep}")
    print(f"  {'MODEL':<12} {'RUN':<22} {'FOLDS':>5} {'EP':>4}  "
          f"{'BEST_SH':>8}  {'MEAN_SH':>8}  {'BEST_ACC':>8}  {'CKPT_SH':>8}")
    print(sep)

    for key, entry in sorted(all_logs.items()):
        model  = entry["model"]
        run    = entry["run"]
        folds  = entry["folds"]
        epochs = [len(f["history"].get("train_loss", [])) for f in folds]
        avg_ep = int(np.mean(epochs)) if epochs else 0
        sharpes  = [f.get("best_metric", 0) for f in folds]
        best_sh  = max(sharpes)
        mean_sh  = float(np.mean(sharpes))
        accs     = [max(f["history"].get("dir_acc", [0])) for f in folds]
        best_acc = max(accs)
        ckpt_sh  = "—"
        if model in ckpt_cfgs:
            ckpt_sh = f"{max(c.get('best_val_sharpe_proxy', -999) for c in ckpt_cfgs[model]):.4f}"

        print(f"  {model.upper():<12} {run[:22]:<22} {len(folds):>5} {avg_ep:>4}  "
              f"{best_sh:>8.4f}  {mean_sh:>8.4f}  {best_acc:>8.2%}  {ckpt_sh:>8}")

    if ckpt_cfgs:
        print(f"\n  Checkpoint-only models (no log file):")
        for model, cfgs in sorted(ckpt_cfgs.items()):
            if not any(e["model"] == model for e in all_logs.values()):
                best = max(c.get("best_val_sharpe_proxy", -999) for c in cfgs)
                print(f"    {model.upper():<12}  {len(cfgs)} fold(s)  best_ckpt_sharpe={best:.4f}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forex Scaling Model — Training Performance Dashboard",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--log-dir",      type=str, default="logs",
                   help="Directory containing *_cv.json training logs (default: logs/)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory containing *_config.json files (default: checkpoints/)")
    p.add_argument("--model",  type=str, default=None,
                   help="Filter to one model name  (e.g. haelt)")
    p.add_argument("--run",    type=str, default=None,
                   help="Filter to one run prefix  (e.g. haelt_0331)")
    p.add_argument("--out",    type=str, default=None,
                   help="Output PNG path  (default: logs/performance_dashboard.png)")
    p.add_argument("--show",   action="store_true",
                   help="Open interactive matplotlib window (in addition to saving)")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save PNG — useful with --show on headless servers")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    log_dir  = Path(args.log_dir)
    ckpt_dir = Path(args.checkpoint_dir)

    if not log_dir.exists():
        print(f"[WARN] Log directory not found: {log_dir}")
    if not ckpt_dir.exists():
        print(f"[WARN] Checkpoint directory not found: {ckpt_dir}")

    print(f"[Dashboard] Scanning  logs : {log_dir.resolve()}")
    print(f"[Dashboard] Scanning  ckpts: {ckpt_dir.resolve()}")

    all_logs  = load_cv_logs(log_dir,  model_filter=args.model, run_filter=args.run)
    ckpt_cfgs = load_checkpoint_configs(ckpt_dir, model_filter=args.model)

    print(f"[Dashboard] Found {len(all_logs)} log file(s), "
          f"{sum(len(v) for v in ckpt_cfgs.values())} checkpoint config(s)")

    print_summary(all_logs, ckpt_cfgs)

    out_path: Optional[Path] = None
    if not args.no_save:
        raw = args.out or str(log_dir / "performance_dashboard.png")
        out_path = Path(raw)

    build_dashboard(all_logs, ckpt_cfgs, out_path=out_path, show=args.show)


if __name__ == "__main__":
    main()
