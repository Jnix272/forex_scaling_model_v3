"""
training/train_gpu.py  (v4 — 20M tick scale)
=============================================
Purpose-built to train on 20,000,000 ticks without running out of RAM.

Memory math
-----------
  20M ticks → ~333k 1-min bars → ~333k labeled samples
  All sequences in RAM at once = ~4.5 GB  (too much on most pods)
  Solution: chunk pipeline + memory-mapped HDF5/NPY sequences

Architecture
------------
  Phase 1 — CHUNK INGESTION
    Split 20M ticks into 500k-tick chunks.
    Each chunk: ticks → bars → features → RL labels → append to HDF5.
    Peak RAM per chunk: ~120 MB.  Total disk: ~250 MB HDF5.

  Phase 2 — MEMORY-MAPPED TRAINING
    MemmapSequenceDataset reads sequences directly from HDF5/NPY on disk.
    Workers pre-fetch batches asynchronously — GPU is never waiting.
    Effective throughput: ~1,200 batches/sec on RTX 4090 with AMP.

  Phase 3 — RL TRAINING
    ForexTradingEnv streams samples from the same memory-mapped arrays.
    DQN replay buffer stays on GPU (pinned memory).

Usage
-----
    # Full 20M tick pipeline (cloud GPU or local workstation with enough VRAM)
    python training/train_gpu.py --n-ticks 20000000 --model haelt --epochs 100

    # With real Dukascopy data
    python training/train_gpu.py --data-source dukascopy \\
        --data-start 2020-01-01 --data-end 2023-12-31 \\
        --model haelt --epochs 100

    # With TDS export
    python training/train_gpu.py --data-source tds --model haelt --epochs 100

    # All 6 architectures sequentially
    python training/train_gpu.py --n-ticks 20000000 --all-models --epochs 50

    # Resume after interruption
    python training/train_gpu.py --n-ticks 20000000 --model haelt --resume

    # RL agents on top of supervised
    python training/train_gpu.py --n-ticks 20000000 --rl-train --rl-algo dqn
"""

import os, sys, gc, json, time, argparse, warnings, shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml as _yaml; _YAML = True
except ImportError:
    _YAML = False

try:
    from tqdm import tqdm as _tqdm
    def _pbar(it=None, **kw): return _tqdm(it, **kw)
except ImportError:
    class _DummyBar:
        """No-op progress bar used when tqdm is not installed."""
        def __init__(self, *a, **kw): pass
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass
        def close(self): pass
        def __iter__(self): return iter([])
    def _pbar(it=None, **kw):
        return iter(it) if it is not None else _DummyBar()

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Core project imports ───────────────────────────────────────────────────────
from data.data_ingestion import generate_synthetic_tick_data, ForexDataPipeline
from data.sources import ForexDataManager, TICK_COLUMNS
from data.cross_asset import load_cross_asset_panel
from features.feature_engineering import FeatureEngineer
from labeling.rl_reward_labeling import compute_rl_reward_labels, align_labels_with_features
from labeling.triple_barrier_labeling import compute_triple_barrier_labels
from models.architectures import (
    TFTScalper, iTransformerScalper, HAELTHybrid,
    MambaScalper, GNNFromSequence, EXPERTEncoder,
    HuberLoss, AsymmetricDirectionalLoss, MODEL_REGISTRY,
    MultiTaskHead, MultiTaskLoss, MultiTaskWrapper,
    MultiPairWrapper,
)
from models.rl_agents import ForexTradingEnv, DQNAgent, PPOAgent, train_agent
from pretrain.contrastive import TimeSeriesAugmenter
try:
    from pretrain.contrastive import TSCLTrainer, RegimeAwareTSCLTrainer
except ImportError:
    TSCLTrainer = None
    RegimeAwareTSCLTrainer = None

try:
    from models.ensemble import EnsembleMetaLearner, train_meta_learner
    ENSEMBLE = True
except ImportError:
    ENSEMBLE = False
from config.settings import (
    TRAINING, PRETRAIN, RL, FEATURES, SIZING, RISK, LABELING, HARDWARE_PROFILES, PATHS,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset
    TORCH = True
except ImportError:
    print("[ERROR] PyTorch not installed. pip install torch"); sys.exit(1)

try:
    import h5py; HDF5 = True
except ImportError:
    HDF5 = False
    print("[WARN] h5py not installed — using NPY fallback. pip install h5py")

try:
    import wandb; WANDB = True
except ImportError:
    WANDB = False

try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); OPTUNA = True
except ImportError:
    OPTUNA = False

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM LOSSES (TRADING-AWARE)
# ─────────────────────────────────────────────────────────────────────────────

class DirectionalHuberLoss(nn.Module):
    """Huber magnitude loss + extra penalty when direction is wrong."""

    def __init__(self, delta: float = 1.0, direction_weight: float = 0.5):
        super().__init__()
        self.huber = HuberLoss(delta=delta)
        self.direction_weight = float(direction_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.huber(pred, target)
        # Penalize opposite-sign predictions more than near-zero misses.
        wrong_sign = (pred * target) < 0
        dir_pen = wrong_sign.float() * (pred - target).abs()
        return base + self.direction_weight * dir_pen.mean()


class SharpeProxyLoss(nn.Module):
    """Minimize -Sharpe proxy while keeping pointwise stability via Huber."""

    def __init__(self, delta: float = 1.0, sharpe_weight: float = 0.2, eps: float = 1e-8):
        super().__init__()
        self.huber = HuberLoss(delta=delta)
        self.sharpe_weight = float(sharpe_weight)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.huber(pred, target)
        direction = torch.sign(pred)
        returns = (direction * target).flatten()
        mean = returns.mean()
        std = returns.std(unbiased=False)
        sharpe = mean / (std + self.eps)
        # Minimize negative Sharpe to maximize risk-adjusted returns.
        return base - self.sharpe_weight * sharpe


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

# Maps YAML keys (section.key) → argparse dest names
_YAML_MAP = {
    "data.source":              "data_source",
    "data.pair":                "pair",
    "data.pairs":               "pairs",
    "data.pair_embed_dim":      "pair_embed_dim",
    "data.pair_align":          "pair_align",
    "data.start":               "data_start",
    "data.end":                 "data_end",
    "data.full_day_data":       "full_day_data",
    "data.n_ticks":             "n_ticks",
    "data.chunk_size":          "chunk_size",
    "data.use_cache":           None,          # handled as force_rebuild inversion below
    "model.name":               "model",
    "model.all_models":         "all_models",
    "model.hidden_size":        "hidden_size",
    "model.d_model":            "d_model",
    "model.nhead":              "nhead",
    "model.num_layers":         "num_layers",
    "model.dropout":            "dropout",
    "training.epochs":          "epochs",
    "training.batch_size":      "batch_size",
    "training.lr":              "lr",
    "training.seq_len":         "seq_len",
    "training.patience":        "patience",
    "training.val_split":       "val_split",
    "training.loss":            "loss",
    "training.label_method":    "label_method",
    "training.early_stop_metric": "early_stop_metric",
    "training.save_every":      "save_every",
    "training.grad_clip":       "grad_clip",
    "training.weight_decay":    "weight_decay",
    "training.amp":             "amp",
    "training.cross_asset_mode": "cross_asset_mode",
    "walk_forward.enabled":     "walk_forward_cv",
    "walk_forward.folds":       "walk_forward_folds",
    "multitask.enabled":        "multitask",
    "multitask.w_ret":          "mt_w_ret",
    "multitask.w_conf":         "mt_w_conf",
    "pretrain.enabled":         "pretrain",
    "pretrain.epochs":          "pretrain_epochs",
    "pretrain.regime_aware":    "pretrain_regime",
    "ensemble.enabled":         "train_ensemble",
    "ensemble.epochs":          "ensemble_epochs",
    "ensemble.div_weight":      "ensemble_div_weight",
    "rl.enabled":               "rl_train",
    "rl.algo":                  "rl_algo",
    "rl.episodes":              "rl_episodes",
    "hardware.profile":         "hardware_profile",
    "hardware.num_workers":     "num_workers",
    "hardware.prefetch_factor": "prefetch_factor",
    "tracking.wandb_project":   "wandb_project",
    "tracking.run_name":        "run_name",
    "tracking.no_wandb":        "no_wandb",
    "quick.enabled":            "quick_mode",
    "data.integrity_gate":      "integrity_gate",
    "data.auto_rebuild_on_mismatch": "auto_rebuild_on_mismatch",
    "paths.checkpoint_dir":     "checkpoint_dir",
    "paths.data_cache":         "data_cache",
}


def _apply_yaml_config(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Load config/run.yaml and set argparse defaults from it."""
    if not _YAML:
        print("[Config] PyYAML not installed — ignoring --config. pip install pyyaml")
        return
    path = Path(config_path)
    if not path.exists():
        print(f"[Config] WARN: config file not found: {config_path}")
        return

    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh)
    except Exception as e:
        print(
            f"[Config] YAML parse failed for {config_path}: {e}\n"
            "[Config] Continuing with argparse defaults + explicit CLI flags."
        )
        return
    if cfg is None:
        cfg = {}

    defaults: dict = {}
    for yaml_key, dest in _YAML_MAP.items():
        section, key = yaml_key.split(".", 1)
        val = (cfg.get(section) or {}).get(key)
        if val is None:
            continue
        if dest is None:
            # data.use_cache=false → force_rebuild=true
            if yaml_key == "data.use_cache":
                defaults["force_rebuild"] = not bool(val)
            continue
        # Blank strings mean "use the hardcoded default"
        if isinstance(val, str) and val.strip() == "":
            continue
        defaults[dest] = val

    parser.set_defaults(**defaults)
    print(f"[Config] Loaded {config_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Forex Model — 20M Tick GPU Trainer")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML run config (e.g. config/run.yaml). "
                        "Values are used as defaults; explicit CLI flags override them.")

    # Scale
    p.add_argument("--n-ticks",      type=int,   default=20_000_000,
                   help="Total tick count to train on (default: 20M)")
    p.add_argument("--chunk-size",   type=int,   default=500_000,
                   help="Ticks per processing chunk (RAM safety valve)")

    # Data source
    p.add_argument("--data-source",  type=str,   default="synthetic",
                   choices=["synthetic","dukascopy","tds","lmax_historical","auto"],
                   help="Which data source to use")
    p.add_argument("--data-start",   type=str,   default="2020-01-01")
    p.add_argument("--data-end",     type=str,   default="2023-12-31")
    p.add_argument("--pair",          type=str,   default="EURUSD")
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pairs for joint multi-pair training, e.g. EURUSD,GBPUSD,USDJPY. "
             "Overrides --pair when set. Can also be a list in config/run.yaml under data.pairs.",
    )
    p.add_argument(
        "--pair-embed-dim",
        type=int,
        default=0,
        help="Learnable pair embedding size (int). Appended to each pair's features before "
             "the backbone. 0 = disabled (pairs are simply concatenated on the feature axis).",
    )
    p.add_argument(
        "--pair-align",
        type=str,
        default="inner",
        choices=["inner", "outer"],
        help="Timestamp alignment across pairs: inner=common bars only (default), "
             "outer=fill missing bars with NaN.",
    )
    p.add_argument(
        "--full-day-data",
        action="store_true",
        help="Dukascopy: load all 24h (00–23 UTC). Default is session-only (07–17 UTC).",
    )

    # Model
    p.add_argument("--model",        type=str,   default="haelt",
                   choices=["tft","transformer","haelt","mamba","gnn","expert"])
    p.add_argument("--all-models",   action="store_true")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=2048,
                   help="Batch size — 2048 optimal for 20M samples on RTX 4090")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--seq-len",      type=int,   default=60)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Validation fraction (default: TRAINING['val_split'] from config.settings)",
    )
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument(
        "--cross-asset-mode",
        type=str,
        default="auto",
        choices=["auto", "real", "synthetic", "off"],
        help="Cross-asset features source: auto=real for real FX data, synthetic for synthetic FX; "
             "real=attempt external commodities/yields download; synthetic/off disables external fetch",
    )
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--label-method",
        type=str,
        default="rl_reward",
        choices=["rl_reward", "triple_barrier"],
        help="Supervised targets: RL forward P&L labels vs triple-barrier (ATR barriers + vertical)",
    )
    p.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=["huber", "asymmetric", "cross_entropy", "directional_huber", "sharpe_huber"],
        help="huber/asymmetric/directional_huber/sharpe_huber on scalar targets; "
             "cross_entropy=3-class {-1,0,1} with balanced weights",
    )
    p.add_argument("--direction-weight", type=float, default=0.5,
                   help="Extra wrong-direction penalty multiplier for directional_huber loss")
    p.add_argument("--sharpe-weight", type=float, default=0.2,
                   help="Sharpe proxy weight for sharpe_huber loss")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                   help="Minimum validation improvement required to reset patience")
    p.add_argument("--num-workers",  type=int,   default=8,
                   help="DataLoader workers — 8 is sweet spot for H100/A100")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="DataLoader prefetch (per worker); lower on 16GB RAM PCs")
    p.add_argument("--hardware-profile", type=str, default=None,
                   choices=list(HARDWARE_PROFILES.keys()) if HARDWARE_PROFILES else None,
                   help="Apply tuned defaults (batch/workers/chunk/prefetch/paths). "
                        "rtx_4060_16gb_ram: RTX 4060 8GB VRAM + 16GB system RAM")

    # Architecture
    p.add_argument("--hidden-size",  type=int,   default=256)
    p.add_argument("--num-layers",   type=int,   default=3)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--d-model",      type=int,   default=256)
    p.add_argument("--nhead",        type=int,   default=8)

    # Pre-training
    p.add_argument("--pretrain",         action="store_true")
    p.add_argument("--pretrain-epochs",  type=int,   default=30)
    p.add_argument(
        "--pretrain-regime",
        action="store_true",
        help="Use regime-aware TSCL: same-regime positives + cross-regime hard negatives",
    )
    p.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Delete existing contrastive encoder checkpoint and pretrain from scratch",
    )
    p.add_argument(
        "--multitask",
        action="store_true",
        help="Replace single prediction head with MultiTaskHead "
             "(direction CE + magnitude Huber + confidence BCE)",
    )
    p.add_argument("--mt-w-ret",  type=float, default=0.5,
                   help="Multi-task loss weight for return_hat Huber term (default 0.5)")
    p.add_argument("--mt-w-conf", type=float, default=0.3,
                   help="Multi-task loss weight for confidence BCE term (default 0.3)")
    p.add_argument(
        "--train-ensemble",
        action="store_true",
        help="After supervised training, train the EnsembleMetaLearner "
             "with diversity penalty across all trained base models",
    )
    p.add_argument("--ensemble-epochs",     type=int,   default=10,
                   help="Epochs to train the meta-learner (default 10)")
    p.add_argument("--ensemble-div-weight", type=float, default=0.1,
                   help="Diversity penalty weight for meta-learner training (default 0.1)")

    # RL
    p.add_argument("--rl-train",     action="store_true")
    p.add_argument("--rl-algo",      type=str,   default="dqn", choices=["dqn","ppo"])
    p.add_argument("--rl-episodes",  type=int,   default=500)

    # HPO
    p.add_argument("--hparam-search",action="store_true")
    p.add_argument("--n-trials",     type=int,   default=30)

    # Tracking
    p.add_argument("--wandb-project",type=str,   default="forex-scaling-model")
    p.add_argument("--run-name",     type=str,   default=None)
    p.add_argument("--no-wandb",     action="store_true")
    p.add_argument("--save-every",   type=int,   default=5)

    # Paths
    p.add_argument("--checkpoint-dir", type=str, default=PATHS["checkpoints"])
    p.add_argument("--data-cache", type=str, default=PATHS["data_processed"])
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--force-rebuild",action="store_true",
                   help="Ignore cached HDF5 and rebuild from scratch")
    p.add_argument("--quick-mode", action="store_true",
                   help="Fast sanity run: fewer folds/epochs, no ensemble or RL.")
    p.add_argument("--integrity-gate", dest="integrity_gate", action="store_true", default=True,
                   help="Fail fast when cached X/y lengths are inconsistent.")
    p.add_argument("--no-integrity-gate", dest="integrity_gate", action="store_false",
                   help="Disable strict cache integrity gate (not recommended).")
    p.add_argument("--auto-rebuild-on-mismatch", action="store_true",
                   help="If cache integrity fails, delete cache/sidecars and rebuild automatically.")
    p.add_argument(
        "--walk-forward-cv",
        action="store_true",
        help="Purged walk-forward CV (train past / val future, embargo=seq_len) instead of one split",
    )
    p.add_argument(
        "--walk-forward-folds",
        type=int,
        default=None,
        help="Number of walk-forward folds (default: TRAINING['walk_forward_folds'])",
    )
    p.add_argument(
        "--early-stop-metric",
        type=str,
        default=None,
        choices=["loss", "sharpe"],
        help="Checkpoint early stopping on val loss or validation Sharpe proxy (default: TRAINING)",
    )

    # ── Pre-parse to find --config, then apply YAML defaults before full parse ──
    pre, _ = p.parse_known_args()
    if pre.config:
        _apply_yaml_config(p, pre.config)

    args = p.parse_args()
    if args.val_split is None:
        args.val_split = float(TRAINING["val_split"])
    if args.loss is None:
        args.loss = str(TRAINING.get("loss", "huber"))
    if args.walk_forward_folds is None:
        args.walk_forward_folds = int(TRAINING.get("walk_forward_folds", 6))
    if args.early_stop_metric is None:
        args.early_stop_metric = str(TRAINING.get("early_stop_metric", "sharpe"))
    if args.quick_mode:
        args.walk_forward_cv = True
        args.walk_forward_folds = min(max(int(args.walk_forward_folds), 1), 2)
        args.epochs = min(int(args.epochs), 8)
        args.pretrain_epochs = min(int(args.pretrain_epochs), 5)
        args.patience = min(int(args.patience), 4)
        args.train_ensemble = False
        args.rl_train = False
        print(f"[Quick] ON | folds={args.walk_forward_folds} | epochs={args.epochs} | "
              f"pretrain_epochs={args.pretrain_epochs} | ensemble=off | rl=off")
    return args


def apply_hardware_profile(args):
    """Override training paths and loader settings for a known local GPU/RAM combo."""
    name = getattr(args, "hardware_profile", None)
    if not name:
        return
    prof = HARDWARE_PROFILES.get(name)
    if not prof:
        return
    for k, v in prof.items():
        if k == "local_project_paths":
            continue
        setattr(args, k, v)
    if prof.get("local_project_paths"):
        args.checkpoint_dir = PATHS["checkpoints"]
        args.data_cache = PATHS["data_processed"]
    print(f"[Hardware] profile={name} | batch={args.batch_size} | workers={args.num_workers} | "
          f"chunk={args.chunk_size} | prefetch={args.prefetch_factor}")
    print(f"             checkpoint_dir={args.checkpoint_dir} | data_cache={args.data_cache}")


# ─────────────────────────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_device():
    if not torch.cuda.is_available():
        print("[GPU] No CUDA — using CPU (very slow for 20M ticks)")
        return torch.device("cpu"), 1
    n   = torch.cuda.device_count()
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    for i in range(n):
        g = torch.cuda.get_device_properties(i)
        vram = g.total_memory / 1e9
        print(f"[GPU {i}] {g.name} | {vram:.0f}GB VRAM | CUDA {torch.version.cuda}")
        if vram < 12:
            print(f"  NOTE: {vram:.0f}GB VRAM (e.g. RTX 4060). Use --hardware-profile "
                  f"rtx_4060_16gb_ram or --batch-size 384–512 if you hit OOM.")
    return dev, n


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CHUNKED DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _get_pairs(args) -> List[str]:
    """Return the list of pairs to train on, from --pairs or --pair."""
    raw = getattr(args, "pairs", None)
    if not raw:
        return [args.pair.upper()]
    if isinstance(raw, list):
        return [p.strip().upper() for p in raw if p and p.strip()]
    return [p.strip().upper() for p in str(raw).split(",") if p.strip()]


def _get_cache_path(args) -> Path:
    pairs    = _get_pairs(args)
    pair_tag = "-".join(sorted(pairs))
    tag      = f"{pair_tag}_{args.n_ticks}_{args.data_source}_{args.seq_len}_{args.label_method}"
    return Path(args.data_cache) / f"dataset_{tag}.h5"


def _load_scaler_from_attrs(f) -> Optional[StandardScaler]:
    """Restore StandardScaler from HDF5 attrs or None if missing (legacy cache)."""
    attrs = f.attrs
    if "scaler_mean" not in attrs:
        return None
    s = StandardScaler()
    s.mean_ = np.asarray(attrs["scaler_mean"], dtype=np.float64)
    s.scale_ = np.asarray(attrs["scaler_scale"], dtype=np.float64)
    s.var_ = np.asarray(attrs["scaler_var"], dtype=np.float64)
    s.n_features_in_ = int(attrs["scaler_n_features"])
    if "scaler_n_samples_seen" in attrs:
        s.n_samples_seen_ = int(attrs["scaler_n_samples_seen"])
    return s


def _write_scaler_attrs(h5_file, scaler: StandardScaler) -> None:
    if not hasattr(scaler, "mean_") or scaler.mean_ is None:
        return
    h5_file.attrs["scaler_mean"] = scaler.mean_
    h5_file.attrs["scaler_scale"] = scaler.scale_
    h5_file.attrs["scaler_var"] = scaler.var_
    h5_file.attrs["scaler_n_features"] = int(scaler.n_features_in_)
    h5_file.attrs["scaler_n_samples_seen"] = int(getattr(scaler, "n_samples_seen_", 0) or 0)


def _scaler_npz_path(cache_path: Path) -> Path:
    return Path(str(cache_path).replace(".h5", "_scaler.npz"))


def _x_path(cache_path) -> str:
    return str(cache_path).replace(".h5", "_X.npy")


def _y_path(cache_path) -> str:
    return str(cache_path).replace(".h5", "_y.npy")


def _on_disk_sequence_count(cache_path: str) -> Optional[int]:
    """
    Rows actually readable by MemmapSequenceDataset.
    NPY sidecars take precedence when both exist (same rule as the Dataset).
    """
    px, py = Path(_x_path(cache_path)), Path(_y_path(cache_path))
    if px.exists() and py.exists():
        X = np.load(str(px), mmap_mode="r")
        y = np.load(str(py), mmap_mode="r")
        return int(min(X.shape[0], y.shape[0]))
    if HDF5 and str(cache_path).endswith(".h5") and Path(cache_path).exists():
        with h5py.File(cache_path, "r") as f:
            return int(min(f["X"].shape[0], f["y"].shape[0]))
    return None


def _clamp_n_samples_to_disk(cache_path: str, n_samples: int) -> int:
    """Avoid OOB in workers when HDF5 metadata and NPY/HDF5 lengths disagree."""
    n_disk = _on_disk_sequence_count(cache_path)
    if n_disk is None or n_disk >= n_samples:
        return n_samples
    print(f"[Data] WARN: on-disk arrays have {n_disk:,} rows but pipeline reported "
          f"{n_samples:,} — clamping to {n_disk:,} (check X/Y export parity)")
    return n_disk


def _cache_length_snapshot(cache_path: str) -> dict:
    """
    Return cache lengths for integrity checks.
    Keys may include: h5_X, h5_y, npy_X, npy_y.
    """
    out: dict = {}
    p = Path(cache_path)
    if HDF5 and p.exists() and str(p).endswith(".h5"):
        try:
            with h5py.File(p, "r") as f:
                if "X" in f:
                    out["h5_X"] = int(f["X"].shape[0])
                if "y" in f:
                    out["h5_y"] = int(f["y"].shape[0])
        except Exception:
            out["h5_unreadable"] = 1
    px, py = Path(_x_path(cache_path)), Path(_y_path(cache_path))
    if px.exists():
        out["npy_X"] = int(np.load(str(px), mmap_mode="r").shape[0])
    if py.exists():
        out["npy_y"] = int(np.load(str(py), mmap_mode="r").shape[0])
    return out


def _validate_cache_integrity(cache_path: str) -> tuple[bool, str]:
    snap = _cache_length_snapshot(cache_path)
    problems = []
    p = Path(cache_path)
    if p.exists() and str(p).endswith(".h5") and HDF5:
        if snap.get("h5_unreadable", 0) == 1:
            problems.append("HDF5 file unreadable/corrupt")
        elif "h5_X" not in snap or "h5_y" not in snap:
            problems.append("HDF5 missing required dataset(s): X and/or y")
    if "h5_X" in snap and "h5_y" in snap and snap["h5_X"] != snap["h5_y"]:
        problems.append(f"HDF5 X={snap['h5_X']:,} != y={snap['h5_y']:,}")
    if "npy_X" in snap and "npy_y" in snap and snap["npy_X"] != snap["npy_y"]:
        problems.append(f"NPY X={snap['npy_X']:,} != y={snap['npy_y']:,}")
    if not problems:
        return True, ""
    return False, " | ".join(problems)


def _delete_cache_artifacts(cache_path: str) -> None:
    p = Path(cache_path)
    for fp in (p, Path(_x_path(cache_path)), Path(_y_path(cache_path)), _scaler_npz_path(p)):
        if fp.exists():
            fp.unlink()
            print(f"[Data] Removed corrupt cache artifact: {fp}")


def _core_model(model: "nn.Module") -> "nn.Module":
    return model.module if hasattr(model, "module") else model


def _identity_scaler(n_features: int) -> StandardScaler:
    s = StandardScaler()
    s.mean_ = np.zeros(n_features)
    s.scale_ = np.ones(n_features)
    s.var_ = np.ones(n_features)
    s.n_features_in_ = n_features
    return s


def _save_scaler_npz(cache_path: Path, scaler: StandardScaler) -> None:
    if not hasattr(scaler, "mean_") or scaler.mean_ is None:
        return
    p = _scaler_npz_path(cache_path)
    np.savez(
        p,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_features_in_=int(scaler.n_features_in_),
        n_samples_seen_=int(getattr(scaler, "n_samples_seen_", 0) or 0),
    )


def _load_scaler_npz(cache_path: Path) -> Optional[StandardScaler]:
    p = _scaler_npz_path(cache_path)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    s = StandardScaler()
    s.mean_ = np.asarray(z["mean"], dtype=np.float64)
    s.scale_ = np.asarray(z["scale"], dtype=np.float64)
    s.var_ = np.asarray(z["var"], dtype=np.float64)
    s.n_features_in_ = int(z["n_features_in_"])
    if "n_samples_seen_" in z.files:
        s.n_samples_seen_ = int(z["n_samples_seen_"])
    return s


def _build_chunk(
    ticks_chunk: "pd.DataFrame",
    fe:          FeatureEngineer,
    scaler:      StandardScaler,
    seq_len:     int,
    chunk_idx:   int,
    label_method: str = "rl_reward",
    cross_asset: Optional[Dict[str, "pd.Series"]] = None,
) -> "tuple[np.ndarray, np.ndarray, int]":
    """
    Process one tick chunk → (X_sequences, y_labels, n_features).
    Returns empty arrays if chunk is too small for sequences.
    """
    import pandas as pd

    # Graceful exit for empty/bad chunks (e.g., all vendor hour files missing/empty)
    if ticks_chunk is None or len(ticks_chunk) == 0:
        return np.array([]), np.array([]), 0
    if not isinstance(getattr(ticks_chunk, "index", None), pd.DatetimeIndex):
        return np.array([]), np.array([]), 0

    pipeline = ForexDataPipeline(bar_freq="1min", session_filter=False,
                                  apply_frac_diff=False)
    bars = pipeline.run(ticks_chunk)
    if len(bars) < seq_len + 20:
        return np.array([]), np.array([]), 0

    feats = fe.build(bars, cross_asset=cross_asset)
    if len(feats) < seq_len + 10:
        return np.array([]), np.array([]), 0

    if label_method == "triple_barrier":
        labels = compute_triple_barrier_labels(
            bars,
            feats,
            vertical_bars=LABELING["lookahead_bars"],
            profit_atr_mult=LABELING["profit_target_atr"],
            stop_atr_mult=LABELING["stop_loss_atr"],
            pip_size=LABELING["pip_size"],
        )
    else:
        labels = compute_rl_reward_labels(
            bars,
            feats,
            lookahead_bars=LABELING["lookahead_bars"],
            profit_atr_mult=LABELING["profit_target_atr"],
            stop_atr_mult=LABELING["stop_loss_atr"],
            tx_cost_pips=LABELING["transaction_cost_pips"],
            pip_size=LABELING["pip_size"],
        )
    X, y = align_labels_with_features(labels, feats)
    if len(X) < seq_len:
        return np.array([]), np.array([]), 0

    X_arr = X.values.astype(np.float32)
    n_feat = X_arr.shape[1]

    scaler.partial_fit(X_arr)
    X_arr = scaler.transform(X_arr).astype(np.float32)
    y_arr = y.values.astype(np.float32)

    # Build sliding window sequences
    # Window i uses rows [i, i+seq_len), label is the bar at i+seq_len-1 (last bar).
    # sliding_window_view produces (N - seq_len + 1) windows;
    # labels start at index seq_len-1 so there are (N - seq_len + 1) of them too.
    n_seq = len(X_arr) - seq_len + 1
    if n_seq <= 0:
        return np.array([]), np.array([]), n_feat

    X_seq = np.lib.stride_tricks.sliding_window_view(
        X_arr, (seq_len, n_feat)
    ).squeeze(1)                        # (n_seq, seq_len, n_feat)
    y_seq = y_arr[seq_len - 1:]        # label = last bar of each window

    return X_seq, y_seq, n_feat


def _scaler_npz_path_pair(cache_path: Path, pair: str) -> Path:
    """Per-pair scaler sidecar: dataset_EURUSD-GBPUSD_..._scaler_EURUSD.npz"""
    return Path(str(cache_path).replace(".h5", f"_scaler_{pair}.npz"))


def _build_multipair_chunk(
    pair_ticks:   dict,
    fe:           "FeatureEngineer",
    scalers:      dict,
    seq_len:      int,
    chunk_idx:    int,
    label_method: str,
    cross_asset: Optional[Dict[str, "pd.Series"]] = None,
) -> "tuple[np.ndarray, np.ndarray, int]":
    """
    Process raw ticks for P pairs into joint sequences.

    pair_ticks : {pair_name: pd.DataFrame}
    scalers    : {pair_name: StandardScaler}  — one per pair, fitted in-place

    Returns
    -------
    X : (N, T, P * F_per_pair)   — pairs concatenated on feature axis
    y : (N,)                      — mean label across pairs
    n_features_total : int
    """
    pair_Xs: dict = {}
    pair_ys: dict = {}

    for pair, ticks in pair_ticks.items():
        X_seq, y_seq, _ = _build_chunk(
            ticks, fe, scalers[pair], seq_len, chunk_idx, label_method,
            cross_asset=cross_asset,
        )
        if X_seq.size == 0:
            continue
        pair_Xs[pair] = X_seq   # (N, T, F)
        pair_ys[pair] = y_seq   # (N,)

    if not pair_Xs:
        return np.array([]), np.array([]), 0

    # Align to shortest pair (inner join on sample axis)
    n_min  = min(v.shape[0] for v in pair_Xs.values())
    X_list = [v[:n_min] for v in pair_Xs.values()]
    y_list = [v[:n_min] for v in pair_ys.values()]

    X_multi = np.concatenate(X_list, axis=2)                                  # (N, T, P*F)
    y_multi = np.mean(np.stack(y_list, axis=1), axis=1).astype(np.float32)   # (N,)
    return X_multi, y_multi, X_multi.shape[2]


def _build_multipair_dataset(
    args,
    pairs:      List[str],
    cache_path: Path,
    fe:         "FeatureEngineer",
) -> "tuple[str, int, int, StandardScaler]":
    """
    Multi-pair variant of build_dataset_chunked.
    Loads ticks for all pairs in parallel (dukascopy) or sequentially (other sources),
    builds joint (N, T, P*F) sequences, and writes them to the same HDF5 format.
    Returns (cache_path, n_samples, n_features, first_pair_scaler).
    """
    print(f"\n[MultiPair] {len(pairs)} pairs: {', '.join(pairs)}")
    print(f"            Source: {args.data_source} | {args.data_start} → {args.data_end}")
    use_real_cross = (
        args.cross_asset_mode == "real"
        or (args.cross_asset_mode == "auto" and args.data_source != "synthetic")
    )
    cross_asset_source = str(os.getenv("CROSS_ASSET_SOURCE", "auto") or "auto").strip().lower()
    cross_asset_cache_dir = str(
        os.getenv("CROSS_ASSET_CACHE_DIR", str(Path(args.data_cache) / "cross_asset"))
    ).strip()
    cross_asset = None
    if use_real_cross:
        try:
            cross_asset = load_cross_asset_panel(
                start=args.data_start,
                end=args.data_end,
                cache_dir=cross_asset_cache_dir,
                source=cross_asset_source,
            )
            print(f"[CrossAsset] Loaded external assets: {len(cross_asset)} "
                  f"(source={cross_asset_source}, cache={cross_asset_cache_dir})")
        except Exception as e:
            print(f"[CrossAsset] WARN: external load failed ({e}) — falling back to synthetic")
            cross_asset = None

    scalers       = {p: StandardScaler() for p in pairs}
    n_features    = 0
    total_samples = 0
    h5_file       = None

    if HDF5:
        h5_file = h5py.File(cache_path, "w")
    else:
        X_chunks: list = []
        y_chunks: list = []

    # ── Load ticks ──────────────────────────────────────────────────────────
    if args.data_source == "synthetic":
        n_remaining = args.n_ticks
        chunk_n     = 0
        while n_remaining > 0:
            chunk_n_ticks = min(args.chunk_size, n_remaining)
            pair_ticks = {p: generate_synthetic_tick_data(n_rows=chunk_n_ticks) for p in pairs}
            X_seq, y_seq, n_feat = _build_multipair_chunk(
                pair_ticks, fe, scalers, args.seq_len, chunk_n, args.label_method,
                cross_asset=cross_asset,
            )
            if X_seq.size > 0:
                n_features     = n_feat
                total_samples += len(X_seq)
                if HDF5:
                    if "X" not in h5_file:
                        chunk0 = max(1, min(512, int(len(X_seq))))
                        h5_file.create_dataset("X", data=X_seq, maxshape=(None,) + X_seq.shape[1:],
                                               compression="lzf", chunks=(chunk0,) + X_seq.shape[1:])
                        h5_file.create_dataset("y", data=y_seq, maxshape=(None,),
                                               compression="lzf", chunks=(chunk0,))
                    else:
                        old = h5_file["X"].shape[0]
                        h5_file["X"].resize(old + len(X_seq), axis=0)
                        h5_file["y"].resize(old + len(y_seq), axis=0)
                        h5_file["X"][old:] = X_seq
                        h5_file["y"][old:] = y_seq
                    h5_file.flush()
                else:
                    X_chunks.append(X_seq)
                    y_chunks.append(y_seq)
                pct = min((args.n_ticks - n_remaining + chunk_n_ticks) / args.n_ticks * 100, 100)
                print(f"  Chunk {chunk_n+1} | {len(X_seq):,} seqs | {pct:.0f}%")
            n_remaining -= chunk_n_ticks
            chunk_n     += 1

    else:
        # Real data: load all pairs at once then process
        if args.data_source == "dukascopy":
            from data.sources import DukascopyLoader
            hours  = None if getattr(args, "full_day_data", False) else list(range(7, 18))
            loader = DukascopyLoader(verbose=True)
            pair_ticks = loader.load_multiple(
                pairs=pairs, start=args.data_start, end=args.data_end, hours=hours,
            )
        else:
            mgr        = ForexDataManager(verbose=True)
            pair_ticks = {}
            for p in pairs:
                print(f"  Loading {p}...")
                pair_ticks[p] = mgr.load(
                    pair         = p,
                    source       = args.data_source,
                    start        = args.data_start,
                    end          = args.data_end,
                    session_only = not getattr(args, "full_day_data", False),
                )

        X_seq, y_seq, n_feat = _build_multipair_chunk(
            pair_ticks, fe, scalers, args.seq_len, 0, args.label_method,
            cross_asset=cross_asset,
        )
        if X_seq.size > 0:
            n_features    = n_feat
            total_samples = len(X_seq)
            if HDF5:
                chunk0 = max(1, min(512, int(len(X_seq))))
                h5_file.create_dataset("X", data=X_seq, compression="lzf", chunks=(chunk0,) + X_seq.shape[1:])
                h5_file.create_dataset("y", data=y_seq, compression="lzf", chunks=(chunk0,))
                h5_file.flush()
            else:
                X_chunks.append(X_seq)
                y_chunks.append(y_seq)
            print(f"  {total_samples:,} joint sequences × {n_features} features")

    if total_samples == 0:
        raise RuntimeError(
            "[MultiPair] No usable samples produced. Check date range and data source."
        )

    # ── Finalise cache ───────────────────────────────────────────────────────
    if HDF5:
        h5_file.attrs["n_features"] = n_features
        h5_file.attrs["seq_len"]    = args.seq_len
        h5_file.attrs["n_pairs"]    = len(pairs)
        h5_file.attrs["pairs"]      = ",".join(pairs)
        _write_scaler_attrs(h5_file, scalers[pairs[0]])   # first pair for compat
        h5_file.close()
    else:
        X_all = np.concatenate(X_chunks, axis=0)
        y_all = np.concatenate(y_chunks, axis=0)
        np.save(_x_path(cache_path), X_all)
        np.save(_y_path(cache_path), y_all)
        del X_all, y_all

    # Save per-pair scalers
    for p, sc in scalers.items():
        _save_scaler_npz(_scaler_npz_path_pair(cache_path, p), sc)

    print(f"\n[MultiPair] Dataset built: {total_samples:,} samples × {n_features} features")
    print(f"            Cached: {cache_path}")
    return str(cache_path), total_samples, n_features, scalers[pairs[0]]


def build_dataset_chunked(args) -> "tuple[str, int, int, StandardScaler]":
    """
    Ingest up to 20M ticks in chunks, write sequences to HDF5.

    Returns: (cache_path, n_samples, n_features, scaler)
    """
    import pandas as pd

    pairs    = _get_pairs(args)
    is_multi = len(pairs) > 1

    # Multi-pair: delegate to dedicated function
    if is_multi:
        Path(args.data_cache).mkdir(parents=True, exist_ok=True)
        cache_path = _get_cache_path(args)
        if cache_path.exists() and not args.force_rebuild:
            ok, reason = _validate_cache_integrity(str(cache_path))
            if not ok:
                if getattr(args, "auto_rebuild_on_mismatch", False):
                    print(f"[MultiPair] Cache mismatch ({reason}) — rebuilding.")
                    _delete_cache_artifacts(str(cache_path))
                elif getattr(args, "integrity_gate", True):
                    raise RuntimeError(f"Cache integrity failed: {reason}. Use --force-rebuild.")
        if cache_path.exists() and not args.force_rebuild:
            if HDF5:
                with h5py.File(cache_path, "r") as f:
                    n_samples  = min(int(f["X"].shape[0]), int(f["y"].shape[0]))
                    n_features = f["X"].shape[2]
                    scaler     = _load_scaler_from_attrs(f) or _identity_scaler(n_features)
                n_samples = _clamp_n_samples_to_disk(str(cache_path), n_samples)
                print(f"[MultiPair] {n_samples:,} samples × {n_features} features (cached)")
                args._n_pairs    = len(pairs)
                args._f_per_pair = n_features // len(pairs)
                return str(cache_path), n_samples, n_features, scaler
        fe = FeatureEngineer(
            atr_window=FEATURES["atr_window"], ofi_window=FEATURES["ofi_window"],
            tar_window=FEATURES["trade_arrival_window"], rsi_period=FEATURES["rsi_period"],
            macd_fast=FEATURES["macd_fast"], macd_slow=FEATURES["macd_slow"],
            macd_signal=FEATURES["macd_signal"], bb_window=FEATURES["bollinger_window"],
            bb_std=FEATURES["bollinger_std"], lag_windows=FEATURES["lag_windows"],
        )
        cache_str, n_samples, n_features, scaler = _build_multipair_dataset(
            args, pairs, cache_path, fe,
        )
        args._n_pairs    = len(pairs)
        args._f_per_pair = n_features // len(pairs)
        return cache_str, n_samples, n_features, scaler

    cache_path = _get_cache_path(args)
    Path(args.data_cache).mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not args.force_rebuild:
        ok, reason = _validate_cache_integrity(str(cache_path))
        if not ok:
            if getattr(args, "auto_rebuild_on_mismatch", False):
                print(f"[Data] WARN: cache integrity mismatch ({reason}) — auto rebuilding.")
                _delete_cache_artifacts(str(cache_path))
            elif getattr(args, "integrity_gate", True):
                raise RuntimeError(
                    f"Cache integrity check failed: {reason}. "
                    "Run with --force-rebuild or --auto-rebuild-on-mismatch."
                )
        print(f"\n[Data] Found cached dataset: {cache_path}")
    if cache_path.exists() and not args.force_rebuild:
        if HDF5:
            with h5py.File(cache_path, "r") as f:
                n_samples  = min(int(f["X"].shape[0]), int(f["y"].shape[0]))
                n_features = f["X"].shape[2]
                scaler = _load_scaler_from_attrs(f)
                if scaler is None:
                    scaler = _identity_scaler(n_features)
                    print("[Data] WARN: cached HDF5 has no scaler attrs — identity scaling "
                          "(rebuild with --force-rebuild to fit partial StandardScaler)")
            # MemmapSequenceDataset reads NPY when sidecars exist — lengths must match
            n_samples = _clamp_n_samples_to_disk(str(cache_path), n_samples)
            print(f"[Data] {n_samples:,} samples × {n_features} features (from cache)")
            return str(cache_path), n_samples, n_features, scaler
        else:
            X_path = _x_path(cache_path)
            if Path(X_path).exists():
                X_mmap = np.load(X_path, mmap_mode="r")
                y_mmap = np.load(_y_path(str(cache_path)), mmap_mode="r")
                n_samples = min(X_mmap.shape[0], y_mmap.shape[0])
                n_features_cached = X_mmap.shape[2]
                scaler = _load_scaler_npz(Path(cache_path))
                if scaler is None:
                    scaler = _identity_scaler(n_features_cached)
                    print("[Data] WARN: no _scaler.npz next to cache — identity scaling")
                if X_mmap.shape[0] != y_mmap.shape[0]:
                    print(f"[Data] WARN: X has {X_mmap.shape[0]:,} rows but y has {y_mmap.shape[0]:,} — using {n_samples:,}")
                n_samples = _clamp_n_samples_to_disk(str(cache_path), n_samples)
                print(f"[Data] {n_samples:,} samples × {n_features_cached} features (from NPY cache)")
                return str(cache_path), n_samples, n_features_cached, scaler

    print(f"\n[Data] Building 20M tick dataset — chunk size: {args.chunk_size:,}")
    _pairs_display = ", ".join(_get_pairs(args))
    print(f"       Source: {args.data_source} | Pairs: {_pairs_display}")
    use_real_cross = (
        args.cross_asset_mode == "real"
        or (args.cross_asset_mode == "auto" and args.data_source != "synthetic")
    )
    cross_asset_source = str(os.getenv("CROSS_ASSET_SOURCE", "auto") or "auto").strip().lower()
    cross_asset_cache_dir = str(
        os.getenv("CROSS_ASSET_CACHE_DIR", str(Path(args.data_cache) / "cross_asset"))
    ).strip()
    cross_asset = None
    if use_real_cross:
        try:
            cross_asset = load_cross_asset_panel(
                start=args.data_start,
                end=args.data_end,
                cache_dir=cross_asset_cache_dir,
                source=cross_asset_source,
            )
            print(f"[CrossAsset] Loaded external assets: {len(cross_asset)} "
                  f"(source={cross_asset_source}, cache={cross_asset_cache_dir})")
        except Exception as e:
            print(f"[CrossAsset] WARN: external load failed ({e}) — falling back to synthetic")
            cross_asset = None

    fe     = FeatureEngineer(
        atr_window  = FEATURES["atr_window"],
        ofi_window  = FEATURES["ofi_window"],
        tar_window  = FEATURES["trade_arrival_window"],
        rsi_period  = FEATURES["rsi_period"],
        macd_fast   = FEATURES["macd_fast"],
        macd_slow   = FEATURES["macd_slow"],
        macd_signal = FEATURES["macd_signal"],
        bb_window   = FEATURES["bollinger_window"],
        bb_std      = FEATURES["bollinger_std"],
        lag_windows = FEATURES["lag_windows"],
    )
    scaler  = StandardScaler()
    n_remaining = args.n_ticks
    chunk_n     = 0
    total_samples = 0
    n_features  = 0
    h5_file     = None

    if HDF5:
        h5_file = h5py.File(cache_path, "w")
    else:
        X_chunks: list = []; y_chunks: list = []

    # ── Load all ticks or generate in chunks ─────────────────────────────────
    if args.data_source == "synthetic":
        print(f"[Data] Generating {args.n_ticks:,} synthetic ticks in chunks...")

    while n_remaining > 0:
        chunk_ticks = min(args.chunk_size, n_remaining)
        t0 = time.time()

        # Load / generate this chunk
        if args.data_source == "synthetic":
            ticks_chunk = generate_synthetic_tick_data(n_rows=chunk_ticks)
        else:
            print(
                f"[Data] Loading {args.data_source} for {args.pair} "
                f"({args.data_start} → {args.data_end}). "
                "First run downloads many hourly files; this can take tens of minutes to hours."
            )
            mgr = ForexDataManager(verbose=True)
            ticks_chunk = mgr.load(
                pair   = args.pair,
                source = args.data_source,
                start  = args.data_start,
                end    = args.data_end,
                session_only = (not getattr(args, "full_day_data", False)),
            )
            # If real data, use all of it (ignore n_ticks cap)
            n_remaining = 0

        X_seq, y_seq, n_feat = _build_chunk(
            ticks_chunk, fe, scaler,
            seq_len    = args.seq_len,
            chunk_idx  = chunk_n,
            label_method = args.label_method,
            cross_asset = cross_asset,
        )
        del ticks_chunk; gc.collect()

        if X_seq.size == 0:
            n_remaining -= chunk_ticks; chunk_n += 1; continue

        n_features = n_feat
        total_samples += len(X_seq)

        if HDF5:
            if "X" not in h5_file:
                chunk0 = max(1, min(512, int(len(X_seq))))
                h5_file.create_dataset("X", data=X_seq, maxshape=(None,)+X_seq.shape[1:],
                                        compression="lzf", chunks=(chunk0,)+X_seq.shape[1:])
                h5_file.create_dataset("y", data=y_seq, maxshape=(None,),
                                        compression="lzf", chunks=(chunk0,))
            else:
                old = h5_file["X"].shape[0]
                h5_file["X"].resize(old + len(X_seq), axis=0)
                h5_file["y"].resize(old + len(y_seq), axis=0)
                h5_file["X"][old:] = X_seq
                h5_file["y"][old:] = y_seq
            h5_file.flush()
        else:
            X_chunks.append(X_seq); y_chunks.append(y_seq)

        elapsed = time.time() - t0
        done    = args.n_ticks - n_remaining + chunk_ticks
        pct     = min(done / args.n_ticks * 100, 100)
        print(f"  Chunk {chunk_n+1} | {len(X_seq):,} seqs | "
              f"{elapsed:.1f}s | {pct:.0f}% ({total_samples:,} total)")

        n_remaining -= chunk_ticks; chunk_n += 1

    if total_samples == 0:
        raise RuntimeError(
            "[Data] No usable samples were produced from the selected date range/source. "
            "Likely causes: vendor returned mostly empty hour files, wrong pair/date range, "
            "or blocked data endpoint. Try a shorter recent range first and verify raw cache."
        )

    if HDF5:
        h5_file.attrs["n_features"] = n_features
        h5_file.attrs["seq_len"]    = args.seq_len
        _write_scaler_attrs(h5_file, scaler)
        h5_file.close()
    else:
        X_all = np.concatenate(X_chunks, axis=0)
        y_all = np.concatenate(y_chunks, axis=0)
        np.save(_x_path(cache_path), X_all); np.save(_y_path(cache_path), y_all)
        _save_scaler_npz(cache_path, scaler)
        del X_all, y_all; gc.collect()

    print(f"\n[Data] Dataset built: {total_samples:,} samples × "
          f"{n_features} features × seq_len {args.seq_len}")
    print(f"       Cached at: {cache_path}")
    return str(cache_path), total_samples, n_features, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY-MAPPED SEQUENCE DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MemmapSequenceDataset(Dataset):
    """
    Reads pre-built sequences directly from HDF5 or NPY on disk.
    Never loads the full dataset into RAM — GPU workers stream batches
    asynchronously, keeping the GPU fed at full throughput.

    Why this is fast:
      - Linux page cache pre-fetches the next HDF5 chunk while GPU trains
      - num_workers=8 parallel readers
      - pin_memory=True eliminates GPU↔CPU copy latency
      - persistent_workers=True avoids worker restart overhead
    """

    def __init__(self, cache_path: str, indices: np.ndarray):
        self.cache_path = cache_path
        # Own memory: avoids numpy pickling a *view* whose base is huge, and
        # keeps worker pickles small for Windows spawn + DataLoader.
        self.indices = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
        self._h5 = None   # Opened per-worker in __getitem__

        # Prefer NPY memmap over HDF5 when available — O(1) random access,
        # no file locking, no per-seek overhead. HDF5 is kept as fallback.
        npy_x = Path(_x_path(cache_path))
        npy_y = Path(_y_path(cache_path))
        self.use_hdf5 = not (npy_x.exists() and npy_y.exists()) and cache_path.endswith(".h5") and HDF5
        if not self.use_hdf5:
            self.X_mmap = np.load(str(npy_x), mmap_mode="r")
            self.y_mmap = np.load(str(npy_y), mmap_mode="r")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        if self.use_hdf5:
            if self._h5 is None:
                self._h5 = h5py.File(self.cache_path, "r", swmr=True)
            X = self._h5["X"][real_idx]
            y = float(self._h5["y"][real_idx])
        else:
            X = np.array(self.X_mmap[real_idx])
            y = float(self.y_mmap[real_idx])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __getstate__(self):
        # Never pickle memmaps or h5py handles — default pickling can materialize
        # the full array on Windows (MemoryError in multiprocessing.spawn).
        return {
            "cache_path": self.cache_path,
            "indices": self.indices,
            "use_hdf5": self.use_hdf5,
        }

    def __setstate__(self, state):
        self.cache_path = state["cache_path"]
        self.indices = state["indices"]
        self.use_hdf5 = state["use_hdf5"]
        self._h5 = None
        if not self.use_hdf5:
            npy_x = Path(_x_path(self.cache_path))
            npy_y = Path(_y_path(self.cache_path))
            self.X_mmap = np.load(str(npy_x), mmap_mode="r")
            self.y_mmap = np.load(str(npy_y), mmap_mode="r")

    def __del__(self):
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# SPLITS + LABEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(n_samples: int, n_folds: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward: each fold trains on [0, val_start - embargo),
    validates on [val_start, val_end). Prevents overlap leakage via embargo (e.g. seq_len).
    """
    if n_samples < max(embargo + n_folds * 2, 500):
        split = int(n_samples * (1 - float(TRAINING.get("val_split", 0.2))))
        return [(np.arange(0, split), np.arange(split, n_samples))]
    edges = np.linspace(0, n_samples, n_folds + 2, dtype=np.int64)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        va, vb = int(edges[k + 1]), int(edges[k + 2])
        train_end = max(0, va - int(embargo))
        tr = np.arange(0, train_end, dtype=np.int64)
        va_idx = np.arange(va, vb, dtype=np.int64)
        if len(tr) < 100 or len(va_idx) < 10:
            continue
        out.append((tr, va_idx))
    if not out:
        split = int(n_samples * 0.8)
        return [(np.arange(0, split), np.arange(split, n_samples))]
    return out


def _read_y_indices(cache_path: str, indices: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    parts: list[np.ndarray] = []
    if cache_path.endswith(".h5") and HDF5:
        with h5py.File(cache_path, "r") as f:
            y = f["y"]
            for s in range(0, len(indices), chunk):
                sl = indices[s : s + chunk]
                parts.append(np.asarray(y[sl]))
    else:
        ym = np.load(_y_path(cache_path), mmap_mode="r")
        for s in range(0, len(indices), chunk):
            sl = indices[s : s + chunk]
            parts.append(np.asarray(ym[sl]))
    return np.concatenate(parts) if parts else np.array([])


def _class_weights_tensor(
    cache_path: str, train_idx: np.ndarray, device: "torch.device", max_samples: int = 2_000_000,
) -> torch.Tensor:
    if len(train_idx) > max_samples:
        sub = np.random.choice(train_idx, max_samples, replace=False)
    else:
        sub = train_idx
    y_raw = _read_y_indices(cache_path, np.sort(sub))

    # Round to the nearest integer label and cast to int8 so sklearn
    # never hits float32 vs float64 comparison issues.
    y = np.round(y_raw.astype(np.float64)).astype(np.int8)
    # Keep only the three valid direction labels; drop any noise values.
    y = y[np.isin(y, [-1, 0, 1])]

    # Default weight 1.0 for every class slot: index 0=-1, 1=0, 2=+1
    weights = np.ones(3, dtype=np.float32)

    present = np.unique(y)          # guaranteed int8 subset of {-1, 0, 1}
    if len(present) > 0:
        w = compute_class_weight("balanced",
                                  classes=present.astype(np.int64),
                                  y=y.astype(np.int64))
        for label, val in zip(present, w):
            idx = int(label) + 1    # -1 → 0,  0 → 1,  +1 → 2
            if 0 <= idx < 3:
                weights[idx] = float(val)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def labels_to_class_index(yb: "torch.Tensor") -> "torch.Tensor":
    return (yb + 1.0).round().long().clamp(0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _multitask_head_in(model_name: str, args, n_features: int) -> int:
    """
    Return the dimensionality of each backbone's pre-head hidden state.
    This is the input size for MultiTaskWrapper / MultiTaskHead.

    Derived from each architecture's forward() — the tensor fed into self.head
    before the MultiTaskWrapper replaces it with nn.Identity().
    """
    m = model_name.lower()
    if m == "tft":         return args.hidden_size
    # iTransformer: out.reshape(B, F * d_model) → large, wrapper auto-projects
    if m == "transformer": return args.d_model * n_features
    # HAELT: cat([lstm_feat, trf_feat]) of size lstm_hidden + d_model
    if m == "haelt":       return (args.hidden_size // 2) + (args.d_model // 2)
    if m == "mamba":       return args.d_model
    # GNNFromSequence: h.reshape(B, hidden * n_nodes)
    if m == "gnn":         return args.hidden_size * 6
    if m == "expert":      return args.d_model
    return args.hidden_size


def build_model(name: str, n_features: int, args) -> nn.Module:
    # ── Multi-pair embedding expansion ────────────────────────────────────────
    n_pairs     = getattr(args, "_n_pairs", 1)
    f_per_pair  = getattr(args, "_f_per_pair", n_features)
    embed_dim   = getattr(args, "pair_embed_dim", 0)
    use_pair_emb = n_pairs > 1 and embed_dim > 0
    # When using pair embeddings the backbone sees a wider feature space
    backbone_input = n_pairs * (f_per_pair + embed_dim) if use_pair_emb else n_features

    # Multitask wrapper adds its own 3-class head; base always uses nc=1
    multitask = getattr(args, "multitask", False)
    nc = 1 if multitask else (3 if args.loss == "cross_entropy" else 1)
    builders = {
        "tft":         lambda: TFTScalper(
                           input_size=backbone_input, hidden=args.hidden_size,
                           heads=min(8,args.nhead), lstm_layers=args.num_layers,
                           dropout=args.dropout, num_classes=nc),
        "transformer": lambda: iTransformerScalper(
                           input_size=backbone_input, seq_len=args.seq_len,
                           d_model=args.d_model, nhead=args.nhead,
                           num_layers=args.num_layers, dropout=args.dropout,
                           num_classes=nc),
        "haelt":       lambda: HAELTHybrid(
                           input_size=backbone_input, seq_len=args.seq_len,
                           lstm_hidden=args.hidden_size//2, d_model=args.d_model//2,
                           nhead=max(2,args.nhead//2), n_layers=args.num_layers,
                           dropout=args.dropout, num_classes=nc),
        "mamba":       lambda: MambaScalper(
                           input_size=backbone_input, d_model=args.d_model,
                           num_layers=args.num_layers, dropout=args.dropout,
                           num_classes=nc),
        "gnn":         lambda: GNNFromSequence(
                           input_size=backbone_input, hidden=args.hidden_size,
                           num_layers=args.num_layers, dropout=args.dropout,
                           n_nodes=6, num_classes=nc, nhead=min(4, args.nhead)),
        "expert":      lambda: EXPERTEncoder(
                           input_size=backbone_input, d_model=args.d_model,
                           nhead=args.nhead, num_layers=args.num_layers,
                           dropout=args.dropout, num_classes=nc),
    }
    m = builders[name]()

    # Pair embedding wrapper (only when embed_dim > 0 and training on multiple pairs)
    if use_pair_emb:
        m = MultiPairWrapper(m, n_pairs=n_pairs, f_per_pair=f_per_pair, embed_dim=embed_dim)
        print(f"[Model] {name.upper()} | MultiPair wrapper "
              f"({n_pairs} pairs × {f_per_pair}F + {embed_dim}E embed) | "
              f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")
    elif n_pairs > 1:
        print(f"[Model] {name.upper()} | {n_pairs} pairs × {f_per_pair}F concatenated | "
              f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

    if multitask:
        head_in = _multitask_head_in(name, args, backbone_input)
        m = MultiTaskWrapper(
            m, head_in=head_in,
            hidden=64, dropout=args.dropout,
            proj_threshold=1024, proj_to=256,
        )
        print(f"[Model] {name.upper()} | MultiTask wrapper (head_in={head_in}) | "
              f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")
    elif n_pairs == 1:
        n = sum(p.numel() for p in m.parameters())
        print(f"[Model] {name.upper()} | {n/1e6:.2f}M parameters")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _compute_loss(model_out, crit, yb, classification: bool) -> "torch.Tensor":
    """
    Unified loss computation for both single-head and MultiTaskWrapper outputs.
    MultiTaskWrapper returns (direction_logits, return_hat, confidence) tuple.
    Single-head models return a scalar / 3-class logit tensor.
    """
    if isinstance(model_out, tuple):
        # MultiTask: crit is MultiTaskLoss
        logits, ret_hat, conf = model_out
        return crit(logits, ret_hat, conf, labels_to_class_index(yb), yb)
    elif classification:
        return crit(model_out, labels_to_class_index(yb))
    else:
        return crit(model_out, yb)


def train_epoch(model, loader, opt, crit, scaler_amp, device, use_amp, classification: bool,
                grad_clip: float = 1.0, pbar=None):
    model.train()
    total = 0.0; n = 0; oom_skips = 0
    for xb, yb in loader:
        try:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                if device.type == "cuda":
                    with autocast("cuda"):
                        loss = _compute_loss(model(xb), crit, yb, classification)
                else:
                    loss = _compute_loss(model(xb), crit, yb, classification)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler_amp.step(opt); scaler_amp.update()
            else:
                loss = _compute_loss(model(xb), crit, yb, classification)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            total += loss.item(); n += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        except RuntimeError as e:
            is_oom = ("out of memory" in str(e).lower()) or ("cuda error" in str(e).lower())
            if not (device.type == "cuda" and is_oom):
                raise
            oom_skips += 1
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss="OOM-skip")
            if oom_skips <= 3 or oom_skips % 10 == 0:
                print(f"[Train] WARN: CUDA OOM on batch, skipped ({oom_skips} total). "
                      "Tip: resume with smaller --batch-size (e.g., 256).")
            continue
    if oom_skips:
        print(f"[Train] OOM summary: skipped {oom_skips} batch(es) this epoch.")
    return total / max(n, 1)


def build_criterion(
    args,
    device: "torch.device",
    cache_path: Optional[str] = None,
    train_idx: Optional[np.ndarray] = None,
):
    """
    Huber / asymmetric / directional_huber / sharpe_huber regression,
    weighted CE on {-1,0,+1}, or MultiTaskLoss.
    MultiTaskLoss is selected when --multitask is passed and combines:
      w_dir*CE(direction) + w_ret*Huber(return_hat) + w_conf*BCE(confidence)
    """
    multitask = getattr(args, "multitask", False)
    d = float(TRAINING.get("huber_delta", 1.0))

    if multitask:
        # Compute balanced class weights for the direction head
        cw = None
        if cache_path is not None and train_idx is not None:
            cw = _class_weights_tensor(cache_path, train_idx, device)
        return MultiTaskLoss(
            class_weights=cw,
            w_dir=1.0,
            w_ret=float(getattr(args, "mt_w_ret",  0.5)),
            w_conf=float(getattr(args, "mt_w_conf", 0.3)),
            huber_delta=d,
        ).to(device)

    if args.loss == "cross_entropy":
        if cache_path is None or train_idx is None:
            raise ValueError("cross_entropy requires cache_path and train_idx")
        w = _class_weights_tensor(cache_path, train_idx, device)
        return nn.CrossEntropyLoss(weight=w)
    if args.loss == "asymmetric":
        sw = float(TRAINING.get("asymmetric_sign_weight", 2.0))
        return AsymmetricDirectionalLoss(delta=d, sign_weight=sw).to(device)
    if args.loss == "directional_huber":
        return DirectionalHuberLoss(
            delta=d,
            direction_weight=float(getattr(args, "direction_weight", 0.5)),
        ).to(device)
    if args.loss == "sharpe_huber":
        return SharpeProxyLoss(
            delta=d,
            sharpe_weight=float(getattr(args, "sharpe_weight", 0.2)),
        ).to(device)
    return HuberLoss(delta=d).to(device)


@torch.no_grad()
def validate_epoch(model, loader, crit, device, classification: bool, pbar=None):
    model.eval()
    total = 0.0; correct = 0; nt = 0
    r_sum = torch.zeros(1, device=device)
    r_sq_sum = torch.zeros(1, device=device)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred    = model(xb)
        y_cls   = labels_to_class_index(yb)
        if pbar is not None:
            pbar.update(1)

        if isinstance(pred, tuple):
            logits, ret_hat, conf = pred
            total   += _compute_loss(pred, crit, yb, classification).item()
            correct += (logits.argmax(-1) == y_cls).sum().item()
            d = logits.argmax(-1).float() - 1.0
        elif classification:
            total   += crit(pred, y_cls).item()
            correct += (pred.argmax(-1) == y_cls).sum().item()
            d = pred.argmax(-1).float() - 1.0
        else:
            total   += crit(pred, yb).item()
            correct += (torch.sign(pred) == torch.sign(yb)).sum().item()
            d = torch.sign(pred)

        r = (d * yb.float()).flatten()
        r_sum    += r.sum()
        r_sq_sum += (r * r).sum()
        nt += len(yb)

    ann = float(TRAINING.get("sharpe_annualization_factor", 1.0))
    if nt == 0:
        sharpe = 0.0
    else:
        r_mean = r_sum / nt
        r_var  = r_sq_sum / nt - r_mean ** 2
        sharpe = (r_mean / (r_var.sqrt() + 1e-8)).item() * ann
    return total / max(len(loader), 1), correct / max(nt, 1), sharpe


def supervised_train(
    model_name: str,
    cache_path: str,
    n_samples:  int,
    n_features: int,
    args,
    device:     "torch.device",
    n_gpus:     int,
    run: Any = None,
    train_idx:  Optional[np.ndarray] = None,
    val_idx:    Optional[np.ndarray] = None,
    fold_id:    Optional[int] = None,
):
    classification = args.loss == "cross_entropy"
    fold_suffix = f"_fold{fold_id}" if fold_id is not None else ""

    print(f"\n{'─'*60}")
    print(f"  Training: {model_name.upper()} | {n_samples:,} samples | "
          f"batch={args.batch_size} | AMP={args.amp} | loss={args.loss}")
    print(f"{'─'*60}")

    if train_idx is None or val_idx is None:
        split     = int(n_samples * (1 - args.val_split))
        train_idx = np.arange(0, split)
        val_idx   = np.arange(split, n_samples)
    print(f"[Split] Train: {len(train_idx):,} | Val: {len(val_idx):,}"
          f"{fold_suffix if fold_id is not None else ''}")

    train_ds = MemmapSequenceDataset(cache_path, train_idx)
    val_ds   = MemmapSequenceDataset(cache_path, val_idx)

    nw = min(args.num_workers, os.cpu_count() or 4)
    pf = args.prefetch_factor if nw > 0 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=nw, pin_memory=True, persistent_workers=nw>0,
                          prefetch_factor=pf)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False,
                          num_workers=max(2, nw//2), pin_memory=True,
                          persistent_workers=nw>0,
                          prefetch_factor=pf)
    print(f"[Loader] {len(train_dl)} train batches | {len(val_dl)} val batches | "
          f"{nw} workers | prefetch={pf if pf is not None else 0}")

    model = build_model(model_name, n_features, args).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model); print(f"[Model] DataParallel × {n_gpus} GPUs")

    crit = build_criterion(
        args, device,
        cache_path=cache_path if classification else None,
        train_idx=train_idx if classification else None,
    )
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    max_lr = float(TRAINING.get("onecycle_max_lr_mult", 10.0)) * args.lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=max_lr,
        epochs=args.epochs,
        steps_per_epoch=max(len(train_dl), 1),
        pct_start=float(TRAINING.get("onecycle_pct_start", 0.1)),
        anneal_strategy="cos",
    )
    amp_sc    = GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    ckpt_dir  = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{model_name}{fold_suffix}_best.pt"
    cfg_path  = ckpt_dir / f"{model_name}{fold_suffix}_config.json"

    stop_on_sharpe = args.early_stop_metric == "sharpe"

    # Resume (single-split only; skip per-fold resume id)
    start_ep = 0
    last_path = ckpt_dir / f"{model_name}{fold_suffix}_last.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        core = _core_model(model)
        core.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        if "scaler_state" in ck and ck["scaler_state"] is not None:
            amp_sc.load_state_dict(ck["scaler_state"])
        start_ep = int(ck.get("epoch", -1)) + 1
        best_val_loss = float(ck.get("best_val_loss", float("inf")))
        best_sharpe = float(ck.get("best_sharpe", float("-inf")))
        no_improve = int(ck.get("no_improve", 0))
        history = ck.get("history", {"train_loss":[], "val_loss":[], "dir_acc":[], "val_sharpe":[], "lr":[]})
        print(f"[Resume] Loaded exact state from {last_path} (epoch {start_ep})")
    elif args.resume and best_path.exists() and fold_id is None:
        core = _core_model(model)
        core.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[Resume] Loaded weights from {best_path}")

    if start_ep == 0:
        best_val_loss = float("inf")
        best_sharpe = float("-inf")
        no_improve = 0
        history  = {"train_loss":[], "val_loss":[], "dir_acc":[], "val_sharpe":[], "lr":[]}

    print(f"\n{'Ep':>5} {'Train':>11} {'Val':>11} {'DirAcc':>8} {'vSharpe':>9} "
          f"{'LR':>10} {'Time':>7} {'GPU MB':>8}")
    print("─" * 72)

    epoch_bar = _pbar(range(start_ep, args.epochs), desc=f"Train {model_name.upper()}", unit="ep")
    for ep in epoch_bar:
        # Batch-level progress bars
        train_pbar = _pbar(total=len(train_dl), desc=f"  Ep {ep+1:3d} [Tr]", unit="batch", leave=False)
        val_pbar   = _pbar(total=len(val_dl),   desc=f"  Ep {ep+1:3d} [Va]", unit="batch", leave=False)

        t0 = time.time()
        tl = train_epoch(
            model, train_dl, opt, crit, amp_sc, device, args.amp, classification,
            grad_clip=args.grad_clip, pbar=train_pbar
        )
        train_pbar.close()

        vl, da, v_sh = validate_epoch(model, val_dl, crit, device, classification, pbar=val_pbar)
        val_pbar.close()

        scheduler.step()

        lr = opt.param_groups[0]["lr"]
        el = time.time() - t0
        gm = torch.cuda.max_memory_allocated(device) // 1_000_000 if device.type=="cuda" else 0
        torch.cuda.reset_peak_memory_stats(device) if device.type=="cuda" else None

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["dir_acc"].append(da);    history["lr"].append(lr)
        history["val_sharpe"].append(v_sh)

        if WANDB and run:
            run.log({
                "train/loss": tl, "val/loss": vl, "val/dir_acc": da,
                "val/sharpe_proxy": v_sh, "train/lr": lr, "gpu_mb": gm, "epoch": ep,
                **({"fold": fold_id} if fold_id is not None else {}),
            })

        print(f"{ep+1:>5} {tl:>11.6f} {vl:>11.6f} {da:>8.4f} {v_sh:>9.4f} "
              f"{lr:>10.2e} {el:>6.1f}s {gm:>7.0f}M")

        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        improved = (v_sh > (best_sharpe + min_delta)) if stop_on_sharpe else (vl < (best_val_loss - min_delta))
        if improved:
            if stop_on_sharpe:
                best_sharpe = v_sh
            else:
                best_val_loss = vl
            no_improve = 0
            core = _core_model(model)
            torch.save(core.state_dict(), best_path)
            json.dump({
                "model": model_name, "n_features": n_features,
                "seq_len": args.seq_len, "d_model": args.d_model,
                "nhead": args.nhead, "hidden_size": args.hidden_size,
                "num_layers": args.num_layers, "dropout": args.dropout,
                "best_val_loss": vl, "best_val_sharpe_proxy": v_sh,
                "early_stop_metric": args.early_stop_metric,
                "epoch": ep, "n_samples": n_samples, "loss": args.loss,
                "fold_id": fold_id,
            }, open(cfg_path,"w"), indent=2)
            if WANDB and run:
                run.summary.update({
                    "best_val_loss": vl, "best_val_sharpe_proxy": v_sh, "best_epoch": ep,
                })
        else:
            no_improve += 1

        if (ep+1) % args.save_every == 0:
            core = _core_model(model)
            ep_tag = f"{fold_suffix}_ep{ep+1}" if fold_suffix else f"_ep{ep+1}"
            torch.save(core.state_dict(), ckpt_dir / f"{model_name}{ep_tag}.pt")

        # Exact resume checkpoint (model + optimizer + scheduler + AMP scaler + history)
        core = _core_model(model)
        torch.save({
            "epoch": ep,
            "model_state": core.state_dict(),
            "opt_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": amp_sc.state_dict() if args.amp and device.type == "cuda" else None,
            "best_val_loss": best_val_loss,
            "best_sharpe": best_sharpe,
            "no_improve": no_improve,
            "history": history,
            "fold_id": fold_id,
        }, last_path)

        if no_improve >= args.patience:
            print(f"\n[Train] Early stop (patience={args.patience}, "
                  f"metric={args.early_stop_metric})")
            break

    if stop_on_sharpe:
        print(f"\n[Train] Best val Sharpe (proxy): {best_sharpe:.4f}  →  {best_path}")
        return history, best_sharpe
    print(f"\n[Train] Best val loss: {best_val_loss:.6f}  →  {best_path}")
    return history, best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# CONTRASTIVE PRE-TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_pretrain(model, cache_path, n_features, args, device):
    use_regime = getattr(args, "pretrain_regime", False) and RegimeAwareTSCLTrainer is not None
    mode_str   = "RegimeAware-TSCL" if use_regime else "TSCL"
    print(f"\n[Pretrain] {mode_str} on 20M-tick sequences | {args.pretrain_epochs} epochs")

    encoder = model.backbone if hasattr(model, "backbone") else model

    # --force-pretrain: wipe previous encoder checkpoint so we start fresh
    if getattr(args, "force_pretrain", False):
        for suffix in ("", "_regime"):
            old = Path(args.checkpoint_dir) / f"contrastive_encoder{suffix}.pt"
            if old.exists():
                old.unlink()
                print(f"[Pretrain] Deleted old checkpoint: {old}")

    n_windows = 100_000
    if HDF5:
        with h5py.File(cache_path, "r") as f:
            n_total = min(int(f["X"].shape[0]), int(f["y"].shape[0]))
            idx     = np.random.choice(n_total, min(n_windows, n_total), replace=False)
            idx.sort()
            windows  = f["X"][idx[:n_windows]]
            y_sample = np.asarray(f["y"][idx[:n_windows]])
    else:
        X_mmap  = np.load(_x_path(cache_path), mmap_mode="r")
        y_mmap  = np.load(_y_path(cache_path), mmap_mode="r")
        idx     = np.random.choice(len(X_mmap), min(n_windows, len(X_mmap)), replace=False)
        windows = X_mmap[np.sort(idx)]
        y_sample = np.asarray(y_mmap[np.sort(idx)])

    print(f"[Pretrain] Sampled {len(windows):,} windows | shape {windows.shape[1:]}")

    ckpt   = str(Path(args.checkpoint_dir) / f"contrastive_encoder{'_regime' if use_regime else ''}.pt")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    pt_bs  = min(1024, max(256, args.batch_size))

    # Infer encoder output dim via dummy forward pass — works for any architecture
    # regardless of how hidden_size / d_model combine inside the backbone.
    with torch.no_grad():
        _dummy = torch.zeros(1, args.seq_len, n_features, device=device)
        _out   = encoder(_dummy)
        if _out.ndim == 2:
            encoder_dim = int(_out.shape[-1])
        elif _out.ndim >= 3:
            encoder_dim = int(_out[:, -1, :].shape[-1])
        else:
            encoder_dim = int(_out.shape[0])

    common = dict(
        d_model     = encoder_dim,
        proj_dim    = PRETRAIN["projection_dim"],
        temperature = PRETRAIN["temperature"],
        lr          = PRETRAIN["pretrain_lr"],
        device      = str(device),
    )

    if use_regime:
        # Derive regime labels from the continuous label values:
        #   y > 0.1  → trending (+1), y < -0.1 → mean-reverting (-1), else neutral (0)
        regime_labels = np.where(y_sample > 0.1, 1,
                        np.where(y_sample < -0.1, -1, 0)).astype(np.int8)
        trainer = RegimeAwareTSCLTrainer(
            encoder=encoder,
            regime_labels=regime_labels,
            hard_negative_weight=1.0,
            **common,
        )
    else:
        trainer = TSCLTrainer(encoder=encoder, **common)

    history = trainer.pretrain(windows, epochs=args.pretrain_epochs,
                               batch_size=pt_bs, checkpoint_path=ckpt)
    print(f"[Pretrain] Final loss: {history['loss'][-1]:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# RL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_rl(cache_path, n_features, args, device):
    print(f"\n[RL] {args.rl_algo.upper()} | {args.rl_episodes} episodes")

    # Stream a 100k-sample slice for the RL environment
    n_env = 100_000
    if HDF5:
        with h5py.File(cache_path, "r") as f:
            X_env = f["X"][-n_env:, -1, :]    # Last bar of each window as obs
            y_env = f["y"][-n_env:]
    else:
        X_env = np.load(_x_path(cache_path), mmap_mode="r")[-n_env:, -1, :]
        y_env = np.load(_y_path(cache_path), mmap_mode="r")[-n_env:]

    prices  = np.ones(n_env, dtype=np.float32) * 1.085   # Proxy close prices
    atr     = np.full(n_env, 0.0005, dtype=np.float32)
    spreads = np.full(n_env, 0.00005, dtype=np.float32)
    feat    = np.array(X_env, dtype=np.float32)

    env = ForexTradingEnv(
        features       = feat, prices=prices, atr=atr, spreads=spreads,
        atr_sl_mult    = RISK["atr_multiplier"],
        trail_activation_r = RISK["trail_activation_r"],
        breakeven_at_r = RISK["breakeven_at_r"],
        pyramid_pct    = SIZING["pyramid_add_pct"],
        martingale_pct = SIZING["martingale_add_pct"],
        max_lots       = SIZING["max_total_lots"],
    )
    dev = str(device)
    if args.rl_algo == "dqn":
        agent = DQNAgent(obs_size=env.obs_size, device=dev, **RL["dqn"])
    else:
        agent = PPOAgent(obs_size=env.obs_size, device=dev, **RL["ppo"])

    returns = train_agent(agent, env, n_episodes=args.rl_episodes,
                          agent_type=args.rl_algo)

    ckpt_dir = Path(args.checkpoint_dir)
    if hasattr(agent, "policy_net"):
        torch.save(agent.policy_net.state_dict(),
                   ckpt_dir / f"rl_{args.rl_algo}_best.pt")
    elif hasattr(agent, "net"):
        torch.save(agent.net.state_dict(),
                   ckpt_dir / f"rl_{args.rl_algo}_best.pt")

    s = env.summary()
    print(f"[RL] Done | Return: {s['total_return_pct']:+.2f}% | "
          f"Sharpe: {s['sharpe']:.3f} | Trades: {s['n_trades']}")
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE META-LEARNER TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble_meta(
    cache_path:  str,
    n_features:  int,
    args,
    device:      "torch.device",
) -> None:
    """
    Load all trained base model checkpoints, build an EnsembleMetaLearner,
    then train the meta-network with a diversity penalty.

    The diversity penalty has two components:
      1. Weight entropy maximisation — prevents the meta from collapsing to
         a single model (all weight on the best base model).
      2. Base-output correlation penalty — rewards the meta for up-weighting
         models whose predictions disagree with each other.

    Only runs when --train-ensemble is passed and at least 2 base checkpoints
    are found in the checkpoint directory.
    """
    if not ENSEMBLE:
        print("[EnsembleMeta] models.ensemble not available — skipping.")
        return

    ckpt_dir = Path(args.checkpoint_dir)
    loaded_bases: list = []
    loaded_names: list = []

    for model_name in MODEL_REGISTRY:
        ckpt = ckpt_dir / f"{model_name}_best.pt"
        if not ckpt.exists():
            continue
        try:
            base = build_model(model_name, n_features, args).to(device)
            core = base.backbone if isinstance(base, MultiTaskWrapper) else base
            core.load_state_dict(
                torch.load(ckpt, map_location=device), strict=False
            )
            base.eval()
            loaded_bases.append(base)
            loaded_names.append(model_name)
            print(f"  [EnsembleMeta] Loaded {model_name} from {ckpt.name}")
        except Exception as e:
            print(f"  [EnsembleMeta] Could not load {model_name}: {e}")

    if len(loaded_bases) < 2:
        print("[EnsembleMeta] Need ≥ 2 trained base models — skipping "
              f"(found {len(loaded_bases)}: {loaded_names}). "
              "Train with --all-models first.")
        return

    print(f"\n[EnsembleMeta] Training meta-learner on {len(loaded_bases)} bases: "
          f"{loaded_names}")

    meta = EnsembleMetaLearner(loaded_bases, context_dim=32, hidden=64).to(device)

    # Use a random 10 % subset of the dataset for meta training
    if cache_path.endswith(".h5") and HDF5:
        with h5py.File(cache_path, "r") as _f:
            _total = _f["X"].shape[0]
    else:
        _total = np.load(_x_path(cache_path), mmap_mode="r").shape[0]
    n_meta   = min(200_000, int(0.1 * _total))
    meta_idx = np.random.choice(_total, n_meta, replace=False)
    meta_ds = MemmapSequenceDataset(cache_path, meta_idx)
    meta_dl = DataLoader(
        meta_ds, batch_size=min(args.batch_size, 512),
        shuffle=True, num_workers=min(4, args.num_workers),
        pin_memory=True,
    )

    history = train_meta_learner(
        meta,
        meta_dl,
        epochs=getattr(args, "ensemble_epochs", 10),
        lr=1e-3,
        diversity_weight=getattr(args, "ensemble_div_weight", 0.1),
        device=str(device),
        verbose=True,
    )

    out = ckpt_dir / "ensemble_meta_best.pt"
    torch.save(meta.state_dict(), out)
    print(f"[EnsembleMeta] Saved → {out}  |  Final loss: {history[-1]:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    apply_hardware_profile(args)
    run_name = args.run_name or f"{args.model}_{datetime.now():%m%d_%H%M}"

    _all_pairs   = _get_pairs(args)
    _pairs_str   = ", ".join(_all_pairs)
    _embed_str   = (f"  embed={getattr(args,'pair_embed_dim',0)}d"
                    if len(_all_pairs) > 1 else "")

    print(f"\n{'━'*62}")
    print(f"  Forex Scaling Model — 20M Tick GPU Trainer")
    print(f"  Run: {run_name}  |  Ticks: {args.n_ticks:,}  |  Model: {args.model.upper()}")
    print(f"  Pairs: {_pairs_str}{_embed_str}")
    print(f"  Batch: {args.batch_size}  |  Epochs: {args.epochs}  |  AMP: {args.amp}")
    print(f"  Labels: {args.label_method}  |  Loss: {args.loss}  |  "
          f"Early-stop: {args.early_stop_metric}")
    if getattr(args, "multitask", False):
        print(f"  MultiTask: ON  (w_ret={args.mt_w_ret}  w_conf={args.mt_w_conf})")
    if getattr(args, "pretrain_regime", False):
        print(f"  Pretrain: regime-aware TSCL (hard negatives from opposite regime)")
    if getattr(args, "train_ensemble", False):
        print(f"  Ensemble meta-learner: ON  (epochs={args.ensemble_epochs}  "
              f"div_weight={args.ensemble_div_weight})")
    print(f"{'━'*62}")

    device, n_gpus = setup_device()

    # ── Phase 1: Build / load chunked dataset ─────────────────────────────────
    cache_path, n_samples, n_features, scaler = build_dataset_chunked(args)
    n_samples = _clamp_n_samples_to_disk(cache_path, n_samples)
    print(f"\n[Dataset] {n_samples:,} sequences × {n_features} features × "
          f"seq_len {args.seq_len}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run: Any = None
    if WANDB and not args.no_wandb and os.getenv("WANDB_API_KEY"):
        wandb_run = wandb.init(
            project = args.wandb_project,
            name    = run_name,
            config  = {**vars(args), "n_samples": n_samples,
                       "n_features": n_features},
            tags    = [args.model, args.data_source, "20M"],
        )

    models_to_train = list(MODEL_REGISTRY.keys()) if args.all_models else [args.model]

    for model_name in models_to_train:
        args.model = model_name

        model = build_model(model_name, n_features, args).to(device)

        # Optional HPO
        if args.hparam_search and OPTUNA:
            print(f"\n[HPO] Optuna {args.n_trials} trials...")
            def objective(trial):
                ta = argparse.Namespace(**vars(args))
                ta.lr          = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
                ta.hidden_size = trial.suggest_categorical("hidden_size", [128,256,512])
                ta.d_model     = trial.suggest_categorical("d_model", [128,256,512])
                ta.dropout     = trial.suggest_float("dropout", 0.05, 0.3)
                ta.batch_size  = trial.suggest_categorical("batch_size", [1024,2048,4096])
                ta.epochs      = 5; ta.patience = 3; ta.resume = False
                ta.all_models  = False
                m2 = build_model(model_name, n_features, ta).to(device)
                h, bv = supervised_train(model_name, cache_path, n_samples,
                                          n_features, ta, device, n_gpus, run=None)
                return bv
            direction = "maximize" if args.early_stop_metric == "sharpe" else "minimize"
            study = optuna.create_study(direction=direction,
                                        pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
            for k,v in study.best_params.items():
                setattr(args, k.replace("-","_"), v)
            print(f"[HPO] Best: {study.best_params}  val={study.best_value:.6f}")
            model = build_model(model_name, n_features, args).to(device)

        # Optional contrastive pre-training (TSCL expects dense embeddings, not class logits)
        if args.pretrain:
            if args.loss == "cross_entropy":
                pt_ns = argparse.Namespace(**vars(args))
                pt_ns.loss = "huber"
                model = build_model(model_name, n_features, pt_ns).to(device)
            model = run_pretrain(model, cache_path, n_features, args, device)

        # Supervised training (single split or walk-forward CV)
        log_dir = Path(args.checkpoint_dir).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if args.walk_forward_cv:
            splits = walk_forward_splits(
                n_samples, args.walk_forward_folds, args.seq_len,
            )
            print(f"[WalkForward] {len(splits)} folds | embargo bars = {args.seq_len}")
            cv_hist: list[dict] = []
            for fi, (tr_i, va_i) in enumerate(splits):
                history, best_val = supervised_train(
                    model_name, cache_path, n_samples, n_features,
                    args, device, n_gpus, run=wandb_run,
                    train_idx=tr_i, val_idx=va_i, fold_id=fi,
                )
                cv_hist.append({"fold": fi, "best_metric": best_val, "history": history})
            with open(log_dir / f"{run_name}_{model_name}_cv.json", "w", encoding="utf-8") as fp:
                json.dump(cv_hist, fp)
        else:
            history, best_val = supervised_train(
                model_name, cache_path, n_samples, n_features,
                args, device, n_gpus, run=wandb_run,
            )
            with open(log_dir / f"{run_name}_{model_name}.json", "w", encoding="utf-8") as fp:
                json.dump(history, fp)

    # Ensemble meta-learner training (with diversity penalty)
    if getattr(args, "train_ensemble", False):
        run_ensemble_meta(cache_path, n_features, args, device)

    # RL training
    if args.rl_train:
        run_rl(cache_path, n_features, args, device)

    if wandb_run: wandb_run.finish()

    print(f"\n{'━'*62}")
    print(f"  Training complete!")
    print(f"  Checkpoints: {args.checkpoint_dir}/")
    print(f"  Dataset cache: {cache_path}  (reused on --resume)")
    print(f"{'━'*62}")


if __name__ == "__main__":
    main()
