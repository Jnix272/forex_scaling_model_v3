"""
config/settings.py  —  Full System Specification
=================================================
Single source of truth for every component in the pipeline.
Covers all 25 architectural choices specified.
"""

from pathlib import Path
from typing import Dict

from config.models import MODELS

# Repository root (parent of config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> str:
    """Absolute path under the repo root (portable; works from any CWD)."""
    return str(PROJECT_ROOT.joinpath(*parts))


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT PATHS  —  separate top-level folders under the repo
# ─────────────────────────────────────────────────────────────────────────────
# checkpoints/   model weights, ONNX, lockbox markers
# exports/       ONNX and other export artifacts (monitoring ONNXExporter)
# logs/          training history, shadow mode, SHAP, reports, live engine
# data/          processed caches, embeddings, raw vendor feeds (incl. data/raw/*)
PATHS: Dict[str, str] = {
    "checkpoints": project_path("checkpoints"),
    "exports": project_path("exports"),
    "logs": project_path("logs"),
    "data": project_path("data"),
    "data_processed": project_path("data", "processed"),
    "data_embeddings": project_path("data", "embeddings"),
    "data_raw_cot": project_path("data", "raw", "cot"),
    "logs_shadow": project_path("logs", "shadow"),
    "logs_shap": project_path("logs", "shap"),
    "logs_reports": project_path("logs", "reports"),
    "logs_live": project_path("logs", "live"),
    "file_contrastive_encoder": project_path("checkpoints", "contrastive_encoder.pt"),
    "file_lockbox_used": project_path("checkpoints", "lockbox_used.json"),
    "file_lockbox_log": project_path("checkpoints", "lockbox_log.json"),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
DATA = {
    "resolution":      "tick",
    "storage_engine":  "timescaledb",
    "price_type":      "bid_ask",
    "pairs":           ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
    "primary_pair":    "EURUSD",
    "bar_freq":        "1min",
    "timescale": {
        "host":        "localhost",
        "port":        5432,
        "dbname":      "forex_ticks",
        "user":        "forex_user",
        "tick_table":  "tick_data",
        "bar_table":   "ohlcv_bars",
        "chunk_interval": "1 day",
    },
    "kafka": {
        "bootstrap_servers": "localhost:9092",
        "tick_topic":        "forex.ticks",
        "news_topic":        "forex.news",
        "cross_asset_topic": "forex.cross_asset",
        "consumer_group":    "forex_scaler",
        "batch_size":        1000,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = {
    "atr_window":           6,             # 6-period ATR (spec)
    "ofi_window":           20,
    "trade_arrival_window": 30,            # Trade Arrival Rate window
    "rsi_period":           14,
    "macd_fast":            12,
    "macd_slow":            26,
    "macd_signal":          9,
    "bollinger_window":     20,
    "bollinger_std":        2.0,
    "lag_windows":          [5, 20, 60],
    "sentiment_decay_lambda": 0.1,
    "buzz_window_minutes":  5,
}

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ASSET INPUTS
# ─────────────────────────────────────────────────────────────────────────────
CROSS_ASSET = {
    "enabled": True,
    "assets": {
        "US10Y":    {"ticker": "^TNX",  "related_pair": "USDJPY",  "lag_bars": [1, 5]},
        "WTI":      {"ticker": "CL=F",  "related_pair": "USDCAD",  "lag_bars": [1, 5]},
        "COPPER":   {"ticker": "HG=F",  "related_pair": "AUDUSD",  "lag_bars": [1, 5]},
        "IRON_ORE": {"ticker": "TIO=F", "related_pair": "AUDUSD",  "lag_bars": [1, 5]},
        "VIX":      {"ticker": "^VIX",  "related_pair": "all",     "lag_bars": [1, 5]},
    },
    "correlation_window":  60,
    "gnn_update_freq":     5,
}

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT / NEWS  — DUAL STREAM
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT = {
    "enabled":              True,
    "offline_model":        "ProsusAI/finbert",
    "embedding_dim":        768,
    "embedding_cache":      PATHS["data_embeddings"],
    "online_model":         "mistralai/Mistral-7B-Instruct-v0.2",
    "slm_max_tokens":       64,
    "global_brain_update_sec": 60,
    "decay_lambda":         0.1,
    "buzz_window_min":      5,
    "eco_calendar_enabled": True,
    "high_impact_events":   ["NFP", "CPI", "FOMC", "GDP", "PMI", "ECB", "BOE"],
    "news_api":             "alpha_vantage",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES  (defined in config/models.py, re-exported here)
# ─────────────────────────────────────────────────────────────────────────────

TRAINING = {
    "batch_size":       512,
    "epochs":           100,
    "patience":         10,
    "loss":             "cross_entropy",
    "huber_delta":      1.0,
    "asymmetric_sign_weight": 2.0,
    "grad_clip":        1.0,
    "weight_decay":     1e-4,
    "amp":              True,
    "val_split":        0.2,
    "seq_len":          60,
    "checkpoint_dir":   PATHS["checkpoints"],
    "walk_forward_folds": 6,
    "early_stop_metric": "sharpe",
    "sharpe_annualization_factor": 1.0,
    "onecycle_pct_start": 0.1,
    "onecycle_max_lr_mult": 10.0,
}

# Presets for local machines (use: python training/train_gpu.py --hardware-profile <name>)
# RTX 4060 = 8GB VRAM; pair with 16GB system RAM → keep workers/prefetch low to avoid host OOM.
HARDWARE_PROFILES = {
    "rtx_4060_16gb_ram": {
        "batch_size":         512,
        "num_workers":        5,
        "chunk_size":         250_000,
        "prefetch_factor":    2,
        "local_project_paths": True,
    },
    "rtx_4000_ada_cloud": {
        # RTX 4000 Ada Generation — 20GB VRAM, cloud pod (RunPod / Vast.ai)
        "batch_size":         8192,
        "num_workers":        2,
        "chunk_size":         500_000,
        "prefetch_factor":    2,
        "local_project_paths": False,
    },
    "a5000_24gb": {
        # RTX A5000 24GB — good cloud/workstation balance for long runs
        "batch_size":         1536,
        "num_workers":        8,
        "chunk_size":         500_000,
        "prefetch_factor":    4,
        "local_project_paths": False,
    },
    "a40_48gb": {
        # NVIDIA A40 48GB — high VRAM cloud profile tuned for stable throughput
        "batch_size":         384,
        "num_workers":        6,
        "chunk_size":         500_000,
        "prefetch_factor":    4,
        "local_project_paths": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTRASTIVE PRE-TRAINING
# ─────────────────────────────────────────────────────────────────────────────
PRETRAIN = {
    "enabled":          True,
    "method":           "tscl",            # Time-Series Contrastive Learning
    "temperature":      0.07,
    "projection_dim":   128,
    "augmentations": {
        "jitter_std":       0.01,
        "scaling_range":    (0.8, 1.2),
        "permutation_segs": 4,
        "dropout_prob":     0.1,
    },
    "pretrain_epochs":  50,
    "pretrain_lr":      1e-4,
    "pretrain_batch":   256,
    "checkpoint":       PATHS["file_contrastive_encoder"],
}

# ─────────────────────────────────────────────────────────────────────────────
# RL AGENT  — PPO + DQN, 3-ACTION
# ─────────────────────────────────────────────────────────────────────────────
RL = {
    "algorithms":   ["PPO", "DQN"],
    "action_space": 3,                     # 0=Buy, 1=Hold, 2=Sell
    "ppo": {
        "gamma":        0.99,
        "lr":           3e-4,
        "clip_epsilon": 0.2,
        "entropy_coeff": 0.01,
        "value_coeff":  0.5,
        "n_steps":      2048,
        "n_epochs":     10,
        "gae_lambda":   0.95,
    },
    "dqn": {
        "gamma":         0.99,
        "lr":            1e-4,
        "eps_start":     1.0,
        "eps_end":       0.01,
        "eps_decay":     0.995,
        "buf_size":      1_000_000,
        "batch":         64,
        "target_update": 100,
        "double_dqn":    True,
    },
    "reward": {
        "pnl_weight":               1.0,
        "drawdown_penalty":         0.5,
        "transaction_cost_penalty": 0.3,
        "overtrading_penalty":      0.2,
        "holding_cost":             0.01,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# LABELING — RL REWARD SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
LABELING = {
    "method":               "rl_reward",
    "lookahead_bars":       10,
    "profit_target_atr":    1.5,
    "stop_loss_atr":        1.0,
    "transaction_cost_pips": 1.5,
    "pip_size":             0.0001,
    # Triple-barrier: Numba-accelerated scans (parallel over bars; auto fallback if Numba missing)
    "tbm_numba":            True,
    "tbm_parallel":       True,
}

# ─────────────────────────────────────────────────────────────────────────────
# SIZING + SCALING STRATEGY (BOTH COMBINED)
# ─────────────────────────────────────────────────────────────────────────────
SIZING = {
    "method":           "fractional_kelly",
    "kelly_fraction":   0.25,
    "max_position_pct": 0.05,
    "target_annual_vol": 0.10,
    "pip_risk":         20.0,
    "scaling_strategy": "both",            # pyramid winners + martingale losers
    "pyramid_add_pct":  0.25,
    "martingale_add_pct": 0.25,
    "max_total_lots":   3.0,
    "scale_out_targets": [0.25, 0.50, 0.75],
}

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT — DYNAMIC STOP LOSS
# ─────────────────────────────────────────────────────────────────────────────
RISK = {
    "stop_type":            "dynamic",
    "atr_multiplier":       1.5,
    "trail_activation_r":   1.0,
    "breakeven_at_r":       0.5,
    "max_drawdown_halt":    0.10,
    "daily_loss_limit":     0.03,
}

# ─────────────────────────────────────────────────────────────────────────────
# TRADE FILTERS — BOTH ENABLED
# ─────────────────────────────────────────────────────────────────────────────
FILTERS = {
    "vol_filter_enabled":   True,
    "vol_multiplier":       3.0,
    "vol_lookback":         60,
    "news_filter_enabled":  True,
    "news_buffer_minutes":  15,
    "sentiment_threshold":  0.6,
    "dust_settle_bars":     3,
}

# ─────────────────────────────────────────────────────────────────────────────
# LATENCY — TIP-SEARCH
# ─────────────────────────────────────────────────────────────────────────────
LATENCY = {
    "strategy":             "tip_search",
    "fast_model":           "dqn",         # ~2ms
    "slow_model":           "haelt",       # ~5ms
    "fast_latency_ms":      2.0,
    "slow_latency_ms":      5.0,
    "switch_threshold_mult": 2.0,          # Use fast if ATR > 2× avg
    "max_acceptable_ms":    10.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — EMBARGOING + PURGED K-FOLD
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION = {
    "method":               "purged_embargo",
    "n_splits":             5,
    "purge_bars":           30,
    "embargo_bars":         10,
    "train_window_bars":    50_000,
    "test_window_bars":     10_000,
}

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING + DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
MONITORING = {
    "enabled":              True,
    "drift_window":         1000,
    "psi_threshold":        0.2,
    "ks_pvalue_threshold":  0.05,
    "sharpe_drop_threshold": 0.5,
    "check_freq_bars":      500,
    "retrain_trigger":      "auto",
    "wandb_project":        "forex-scaling-model",
}

# ─────────────────────────────────────────────────────────────────────────────
# RETRAINING — WALK-FORWARD ROLLING
# ─────────────────────────────────────────────────────────────────────────────
RETRAINING = {
    "strategy":             "walk_forward_rolling",
    "retrain_every_bars":   10_000,
    "rolling_window_bars":  50_000,
    "warm_start":           True,
    "auto_deploy":          True,
    "min_improvement":      0.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE — KAFKA + TIMESCALEDB
# ─────────────────────────────────────────────────────────────────────────────
INFRA = {
    "kafka_enabled":        True,
    "timescale_enabled":    True,
    "log_dir":              PATHS["logs"],
    "checkpoint_dir":       PATHS["checkpoints"],
    "data_dir":             PATHS["data"],
}

# GOVERNANCE
GOVERNANCE = {'min_sharpe_promote': 1.5, 'min_sharpe_emergency': 1.3, 'min_profit_factor': 1.5, 'max_drawdown_pct': 0.20, 'min_trades': 500, 'max_regime_concentration': 0.75, 'max_cost_pct_gross_pnl': 0.30, 'strict_psr': False, 'demotion_sharpe_floor': 0.5, 'demotion_winrate_floor': 0.45, 'demotion_window_trades': 300, 'page_hinkley_delta': 0.005, 'page_hinkley_lambda': 50.0, 'mlflow_tracking_uri': 'http://localhost:5000', 'mlflow_experiment': 'forex-scaling-model'}

# ALERTS
ALERTS = {'discord_webhook_url': '', 'discord_min_interval_s': 300, 'discord_environment': 'production', 'prometheus_port': 8000, 'prometheus_enabled': True, 'alert_on_circuit_breaker': True, 'alert_on_drift': True, 'alert_on_promotion': True, 'alert_on_demotion': True, 'alert_on_retrain': True, 'alert_on_tca_breach': True}

# MACRO_DATA
MACRO_DATA = {'fred_api_key': '', 'yield_momentum_windows': [5, 20], 'yield_vol_window': 20, 'av_api_key': '', 'eco_news_buffer_minutes': 15, 'ollama_url': 'http://localhost:11434', 'ollama_model': 'mistral', 'bad_tick_z_thresh': 8.0, 'bad_tick_window': 60}
