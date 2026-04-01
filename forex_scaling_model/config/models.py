"""
config/models.py  —  Supervised architecture hyperparameters
===========================================================
Defaults aligned with `training/train_gpu.py` builders and `models/architectures.py`.
Edit here to change per-architecture specs; `settings.MODELS` re-exports this dict.
"""

from typing import Any, Dict, FrozenSet

# Keys accepted by train_gpu --model and build_model()
SUPPORTED_SUPERVISED: FrozenSet[str] = frozenset(
    {"tft", "transformer", "haelt", "mamba", "gnn", "expert"}
)

MODELS: Dict[str, Dict[str, Any]] = {
    "tft": {
        "hidden_size": 128,
        "attention_head_size": 4,
        "dropout": 0.1,
        "lstm_layers": 2,
        "seq_len": 60,
        "learning_rate": 1e-3,
    },
    "transformer": {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "seq_len": 60,
        "learning_rate": 1e-4,
    },
    "haelt": {
        "lstm_hidden": 64,
        "d_model": 64,
        "nhead": 4,
        "n_transformer_layers": 2,
        "dropout": 0.1,
        "seq_len": 60,
        "learning_rate": 1e-4,
    },
    "mamba": {
        "d_model": 128,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 4,
        "dropout": 0.1,
        "seq_len": 60,
        "learning_rate": 1e-4,
    },
    "gnn": {
        "node_features": 32,
        "hidden_channels": 64,
        "num_layers": 3,
        "heads": 4,
        "dropout": 0.1,
        "correlation_threshold": 0.3,
    },
    "expert": {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "use_conv_ffn": True,
        "no_pos_encoding": True,
        "dropout": 0.1,
        "seq_len": 60,
        "learning_rate": 1e-4,
    },
}


def architecture_config(name: str) -> Dict[str, Any]:
    """Return hyperparameter dict for a supervised architecture key."""
    key = name.lower().strip()
    if key not in MODELS:
        raise KeyError(
            f"Unknown architecture {name!r}; expected one of {sorted(SUPPORTED_SUPERVISED)}"
        )
    return dict(MODELS[key])
