"""Integration-style checks: configuration, registry alignment, data → features → env."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from config.models import MODELS, SUPPORTED_SUPERVISED, architecture_config
from config.settings import MODELS as SETTINGS_MODELS, PATHS, PROJECT_ROOT, project_path
from data.data_ingestion import ForexDataPipeline, generate_synthetic_tick_data
from features.feature_engineering import FeatureEngineer
from models.architectures import MODEL_REGISTRY
from models.rl_agents import ForexTradingEnv, HOLD


class TestConfiguration:
    def test_project_root_is_repo(self):
        assert PROJECT_ROOT.is_dir()
        assert (PROJECT_ROOT / "config" / "settings.py").is_file()

    def test_project_path_joins_under_root(self):
        p = project_path("config", "settings.py")
        assert Path(p).resolve() == (PROJECT_ROOT / "config" / "settings.py").resolve()

    def test_paths_use_project_root_strings(self):
        for key, val in PATHS.items():
            assert isinstance(val, str)
            assert Path(val).is_absolute() or key.startswith("file_")

    def test_settings_models_matches_config_models(self):
        assert SETTINGS_MODELS is MODELS

    def test_supervised_keys_match_model_registry(self):
        assert SUPPORTED_SUPERVISED == frozenset(MODEL_REGISTRY.keys())

    def test_architecture_config_roundtrip(self):
        cfg = architecture_config("expert")
        assert "d_model" in cfg
        with pytest.raises(KeyError):
            architecture_config("unknown_arch")


class TestDataPipelineSmoke:
    def test_synthetic_ticks_to_env_observation(self):
        ticks = generate_synthetic_tick_data(n_rows=25_000, seed=1)
        bars = ForexDataPipeline(
            bar_freq="5min", session_filter=False, apply_frac_diff=False
        ).run(ticks)
        fe = FeatureEngineer()
        feats = fe.build(bars)
        bars_a = bars.reindex(feats.index).dropna()
        assert len(bars_a) > 10

        prices = bars_a["close"].values.astype(np.float32)
        atr_col = "atr_6" if "atr_6" in feats.columns else feats.columns[0]
        atr = feats[atr_col].values.astype(np.float32)
        spreads = np.full(len(prices), 0.00005, dtype=np.float32)
        feat_arr = feats.values.astype(np.float32)
        assert feat_arr.shape[0] == len(prices)

        env = ForexTradingEnv(feat_arr, prices, atr, spreads)
        obs = env.reset()
        assert obs.shape[0] == env.obs_size
        obs2, _, done, _ = env.step(HOLD)
        assert not done or obs2.shape == obs.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_creation():
    x = torch.zeros(1, device="cuda")
    assert x.device.type == "cuda"
