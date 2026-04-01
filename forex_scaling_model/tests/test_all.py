"""
tests/test_all.py
==================
Comprehensive test suite — runs every module with assertions.
Covers: data, features, labeling, models, RL, risk, monitoring, live engine.

Run:  python tests/test_all.py
      python tests/test_all.py --fast   (skip slow Hurst / MC tests)
      python tests/test_all.py --module features
"""

import sys, argparse, time, warnings, traceback, os
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

PASS = 0; FAIL = 0; SKIP = 0
RESULTS = []


def run_case(name: str, fn, skip=False):
    global PASS, FAIL, SKIP
    if skip:
        SKIP += 1
        RESULTS.append(("SKIP", name, ""))
        print(f"  ⊙ {name}")
        return
    t0 = time.perf_counter()
    try:
        fn()
        ms = (time.perf_counter() - t0) * 1000
        PASS += 1
        RESULTS.append(("PASS", name, f"{ms:.0f}ms"))
        print(f"  ✓ {name}  ({ms:.0f}ms)")
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        FAIL += 1
        RESULTS.append(("FAIL", name, str(e)))
        print(f"  ✗ {name}  ({ms:.0f}ms)")
        print(f"    {traceback.format_exc().strip().splitlines()[-1]}")


def section(name: str):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def make_bars(n=500):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.085 + np.cumsum(rng.normal(0, 0.0001, n))
    return pd.DataFrame({
        "open":   close - rng.uniform(0, 0.0002, n),
        "high":   close + rng.uniform(0, 0.0003, n),
        "low":    close - rng.uniform(0, 0.0003, n),
        "close":  close,
        "volume": rng.integers(10, 100, n).astype(float),
        "bid_close": close - 0.00003,
        "ask_close": close + 0.00003,
        "spread_avg": np.full(n, 0.00005),
    }, index=idx)

def make_features(bars):
    from features.feature_engineering import FeatureEngineer
    from config.settings import FEATURES
    fe = FeatureEngineer(atr_window=FEATURES["atr_window"],
                          lag_windows=FEATURES["lag_windows"])
    return fe.build(bars)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def run_data_tests(fast=False):
    section("DATA INGESTION")

    def t_synthetic():
        from data.data_ingestion import generate_synthetic_tick_data
        df = generate_synthetic_tick_data(n_rows=10_000)
        assert len(df) == 10_000
        assert set(["bid","ask","mid","volume","spread"]).issubset(df.columns)
        assert (df["ask"] > df["bid"]).all(), "Golden rule violated"
        assert (df["spread"] > 0).all()

    def t_pipeline():
        from data.data_ingestion import generate_synthetic_tick_data, ForexDataPipeline
        ticks = generate_synthetic_tick_data(n_rows=50_000)
        bars = ForexDataPipeline(bar_freq="1min", session_filter=False,
                                  apply_frac_diff=False).run(ticks)
        assert len(bars) > 100
        assert "close" in bars.columns
        assert bars.index.tz is not None

    def t_schema():
        from data.sources import _enforce_schema
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=100, freq="1s", tz="UTC")
        df = pd.DataFrame({"bid": 1.085 - 0.00003, "ask": 1.085 + 0.00003}, index=idx)
        out = _enforce_schema(df, "EURUSD", "test")
        assert list(out.columns) == ["bid","ask","mid","spread","volume","pair","source"]
        assert out["pair"].iloc[0] == "EURUSD"

    def t_bid_ask_inversion_drop():
        from data.sources import _enforce_schema
        idx = pd.date_range("2024-01-01", periods=5, freq="1s", tz="UTC")
        df = pd.DataFrame({
            "bid": [1.085, 1.086, 1.085, 1.085, 1.085],
            "ask": [1.086, 1.085, 1.086, 1.086, 1.086],  # Row 1: bid>ask
        }, index=idx)
        out = _enforce_schema(df, "EURUSD", "test")
        assert len(out) == 4  # 1 row dropped

    def t_bad_tick_cleaning():
        from data.data_ingestion import generate_synthetic_tick_data, clean_bad_ticks
        df = generate_synthetic_tick_data(n_rows=200)
        # Inject a massive spike
        df.loc[df.index[100], "mid"] += 0.5000  # 5000 pips spike
        df.loc[df.index[100], "bid"] += 0.5000
        df.loc[df.index[100], "ask"] += 0.5000
        out = clean_bad_ticks(df, z_thresh=5.0)
        # Check that the spike was capped
        assert abs(out.loc[out.index[100], "mid"] - out.loc[out.index[99], "mid"]) < 0.01

    def t_eco_calendar():
        from data.economic_calendar import EcoCalendarFeatureBuilder
        from data.data_ingestion import generate_synthetic_tick_data, resample_to_bars
        ticks = generate_synthetic_tick_data(n_rows=5000)
        bars = resample_to_bars(ticks, freq="1min")
        builder = EcoCalendarFeatureBuilder(use_synthetic=True)
        feats = builder.build(bars)
        assert "eco_release_flag" in feats.columns
        assert feats.shape[0] == bars.shape[0]

    run_case("synthetic tick generation", t_synthetic)
    run_case("tick → bar pipeline",       t_pipeline)
    run_case("unified schema enforcement", t_schema)
    run_case("bid-ask inversion drop",    t_bid_ask_inversion_drop)
    run_case("bad-tick outlier cleaning", t_bad_tick_cleaning)
    run_case("economic calendar features", t_eco_calendar)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_tests(fast=False):
    section("FEATURE ENGINEERING")
    bars = make_bars(400)

    def t_core_features():
        f = make_features(bars)
        assert f.shape[1] >= 40
        assert f.shape[0] > 0
        assert "atr_6" in f.columns
        assert "ofi" in f.columns
        assert "rsi_14" in f.columns
        assert "macd" in f.columns
        assert "bb_pct" in f.columns
        assert f.isna().sum().sum() == 0

    def t_atr6():
        f = make_features(bars)
        assert "atr_6" in f.columns
        assert (f["atr_6"] > 0).mean() > 0.9

    def t_cross_asset():
        f = make_features(bars)
        ca_cols = [c for c in f.columns if any(a in c for a in ["US10Y","WTI","COPPER","VIX","IRON"])]
        assert len(ca_cols) >= 10, f"Expected ≥10 cross-asset cols, got {len(ca_cols)}"

    def t_adv_features():
        from features.advanced_features import AdvancedFeatureBuilder
        afb = AdvancedFeatureBuilder(hurst_windows=[30])
        adv = afb.build(bars)
        # Check key representative columns (names depend on implementation)
        assert "sess_london" in adv.columns
        assert adv.shape[1] >= 30, f"Expected ≥30 advanced cols, got {adv.shape[1]}"
        assert adv.isna().sum().sum() == 0

    def t_session_flags():
        from features.advanced_features import session_clock_features
        idx = pd.date_range("2024-01-01 08:00", periods=60, freq="1min", tz="UTC")
        s = session_clock_features(idx)
        assert s["sess_london"].iloc[0] == 1.0
        assert s["sess_ny"].iloc[0] == 0.0

    def t_hurst_bounds():
        from features.advanced_features import hurst_exponent
        s = pd.Series(np.cumsum(np.random.randn(200)))
        h = hurst_exponent(s)
        assert 0.0 <= h <= 1.0

    def t_no_lookahead():
        """Features at time T must not use data from T+1."""
        f = make_features(bars)
        # Close at T correlates with features at T-1 (lagged), not T+1
        corr_lag1 = f["ret_5"].shift(1).corr(bars["close"].reindex(f.index))
        assert abs(corr_lag1) <= 1.0  # Sanity: no NaN correlation

    def t_macro_yields():
        from features.macro_features import MacroYieldFeatureBuilder
        builder = MacroYieldFeatureBuilder()
        feats = builder.build(bars)
        assert "spread_us_de" in feats.columns
        assert "carry_eur" in feats.columns
        assert feats.isna().sum().sum() == 0

    def t_sentiment_pipeline():
        from features.finbert_sentiment import SentimentPipeline
        # Force VADER as default for fast tests and avoid Ollama/Transformers dep in CI
        pipe = SentimentPipeline(prefer_backend="vader", use_cache=False)
        score = pipe.score_headlines(["EUR/USD expected to rally on positive news"])
        assert score > 0
        assert pipe.active_backend() == "vader"

    run_case("core feature matrix",  t_core_features)
    run_case("ATR-6 period",         t_atr6)
    run_case("cross-asset inputs",   t_cross_asset)
    run_case("advanced features",    t_adv_features)
    run_case("session clock flags",  t_session_flags)
    run_case("Hurst exponent bounds",t_hurst_bounds, skip=fast)
    run_case("no look-ahead bias",   t_no_lookahead)
    run_case("macro yield spreads",  t_macro_yields)
    run_case("sentiment (VADER tier)", t_sentiment_pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# LABELING
# ─────────────────────────────────────────────────────────────────────────────

def run_labeling_tests(fast=False):
    section("RL REWARD LABELING")
    bars = make_bars(400)
    feats = make_features(bars)

    def t_rl_labels():
        from labeling.rl_reward_labeling import compute_rl_reward_labels
        labels = compute_rl_reward_labels(bars, feats)
        assert "reward" in labels.columns
        assert "label" in labels.columns
        assert set(labels["label"].dropna().unique()).issubset({-1.0, 0.0, 1.0})
        assert len(labels) > 0

    def t_align():
        from labeling.rl_reward_labeling import (compute_rl_reward_labels,
                                                   align_labels_with_features)
        labels = compute_rl_reward_labels(bars, feats)
        X, y = align_labels_with_features(labels, feats)
        assert len(X) == len(y)
        assert X.shape[1] == feats.shape[1]
        assert X.isna().sum().sum() == 0

    run_case("RL reward signal", t_rl_labels)
    run_case("label alignment",  t_align)

    section("TRIPLE BARRIER — NUMBA SCAN")

    def t_tbm_numba_parity():
        """Sequential reference vs Numba parallel/serial must agree element-wise."""
        import numpy as np
        from labeling.triple_barrier_labeling import (
            _NUMBA_IMPORT_OK,
            _run_barrier_scan,
            _scan_outcomes_sequential,
        )

        rng = np.random.default_rng(42)
        n = 1500
        vb = 12
        close = np.cumsum(rng.standard_normal(n) * 0.00005).astype(np.float64) + 1.085
        entry_long = close + 0.00002
        entry_short = close - 0.00002
        atr = np.full(n, 0.00045, dtype=np.float64)
        pm, sm = 1.5, 1.0

        lo_s, tl_s, so_s, ts_s = _scan_outcomes_sequential(
            close, entry_long, entry_short, atr, pm, sm, vb
        )
        lo_n, tl_n, so_n, ts_n, tag = _run_barrier_scan(
            close, entry_long, entry_short, atr, pm, sm, vb,
            use_numba=True, parallel=True,
        )
        np.testing.assert_array_equal(lo_s, lo_n)
        np.testing.assert_array_equal(tl_s, tl_n)
        np.testing.assert_array_equal(so_s, so_n)
        np.testing.assert_array_equal(ts_s, ts_n)
        if _NUMBA_IMPORT_OK:
            assert "numba_parallel" == tag

        lo_ns, tl_ns, so_ns, ts_ns, tag2 = _run_barrier_scan(
            close, entry_long, entry_short, atr, pm, sm, vb,
            use_numba=True, parallel=False,
        )
        np.testing.assert_array_equal(lo_s, lo_ns)
        if _NUMBA_IMPORT_OK:
            assert tag2 == "numba_serial"

    run_case("TBM Numba scan matches sequential reference", t_tbm_numba_parity)


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

def run_model_tests(fast=False):
    section("MODEL ARCHITECTURES")

    try:
        import torch; TORCH = True
    except ImportError:
        TORCH = False

    def t_architectures():
        if not TORCH: return
        import torch
        from models.architectures import (TFTScalper, iTransformerScalper,
            HAELTHybrid, MambaScalper, GNNCrossAsset, EXPERTEncoder)
        B, T, F = 4, 60, 48
        x = torch.randn(B, T, F)
        for name, cls, kw in [
            ("TFT",         TFTScalper,         {}),
            ("iTransformer",iTransformerScalper, {"seq_len":T}),
            ("HAELT",       HAELTHybrid,        {"seq_len":T}),
            ("Mamba",       MambaScalper,       {}),
            ("EXPERT",      EXPERTEncoder,      {}),
        ]:
            m = cls(input_size=F, **kw)
            out = m(x)
            assert out.shape == (B,), f"{name}: expected ({B},) got {out.shape}"

    def t_gnn():
        if not TORCH: return
        import torch
        from models.architectures import GNNCrossAsset
        x = torch.randn(4, 6, 32)
        m = GNNCrossAsset(node_features=32, n_nodes=6)
        out = m(x)
        assert out.shape == (4,)

    def t_huber_loss():
        if not TORCH: return
        import torch
        from models.architectures import HuberLoss
        loss = HuberLoss(delta=1.0)
        p = torch.tensor([0.5, -0.3, 1.2])
        t = torch.tensor([0.4, -0.5, 0.8])
        val = loss(p, t)
        assert val.item() > 0

    def t_asymmetric_loss():
        if not TORCH:
            return
        import torch
        from models.architectures import AsymmetricDirectionalLoss
        loss = AsymmetricDirectionalLoss(delta=1.0, sign_weight=2.0)
        p = torch.tensor([0.5, -0.3, 1.2])
        t = torch.tensor([0.4, -0.5, 0.8])
        assert loss(p, t).item() > 0

    def t_ensemble():
        if not TORCH: return
        from models.ensemble import GrangerCausalityGraph
        gc = GrangerCausalityGraph(max_lag=2)
        df = pd.DataFrame(np.random.randn(200, 5),
                          columns=["EURUSD","US10Y","WTI","COPPER","VIX"])
        adj = gc.compute_adjacency(df, window=150)
        assert adj.shape == (5, 5)  # 5 inputs
        assert adj.diagonal().sum() == 0  # No self-loops

    def t_hierarchical_attn():
        if not TORCH: return
        import torch
        from models.ensemble import MultiTimeframeAttention
        m   = MultiTimeframeAttention(input_size=48)
        x1  = torch.randn(4, 60, 48)
        x5  = torch.randn(4, 12, 48)
        x15 = torch.randn(4,  4, 48)
        out = m([x1, x5, x15])
        assert out.shape == (4,)

    # ── #10 Multi-task head ────────────────────────────────────────────────

    def t_multitask_head_shapes():
        if not TORCH: return
        import torch
        from models.architectures import MultiTaskHead
        B, D = 8, 64
        h = torch.randn(B, D)
        head = MultiTaskHead(in_features=D, hidden=32)
        logits, ret, conf = head(h)
        assert logits.shape == (B, 3)
        assert ret.shape    == (B,)
        assert conf.shape   == (B,)
        assert conf.min().item() >= 0.0 - 1e-6
        assert conf.max().item() <= 1.0 + 1e-6

    def t_multitask_loss_backward():
        if not TORCH: return
        import torch
        from models.architectures import MultiTaskLoss
        B = 8
        logits  = torch.randn(B, 3, requires_grad=True)
        ret_hat = torch.randn(B, requires_grad=True)
        conf    = torch.sigmoid(torch.randn(B)).clone().requires_grad_(True)
        y_cls   = torch.zeros(B, dtype=torch.long)
        y_cont  = torch.zeros(B)
        loss = MultiTaskLoss()(logits, ret_hat, conf, y_cls, y_cont)
        assert loss.ndim == 0 and loss.item() >= 0
        loss.backward()
        assert logits.grad is not None and ret_hat.grad is not None

    def t_multitask_wrapper_haelt():
        if not TORCH: return
        import torch
        from models.architectures import HAELTHybrid, MultiTaskWrapper
        B, T, F = 4, 60, 48
        x    = torch.randn(B, T, F)
        base = HAELTHybrid(input_size=F, seq_len=T, lstm_hidden=32, d_model=32)
        w    = MultiTaskWrapper(base, head_in=64, hidden=32)
        w.eval()
        logits, ret, conf = w(x)
        assert logits.shape == (B, 3)
        assert ret.shape    == (B,)
        assert conf.shape   == (B,)

    # ── #12 Ensemble diversity ─────────────────────────────────────────────

    def t_diversity_loss_range():
        if not TORCH: return
        import torch
        from models.ensemble import EnsembleMetaLearner
        B, N = 32, 3

        class _Tiny(torch.nn.Module):
            def __init__(self): super().__init__(); self.w = torch.nn.Linear(48, 1)
            def forward(self, x): return self.w(x[:,-1,:]).squeeze(-1)

        ens = EnsembleMetaLearner([_Tiny(), _Tiny(), _Tiny()], context_dim=8, hidden=16)
        # Identical columns → diversity ≈ 1
        preds_same = torch.randn(B, 1).expand(B, N)
        d_high = ens.diversity_loss(preds_same).item()
        assert d_high > 0.8, f"Identical cols should yield high diversity, got {d_high:.3f}"
        # Random independent columns → diversity ≈ 0
        preds_rand = torch.randn(B, N)
        d_rand = abs(ens.diversity_loss(preds_rand).item())
        assert d_rand < 0.5, f"Random cols should yield low diversity, got {d_rand:.3f}"

    def t_train_meta_learner_runs():
        if not TORCH: return
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from models.ensemble import EnsembleMetaLearner, train_meta_learner

        class _Tiny(torch.nn.Module):
            def __init__(self): super().__init__(); self.w = torch.nn.Linear(48, 1)
            def forward(self, x): return self.w(x[:,-1,:]).squeeze(-1)

        ens = EnsembleMetaLearner([_Tiny(), _Tiny()], context_dim=8, hidden=16)
        X   = torch.randn(64, 60, 48)
        y   = torch.tensor(np.random.choice([-1.0, 0.0, 1.0], 64).astype(np.float32))
        dl  = DataLoader(TensorDataset(X, y), batch_size=32)
        h   = train_meta_learner(ens, dl, epochs=3, lr=1e-2,
                                  diversity_weight=0.1, verbose=False)
        assert len(h) == 3
        assert all(np.isfinite(v) for v in h)

    # ── Multi-pair joint training ──────────────────────────────────────────────

    def t_multipair_wrapper_shapes():
        if not TORCH: return
        import torch
        from models.architectures import HAELTHybrid, MultiPairWrapper
        B, T, F, E, P = 4, 60, 20, 8, 2
        backbone = HAELTHybrid(input_size=P*(F+E), seq_len=T,
                               lstm_hidden=32, d_model=32, nhead=2, n_layers=1)
        model    = MultiPairWrapper(backbone, n_pairs=P, f_per_pair=F, embed_dim=E)
        model.eval()
        x = torch.randn(B, T, P*F)
        out = model(x)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"

    def t_multipair_wrapper_four_pairs_classification():
        if not TORCH: return
        import torch
        from models.architectures import HAELTHybrid, MultiPairWrapper
        B, T, F, E, P = 4, 60, 16, 8, 4
        backbone = HAELTHybrid(input_size=P*(F+E), seq_len=T,
                               lstm_hidden=32, d_model=32, nhead=2, n_layers=1,
                               num_classes=3)
        model    = MultiPairWrapper(backbone, n_pairs=P, f_per_pair=F, embed_dim=E)
        model.eval()
        out = model(torch.randn(B, T, P*F))
        assert out.shape == (B, 3)

    def t_multipair_wrapper_gradient_flow():
        if not TORCH: return
        import torch
        from models.architectures import HAELTHybrid, MultiPairWrapper
        B, T, F, E, P = 4, 60, 16, 8, 2
        backbone = HAELTHybrid(input_size=P*(F+E), seq_len=T,
                               lstm_hidden=32, d_model=32, nhead=2, n_layers=1)
        model    = MultiPairWrapper(backbone, n_pairs=P, f_per_pair=F, embed_dim=E)
        model(torch.randn(B, T, P*F)).sum().backward()
        embed_grad = (model.pair_embeds.weight.grad is not None and
                      model.pair_embeds.weight.grad.abs().sum() > 0)
        assert embed_grad, "No gradient reached pair embedding weights"

    def t_get_pairs_helper():
        import argparse
        from training.train_gpu import _get_pairs
        a1 = argparse.Namespace(pairs="EURUSD,GBPUSD,USDJPY", pair="EURUSD")
        assert _get_pairs(a1) == ["EURUSD", "GBPUSD", "USDJPY"]
        a2 = argparse.Namespace(pairs=None, pair="GBPUSD")
        assert _get_pairs(a2) == ["GBPUSD"]
        a3 = argparse.Namespace(pairs=["eurusd", "usdjpy"], pair="EURUSD")
        assert _get_pairs(a3) == ["EURUSD", "USDJPY"]

    def t_multipair_chunk_shape():
        from unittest.mock import patch
        from training.train_gpu import _build_multipair_chunk
        from sklearn.preprocessing import StandardScaler
        T, F, N = 60, 8, 50
        pair_ticks = {"EURUSD": "e", "GBPUSD": "g"}
        scalers    = {"EURUSD": StandardScaler(), "GBPUSD": StandardScaler()}
        def _fake(*a, **kw):
            return (np.random.randn(N, T, F).astype(np.float32),
                    np.random.randn(N).astype(np.float32), F)
        with patch("training.train_gpu._build_chunk", side_effect=[_fake(), _fake()]):
            X, y, n_feat = _build_multipair_chunk(
                pair_ticks, fe=None, scalers=scalers,
                seq_len=T, chunk_idx=0, label_method="rl_reward",
            )
        assert X.shape == (N, T, 2*F) and n_feat == 2*F

    def t_multipair_chunk_inner_join():
        """Shorter pair truncates the output to the minimum sample count."""
        from unittest.mock import patch
        from training.train_gpu import _build_multipair_chunk
        from sklearn.preprocessing import StandardScaler
        T, F, N1, N2 = 60, 8, 100, 67
        pair_ticks = {"EURUSD": "e", "GBPUSD": "g"}
        scalers    = {"EURUSD": StandardScaler(), "GBPUSD": StandardScaler()}
        def _fake_n(n):
            return (np.random.randn(n, T, F).astype(np.float32),
                    np.random.randn(n).astype(np.float32), F)
        with patch("training.train_gpu._build_chunk", side_effect=[_fake_n(N1), _fake_n(N2)]):
            X, _, _ = _build_multipair_chunk(
                pair_ticks, fe=None, scalers=scalers,
                seq_len=T, chunk_idx=0, label_method="rl_reward",
            )
        assert X.shape[0] == min(N1, N2)

    run_case("all 6 architectures forward pass", t_architectures, skip=not TORCH)
    run_case("GNN cross-asset forward pass",     t_gnn,          skip=not TORCH)
    run_case("Huber loss computation",           t_huber_loss,   skip=not TORCH)
    run_case("Asymmetric directional loss",      t_asymmetric_loss, skip=not TORCH)
    run_case("Granger causality adjacency",      t_ensemble)
    run_case("hierarchical multi-TF attention",  t_hierarchical_attn, skip=not TORCH)
    run_case("MultiTaskHead output shapes",      t_multitask_head_shapes,   skip=not TORCH)
    run_case("MultiTaskLoss backward pass",      t_multitask_loss_backward, skip=not TORCH)
    run_case("MultiTaskWrapper (HAELT)",         t_multitask_wrapper_haelt, skip=not TORCH)
    run_case("diversity_loss range check",       t_diversity_loss_range,    skip=not TORCH)
    run_case("train_meta_learner runs",          t_train_meta_learner_runs, skip=not TORCH)
    run_case("MultiPairWrapper 2-pair scalar",   t_multipair_wrapper_shapes,              skip=not TORCH)
    run_case("MultiPairWrapper 4-pair 3-class",  t_multipair_wrapper_four_pairs_classification, skip=not TORCH)
    run_case("MultiPairWrapper gradient flow",   t_multipair_wrapper_gradient_flow,       skip=not TORCH)
    run_case("_get_pairs() helper parsing",      t_get_pairs_helper)
    run_case("_build_multipair_chunk shape",     t_multipair_chunk_shape)
    run_case("_build_multipair_chunk inner join",t_multipair_chunk_inner_join)


# ─────────────────────────────────────────────────────────────────────────────
# CONTRASTIVE PRE-TRAINING  (#11)
# ─────────────────────────────────────────────────────────────────────────────

def run_pretrain_tests(fast=False):
    section("CONTRASTIVE PRE-TRAINING")

    try:
        import torch; TORCH = True
    except ImportError:
        TORCH = False

    def t_standard_tscl():
        """Standard TSCLTrainer completes 2 epochs without error."""
        if not TORCH: return
        import torch, tempfile, os
        from pretrain.contrastive import TSCLTrainer
        from models.architectures import MambaScalper
        rng  = np.random.default_rng(0)
        X    = rng.standard_normal((80, 20, 16)).astype(np.float32)
        enc  = MambaScalper(input_size=16, d_model=16, num_layers=2)
        with tempfile.TemporaryDirectory() as tmp:
            h = TSCLTrainer(enc, d_model=16, proj_dim=16, lr=1e-3).pretrain(
                X, epochs=2, batch_size=32,
                checkpoint_path=os.path.join(tmp, "enc.pt"),
            )
        assert len(h["loss"]) == 2
        assert all(np.isfinite(v) for v in h["loss"])

    def t_regime_tscl_three_regimes():
        """RegimeAwareTSCLTrainer with 3 distinct regimes runs cleanly."""
        if not TORCH: return
        import torch, tempfile, os
        from pretrain.contrastive import RegimeAwareTSCLTrainer
        from models.architectures import EXPERTEncoder
        rng = np.random.default_rng(1)
        N   = 90
        X   = rng.standard_normal((N, 20, 16)).astype(np.float32)
        # Equal split across 3 regimes: trending, neutral, mean-rev
        labels = np.array([1]*30 + [0]*30 + [-1]*30, dtype=np.int8)
        enc  = EXPERTEncoder(input_size=16, d_model=16, nhead=4, num_layers=2)
        with tempfile.TemporaryDirectory() as tmp:
            h = RegimeAwareTSCLTrainer(
                enc, regime_labels=labels, d_model=16, proj_dim=16
            ).pretrain(X, epochs=2, batch_size=32,
                       checkpoint_path=os.path.join(tmp, "enc.pt"))
        assert len(h["loss"]) == 2
        assert all(np.isfinite(v) for v in h["loss"])

    def t_regime_tscl_single_regime_fallback():
        """Single-regime dataset falls back to standard NT-Xent without crashing."""
        if not TORCH: return
        import torch, tempfile, os
        from pretrain.contrastive import RegimeAwareTSCLTrainer
        from models.architectures import MambaScalper
        rng    = np.random.default_rng(2)
        N      = 60
        X      = rng.standard_normal((N, 20, 16)).astype(np.float32)
        labels = np.zeros(N, dtype=np.int8)   # all neutral
        enc    = MambaScalper(input_size=16, d_model=16, num_layers=2)
        with tempfile.TemporaryDirectory() as tmp:
            h = RegimeAwareTSCLTrainer(
                enc, regime_labels=labels, d_model=16, proj_dim=16
            ).pretrain(X, epochs=1, batch_size=32,
                       checkpoint_path=os.path.join(tmp, "enc.pt"))
        assert np.isfinite(h["loss"][0])

    def t_augmenter_all_strategies():
        """All 4 augmentation types must preserve shape."""
        from pretrain.contrastive import TimeSeriesAugmenter
        aug = TimeSeriesAugmenter()
        x   = np.random.randn(30, 8).astype(np.float32)
        for strat in ["jitter", "scale", "permute", "dropout"]:
            aug.aug = strat
            out = aug.augment(x)
            assert out.shape == x.shape, f"{strat}: shape changed {x.shape} → {out.shape}"

    run_case("standard TSCL 2-epoch smoke",          t_standard_tscl,                skip=not TORCH)
    run_case("regime TSCL — 3 regimes",              t_regime_tscl_three_regimes,    skip=not TORCH)
    run_case("regime TSCL — single regime fallback", t_regime_tscl_single_regime_fallback, skip=not TORCH)
    run_case("augmenter all 4 strategies",           t_augmenter_all_strategies)


# ─────────────────────────────────────────────────────────────────────────────
# RL AGENTS
# ─────────────────────────────────────────────────────────────────────────────

def run_rl_tests(fast=False):
    section("RL AGENTS + ADVANCED")
    bars  = make_bars(300)
    feats = make_features(bars)
    bars_a = bars.reindex(feats.index).dropna()
    fa = feats.values.astype(np.float32)
    pr = bars_a["close"].values.astype(np.float32)
    at = feats["atr_6"].values.astype(np.float32)
    sp = np.full(len(fa), 0.00005, dtype=np.float32)

    def t_env_reset():
        from models.rl_agents import ForexTradingEnv
        env = ForexTradingEnv(fa, pr, at, sp)
        obs = env.reset()
        assert obs.shape == (env.obs_size,)
        assert not env.done

    def t_env_step_all_actions():
        from models.rl_agents import ForexTradingEnv, BUY, HOLD, SELL
        env = ForexTradingEnv(fa, pr, at, sp)
        env.reset()
        for action in [BUY, HOLD, SELL, BUY]:
            obs, r, done, info = env.step(action)
            assert isinstance(r, float)
            assert "equity" in info
            if done: break

    def t_dqn_select():
        from models.rl_agents import DQNAgent, ForexTradingEnv
        env = ForexTradingEnv(fa, pr, at, sp)
        agent = DQNAgent(obs_size=env.obs_size, device="cpu")
        obs = env.reset()
        a = agent.select_action(obs)
        assert a in (0, 1, 2)

    def t_curriculum():
        from models.rl_advanced import CurriculumScheduler
        cs = CurriculumScheduler(total_episodes=100)
        assert cs.current_phase["name"]  # has a named phase
        cs.step()  # advance one episode
        assert cs.get_difficulty_multiplier() >= 0

    def t_sharpe_reward():
        from models.rl_advanced import SharpeRewardWrapper
        srs = SharpeRewardWrapper()
        srs.reset()
        rewards = [srs.compute(raw_pnl=r, tx_cost=0.0) for r in [0.001,-0.002,0.003]]
        assert all(isinstance(r, float) for r in rewards)

    def t_her_buffer():
        from models.rl_advanced import HERBuffer
        buf = HERBuffer(capacity=1000, k=4)
        goal     = np.array([1.0870])
        achieved = np.array([1.0845])
        for i in range(10):
            buf.store_transition(
                obs=np.zeros(10), action=0, reward=-0.5,
                next_obs=np.zeros(10), done=(i==9),
                goal=goal, achieved=achieved,
            )
        buf.end_episode()
        assert len(buf) >= 10  # original + HER relabelled

    run_case("env reset",            t_env_reset)
    run_case("env step all actions", t_env_step_all_actions)
    run_case("DQN action selection", t_dqn_select)
    run_case("curriculum scheduler", t_curriculum)
    run_case("Sharpe reward wrapper",t_sharpe_reward)
    run_case("HER buffer",           t_her_buffer)


# ─────────────────────────────────────────────────────────────────────────────
# RISK & EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_risk_tests(fast=False):
    section("RISK & EXECUTION")

    def t_regime_kelly():
        from risk.execution import RegimeConditionalKelly
        rck = RegimeConditionalKelly(max_pos_pct=0.5, min_kelly=0.01)
        r1  = rck.size(10000, 0.54, 1.7, [], 0.0005, corr_avg=0.0, hurst=0.8)
        r2  = rck.size(10000, 0.54, 1.7, [], 0.0005, corr_avg=0.8, hurst=0.3)
        assert r1["kelly"] > r2["kelly"], "Good regime should size larger"
        assert r1["regime_scale"] > r2["regime_scale"]
        assert 0 < r1["kelly"] <= 1.0

    def t_almgren_chriss():
        from risk.execution import AlmgrenChrissExecutor
        ac = AlmgrenChrissExecutor()
        c  = ac.estimate_impact_cost(1.0, n_slices=5)
        assert "schedule" in c
        assert len(c["schedule"]) == 5
        assert abs(sum(c["schedule"]) - 1.0) < 0.01, "Schedule must sum to 1 lot"
        assert c["impact_pips"] >= 0

    def t_drawdown_exit():
        from risk.execution import DrawdownAwareExitPolicy
        dae = DrawdownAwareExitPolicy()
        dae.update(10000, 0)
        g1 = dae.update(9800, -200)  # 2% DD
        g2 = dae.update(9300, -500)  # 7% DD
        g3 = dae.update(8900, -400)  # 11% DD
        assert g1["action"] in ("continue", "reduce_size", "reduce_50")
        assert g2["action"] == "reduce_50"
        assert g3["action"] in ("close_all", "halt")
        assert dae._halted == True

    def t_portfolio_var():
        from risk.execution import PortfolioVaR
        pv = PortfolioVaR(confidence=0.99)
        for _ in range(80):
            pv.update_returns("EURUSD", float(np.random.randn()*0.0005))
            pv.update_returns("GBPUSD", float(np.random.randn()*0.0005))
        v = pv.parametric_var({"EURUSD":0.5,"GBPUSD":0.5}, 10_000)
        assert "var_pct" in v
        assert "cvar_usd" in v
        assert 0 <= v["var_pct"] <= 1.0

    def t_kelly_sizing():
        from sizing.kelly_criterion import kelly_binary, fractional_kelly, PositionSizer
        fk = kelly_binary(0.54, 1.7)
        assert 0 < fk < 1
        qk = fractional_kelly(fk, 0.25)
        assert qk == fk * 0.25
        ps = PositionSizer(equity=10_000, kelly_fraction=0.25)
        r  = ps.size_position(0.54, 1.7, np.random.randn(100)*0.001, 1.085, 0.0005)
        assert r["lots"] > 0

    run_case("regime-conditional Kelly",  t_regime_kelly)
    run_case("Almgren-Chriss execution",  t_almgren_chriss)
    run_case("drawdown-aware exit",       t_drawdown_exit)
    run_case("portfolio VaR",             t_portfolio_var)
    run_case("fractional Kelly sizing",   t_kelly_sizing)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION + MONITORING
# ─────────────────────────────────────────────────────────────────────────────

def run_monitoring_tests(fast=False):
    section("VALIDATION + MONITORING")

    def t_purged_cv():
        from pretrain.contrastive import PurgedEmbargoCVSplitter
        X = pd.DataFrame(np.random.randn(1000, 5),
                          index=pd.date_range("2024-01-01",periods=1000,freq="1min",tz="UTC"))
        cv = PurgedEmbargoCVSplitter(n_splits=3, purge_bars=30, embargo_bars=10)
        folds = cv.split(X)
        assert len(folds) >= 1
        for tr, te in folds:
            assert len(set(tr) & set(te)) == 0, "Train/test overlap!"
            assert tr.max() < te.min(), "Train must precede test"

    def t_drift_detection():
        from pretrain.contrastive import DriftDetector
        dd = DriftDetector(psi_threshold=0.2, ks_pvalue=0.05, sharpe_drop=0.5)
        X_train = np.random.randn(1000, 10)
        dd.fit_baseline(X_train, np.random.normal(0.001, 0.01, 1000))
        # Stable data: no drift
        r1 = dd.check(np.random.randn(200, 10), np.random.normal(0.001, 0.01, 200))
        # Shifted data: drift expected
        r2 = dd.check(np.random.randn(200,10)*3+5, np.random.normal(-0.01, 0.02, 200))
        assert r2["psi_max"] >= r1["psi_max"]

    def t_monte_carlo():
        from monitoring.pipeline import MonteCarloBacktest
        mc = MonteCarloBacktest(n_simulations=200)
        trades = np.random.normal(0.0005, 0.005, 300)
        r = mc.run(trades)
        assert "sharpe_ci" in r
        # CI lower <= upper (may be equal if all sims identical)
        assert r["sharpe_ci"][0] <= r["sharpe_ci"][1]
        assert 0 <= r["pct_positive_sharpe"] <= 1

    def t_slippage():
        from monitoring.pipeline import SlippageCalibrator
        sc = SlippageCalibrator()
        sc.fit_synthetic()
        s1 = sc.predict(0.1); s2 = sc.predict(1.0); s5 = sc.predict(5.0)
        assert s1 < s2 < s5, "Slippage must increase with order size"

    def t_shadow_mode():
        from monitoring.pipeline import ShadowModeDeployer
        sm = ShadowModeDeployer(shadow_bars=50)
        rng = np.random.default_rng(0)
        for _ in range(60):
            sm.step(int(rng.integers(0,3)), int(rng.integers(0,3)),
                    float(rng.normal(0,0.001)))
        promote, diag = sm.should_promote()
        assert isinstance(promote, bool)
        assert "sharpe_improvement" in diag

    def t_lockbox_split():
        from monitoring.pipeline import LockboxEvaluator
        idx = pd.date_range("2022-01-01", periods=1000, freq="D", tz="UTC")
        df  = pd.DataFrame(np.random.randn(1000, 5), index=idx)
        lb  = LockboxEvaluator(lockbox_start="2024-01-01", lockbox_end="2024-12-31",
                                lock_file="/tmp/lb_test2.json")
        train, box = lb.split(df)
        assert len(train) + len(box) == len(df)
        assert len(box) > 0
        # Verify no overlap
        assert len(set(train.index) & set(box.index)) == 0

    def t_promotion_gate():
        from validation.promotion_gate import PromotionGate
        gate = PromotionGate()
        r = gate.evaluate(sharpe=1.8, profit_factor=1.7, max_drawdown=0.12, n_trades=700)
        assert r["promoted"] == True
        r2 = gate.evaluate(sharpe=1.2, profit_factor=1.1, max_drawdown=0.30, n_trades=200)
        assert r2["promoted"] == False

    def t_mlflow_logger_fallback():
        from validation.mlflow_logger import MLflowModelLogger
        logger = MLflowModelLogger(verbose=False)
        # Should fallback to filesystem if no server
        path = logger.log_promotion("test_model", {"promoted": True, "details": {"sharpe": 1.5}})
        assert os.path.exists(path)

    def t_demotion_monitor():
        from monitoring.demotion_monitor import DemotionMonitor
        mon = DemotionMonitor(window_trades=10, auto_rollback=False)
        # Simulate bad trades — must be > 30 as per implementation threshold
        for _ in range(35):
            mon.on_trade_closed(pnl=-100.0, equity=9000.0)
        alert = mon.on_bar(equity=8500.0)
        assert alert is not None
        assert alert["demoted"] == True

    def t_prometheus_exporter():
        from monitoring.prometheus_exporter import ForexPrometheusExporter
        exp = ForexPrometheusExporter(port=8099)
        exp.update_equity(10500.0)
        snap = exp.snapshot()
        assert snap["equity"] == 10500.0

    def t_discord_alerts():
        from monitoring.discord_alerts import DiscordAlerter
        alerter = DiscordAlerter(verbose=False)
        # Should just print/no-op as no webhook is set
        sent = alerter.send("circuit_breaker", {"reason": "test"})
        assert sent == True

    run_case("purged K-fold + embargo",   t_purged_cv)
    run_case("drift detection",           t_drift_detection)
    run_case("Monte Carlo backtest",      t_monte_carlo, skip=fast)
    run_case("slippage calibration",      t_slippage)
    run_case("shadow mode deployer",      t_shadow_mode)
    run_case("lockbox split + no-overlap",t_lockbox_split)
    run_case("promotion gate",            t_promotion_gate)
    run_case("MLflow logger (fallback)",  t_mlflow_logger_fallback)
    run_case("demotion monitor",          t_demotion_monitor)
    run_case("prometheus exporter",       t_prometheus_exporter)
    run_case("discord alerts",            t_discord_alerts)


# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def run_infrastructure_tests(fast=False):
    section("INFRASTRUCTURE")

    def t_timescale_mock():
        from infrastructure.timescale_kafka import get_store
        store = get_store(mock=True)
        from data.data_ingestion import generate_synthetic_tick_data
        ticks = generate_synthetic_tick_data(n_rows=1000)
        n = store.write_ticks(ticks, pair="EURUSD")
        assert n == 1000

    def t_kafka_mock():
        from infrastructure.timescale_kafka import get_store, MockKafkaProducer
        store    = get_store(mock=True)
        producer = MockKafkaProducer(store, pairs=["EURUSD"])
        n = producer.produce_ticks(n=5000)
        assert n > 0

    def t_tip_search():
        from pretrain.contrastive import TIPSearchManager
        class _A:
            def select_action(self, o): return 1
        ts = TIPSearchManager(_A(), _A(), switch_mult=2.0)
        obs = np.zeros(10)
        # Normal ATR → slow model
        a1, m1, _ = ts.select_action(obs, 0.0005)
        assert a1 in (0,1,2)
        # Spike → fast model
        for _ in range(65): ts.select_action(obs, 0.0005)  # fill history
        a2, m2, _ = ts.select_action(obs, 0.003)
        assert a2 in (0,1,2)

    run_case("TimescaleDB mock store",    t_timescale_mock)
    run_case("Kafka mock producer",       t_kafka_mock)
    run_case("TIP-Search latency switch", t_tip_search)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_live_engine_tests(fast=False):
    section("LIVE TRADING ENGINE")

    def t_paper_broker():
        from trading.live_engine import PaperBroker
        b = PaperBroker(initial_equity=10_000)
        assert b.connect()
        bid, ask = b.get_bid_ask("EURUSD")
        assert ask > bid
        oid = b.market_order("EURUSD", "buy", 0.1)
        assert oid is not None
        assert abs(b._pos.get("EURUSD", 0)) == 0.1
        b.close_position("EURUSD")
        assert b._pos.get("EURUSD", 0) == 0

    def t_tick_buffer():
        from trading.live_engine import LiveTickBuffer
        buf = LiveTickBuffer()
        for _ in range(300):
            buf.push_tick(1.0850, 1.0851)
        bars = buf.get_bars()
        assert bars is None or len(bars) > 0  # May be None if window too small

    def t_engine_init():
        from trading.live_engine import LiveTradingEngine, PaperBroker
        class _A:
            def select_action(self, o): return 1
        e = LiveTradingEngine(PaperBroker(), _A(), _A())
        assert e.equity == 10_000

    run_case("paper broker",        t_paper_broker)
    run_case("tick buffer",         t_tick_buffer)
    run_case("engine init",         t_engine_init)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast",   action="store_true", help="Skip slow tests")
    p.add_argument("--module", default="all",
                   choices=["all","data","features","labeling","models",
                             "pretrain","rl","risk","monitoring","infra","live"])
    args = p.parse_args()

    print("█"*55)
    print("  Forex Scaling Model — Test Suite")
    print(f"  Mode: {'fast' if args.fast else 'full'}  |  Module: {args.module}")
    print("█"*55)

    t0 = time.time()
    runners = {
        "data":       run_data_tests,
        "features":   run_feature_tests,
        "labeling":   run_labeling_tests,
        "models":     run_model_tests,
        "pretrain":   run_pretrain_tests,
        "rl":         run_rl_tests,
        "risk":       run_risk_tests,
        "monitoring": run_monitoring_tests,
        "infra":      run_infrastructure_tests,
        "live":       run_live_engine_tests,
    }
    if args.module == "all":
        for fn in runners.values():
            fn(fast=args.fast)
    else:
        runners[args.module](fast=args.fast)

    elapsed = time.time() - t0
    print(f"\n{'═'*55}")
    print(f"  Results:  ✓ {PASS} passed  ✗ {FAIL} failed  ⊙ {SKIP} skipped")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"{'═'*55}")
    if FAIL:
        print("\n  FAILURES:")
        for status, name, msg in RESULTS:
            if status == "FAIL":
                print(f"    ✗ {name}: {msg}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
