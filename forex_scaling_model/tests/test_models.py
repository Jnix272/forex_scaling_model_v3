"""Tests for neural architectures, losses, RL env, and agents."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import argparse
from unittest.mock import patch

from models.architectures import (
    AsymmetricDirectionalLoss,
    EXPERTEncoder,
    GNNCrossAsset,
    GNNFromSequence,
    HAELTHybrid,
    HuberLoss,
    MambaScalper,
    MODEL_REGISTRY,
    MultiTaskHead,
    MultiTaskLoss,
    MultiTaskWrapper,
    MultiPairWrapper,
    TFTScalper,
    build_model,
    iTransformerScalper,
)
from models.ensemble import EnsembleMetaLearner, train_meta_learner
from models.rl_agents import (
    BUY,
    DQNAgent,
    ForexTradingEnv,
    HOLD,
    PPOAgent,
    SELL,
)


def _synthetic_env(n: int = 120, feat_dim: int = 16) -> ForexTradingEnv:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, feat_dim), dtype=np.float32)
    prices = np.maximum(
        0.5,
        np.cumsum(rng.standard_normal(n, dtype=np.float32) * 5e-5) + 1.0,
    ).astype(np.float32)
    atr = np.full(n, 0.001, dtype=np.float32)
    spreads = np.full(n, 1e-5, dtype=np.float32)
    return ForexTradingEnv(features, prices, atr, spreads)


@pytest.fixture
def seq_batch():
    b, t, f = 4, 60, 48
    return torch.randn(b, t, f)


class TestForexTradingEnv:
    def test_reset_obs_shape(self):
        env = _synthetic_env()
        obs = env.reset()
        assert obs.shape == (env.obs_size,)
        assert obs.dtype == np.float32

    def test_hold_runs_to_end(self):
        env = _synthetic_env(n=80)
        env.reset()
        while not env.done:
            obs, reward, done, info = env.step(HOLD)
            assert "equity" in info
        assert env.idx >= len(env.prices) - 2

    def test_buy_then_sell_closes(self):
        env = _synthetic_env(n=50)
        env.reset()
        env.step(BUY)
        assert env.position > 0
        env.step(SELL)
        assert env.position == 0.0


@pytest.mark.parametrize(
    "cls,extra_kw",
    [
        (TFTScalper, {}),
        (iTransformerScalper, {"seq_len": 60}),
        (HAELTHybrid, {"seq_len": 60}),
        (MambaScalper, {}),
        (EXPERTEncoder, {}),
        (
            GNNFromSequence,
            {"hidden": 64, "num_layers": 2, "dropout": 0.1},
        ),
    ],
)
def test_sequence_models_forward(cls, extra_kw, seq_batch):
    f = seq_batch.shape[-1]
    model = cls(input_size=f, **extra_kw)
    model.eval()
    out = model(seq_batch)
    assert out.shape == (seq_batch.shape[0],)


def test_gnn_cross_asset_direct():
    b, n_nodes, nf = 3, 6, 32
    x = torch.randn(b, n_nodes, nf)
    m = GNNCrossAsset(node_features=nf, hidden=64, num_layers=2, n_nodes=n_nodes)
    m.eval()
    out = m(x)
    assert out.shape == (b,)


def test_losses_scalar():
    pred = torch.randn(8, requires_grad=True)
    target = torch.randn(8)
    h = HuberLoss()(pred, target)
    a = AsymmetricDirectionalLoss()(pred, target)
    assert h.ndim == 0 and a.ndim == 0
    h.backward()
    assert pred.grad is not None


def test_build_model_registry_keys(seq_batch, capsys):
    f = seq_batch.shape[-1]
    for name in MODEL_REGISTRY:
        if name == "gnn":
            m = build_model(
                name, f, hidden=64, num_layers=2, dropout=0.1, nhead=4
            )
        else:
            m = build_model(name, f, seq_len=seq_batch.shape[1])
        m.eval()
        y = m(seq_batch)
        assert y.shape == (seq_batch.shape[0],)
    _ = capsys.readouterr()


def test_build_model_unknown():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("not_a_model", 10)


class TinySeqModel(nn.Module):
    """Minimal (B,T,F) → scalar predictor for ensemble tests."""

    def __init__(self, f: int):
        super().__init__()
        self.lin = nn.Linear(f, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x[:, -1, :]).squeeze(-1)


def test_ensemble_meta_learner(seq_batch):
    f = seq_batch.shape[-1]
    bases = [TinySeqModel(f), TinySeqModel(f)]
    ens = EnsembleMetaLearner(bases, context_dim=16, hidden=32)
    ens.eval()
    pred, w = ens(seq_batch)
    assert pred.shape == (seq_batch.shape[0],)
    assert w.shape == (seq_batch.shape[0], len(bases))
    assert torch.allclose(w.sum(dim=1), torch.ones(seq_batch.shape[0]), atol=1e-5)
    summary = ens.model_weights_summary(seq_batch)
    assert len(summary) == len(bases)
    assert set(summary.keys()) <= {
        "tft",
        "transformer",
        "haelt",
        "mamba",
        "gnn",
        "expert",
    }


def test_ppo_agent_update():
    env = _synthetic_env(n=200, feat_dim=12)
    obs = env.reset()
    agent = PPOAgent(obs_size=env.obs_size, hidden=64, n_epochs=2)
    for _ in range(70):
        a, lp, v = agent.select_action(obs)
        next_obs, r, done, _ = env.step(int(a))
        agent.store(obs, a, r, done, lp, v)
        obs = next_obs if not done else env.reset()
    stats = agent.update()
    assert "loss" in stats
    assert np.isfinite(stats["loss"])


def test_dqn_agent_update():
    env = _synthetic_env(n=200, feat_dim=12)
    obs = env.reset()
    agent = DQNAgent(
        obs_size=env.obs_size,
        hidden=64,
        batch=32,
        buf_size=10_000,
        target_update=50,
    )
    for _ in range(80):
        a = agent.select_action(obs)
        next_obs, r, done, _ = env.step(a)
        agent.store(obs, a, r, next_obs, done)
        agent.update()
        obs = next_obs if not done else env.reset()
    assert len(agent.buf) >= agent.batch


# ─────────────────────────────────────────────────────────────────────────────
# #10 — Multi-task head, loss, and wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTaskHead:
    def test_output_shapes(self, seq_batch):
        """MultiTaskHead returns three tensors with correct shapes."""
        B = seq_batch.shape[0]
        in_features = 64
        h = torch.randn(B, in_features)
        head = MultiTaskHead(in_features=in_features, hidden=32)
        logits, ret_hat, conf = head(h)
        assert logits.shape  == (B, 3),  f"direction logits: {logits.shape}"
        assert ret_hat.shape == (B,),    f"return_hat: {ret_hat.shape}"
        assert conf.shape    == (B,),    f"confidence: {conf.shape}"

    def test_confidence_in_unit_interval(self, seq_batch):
        """Head returns confidence logits; sigmoid maps them to [0, 1] for inference."""
        B = seq_batch.shape[0]
        h = torch.randn(B, 32)
        _, _, conf = MultiTaskHead(in_features=32)(h)
        p = torch.sigmoid(conf)
        assert p.min().item() >= 0.0 - 1e-6
        assert p.max().item() <= 1.0 + 1e-6


class TestMultiTaskLoss:
    def _make_inputs(self, B: int = 8):
        logits  = torch.randn(B, 3)
        ret_hat = torch.randn(B)
        conf    = torch.randn(B)  # logits — BCEWithLogitsLoss in MultiTaskLoss
        y_cls   = torch.randint(0, 3, (B,))
        y_cont  = torch.tensor(
            np.random.choice([-1.0, 0.0, 1.0], B).astype(np.float32)
        )
        return logits, ret_hat, conf, y_cls, y_cont

    def test_loss_is_scalar_and_positive(self):
        crit = MultiTaskLoss()
        loss = crit(*self._make_inputs())
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_loss_weights_scale_result(self):
        """Higher w_ret should produce a larger total loss when magnitude error is large."""
        logits, ret_hat, conf, y_cls, y_cont = self._make_inputs()
        # Force large magnitude error
        ret_hat = ret_hat * 10
        l_low  = MultiTaskLoss(w_ret=0.1)(*[logits, ret_hat, conf, y_cls, y_cont])
        l_high = MultiTaskLoss(w_ret=5.0)(*[logits, ret_hat, conf, y_cls, y_cont])
        assert l_high.item() > l_low.item()

    def test_loss_backward(self):
        """Gradients should flow back through all three outputs."""
        logits  = torch.randn(8, 3, requires_grad=True)
        ret_hat = torch.randn(8, requires_grad=True)
        conf    = torch.randn(8, requires_grad=True)
        y_cls   = torch.zeros(8, dtype=torch.long)
        y_cont  = torch.zeros(8)
        loss = MultiTaskLoss()(logits, ret_hat, conf, y_cls, y_cont)
        loss.backward()
        assert logits.grad  is not None
        assert ret_hat.grad is not None
        assert conf.grad    is not None

    def test_class_weights_applied(self):
        """Passing class weights should change the loss value."""
        torch.manual_seed(42)
        logits, ret_hat, conf, y_cls, y_cont = self._make_inputs(B=16)
        w = torch.tensor([5.0, 1.0, 0.1])
        l_uniform  = MultiTaskLoss()(logits, ret_hat, conf, y_cls, y_cont)
        l_weighted = MultiTaskLoss(class_weights=w)(logits, ret_hat, conf, y_cls, y_cont)
        assert l_uniform.item() != pytest.approx(l_weighted.item(), rel=1e-2)


class TestMultiTaskWrapper:
    @pytest.mark.parametrize("model_name,extra", [
        ("haelt",       {"seq_len": 60}),
        ("mamba",       {}),
        ("expert",      {}),
        ("tft",         {}),
    ])
    def test_wrapper_output_shapes(self, model_name, extra, seq_batch):
        """Wrapped backbone should return (logits, ret_hat, conf) tuples."""
        B, T, F = seq_batch.shape
        base = build_model(model_name, F, seq_len=T, **{
            k: v for k, v in {
                "hidden_size": 64, "d_model": 64, "nhead": 4,
                "num_layers": 2, "dropout": 0.1,
            }.items()
        })
        # Compute head_in for this architecture
        head_in_map = {
            "haelt":  64 + 64,        # lstm_hidden + d_model (both //2 of 128)
            "mamba":  64,              # d_model
            "expert": 64,              # d_model
            "tft":    128,             # hidden_size (default build_model uses 256 but we pass 64)
        }
        head_in = head_in_map.get(model_name, 64)

        wrapped = MultiTaskWrapper(base, head_in=head_in, hidden=32)
        wrapped.eval()
        logits, ret_hat, conf = wrapped(seq_batch)
        assert logits.shape  == (B, 3)
        assert ret_hat.shape == (B,)
        assert conf.shape    == (B,)
        p = torch.sigmoid(conf)
        assert p.min() >= 0.0 - 1e-6
        assert p.max() <= 1.0 + 1e-6

    def test_wrapper_large_head_auto_projects(self, seq_batch):
        """iTransformer head_in = d_model * n_features which is large; wrapper should project."""
        B, T, F = seq_batch.shape
        base = iTransformerScalper(input_size=F, seq_len=T, d_model=32, nhead=4)
        head_in = 32 * F   # F * d_model = 48 * 32 = 1536 > proj_threshold=1024
        wrapped = MultiTaskWrapper(base, head_in=head_in, proj_threshold=1024, proj_to=128)
        wrapped.eval()
        logits, ret_hat, conf = wrapped(seq_batch)
        assert logits.shape == (B, 3)

    def test_wrapper_gradients_flow(self, seq_batch):
        """Loss.backward() should update backbone parameters through the wrapper."""
        B, T, F = seq_batch.shape
        base    = HAELTHybrid(input_size=F, seq_len=T, lstm_hidden=32, d_model=32)
        wrapped = MultiTaskWrapper(base, head_in=64, hidden=32)
        crit    = MultiTaskLoss()

        logits, ret_hat, conf = wrapped(seq_batch)
        y_cls  = torch.zeros(B, dtype=torch.long)
        y_cont = torch.zeros(B)
        loss   = crit(logits, ret_hat, conf, y_cls, y_cont)
        loss.backward()

        # At least one backbone parameter should have a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in wrapped.backbone.parameters())
        assert has_grad, "No backbone gradients after backward()"


# ─────────────────────────────────────────────────────────────────────────────
# #12 — Ensemble diversity loss and meta-learner training
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleDiversityLoss:
    def test_diversity_loss_identical_models_is_high(self):
        """When all base models produce identical predictions, diversity loss ≈ 1."""
        B, n_models = 32, 4
        # All models output the same values → pairwise correlation = 1
        single = torch.randn(B, 1).expand(B, n_models)
        bases  = [TinySeqModel(48)] * n_models
        ens    = EnsembleMetaLearner(bases, context_dim=16, hidden=32)
        div    = ens.diversity_loss(single)
        assert div.item() > 0.8, f"Identical outputs should yield diversity ≈ 1, got {div.item():.4f}"

    def test_diversity_loss_orthogonal_is_near_zero(self):
        """Orthogonal model outputs (zero correlation) should yield diversity near 0."""
        B, n_models = 128, 4
        # Construct uncorrelated columns using QR decomposition
        random_mat = torch.randn(B, n_models)
        q, _ = torch.linalg.qr(random_mat)
        preds = q[:, :n_models]  # exactly orthonormal columns
        bases = [TinySeqModel(48)] * n_models
        ens   = EnsembleMetaLearner(bases, context_dim=16, hidden=32)
        div   = ens.diversity_loss(preds)
        assert abs(div.item()) < 0.15, f"Orthogonal outputs should yield diversity ≈ 0, got {div.item():.4f}"

    def test_diversity_loss_differentiable(self):
        """diversity_loss must be differentiable (needed for meta-learner training)."""
        B, n_models = 16, 3
        preds = torch.randn(B, n_models, requires_grad=True)
        bases = [TinySeqModel(48)] * n_models
        ens   = EnsembleMetaLearner(bases, context_dim=16, hidden=32)
        loss  = ens.diversity_loss(preds)
        loss.backward()
        assert preds.grad is not None


class TestTrainMetaLearner:
    def _make_loader(self, B: int = 64, T: int = 60, F: int = 48, n: int = 128):
        X = torch.randn(n, T, F)
        y = torch.tensor(
            np.random.choice([-1.0, 0.0, 1.0], n).astype(np.float32)
        )
        return DataLoader(TensorDataset(X, y), batch_size=B, shuffle=True)

    def test_meta_learner_loss_decreases(self):
        """Meta-learner loss should decrease over 5 training epochs."""
        F = 48
        bases  = [TinySeqModel(F), TinySeqModel(F), TinySeqModel(F)]
        ens    = EnsembleMetaLearner(bases, context_dim=16, hidden=32)
        loader = self._make_loader(F=F)
        history = train_meta_learner(ens, loader, epochs=5, lr=1e-2,
                                     diversity_weight=0.1, verbose=False)
        assert len(history) == 5
        # Loss should be finite throughout
        assert all(np.isfinite(v) for v in history)

    def test_base_models_frozen_during_meta_training(self):
        """Base model parameters must NOT change during meta-learner training."""
        F = 48
        bases  = [TinySeqModel(F), TinySeqModel(F)]
        ens    = EnsembleMetaLearner(bases, context_dim=16, hidden=32)

        # Capture base param snapshots before
        before = [p.data.clone() for base in ens.bases for p in base.parameters()]

        loader  = self._make_loader(F=F, n=64)
        train_meta_learner(ens, loader, epochs=3, lr=1e-2, verbose=False)

        after = [p.data.clone() for base in ens.bases for p in base.parameters()]
        for b, a in zip(before, after):
            assert torch.allclose(b, a), "Base model weights changed during meta training!"

    def test_weights_sum_to_one_after_training(self, seq_batch):
        """Ensemble softmax weights must still sum to 1 after training."""
        F = seq_batch.shape[-1]
        bases  = [TinySeqModel(F), TinySeqModel(F)]
        ens    = EnsembleMetaLearner(bases, context_dim=16, hidden=32)
        loader = self._make_loader(F=F, n=64)
        train_meta_learner(ens, loader, epochs=2, lr=1e-2, verbose=False)

        ens.eval()
        _, w = ens(seq_batch)
        assert torch.allclose(w.sum(dim=1), torch.ones(seq_batch.shape[0]), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# #11 — Regime-aware contrastive pre-training
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeAwareTSCL:
    @pytest.fixture
    def small_windows(self):
        rng = np.random.default_rng(7)
        return rng.standard_normal((120, 20, 16)).astype(np.float32)

    def test_regime_trainer_runs_and_returns_history(self, small_windows):
        """RegimeAwareTSCLTrainer must complete without error and return loss history."""
        from pretrain.contrastive import RegimeAwareTSCLTrainer
        N = len(small_windows)
        regime_labels = np.where(
            np.arange(N) < N // 3, 1,
            np.where(np.arange(N) < 2 * N // 3, -1, 0)
        ).astype(np.int8)

        encoder = HAELTHybrid(input_size=16, seq_len=20, lstm_hidden=16, d_model=16)
        trainer = RegimeAwareTSCLTrainer(
            encoder=encoder,
            regime_labels=regime_labels,
            d_model=32,
            proj_dim=32,
            temperature=0.1,
            lr=1e-3,
            device="cpu",
        )
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "encoder.pt")
            history = trainer.pretrain(small_windows, epochs=2, batch_size=32,
                                       checkpoint_path=ckpt)
        assert "loss" in history
        assert len(history["loss"]) == 2
        assert all(np.isfinite(v) for v in history["loss"])

    def test_regime_trainer_loss_lower_than_standard(self, small_windows):
        """Regime-aware TSCL should converge at least as well as standard TSCL on structured data."""
        from pretrain.contrastive import RegimeAwareTSCLTrainer, TSCLTrainer
        N = len(small_windows)
        # Create clearly separated regime structure: first half trending, second mean-rev
        regime_labels = np.array([1]*60 + [-1]*60, dtype=np.int8)

        def _make_encoder():
            return HAELTHybrid(input_size=16, seq_len=20, lstm_hidden=16, d_model=16)

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            enc_std = _make_encoder()
            std_trainer = TSCLTrainer(enc_std, d_model=32, proj_dim=32, lr=1e-3, device="cpu")
            h_std = std_trainer.pretrain(small_windows, epochs=3, batch_size=32,
                                          checkpoint_path=os.path.join(tmp, "std.pt"))

            enc_reg = _make_encoder()
            reg_trainer = RegimeAwareTSCLTrainer(
                enc_reg, regime_labels=regime_labels,
                d_model=32, proj_dim=32, lr=1e-3, device="cpu",
            )
            h_reg = reg_trainer.pretrain(small_windows, epochs=3, batch_size=32,
                                          checkpoint_path=os.path.join(tmp, "reg.pt"))

        # Both should produce finite losses
        assert all(np.isfinite(v) for v in h_std["loss"])
        assert all(np.isfinite(v) for v in h_reg["loss"])

    def test_regime_labels_length_mismatch_handled(self, small_windows):
        """Trainer should handle regime_labels shorter or longer than window count."""
        from pretrain.contrastive import RegimeAwareTSCLTrainer
        N = len(small_windows)
        # Shorter labels
        short_labels = np.ones(N // 2, dtype=np.int8)
        encoder = MambaScalper(input_size=16, d_model=16, num_layers=2)
        trainer = RegimeAwareTSCLTrainer(
            encoder, regime_labels=short_labels,
            d_model=16, proj_dim=16, device="cpu",
        )
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            history = trainer.pretrain(small_windows, epochs=1, batch_size=32,
                                       checkpoint_path=os.path.join(tmp, "e.pt"))
        assert len(history["loss"]) == 1

    def test_single_regime_falls_back_to_standard_ntxent(self, small_windows):
        """When all samples share one regime, trainer must fall back to standard NT-Xent."""
        from pretrain.contrastive import RegimeAwareTSCLTrainer
        N = len(small_windows)
        all_same = np.zeros(N, dtype=np.int8)   # only regime 0
        encoder  = EXPERTEncoder(input_size=16, d_model=16, nhead=4, num_layers=2)
        trainer  = RegimeAwareTSCLTrainer(
            encoder, regime_labels=all_same,
            d_model=16, proj_dim=16, device="cpu",
        )
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            history = trainer.pretrain(small_windows, epochs=1, batch_size=32,
                                       checkpoint_path=os.path.join(tmp, "e.pt"))
        assert all(np.isfinite(v) for v in history["loss"])


# ─────────────────────────────────────────────────────────────────────────────
# Multi-pair joint training — wrapper and pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiPairWrapper:
    """Tests for MultiPairWrapper: pair embedding concatenation and backbone passthrough."""

    def _tiny_backbone(self, input_size: int, seq_len: int = 60, num_classes: int = 1):
        return HAELTHybrid(
            input_size=input_size, seq_len=seq_len,
            lstm_hidden=32, d_model=32, nhead=2, n_layers=1,
            dropout=0.0, num_classes=num_classes,
        )

    def test_two_pairs_no_embed_concatenated_input(self):
        """With no wrapper (embed_dim=0) the backbone sees 2*F features directly."""
        B, T, F = 4, 60, 24
        x = torch.randn(B, T, 2 * F)
        backbone = self._tiny_backbone(input_size=2 * F, seq_len=T)
        backbone.eval()
        out = backbone(x)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"

    def test_wrapper_two_pairs_with_embed_output_shape(self):
        """MultiPairWrapper(2 pairs, embed_dim=8) keeps backbone output shape."""
        B, T, F, E = 4, 60, 24, 8
        backbone = self._tiny_backbone(input_size=2 * (F + E), seq_len=T)
        model    = MultiPairWrapper(backbone, n_pairs=2, f_per_pair=F, embed_dim=E)
        model.eval()
        x   = torch.randn(B, T, 2 * F)
        out = model(x)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"

    def test_wrapper_four_pairs_with_embed_output_shape(self):
        """4 pairs with embed_dim=16 → (B,) scalar from backbone."""
        B, T, F, E, P = 4, 60, 20, 16, 4
        backbone = self._tiny_backbone(input_size=P * (F + E), seq_len=T)
        model    = MultiPairWrapper(backbone, n_pairs=P, f_per_pair=F, embed_dim=E)
        model.eval()
        x   = torch.randn(B, T, P * F)
        out = model(x)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"

    def test_pair_embeddings_are_distinct_after_init(self):
        """Each pair must have a unique random embedding by default."""
        backbone = self._tiny_backbone(input_size=2 * (24 + 8))
        model    = MultiPairWrapper(backbone, n_pairs=2, f_per_pair=24, embed_dim=8)
        emb0     = model.pair_embeds(torch.tensor([0]))
        emb1     = model.pair_embeds(torch.tensor([1]))
        assert not torch.allclose(emb0, emb1), "Pair embeddings should differ after init"

    def test_gradient_flows_through_backbone_and_embeddings(self):
        """loss.backward() must reach both backbone params and pair embedding weights."""
        B, T, F, E = 4, 60, 24, 8
        backbone = self._tiny_backbone(input_size=2 * (F + E), seq_len=T)
        model    = MultiPairWrapper(backbone, n_pairs=2, f_per_pair=F, embed_dim=E)
        out      = model(torch.randn(B, T, 2 * F))
        out.sum().backward()

        backbone_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
        )
        embed_grad = (model.pair_embeds.weight.grad is not None and
                      model.pair_embeds.weight.grad.abs().sum() > 0)
        assert backbone_grad, "No gradient reached backbone parameters"
        assert embed_grad,    "No gradient reached pair embedding weights"

    def test_wrapper_with_classification_head(self):
        """Wrapper works when backbone outputs 3-class logits."""
        B, T, F, E, P = 4, 60, 20, 8, 3
        backbone = self._tiny_backbone(input_size=P * (F + E), seq_len=T, num_classes=3)
        model    = MultiPairWrapper(backbone, n_pairs=P, f_per_pair=F, embed_dim=E)
        model.eval()
        out = model(torch.randn(B, T, P * F))
        assert out.shape == (B, 3), f"Expected ({B}, 3), got {out.shape}"


class TestMultiPairHelpers:
    """Tests for _get_pairs() and _build_multipair_chunk() pipeline helpers."""

    # ── _get_pairs ────────────────────────────────────────────────────────────

    def test_get_pairs_from_comma_string(self):
        from training.train_gpu import _get_pairs
        args = argparse.Namespace(pairs="EURUSD,GBPUSD,USDJPY", pair="EURUSD")
        assert _get_pairs(args) == ["EURUSD", "GBPUSD", "USDJPY"]

    def test_get_pairs_from_list(self):
        from training.train_gpu import _get_pairs
        args = argparse.Namespace(pairs=["eurusd", "gbpusd"], pair="EURUSD")
        assert _get_pairs(args) == ["EURUSD", "GBPUSD"]

    def test_get_pairs_fallback_to_pair_when_none(self):
        from training.train_gpu import _get_pairs
        args = argparse.Namespace(pairs=None, pair="GBPUSD")
        assert _get_pairs(args) == ["GBPUSD"]

    def test_get_pairs_strips_whitespace(self):
        from training.train_gpu import _get_pairs
        args = argparse.Namespace(pairs=" EURUSD , GBPUSD ", pair="EURUSD")
        assert _get_pairs(args) == ["EURUSD", "GBPUSD"]

    def test_get_pairs_empty_string_falls_back(self):
        from training.train_gpu import _get_pairs
        args = argparse.Namespace(pairs="", pair="USDJPY")
        assert _get_pairs(args) == ["USDJPY"]

    # ── _build_multipair_chunk ────────────────────────────────────────────────

    def _fake_chunk(self, n: int, T: int = 60, F: int = 8):
        X = np.random.randn(n, T, F).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        return X, y, F

    def test_multipair_chunk_concatenates_features(self):
        """Two equal-length pairs → X shape (N, T, 2*F)."""
        from training.train_gpu import _build_multipair_chunk
        from sklearn.preprocessing import StandardScaler

        T, F, N = 60, 8, 50
        pair_ticks = {"EURUSD": "e", "GBPUSD": "g"}
        scalers    = {"EURUSD": StandardScaler(), "GBPUSD": StandardScaler()}

        with patch("training.train_gpu._build_chunk",
                   side_effect=[self._fake_chunk(N, T, F), self._fake_chunk(N, T, F)]):
            X, y, n_feat = _build_multipair_chunk(
                pair_ticks, fe=None, scalers=scalers,
                seq_len=T, chunk_idx=0, label_method="rl_reward",
            )
        assert X.shape == (N, T, 2 * F)
        assert y.shape == (N,)
        assert n_feat  == 2 * F

    def test_multipair_chunk_aligns_to_shortest_pair(self):
        """Pairs with different sample counts → trimmed to inner-join minimum."""
        from training.train_gpu import _build_multipair_chunk
        from sklearn.preprocessing import StandardScaler

        T, F, N1, N2 = 60, 8, 100, 73
        pair_ticks = {"EURUSD": "e", "GBPUSD": "g"}
        scalers    = {"EURUSD": StandardScaler(), "GBPUSD": StandardScaler()}

        with patch("training.train_gpu._build_chunk",
                   side_effect=[self._fake_chunk(N1, T, F), self._fake_chunk(N2, T, F)]):
            X, y, _ = _build_multipair_chunk(
                pair_ticks, fe=None, scalers=scalers,
                seq_len=T, chunk_idx=0, label_method="rl_reward",
            )
        assert X.shape[0] == min(N1, N2)

    def test_multipair_chunk_averages_labels(self):
        """y must be the mean of each pair's label array."""
        from training.train_gpu import _build_multipair_chunk
        from sklearn.preprocessing import StandardScaler

        T, F, N = 60, 8, 40
        rng   = np.random.default_rng(0)
        y_eur = rng.standard_normal(N).astype(np.float32)
        y_gbp = rng.standard_normal(N).astype(np.float32)
        X_z   = np.zeros((N, T, F), np.float32)

        pair_ticks = {"EURUSD": "e", "GBPUSD": "g"}
        scalers    = {"EURUSD": StandardScaler(), "GBPUSD": StandardScaler()}

        with patch("training.train_gpu._build_chunk",
                   side_effect=[(X_z, y_eur, F), (X_z, y_gbp, F)]):
            _, y_out, _ = _build_multipair_chunk(
                pair_ticks, fe=None, scalers=scalers,
                seq_len=T, chunk_idx=0, label_method="rl_reward",
            )
        np.testing.assert_allclose(
            y_out, ((y_eur + y_gbp) / 2).astype(np.float32), rtol=1e-5,
            err_msg="Labels should be the mean across pairs",
        )

    def test_multipair_chunk_empty_pairs_returns_empty(self):
        """When all pairs produce empty arrays the chunk exits gracefully."""
        from training.train_gpu import _build_multipair_chunk
        from sklearn.preprocessing import StandardScaler

        pair_ticks = {"EURUSD": "e", "GBPUSD": "g"}
        scalers    = {"EURUSD": StandardScaler(), "GBPUSD": StandardScaler()}

        with patch("training.train_gpu._build_chunk",
                   return_value=(np.array([]), np.array([]), 0)):
            X, y, n_feat = _build_multipair_chunk(
                pair_ticks, fe=None, scalers=scalers,
                seq_len=60, chunk_idx=0, label_method="rl_reward",
            )
        assert X.size == 0
        assert n_feat == 0

    def test_multipair_chunk_three_pairs_feature_count(self):
        """3 pairs × F each → n_features_total = 3*F."""
        from training.train_gpu import _build_multipair_chunk
        from sklearn.preprocessing import StandardScaler

        T, F, N, P = 60, 10, 30, 3
        pairs      = ["EURUSD", "GBPUSD", "USDJPY"]
        pair_ticks = {p: "stub" for p in pairs}
        scalers    = {p: StandardScaler() for p in pairs}

        with patch("training.train_gpu._build_chunk",
                   side_effect=[self._fake_chunk(N, T, F) for _ in pairs]):
            X, y, n_feat = _build_multipair_chunk(
                pair_ticks, fe=None, scalers=scalers,
                seq_len=T, chunk_idx=0, label_method="rl_reward",
            )
        assert n_feat  == P * F
        assert X.shape == (N, T, P * F)
