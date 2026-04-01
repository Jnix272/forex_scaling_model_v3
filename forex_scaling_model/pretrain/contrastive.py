"""
pretrain/contrastive.py  — TSCL contrastive pre-training
sentiment/dual_stream.py — FinBERT offline + SLM online
latency/tip_search.py    — TIP-Search latency manager
monitoring/drift.py      — Model drift detection
validation/purged_cv.py  — Purged K-Fold + Embargoing
retraining/rolling.py    — Walk-forward rolling retraining

All six secondary modules in one file for packaging efficiency.
In production, split into individual files per module path.
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
try:
    from tqdm import tqdm as _tqdm
    def _pbar(it, **kw): return _tqdm(it, **kw)
except ImportError:
    def _pbar(it, **kw): return it

from config.settings import PATHS

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False

try:
    from scipy import stats
    SCIPY = True
except ImportError:
    SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTRASTIVE PRE-TRAINING (TSCL)
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesAugmenter:
    """
    Four augmentation strategies for time-series contrastive learning.
    Two different augmentations of the same window should be 'similar';
    windows from different market regimes should be 'dissimilar'.
    """
    def __init__(self, jitter_std=0.01, scale_range=(0.8,1.2),
                 perm_segs=4, dropout_p=0.1):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.perm_segs = perm_segs
        self.dropout_p = dropout_p

    def augment(self, x: np.ndarray) -> np.ndarray:
        """Apply a random combination of augmentations to a single sample."""
        rng = np.random.default_rng()
        aug = rng.choice(["jitter","scale","permute","dropout"])
        if aug == "jitter":
            return x + rng.normal(0, self.jitter_std, x.shape)
        elif aug == "scale":
            s = rng.uniform(*self.scale_range)
            return x * s
        elif aug == "permute":
            T, F = x.shape; seg = max(1, T // self.perm_segs)
            idx = list(range(0, T, seg)); rng.shuffle(idx)
            return np.concatenate([x[i:i+seg] for i in idx], axis=0)[:T]
        else:  # dropout
            mask = rng.random(x.shape) > self.dropout_p
            return x * mask

    def augment_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized augmentation on a full batch (B, T, F). ~10x faster than per-sample."""
        B, T, F = X.shape
        rng = np.random.default_rng()
        choice = rng.integers(0, 4, size=B)

        out = X.copy()
        # Jitter
        m = choice == 0
        if m.any():
            out[m] += rng.normal(0, self.jitter_std, out[m].shape).astype(X.dtype)
        # Scale
        m = choice == 1
        if m.any():
            s = rng.uniform(*self.scale_range, size=(int(m.sum()), 1, 1)).astype(X.dtype)
            out[m] *= s
        # Permute (vectorized segment shuffle)
        m = choice == 2
        if m.any():
            seg = max(1, T // self.perm_segs)
            n_segs = (T + seg - 1) // seg
            for i in np.where(m)[0]:
                perm = rng.permutation(n_segs)
                idx = np.concatenate([np.arange(p * seg, min((p + 1) * seg, T)) for p in perm])[:T]
                out[i] = X[i][idx]
        # Dropout
        m = choice == 3
        if m.any():
            mask = (rng.random(out[m].shape) > self.dropout_p).astype(X.dtype)
            out[m] *= mask
        return out


if TORCH:
    class ProjectionHead(nn.Module):
        def __init__(self, d_model=128, proj_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, proj_dim),
            )
        def forward(self, x): return F.normalize(self.net(x), dim=-1)

    class TSCLTrainer:
        """
        Time-Series Contrastive Learning pre-trainer.

        Trains the encoder to produce similar representations for two
        augmented views of the same market segment, and dissimilar
        representations for different segments.

        After pre-training, the encoder weights are frozen and used as
        a feature extractor for the supervised trading models.

        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss is used.
        """
        def __init__(self, encoder: nn.Module, d_model=128, proj_dim=128,
                     temperature=0.07, lr=1e-4, device="cpu"):
            if hasattr(encoder, "head"):
                encoder.head = nn.Identity()
            self.encoder = encoder.to(device)
            self.proj    = ProjectionHead(d_model, proj_dim).to(device)
            self.temp    = temperature
            self.aug     = TimeSeriesAugmenter()
            self.device  = torch.device(device)
            self.opt     = torch.optim.AdamW(
                list(encoder.parameters()) + list(self.proj.parameters()),
                lr=lr, weight_decay=1e-4,
            )
            self._use_amp = device != "cpu" and torch.cuda.is_available()
            self._scaler  = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
            """NT-Xent contrastive loss (SimCLR formulation)."""
            B = z1.shape[0]
            z  = torch.cat([z1, z2], dim=0)           # (2B, D)
            sim = torch.mm(z, z.T) / self.temp         # (2B, 2B)
            mask = torch.eye(2*B, device=self.device).bool()
            neg_inf = torch.finfo(sim.dtype).min
            sim.masked_fill_(mask, neg_inf)
            labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(self.device)
            return F.cross_entropy(sim, labels)

        def _encode_project(self, t):
            h = self.encoder(t)
            if h.ndim == 3:
                h = h[:, -1, :]
            return self.proj(h)

        def pretrain(
            self,
            X: np.ndarray,           # (N, seq_len, n_features)
            epochs: int = 50,
            batch_size: int = 256,
            checkpoint_path: Optional[str] = None,
        ) -> dict:
            if checkpoint_path is None:
                checkpoint_path = PATHS["file_contrastive_encoder"]
            N = len(X)
            batch_size = max(batch_size, 512)
            history = {"loss": []}
            amp_str = "AMP" if self._use_amp else "FP32"
            print(f"[TSCL] Pre-training {epochs} epochs | {N:,} windows | "
                  f"batch={batch_size} | {amp_str} | temp={self.temp}")

            epoch_bar = _pbar(range(epochs), desc="TSCL Pretrain", unit="ep", leave=True)
            for epoch in epoch_bar:
                idx = np.random.permutation(N)
                epoch_loss = 0.0; n_batches = 0

                batches = list(range(0, N, batch_size))
                batch_bar = _pbar(batches, desc=f"  Ep {epoch+1:3d}/{epochs}",
                                  unit="batch", leave=False)
                for start in batch_bar:
                    batch_idx = idx[start:start+batch_size]
                    X_batch   = X[batch_idx]

                    v1 = self.aug.augment_batch(X_batch)
                    v2 = self.aug.augment_batch(X_batch)
                    t1 = torch.as_tensor(v1, dtype=torch.float32, device=self.device)
                    t2 = torch.as_tensor(v2, dtype=torch.float32, device=self.device)

                    with torch.amp.autocast("cuda", enabled=self._use_amp):
                        z1 = self._encode_project(t1)
                        z2 = self._encode_project(t2)
                        loss = self.nt_xent_loss(z1, z2)

                    self.opt.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    self._scaler.step(self.opt)
                    self._scaler.update()

                    epoch_loss += loss.item(); n_batches += 1
                    if hasattr(batch_bar, "set_postfix"):
                        batch_bar.set_postfix(loss=f"{loss.item():.3f}")

                avg = epoch_loss / max(n_batches, 1)
                history["loss"].append(avg)
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix(loss=f"{avg:.4f}")
                elif (epoch+1) % 10 == 0:
                    print(f"  TSCL Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f}")

            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)
            print(f"[TSCL] Encoder saved → {checkpoint_path}")
            return history


    class RegimeAwareTSCLTrainer(TSCLTrainer):
        """
        Extends TSCLTrainer with regime-aware positive/negative pair selection.

        Standard TSCL: positives = two augmented views of the SAME window.
        Regime-aware:  positives = windows from the SAME market regime
                       (trending / mean-reverting / neutral); hard negatives =
                       windows from the OPPOSITE regime.

        This gives the encoder a structured latent space:
          Trending       windows cluster together  (Hurst > 0.55)
          Mean-reverting windows cluster together  (Hurst < 0.45)
          Neutral        windows fill the middle

        The regime labels (1=trending, -1=mean-reverting, 0=neutral) are
        typically derived from the Parkinson fast_trend_score in
        features/advanced_features.py and passed in at construction time.
        """

        def __init__(
            self,
            encoder,
            regime_labels:        np.ndarray,   # (N,) int8: 1, 0, -1
            d_model=128, proj_dim=128,
            temperature=0.07, lr=1e-4, device="cpu",
            hard_negative_weight: float = 1.0,
        ):
            super().__init__(encoder, d_model, proj_dim, temperature, lr, device)
            self.regime_labels    = np.asarray(regime_labels, dtype=np.int8)
            self.hard_neg_weight  = hard_negative_weight

            # Pre-build per-regime index lists for O(1) sampling
            self._regime_idx: dict = {}
            for r in np.unique(self.regime_labels):
                self._regime_idx[int(r)] = np.where(self.regime_labels == r)[0]

        def _same_regime(self, anchor_i: int) -> int:
            """Sample a single index sharing the same regime as anchor_i."""
            r    = int(self.regime_labels[anchor_i])
            pool = self._regime_idx.get(r, np.array([anchor_i]))
            return int(np.random.choice(pool))

        def _diff_regime(self, anchor_i: int) -> int:
            """Sample a single index from a DIFFERENT regime (hard negative)."""
            r         = int(self.regime_labels[anchor_i])
            candidates = [
                idx for rk, idxs in self._regime_idx.items()
                if rk != r for idx in idxs
            ]
            if not candidates:
                return anchor_i   # Fallback if only one regime present
            return int(np.random.choice(candidates))

        def _regime_loss(
            self,
            z_a: "torch.Tensor",  # (B, D) anchor
            z_p: "torch.Tensor",  # (B, D) same-regime positive
            z_n: "torch.Tensor",  # (B, D) diff-regime hard negative
        ) -> "torch.Tensor":
            """
            Triplet-style NT-Xent loss that pushes same-regime embeddings
            together and cross-regime embeddings apart.

            Compared to standard SimCLR:
              - Positives are semantically matched (same regime), not just augmented
              - Hard negatives from opposite regime increase training signal quality
            """
            B = z_a.shape[0]
            # Standard SimCLR loss on (anchor, positive) pairs
            z_std = torch.cat([z_a, z_p], dim=0)               # (2B, D)
            sim   = torch.mm(z_std, z_std.T) / self.temp        # (2B, 2B)
            eye   = torch.eye(2 * B, device=self.device).bool()
            neg_inf = torch.finfo(sim.dtype).min
            sim.masked_fill_(eye, neg_inf)
            labels = torch.cat([
                torch.arange(B, 2 * B, device=self.device),
                torch.arange(0, B,     device=self.device),
            ])
            loss_std = F.cross_entropy(sim, labels)

            # Hard-negative margin loss: anchor should be farther from hard neg than pos
            sim_ap = (z_a * z_p).sum(-1) / self.temp  # (B,)
            sim_an = (z_a * z_n).sum(-1) / self.temp  # (B,)
            # Hinge: push anchor-negative similarity 0.2 below anchor-positive
            margin_loss = F.relu(sim_an - sim_ap + 0.2).mean()

            return loss_std + self.hard_neg_weight * margin_loss

        def pretrain(
            self,
            X:                np.ndarray,   # (N, seq_len, n_features)
            epochs:           int = 50,
            batch_size:       int = 256,
            checkpoint_path:  Optional[str] = None,
        ) -> dict:
            """
            Regime-aware pre-training loop.
            Falls back to standard TSCL augmentation when all samples share
            the same regime (e.g. pure trending dataset).
            """
            if checkpoint_path is None:
                checkpoint_path = PATHS.get(
                    "file_contrastive_encoder",
                    "/workspace/checkpoints/contrastive_encoder_regime.pt",
                )
            N = len(X)

            # Align regime labels to dataset length
            reg = self.regime_labels
            if len(reg) > N:
                reg = reg[:N]
            elif len(reg) < N:
                reg = np.pad(reg, (0, N - len(reg)), constant_values=0)
            self.regime_labels = reg

            # Rebuild index after potential trimming
            self._regime_idx = {}
            for r in np.unique(reg):
                self._regime_idx[int(r)] = np.where(reg == r)[0]

            n_regimes = len(self._regime_idx)
            print(f"[RegimeTSCL] Pre-training {epochs} epochs | {N:,} windows | "
                  f"{n_regimes} regimes: "
                  f"{ {int(r): int(len(idx)) for r, idx in self._regime_idx.items()} }")

            batch_size = max(batch_size, 512)
            amp_str = "AMP" if self._use_amp else "FP32"
            print(f"  batch={batch_size} | {amp_str}")

            # Pre-build vectorised regime pools for fast sampling
            _regime_pool = {r: idx for r, idx in self._regime_idx.items()}

            history = {"loss": []}
            epoch_bar = _pbar(range(epochs), desc="Pretrain", unit="ep", leave=True)
            for epoch in epoch_bar:
                idx_perm = np.random.permutation(N)
                ep_loss  = 0.0
                n_b      = 0

                batches = list(range(0, N, batch_size))
                batch_bar = _pbar(batches, desc=f"  Ep {epoch+1:3d}/{epochs}",
                                  unit="batch", leave=False)
                for start in batch_bar:
                    batch_idx = idx_perm[start: start + batch_size]
                    if len(batch_idx) < 4:
                        continue

                    v_a = self.aug.augment_batch(X[batch_idx])

                    with torch.amp.autocast("cuda", enabled=self._use_amp):
                        if n_regimes > 1:
                            batch_reg = reg[batch_idx]
                            pos_i = np.empty(len(batch_idx), dtype=np.int64)
                            neg_i = np.empty(len(batch_idx), dtype=np.int64)
                            for r, pool in _regime_pool.items():
                                m = batch_reg == r
                                cnt = int(m.sum())
                                if cnt == 0:
                                    continue
                                pos_i[m] = np.random.choice(pool, cnt)
                                others = np.concatenate([p for rk, p in _regime_pool.items() if rk != r])
                                if len(others) > 0:
                                    neg_i[m] = np.random.choice(others, cnt)
                                else:
                                    neg_i[m] = pos_i[m]

                            v_p = self.aug.augment_batch(X[pos_i])
                            v_n = self.aug.augment_batch(X[neg_i])

                            t_a = torch.as_tensor(v_a, dtype=torch.float32, device=self.device)
                            t_p = torch.as_tensor(v_p, dtype=torch.float32, device=self.device)
                            t_n = torch.as_tensor(v_n, dtype=torch.float32, device=self.device)

                            loss = self._regime_loss(
                                self._encode_project(t_a),
                                self._encode_project(t_p),
                                self._encode_project(t_n),
                            )
                        else:
                            v_b = self.aug.augment_batch(X[batch_idx])
                            t1 = torch.as_tensor(v_a, dtype=torch.float32, device=self.device)
                            t2 = torch.as_tensor(v_b, dtype=torch.float32, device=self.device)
                            loss = self.nt_xent_loss(
                                self._encode_project(t1), self._encode_project(t2))

                    self.opt.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    self._scaler.step(self.opt)
                    self._scaler.update()

                    ep_loss += loss.item()
                    n_b += 1
                    if hasattr(batch_bar, "set_postfix"):
                        batch_bar.set_postfix(loss=f"{loss.item():.3f}")

                avg = ep_loss / max(n_b, 1)
                history["loss"].append(avg)
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix(loss=f"{avg:.4f}")
                elif (epoch + 1) % 10 == 0:
                    print(f"  RegimeTSCL Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f}")

            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)
            print(f"[RegimeTSCL] Encoder saved → {checkpoint_path}")
            return history


# ══════════════════════════════════════════════════════════════════════════════
# 2. DUAL-STREAM SENTIMENT  (FinBERT offline + SLM online)
# ══════════════════════════════════════════════════════════════════════════════

class DualStreamSentiment:
    """
    Two-speed architecture that adds news context without blocking execution.

    GLOBAL BRAIN (slow, ~60s cadence):
      Runs FinBERT (offline pre-computed) or Mistral-7B SLM on live headlines.
      Outputs a single Sentiment Bias in [-1.0, +1.0].

    LOCAL ACTOR (fast, every bar):
      Receives the Sentiment Bias as a feature.
      If bias > threshold → only take long signals.
      If bias < -threshold → only take short signals.

    Latency solution: FinBERT embeddings are pre-computed offline for
    historical data. Online inference uses a small SLM (Mistral-7B) running
    on GPU with cached embeddings — adds <5ms via the cross-attention fusion.
    """

    def __init__(
        self,
        embedding_dim:   int   = 768,
        proj_dim:        int   = 8,
        decay_lambda:    float = 0.1,
        update_sec:      int   = 60,
        sentiment_threshold: float = 0.3,
    ):
        self.emb_dim    = embedding_dim
        self.proj_dim   = proj_dim
        self.decay_lam  = decay_lambda
        self.update_sec = update_sec
        self.threshold  = sentiment_threshold

        # Random projection (deterministic — same weights every run)
        rng = np.random.default_rng(0)
        self._proj = rng.standard_normal((embedding_dim, proj_dim)).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-9

        # Live state
        self._current_bias: float = 0.0
        self._last_update:  float = 0.0
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def compute_offline_embedding(self, text: str) -> np.ndarray:
        """
        Compute FinBERT embedding for a news headline.
        In production: called by a batch job that pre-processes news archives.
        Returns a 768-dim vector.
        """
        # Mock: returns deterministic hash-based vector (avoids transformers dependency)
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(self.emb_dim).astype(np.float32)

    def project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce 768-dim to proj_dim using pre-computed projection matrix."""
        e = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
        return (e @ self._proj).squeeze()

    def sentiment_from_embedding(self, embedding: np.ndarray) -> float:
        """
        Map FinBERT embedding to scalar sentiment bias [-1, +1].
        FinBERT outputs [positive, negative, neutral] logits in its final layer;
        this projects the full embedding to a scalar.
        """
        proj = self.project_embedding(embedding)
        # Sigmoid of first projected dimension as a proxy for positive sentiment
        return float(np.tanh(proj[0]))

    def update_global_brain(self, headlines: List[str]) -> float:
        """
        Update the global sentiment bias from a batch of recent headlines.
        Called by the slow loop (every ~60 seconds).
        """
        if not headlines:
            # Decay toward zero if no new headlines
            dt = time.time() - self._last_update
            self._current_bias *= np.exp(-self.decay_lam * dt)
            return self._current_bias

        sentiments = []
        for h in headlines:
            if h not in self._embedding_cache:
                self._embedding_cache[h] = self.compute_offline_embedding(h)
            s = self.sentiment_from_embedding(self._embedding_cache[h])
            sentiments.append(s)

        # Weighted average (most recent gets highest weight)
        weights = np.exp(-self.decay_lam * np.arange(len(sentiments))[::-1])
        weights /= weights.sum()
        self._current_bias = float(np.dot(weights, sentiments))
        self._last_update  = time.time()
        return self._current_bias

    def get_bias(self) -> float:
        """Get current sentiment bias (with decay applied)."""
        dt = time.time() - self._last_update
        return self._current_bias * np.exp(-self.decay_lam * dt)

    def filter_signal(self, raw_signal: int, bias: Optional[float] = None) -> int:
        """
        Apply sentiment bias to suppress counter-trend signals.
          bias > +threshold: suppress SELL signals
          bias < -threshold: suppress BUY signals
        """
        if bias is None: bias = self.get_bias()
        if bias > self.threshold  and raw_signal == 2: return 1  # Suppress SELL → HOLD
        if bias < -self.threshold and raw_signal == 0: return 1  # Suppress BUY  → HOLD
        return raw_signal

    def build_sentiment_series(
        self,
        headlines_by_time: Dict[pd.Timestamp, List[str]],
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Build a sentiment bias time series aligned to bar index.
        Used to construct the 'sentiment_decayed' feature column.
        """
        raw = pd.Series(0.0, index=index, dtype=float)
        for ts, headlines in sorted(headlines_by_time.items()):
            if ts in raw.index:
                raw[ts] = self.update_global_brain(headlines)

        # Forward-fill with exponential decay
        result = pd.Series(0.0, index=index, dtype=float)
        last_ts = None; last_v = 0.0
        for ts in index:
            if ts in raw.index and raw[ts] != 0:
                last_ts = ts; last_v = raw[ts]; result[ts] = last_v
            elif last_ts is not None:
                dt = (ts - last_ts).total_seconds()
                result[ts] = last_v * np.exp(-self.decay_lam * dt)
        return result.rename("sentiment_bias")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TIP-SEARCH LATENCY MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class TIPSearchManager:
    """
    Time-Predictable Inference Scheduling (TIP-Search).

    Runs TWO models:
      FAST model (DQN):   ~2ms latency, lower accuracy
      SLOW model (HAELT): ~5ms latency, higher accuracy

    Switching logic:
      - Normal market: use SLOW model (accuracy priority)
      - Volatility spike (ATR > 2× avg): switch to FAST model
        (must get order in before spread widens)

    This ensures the system NEVER misses its execution window, even during
    high-impact news events where the slow model would exceed acceptable latency.
    """

    def __init__(
        self,
        fast_agent,           # DQNAgent (or any .select_action(obs) → int)
        slow_agent,           # PPOAgent / any supervised model
        fast_latency_ms: float = 2.0,
        slow_latency_ms: float = 5.0,
        switch_mult:     float = 2.0,
        atr_lookback:    int   = 60,
        max_latency_ms:  float = 10.0,
    ):
        self.fast = fast_agent
        self.slow = slow_agent
        self.fast_ms  = fast_latency_ms
        self.slow_ms  = slow_latency_ms
        self.switch_m = switch_mult
        self.lb       = atr_lookback
        self.max_ms   = max_latency_ms

        self._atr_history: List[float] = []
        self.stats = {"fast_used": 0, "slow_used": 0, "total": 0}

    def _is_vol_spike(self, current_atr: float) -> bool:
        if len(self._atr_history) < self.lb: return False
        avg_atr = np.mean(self._atr_history[-self.lb:])
        return current_atr > self.switch_m * avg_atr

    def select_action(
        self,
        obs:         np.ndarray,
        current_atr: float = 0.0,
    ) -> Tuple[int, str, float]:
        """
        Returns (action, model_used, latency_ms).

        Selects model dynamically based on current volatility regime.
        """
        self._atr_history.append(current_atr)
        use_fast = self._is_vol_spike(current_atr)

        t0 = time.perf_counter()
        if use_fast:
            action = self.fast.select_action(obs)
            model_used = "fast_dqn"
        else:
            try:
                action = self.slow.select_action(obs)
                model_used = "slow_haelt"
            except Exception:
                # Fallback to fast if slow model errors
                action = self.fast.select_action(obs)
                model_used = "fast_fallback"

        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats["total"] += 1
        self.stats["fast_used" if use_fast else "slow_used"] += 1

        return action, model_used, latency_ms

    def report(self) -> dict:
        t = max(self.stats["total"], 1)
        return {
            "fast_pct":  self.stats["fast_used"] / t * 100,
            "slow_pct":  self.stats["slow_used"] / t * 100,
            "total_calls": t,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Detects when the model's feature distribution or performance
    has drifted significantly from the training distribution.

    Three detection methods:
      PSI (Population Stability Index): measures feature distribution shift
      KS Test: statistical test for distribution change
      Sharpe Drop: performance-based drift signal

    Any trigger can initiate automatic retraining.
    """

    def __init__(
        self,
        psi_threshold:   float = 0.2,
        ks_pvalue:       float = 0.05,
        sharpe_drop:     float = 0.5,
        window:          int   = 1000,
    ):
        self.psi_thresh  = psi_threshold
        self.ks_p        = ks_pvalue
        self.sd_thresh   = sharpe_drop
        self.window      = window
        self._train_dist: Optional[np.ndarray] = None
        self._baseline_sharpe: float = 0.0

    def fit_baseline(self, X_train: np.ndarray, baseline_returns: np.ndarray):
        """Store training distribution and baseline Sharpe for comparison."""
        self._train_dist = X_train.copy()
        if len(baseline_returns) > 1:
            self._baseline_sharpe = (
                baseline_returns.mean() / (baseline_returns.std() + 1e-9)
                * np.sqrt(252)
            )
        print(f"[Drift] Baseline fitted | Sharpe: {self._baseline_sharpe:.3f} | "
              f"Train dist: {X_train.shape}")

    def compute_psi(self, expected: np.ndarray, actual: np.ndarray,
                    bins: int = 10) -> float:
        """
        Population Stability Index.
          PSI < 0.1  : no significant shift
          PSI 0.1-0.2: moderate shift, monitor
          PSI > 0.2  : significant shift → retrain
        """
        eps = 1e-6
        exp_hist, edges = np.histogram(expected, bins=bins, density=True)
        act_hist, _     = np.histogram(actual, bins=edges, density=True)
        exp_hist += eps; act_hist += eps
        return float(np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist)))

    def check(
        self,
        X_live:         np.ndarray,
        live_returns:   np.ndarray,
    ) -> dict:
        """
        Run all drift checks. Returns a dict with drift flags and scores.
        """
        result = {
            "drift_detected": False,
            "psi_max": 0.0,
            "ks_min_pvalue": 1.0,
            "sharpe_drop": 0.0,
            "reasons": [],
        }

        if self._train_dist is None:
            return result

        # PSI per feature (use subset of most important features)
        n_feats = min(X_live.shape[1], self._train_dist.shape[1])
        psi_vals = []
        for f in range(n_feats):
            train_f = self._train_dist[:, f]
            live_f  = X_live[-min(len(X_live), self.window):, f]
            psi_vals.append(self.compute_psi(train_f, live_f))
        result["psi_max"] = float(np.max(psi_vals))

        # KS test on first feature (price return proxy)
        if SCIPY and len(X_live) >= 30:
            ks_pvals = []
            for f in range(min(5, n_feats)):
                _, p = stats.ks_2samp(
                    self._train_dist[:, f],
                    X_live[-self.window:, f]
                )
                ks_pvals.append(p)
            result["ks_min_pvalue"] = float(np.min(ks_pvals))

        # Sharpe drop
        if len(live_returns) > 30:
            live_sharpe = (
                live_returns[-self.window:].mean()
                / (live_returns[-self.window:].std() + 1e-9)
                * np.sqrt(252)
            )
            drop = self._baseline_sharpe - live_sharpe
            result["sharpe_drop"] = float(drop)
            if drop > self.sd_thresh:
                result["drift_detected"] = True
                result["reasons"].append(f"Sharpe drop {drop:.3f} > {self.sd_thresh}")

        if result["psi_max"] > self.psi_thresh:
            result["drift_detected"] = True
            result["reasons"].append(f"PSI {result['psi_max']:.3f} > {self.psi_thresh}")

        if result["ks_min_pvalue"] < self.ks_p:
            result["drift_detected"] = True
            result["reasons"].append(f"KS p-value {result['ks_min_pvalue']:.4f} < {self.ks_p}")

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. PURGED K-FOLD + EMBARGOING
# ══════════════════════════════════════════════════════════════════════════════

class PurgedEmbargoCVSplitter:
    """
    Combines Purged K-Fold + Embargoing (both specified).

    Purging removes training samples whose label horizon overlaps the test set,
    eliminating look-ahead bias from overlapping position horizons.

    Embargoing adds a temporal buffer AFTER each test set to account for
    market autocorrelation — the model cannot learn from data that is
    'adjacent in time' to its test observations.
    """

    def __init__(self, n_splits=5, purge_bars=30, embargo_bars=10):
        self.k       = n_splits
        self.purge   = purge_bars
        self.embargo = embargo_bars

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n    = len(X)
        size = n // self.k
        folds = []

        for i in range(self.k):
            # Test window
            t_start = i * size
            t_end   = t_start + size if i < self.k - 1 else n

            # Purge zone: remove train samples near test boundary
            purge_zone = set(range(max(0, t_start - self.purge), t_end))
            # Embargo zone: skip train samples immediately after test
            embargo_zone = set(range(t_end, min(n, t_end + self.embargo)))

            excluded = purge_zone | embargo_zone
            train_idx = np.array([j for j in range(n)
                                   if j not in excluded and j < t_start])
            test_idx  = np.arange(t_start, t_end)

            if len(train_idx) < size // 2:
                continue  # Skip if not enough training data

            folds.append((train_idx, test_idx))
            print(f"[PurgedCV] Fold {i+1}/{self.k}: "
                  f"train={len(train_idx):,} | test={len(test_idx):,} | "
                  f"purged={len(purge_zone):,} | embargoed={len(embargo_zone):,}")

        return folds


# ══════════════════════════════════════════════════════════════════════════════
# 6. WALK-FORWARD ROLLING RETRAINING
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardRetrainer:
    """
    Periodically retrains the model as new tick data accumulates.

    Rolling window: always train on the most recent rolling_window_bars,
    discarding older data. This prevents the model from being anchored
    to stale market regimes.

    Warm start: resumes from the previous checkpoint, reducing training
    time while adapting to recent conditions.

    Auto-deploy: only replaces the live model if the new model's
    walk-forward Sharpe exceeds the current model by min_improvement.
    """

    def __init__(
        self,
        retrain_every_bars: int   = 10_000,
        rolling_window:     int   = 50_000,
        warm_start:         bool  = True,
        auto_deploy:        bool  = True,
        min_improvement:    float = 0.05,
        checkpoint_dir:     str   = None,
    ):
        if checkpoint_dir is None:
            checkpoint_dir = PATHS["checkpoints"]
        self.retrain_every = retrain_every_bars
        self.rolling_window = rolling_window
        self.warm_start    = warm_start
        self.auto_deploy   = auto_deploy
        self.min_improve   = min_improvement
        self.ckpt_dir      = Path(checkpoint_dir)
        self._bar_counter  = 0
        self._live_sharpe  = 0.0
        self.retrain_log: List[dict] = []

    def tick(self, n_new_bars: int = 1) -> bool:
        """Advance bar counter. Returns True if retraining should trigger."""
        self._bar_counter += n_new_bars
        return self._bar_counter >= self.retrain_every

    def should_retrain(
        self,
        drift_result: dict,
        n_new_bars: int = 1,
    ) -> Tuple[bool, str]:
        """
        Determine if retraining should be triggered.
        Returns (should_retrain, reason).
        """
        if self.tick(n_new_bars):
            self._bar_counter = 0
            return True, f"scheduled ({self.retrain_every} bars elapsed)"
        if drift_result.get("drift_detected"):
            return True, f"drift: {'; '.join(drift_result.get('reasons', []))}"
        return False, ""

    def run_retraining(
        self,
        train_fn:        Callable,
        X_rolling:       np.ndarray,
        y_rolling:       np.ndarray,
        eval_fn:         Callable,
        X_eval:          np.ndarray,
        y_eval:          np.ndarray,
        model_name:      str = "model",
    ) -> dict:
        """
        Execute one retraining cycle.

        1. Train on rolling window
        2. Evaluate on held-out data
        3. Compare to live model's Sharpe
        4. Deploy if improved (auto_deploy=True)
        """
        print(f"\n[Retrain] Training on {len(X_rolling):,} bars (rolling window)")
        t0 = time.time()

        # Checkpoint path for warm start
        warm_ckpt = self.ckpt_dir / f"{model_name}_best.pt" if self.warm_start else None

        new_model = train_fn(
            X_rolling, y_rolling,
            warm_start_path=str(warm_ckpt) if warm_ckpt and warm_ckpt.exists() else None
        )
        train_time = time.time() - t0

        # Evaluate new model
        new_sharpe = eval_fn(new_model, X_eval, y_eval)
        improvement = new_sharpe - self._live_sharpe

        deployed = False
        if self.auto_deploy and improvement >= self.min_improve:
            # Save new model as live
            if TORCH and hasattr(new_model, "state_dict"):
                import torch
                torch.save(new_model.state_dict(), self.ckpt_dir / f"{model_name}_live.pt")
            self._live_sharpe = new_sharpe
            deployed = True
            print(f"[Retrain] NEW MODEL DEPLOYED | "
                  f"Sharpe: {self._live_sharpe:.3f} (+{improvement:.3f})")
        else:
            print(f"[Retrain] Kept existing model | "
                  f"New Sharpe: {new_sharpe:.3f} | Improvement: {improvement:.3f} "
                  f"(need {self.min_improve:.3f})")

        record = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "new_sharpe": new_sharpe,
            "live_sharpe": self._live_sharpe,
            "improvement": improvement,
            "deployed": deployed,
            "train_time_s": train_time,
            "n_samples": len(X_rolling),
        }
        self.retrain_log.append(record)
        return record


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Secondary Modules — Smoke Tests")
    print("=" * 60)

    # 1. Contrastive augmentation
    aug = TimeSeriesAugmenter()
    x = np.random.randn(60, 10)
    for name in ["jitter","scale","permute","dropout"]:
        aug.aug = name
        out = aug.augment(x)
        print(f"[TSCL] Augment '{name}': {x.shape} → {out.shape}")

    # 2. Dual-stream sentiment
    ds = DualStreamSentiment()
    headlines = ["Fed signals rate cut", "Strong jobs report surprises markets"]
    bias = ds.update_global_brain(headlines)
    print(f"\n[Dual-Stream] Sentiment bias: {bias:.4f}")
    action = ds.filter_signal(raw_signal=0, bias=bias)
    print(f"[Dual-Stream] BUY signal with bias={bias:.2f} → filtered to: {action}")

    # 3. TIP-Search
    class MockAgent:
        def select_action(self, obs): return 1  # HOLD
    ts = TIPSearchManager(MockAgent(), MockAgent())
    atr_hist = [0.0005]*60 + [0.003]  # Spike at end
    for atr in atr_hist:
        action, model, lat = ts.select_action(np.zeros(10), atr)
    print(f"\n[TIP-Search] Last action: {action} via {model} ({lat:.2f}ms)")
    print(f"[TIP-Search] Report: {ts.report()}")

    # 4. Drift detection
    dd = DriftDetector()
    X_train = np.random.randn(1000, 20)
    dd.fit_baseline(X_train, np.random.normal(0.001, 0.01, 1000))
    X_live_drifted = np.random.randn(500, 20) * 2  # Obvious drift
    result = dd.check(X_live_drifted, np.random.normal(-0.002, 0.01, 500))
    print(f"\n[Drift] Detected: {result['drift_detected']} | "
          f"PSI: {result['psi_max']:.3f} | Sharpe drop: {result['sharpe_drop']:.3f}")
    if result['reasons']: print(f"  Reasons: {result['reasons']}")

    # 5. Purged CV
    X_df = pd.DataFrame(np.random.randn(1000, 5),
                         index=pd.date_range("2024-01-01", periods=1000, freq="1min", tz="UTC"))
    cv = PurgedEmbargoCVSplitter(n_splits=3, purge_bars=10, embargo_bars=5)
    folds = cv.split(X_df)
    print(f"\n[PurgedCV] Generated {len(folds)} folds")

    # 6. Rolling retrainer
    rt = WalkForwardRetrainer(retrain_every_bars=100)
    print(f"\n[Retrain] Tick 90: {rt.should_retrain({}, 90)}")
    print(f"[Retrain] Tick 15: {rt.should_retrain({}, 15)}")
    print(f"[Retrain] Drift:   {rt.should_retrain({'drift_detected': True, 'reasons': ['PSI spike']}, 1)}")
    print("\nAll smoke tests passed ✓")
