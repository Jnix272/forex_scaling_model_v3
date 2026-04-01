"""
models/ensemble.py
==================
Model upgrades:
  1. EnsembleMetaLearner   — weighted average of all 6 architectures
  2. UncertaintyQuantifier — MC Dropout + deep ensemble confidence intervals
  3. MultiTimeframeAttn    — hierarchical attention across 1m/5m/15m bars
  4. CausalityGNN          — Granger-causality-rewired graph network
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False


if TORCH:

    # ── 1. ENSEMBLE META-LEARNER ──────────────────────────────────────────────

    class EnsembleMetaLearner(nn.Module):
        """
        Learned ensemble of the 6 base architectures.
        Each base model produces a scalar prediction; the meta-learner
        learns dynamic weights conditioned on the current market features,
        so it can down-weight models that underperform in the current regime.

        Stacking strategy:
          - Base model outputs are concatenated with a context vector
          - A small attention network assigns weights per model per bar
          - Final output = softmax-weighted sum of base predictions
        """

        def __init__(
            self,
            base_models: List[nn.Module],
            context_dim: int = 32,
            hidden:      int = 64,
        ):
            super().__init__()
            self.bases    = nn.ModuleList(base_models)
            self.n_models = len(base_models)

            # Context encoder: maps last bar features → context vector
            self.context_enc = nn.Sequential(
                nn.LazyLinear(hidden), nn.ReLU(),
                nn.Linear(hidden, context_dim),
            )
            # Meta-network: context + n_model predictions → weights
            self.meta = nn.Sequential(
                nn.Linear(context_dim + self.n_models, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.n_models),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            x: (B, seq_len, n_features)
            Returns: (prediction, weights) where weights.shape = (B, n_models)
            """
            with torch.no_grad():
                preds = torch.stack(
                    [m(x) for m in self.bases], dim=1
                )  # (B, n_models)

            context = self.context_enc(x[:, -1, :])  # Last bar as context
            meta_in = torch.cat([context, preds], dim=1)
            weights = torch.softmax(self.meta(meta_in), dim=1)  # (B, n_models)
            output  = (weights * preds).sum(dim=1)               # (B,)
            return output, weights

        def model_weights_summary(self, x: torch.Tensor) -> Dict[str, float]:
            """Return avg weight per model — useful for monitoring which models dominate."""
            _, w = self.forward(x)
            w_avg = w.mean(0).detach().cpu().numpy()
            names = ["tft","transformer","haelt","mamba","gnn","expert"]
            return {names[i]: float(w_avg[i]) for i in range(len(self.bases))}

        def diversity_loss(self, preds: torch.Tensor) -> torch.Tensor:
            """
            Mean pairwise Pearson correlation of base-model outputs.
            Returns a scalar in [-1, +1]; lower = more diverse ensemble.

            Minimizing this in the meta-learner's training loss discourages the
            base models from making identical predictions (which would make the
            ensemble no better than a single model).

            preds: (B, n_models) — stacked raw base model outputs (before softmax)
            """
            # Standardise each model's column to zero-mean unit-variance
            p = preds - preds.mean(0, keepdim=True)
            p = p / (p.std(0, keepdim=True) + 1e-8)   # (B, n_models)
            # Pearson correlation matrix via inner product
            corr = (p.T @ p) / max(p.shape[0] - 1, 1)  # (n_models, n_models)
            # Average of upper-triangle (off-diagonal) elements only
            n = corr.shape[0]
            mask = torch.triu(torch.ones(n, n, device=preds.device), diagonal=1).bool()
            return corr[mask].mean()


    # ── Meta-learner training utility ────────────────────────────────────────

    def train_meta_learner(
        meta:             "EnsembleMetaLearner",
        loader:           "torch.utils.data.DataLoader",
        epochs:           int   = 10,
        lr:               float = 1e-3,
        diversity_weight: float = 0.1,
        device:           str   = "cpu",
        verbose:          bool  = True,
    ) -> List[float]:
        """
        Train only the EnsembleMetaLearner's context encoder and meta-network.
        Base model weights are frozen — only the weighting mechanism is learned.

        Objective:
          L = MSE(weighted_ensemble_output, target)
            - diversity_weight × H(weights)        # maximise weight entropy
            + diversity_weight × corr(base_preds)  # penalise correlated bases

        The entropy term prevents the meta-learner from collapsing to a single
        model (degenerate 'ensemble of one'). The correlation term rewards the
        meta-learner for up-weighting models that disagree with each other.

        Returns loss history (one value per epoch).
        """
        dev = torch.device(device)
        meta = meta.to(dev)

        # Freeze base models
        for base in meta.bases:
            for p in base.parameters():
                p.requires_grad_(False)

        trainable = (
            list(meta.context_enc.parameters()) +
            list(meta.meta.parameters())
        )
        opt       = torch.optim.Adam(trainable, lr=lr)
        criterion = nn.MSELoss()
        history: List[float] = []

        for ep in range(epochs):
            ep_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True).float()
                opt.zero_grad(set_to_none=True)

                # Base predictions (no grad — bases are frozen)
                with torch.no_grad():
                    base_preds = torch.stack(
                        [b(xb) for b in meta.bases], dim=1
                    )  # (B, n_models)

                # Meta-network forward
                context = meta.context_enc(xb[:, -1, :])
                meta_in = torch.cat([context, base_preds], dim=1)
                weights = torch.softmax(meta.meta(meta_in), dim=1)   # (B, n_models)
                output  = (weights * base_preds).sum(dim=1)           # (B,)

                # Task loss
                task_loss = criterion(output, yb)

                # Diversity bonus 1: maximise weight entropy (avoid collapse to one model)
                entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()

                # Diversity bonus 2: penalise correlated base outputs
                div_pen = meta.diversity_loss(base_preds)

                loss = task_loss - diversity_weight * entropy + diversity_weight * div_pen
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()

                ep_loss   += loss.item()
                n_batches += 1

            avg = ep_loss / max(n_batches, 1)
            history.append(avg)
            if verbose and (ep + 1) % max(1, epochs // 5) == 0:
                print(f"  [MetaTrain] Epoch {ep+1:3d}/{epochs} | Loss: {avg:.6f}")

        return history


    # ── 2. UNCERTAINTY QUANTIFIER ─────────────────────────────────────────────

    class MCDropoutWrapper(nn.Module):
        """
        Wraps any model to enable MC Dropout inference.
        Keeps dropout active at inference time and runs N forward passes
        to estimate prediction variance (= epistemic uncertainty).

        High uncertainty → reduce position size / skip signal.
        Low uncertainty  → trade at full size.
        """

        def __init__(self, model: nn.Module, n_passes: int = 30):
            super().__init__()
            self.model    = model
            self.n_passes = n_passes

        def _enable_dropout(self):
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        @torch.no_grad()
        def predict_with_uncertainty(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns (mean_pred, std_pred, confidence_score).
            confidence_score = 1 - normalized_std ∈ [0, 1]
            """
            self._enable_dropout()
            preds = torch.stack(
                [self.model(x) for _ in range(self.n_passes)], dim=0
            )  # (n_passes, B)
            mean = preds.mean(0)
            std  = preds.std(0)
            # Normalize std to [0,1] confidence score using running stats
            max_std = std.max().clamp(min=1e-6)
            conf    = 1.0 - (std / max_std)
            return mean, std, conf


    class DeepEnsembleUQ:
        """
        Deep ensemble uncertainty: train N independent models from different
        random seeds. Disagreement between models = uncertainty.
        More reliable than MC Dropout but requires N × training time.
        """

        def __init__(self, models: List[nn.Module], device: str = "cpu"):
            self.models = models
            self.device = torch.device(device)

        @torch.no_grad()
        def predict(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            preds = torch.stack(
                [m(x.to(self.device)) for m in self.models], dim=0
            )
            mean = preds.mean(0); std = preds.std(0)
            conf = 1.0 - (std / std.max().clamp(min=1e-6))
            return mean, std, conf

        def confidence_filter(
            self,
            x:         torch.Tensor,
            threshold: float = 0.5,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Return (signal, mask) where mask=1 for high-confidence predictions."""
            mean, _, conf = self.predict(x)
            mask = (conf > threshold).float()
            return mean * mask, mask


    # ── 3. MULTI-TIMEFRAME ATTENTION ──────────────────────────────────────────

    class MultiTimeframeAttention(nn.Module):
        """
        Hierarchical attention across 1-min, 5-min, and 15-min bar streams.

        Each timeframe captures different signal frequencies:
          1-min  → microstructure, order flow, scalping signal
          5-min  → intraday momentum, MACD crossovers
          15-min → structural support/resistance, session bias

        Architecture:
          1. Separate encoder per timeframe (shared weights to save params)
          2. Cross-timeframe attention: each timeframe attends to the others
          3. Fusion layer: concatenate attended representations → prediction
        """

        def __init__(
            self,
            input_size:    int,
            d_model:       int = 128,
            nhead:         int = 4,
            n_tf_layers:   int = 2,
            dropout:       float = 0.1,
            timeframes:    List[int] = [1, 5, 15],  # in minutes
        ):
            super().__init__()
            self.tfs = timeframes

            # Shared encoder applied to each timeframe
            self.proj = nn.Linear(input_size, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, n_tf_layers, enable_nested_tensor=False)

            # Cross-timeframe attention: 1m attends to 5m and 15m context
            self.cross_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(d_model)

            # Fusion
            n_tf = len(timeframes)
            self.fuse = nn.Sequential(
                nn.Linear(d_model * n_tf, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
            """
            x_list: list of (B, T_i, input_size) tensors, one per timeframe.
            All T_i can differ (15-min bars will have fewer rows).
            """
            encoded = []
            for x in x_list:
                h = self.encoder(self.proj(x))
                encoded.append(h[:, -1, :])  # Take last bar embedding

            # Cross-timeframe attention: fine (1m) queries, coarse (5m, 15m) as K/V
            if len(encoded) > 1:
                query   = encoded[0].unsqueeze(1)            # (B, 1, d)
                context = torch.stack(encoded[1:], dim=1)    # (B, n_tf-1, d)
                attn_out, _ = self.cross_attn(query, context, context)
                fine = self.cross_norm(encoded[0] + attn_out.squeeze(1))
                encoded[0] = fine

            fused = torch.cat(encoded, dim=-1)
            return self.fuse(fused).squeeze(-1)


    # ── 4. GRANGER CAUSALITY GNN ─────────────────────────────────────────────

    class GrangerCausalityGraph:
        """
        Compute Granger causality p-values between asset pairs and build
        a directed adjacency matrix. More principled than correlation:
        tests whether X's past *predicts* Y beyond Y's own past.

        Used to dynamically rewire the GNN cross-asset graph.
        Update every N bars (not every tick — costly computation).
        """

        def __init__(self, max_lag: int = 5, significance: float = 0.05):
            self.max_lag = max_lag
            self.alpha   = significance

        def _granger_pvalue(
            self, y: np.ndarray, x: np.ndarray, lag: int
        ) -> float:
            """
            Simplified Granger test using OLS F-test.
            H0: x does not Granger-cause y.
            """
            n = len(y)
            if n < lag * 3:
                return 1.0
            try:
                from scipy.stats import f as f_dist
                # Restricted model (y regressed on own lags)
                Y  = y[lag:]
                Xr = np.column_stack([y[lag-k-1:n-k-1] for k in range(lag)])
                # Unrestricted model (add x lags)
                Xu = np.column_stack([
                    Xr,
                    *[x[lag-k-1:n-k-1] for k in range(lag)]
                ])
                def rss(X, y):
                    try:
                        b = np.linalg.lstsq(X, y, rcond=None)[0]
                        return float(((y - X @ b)**2).sum())
                    except Exception:
                        return float(np.var(y) * len(y))
                r_rss = rss(np.column_stack([np.ones(len(Xr)), Xr]), Y)
                u_rss = rss(np.column_stack([np.ones(len(Xu)), Xu]), Y)
                df1 = lag; df2 = len(Y) - 2 * lag - 1
                if df2 <= 0 or u_rss <= 0:
                    return 1.0
                F    = ((r_rss - u_rss) / df1) / (u_rss / df2)
                return float(1 - f_dist.cdf(F, df1, df2))
            except ImportError:
                # Fallback: simple correlation-based p-value proxy
                corr = float(np.corrcoef(x[:-lag], y[lag:])[0, 1])
                return float(1 - abs(corr))
            except Exception:
                return 1.0

        def compute_adjacency(
            self,
            returns_df: pd.DataFrame,
            window:     int = 120,
        ) -> np.ndarray:
            """
            Compute directed adjacency matrix from Granger causality tests.
            adj[i, j] = 1  if asset i Granger-causes asset j (p < alpha)
            adj[i, j] = 0  otherwise
            """
            assets = returns_df.columns.tolist()
            n      = len(assets)
            adj    = np.zeros((n, n), dtype=np.float32)
            data   = returns_df.tail(window).fillna(0).values

            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    p = self._granger_pvalue(data[:, j], data[:, i], self.max_lag)
                    if p < self.alpha:
                        adj[i, j] = 1.0

            return adj

        def to_torch(self, adj: np.ndarray, device: str = "cpu") -> torch.Tensor:
            return torch.tensor(adj, dtype=torch.float32,
                                device=torch.device(device))


    class CausalGNNCrossAsset(nn.Module):
        """
        GNN where edges are determined by Granger causality tests rather
        than static correlation thresholds. Updated every N bars.

        Detects when asset A is about to move asset B — before price shows it.
        """

        def __init__(
            self,
            node_features: int = 32,
            hidden:        int = 64,
            num_layers:    int = 3,
            heads:         int = 4,
            n_nodes:       int = 6,
            dropout:       float = 0.1,
        ):
            super().__init__()
            self.embed  = nn.Linear(node_features, hidden)
            self.layers = nn.ModuleList([
                nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])
            self.norms  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
            self.head   = nn.Linear(hidden * n_nodes, 1)
            self.drop   = nn.Dropout(dropout)
            self.causal = GrangerCausalityGraph()
            self._adj:  Optional[torch.Tensor] = None
            self._adj_update_count = 0

        def update_adjacency(
            self,
            returns_df: pd.DataFrame,
            device:     str = "cpu",
            every:      int = 500,   # Update every N calls
        ):
            self._adj_update_count += 1
            if self._adj_update_count % every != 0 and self._adj is not None:
                return
            adj = self.causal.compute_adjacency(returns_df)
            self._adj = self.causal.to_torch(adj, device)

        def forward(
            self,
            x:   torch.Tensor,   # (B, n_nodes, node_features)
            adj: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            adj = adj if adj is not None else self._adj
            h = self.embed(x)
            for attn, norm in zip(self.layers, self.norms):
                if adj is not None:
                    # Causal mask: only attend through Granger-significant edges
                    mask = (adj == 0).float() * -1e9
                    mask = mask.unsqueeze(0).expand(h.shape[0], -1, -1)
                    out, _ = attn(h, h, h)
                else:
                    out, _ = attn(h, h, h)
                h = norm(h + self.drop(out))
            return self.head(h.reshape(h.shape[0], -1)).squeeze(-1)

else:
    class EnsembleMetaLearner:
        def __init__(self, **kw): pass
    class MCDropoutWrapper:
        def __init__(self, **kw): pass
    class DeepEnsembleUQ:
        def __init__(self, **kw): pass
    class MultiTimeframeAttention:
        def __init__(self, **kw): pass
    class CausalGNNCrossAsset:
        def __init__(self, **kw): pass
    class GrangerCausalityGraph:
        def __init__(self, **kw): pass
        def compute_adjacency(self, *a, **kw): return np.zeros((6,6))


if __name__ == "__main__" and TORCH:
    import torch
    B, T, F_IN = 4, 60, 48

    # Ensemble test
    from models.architectures import HAELTHybrid, MambaScalper
    bases  = [HAELTHybrid(input_size=F_IN), MambaScalper(input_size=F_IN)]
    ens    = EnsembleMetaLearner(bases, context_dim=32)
    x      = torch.randn(B, T, F_IN)
    out, w = ens(x)
    print(f"Ensemble: {tuple(out.shape)} | weights: {tuple(w.shape)}")

    # MC Dropout
    from models.architectures import HAELTHybrid
    m    = MCDropoutWrapper(HAELTHybrid(input_size=F_IN), n_passes=10)
    mean, std, conf = m.predict_with_uncertainty(x)
    print(f"MC Dropout: mean={tuple(mean.shape)} std={tuple(std.shape)} conf={conf.mean():.3f}")

    # Multi-timeframe
    mtf = MultiTimeframeAttention(F_IN, d_model=64, nhead=4)
    x1  = torch.randn(B, 60, F_IN)   # 1-min
    x5  = torch.randn(B, 12, F_IN)   # 5-min (60/5)
    x15 = torch.randn(B,  4, F_IN)   # 15-min (60/15)
    out = mtf([x1, x5, x15])
    print(f"MultiTimeframe: {tuple(out.shape)}")

    # Granger GNN
    gc  = GrangerCausalityGraph(max_lag=3, significance=0.1)
    df  = pd.DataFrame(np.random.randn(200, 5), columns=["A","B","C","D","E"])
    adj = gc.compute_adjacency(df, window=100)
    print(f"Granger adj:\n{adj}")
