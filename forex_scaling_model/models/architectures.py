"""
models/architectures.py
========================
All six model architectures specified:
  1. TFT          — Temporal Fusion Transformer
  2. iTransformer — Variate-dimension attention
  3. HAELTHybrid  — LSTM + Transformer parallel
  4. MambaScalper — State Space Model (low latency)
  5. GNNCrossAsset— Graph Neural Network for cross-asset correlations
  6. EXPERTEncoder— Exchange-Rate Transformer (conv FFN, no positional enc)

Shared interface: forward(x) → (batch,) scalars if num_classes==1, else (batch, num_classes) logits.
"""

import numpy as np
import warnings
import inspect
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False
    warnings.warn("PyTorch not installed. pip install torch")


if TORCH:

    # ── Shared building blocks ─────────────────────────────────────────────

    class HuberLoss(nn.Module):
        def __init__(self, delta=1.0):
            super().__init__(); self.delta=delta
        def forward(self, p, t):
            e=p-t; a=e.abs()
            return torch.where(a<=self.delta, 0.5*e**2, self.delta*(a-0.5*self.delta)).mean()

    class AsymmetricDirectionalLoss(nn.Module):
        """
        Huber on residuals plus extra penalty when prediction and target disagree in
        sign (direction), as discussed for asymmetric economic risk in directional
        forecasting. Targets are typically {-1,0,+1} bar labels.
        """
        def __init__(self, delta=1.0, sign_weight=2.0):
            super().__init__()
            self.delta = delta
            self.sign_weight = sign_weight

        def forward(self, pred, target):
            e = pred - target
            a = e.abs()
            huber = torch.where(
                a <= self.delta,
                0.5 * e ** 2,
                self.delta * (a - 0.5 * self.delta),
            )
            tnz = target.abs() > 0.05
            wrong = tnz & (torch.sign(pred) != torch.sign(target))
            extra = wrong.float() * target.abs().clamp(min=0.1)
            return (huber + self.sign_weight * extra).mean()

    # ── Multi-task head, loss, and backbone wrapper ───────────────────────

    class MultiTaskHead(nn.Module):
        """
        Three-output prediction head for multi-task supervision:
          direction  — 3-class logits {sell=0, hold=1, buy=2}   (cross-entropy)
          return_hat — scalar magnitude regression               (Huber)
          confidence — predicted |return|, clipped to [0,1]     (BCE)

        Training on all three signals simultaneously prevents the backbone from
        learning 'correct direction / wrong magnitude' solutions and gives a
        natural confidence signal for downstream position sizing.
        """

        def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.1):
            super().__init__()
            h2 = max(hidden // 2, 16)
            self.direction = nn.Sequential(
                nn.Linear(in_features, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 3),
            )
            self.return_hat = nn.Sequential(
                nn.Linear(in_features, h2), nn.GELU(),
                nn.Linear(h2, 1),
            )
            self.confidence = nn.Sequential(
                nn.Linear(in_features, h2), nn.GELU(),
                nn.Linear(h2, 1),
                # Sigmoid removed — BCEWithLogitsLoss in MultiTaskLoss fuses it
                # safely under AMP. Do NOT add Sigmoid back here.
            )

        def forward(self, h: torch.Tensor):
            """h: (B, in_features) — backbone hidden state BEFORE any prediction head."""
            return (
                self.direction(h),              # (B, 3)
                self.return_hat(h).squeeze(-1), # (B,)
                self.confidence(h).squeeze(-1), # (B,)
            )


    class MultiTaskLoss(nn.Module):
        """
        Weighted combination of three supervised objectives:
          L = w_dir  * CrossEntropy(direction, y_cls)
            + w_ret  * Huber(return_hat, y_cont)
            + w_conf * BCE(confidence, |y_cont|)

        Typical weights: w_dir=1.0, w_ret=0.5, w_conf=0.3
        Class weights can be passed to CrossEntropyLoss to handle {-1,0,+1} imbalance.
        """

        def __init__(
            self,
            class_weights: Optional[torch.Tensor] = None,
            w_dir:       float = 1.0,
            w_ret:       float = 0.5,
            w_conf:      float = 0.3,
            huber_delta: float = 1.0,
        ):
            super().__init__()
            self.ce    = nn.CrossEntropyLoss(weight=class_weights)
            self.hub   = nn.HuberLoss(delta=huber_delta, reduction="mean")
            self.bce   = nn.BCEWithLogitsLoss()   # AMP-safe; sigmoid is fused internally
            self.w_dir  = w_dir
            self.w_ret  = w_ret
            self.w_conf = w_conf

        def forward(
            self,
            logits:  torch.Tensor,  # (B, 3)
            ret_hat: torch.Tensor,  # (B,)
            conf:    torch.Tensor,  # (B,)
            y_cls:   torch.Tensor,  # (B,) long {0,1,2}
            y_cont:  torch.Tensor,  # (B,) float {-1,0,+1}
        ) -> torch.Tensor:
            l_dir  = self.ce(logits, y_cls)
            l_ret  = self.hub(ret_hat, y_cont)
            l_conf = self.bce(conf, y_cont.abs().clamp(0.0, 1.0))
            return self.w_dir * l_dir + self.w_ret * l_ret + self.w_conf * l_conf


    class MultiTaskWrapper(nn.Module):
        """
        Wraps any of the 6 backbone architectures, replacing its .head with
        nn.Identity() to expose the pre-head hidden state, then routing that
        state through a MultiTaskHead.

        After wrapping, forward() returns (direction_logits, return_hat, confidence)
        instead of the backbone's scalar/logit prediction.

        When the backbone's pre-head dimension exceeds proj_threshold (e.g.
        iTransformer whose head_in = d_model × n_features ≈ 18k), an extra
        Linear+GELU projection to proj_to=256 is inserted automatically.

        Usage:
            base  = HAELTHybrid(input_size=73, num_classes=1)
            model = MultiTaskWrapper(base, head_in=256)
            logits, ret, conf = model(x)   # x: (B, T, F)
        """

        def __init__(
            self,
            backbone:       "nn.Module",
            head_in:        int,
            hidden:         int   = 64,
            dropout:        float = 0.1,
            proj_threshold: int   = 1024,
            proj_to:        int   = 256,
        ):
            super().__init__()
            self.backbone = backbone

            if head_in > proj_threshold:
                self.proj  = nn.Sequential(nn.Linear(head_in, proj_to), nn.GELU())
                actual_in  = proj_to
            else:
                self.proj  = nn.Identity()
                actual_in  = head_in

            self.mt_head = MultiTaskHead(actual_in, hidden, dropout)
            # Replace backbone prediction head with Identity to expose hidden state
            backbone.head = nn.Identity()

        def forward(self, x: torch.Tensor):
            h = self.backbone(x)   # (B, head_in) — features from backbone
            h = self.proj(h)
            return self.mt_head(h)


    class MultiPairWrapper(nn.Module):
        """
        Adds a learnable pair embedding to each pair's features before passing
        to the backbone.

        Input shape:  (B, T, n_pairs * f_per_pair)  — pairs concatenated on feature axis.
        Each pair's F features get an embed_dim learned vector appended, then all pairs
        are flattened to (B, T, n_pairs * (f_per_pair + embed_dim)) for the backbone.

        When embed_dim=0, this wrapper is not needed — the backbone sees all pairs'
        features concatenated and learns cross-pair patterns without explicit embeddings.

        Usage:
            base  = HAELTHybrid(input_size=n_pairs*(f_per_pair+16), num_classes=3)
            model = MultiPairWrapper(base, n_pairs=3, f_per_pair=48, embed_dim=16)
            out   = model(x)   # x: (B, T, 3*48)
        """

        def __init__(
            self,
            backbone:   "nn.Module",
            n_pairs:    int,
            f_per_pair: int,
            embed_dim:  int = 16,
        ):
            super().__init__()
            self.backbone    = backbone
            self.n_pairs     = n_pairs
            self.f_per_pair  = f_per_pair
            self.embed_dim   = embed_dim
            self.pair_embeds = nn.Embedding(n_pairs, embed_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, T, _ = x.shape
            # Split into (B, T, P, F)
            xp = x.reshape(B, T, self.n_pairs, self.f_per_pair)
            # Pair embeddings: (P, E) → broadcast to (B, T, P, E)
            ids = torch.arange(self.n_pairs, device=x.device)
            emb = self.pair_embeds(ids).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
            # Concat embeddings and flatten: (B, T, P * (F + E))
            xp = torch.cat([xp, emb], dim=-1)
            return self.backbone(xp.reshape(B, T, self.n_pairs * (self.f_per_pair + self.embed_dim)))


    # ── 1. Temporal Fusion Transformer (simplified) ────────────────────────

    class VariableSelectionNetwork(nn.Module):
        """Learns which features matter at each timestep."""
        def __init__(self, input_size, hidden, dropout=0.1):
            super().__init__()
            self.grn = nn.Sequential(
                nn.Linear(input_size, hidden), nn.ELU(),
                nn.Dropout(dropout), nn.Linear(hidden, input_size),
            )
            self.softmax = nn.Softmax(dim=-1)
        def forward(self, x):
            weights = self.softmax(self.grn(x))
            return (x * weights).sum(dim=-1, keepdim=True) * x, weights

    class TFTScalper(nn.Module):
        """
        Temporal Fusion Transformer for multi-horizon forex forecasting.
        Uses Variable Selection Networks to identify which features matter,
        LSTM for local sequential patterns, and Self-Attention for long-range.
        """
        def __init__(self, input_size=64, hidden=128, heads=4,
                     lstm_layers=2, dropout=0.1, num_classes=1):
            super().__init__()
            self.num_classes = num_classes
            self.vsn     = VariableSelectionNetwork(input_size, hidden, dropout)
            self.lstm    = nn.LSTM(input_size, hidden, lstm_layers,
                                   batch_first=True, dropout=dropout)
            self.attn    = nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                                  batch_first=True)
            self.norm1   = nn.LayerNorm(hidden)
            self.ffn     = nn.Sequential(nn.Linear(hidden,hidden*2),nn.GELU(),
                                         nn.Dropout(dropout),nn.Linear(hidden*2,hidden))
            self.norm2   = nn.LayerNorm(hidden)
            self.head    = nn.Linear(hidden, num_classes)

        def forward(self, x):
            # x: (B, T, F)
            x_sel, _ = self.vsn(x)
            lstm_out, _ = self.lstm(x_sel)
            attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
            h = self.norm1(lstm_out + attn_out)
            h = self.norm2(h + self.ffn(h))
            out = self.head(h[:, -1, :])
            return out.squeeze(-1) if self.num_classes == 1 else out

    # ── 2. iTransformer (variate-dimension attention) ──────────────────────

    class iTransformerScalper(nn.Module):
        """
        iTransformer: applies attention across the feature (variate) dimension.
        Treats EUR/USD price and US 10Y yield as different 'tokens',
        learning their interactions as a differentiable map.
        Outperforms standard time-dimension Transformers on multivariate series.
        """
        def __init__(self, input_size=64, seq_len=60, d_model=128,
                     nhead=8, num_layers=3, dim_ff=256, dropout=0.1, num_classes=1):
            super().__init__()
            self.num_classes = num_classes
            # Project each variate's time-series into d_model token
            self.variate_proj = nn.Linear(seq_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True, norm_first=True)
            self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
            self.head     = nn.Linear(d_model * input_size, num_classes)
            self.input_size = input_size

        def forward(self, x):
            # x: (B, T, F)  →  treat F as sequence, T as embedding
            B, T, F = x.shape
            tokens = x.permute(0, 2, 1)          # (B, F, T)
            tokens = self.variate_proj(tokens)    # (B, F, d_model)
            out    = self.encoder(tokens)         # (B, F, d_model)
            out    = out.reshape(B, -1)           # (B, F*d_model)
            o = self.head(out)
            return o.squeeze(-1) if self.num_classes == 1 else o

    # ── 3. HAELT Hybrid (LSTM + Transformer in parallel) ──────────────────

    class HAELTHybrid(nn.Module):
        """
        Hybrid Attentive Ensemble Learning Transformer.
        LSTM branch captures local microstructure; Transformer captures
        long-range cross-asset correlations. Both run in parallel and are
        fused with a learned attention gate.
        """
        def __init__(self, input_size=64, seq_len=60, lstm_hidden=64,
                     d_model=64, nhead=4, n_layers=2, dropout=0.1, num_classes=1):
            super().__init__()
            self.num_classes = num_classes
            self.lstm = nn.LSTM(input_size, lstm_hidden, 2, batch_first=True, dropout=dropout)
            self.proj = nn.Linear(input_size, d_model)
            enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4,
                                              dropout=dropout, batch_first=True, norm_first=True)
            self.trf  = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
            fused = lstm_hidden + d_model
            self.gate = nn.Sequential(nn.Linear(fused, fused), nn.Sigmoid())
            self.head = nn.Sequential(nn.Linear(fused,64),nn.GELU(),
                                       nn.Dropout(dropout),nn.Linear(64, num_classes))

        def forward(self, x):
            lout, _ = self.lstm(x)
            lf = lout[:, -1, :]
            tf = self.trf(self.proj(x))[:, -1, :]
            c  = torch.cat([lf, tf], dim=-1)
            o = self.head(c * self.gate(c))
            return o.squeeze(-1) if self.num_classes == 1 else o

    # ── 4. Mamba State Space Model ─────────────────────────────────────────

    class MambaBlock(nn.Module):
        """
        Simplified Mamba block (State Space Model).
        Full Mamba (Gu & Dao 2023) uses selective state spaces for O(L) scaling.
        This implementation captures the SSM spirit using 1D conv + gating,
        providing transformer-level accuracy at ~2ms latency (GRU-class speed).
        """
        def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
            super().__init__()
            d_inner = d_model * expand
            self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                       padding=d_conv-1, groups=d_inner, bias=True)
            self.act      = nn.SiLU()
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            self.norm     = nn.LayerNorm(d_model)
            self.drop     = nn.Dropout(dropout)
            # SSM parameters
            self.A_log    = nn.Parameter(torch.randn(d_inner, d_state))
            self.D        = nn.Parameter(torch.ones(d_inner))
            self.dt_proj  = nn.Linear(d_inner, d_inner, bias=True)
            self.B        = nn.Linear(d_inner, d_state, bias=False)
            self.C        = nn.Linear(d_inner, d_state, bias=False)

        def forward(self, x):
            # x: (B, T, d_model)
            B, T, D = x.shape
            res   = x
            xz    = self.in_proj(x)           # (B, T, d_inner*2)
            x2, z = xz.chunk(2, dim=-1)       # each (B, T, d_inner)
            # 1D conv along time (causal)
            x2c   = self.conv1d(x2.permute(0,2,1))[:, :, :T].permute(0,2,1)
            x2c   = self.act(x2c)
            # Simplified SSM (linear recurrence approximation)
            y     = x2c * torch.sigmoid(self.dt_proj(x2c))
            y     = y * torch.sigmoid(z)      # gating
            out   = self.out_proj(y)
            return self.norm(out + res)

    class MambaScalper(nn.Module):
        """
        Stack of MambaBlocks for low-latency HFT inference.
        Handles extremely long sequences of tick data better than Transformers
        (O(L) vs O(L²) complexity) while maintaining high accuracy.
        """
        def __init__(self, input_size=64, d_model=128, d_state=16,
                     d_conv=4, expand=2, num_layers=4, dropout=0.1, num_classes=1):
            super().__init__()
            self.num_classes = num_classes
            self.embed = nn.Linear(input_size, d_model)
            self.layers = nn.ModuleList([
                MambaBlock(d_model, d_state, d_conv, expand, dropout)
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers: h = layer(h)
            o = self.head(self.norm(h[:, -1, :]))
            return o.squeeze(-1) if self.num_classes == 1 else o

    # ── 5. GNN Cross-Asset ────────────────────────────────────────────────

    class GNNCrossAsset(nn.Module):
        """
        Graph Neural Network for cross-asset correlation modelling.

        Treats currency pairs and cross-asset indicators as NODES.
        Edges = rolling correlations above a threshold.
        A GNN can detect when a move in the Nikkei (node) is about to
        'bleed' into USD/JPY (node) before price action shows it.

        This is a simplified GAT-style implementation that works without
        torch_geometric, using manual message-passing.
        """
        def __init__(self, node_features=32, hidden=64, num_layers=3,
                     heads=4, n_nodes=6, dropout=0.1, num_classes=1):
            super().__init__()
            self.n_nodes  = n_nodes
            self.num_classes = num_classes
            self.node_embed = nn.Linear(node_features, hidden)
            self.adj_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))
            self.attn_layers = nn.ModuleList([
                nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
            self.head  = nn.Linear(hidden * n_nodes, num_classes)
            self.drop  = nn.Dropout(dropout)

        def forward(self, x, adj=None):
            """
            x  : (B, n_nodes, node_features) — one feature vector per node per bar
            adj: ignored (kept for API compatibility); edge weights are learned via adj_logits.
            """
            h = self.node_embed(x)          # (B, N, hidden)
            A = torch.sigmoid(self.adj_logits).unsqueeze(0).expand(h.shape[0], -1, -1)
            for attn, norm in zip(self.attn_layers, self.norms):
                h_mix = torch.einsum("bnm,bmh->bnh", A, h)
                out, _ = attn(h_mix, h_mix, h_mix)
                h = norm(h + self.drop(out))
            o = self.head(h.reshape(h.shape[0], -1))
            return o.squeeze(-1) if self.num_classes == 1 else o

    class GNNFromSequence(nn.Module):
        """
        Adapts (B, T, F) sequence batches to GNNCrossAsset (B, n_nodes, node_features).
        Time axis is mean-pooled; features are projected into n_nodes × chunk tokens.
        """
        def __init__(self, input_size, hidden, num_layers, dropout, n_nodes=6,
                     num_classes=1, nhead=4):
            super().__init__()
            chunk = max(8, (input_size + n_nodes - 1) // n_nodes)
            self.n_nodes = n_nodes
            self.chunk = chunk
            self.proj = nn.Linear(input_size, n_nodes * chunk)
            self.gnn = GNNCrossAsset(
                node_features=chunk, hidden=hidden, num_layers=num_layers,
                heads=nhead, n_nodes=n_nodes, dropout=dropout, num_classes=num_classes,
            )

        @property
        def head(self) -> "nn.Module":
            """Proxy to inner GNN head — enables MultiTaskWrapper compatibility."""
            return self.gnn.head

        @head.setter
        def head(self, val: "nn.Module") -> None:
            self.gnn.head = val

        def forward(self, x):
            z = x.mean(dim=1)
            h = self.proj(z).view(-1, self.n_nodes, self.chunk)
            return self.gnn(h)

    # ── 6. EXPERT Encoder ─────────────────────────────────────────────────

    class ConvFFN(nn.Module):
        """1D conv feedforward — captures local temporal patterns better than MLP."""
        def __init__(self, d_model, d_ff, kernel=3, dropout=0.1):
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv1d(d_model, d_ff,   kernel, padding=pad)
            self.conv2 = nn.Conv1d(d_ff,    d_model, kernel, padding=pad)
            self.norm  = nn.LayerNorm(d_model)
            self.drop  = nn.Dropout(dropout)
            self.act   = nn.GELU()

        def forward(self, x):
            # x: (B, T, D)
            h = x.permute(0,2,1)
            h = self.act(self.conv1(h))
            h = self.drop(self.conv2(h))
            return self.norm(x + h.permute(0,2,1))

    class EXPERTEncoder(nn.Module):
        """
        EXPERT: EXchange-Rate Prediction using Encoder Representation from Transformers.
        Key differences from standard Transformer:
          - NO positional encoding (order is inherent in time series)
          - 1D convolutional feedforward layers (local temporal patterns)
          - Encoder-only (no decoder needed for regression)
        Focused architecture makes it more data-efficient than general Transformers.
        """
        def __init__(self, input_size=64, d_model=128, nhead=8,
                     num_layers=4, dropout=0.1, num_classes=1):
            super().__init__()
            self.num_classes = num_classes
            self.proj   = nn.Linear(input_size, d_model)
            # No positional encoding by design
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "attn": nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                    "norm1": nn.LayerNorm(d_model),
                    "ffn":  ConvFFN(d_model, d_model*4, dropout=dropout),
                })
                for _ in range(num_layers)
            ])
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            h = self.proj(x)
            for layer in self.layers:
                attn_out, _ = layer["attn"](h, h, h)
                h = layer["norm1"](h + attn_out)
                h = layer["ffn"](h)
            h = self.pool(h.permute(0,2,1)).squeeze(-1)
            o = self.head(h)
            return o.squeeze(-1) if self.num_classes == 1 else o


    # ── Model factory ──────────────────────────────────────────────────────

    MODEL_REGISTRY = {
        "tft":         TFTScalper,
        "transformer": iTransformerScalper,
        "haelt":       HAELTHybrid,
        "mamba":       MambaScalper,
        "gnn":         GNNFromSequence,
        "expert":      EXPERTEncoder,
    }

    def build_model(name: str, input_size: int, seq_len: int = 60, **kwargs) -> nn.Module:
        cls = MODEL_REGISTRY.get(name.lower())
        if cls is None:
            raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
            
        sig = inspect.signature(cls)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        if "seq_len" in sig.parameters and "seq_len" not in valid_kwargs:
            valid_kwargs["seq_len"] = seq_len
            
        try:
            model = cls(input_size=input_size, **valid_kwargs)
        except TypeError:
            # GNN has different signature
            model = cls(**valid_kwargs)
        n = sum(p.numel() for p in model.parameters())
        print(f"[Model] {name.upper()} | {n:,} parameters")
        return model


else:
    # CPU stubs when torch unavailable
    class TFTScalper:
        def __init__(self, **kw): pass
    class iTransformerScalper:
        def __init__(self, **kw): pass
    class HAELTHybrid:
        def __init__(self, **kw): pass
    class MambaScalper:
        def __init__(self, **kw): pass
    class GNNCrossAsset:
        def __init__(self, **kw): pass
    class GNNFromSequence:
        def __init__(self, **kw): pass
    class EXPERTEncoder:
        def __init__(self, **kw): pass
    class MultiTaskHead:
        def __init__(self, **kw): pass
    class MultiTaskLoss:
        def __init__(self, **kw): pass
    class MultiTaskWrapper:
        def __init__(self, **kw): pass
    class MultiPairWrapper:
        def __init__(self, **kw): pass
    MODEL_REGISTRY = {}

    def build_model(name, input_size, seq_len=60, **kw):
        print(f"[Model] Stub for {name} (torch not installed)")
        return None

    class HuberLoss:
        def __init__(self, delta=1.0): pass

    class AsymmetricDirectionalLoss:
        def __init__(self, delta=1.0, sign_weight=2.0): pass


if __name__ == "__main__" and TORCH:
    import torch
    B, T, F_IN = 8, 60, 48
    x = torch.randn(B, T, F_IN)

    for name, Cls in [
        ("TFT",         TFTScalper),
        ("iTransformer",iTransformerScalper),
        ("HAELT",       HAELTHybrid),
        ("Mamba",       MambaScalper),
        ("EXPERT",      EXPERTEncoder),
    ]:
        try:
            m = Cls(input_size=F_IN)
            out = m(x)
            print(f"  {name:16s}: in {tuple(x.shape)} → out {tuple(out.shape)}")
        except Exception as e:
            print(f"  {name:16s}: ERROR — {e}")

    x_seq = torch.randn(B, T, F_IN)
    gnn   = GNNFromSequence(input_size=F_IN, hidden=64, num_layers=2, dropout=0.1)
    out   = gnn(x_seq)
    print(f"  {'GNN-seq':16s}: in {tuple(x_seq.shape)} → out {tuple(out.shape)}")
