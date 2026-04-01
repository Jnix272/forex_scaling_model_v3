"""
labeling/rl_reward_labeling.py
===============================
RL Reward Signal labeling: assign a reward to each bar based on the
forward P&L of entering at that bar, minus transaction costs.

Unlike Triple Barrier Method (which assigns fixed {-1,0,+1} labels),
RL reward labeling produces a continuous signal that directly represents
the risk-adjusted P&L the agent would have earned — making it the natural
training signal for PPO/DQN agents.

Label values:
  positive  → profitable entry at this bar
  negative  → losing entry
  near-zero → no clear edge (hold)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def compute_rl_reward_labels(
    bars:              pd.DataFrame,
    features:          pd.DataFrame,
    atr_col:           str   = "atr_6",
    lookahead_bars:    int   = 10,
    profit_atr_mult:   float = 1.5,
    stop_atr_mult:     float = 1.0,
    tx_cost_pips:      float = 1.5,
    pip_size:          float = 0.0001,
) -> pd.DataFrame:
    """
    Compute per-bar RL reward signals for long and short entries.

    For each bar t, simulates:
      - Long entry: buy at ask_close[t], exit at the first of:
          (a) price hits profit target (entry + ATR × profit_mult)
          (b) price hits stop loss   (entry - ATR × stop_mult)
          (c) lookahead_bars elapsed
      - Short entry: symmetric

    Returns a DataFrame with columns:
      reward_long  : net reward for a long entry at this bar
      reward_short : net reward for a short entry
      reward       : max(reward_long, reward_short)  ← training label
      label        : sign of reward → {-1, 0, +1} for classifier use
      optimal_side : 'long' | 'short' | 'hold'
    """
    close = bars["close"].reindex(features.index).ffill().values
    if "ask_close" in bars.columns:
        entry_long  = bars["ask_close"].reindex(features.index).ffill().values
        entry_short = bars["bid_close"].reindex(features.index).ffill().values
    else:
        spread_half  = features["spread_pips"].values * pip_size / 2
        entry_long   = close + spread_half
        entry_short  = close - spread_half

    atr = features[atr_col].values if atr_col in features.columns else np.full(len(close), 0.0005)
    tx_cost = tx_cost_pips * pip_size
    n = len(close)

    reward_long  = np.zeros(n, dtype=np.float32)
    reward_short = np.zeros(n, dtype=np.float32)

    for i in range(n - lookahead_bars):
        el = entry_long[i];  es = entry_short[i]
        tp_l = el + profit_atr_mult * atr[i]
        sl_l = el - stop_atr_mult  * atr[i]
        tp_s = es - profit_atr_mult * atr[i]
        sl_s = es + stop_atr_mult  * atr[i]

        # Simulate forward path
        horizon = close[i+1 : i+1+lookahead_bars]

        # Long
        pnl_l = None
        for p in horizon:
            if p >= tp_l:   pnl_l = (tp_l - el) / pip_size;  break
            elif p <= sl_l: pnl_l = (sl_l - el) / pip_size;  break
        if pnl_l is None:   pnl_l = (horizon[-1] - el) / pip_size
        reward_long[i]  = pnl_l - tx_cost_pips

        # Short
        pnl_s = None
        for p in horizon:
            if p <= tp_s:   pnl_s = (es - tp_s) / pip_size; break
            elif p >= sl_s: pnl_s = (es - sl_s) / pip_size; break
        if pnl_s is None:   pnl_s = (es - horizon[-1]) / pip_size
        reward_short[i] = pnl_s - tx_cost_pips

    # Combined label — pick the best (most profitable) direction
    reward = np.maximum(reward_long, reward_short)
    label = np.select(
        [reward > tx_cost_pips, reward < -tx_cost_pips],
        [1, -1], default=0
    )
    optimal = np.select(
        [reward_long > reward_short, reward_short > reward_long],
        ["long", "short"], default="hold"
    )

    result = pd.DataFrame({
        "reward_long":  reward_long,
        "reward_short": reward_short,
        "reward":       reward,
        "label":        label,
        "optimal_side": optimal,
    }, index=features.index)

    # Drop last rows (no complete lookahead)
    result.iloc[-lookahead_bars:] = np.nan
    result = result.dropna()

    counts = pd.Series(result["label"]).value_counts()
    print(f"[RLLabeling] {len(result):,} labels | "
          f"Long: {counts.get(1,0):,}  Hold: {counts.get(0,0):,}  Short: {counts.get(-1,0):,}")
    return result


def align_labels_with_features(
    labels_df:    pd.DataFrame,
    features_df:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Inner-join and return (X, y) ready for supervised or RL training."""
    combined = features_df.join(labels_df[["reward", "label"]], how="inner")
    combined = combined.dropna()
    X = combined.drop(columns=["reward", "label"])
    y = combined["label"]
    print(f"[Align] {len(X):,} samples × {X.shape[1]} features | "
          f"Balance: {y.value_counts().to_dict()}")
    return X, y


if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.data_ingestion import generate_synthetic_tick_data, ForexDataPipeline
    from features.feature_engineering import FeatureEngineer

    ticks = generate_synthetic_tick_data(n_rows=500_000)
    bars  = ForexDataPipeline(bar_freq="1min", session_filter=False,
                               apply_frac_diff=False).run(ticks)
    fe = FeatureEngineer()
    feats = fe.build(bars)

    labels = compute_rl_reward_labels(bars, feats)
    print(f"\nSample labels:\n{labels.head(5)}")
    X, y = align_labels_with_features(labels, feats)
    print(f"Training shapes: X={X.shape} y={y.shape}")
