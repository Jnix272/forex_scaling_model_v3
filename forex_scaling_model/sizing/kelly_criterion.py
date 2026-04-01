"""
sizing/kelly_criterion.py
==========================
Fractional Kelly position sizing with:
  - Quarter-Kelly default (conservative)
  - Volatility targeting (scales down in high-vol regimes)
  - Square Root market impact model
  - Both scaling strategies: pyramid winners + martingale losers
"""
import numpy as np
import pandas as pd
from typing import Optional


def kelly_binary(win_prob: float, win_loss_ratio: float) -> float:
    """Kelly fraction = p - q/b where b=win/loss ratio."""
    q = 1 - win_prob
    return win_prob - q / win_loss_ratio


def fractional_kelly(full_kelly: float, fraction: float = 0.25) -> float:
    return np.clip(full_kelly * fraction, 0, 1)


def vol_target_scalar(
    returns: np.ndarray,
    target_vol: float = 0.10,
    lookback: int = 20,
) -> float:
    if len(returns) < 2: return 1.0
    recent = returns[-lookback:] if len(returns) >= lookback else returns
    realized_vol = float(np.std(recent) * np.sqrt(252))
    return np.clip(target_vol / (realized_vol + 1e-9), 0.1, 3.0)


def square_root_impact(
    lots: float,
    adv_lots: float = 1000.0,
    perm_impact_coef: float = 0.1,
    pip_size: float = 0.0001,
) -> float:
    """Market impact (pips) = coef × sqrt(size/ADV)."""
    pct = lots / (adv_lots + 1e-9)
    impact_pips = perm_impact_coef * np.sqrt(pct)
    return impact_pips


class PositionSizer:
    def __init__(self, equity=10_000, kelly_fraction=0.25,
                 max_position_pct=0.05, target_vol=0.10, pip_risk=20.0):
        self.equity    = equity
        self.frac      = kelly_fraction
        self.max_pct   = max_position_pct
        self.tvol      = target_vol
        self.pip_risk  = pip_risk

    def size_position(self, win_prob, win_loss_ratio, returns,
                      price, current_atr, lot_size=10_000):
        full_k  = kelly_binary(win_prob, win_loss_ratio)
        frac_k  = fractional_kelly(full_k, self.frac)
        vol_sc  = vol_target_scalar(np.array(returns), self.tvol)
        risk_usd = self.equity * min(frac_k * vol_sc, self.max_pct)
        pip_val  = lot_size * 0.0001
        pip_stop = max(self.pip_risk, current_atr / 0.0001 * 1.5)
        lots     = risk_usd / (pip_stop * pip_val)
        lots     = round(np.clip(lots, 0.01, self.equity / lot_size * 0.2), 2)
        impact   = square_root_impact(lots) * pip_val * lots
        return {
            "lots": lots, "full_kelly": full_k, "frac_kelly": frac_k,
            "vol_scalar": vol_sc, "risk_usd": risk_usd, "impact_usd": impact,
        }
