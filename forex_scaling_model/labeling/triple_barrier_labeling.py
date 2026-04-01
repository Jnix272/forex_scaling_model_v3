"""
labeling/triple_barrier_labeling.py
====================================
Triple Barrier Method (TBM): ATR-scaled TP/SL + vertical horizon; first touch
wins. Parallel **Numba** scan over all bars (optional; falls back to a sequential
reference implementation if Numba is unavailable).

Fully automated: enable/disable via LABELING["tbm_numba"] / ["tbm_parallel"] in
config/settings.py (no manual steps in training).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── Optional Numba (required for fast path on large series) ─────────────────
try:
    from numba import njit, prange
    _NUMBA_IMPORT_OK = True
except ImportError:
    _NUMBA_IMPORT_OK = False
    njit = None  # type: ignore
    prange = range  # type: ignore


def _default_labeling() -> Dict[str, Any]:
    try:
        from config.settings import LABELING as L
        return L
    except Exception:
        return {}


def _scan_outcomes_sequential(
    close: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    atr: np.ndarray,
    profit_mult: float,
    stop_mult: float,
    vertical_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation (single-threaded). Used for tests and fallback."""
    n = close.shape[0]
    n_valid = n - vertical_bars - 1
    if n_valid <= 0:
        z = np.zeros(0, dtype=np.int8)
        return z, z.astype(np.int32), z, z.astype(np.int32)

    lo_o = np.zeros(n_valid, dtype=np.int8)
    tl_o = np.zeros(n_valid, dtype=np.int32)
    so_o = np.zeros(n_valid, dtype=np.int8)
    ts_o = np.zeros(n_valid, dtype=np.int32)

    for i in range(n_valid):
        el = entry_long[i]
        tp_l = el + profit_mult * atr[i]
        sl_l = el - stop_mult * atr[i]
        es = entry_short[i]
        tp_s = es - profit_mult * atr[i]
        sl_s = es + stop_mult * atr[i]

        lo = np.int8(0)
        tl = np.int32(vertical_bars)
        so = np.int8(0)
        ts = np.int32(vertical_bars)

        for t in range(vertical_bars):
            p = close[i + 1 + t]
            if lo == 0:
                if p >= tp_l:
                    lo = 1
                    tl = t
                elif p <= sl_l:
                    lo = -1
                    tl = t
            if so == 0:
                if p <= tp_s:
                    so = 1
                    ts = t
                elif p >= sl_s:
                    so = -1
                    ts = t
            if lo != 0 and so != 0:
                break

        lo_o[i] = lo
        tl_o[i] = tl
        so_o[i] = so
        ts_o[i] = ts

    return lo_o, tl_o, so_o, ts_o


if _NUMBA_IMPORT_OK:

    @njit(cache=True, fastmath=True, parallel=True)
    def _scan_outcomes_numba(
        close,
        entry_long,
        entry_short,
        atr,
        profit_mult,
        stop_mult,
        vertical_bars,
        n_valid,
    ):
        lo_o = np.zeros(n_valid, dtype=np.int8)
        tl_o = np.zeros(n_valid, dtype=np.int32)
        so_o = np.zeros(n_valid, dtype=np.int8)
        ts_o = np.zeros(n_valid, dtype=np.int32)

        for i in prange(n_valid):
            el = entry_long[i]
            tp_l = el + profit_mult * atr[i]
            sl_l = el - stop_mult * atr[i]
            es = entry_short[i]
            tp_s = es - profit_mult * atr[i]
            sl_s = es + stop_mult * atr[i]

            lo = np.int8(0)
            tl = vertical_bars
            so = np.int8(0)
            ts = vertical_bars

            for t in range(vertical_bars):
                p = close[i + 1 + t]
                if lo == 0:
                    if p >= tp_l:
                        lo = 1
                        tl = t
                    elif p <= sl_l:
                        lo = -1
                        tl = t
                if so == 0:
                    if p <= tp_s:
                        so = 1
                        ts = t
                    elif p >= sl_s:
                        so = -1
                        ts = t
                if lo != 0 and so != 0:
                    break

            lo_o[i] = lo
            tl_o[i] = tl
            so_o[i] = so
            ts_o[i] = ts

        return lo_o, tl_o, so_o, ts_o

    @njit(cache=True, fastmath=True)
    def _scan_outcomes_numba_serial(
        close,
        entry_long,
        entry_short,
        atr,
        profit_mult,
        stop_mult,
        vertical_bars,
        n_valid,
    ):
        lo_o = np.zeros(n_valid, dtype=np.int8)
        tl_o = np.zeros(n_valid, dtype=np.int32)
        so_o = np.zeros(n_valid, dtype=np.int8)
        ts_o = np.zeros(n_valid, dtype=np.int32)

        for i in range(n_valid):
            el = entry_long[i]
            tp_l = el + profit_mult * atr[i]
            sl_l = el - stop_mult * atr[i]
            es = entry_short[i]
            tp_s = es - profit_mult * atr[i]
            sl_s = es + stop_mult * atr[i]

            lo = np.int8(0)
            tl = vertical_bars
            so = np.int8(0)
            ts = vertical_bars

            for t in range(vertical_bars):
                p = close[i + 1 + t]
                if lo == 0:
                    if p >= tp_l:
                        lo = 1
                        tl = t
                    elif p <= sl_l:
                        lo = -1
                        tl = t
                if so == 0:
                    if p <= tp_s:
                        so = 1
                        ts = t
                    elif p >= sl_s:
                        so = -1
                        ts = t
                if lo != 0 and so != 0:
                    break

            lo_o[i] = lo
            tl_o[i] = tl
            so_o[i] = so
            ts_o[i] = ts

        return lo_o, tl_o, so_o, ts_o


def _combine_directional_labels(
    lo: np.ndarray,
    tl: np.ndarray,
    so: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    """Vectorized merge: bullish/bearish/neutral from long/short barrier outcomes."""
    n = lo.shape[0]
    label = np.zeros(n, dtype=np.int8)
    both_tp = (lo == 1) & (so == 1)
    if np.any(both_tp):
        label[both_tp] = np.where(tl[both_tp] <= ts[both_tp], 1, -1).astype(np.int8)
    rest = ~both_tp
    label[rest & (lo == 1)] = 1
    label[rest & (so == 1)] = -1
    return label


def _run_barrier_scan(
    close: np.ndarray,
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    atr: np.ndarray,
    profit_mult: float,
    stop_mult: float,
    vertical_bars: int,
    use_numba: bool,
    parallel: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Returns (lo, tl, so, ts) each length n_valid, and backend tag for logging.
    """
    n = close.shape[0]
    n_valid = n - vertical_bars - 1
    if n_valid <= 0:
        z = np.zeros(0, dtype=np.int8)
        return z, z.astype(np.int32), z, z.astype(np.int32), "empty"

    if use_numba and _NUMBA_IMPORT_OK:
        try:
            if parallel:
                lo, tl, so, ts = _scan_outcomes_numba(
                    close,
                    entry_long,
                    entry_short,
                    atr,
                    profit_mult,
                    stop_mult,
                    vertical_bars,
                    n_valid,
                )
                return lo, tl, so, ts, "numba_parallel"
            lo, tl, so, ts = _scan_outcomes_numba_serial(
                close,
                entry_long,
                entry_short,
                atr,
                profit_mult,
                stop_mult,
                vertical_bars,
                n_valid,
            )
            return lo, tl, so, ts, "numba_serial"
        except Exception as ex:
            warnings.warn(f"[TBM] Numba scan failed ({ex}); using sequential scan.")
            return (
                *_scan_outcomes_sequential(
                    close,
                    entry_long,
                    entry_short,
                    atr,
                    profit_mult,
                    stop_mult,
                    vertical_bars,
                ),
                "sequential_fallback",
            )

    return (
        *_scan_outcomes_sequential(
            close,
            entry_long,
            entry_short,
            atr,
            profit_mult,
            stop_mult,
            vertical_bars,
        ),
        "sequential",
    )


def compute_triple_barrier_labels(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    atr_col: str = "atr_6",
    vertical_bars: int = 10,
    profit_atr_mult: float = 1.5,
    stop_atr_mult: float = 1.0,
    pip_size: float = 0.0001,
    use_numba: Optional[bool] = None,
    parallel: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Per-bar triple-barrier outcomes; combined directional label.

    When ``use_numba`` / ``parallel`` are None, values are read from
    ``config.settings.LABELING`` (``tbm_numba``, ``tbm_parallel``).
    """
    cfg = _default_labeling()
    if use_numba is None:
        use_numba = bool(cfg.get("tbm_numba", True))
    if parallel is None:
        parallel = bool(cfg.get("tbm_parallel", True))

    close = bars["close"].reindex(features.index).ffill().values.astype(np.float64)
    if "ask_close" in bars.columns:
        entry_long = bars["ask_close"].reindex(features.index).ffill().values.astype(np.float64)
        entry_short = bars["bid_close"].reindex(features.index).ffill().values.astype(np.float64)
    else:
        spread_half = features["spread_pips"].values.astype(np.float64) * pip_size / 2
        entry_long = close + spread_half
        entry_short = close - spread_half

    atr = (
        features[atr_col].values.astype(np.float64)
        if atr_col in features.columns
        else np.full(len(close), 0.0005, dtype=np.float64)
    )

    n = len(close)
    n_valid = n - vertical_bars - 1
    reward_long = np.zeros(n, dtype=np.float32)
    reward_short = np.zeros(n, dtype=np.float32)
    label = np.zeros(n, dtype=np.int8)

    if n_valid <= 0:
        return pd.DataFrame(
            {
                "reward_long": reward_long,
                "reward_short": reward_short,
                "reward": reward_long,
                "label": label,
            },
            index=features.index,
        ).iloc[0:0]

    lo, tl, so, ts, backend = _run_barrier_scan(
        close,
        entry_long,
        entry_short,
        atr,
        float(profit_atr_mult),
        float(stop_atr_mult),
        int(vertical_bars),
        use_numba=use_numba,
        parallel=parallel,
    )

    comb = _combine_directional_labels(lo, tl, so, ts)
    reward_long[:n_valid] = lo.astype(np.float32)
    reward_short[:n_valid] = so.astype(np.float32)
    label[:n_valid] = comb

    reward = np.where(
        np.abs(reward_long) >= np.abs(reward_short),
        reward_long,
        reward_short,
    )

    result = pd.DataFrame(
        {
            "reward_long": reward_long,
            "reward_short": reward_short,
            "reward": reward,
            "label": label,
        },
        index=features.index,
    )
    result.iloc[-vertical_bars:] = np.nan
    result = result.dropna()

    vc = pd.Series(result["label"]).value_counts()
    print(
        f"[TBMLabeling] {len(result):,} labels | backend={backend} | "
        f"Long+: {vc.get(1, 0):,}  Hold: {vc.get(0, 0):,}  Short+: {vc.get(-1, 0):,}"
    )
    return result
