"""
features/macro_features.py
===========================
Yield-spread and carry features for Forex models.

Interest-rate differentials are the dominant long-run driver of major FX pairs:
  EUR/USD  ← US10Y − DE10Y  (US-Germany spread)
  USD/JPY  ← US10Y − JP10Y  (US-Japan spread)
  GBP/USD  ← US10Y − GB10Y
  AUD/USD  ← AU10Y − US10Y (inverted — AUD high-yielder)
  USD/CAD  ← US10Y − CA10Y

Data sources (in priority order):
  1. FRED API (fredapi library) — free, no rate limit for daily data.
     Set FRED_API_KEY env var.  Key at https://fred.stlouisfed.org/docs/api/api_key.html
  2. Synthetic fallback — correlated random walks calibrated to historical levels.

Features produced (all filled from daily → minute resolution via ffill):
  spread_us_de, spread_us_jp, spread_us_gb, spread_us_au, spread_us_ca
  carry_eur, carry_jpy, carry_gbp, carry_aud, carry_cad
  yield_momentum_5d, yield_momentum_20d  (on primary EURUSD spread)
  yield_vol_20d                          (rolling 20-day std of primary spread)
"""

import os
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── FRED series IDs ──────────────────────────────────────────────────────────
FRED_SERIES = {
    "US10Y": "DGS10",          # US 10-year Treasury yield
    "DE10Y": "IRLTLT01DEM156N", # Germany 10-year Bund yield
    "JP10Y": "IRLTLT01JPM156N", # Japan 10-year JGB yield
    "GB10Y": "IRLTLT01GBM156N", # UK 10-year Gilt yield
    "AU10Y": "IRLTLT01AUM156N", # Australia 10-year yield
    "CA10Y": "IRLTLT01CAM156N", # Canada 10-year yield
}

# Approximate long-run mean values (used for synthetic fallback)
YIELD_DEFAULTS = {
    "US10Y": 4.50, "DE10Y": 2.40, "JP10Y": 0.80,
    "GB10Y": 4.20, "AU10Y": 4.40, "CA10Y": 3.80,
}


# ── FRED fetch ───────────────────────────────────────────────────────────────

def _fetch_fred_series(series_id: str, start: str, end: str,
                       fred_key: str) -> pd.Series:
    """Fetch a single FRED series using the fredapi library."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s.index = pd.DatetimeIndex(s.index).tz_localize("UTC")
        return s.dropna()
    except ImportError:
        raise RuntimeError("fredapi not installed — pip install fredapi")


def _fetch_all_yields(start: pd.Timestamp, end: pd.Timestamp,
                      fred_key: str) -> Dict[str, pd.Series]:
    """Fetch all yield curves from FRED. Returns dict of pd.Series indexed UTC."""
    start_s = str(start.date())
    end_s   = str(end.date())
    result  = {}
    for name, series_id in FRED_SERIES.items():
        try:
            result[name] = _fetch_fred_series(series_id, start_s, end_s, fred_key)
            print(f"[MacroFeatures] FRED {name}: {len(result[name])} observations")
        except Exception as e:
            print(f"[MacroFeatures] FRED {name} failed ({e}) — using synthetic")
            result[name] = None
    return result


def _synthetic_yields(start: pd.Timestamp, end: pd.Timestamp,
                      seed: int = 42) -> Dict[str, pd.Series]:
    """
    Generate synthetic yield series calibrated to realistic levels and
    correlations. Used when FRED API is unavailable.
    """
    rng  = np.random.default_rng(seed)
    days = pd.date_range(start.date(), end.date(), freq="B", tz="UTC")
    n    = len(days)

    # Correlated random walks
    corr_matrix = np.array([
        [1.00, 0.65, 0.40, 0.80, 0.70, 0.75],   # US10Y
        [0.65, 1.00, 0.30, 0.72, 0.55, 0.60],   # DE10Y
        [0.40, 0.30, 1.00, 0.35, 0.38, 0.42],   # JP10Y
        [0.80, 0.72, 0.35, 1.00, 0.65, 0.68],   # GB10Y
        [0.70, 0.55, 0.38, 0.65, 1.00, 0.62],   # AU10Y
        [0.75, 0.60, 0.42, 0.68, 0.62, 1.00],   # CA10Y
    ])
    L  = np.linalg.cholesky(corr_matrix)
    z  = rng.standard_normal((n, 6))
    cz = z @ L.T
    sigma = 0.03  # daily vol in bps * 100

    names  = list(YIELD_DEFAULTS.keys())
    levels = np.array([YIELD_DEFAULTS[k] for k in names])
    paths  = levels + np.cumsum(cz * sigma, axis=0)
    paths  = np.clip(paths, 0.01, 20.0)

    return {name: pd.Series(paths[:, i], index=days) for i, name in enumerate(names)}


# ── feature builder ──────────────────────────────────────────────────────────

class MacroYieldFeatureBuilder:
    """
    Builds yield-spread and carry features aligned to 1-minute bars.

    Usage:
        builder = MacroYieldFeatureBuilder()
        df = builder.build(bars)
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        momentum_windows: tuple = (5, 20),
        vol_window: int = 20,
    ):
        self._fred_key = fred_api_key or os.getenv("FRED_API_KEY", "")
        self._mom_wins = momentum_windows
        self._vol_win  = vol_window

    # ── public API ──────────────────────────────────────────────────────────

    def load_yields(self, start: pd.Timestamp, end: pd.Timestamp
                    ) -> Dict[str, pd.Series]:
        """Load yield series; fills missing with synthetic."""
        if self._fred_key:
            try:
                raw = _fetch_all_yields(start, end, self._fred_key)
                # Replace any failed series with synthetic
                synth = _synthetic_yields(start, end)
                for k in raw:
                    if raw[k] is None:
                        raw[k] = synth[k]
                return raw
            except Exception as e:
                print(f"[MacroFeatures] FRED batch failed ({e}) — full synthetic")

        return _synthetic_yields(start, end)

    def build(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Build macro yield features aligned to bars.index.
        All yields are forward-filled from daily → minute frequency.
        """
        idx   = bars.index
        start = pd.Timestamp(idx[0]).tz_convert("UTC") if idx.tzinfo else \
                pd.Timestamp(idx[0]).tz_localize("UTC")
        start -= pd.Timedelta(days=30)   # warm-up for momentum windows
        end   = pd.Timestamp(idx[-1]).tz_convert("UTC") if idx.tzinfo else \
                pd.Timestamp(idx[-1]).tz_localize("UTC")

        yields = self.load_yields(start, end)

        # Resample each yield series from daily → bars.index via ffill
        def _align(s: pd.Series) -> pd.Series:
            combined = s.reindex(s.index.union(bars.index)).sort_index()
            return combined.ffill().bfill().reindex(bars.index)

        Y: Dict[str, pd.Series] = {k: _align(v) for k, v in yields.items()}

        out = pd.DataFrame(index=bars.index)

        # ── Spreads ──────────────────────────────────────────────────────────
        out["spread_us_de"] = Y["US10Y"] - Y["DE10Y"]   # EUR/USD driver
        out["spread_us_jp"] = Y["US10Y"] - Y["JP10Y"]   # USD/JPY driver
        out["spread_us_gb"] = Y["US10Y"] - Y["GB10Y"]   # GBP/USD driver
        out["spread_us_au"] = Y["AU10Y"] - Y["US10Y"]   # AUD/USD driver (inverted)
        out["spread_us_ca"] = Y["US10Y"] - Y["CA10Y"]   # USD/CAD driver

        # ── Carry signal: positive = long base currency ───────────────────
        out["carry_eur"] = -out["spread_us_de"]  # long EUR when DE > US
        out["carry_jpy"] =  out["spread_us_jp"]  # long USD when US > JP
        out["carry_gbp"] = -out["spread_us_gb"]  # long GBP when GB > US
        out["carry_aud"] =  out["spread_us_au"]  # long AUD when AU > US
        out["carry_cad"] = -out["spread_us_ca"]  # long CAD when CA > US

        # ── Primary spread momentum + vol (USD−EUR as primary) ────────────
        primary = out["spread_us_de"]
        for w in self._mom_wins:
            out[f"yield_momentum_{w}d"] = primary - primary.shift(w)
        out["yield_vol_20d"] = primary.rolling(self._vol_win).std().fillna(0)

        return out.ffill().bfill().fillna(0)


# ── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Macro Yield Features — smoke test")
    idx  = pd.date_range("2024-01-01", periods=1440, freq="1min", tz="UTC")
    bars = pd.DataFrame({"close": np.random.randn(1440).cumsum() + 1.085}, index=idx)

    builder = MacroYieldFeatureBuilder()   # no FRED key → synthetic
    df = builder.build(bars)
    print(f"  Feature shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  NaNs: {df.isna().sum().sum()}")
    print(df.head(3))
    print("OK ✓")
