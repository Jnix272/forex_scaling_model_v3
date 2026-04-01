"""
data/data_ingestion.py
======================
High-resolution forex data ingestion and preprocessing.

Handles:
  - Loading tick / 1-min OHLCV data from CSV or Parquet
  - Fractional differentiation for stationarity
  - Bid-Ask spread validation (Golden Rule)
  - UTC timestamp normalization
  - Train/test splitting for walk-forward validation
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (use when real broker data is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_tick_data(
    n_rows: int = 100_000,
    pair: str = "EURUSD",
    base_price: float = 1.0850,
    spread_pips: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic EUR/USD tick data for development/testing.

    Uses a Geometric Brownian Motion (GBM) process to simulate mid-price,
    then adds a fixed spread to produce bid/ask columns.

    Parameters
    ----------
    n_rows       : Number of ticks to generate
    pair         : Currency pair label (cosmetic)
    base_price   : Starting mid-price
    spread_pips  : Fixed spread in pips (1 pip = 0.0001 for EURUSD)
    seed         : Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: [timestamp, bid, ask, mid, volume, spread]
    """
    rng = np.random.default_rng(seed)

    # GBM parameters (annualised, scaled to tick frequency ~1s)
    mu = 0.0      # drift (roughly zero for FX)
    sigma = 0.10  # annualised volatility
    dt = 1 / (252 * 24 * 3600)  # 1-second tick in year fraction

    # Simulate log-returns and compute price path
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_rows)
    mid_prices = base_price * np.exp(np.cumsum(log_returns))

    # Build timestamps: 1 tick per second starting from a recent date
    start = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start, periods=n_rows, freq="1s")

    # Spread and bid/ask construction
    half_spread = (spread_pips * 0.0001) / 2
    bid = mid_prices - half_spread
    ask = mid_prices + half_spread

    # Simulate volume (log-normal, spikes during London/NY session)
    hour = timestamps.hour
    session_multiplier = np.where(
        ((hour >= 7) & (hour <= 10)) | ((hour >= 13) & (hour <= 16)),
        rng.uniform(2.0, 4.0, n_rows),   # active session
        rng.uniform(0.5, 1.5, n_rows),   # quiet session
    )
    volume = np.round(np.exp(rng.normal(3.5, 0.8, n_rows)) * session_multiplier).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "bid": np.round(bid, 5),
        "ask": np.round(ask, 5),
        "mid": np.round(mid_prices, 5),
        "volume": volume,
        "spread": np.round(ask - bid, 5),
        "pair": pair,
    }).set_index("timestamp")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_tick_data(filepath: str) -> pd.DataFrame:
    """
    Load tick data from CSV or Parquet.
    Expects at minimum: timestamp (UTC), bid, ask columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(
            filepath,
            parse_dates=("timestamp",),
            iterator=False,
        )
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return _standardize_dataframe(df)


def enforce_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame index is UTC-aware. Raises if no timezone-aware index
    can be established. Logs conversions.
    """
    if df.index.tz is None:
        print("[DataIngestion] Localizing index to UTC")
        df.index = df.index.tz_localize("UTC")
    elif df.index.tz != "UTC":
        print(f"[DataIngestion] Converting index from {df.index.tz} to UTC")
        df.index = df.index.tz_convert("UTC")
    return df


def clean_bad_ticks(df: pd.DataFrame, z_thresh: float = 8.0, window: int = 60) -> pd.DataFrame:
    """
    Forex data often contains "bad ticks" (price spikes).
    Uses a rolling Z-score and median filter to remove outliers.
    """
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2

    # Rolling Z-score calculation
    rolling_mean = df["mid"].rolling(window=window).mean()
    rolling_std = df["mid"].rolling(window=window).std()
    z_score = (df["mid"] - rolling_mean) / (rolling_std + 1e-9)

    # Detect outliers
    is_outlier = z_score.abs() > z_thresh
    outlier_count = is_outlier.sum()

    if outlier_count > 0:
        print(f"[DataIngestion] Cleaning {outlier_count} bad ticks (> {z_thresh} sigma)")
        # Replace outliers with rolling median
        rolling_median = df["mid"].rolling(window=window).median()
        df.loc[is_outlier, "mid"] = rolling_median[is_outlier]
        # Re-derive bid/ask if they were shifted away from mid
        half_spread = (df["ask"] - df["bid"]) / 2
        df.loc[is_outlier, "bid"] = df.loc[is_outlier, "mid"] - half_spread[is_outlier]
        df.loc[is_outlier, "ask"] = df.loc[is_outlier, "mid"] + half_spread[is_outlier]

    return df


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce UTC timezone, sort by time, validate required columns.
    """
    required = {"bid", "ask"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}. Got: {set(df.columns)}")

    # Ensure UTC timestamp index and clean bad ticks
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    df = enforce_utc(df)
    df = df.sort_index()

    # Derived mid and cleaning
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2

    df = clean_bad_ticks(df)

    df = df.sort_index()

    # Derive mid-price if not present
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2

    # Derive spread
    df["spread"] = df["ask"] - df["bid"]

    # Golden Rule check
    if (df["spread"] <= 0).any():
        bad_count = (df["spread"] <= 0).sum()
        warnings.warn(
            f"⚠️  {bad_count} rows have zero or negative spread. "
            "Filtering these out — always include real bid/ask data!"
        )
        df = df[df["spread"] > 0]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING (tick → OHLCV bars)
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_bars(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Resample tick data to OHLCV bars.

    Parameters
    ----------
    df   : Tick DataFrame with mid, bid, ask, volume columns
    freq : Pandas offset alias e.g. '1min', '5min', '1H'

    Returns
    -------
    pd.DataFrame with [open, high, low, close, volume, spread_avg]
    """
    ohlcv = df["mid"].resample(freq).ohlc()
    ohlcv["volume"] = df["volume"].resample(freq).sum()
    ohlcv["spread_avg"] = df["spread"].resample(freq).mean()
    ohlcv["bid_close"] = df["bid"].resample(freq).last()
    ohlcv["ask_close"] = df["ask"].resample(freq).last()
    ohlcv = ohlcv.dropna(subset=["open"])  # type: ignore[call-overload]
    return ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# FRACTIONAL DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
    """
    Compute weights for Fixed-Window Fractional Differentiation (FFD).
    Preserves memory while achieving approximate stationarity.
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])


def fracDiff_FFD(series: pd.Series, d: float = 0.4, thres: float = 1e-5) -> pd.Series:
    """
    Apply Fixed-Window Fractional Differentiation to a price series.

    Why: Raw price series are non-stationary (integrated), breaking most ML models.
    Standard differencing (d=1) destroys memory. Fractional diff (d≈0.3-0.5)
    achieves stationarity while preserving long-memory correlations.

    Parameters
    ----------
    series : Raw price series (e.g., close prices)
    d      : Fractional order (0 < d < 1). Higher = more stationary, less memory.
    thres  : Weight threshold for truncating the infinite series

    Returns
    -------
    pd.Series: Fractionally differentiated series (same index, NaN at start)
    """
    w = _get_weights_ffd(d, thres)
    width = len(w) - 1
    output = {}
    for i in range(width, len(series)):
        loc = series.index[i - width: i + 1]
        output[series.index[i]] = float(np.dot(w, series.loc[loc]))
    return pd.Series(output, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class ForexDataPipeline:
    """
    End-to-end preprocessing pipeline for forex tick data.

    Usage
    -----
        pipeline = ForexDataPipeline(bar_freq="1min")
        bars = pipeline.run(tick_df)
        # bars is a clean, stationary OHLCV DataFrame ready for feature engineering
    """

    def __init__(
        self,
        bar_freq: str = "1min",
        frac_diff_order: float = 0.4,
        apply_frac_diff: bool = True,
        session_filter: bool = True,
        session_start_utc: str = "07:00",
        session_end_utc: str = "21:00",
    ):
        self.bar_freq = bar_freq
        self.frac_diff_order = frac_diff_order
        self.apply_frac_diff = apply_frac_diff
        self.session_filter = session_filter
        self.session_start = pd.to_datetime(session_start_utc).time()
        self.session_end = pd.to_datetime(session_end_utc).time()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing: resample → session filter → frac diff → clean."""
        print(f"[Pipeline] Raw tick rows: {len(df):,}")

        # Step 1: Resample to bars
        bars = resample_to_bars(df, freq=self.bar_freq)
        print(f"[Pipeline] Bars after resampling ({self.bar_freq}): {len(bars):,}")

        # Step 2: Session filter (avoid flat weekend / low-liquidity markets)
        if self.session_filter:
            times = bars.index.time
            mask = (times >= self.session_start) & (times <= self.session_end)
            # Also remove weekends
            mask = mask & (bars.index.dayofweek < 5)
            bars = bars[mask]
            print(f"[Pipeline] Bars after session filter: {len(bars):,}")

        # Step 3: Fractional differentiation on close price
        if self.apply_frac_diff:
            bars["close_ffd"] = fracDiff_FFD(bars["close"], d=self.frac_diff_order)
            print(f"[Pipeline] Fractional diff applied (d={self.frac_diff_order})")

        # Step 4: Drop NaN rows introduced by differencing
        bars = bars.dropna()
        print(f"[Pipeline] Final bar count after cleaning: {len(bars):,}")

        return bars

    def train_test_split(
        self,
        bars: pd.DataFrame,
        train_ratio: float = 0.7,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simple chronological split (no shuffling — time series!)."""
        split_idx = int(len(bars) * train_ratio)
        return bars.iloc[:split_idx], bars.iloc[split_idx:]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-START HELPER
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate(filepath: Optional[str] = None, n_rows: int = 50_000) -> pd.DataFrame:
    """
    Load real tick data if filepath provided, otherwise generate synthetic data.
    Ideal for rapid prototyping before connecting a live broker feed.
    """
    if filepath and Path(filepath).exists():
        print(f"[DataLoader] Loading real data from {filepath}")
        return load_tick_data(filepath)
    else:
        print(f"[DataLoader] Generating {n_rows:,} synthetic EUR/USD ticks for development")
        return generate_synthetic_tick_data(n_rows=n_rows)


if __name__ == "__main__":
    # Smoke test
    ticks = load_or_generate(n_rows=10_000)
    print(f"\nSample tick data:\n{ticks.head()}")
    print(f"\nData types:\n{ticks.dtypes}")

    pipeline = ForexDataPipeline(bar_freq="5min")
    bars = pipeline.run(ticks)
    train, test = pipeline.train_test_split(bars)
    print(f"\nTrain: {len(train):,} bars | Test: {len(test):,} bars")
