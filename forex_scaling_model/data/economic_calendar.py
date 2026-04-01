"""
data/economic_calendar.py
==========================
Economic event features for Forex models.

Produces per-bar features:
  eco_minutes_to_next  — minutes until next High-impact event (clipped 0–1440)
  eco_minutes_since_last — minutes since last High-impact event
  eco_surprise_norm    — (actual − forecast) / |prior|, forward-filled after each release
  eco_release_flag     — 1.0 within news_buffer_minutes of any High-impact event

Data sources (in priority order):
  1. Live: Alpha Vantage "REAL_GDP", "CPI", "FEDERAL_FUNDS_RATE" endpoints —
     set AV_API_KEY env var (free tier, 25 req/day).
  2. Offline CSV cache in data/raw/eco_calendar/ (auto-populated on first live pull).
  3. Synthetic fallback: random event schedule matching density of real calendars.

Usage:
    from data.economic_calendar import EcoCalendarFeatureBuilder
    eco = EcoCalendarFeatureBuilder()
    df_features = eco.build(bars)   # bars: pd.DataFrame with DatetimeTZIndex (UTC)
"""

import os
import hashlib
import json
import warnings
from datetime import timezone
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────

HIGH_IMPACT_EVENTS = [
    "NFP", "NonFarm", "Non-Farm", "Payroll",
    "CPI", "Inflation", "Core CPI",
    "FOMC", "Fed", "Interest Rate Decision",
    "GDP", "Gross Domestic",
    "PMI", "Purchasing Managers",
    "ECB", "European Central",
    "BOE", "Bank of England",
    "BOJ", "Bank of Japan",
    "RBA", "Reserve Bank of Australia",
    "Retail Sales",
    "Unemployment", "Jobless",
]

CACHE_DIR = Path(os.getenv("ECO_CACHE_DIR",
                           str(Path(__file__).resolve().parent.parent / "data" / "raw" / "eco_calendar")))


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_high_impact(event_name: str) -> bool:
    """True if the event name matches any high-impact keyword."""
    name_l = str(event_name).lower()
    return any(kw.lower() in name_l for kw in HIGH_IMPACT_EVENTS)


def _to_utc(ts) -> pd.Timestamp:
    """Coerce any timestamp to UTC-aware."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _synthetic_calendar(start: pd.Timestamp, end: pd.Timestamp,
                         seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic economic calendar for testing / offline mode.
    Produces ~4–6 high-impact events per week, distributed across US / EU sessions.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(start.date(), end.date(), freq="D")
    events: List[Dict] = []

    event_menu = [
        ("NFP", 200e3, 185e3, 170e3),
        ("CPI YoY", 3.2, 3.0, 3.1),
        ("FOMC Rate Decision", 5.25, 5.25, 5.00),
        ("GDP QoQ", 2.4, 2.1, 1.9),
        ("Core PMI", 53.0, 52.5, 51.8),
        ("ECB Rate Decision", 4.0, 4.0, 3.75),
        ("BOE Rate Decision", 5.25, 5.25, 5.00),
        ("Retail Sales", 0.4, 0.3, 0.1),
        ("Unemployment Rate", 3.7, 3.8, 3.9),
    ]

    for day in days:
        if rng.random() < 0.45:          # ~45% of days have an event
            n_events = rng.integers(1, 3)
            for _ in range(n_events):
                ev_tmpl = event_menu[rng.integers(len(event_menu))]
                hour = rng.choice([8, 12, 13, 14, 15])  # NY/EU session hours
                minute = rng.choice([0, 30])
                ts = _to_utc(pd.Timestamp(day) + pd.Timedelta(hours=int(hour), minutes=int(minute)))
                actual   = float(ev_tmpl[1]) * (1 + rng.normal(0, 0.02))
                forecast = float(ev_tmpl[2]) * (1 + rng.normal(0, 0.01))
                prior    = float(ev_tmpl[3])
                events.append({
                    "datetime": ts,
                    "event":    ev_tmpl[0],
                    "actual":   round(actual, 4),
                    "forecast": round(forecast, 4),
                    "prior":    round(prior, 4),
                    "impact":   "High",
                })

    df = pd.DataFrame(events) if events else pd.DataFrame(
        columns=["datetime", "event", "actual", "forecast", "prior", "impact"])
    return df.sort_values("datetime").reset_index(drop=True)


def _load_alpha_vantage(av_key: str, start: pd.Timestamp,
                        end: pd.Timestamp) -> pd.DataFrame:
    """
    Pull economic indicator data from Alpha Vantage.
    Covers: REAL_GDP, CPI, FEDERAL_FUNDS_RATE, UNEMPLOYMENT, RETAIL_SALES.
    Free tier: 25 requests/day — results are cached locally.
    """
    import urllib.request, urllib.parse

    functions = [
        ("REAL_GDP",            "GDP"),
        ("CPI",                 "CPI"),
        ("FEDERAL_FUNDS_RATE",  "FOMC Rate Decision"),
        ("UNEMPLOYMENT",        "Unemployment Rate"),
        ("RETAIL_SALES",        "Retail Sales"),
    ]

    rows: List[Dict] = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for fn, label in functions:
        cache_path = CACHE_DIR / f"av_{fn}.json"
        data: Optional[dict] = None

        # Load from cache if fresh (< 24 h)
        if cache_path.exists():
            age_h = (pd.Timestamp.utcnow().timestamp() - cache_path.stat().st_mtime) / 3600
            if age_h < 24:
                try:
                    with open(cache_path) as f:
                        data = json.load(f)
                except Exception:
                    data = None

        if data is None:
            url = (f"https://www.alphavantage.co/query?"
                   f"function={fn}&interval=monthly&apikey={av_key}")
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    data = json.loads(r.read())
                with open(cache_path, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"[EcoCalendar] AV fetch failed for {fn}: {e} — using cache/synth")
                continue

        # Parse AV response
        for key in ("data", "Time Series (Monthly)"):
            if key in data:
                series = data[key]
                break
        else:
            continue

        if isinstance(series, list):
            for item in series:
                try:
                    ts = _to_utc(item.get("date", ""))
                    if not (start <= ts <= end):
                        continue
                    actual = float(item.get("value", 0))
                    rows.append({
                        "datetime": ts,
                        "event":    label,
                        "actual":   actual,
                        "forecast": actual * 0.99,   # AV doesn't give forecast
                        "prior":    actual * 0.98,
                        "impact":   "High",
                    })
                except Exception:
                    continue

    if not rows:
        return pd.DataFrame(columns=["datetime", "event", "actual", "forecast", "prior", "impact"])
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


# ── main class ───────────────────────────────────────────────────────────────

class EcoCalendarFeatureBuilder:
    """
    Builds economic calendar features aligned to a bar DataFrame.

    Parameters
    ----------
    news_buffer_minutes : int
        Bars within this window around a release get eco_release_flag=1.
    av_api_key : str or None
        Alpha Vantage API key. If None, falls back to synthetic calendar.
    use_synthetic : bool
        Force synthetic calendar (useful in tests / CI).
    """

    def __init__(
        self,
        news_buffer_minutes: int = 15,
        av_api_key: Optional[str] = None,
        use_synthetic: bool = False,
    ):
        self.buffer_min   = news_buffer_minutes
        self._av_key      = av_api_key or os.getenv("AV_API_KEY", "")
        self._use_synth   = use_synthetic or not self._av_key

    # ── public API ──────────────────────────────────────────────────────────

    def load_events(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return a DataFrame of high-impact economic events in [start, end]."""
        if self._use_synth:
            df = _synthetic_calendar(start, end)
        else:
            try:
                df = _load_alpha_vantage(self._av_key, start, end)
                if len(df) == 0:
                    df = _synthetic_calendar(start, end)
            except Exception as e:
                print(f"[EcoCalendar] Falling back to synthetic: {e}")
                df = _synthetic_calendar(start, end)

        # Filter to high-impact only
        df = df[df["impact"] == "High"].copy()
        df["datetime"] = df["datetime"].apply(_to_utc)
        return df.sort_values("datetime").reset_index(drop=True)

    def build(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame aligned to `bars.index` with columns:
          eco_minutes_to_next, eco_minutes_since_last,
          eco_surprise_norm, eco_release_flag
        """
        idx = bars.index
        if idx.tzinfo is None:
            idx = idx.tz_localize("UTC")

        start = idx[0]  - pd.Timedelta(days=7)    # load some history
        end   = idx[-1] + pd.Timedelta(days=1)

        events = self.load_events(start, end)

        # Initialise output
        out = pd.DataFrame(index=bars.index)
        out["eco_minutes_to_next"]    = 1440.0
        out["eco_minutes_since_last"] = 1440.0
        out["eco_surprise_norm"]      = 0.0
        out["eco_release_flag"]       = 0.0

        if len(events) == 0:
            return out

        ev_times   = events["datetime"].values            # numpy datetime64
        buf        = pd.Timedelta(minutes=self.buffer_min)
        bar_times  = pd.DatetimeIndex(bars.index)

        # Vectorised: distance from each bar to nearest future event
        for i, bt in enumerate(bar_times):
            future = events[events["datetime"] > bt]["datetime"]
            past   = events[events["datetime"] <= bt]["datetime"]

            delta_fwd = 1440.0
            delta_bk  = 1440.0

            if len(future):
                delta_fwd = (future.iloc[0] - bt).total_seconds() / 60
                out.iloc[i, out.columns.get_loc("eco_minutes_to_next")] = float(
                    np.clip(delta_fwd, 0, 1440))

            if len(past):
                last_ev   = past.iloc[-1]
                delta_bk  = (bt - last_ev).total_seconds() / 60
                out.iloc[i, out.columns.get_loc("eco_minutes_since_last")] = float(
                    np.clip(delta_bk, 0, 1440))

                # Surprise: find the event row
                ev_row = events[events["datetime"] == last_ev].iloc[-1]
                prior  = float(ev_row["prior"])
                if abs(prior) > 1e-9:
                    surprise = (float(ev_row["actual"]) - float(ev_row["forecast"])) / abs(prior)
                    out.iloc[i, out.columns.get_loc("eco_surprise_norm")] = float(
                        np.clip(surprise, -5, 5))

            # Release flag (true if within news_buffer_minutes of PAST or FUTURE event)
            if delta_bk <= self.buffer_min or delta_fwd <= self.buffer_min:
                out.iloc[i, out.columns.get_loc("eco_release_flag")] = 1.0

        return out


# ── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Economic Calendar — smoke test")
    idx = pd.date_range("2024-01-01", periods=500, freq="1min", tz="UTC")
    bars = pd.DataFrame({"close": np.random.randn(500).cumsum() + 1.085}, index=idx)

    eco = EcoCalendarFeatureBuilder(use_synthetic=True)
    feats = eco.build(bars)
    print(f"  Shape: {feats.shape}")
    print(f"  Release flags set: {int(feats['eco_release_flag'].sum())}")
    print(f"  Min to next (avg): {feats['eco_minutes_to_next'].mean():.0f} min")
    print(feats.head(3))
    print("OK ✓")
