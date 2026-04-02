"""
Load cross-asset market series (commodities, yields, risk proxies) for feature engineering.

Provider order:
  1) Stooq public CSV endpoint
  2) Yahoo Finance (yfinance) fallback
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


STOOQ_SYMBOLS: Dict[str, List[str]] = {
    "WTI": ["cl.f", "cl"],
    "GOLD": ["xauusd", "gc.f"],
    "COPPER": ["hg.f", "hg"],
    "DXY": ["usdidx", "dx.f"],
    "SPX": ["^spx", "spx"],
    "NASDAQ100": ["^ndq", "ndq"],
    "VIX": ["^vix", "vix"],
    "US10Y": ["10usy.b", "us10y"],
    "US2Y": ["2usy.b", "us2y"],
    "DE10Y": ["10dey.b", "de10y"],
    "BTC": ["btc.v"],
}

YAHOO_SYMBOLS: Dict[str, str] = {
    "WTI": "CL=F",
    "GOLD": "GC=F",
    "COPPER": "HG=F",
    "DXY": "DX-Y.NYB",
    "SPX": "^GSPC",
    "NASDAQ100": "^NDX",
    "VIX": "^VIX",
    "US10Y": "^TNX",
    "US2Y": "^IRX",
    "DE10Y": "^TNX",  # fallback proxy if German 10Y is unavailable
    "BTC": "BTC-USD",
}


def _stooq_url(symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={symbol}&i=d"


def _read_stooq_daily(symbol: str) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(_stooq_url(symbol))
    except Exception:
        return None
    if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return None
    if str(df.iloc[0]["Date"]).lower().startswith("no_data"):
        return None
    s = pd.to_numeric(df["Close"], errors="coerce")
    idx = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    out = pd.Series(s.values, index=idx, name=symbol).dropna()
    if out.empty:
        return None
    return out.sort_index()


def _read_yahoo_daily(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        df = yf.download(
            symbol,
            start=str(pd.Timestamp(start).date()),
            end=str(end_ts.date()),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if col is None:
        return None
    col_data = df[col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    s = pd.to_numeric(col_data, errors="coerce")
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    out = pd.Series(s.values, index=idx, name=symbol).dropna()
    if out.empty:
        return None
    return out.sort_index()


def load_cross_asset_panel(
    start: str,
    end: str,
    cache_dir: str,
    source: str = "auto",
) -> Dict[str, pd.Series]:
    """
    Return dict of asset->price series in UTC index, clipped to [start, end].
    """
    source = (source or "auto").strip().lower()
    if source not in {"auto", "stooq", "yahoo"}:
        return {}

    out: Dict[str, pd.Series] = {}
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    for asset, candidates in STOOQ_SYMBOLS.items():
        ser = None
        provider_order = ["stooq", "yahoo"] if source == "auto" else [source]
        for provider in provider_order:
            if ser is not None and not ser.empty:
                break
            if provider == "stooq":
                for sym in candidates:
                    cache_path = cdir / f"{asset}_stooq_{sym.replace('^', 'idx')}.csv"
                    if cache_path.exists():
                        try:
                            cached = pd.read_csv(cache_path)
                            idx = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
                            vals = pd.to_numeric(cached["value"], errors="coerce")
                            ser = pd.Series(vals.values, index=idx, name=asset).dropna()
                        except Exception:
                            ser = None
                    if ser is None or ser.empty:
                        ser = _read_stooq_daily(sym)
                        if ser is not None and not ser.empty:
                            try:
                                pd.DataFrame({
                                    "timestamp": ser.index.astype(str),
                                    "value": ser.values,
                                }).to_csv(cache_path, index=False)
                            except Exception:
                                pass
                    if ser is not None and not ser.empty:
                        break
            elif provider == "yahoo":
                ysym = YAHOO_SYMBOLS.get(asset)
                if ysym:
                    cache_path = cdir / f"{asset}_yahoo_{ysym.replace('^', 'idx').replace('=', '_')}.csv"
                    if cache_path.exists():
                        try:
                            cached = pd.read_csv(cache_path)
                            idx = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
                            vals = pd.to_numeric(cached["value"], errors="coerce")
                            ser = pd.Series(vals.values, index=idx, name=asset).dropna()
                        except Exception:
                            ser = None
                    if ser is None or ser.empty:
                        ser = _read_yahoo_daily(ysym, start, end)
                        if ser is not None and not ser.empty:
                            try:
                                pd.DataFrame({
                                    "timestamp": ser.index.astype(str),
                                    "value": ser.values,
                                }).to_csv(cache_path, index=False)
                            except Exception:
                                pass

        if ser is None or ser.empty:
            continue
        clip = ser[(ser.index >= start_ts) & (ser.index <= end_ts)]
        if clip.empty:
            clip = ser
        out[asset] = clip

    return out

