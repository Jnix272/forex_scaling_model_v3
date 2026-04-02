"""
data/sources.py
================
Production data connectors for all three specified sources:

  1. DukascopyLoader   — Free tick data (EUR/USD, GBP/USD, 2003–present)
  2. TickDataSuite     — Paid Ducascopy-based data with spread reconstruction
  3. LMAXLoader        — Institutional LMAX Exchange Level-2 tick feed

All loaders produce the same output schema:
  DataFrame with UTC DatetimeIndex and columns:
    bid, ask, mid, spread, volume, pair, source

This unified schema means any loader can drop straight into the
feature engineering pipeline with zero downstream changes.

Dukascopy format notes
-----------------------
Raw files are LZMA-compressed binaries at:
  https://datafeed.dukascopy.com/datafeed/{PAIR}/{YEAR}/{MONTH:02d}/{DAY:02d}/{HOUR:02d}h_ticks.bi5

Each .bi5 file covers exactly one hour of ticks.
Binary format per tick (20 bytes):
  milliseconds_offset : uint32 (ms since start of hour)
  ask_price_scaled    : uint32 (price × 100000 for JPY pairs, × 10000 others)
  bid_price_scaled    : uint32
  ask_volume          : float32
  bid_volume          : float32

LMAX format notes
------------------
LMAX provides FIX 4.4 and REST API access. Free historical data is
available at: https://www.lmax.com/exchange/market-data
Institutional feed requires LMAX brokerage account.
The REST endpoint returns per-bar OHLCV + spread data which we
reconstruct to tick resolution using the Ask/Bid bar prices.
"""

import io
import os
import asyncio
import aiohttp
import struct
import time
import lzma
import warnings
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional, List, Dict, Tuple, Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SCHEMA ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

TICK_COLUMNS = ["bid", "ask", "mid", "spread", "volume", "pair", "source"]
PIP_SIZES    = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
                "AUDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001}


def _enforce_schema(df: pd.DataFrame, pair: str, source: str) -> pd.DataFrame:
    """
    Ensure output always has the unified tick schema.
    Adds mid and spread if missing, tags pair and source.
    Enforces UTC index named 'timestamp'.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    if "spread" not in df.columns:
        df["spread"] = df["ask"] - df["bid"]
    if "volume" not in df.columns:
        df["volume"] = 1
    df["pair"]   = pair
    df["source"] = source

    # Golden Rule: drop rows where bid >= ask (data corruption)
    bad = df["bid"] >= df["ask"]
    if bad.any():
        print(f"  [Schema] Dropped {bad.sum()} rows with bid >= ask")
        df = df[~bad]

    return df[TICK_COLUMNS].sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# 1. DUKASCOPY LOADER
# ─────────────────────────────────────────────────────────────────────────────

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# Dukascopy uses pair codes like "EURUSD" but the URL uses uppercase pair without slash
DUKA_PAIR_MAP = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD", "EURGBP": "EURGBP", "EURJPY": "EURJPY",
}

# Local cache roots (next to this package: data/raw/…)
_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DUKASCOPY_CACHE_DIR = str(_DATA_DIR / "raw" / "dukascopy")
DEFAULT_TDS_DATA_DIR = str(_DATA_DIR / "raw" / "tickdatasuite")
DEFAULT_LMAX_DATA_DIR = str(_DATA_DIR / "raw" / "lmax")

# Point value for price scaling (non-JPY: ×10000, JPY: ×1000)
DUKA_POINT = {
    "USDJPY": 1000, "EURJPY": 1000, "GBPJPY": 1000,
}


def _parse_bi5_hour(raw_bytes: bytes, dt_hour: datetime, pair: str) -> pd.DataFrame:
    """
    Parse one Dukascopy .bi5 hour file into a DataFrame.
    Binary layout: 20 bytes per tick.
      [0:4]  ms offset from start of hour (uint32 big-endian)
      [4:8]  ask price scaled (uint32 big-endian)
      [8:12] bid price scaled (uint32 big-endian)
      [12:16] ask volume (float32 big-endian)
      [16:20] bid volume (float32 big-endian)
    """
    if not raw_bytes:
        return pd.DataFrame()

    point = DUKA_POINT.get(pair, 100000)
    n     = len(raw_bytes) // 20
    if n == 0:
        return pd.DataFrame()

    ticks = np.frombuffer(raw_bytes, dtype=">u4,>u4,>u4,>f4,>f4")[:n]
    ms_offsets  = ticks["f0"].astype(np.int64)
    ask_scaled  = ticks["f1"].astype(np.float64)
    bid_scaled  = ticks["f2"].astype(np.float64)
    ask_vol     = ticks["f3"].astype(np.float64)
    bid_vol     = ticks["f4"].astype(np.float64)

    ask = ask_scaled / point
    bid = bid_scaled / point
    vol = ((ask_vol + bid_vol) / 2.0).round(2)

    epoch_ms = int(dt_hour.timestamp() * 1000) + ms_offsets
    idx = pd.to_datetime(epoch_ms, unit="ms", utc=True)

    return pd.DataFrame({
        "bid":    bid,
        "ask":    ask,
        "volume": vol,
    }, index=idx)


class DukascopyLoader:
    """
    High-speed asynchronous loader for Dukascopy tick data.

    Uses asyncio + aiohttp for efficient parallel downloads and connection
    pooling. Automatically caches hour-files to disk as Parquet.

    Usage
    -----
        loader = DukascopyLoader(concurrency=128)
        df = loader.load("EURUSD", start="2024-01-01", end="2024-01-31")
    """

    def __init__(
        self,
        cache_dir:    str  = DEFAULT_DUKASCOPY_CACHE_DIR,
        request_delay: float = 0.0,
        max_retries:   int  = 5,
        verbose:       bool = True,
        concurrency:   int  = 60,     # sweet spot: fast without triggering 429s
    ):
        self.cache_dir    = Path(cache_dir)
        self.delay        = request_delay
        self.max_retries  = max_retries
        self.verbose      = verbose
        self.concurrency  = concurrency
        self._print_lock  = Lock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, pair: str, dt: datetime) -> Path:
        return (self.cache_dir / pair /
                f"{dt.year}" / f"{dt.month:02d}" /
                f"{dt.day:02d}_{dt.hour:02d}.parquet")

    async def _fetch_hour_async(
        self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
        pair: str, dt: datetime,
    ) -> Optional[bytes]:
        """Download one hour of .bi5 data. Releases the semaphore between retries."""
        url = DUKASCOPY_URL.format(
            pair  = DUKA_PAIR_MAP.get(pair, pair),
            year  = dt.year,
            month = dt.month - 1,
            day   = dt.day,
            hour  = dt.hour,
        )
        req_timeout = aiohttp.ClientTimeout(total=15, sock_connect=8, sock_read=12)

        for attempt in range(self.max_retries):
            async with semaphore:
                try:
                    async with session.get(url, timeout=req_timeout) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            return data
                        elif resp.status == 404:
                            return None
                        elif resp.status == 429:
                            pass  # release semaphore, then backoff below
                        else:
                            pass
                except (asyncio.TimeoutError, aiohttp.ClientError):
                    pass
            # Backoff happens OUTSIDE the semaphore so slots stay free
            backoff = min(15, 2 ** attempt) if attempt > 0 else 0.5
            await asyncio.sleep(backoff)
        return None

    async def _load_hour_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        pair: str,
        dt: datetime,
        executor: ThreadPoolExecutor,
    ) -> pd.DataFrame:
        """Load one hour — from disk cache or network fetch."""
        cache_file = self._cache_path(pair, dt)

        if cache_file.exists():
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(executor, pd.read_parquet, cache_file)
            except Exception:
                pass

        try:
            raw_lzma = await asyncio.wait_for(
                self._fetch_hour_async(session, semaphore, pair, dt),
                timeout=90,
            )
        except asyncio.TimeoutError:
            return pd.DataFrame()

        if raw_lzma is None:
            return pd.DataFrame()

        loop = asyncio.get_running_loop()
        def _process():
            try:
                raw_bytes = lzma.decompress(raw_lzma)
                df = _parse_bi5_hour(raw_bytes, dt, pair)
                if not df.empty:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(cache_file)
                return df
            except Exception:
                return pd.DataFrame()

        return await loop.run_in_executor(executor, _process)

    def load(
        self,
        pair:  str,
        start: str,
        end:   str,
        hours: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Synchronous wrapper for modern async loading.
        Maintains compatibility with existing training scripts.
        """
        pair     = pair.upper().replace("/", "")
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        hours    = hours or list(range(24))

        tasks_dt: List[datetime] = []
        current = start_dt
        while current <= end_dt:
            for h in hours:
                tasks_dt.append(current.replace(hour=h))
            current += timedelta(days=1)

        if not tasks_dt:
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        if self.verbose:
            print(f"[Dukascopy] Async Loading {pair} | {start} → {end} | "
                  f"{len(tasks_dt)} hours | Concurrency: {self.concurrency}")

        return asyncio.run(self._load_all_async(pair, tasks_dt))

    async def _load_all_async(self, pair: str, datetimes: List[datetime]) -> pd.DataFrame:
        """Core async orchestrator — fires all tasks at once, semaphore caps in-flight."""
        semaphore = asyncio.Semaphore(self.concurrency)
        conn_limit = self.concurrency + 10
        connector = aiohttp.TCPConnector(
            limit=conn_limit, limit_per_host=conn_limit, ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        session_timeout = aiohttp.ClientTimeout(total=None, connect=15)
        executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4))

        total = len(datetimes)
        completed = 0
        results: List[Optional[pd.DataFrame]] = [None] * total
        log_step = max(total // 40, 25)
        t0 = asyncio.get_event_loop().time()

        async with aiohttp.ClientSession(
            timeout=session_timeout, connector=connector,
            headers={"User-Agent": "ForexScaler/2.0"},
        ) as session:

            async def _wrapped(idx: int, dt: datetime):
                return idx, await self._load_hour_async(session, semaphore, pair, dt, executor)

            tasks = [asyncio.ensure_future(_wrapped(i, dt)) for i, dt in enumerate(datetimes)]

            for coro in asyncio.as_completed(tasks):
                idx, df = await coro
                results[idx] = df
                completed += 1
                if self.verbose and (completed % log_step == 0 or completed == total):
                    elapsed = asyncio.get_event_loop().time() - t0
                    rate = completed / max(elapsed, 0.1)
                    eta = (total - completed) / max(rate, 0.01)
                    print(f"  {completed*100//total:3d}% | {completed}/{total} "
                          f"| {rate:.0f} files/s | ETA {eta:.0f}s")

        executor.shutdown(wait=False)

        non_empty = [df for df in results if df is not None and not df.empty]
        if not non_empty:
            return pd.DataFrame(columns=pd.Index(TICK_COLUMNS))

        non_empty.sort(key=lambda df: df.index[0])
        combined = pd.concat(non_empty, copy=False)
        return _enforce_schema(combined, pair, "dukascopy")

    def load_multiple(
        self,
        pairs: List[str],
        start: str,
        end:   str,
        hours: Optional[List[int]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple pairs concurrently (all pairs share one event loop)."""
        return asyncio.run(self._load_multiple_async(pairs, start, end, hours))

    async def _load_multiple_async(
        self, pairs: List[str], start: str, end: str, hours: Optional[List[int]]
    ) -> Dict[str, pd.DataFrame]:
        hours = hours or list(range(24))
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

        async def _one_pair(pair: str) -> tuple:
            pair = pair.upper().replace("/", "")
            tasks_dt: List[datetime] = []
            cur = start_dt
            while cur <= end_dt:
                for h in hours:
                    tasks_dt.append(cur.replace(hour=h))
                cur += timedelta(days=1)
            if not tasks_dt:
                return pair, pd.DataFrame(columns=pd.Index(TICK_COLUMNS))
            return pair, await self._load_all_async(pair, tasks_dt)

        pair_results = await asyncio.gather(*[_one_pair(p) for p in pairs])
        return {p: df for p, df in pair_results}

    def load_eurusd_gbpusd(
        self,
        start: str,
        end:   str,
        session_only: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience: load EUR/USD and GBP/USD (the two most liquid pairs).
        session_only=True loads only London+NY overlap hours (13–17 UTC)
        to reduce file count by 79% while keeping the most tradeable session.
        """
        hours = list(range(7, 18)) if session_only else None
        return self.load_multiple(["EURUSD", "GBPUSD"], start, end, hours)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TICK DATA SUITE LOADER
# ─────────────────────────────────────────────────────────────────────────────

class TickDataSuiteLoader:
    """
    Loader for data processed by Tick Data Suite (TDS).

    Tick Data Suite is a MetaTrader plugin that:
      - Downloads Dukascopy raw tick data
      - Reconstructs real bid/ask spreads (Dukascopy only provides one-sided
        spread in its free feed — TDS uses a proprietary spread model)
      - Exports to CSV, FXT, or HSTv401 formats for MT4/MT5 backtesting

    TDS export formats supported here:
      - Standard CSV  : timestamp, open, high, low, close, tickvol, vol, spread
      - Tick CSV      : date, time, bid, ask, volume
      - Parquet       : pre-converted for faster loading

    How to export from TDS:
      1. Open TDS → Data → Export
      2. Choose "Tick data" format
      3. Set date range and pair
      4. Export as CSV with semicolon delimiter
      5. Point data_dir to the export folder

    TDS CSV tick format (semicolon-delimited):
      20240101 00:00:00.123;1.10500;1.10502;0.75
      (datetime;bid;ask;volume)
    """

    # TDS uses semicolons and no header by default
    TDS_DTYPES = {"bid": np.float64, "ask": np.float64, "volume": np.float64}

    def __init__(
        self,
        data_dir: str = DEFAULT_TDS_DATA_DIR,
        verbose:  bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.verbose  = verbose

    def _detect_format(self, filepath: Path) -> str:
        """Detect whether file is TDS tick CSV, TDS bar CSV, or Parquet."""
        if filepath.suffix == ".parquet":
            return "parquet"
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        # Tick CSV: date and time in separate columns or combined
        if ";" in first:
            parts = first.split(";")
            if len(parts) >= 4:
                return "tick_csv_semicolon"
        if "," in first:
            parts = first.split(",")
            if len(parts) >= 4:
                return "tick_csv_comma"
        return "bar_csv"

    def _load_tick_csv(self, filepath: Path, delimiter: str = ";") -> pd.DataFrame:
        """
        Parse TDS tick CSV export.
        Handles both:
          (a) "20240101 00:00:00.123;bid;ask;vol"
          (b) "2024.01.01,00:00:00,bid,ask,vol"  (MT4 style)
        """
        # Try to detect column count first
        with open(filepath, "r") as f:
            sample = f.readline().strip()
        parts = sample.split(delimiter)

        if len(parts) == 4:
            # Combined datetime
            names = ["datetime", "bid", "ask", "volume"]
            df = pd.read_csv(
                str(filepath),
                sep=delimiter,
                header=None,
                names=names,
                dtype={"bid": float, "ask": float, "volume": float},
                iterator=False,
            )
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.dropna(subset=("datetime",)).set_index("datetime")

        elif len(parts) == 5:
            # Separate date and time columns
            names = ["date", "time", "bid", "ask", "volume"]
            df = pd.read_csv(
                str(filepath),
                sep=delimiter,
                header=None,
                names=names,
                dtype={"bid": float, "ask": float, "volume": float},
                iterator=False,
            )
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                utc=True, errors="coerce"
            )
            df = df.dropna(subset=("datetime",)).set_index("datetime")

        else:
            raise ValueError(f"Unexpected TDS CSV column count: {len(parts)} in {filepath}")

        return df[["bid", "ask", "volume"]]

    def load_file(self, filepath: str, pair: str) -> pd.DataFrame:
        """Load a single TDS export file."""
        fp  = Path(filepath)
        fmt = self._detect_format(fp)

        if fmt == "parquet":
            df = pd.read_parquet(fp)
        elif fmt == "tick_csv_semicolon":
            df = self._load_tick_csv(fp, delimiter=";")
        elif fmt == "tick_csv_comma":
            df = self._load_tick_csv(fp, delimiter=",")
        else:
            raise ValueError(f"Unsupported TDS format: {fmt}  ({filepath})")

        df = _enforce_schema(df, pair.upper().replace("/",""), "tickdatasuite")
        if self.verbose:
            print(f"[TDS] Loaded {fp.name} | {len(df):,} ticks | "
                  f"Spread: {df['spread'].mean()*10000:.2f} pips avg")
        return df

    def load_directory(
        self,
        pair:  str,
        start: Optional[str] = None,
        end:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load all TDS files for a pair from data_dir.
        Expects files named: EURUSD_2024*.csv  or  EURUSD/*.parquet

        The directory layout TDS exports to by default:
          /data_dir/EURUSD/2024_01.csv
          /data_dir/EURUSD/2024_02.csv
          ...
        """
        pair_clean = pair.upper().replace("/", "")
        search_dirs = [
            self.data_dir / pair_clean,
            self.data_dir,
        ]
        files = []
        for d in search_dirs:
            if d.exists():
                files += list(d.glob(f"*{pair_clean}*.csv"))
                files += list(d.glob(f"*{pair_clean}*.parquet"))
                files += list(d.glob("*.csv"))
                files += list(d.glob("*.parquet"))

        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(
                f"No TDS files found for {pair} in {self.data_dir}\n"
                f"Export tick data from Tick Data Suite and place in:\n"
                f"  {self.data_dir / pair_clean}/"
            )

        frames = []
        for f in files:
            try:
                frames.append(self.load_file(str(f), pair))
            except Exception as e:
                print(f"  [TDS] Skipped {f.name}: {e}")

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # Date filter
        if start: combined = combined[combined.index >= pd.Timestamp(start, tz="UTC")]
        if end:   combined = combined[combined.index <= pd.Timestamp(end,   tz="UTC")]

        if self.verbose:
            print(f"[TDS] {pair_clean}: {len(combined):,} ticks | "
                  f"{combined.index[0]} → {combined.index[-1]}")
        return combined

    def convert_to_parquet(self, pair: str):
        """
        Convert all CSV files for a pair to Parquet for faster future loads.
        Run once after exporting from TDS.
        """
        pair_clean = pair.upper().replace("/", "")
        csv_files  = list((self.data_dir / pair_clean).glob("*.csv"))
        for f in csv_files:
            out = f.with_suffix(".parquet")
            if out.exists():
                continue
            try:
                df = self.load_file(str(f), pair)
                df.to_parquet(out)
                print(f"  Converted {f.name} → {out.name}")
            except Exception as e:
                print(f"  Skip {f.name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LMAX EXCHANGE LOADER
# ─────────────────────────────────────────────────────────────────────────────

LMAX_REST_BASE = "https://trade.lmax.com"
LMAX_HIST_BASE = "https://www.lmax.com/exchange/market-data"

# LMAX instrument IDs for major FX pairs
LMAX_INSTRUMENTS = {
    "EURUSD": "4001",   "GBPUSD": "4002",   "USDJPY": "4003",
    "AUDUSD": "4004",   "USDCAD": "4005",   "EURGBP": "4006",
    "USDCHF": "4007",   "NZDUSD": "4009",   "EURJPY": "4012",
}


class LMAXLoader:
    """
    Loader for LMAX Exchange tick / order book data.

    LMAX is an institutional FX ECN with:
      - No last-look execution (true ECN)
      - Tight raw spreads (0.1–0.3 pips EUR/USD typical)
      - Level-2 order book depth (10 price levels)
      - FIX 4.4 and REST API access

    Two modes
    ----------
    1. Historical CSV (free, no account needed):
       Download from https://www.lmax.com/exchange/market-data
       1-minute OHLCV + spread per pair, 2010–present.

    2. Live REST API (requires LMAX brokerage account):
       Real-time L1 bid/ask + L2 order book depth.
       Set LMAX_USERNAME and LMAX_PASSWORD environment variables.

    Historical data format (LMAX CSV):
      DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose,Volume

    Live API returns JSON order book snapshots at ~100ms resolution.
    """

    def __init__(
        self,
        data_dir:    str  = DEFAULT_LMAX_DATA_DIR,
        username:    Optional[str] = None,
        password:    Optional[str] = None,
        verbose:     bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.username = username or os.getenv("LMAX_USERNAME")
        self.password = password or os.getenv("LMAX_PASSWORD")
        self.verbose  = verbose
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session_token: Optional[str] = None

    # ── Historical (free) ─────────────────────────────────────────────────────

    def load_historical_csv(
        self,
        pair:  str,
        filepath: Optional[str] = None,
        start: Optional[str] = None,
        end:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load LMAX historical 1-minute OHLCV CSV data.

        Download from: https://www.lmax.com/exchange/market-data
        Select pair → Download CSV → place in data_dir/LMAX/{pair}/

        CSV format expected:
          DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose,Volume
        """
        pair_clean = pair.upper().replace("/", "")

        if filepath:
            files = [Path(filepath)]
        else:
            search_dir = self.data_dir / pair_clean
            files = sorted(search_dir.glob("*.csv")) + sorted(search_dir.glob("*.parquet"))

        if not files:
            raise FileNotFoundError(
                f"No LMAX data files found for {pair} in {self.data_dir}/{pair_clean}/\n"
                f"Download from: https://www.lmax.com/exchange/market-data\n"
                f"Place CSVs in: {self.data_dir}/{pair_clean}/"
            )

        frames = []
        for f in files:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
            else:
                df = self._parse_lmax_csv(f)
            frames.append(df)

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        if start: combined = combined[combined.index >= pd.Timestamp(start, tz="UTC")]
        if end:   combined = combined[combined.index <= pd.Timestamp(end,   tz="UTC")]

        combined = _enforce_schema(combined, pair_clean, "lmax_historical")
        if self.verbose:
            print(f"[LMAX] Historical {pair_clean}: {len(combined):,} bars | "
                  f"Spread: {combined['spread'].mean()*10000:.2f} pips avg | "
                  f"{combined.index[0]} → {combined.index[-1]}")
        return combined

    def _parse_lmax_csv(self, filepath: Path) -> pd.DataFrame:
        """Parse LMAX historical CSV into bid/ask OHLCV bars."""
        # Try with and without header
        try:
            df = pd.read_csv(
                str(filepath),
                parse_dates=True,
                index_col=0,
                iterator=False,
            )
        except Exception:
            df = pd.read_csv(
                str(filepath),
                header=None,
                parse_dates=True,
                index_col=0,
                names=["dt", "bo", "bh", "bl", "bc", "ao", "ah", "al", "ac", "vol"],
                iterator=False,
            )

        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.dropna()

        # Normalize column names (case-insensitive)
        df.columns = df.columns.str.lower().str.strip()
        col_map = {}
        for c in df.columns:
            if "bidclose"  in c or c == "bc":  col_map[c] = "bid"
            elif "askclose" in c or c == "ac": col_map[c] = "ask"
            elif "volume"   in c or c == "vol":col_map[c] = "volume"
        df = df.rename(columns=col_map)

        # Fall back to synthesizing bid/ask from BidHigh/Low if BidClose missing
        if "bid" not in df.columns:
            bh_col = next((c for c in df.columns if "bidhi" in c or c=="bh"), None)
            bl_col = next((c for c in df.columns if "bidlo" in c or c=="bl"), None)
            if bh_col and bl_col:
                df["bid"] = (df[bh_col] + df[bl_col]) / 2.0
        if "ask" not in df.columns:
            ah_col = next((c for c in df.columns if "askhi" in c or c=="ah"), None)
            al_col = next((c for c in df.columns if "asklo" in c or c=="al"), None)
            if ah_col and al_col:
                df["ask"] = (df[ah_col] + df[al_col]) / 2.0

        return df[["bid", "ask"] + (["volume"] if "volume" in df.columns else [])]

    # ── Live REST API (requires LMAX account) ─────────────────────────────────

    def login(self) -> bool:
        """Authenticate with LMAX REST API. Returns True on success."""
        if not self.username or not self.password:
            print("[LMAX] No credentials. Set LMAX_USERNAME and LMAX_PASSWORD env vars.")
            return False
        try:
            import json
            payload = json.dumps({"username": self.username,
                                   "password": self.password}).encode()
            req = Request(
                f"{LMAX_REST_BASE}/public/security/authenticate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                self._session_token = data.get("sessionId") or data.get("token")
                if self._session_token:
                    print(f"[LMAX] Authenticated | session: {self._session_token[:8]}...")
                    return True
        except Exception as e:
            print(f"[LMAX] Login failed: {e}")
        return False

    def fetch_orderbook(self, pair: str) -> Optional[dict]:
        """
        Fetch live L2 order book snapshot for a pair.
        Requires active session (call login() first).

        Returns dict with keys: bid_levels, ask_levels, timestamp
        Each level: {"price": float, "quantity": float}
        """
        if not self._session_token:
            if not self.login():
                return None
        try:
            import json
            instr_id = LMAX_INSTRUMENTS.get(pair.upper().replace("/",""), "4001")
            req = Request(
                f"{LMAX_REST_BASE}/public/orderbook/{instr_id}/data",
                headers={
                    "Session-Id": self._session_token,
                    "Content-Type": "application/json",
                },
            )
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return {
                    "timestamp":  pd.Timestamp.utcnow(),
                    "pair":       pair,
                    "bid_levels": data.get("bids", [])[:10],
                    "ask_levels": data.get("asks", [])[:10],
                    "best_bid":   float(data["bids"][0]["price"]) if data.get("bids") else None,
                    "best_ask":   float(data["asks"][0]["price"]) if data.get("asks") else None,
                }
        except Exception as e:
            print(f"[LMAX] Order book error: {e}")
            return None

    def stream_ticks(
        self,
        pair:      str,
        n_ticks:   int = 1000,
        callback = None,
    ) -> pd.DataFrame:
        """
        Poll the LMAX REST API for live tick data.
        In production, replace with FIX 4.4 streaming (lower latency).

        callback: optional function(tick_dict) called on each new tick.
        """
        if not self._session_token:
            if not self.login():
                return pd.DataFrame()

        ticks = []
        print(f"[LMAX] Streaming {n_ticks} ticks for {pair}...")
        while len(ticks) < n_ticks:
            ob = self.fetch_orderbook(pair)
            if ob and ob["best_bid"] and ob["best_ask"]:
                tick = {
                    "bid":    ob["best_bid"],
                    "ask":    ob["best_ask"],
                    "volume": 1,
                }
                ticks.append((ob["timestamp"], tick))
                if callback: callback(tick)
            time.sleep(0.1)   # 10 Hz polling

        idx = pd.DatetimeIndex([t[0] for t in ticks], tz="UTC")
        df  = pd.DataFrame([t[1] for t in ticks], index=idx)
        return _enforce_schema(df, pair.upper().replace("/",""), "lmax_live")

    def convert_to_parquet(self, pair: str):
        """Convert all LMAX CSVs to Parquet for fast future loads."""
        pair_clean = pair.upper().replace("/", "")
        for f in (self.data_dir / pair_clean).glob("*.csv"):
            out = f.with_suffix(".parquet")
            if out.exists():
                continue
            df = self._parse_lmax_csv(f)
            df.to_parquet(out)
            print(f"  {f.name} → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED DATA MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ForexDataManager:
    """
    Single interface across all three data sources.

    Priority order (best quality → lowest latency):
      LMAX live    → best spread accuracy, institutional quality
      TDS          → reconstructed spreads, best for backtesting
      Dukascopy    → free, vast history (2003+), good for research

    Usage
    -----
        mgr = ForexDataManager()

        # Research / backtest — free Dukascopy data
        df = mgr.load("EURUSD", start="2023-01-01", end="2023-12-31",
                       source="dukascopy")

        # Backtest with real spreads — TDS export
        df = mgr.load("EURUSD", start="2023-01-01", end="2023-12-31",
                       source="tds")

        # Live trading — LMAX ECN feed
        df = mgr.load("EURUSD", source="lmax_live", n_ticks=10000)

        # Auto-select best available source
        df = mgr.load("EURUSD", start="2023-01-01", end="2023-12-31")
    """

    SOURCE_PRIORITY = ["lmax_live", "tds", "dukascopy"]

    def __init__(
        self,
        dukascopy_dir: str = DEFAULT_DUKASCOPY_CACHE_DIR,
        tds_dir:       str = DEFAULT_TDS_DATA_DIR,
        lmax_dir:      str = DEFAULT_LMAX_DATA_DIR,
        lmax_username: Optional[str] = None,
        lmax_password: Optional[str] = None,
        verbose:       bool = True,
    ):
        self.duka = DukascopyLoader(dukascopy_dir, verbose=verbose)
        self.tds  = TickDataSuiteLoader(tds_dir, verbose=verbose)
        self.lmax = LMAXLoader(lmax_dir, lmax_username, lmax_password, verbose=verbose)
        self.verbose = verbose

    def load(
        self,
        pair:     str,
        source:   str  = "auto",
        start:    Optional[str] = None,
        end:      Optional[str] = None,
        n_ticks:  int  = 10_000,       # for live sources
        session_only: bool = True,     # London+NY hours only (Dukascopy)
    ) -> pd.DataFrame:

        pair = pair.upper().replace("/", "")

        if source == "dukascopy":
            if not start or not end:
                raise ValueError("Dukascopy requires start and end dates")
            hours = list(range(7, 18)) if session_only else None
            return self.duka.load(pair, start, end, hours)

        elif source == "tds":
            return self.tds.load_directory(pair, start, end)

        elif source == "lmax_historical":
            return self.lmax.load_historical_csv(pair, start=start, end=end)

        elif source == "lmax_live":
            return self.lmax.stream_ticks(pair, n_ticks=n_ticks)

        elif source == "auto":
            # Try sources in priority order
            errors = []
            for src in self.SOURCE_PRIORITY:
                try:
                    return self.load(pair, src, start, end, n_ticks, session_only)
                except (FileNotFoundError, ValueError) as e:
                    errors.append(f"  {src}: {e}")
            # All failed → synthetic fallback
            print(f"[DataManager] All sources failed for {pair}. Using synthetic data.")
            if self.verbose:
                for e in errors: print(e)
            from data.data_ingestion import generate_synthetic_tick_data
            return generate_synthetic_tick_data(n_rows=n_ticks)

        else:
            raise ValueError(f"Unknown source '{source}'. "
                             f"Options: dukascopy, tds, lmax_historical, lmax_live, auto")

    def load_all_pairs(
        self,
        source: str = "dukascopy",
        start:  Optional[str] = None,
        end:    Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load EUR/USD and GBP/USD — the two specified pairs."""
        return {
            "EURUSD": self.load("EURUSD", source, start, end),
            "GBPUSD": self.load("GBPUSD", source, start, end),
        }

    def quality_report(self, df: pd.DataFrame, pair: str) -> dict:
        """Compute data quality metrics for any loaded tick DataFrame."""
        if df.empty:
            return {"error": "empty DataFrame"}

        gaps = df.index.to_series().diff().dt.total_seconds().dropna()
        n_spread_anomalies = (df["spread"] <= 0).sum()
        n_bid_ask_inversion = (df["bid"] >= df["ask"]).sum()

        report = {
            "pair":             pair,
            "source":           df["source"].iloc[0] if "source" in df.columns else "unknown",
            "n_ticks":          len(df),
            "date_range":       f"{df.index[0]} → {df.index[-1]}",
            "avg_spread_pips":  round(df["spread"].mean() / PIP_SIZES.get(pair, 0.0001), 3),
            "min_spread_pips":  round(df["spread"].min()  / PIP_SIZES.get(pair, 0.0001), 3),
            "max_spread_pips":  round(df["spread"].max()  / PIP_SIZES.get(pair, 0.0001), 3),
            "avg_gap_seconds":  round(gaps.mean(), 3),
            "max_gap_seconds":  round(gaps.max(), 1),
            "n_gaps_over_1min": int((gaps > 60).sum()),
            "n_spread_anomalies": int(n_spread_anomalies),
            "n_bid_ask_inversions": int(n_bid_ask_inversion),
            "quality_score":    round(
                100 * (1 - (n_spread_anomalies + n_bid_ask_inversion) / max(len(df), 1)), 2
            ),
        }
        return report


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Data Sources — connection test")
    print("=" * 55)

    mgr = ForexDataManager(verbose=True)

    # 1. Dukascopy — try loading 3 days of EURUSD
    print("\n[1] Dukascopy free tick data")
    try:
        df_duka = mgr.load(
            "EURUSD", source="dukascopy",
            start="2024-01-02", end="2024-01-04",
            session_only=True,
        )
        print(f"    Ticks: {len(df_duka):,}")
        print(f"    Sample:\n{df_duka.head(3)}")
        rpt = mgr.quality_report(df_duka, "EURUSD")
        print(f"    Quality: {rpt['quality_score']}% | "
              f"Avg spread: {rpt['avg_spread_pips']} pips")
    except Exception as e:
        print(f"    Dukascopy error: {e}")
        print("    (Expected if no internet — offline / air-gapped)")

    # 2. TDS — check if export files exist
    print("\n[2] Tick Data Suite")
    tds_dir = Path(DEFAULT_TDS_DATA_DIR) / "EURUSD"
    if tds_dir.exists() and list(tds_dir.glob("*.csv")):
        try:
            df_tds = mgr.load("EURUSD", source="tds",
                               start="2024-01-01", end="2024-01-31")
            print(f"    Ticks: {len(df_tds):,}")
        except Exception as e:
            print(f"    TDS load error: {e}")
    else:
        print(f"    No TDS files found in {tds_dir}/")
        print("    Export tick data from Tick Data Suite → place CSVs there")

    # 3. LMAX — historical CSV check
    print("\n[3] LMAX Exchange")
    lmax_dir = Path(DEFAULT_LMAX_DATA_DIR) / "EURUSD"
    if lmax_dir.exists() and list(lmax_dir.glob("*.csv")):
        try:
            df_lmax = mgr.load("EURUSD", source="lmax_historical",
                                start="2024-01-01", end="2024-01-31")
            print(f"    Bars: {len(df_lmax):,}")
        except Exception as e:
            print(f"    LMAX load error: {e}")
    else:
        print(f"    No LMAX files found in {lmax_dir}/")
        print("    Download from: https://www.lmax.com/exchange/market-data")
        print("    For live feed: set LMAX_USERNAME + LMAX_PASSWORD env vars")

    print("\n[Schema] All sources produce the same columns:")
    print(f"    {TICK_COLUMNS}")
