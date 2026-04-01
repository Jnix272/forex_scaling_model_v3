"""
infrastructure/timescale_kafka.py
==================================
Two components:
  1. TimescaleDBStore  — write/read tick data via hypertables
  2. KafkaTickConsumer — stream live ticks into TimescaleDB in real-time

TimescaleDB is purpose-built for time-series: it chunks data by time,
auto-compresses old chunks, and runs time-series queries 10-100× faster
than plain PostgreSQL on large tick datasets.

Kafka decouples the data feed (broker API) from the storage layer,
meaning the model never blocks waiting for a database write.
"""

import os
import time
import json
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict, Generator
import numpy as np
import pandas as pd
import warnings

# ── Optional heavy dependencies — gracefully degraded ────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2 = True
except ImportError:
    PSYCOPG2 = False
    warnings.warn("psycopg2 not installed. pip install psycopg2-binary")

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA = True
except ImportError:
    KAFKA = False
    warnings.warn("kafka-python not installed. pip install kafka-python")


# ─────────────────────────────────────────────────────────────────────────────
# TIMESCALEDB STORE
# ─────────────────────────────────────────────────────────────────────────────

TICK_DDL = """
-- Hypertable for raw tick data (partitioned by time automatically)
CREATE TABLE IF NOT EXISTS {table} (
    time        TIMESTAMPTZ     NOT NULL,
    pair        TEXT            NOT NULL,
    bid         DOUBLE PRECISION NOT NULL,
    ask         DOUBLE PRECISION NOT NULL,
    mid         DOUBLE PRECISION GENERATED ALWAYS AS ((bid + ask) / 2.0) STORED,
    spread      DOUBLE PRECISION GENERATED ALWAYS AS (ask - bid) STORED,
    volume      BIGINT          DEFAULT 0,
    source      TEXT            DEFAULT 'broker'
);

-- Convert to hypertable (TimescaleDB magic: auto-partitions by time)
SELECT create_hypertable('{table}', 'time', if_not_exists => TRUE);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_{table_safe}_pair_time
    ON {table} (pair, time DESC);

-- Compression policy: compress chunks older than 7 days (saves ~90% disk)
SELECT add_compression_policy('{table}',
    INTERVAL '7 days', if_not_exists => TRUE);
"""

BAR_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    time        TIMESTAMPTZ     NOT NULL,
    pair        TEXT            NOT NULL,
    freq        TEXT            NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    spread_avg  DOUBLE PRECISION,
    bid_close   DOUBLE PRECISION,
    ask_close   DOUBLE PRECISION,
    n_ticks     INTEGER         DEFAULT 0
);
SELECT create_hypertable('{table}', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_{table_safe}_pair_freq_time
    ON {table} (pair, freq, time DESC);
"""


class TimescaleDBStore:
    """
    Manages all tick and bar data in TimescaleDB.

    Usage
    -----
        store = TimescaleDBStore.from_env()
        store.write_ticks(tick_df)
        bars = store.read_bars("EURUSD", "1min", limit=10_000)
    """

    def __init__(
        self,
        host:   str = "localhost",
        port:   int = 5432,
        dbname: str = "forex_ticks",
        user:   str = "forex_user",
        password: str = "",
        tick_table: str = "tick_data",
        bar_table:  str = "ohlcv_bars",
    ):
        self.dsn = (
            f"host={host} port={port} dbname={dbname} "
            f"user={user} password={password}"
        )
        self.tick_table = tick_table
        self.bar_table  = bar_table
        self._conn: Optional[object] = None

    @classmethod
    def from_env(cls) -> "TimescaleDBStore":
        """Construct from environment variables (safe for production)."""
        return cls(
            host=os.getenv("TIMESCALE_HOST", "localhost"),
            port=int(os.getenv("TIMESCALE_PORT", "5432")),
            dbname=os.getenv("TIMESCALE_DB", "forex_ticks"),
            user=os.getenv("TIMESCALE_USER", "forex_user"),
            password=os.getenv("TIMESCALE_PASSWORD", ""),
        )

    @classmethod
    def mock(cls) -> "TimescaleDBStore":
        """
        Returns a mock store backed by in-memory DataFrames.
        Used when TimescaleDB is not running (dev/test mode).
        """
        store = cls.__new__(cls)
        store.tick_table = "tick_data"
        store.bar_table  = "ohlcv_bars"
        store._mock_ticks: List[dict] = []
        store._mock_bars:  List[dict] = []
        store._is_mock = True
        print("[TimescaleDB] Running in MOCK mode (in-memory storage)")
        return store

    def connect(self):
        if not PSYCOPG2:
            raise RuntimeError("psycopg2 not installed. pip install psycopg2-binary")
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = False
        return self._conn

    def setup_schema(self):
        """Create hypertables if they don't exist. Call once on first deploy."""
        conn = self.connect()
        with conn.cursor() as cur:
            safe_tick = self.tick_table.replace(".", "_")
            safe_bar  = self.bar_table.replace(".", "_")
            cur.execute(TICK_DDL.format(
                table=self.tick_table, table_safe=safe_tick))
            cur.execute(BAR_DDL.format(
                table=self.bar_table, table_safe=safe_bar))
        conn.commit()
        print(f"[TimescaleDB] Schema ready: {self.tick_table}, {self.bar_table}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_ticks(self, df: pd.DataFrame, pair: str = "EURUSD"):
        """
        Bulk-insert tick DataFrame into TimescaleDB.
        Uses execute_values for 10-100× faster inserts than row-by-row.

        df must have columns: [bid, ask, volume] with UTC DatetimeIndex.
        """
        if getattr(self, "_is_mock", False):
            records = df.assign(pair=pair).to_dict("records")
            self._mock_ticks.extend(records)
            return len(records)

        conn = self.connect()
        rows = [
            (ts.isoformat(), pair, float(row["bid"]),
             float(row["ask"]), int(row.get("volume", 0)))
            for ts, row in df.iterrows()
        ]
        sql = f"""
            INSERT INTO {self.tick_table} (time, pair, bid, ask, volume)
            VALUES %s ON CONFLICT DO NOTHING
        """
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
        return len(rows)

    def write_bars(self, df: pd.DataFrame, pair: str, freq: str = "1min"):
        """Bulk-insert OHLCV bars."""
        if getattr(self, "_is_mock", False):
            records = df.assign(pair=pair, freq=freq).to_dict("records")
            self._mock_bars.extend(records)
            return len(records)

        conn = self.connect()
        cols = ["open", "high", "low", "close", "volume",
                "spread_avg", "bid_close", "ask_close", "n_ticks"]
        rows = []
        for ts, row in df.iterrows():
            rows.append((
                ts.isoformat(), pair, freq,
                *[float(row.get(c, 0) or 0) for c in cols]
            ))
        sql = f"""
            INSERT INTO {self.bar_table}
            (time, pair, freq, open, high, low, close, volume,
             spread_avg, bid_close, ask_close, n_ticks)
            VALUES %s ON CONFLICT DO NOTHING
        """
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
        return len(rows)

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_ticks(
        self,
        pair:  str = "EURUSD",
        start: Optional[str] = None,
        end:   Optional[str] = None,
        limit: int = 1_000_000,
    ) -> pd.DataFrame:
        """
        Read tick data from TimescaleDB.
        TimescaleDB's time_bucket and continuous aggregates make this fast
        even for millions of rows.
        """
        if getattr(self, "_is_mock", False):
            df = pd.DataFrame(self._mock_ticks)
            if df.empty:
                return _empty_tick_df()
            df = df[df["pair"] == pair].copy()
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], utc=True)
                return df.set_index("time").sort_index()
            return df.sort_index()

        conn = self.connect()
        where = ["pair = %s"]
        params: List = [pair]
        if start:
            where.append("time >= %s")
            params.append(start)
        if end:
            where.append("time <= %s")
            params.append(end)
        where_clause = " AND ".join(where)

        sql = f"""
            SELECT time, bid, ask, mid, spread, volume
            FROM   {self.tick_table}
            WHERE  {where_clause}
            ORDER  BY time DESC
            LIMIT  %s
        """
        params.append(limit)
        df = pd.read_sql(sql, conn, params=params, index_col="time",
                         parse_dates=["time"])
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    def read_bars(
        self,
        pair:  str = "EURUSD",
        freq:  str = "1min",
        start: Optional[str] = None,
        end:   Optional[str] = None,
        limit: int = 100_000,
    ) -> pd.DataFrame:
        """Read pre-aggregated OHLCV bars."""
        if getattr(self, "_is_mock", False):
            df = pd.DataFrame(self._mock_bars)
            if df.empty:
                return _empty_bar_df()
            df = df[(df["pair"] == pair) & (df["freq"] == freq)].copy()
            return df.set_index("time").sort_index()

        conn = self.connect()
        where = ["pair = %s", "freq = %s"]
        params: List = [pair, freq]
        if start:
            where.append("time >= %s")
            params.append(start)
        if end:
            where.append("time <= %s")
            params.append(end)
        where_clause = " AND ".join(where)
        sql = f"""
            SELECT time, open, high, low, close, volume,
                   spread_avg, bid_close, ask_close, n_ticks
            FROM   {self.bar_table}
            WHERE  {where_clause}
            ORDER  BY time DESC
            LIMIT  %s
        """
        params.append(limit)
        df = pd.read_sql(sql, conn, params=params, index_col="time",
                         parse_dates=["time"])
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    def latest_tick(self, pair: str = "EURUSD") -> Optional[dict]:
        """Get the most recent tick for a pair (for live trading sync)."""
        if getattr(self, "_is_mock", False):
            ticks = [t for t in self._mock_ticks if t.get("pair") == pair]
            return ticks[-1] if ticks else None
        conn = self.connect()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                f"SELECT * FROM {self.tick_table} WHERE pair=%s "
                f"ORDER BY time DESC LIMIT 1", (pair,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()


def _empty_tick_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["bid", "ask", "mid", "spread", "volume"])

def _empty_bar_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume",
                                  "spread_avg", "bid_close", "ask_close"])


# ─────────────────────────────────────────────────────────────────────────────
# KAFKA TICK CONSUMER
# ─────────────────────────────────────────────────────────────────────────────

class KafkaTickConsumer:
    """
    Consumes live tick data from Kafka and writes to TimescaleDB.

    The Kafka → TimescaleDB pipeline decouples the feed from the model:
      Broker API → Kafka Producer → [Kafka Topic] → KafkaTickConsumer → TimescaleDB
                                                                      ↓
                                                              Feature Engineering
                                                                      ↓
                                                              Model Inference

    This means the model never waits for I/O — it reads from TimescaleDB
    while Kafka handles the upstream buffering independently.

    Usage (in a background thread)
    -----
        consumer = KafkaTickConsumer(store=TimescaleDBStore.from_env())
        consumer.start()   # non-blocking
        # ... later ...
        consumer.stop()
    """

    def __init__(
        self,
        store:            TimescaleDBStore,
        bootstrap_servers: str  = "localhost:9092",
        tick_topic:        str  = "forex.ticks",
        news_topic:        str  = "forex.news",
        consumer_group:    str  = "forex_scaler",
        batch_size:        int  = 1000,
        flush_interval_ms: int  = 500,
    ):
        self.store             = store
        self.bootstrap_servers = bootstrap_servers
        self.tick_topic        = tick_topic
        self.news_topic        = news_topic
        self.consumer_group    = consumer_group
        self.batch_size        = batch_size
        self.flush_interval_ms = flush_interval_ms
        self._running          = False
        self._thread: Optional[threading.Thread] = None
        self._tick_buffer:  List[dict] = []
        self._news_buffer:  List[dict] = []
        self.stats = {"ticks_consumed": 0, "batches_written": 0, "errors": 0}

    def start(self):
        """Start consuming in a background daemon thread."""
        if not KAFKA:
            print("[Kafka] kafka-python not installed — consumer disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[Kafka] Consumer started | Topics: {self.tick_topic}, {self.news_topic}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._flush_buffer()
        print(f"[Kafka] Consumer stopped | Stats: {self.stats}")

    def _run(self):
        if not KAFKA:
            return
        consumer = KafkaConsumer(
            self.tick_topic,
            self.news_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
        )
        last_flush = time.time()
        while self._running:
            try:
                for message in consumer:
                    if not self._running:
                        break
                    self._handle_message(message)

                    # Flush buffer when full or after interval
                    elapsed_ms = (time.time() - last_flush) * 1000
                    if (len(self._tick_buffer) >= self.batch_size
                            or elapsed_ms >= self.flush_interval_ms):
                        self._flush_buffer()
                        last_flush = time.time()

            except Exception as e:
                self.stats["errors"] += 1
                print(f"[Kafka] Consumer error: {e}")
                time.sleep(1)

        consumer.close()

    def _handle_message(self, message):
        topic = message.topic
        data  = message.value

        if topic == self.tick_topic:
            # Expected format: {time, pair, bid, ask, volume}
            self._tick_buffer.append(data)
            self.stats["ticks_consumed"] += 1

        elif topic == self.news_topic:
            # Expected format: {time, headline, sentiment, source}
            self._news_buffer.append(data)

    def _flush_buffer(self):
        """Write buffered ticks to TimescaleDB in bulk."""
        if self._tick_buffer:
            try:
                df = pd.DataFrame(self._tick_buffer)
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.set_index("time")
                for pair, group in df.groupby("pair"):
                    self.store.write_ticks(group, pair=pair)
                self.stats["batches_written"] += 1
                self._tick_buffer.clear()
            except Exception as e:
                print(f"[Kafka] Flush error: {e}")
                self.stats["errors"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# MOCK KAFKA PRODUCER  (for testing without a live Kafka cluster)
# ─────────────────────────────────────────────────────────────────────────────

class MockKafkaProducer:
    """
    Simulates a Kafka producer by feeding synthetic ticks directly
    to a TimescaleDB mock store. Use during development/testing.
    """

    def __init__(self, store: TimescaleDBStore, pairs: List[str] = ["EURUSD"]):
        self.store   = store
        self.pairs   = pairs
        self._rng    = np.random.default_rng(42)
        self._prices = {p: 1.0850 for p in pairs}

    def produce_ticks(self, n: int = 10_000) -> int:
        """Generate and store n synthetic ticks per pair."""
        total = 0
        for pair in self.pairs:
            ticks = self._generate_ticks(pair, n)
            written = self.store.write_ticks(ticks, pair=pair)
            total += written
            self._prices[pair] = float(ticks["ask"].iloc[-1])
        return total

    def _generate_ticks(self, pair: str, n: int) -> pd.DataFrame:
        base = self._prices.get(pair, 1.0850)
        log_ret = self._rng.normal(0, 0.0001, n)
        mid = base * np.exp(np.cumsum(log_ret))
        spread = 0.00005
        ts = pd.date_range(
            start=pd.Timestamp.utcnow() - pd.Timedelta(seconds=n),
            periods=n, freq="1s", tz="UTC"
        )
        return pd.DataFrame({
            "bid":    np.round(mid - spread / 2, 5),
            "ask":    np.round(mid + spread / 2, 5),
            "volume": self._rng.integers(1, 20, n),
        }, index=ts)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: get a connected store (real or mock)
# ─────────────────────────────────────────────────────────────────────────────

def get_store(mock: bool = False) -> TimescaleDBStore:
    """
    Returns a TimescaleDB store.
    Pass mock=True for dev/test — no database required.
    In production, set environment variables and pass mock=False.
    """
    if mock or not PSYCOPG2 or not os.getenv("TIMESCALE_HOST"):
        return TimescaleDBStore.mock()
    return TimescaleDBStore.from_env()


if __name__ == "__main__":
    print("TimescaleDB + Kafka — integration smoke test")
    store = get_store(mock=True)
    producer = MockKafkaProducer(store, pairs=["EURUSD", "GBPUSD"])
    n = producer.produce_ticks(n=5_000)
    print(f"Produced {n} ticks into mock store")

    ticks = store.read_ticks("EURUSD")
    print(f"EURUSD ticks in store: {len(ticks):,}")
    print(ticks.tail(3))

    consumer = KafkaTickConsumer(store=store)
    consumer.start()
    time.sleep(1)
    consumer.stop()
    print(f"Consumer stats: {consumer.stats}")
