"""
monitoring/prometheus_exporter.py
===================================
Prometheus metrics exporter for the Forex Scaling Model live engine.

Exposes 15 real-time metrics at http://localhost:8000/metrics

Metrics:
  forex_equity              Gauge  — current account equity (USD)
  forex_pnl_session         Gauge  — session P&L from session start
  forex_drawdown            Gauge  — current drawdown as fraction [0, 1]
  forex_trade_rate_1h       Gauge  — closed trades per hour (rolling)
  forex_win_rate_100        Gauge  — win rate over last 100 trades [0, 1]
  forex_sharpe_200          Gauge  — rolling Sharpe over last 200 trades
  forex_drift_flag          Gauge  — 1.0 if drift detected, else 0.0
  forex_circuit_breaker     Gauge  — 1.0 if halted, else 0.0
  forex_sentiment_bias      Gauge  — latest sentiment score [-1, +1]
  forex_regime_trending     Gauge  — trending regime probability [0, 1]
  forex_regime_neutral      Gauge  — neutral regime probability [0, 1]
  forex_regime_mean_rev     Gauge  — mean-reverting regime probability [0, 1]
  forex_position_lots       Gauge  — current position in lots (signed)
  forex_latency_ms          Histogram — per-bar inference latency
  forex_promotions_total    Counter   — total model promotions this session

Falls back to a no-op stub when prometheus_client is not installed.

Usage in live engine:
    from monitoring.prometheus_exporter import ForexPrometheusExporter
    prom = ForexPrometheusExporter(port=8000)
    prom.start()              # starts background HTTP server

    prom.update_equity(10_200.0)
    prom.update_trade(pnl=150.0, won=True)
    prom.update_latency(3.5)
    prom.set_drift(True)
    prom.set_circuit_breaker(False)
"""

import os
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Optional

import numpy as np

warnings.filterwarnings("ignore")

_PROM_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))


# ── stub for when prometheus_client is not installed ─────────────────────────

class _NoOpMetric:
    """No-op stub so code doesn't break when prometheus_client is absent."""
    def set(self, *a, **kw): pass
    def inc(self, *a, **kw): pass
    def observe(self, *a, **kw): pass
    def labels(self, **kw): return self


class _NoOpRegistry:
    def start_http_server(self, *a, **kw): pass


# ── main exporter ────────────────────────────────────────────────────────────

class ForexPrometheusExporter:
    """
    Prometheus metrics exporter. Starts a background HTTP server on `port`.
    All update_* / set_* methods are safe to call from any thread.

    Parameters
    ----------
    port           : HTTP port for /metrics endpoint.
    initial_equity : Starting account equity (used for session P&L).
    """

    def __init__(self, port: int = _PROM_PORT, initial_equity: float = 10_000.0):
        self._port           = port
        self._start_equity   = initial_equity
        self._current_equity = initial_equity
        self._peak_equity    = initial_equity
        self._session_start  = time.time()

        # Rolling windows for computed metrics
        self._pnls:    Deque[float]    = deque(maxlen=200)
        self._wins:    Deque[bool]     = deque(maxlen=100)
        self._trade_ts: Deque[float]   = deque(maxlen=200)  # timestamps

        self._available = False
        self._metrics   = {}
        self._server_thread: Optional[threading.Thread] = None

        self._init_prometheus()

    def _init_prometheus(self):
        try:
            from prometheus_client import (
                Gauge, Counter, Histogram, CollectorRegistry, start_http_server
            )
            self._prom_start_server = start_http_server

            self._g_equity     = Gauge("forex_equity",          "Account equity (USD)")
            self._g_pnl        = Gauge("forex_pnl_session",     "Session P&L (USD)")
            self._g_dd         = Gauge("forex_drawdown",        "Current drawdown fraction")
            self._g_trade_rate = Gauge("forex_trade_rate_1h",   "Trades per hour (rolling)")
            self._g_win_rate   = Gauge("forex_win_rate_100",    "Win rate last 100 trades")
            self._g_sharpe     = Gauge("forex_sharpe_200",      "Rolling Sharpe last 200 trades")
            self._g_drift      = Gauge("forex_drift_flag",      "Drift detected (0/1)")
            self._g_cb         = Gauge("forex_circuit_breaker", "Circuit breaker active (0/1)")
            self._g_sentiment  = Gauge("forex_sentiment_bias",  "Latest sentiment score")
            self._g_reg_trend  = Gauge("forex_regime_trending", "Trending regime probability")
            self._g_reg_neut   = Gauge("forex_regime_neutral",  "Neutral regime probability")
            self._g_reg_mr     = Gauge("forex_regime_mean_rev", "Mean-reverting regime probability")
            self._g_lots       = Gauge("forex_position_lots",   "Current position lots (signed)")
            self._h_latency    = Histogram("forex_latency_ms",  "Per-bar inference latency (ms)",
                                           buckets=[1, 2, 3, 5, 8, 13, 21, 34, 55])
            self._c_promotions = Counter("forex_promotions_total", "Total model promotions")

            self._available = True
            print(f"[Prometheus] Metrics registered — will serve on :{self._port}/metrics")
        except ImportError:
            print("[Prometheus] prometheus_client not installed — using no-op stub. "
                  "Install with: pip install prometheus-client")
            self._g_equity = self._g_pnl = self._g_dd = self._g_trade_rate = _NoOpMetric()
            self._g_win_rate = self._g_sharpe = self._g_drift = self._g_cb = _NoOpMetric()
            self._g_sentiment = self._g_reg_trend = self._g_reg_neut = _NoOpMetric()
            self._g_reg_mr = self._g_lots = self._h_latency = self._c_promotions = _NoOpMetric()

    def start(self):
        """Start the Prometheus HTTP server in a background daemon thread."""
        if not self._available:
            print("[Prometheus] No-op mode — HTTP server not started")
            return
        def _serve():
            try:
                self._prom_start_server(self._port)
                print(f"[Prometheus] HTTP server started on :{self._port}/metrics")
            except OSError as e:
                print(f"[Prometheus] Failed to start server: {e}")
        self._server_thread = threading.Thread(target=_serve, daemon=True)
        self._server_thread.start()
        time.sleep(0.2)   # give server time to bind

    # ── update methods ───────────────────────────────────────────────────────

    def update_equity(self, equity: float):
        """Update equity, P&L, and drawdown metrics."""
        self._current_equity = equity
        self._peak_equity    = max(self._peak_equity, equity)
        dd  = max(0.0, (self._peak_equity - equity) / self._peak_equity)
        pnl = equity - self._start_equity
        self._g_equity.set(equity)
        self._g_pnl.set(pnl)
        self._g_dd.set(dd)

    def update_trade(self, pnl: float, won: bool):
        """Record one closed trade and update win rate + Sharpe + trade rate."""
        self._pnls.append(pnl)
        self._wins.append(won)
        self._trade_ts.append(time.time())

        # Win rate (last 100)
        wr = sum(self._wins) / max(len(self._wins), 1)
        self._g_win_rate.set(wr)

        # Sharpe (last 200)
        if len(self._pnls) >= 10:
            arr = np.array(self._pnls)
            std = arr.std()
            sr  = float(arr.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0
            self._g_sharpe.set(sr)

        # Trade rate per hour (last 200 trade timestamps)
        if len(self._trade_ts) >= 2:
            elapsed_h = (self._trade_ts[-1] - self._trade_ts[0]) / 3600
            rate = len(self._trade_ts) / max(elapsed_h, 1/60)
            self._g_trade_rate.set(rate)

    def update_latency(self, latency_ms: float):
        """Record per-bar inference latency."""
        self._h_latency.observe(latency_ms)

    def set_drift(self, detected: bool):
        """Set drift detection flag."""
        self._g_drift.set(1.0 if detected else 0.0)

    def set_circuit_breaker(self, active: bool):
        """Set circuit breaker flag."""
        self._g_cb.set(1.0 if active else 0.0)

    def set_sentiment(self, score: float):
        """Set current sentiment bias."""
        self._g_sentiment.set(float(np.clip(score, -1.0, 1.0)))

    def set_regime(self, trending: float = 0.33,
                   neutral: float = 0.34, mean_rev: float = 0.33):
        """Set regime probabilities. Values should sum to ~1."""
        total = trending + neutral + mean_rev + 1e-9
        self._g_reg_trend.set(trending / total)
        self._g_reg_neut.set(neutral  / total)
        self._g_reg_mr.set(mean_rev   / total)

    def set_position(self, lots: float):
        """Set current position size (signed, negative = short)."""
        self._g_lots.set(lots)

    def inc_promotions(self):
        """Increment the model promotion counter."""
        self._c_promotions.inc()

    def snapshot(self) -> dict:
        """Return current metric values as a dict (useful without Prometheus)."""
        arr = np.array(self._pnls) if self._pnls else np.array([0.0])
        std = arr.std()
        return {
            "equity":        self._current_equity,
            "pnl_session":   self._current_equity - self._start_equity,
            "drawdown":      max(0.0, (self._peak_equity - self._current_equity)
                                 / self._peak_equity),
            "win_rate":      sum(self._wins) / max(len(self._wins), 1),
            "sharpe":        float(arr.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0,
            "n_trades":      len(self._pnls),
        }


# ── smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)

    exp = ForexPrometheusExporter(port=8001, initial_equity=10_000.0)
    exp.start()

    print("\nSimulating 50 bars of data...")
    equity = 10_000.0
    for i in range(50):
        pnl    = float(rng.normal(10, 80))
        equity = max(equity + pnl, 1.0)
        exp.update_equity(equity)
        exp.update_trade(pnl, won=(pnl > 0))
        exp.update_latency(float(rng.uniform(1.5, 6.0)))
        exp.set_sentiment(float(rng.uniform(-0.5, 0.8)))
        exp.set_regime(trending=0.5, neutral=0.3, mean_rev=0.2)
        exp.set_position(0.1)

    print(f"\nSnapshot: {exp.snapshot()}")
    print("OK ✓")
    print(f"Metrics available at http://localhost:8001/metrics (if prometheus_client installed)")
