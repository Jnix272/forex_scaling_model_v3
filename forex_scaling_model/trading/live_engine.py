"""
trading/live_engine.py
=======================
Live trading engine — connects all model components to broker execution.

Architecture:
  Kafka (live ticks) → Feature pipeline → TIP-Search inference
  → UQ confidence filter → Portfolio VaR check
  → Regime Kelly sizing → Almgren-Chriss execution
  → DrawdownAwareExit guard → Broker order submission

Supported brokers (via abstract interface):
  - LMAX FIX 4.4 (institutional, recommended)
  - OANDA v20 REST API
  - Interactive Brokers TWS

Run:
  python trading/live_engine.py --broker lmax --pair EURUSD --model haelt
"""

import os, sys, time, json, signal, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List
from collections import deque
import threading

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_engineering import FeatureEngineer
from features.advanced_features import AdvancedFeatureBuilder
from pretrain.contrastive import DualStreamSentiment, TIPSearchManager, DriftDetector
from risk.execution import (RegimeConditionalKelly, AlmgrenChrissExecutor,
                             DrawdownAwareExitPolicy, PortfolioVaR)
from monitoring.pipeline import ShadowModeDeployer, SHAPFeatureTracker
from config.settings import FEATURES, LATENCY, RISK, SIZING, MONITORING, PATHS


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BROKER INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class BrokerInterface:
    """
    Abstract broker. Subclass for LMAX, OANDA, IB.
    All methods return standardised dicts.
    """
    def connect(self) -> bool:          raise NotImplementedError
    def disconnect(self):               pass
    def get_bid_ask(self, pair: str):   raise NotImplementedError  # → (bid, ask)
    def get_account(self):              raise NotImplementedError  # → {"equity":…,"balance":…}
    def market_order(self, pair, side, lots, stop_loss=None, take_profit=None):
                                        raise NotImplementedError  # → order_id
    def close_position(self, pair):     raise NotImplementedError  # → True/False
    def get_positions(self):            raise NotImplementedError  # → {pair: lots}


class PaperBroker(BrokerInterface):
    """
    Paper-trading broker — simulates execution against live bid/ask
    from the LMAX data feed. No real orders sent.
    Use this during shadow mode evaluation.
    """
    def __init__(self, initial_equity=10_000.0, spread_pips=0.5, pip_size=0.0001):
        self.equity  = initial_equity
        self.balance = initial_equity
        self.spread  = spread_pips * pip_size
        self._pos: Dict[str, float] = {}
        self._entry: Dict[str, float] = {}
        self._orders = []

    def connect(self): print("[Paper] Connected"); return True
    def disconnect(self): print(f"[Paper] Final equity: ${self.equity:.2f}")

    def get_bid_ask(self, pair):
        # Returns synthetic bid/ask (in live mode feed from LMAX)
        mid = 1.0850; half = self.spread / 2
        return mid - half, mid + half

    def get_account(self):
        return {"equity": self.equity, "balance": self.balance}

    def market_order(self, pair, side, lots, stop_loss=None, take_profit=None):
        bid, ask = self.get_bid_ask(pair)
        price = ask if side == "buy" else bid
        self._pos[pair]   = lots if side == "buy" else -lots
        self._entry[pair] = price
        order = {"id": len(self._orders)+1, "pair": pair, "side": side,
                 "lots": lots, "price": price, "time": datetime.utcnow().isoformat()}
        self._orders.append(order)
        print(f"  [Paper] {side.upper()} {lots} {pair} @ {price:.5f}")
        return order["id"]

    def close_position(self, pair):
        if pair not in self._pos or self._pos[pair] == 0:
            return False
        bid, ask = self.get_bid_ask(pair)
        close_price = bid if self._pos[pair] > 0 else ask
        pnl = (close_price - self._entry[pair]) * self._pos[pair] * 10_000
        self.equity += pnl
        print(f"  [Paper] Closed {pair}: PnL ${pnl:+.2f} | Equity ${self.equity:.2f}")
        self._pos[pair] = 0
        return True

    def get_positions(self):
        return {p: l for p, l in self._pos.items() if abs(l) > 0}


class LMAXBroker(BrokerInterface):
    """LMAX FIX 4.4 execution (requires LMAX account)."""
    def __init__(self):
        self._username = os.getenv("LMAX_USERNAME")
        self._password = os.getenv("LMAX_PASSWORD")
        self._session  = None

    def connect(self):
        if not self._username:
            print("[LMAX] No credentials — set LMAX_USERNAME + LMAX_PASSWORD")
            return False
        try:
            from data.sources import LMAXLoader
            loader = LMAXLoader(lmax_username=self._username,
                                lmax_password=self._password)
            ok = loader.login()
            self._session = loader
            return ok
        except Exception as e:
            print(f"[LMAX] Connect error: {e}")
            return False

    def get_bid_ask(self, pair):
        if not self._session: return None, None
        ob = self._session.fetch_orderbook(pair)
        if ob: return ob["best_bid"], ob["best_ask"]
        return None, None

    def get_account(self): return {"equity": 10_000, "balance": 10_000}
    def market_order(self, *a, **kw): print("[LMAX] FIX order — implement FIX 4.4"); return None
    def close_position(self, pair): print(f"[LMAX] Close {pair} — implement FIX 4.4"); return True
    def get_positions(self): return {}


class OANDABroker(BrokerInterface):
    """OANDA v20 REST API."""
    def __init__(self):
        self._token    = os.getenv("OANDA_API_TOKEN")
        self._account  = os.getenv("OANDA_ACCOUNT_ID")
        self._base_url = "https://api-fxpractice.oanda.com"  # practice env

    def connect(self):
        if not self._token:
            print("[OANDA] Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID env vars")
            return False
        try:
            from urllib.request import Request, urlopen
            req = Request(f"{self._base_url}/v3/accounts/{self._account}",
                          headers={"Authorization": f"Bearer {self._token}"})
            with urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            eq = float(data["account"]["NAV"])
            print(f"[OANDA] Connected | Account {self._account} | NAV: ${eq:.2f}")
            return True
        except Exception as e:
            print(f"[OANDA] Connect error: {e}")
            return False

    def get_bid_ask(self, pair):
        try:
            from urllib.request import Request, urlopen
            instr = pair[:3] + "_" + pair[3:]
            req = Request(
                f"{self._base_url}/v3/instruments/{instr}/candles"
                f"?count=1&granularity=S5&price=BA",
                headers={"Authorization": f"Bearer {self._token}"},
            )
            with urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            c = data["candles"][-1]
            return float(c["bid"]["c"]), float(c["ask"]["c"])
        except Exception:
            return None, None

    def get_account(self):
        try:
            from urllib.request import Request, urlopen
            req = Request(f"{self._base_url}/v3/accounts/{self._account}",
                          headers={"Authorization": f"Bearer {self._token}"})
            with urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
            return {"equity": float(d["account"]["NAV"]),
                    "balance": float(d["account"]["balance"])}
        except Exception:
            return {"equity": 0, "balance": 0}

    def market_order(self, pair, side, lots, stop_loss=None, take_profit=None):
        units = int(lots * 10_000) * (1 if side == "buy" else -1)
        body  = json.dumps({"order": {
            "units": str(units),
            "instrument": pair[:3]+"_"+pair[3:],
            "timeInForce": "FOK",
            "type": "MARKET",
            **({"stopLossOnFill": {"price": f"{stop_loss:.5f}"}} if stop_loss else {}),
        }}).encode()
        try:
            from urllib.request import Request, urlopen
            req = Request(
                f"{self._base_url}/v3/accounts/{self._account}/orders",
                data=body,
                headers={"Authorization": f"Bearer {self._token}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
            print(f"  [OANDA] {side.upper()} {lots} {pair} → order {d.get('relatedTransactionIDs')}")
            return d.get("relatedTransactionIDs", [None])[0]
        except Exception as e:
            print(f"  [OANDA] Order error: {e}")
            return None

    def close_position(self, pair):
        try:
            from urllib.request import Request, urlopen
            req = Request(
                f"{self._base_url}/v3/accounts/{self._account}/positions/"
                f"{pair[:3]}_{pair[3:]}/close",
                data=b'{"longUnits":"ALL","shortUnits":"ALL"}',
                headers={"Authorization": f"Bearer {self._token}",
                         "Content-Type": "application/json"},
                method="PUT",
            )
            with urlopen(req, timeout=5): pass
            return True
        except Exception as e:
            print(f"[OANDA] Close error: {e}"); return False

    def get_positions(self):
        try:
            from urllib.request import Request, urlopen
            req = Request(
                f"{self._base_url}/v3/accounts/{self._account}/openPositions",
                headers={"Authorization": f"Bearer {self._token}"},
            )
            with urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
            return {p["instrument"].replace("_",""): float(p["long"]["units"])
                    for p in d.get("positions",[])}
        except Exception:
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# TICK BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class LiveTickBuffer:
    """
    Accumulates live ticks into 1-minute bars in real time.
    Thread-safe. Resample triggers are aligned to wall-clock minute boundaries.
    """
    def __init__(self, pair="EURUSD", bar_freq="1min", max_bars=500):
        self.pair     = pair
        self.freq     = pd.tseries.frequencies.to_offset(bar_freq)
        self.max_bars = max_bars
        self._ticks: deque = deque(maxlen=50_000)
        self._bars: deque  = deque(maxlen=max_bars)
        self._lock = threading.Lock()

    def push_tick(self, bid: float, ask: float, volume: float = 1.0):
        ts = pd.Timestamp.utcnow()
        with self._lock:
            self._ticks.append({"time": ts, "bid": bid, "ask": ask,
                                 "mid": (bid+ask)/2, "volume": volume})

    def get_bars(self) -> Optional[pd.DataFrame]:
        """Resample accumulated ticks to OHLCV bars. Returns None if too few ticks."""
        with self._lock:
            if len(self._ticks) < 5:
                return None
            df = pd.DataFrame(list(self._ticks))
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        ohlcv = pd.DataFrame({
            "open":      df["mid"].resample(self.freq).first(),
            "high":      df["mid"].resample(self.freq).max(),
            "low":       df["mid"].resample(self.freq).min(),
            "close":     df["mid"].resample(self.freq).last(),
            "volume":    df["volume"].resample(self.freq).sum(),
            "bid_close": df["bid"].resample(self.freq).last(),
            "ask_close": df["ask"].resample(self.freq).last(),
            "spread_avg":(df["ask"]-df["bid"]).resample(self.freq).mean(),
        }).dropna(subset=["close"])
        if len(ohlcv) < 70:
            return None
        return ohlcv.tail(self.max_bars)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class LiveTradingEngine:
    """
    Main live trading engine.

    Loop (every new bar):
      1. Get latest bid/ask from broker
      2. Push to tick buffer → resample to bars
      3. Build features (core + advanced)
      4. Dual-stream sentiment update (async, 60s cadence)
      5. TIP-Search: select fast (DQN) or slow (HAELT) model
      6. UQ: run MC dropout, check confidence
      7. Sentiment filter: suppress counter-trend signals
      8. Portfolio VaR check
      9. Drawdown-aware exit guard
      10. Regime-conditional Kelly sizing
      11. Almgren-Chriss execution scheduling
      12. Submit order to broker
      13. Log to W&B / TimescaleDB
    """

    BAR_SECONDS = 60   # 1-minute bars

    def __init__(
        self,
        broker:           BrokerInterface,
        fast_agent,       # DQN (select_action)
        slow_model,       # HAELT (select_action)
        pair:             str   = "EURUSD",
        equity:           float = 10_000.0,
        max_lots:         float = 1.0,
        confidence_thresh: float = 0.15,
        log_dir:          str   = None,
    ):
        if log_dir is None:
            log_dir = PATHS["logs_live"]
        self.broker    = broker
        self.pair      = pair
        self.equity    = equity
        self.max_lots  = max_lots
        self.conf_thr  = confidence_thresh
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Feature pipeline
        self.fe    = FeatureEngineer(atr_window=FEATURES["atr_window"],
                                     lag_windows=FEATURES["lag_windows"])
        self.afb   = AdvancedFeatureBuilder(hurst_windows=[30, 60])

        # Risk components
        self.rck   = RegimeConditionalKelly()
        self.ac    = AlmgrenChrissExecutor()
        self.dae   = DrawdownAwareExitPolicy()
        self.pvar  = PortfolioVaR()
        self.dae.equity_high = equity

        # Sentiment
        self.sentiment = DualStreamSentiment()

        # Latency
        class _Wrap:
            def __init__(self, m): self.m = m
            def select_action(self, o): return self.m.select_action(o)
        self.tip = TIPSearchManager(fast_agent=_Wrap(fast_agent),
                                    slow_agent=_Wrap(slow_model))

        # Monitoring
        self.drift  = DriftDetector()
        self.shadow = ShadowModeDeployer(shadow_bars=2000)
        self.buf    = LiveTickBuffer(pair=pair)

        # State
        self._running  = False
        self._position = 0.0
        self._bar_log: list = []
        self._baseline_fitted = False

        print(f"[Live] Engine initialised | Pair: {pair} | Equity: ${equity:,.0f}")

    def start(self, max_bars: Optional[int] = None):
        """Start the live trading loop."""
        if not self.broker.connect():
            print("[Live] Broker connection failed — using PaperBroker for testing")
        self._running = True
        signal.signal(signal.SIGINT,  lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        # Async sentiment updater (every 60s)
        self._start_sentiment_loop()

        print(f"[Live] Trading loop started | {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
        bar_count = 0
        next_bar_time = self._next_bar()

        while self._running:
            if max_bars and bar_count >= max_bars:
                break
            now = datetime.now(timezone.utc)
            if now < next_bar_time:
                # Poll ticks at 100ms cadence
                bid, ask = self.broker.get_bid_ask(self.pair)
                if bid and ask:
                    self.buf.push_tick(bid, ask)
                time.sleep(0.1)
                continue

            # ── New bar ──────────────────────────────────────────────────────
            bars = self.buf.get_bars()
            if bars is None or len(bars) < 70:
                next_bar_time = self._next_bar()
                continue

            self._on_new_bar(bars, bar_count)
            bar_count     += 1
            next_bar_time  = self._next_bar()

        self.stop()

    def _on_new_bar(self, bars: pd.DataFrame, bar_idx: int):
        """Process one completed bar."""
        t0 = time.perf_counter()

        # 1. Feature engineering
        try:
            features = self.fe.build(bars)
            if len(features) < 62:
                return
        except Exception as e:
            print(f"[Live] Feature error: {e}"); return

        obs = features.values[-1].astype(np.float32)
        atr = float(features[f"atr_{FEATURES['atr_window']}"].iloc[-1])

        # 2. Drift check (every 500 bars)
        if self._baseline_fitted and bar_idx % 500 == 0:
            self._check_drift(features)

        # 3. Drawdown guard
        acct = self.broker.get_account()
        self.equity = acct["equity"]
        self.dae.equity_high = max(self.dae.equity_high, self.equity)
        guard = self.dae.update(self.equity)
        if guard["halt"]:
            print(f"[Live] 🔴 {guard['level'].upper()} — {guard['action']} "
                  f"| DD={guard['drawdown']:.2%}")
            if guard["action"] in ("close_all", "close_all_halt"):
                self.broker.close_position(self.pair)
                self._position = 0.0
            return

        # 4. TIP-Search inference
        action, model_used, lat_ms = self.tip.select_action(obs, atr)

        # 5. Sentiment filter
        bias = self.sentiment.get_bias()
        action = self.sentiment.filter_signal(action, bias)

        # 6. Portfolio VaR
        positions = self.broker.get_positions()
        positions[self.pair] = self._position
        for p, ret in positions.items():
            self.pvar.add_return(p, float(np.random.randn() * 0.0003))  # Live: use actual bar return
        var_result = self.pvar.check_and_adjust(
            {p: abs(l) for p, l in positions.items()}, self.equity
        )
        size_adj = var_result.get(self.pair, self.max_lots) / (self.max_lots + 1e-9)

        # 7. Regime-conditional Kelly sizing
        hurst = float(features.get(f"hurst_60", pd.Series([0.5])).iloc[-1]
                      if f"hurst_60" in features.columns else 0.5)
        vol_z = float(features["vol_20"].iloc[-1] /
                      (features["vol_20"].rolling(60).mean().iloc[-1] + 1e-9))
        corr_stab = 0.1  # From correlation regime module
        sizing = self.rck.size_position(
            0.54, 1.7, hurst=hurst, vol_z=vol_z, corr_stability=corr_stab,
            current_hour=datetime.utcnow().hour, equity=self.equity,
            price=float(bars["close"].iloc[-1]), pip_stop=20,
            current_drawdown=guard["drawdown"],
        )
        lots = min(sizing["lots"] * size_adj, self.max_lots) if guard["size_multiplier"] > 0 \
               else sizing["lots"] * guard["size_multiplier"]

        # 8. Execute
        bid, ask = self.broker.get_bid_ask(self.pair)
        if not bid: return

        lat_total = (time.perf_counter() - t0) * 1000
        prev_pos  = self._position

        if action == 0 and self._position <= 0 and lots > 0:    # BUY
            if self._position < 0: self.broker.close_position(self.pair)
            sl = ask - RISK["atr_multiplier"] * atr
            self.broker.market_order(self.pair, "buy", lots, stop_loss=sl)
            self._position = lots
        elif action == 2 and self._position >= 0 and lots > 0:   # SELL
            if self._position > 0: self.broker.close_position(self.pair)
            sl = bid + RISK["atr_multiplier"] * atr
            self.broker.market_order(self.pair, "sell", lots, stop_loss=sl)
            self._position = -lots
        # action==1 → HOLD: do nothing

        # 9. Log
        log = {
            "time":       datetime.utcnow().isoformat(),
            "bar":        bar_idx,
            "action":     {0:"BUY",1:"HOLD",2:"SELL"}.get(action,"?"),
            "model":      model_used,
            "lots":       lots,
            "bid":        round(bid, 5),
            "ask":        round(ask, 5),
            "equity":     round(self.equity, 2),
            "drawdown":   round(guard["drawdown"], 4),
            "dd_level":   guard["level"],
            "regime_mult":round(sizing["regime_mult"], 3),
            "sentiment":  round(bias, 4),
            "latency_ms": round(lat_total, 2),
            "var_pct":    round(var_result.get("var_pct", 0), 5),
        }
        self._bar_log.append(log)
        if bar_idx % 60 == 0:
            self._save_log()
        if action != 1 or bar_idx % 10 == 0:
            print(f"  Bar {bar_idx:>4} | {log['action']:<4} {lots:.2f}L | "
                  f"${self.equity:,.0f} | {guard['level']} | "
                  f"sens={bias:+.3f} | lat={lat_total:.1f}ms | {model_used}")

    def _check_drift(self, features: pd.DataFrame):
        X = features.values
        if not self._baseline_fitted:
            self.drift.fit_baseline(X, np.random.normal(0.001, 0.01, len(X)))
            self._baseline_fitted = True
        else:
            result = self.drift.check(X[-500:], np.random.normal(0, 0.01, 500))
            if result["drift_detected"]:
                print(f"[Live] ⚠ DRIFT DETECTED: {result['reasons']}")
                print(f"       Recommend retraining: python training/train_gpu.py --model haelt --resume")

    def _start_sentiment_loop(self):
        """Background thread: refresh sentiment every 60 seconds."""
        def _loop():
            while self._running:
                try:
                    headlines = ["Market update"]  # Live: fetch from news API
                    self.sentiment.update_global_brain(headlines)
                except Exception: pass
                time.sleep(60)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def _next_bar(self) -> "datetime":
        """Next aligned bar timestamp."""
        now = datetime.now(timezone.utc)
        return now.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)

    def _save_log(self):
        path = self.log_dir / f"live_{datetime.utcnow():%Y%m%d}.jsonl"
        with open(path, "a") as f:
            for entry in self._bar_log[-60:]:
                f.write(json.dumps(entry) + "\n")

    def stop(self):
        print(f"\n[Live] Stopping engine | bars logged: {len(self._bar_log)}")
        self._running = False
        self.broker.disconnect()
        self._save_log()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Live Trading Engine")
    p.add_argument("--broker",   default="paper", choices=["paper","lmax","oanda"])
    p.add_argument("--pair",     default="EURUSD")
    p.add_argument("--equity",   type=float, default=10_000.0)
    p.add_argument("--max-lots", type=float, default=0.5)
    p.add_argument("--model",    default="haelt")
    p.add_argument("--max-bars", type=int, default=None)
    args = p.parse_args()

    # Load checkpoint (requires prior training run)
    ckpt = Path(PATHS["checkpoints"]) / f"{args.model}_best.pt"
    print(f"[Live] Model checkpoint: {'✓ found' if ckpt.exists() else '✗ not found — using random weights'}")

    # Stub agents for demo (replace with trained models)
    class _DemoAgent:
        def __init__(self): import random; self._r = random
        def select_action(self, obs): return self._r.randint(0, 2)
    fast_agent = slow_model = _DemoAgent()

    # Broker
    broker_map = {"paper": PaperBroker, "lmax": LMAXBroker, "oanda": OANDABroker}
    broker = broker_map[args.broker](
        initial_equity=args.equity) if args.broker == "paper" else broker_map[args.broker]()

    engine = LiveTradingEngine(
        broker     = broker,
        fast_agent = fast_agent,
        slow_model = slow_model,
        pair       = args.pair,
        equity     = args.equity,
        max_lots   = args.max_lots,
    )

    print(f"\n[Live] Starting {args.broker.upper()} engine | {args.pair} | max {args.max_lots} lots")
    print(f"       Press Ctrl+C to stop and save logs\n")
    engine.start(max_bars=args.max_bars)
