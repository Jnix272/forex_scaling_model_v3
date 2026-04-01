"""
backtesting/backtest.py
=======================
Production-grade backtesting engine for forex scaling models.

Accounts for:
  - Bid-ask spread (the Golden Rule)
  - Commission per lot
  - Slippage (execution lag)
  - Market impact (Square Root Law for large orders)
  - Scaling in/out (partial fills and average entry tracking)
  - Dynamic stop-loss management

The primary reason most scalping models fail in production is that they
don't account for execution lag and real transaction costs. This engine
enforces realistic execution at every step.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


# ─────────────────────────────────────────────────────────────────────────────
# TRADE RECORDS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Record of a single trade execution."""
    trade_id: int
    entry_time: pd.Timestamp
    entry_price: float
    entry_lots: float
    direction: int           # +1 = long, -1 = short
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_lots: Optional[float] = None
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    commission: float = 0.0
    slippage_pips: float = 0.0
    exit_reason: str = ""    # 'stop_loss', 'take_profit', 'signal', 'scale_out', 'eod'
    scale_additions: List[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ForexScalingBacktest:
    """
    Event-driven backtesting engine for forex scaling strategies.

    Supports:
      - Scale-in (pyramiding and martingale) at multiple price levels
      - Scale-out (partial profit taking) at predefined targets
      - Dynamic stop-loss trailing
      - Realistic transaction cost modeling

    Usage
    -----
        bt = ForexScalingBacktest(bars_df, signals_df)
        results = bt.run()
        bt.print_performance()
        equity_curve = bt.get_equity_curve()
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        signals: pd.DataFrame,
        initial_equity: float = 10_000.0,
        lot_size: float = 10_000.0,
        commission_per_lot: float = 3.5,
        slippage_pips: float = 0.5,
        pip_size: float = 0.0001,
        pip_value_per_lot: float = 1.0,
        max_lots: float = 3.0,
        execution_delay_bars: int = 1,    # Simulate 1-bar execution delay
        use_bid_ask: bool = True,          # Golden Rule: trade on bid/ask, not mid
        daily_volume_lots: float = 500.0,
        apply_market_impact: bool = True,
    ):
        """
        Parameters
        ----------
        bars     : OHLCV bars with 'open','high','low','close','bid_close','ask_close','spread_avg'
        signals  : DataFrame with columns: 'action' (ScalingAction int), 'stop_loss', 'take_profit'
        """
        self.bars = bars
        self.signals = signals.reindex(bars.index).ffill()

        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.lot_size = lot_size
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.pip_size = pip_size
        self.pip_value_per_lot = pip_value_per_lot
        self.max_lots = max_lots
        self.execution_delay = execution_delay_bars
        self.use_bid_ask = use_bid_ask
        self.daily_volume_lots = daily_volume_lots
        self.apply_market_impact = apply_market_impact

        # State
        self.position: float = 0.0         # Current net lots
        self.avg_entry_price: float = 0.0   # VWAP entry price
        self.current_stop: float = 0.0
        self.current_tp: float = 0.0
        self.holding_bars: int = 0

        # Records
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_pnl: List[float] = []
        self._trade_counter: int = 0
        self._open_trade: Optional[Trade] = None

    # ── Price helpers ────────────────────────────────────────────────────────

    def _get_execution_price(self, idx: int, direction: int, lots: float) -> float:
        """
        Realistic execution price including:
          - Spread: buys at ask, sells at bid
          - Slippage: additional friction from execution lag
          - Market impact: Square Root Law for large orders
        """
        row = self.bars.iloc[idx]

        # Use bid/ask if available (Golden Rule)
        if self.use_bid_ask and "bid_close" in self.bars.columns and "ask_close" in self.bars.columns:
            base_price = row["ask_close"] if direction > 0 else row["bid_close"]
        else:
            spread_half = row.get("spread_avg", 0.0001) / 2
            base_price = row["close"] + direction * spread_half

        # Slippage
        slippage = direction * self.slippage_pips * self.pip_size
        price = base_price + slippage

        # Market impact (Square Root Law)
        if self.apply_market_impact and lots > 0:
            atr = row.get("spread_avg", 0.0005)
            impact_fraction = atr * np.sqrt(abs(lots) / self.daily_volume_lots)
            price += direction * impact_fraction

        return float(price)

    def _compute_cost(self, lots: float) -> float:
        """Commission per round-turn."""
        return abs(lots) * self.commission_per_lot

    # ── Trade execution ──────────────────────────────────────────────────────

    def _open_position(self, idx: int, direction: int, lots: float,
                       stop_loss: float, take_profit: float) -> Trade:
        """Open a new position."""
        exec_price = self._get_execution_price(idx, direction, lots)
        cost = self._compute_cost(lots)
        self.equity -= cost

        self.position = direction * lots
        self.avg_entry_price = exec_price
        self.current_stop = stop_loss
        self.current_tp = take_profit
        self.holding_bars = 0

        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            entry_time=self.bars.index[idx],
            entry_price=exec_price,
            entry_lots=lots,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=cost,
            slippage_pips=self.slippage_pips,
        )
        self._open_trade = trade
        return trade

    def _scale_in(self, idx: int, lots: float):
        """Add to existing position (pyramid or martingale)."""
        if abs(self.position) + lots > self.max_lots:
            lots = self.max_lots - abs(self.position)
        if lots <= 0:
            return

        direction = int(np.sign(self.position))
        exec_price = self._get_execution_price(idx, direction, lots)
        cost = self._compute_cost(lots)
        self.equity -= cost

        # Update weighted average entry
        total_lots = abs(self.position) + lots
        self.avg_entry_price = (
            abs(self.position) * self.avg_entry_price + lots * exec_price
        ) / total_lots

        self.position += direction * lots

        if self._open_trade:
            self._open_trade.scale_additions.append({
                "time": self.bars.index[idx],
                "price": exec_price,
                "lots": lots,
                "cost": cost,
            })
            self._open_trade.commission += cost

    def _close_position(self, idx: int, fraction: float = 1.0,
                        exit_reason: str = "signal") -> float:
        """Close all or part of position. Returns realised P&L in USD."""
        if self.position == 0:
            return 0.0

        close_lots = abs(self.position) * fraction
        direction = int(np.sign(self.position))
        exec_price = self._get_execution_price(idx, -direction, close_lots)
        cost = self._compute_cost(close_lots)

        pnl_pips = direction * (exec_price - self.avg_entry_price) / self.pip_size
        pnl_usd = pnl_pips * self.pip_value_per_lot * close_lots - cost
        self.equity += pnl_usd

        self.position -= direction * close_lots
        if abs(self.position) < 0.001:
            self.position = 0.0
            if self._open_trade:
                self._open_trade.exit_time = self.bars.index[idx]
                self._open_trade.exit_price = exec_price
                self._open_trade.exit_lots = close_lots
                self._open_trade.pnl_pips = pnl_pips
                self._open_trade.pnl_usd = pnl_usd
                self._open_trade.commission += cost
                self._open_trade.exit_reason = exit_reason
                self.trades.append(self._open_trade)
                self._open_trade = None
                self.holding_bars = 0

        return pnl_usd

    # ── Stop/TP checks ───────────────────────────────────────────────────────

    def _check_stops(self, idx: int) -> bool:
        """
        Check if price has hit stop-loss or take-profit during current bar.
        Returns True if position was closed.
        """
        if self.position == 0:
            return False

        row = self.bars.iloc[idx]
        direction = np.sign(self.position)

        # Stop-loss check
        if direction > 0 and row["low"] <= self.current_stop:
            self._close_position(idx, fraction=1.0, exit_reason="stop_loss")
            return True
        if direction < 0 and row["high"] >= self.current_stop:
            self._close_position(idx, fraction=1.0, exit_reason="stop_loss")
            return True

        # Take-profit check
        if direction > 0 and row["high"] >= self.current_tp:
            self._close_position(idx, fraction=0.5, exit_reason="scale_out_tp")
            # Trail stop to breakeven after partial profit
            if self.position != 0:
                self.current_stop = max(self.current_stop, self.avg_entry_price)
            return False  # Position still open (partially)

        if direction < 0 and row["low"] <= self.current_tp:
            self._close_position(idx, fraction=0.5, exit_reason="scale_out_tp")
            if self.position != 0:
                self.current_stop = min(self.current_stop, self.avg_entry_price)
            return False

        return False

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the backtest bar by bar.

        Returns
        -------
        pd.DataFrame: Bar-by-bar equity and P&L records
        """
        print(f"[Backtest] Running {len(self.bars):,} bars | "
              f"Initial equity: ${self.initial_equity:,.2f}")

        records = []

        for i, (ts, row) in enumerate(self.bars.iterrows()):
            # Apply execution delay (signals from bar i arrive at bar i+delay)
            signal_idx = max(0, i - self.execution_delay)
            if signal_idx >= len(self.signals):
                break

            sig = self.signals.iloc[signal_idx]
            action = int(sig.get("action", 0))
            stop_loss = float(sig.get("stop_loss", 0))
            take_profit = float(sig.get("take_profit", 0))
            lots_to_trade = float(sig.get("lots", 0.1))

            # 1. Check stops/take-profits first
            stopped = self._check_stops(i)

            if not stopped:
                # 2. Execute signal
                if action == 1 and self.position == 0:        # Open long
                    self._open_position(i, +1, lots_to_trade, stop_loss, take_profit)

                elif action == 2 and self.position == 0:      # Open short
                    self._open_position(i, -1, lots_to_trade, stop_loss, take_profit)

                elif action == 3 and self.position != 0:      # Scale in 25%
                    self._scale_in(i, lots_to_trade * 0.25)

                elif action == 4 and self.position != 0:      # Scale in 50%
                    self._scale_in(i, lots_to_trade * 0.50)

                elif action == 6 and self.position != 0:      # Scale out 25%
                    self._close_position(i, 0.25, "scale_out_25")

                elif action == 7 and self.position != 0:      # Scale out 50%
                    self._close_position(i, 0.50, "scale_out_50")

                elif action == 9 and self.position != 0:      # Close all
                    self._close_position(i, 1.0, "signal_exit")

            # Update holding time
            if self.position != 0:
                self.holding_bars += 1

            # Mark-to-market unrealised P&L
            if self.position != 0:
                direction = np.sign(self.position)
                unrealised = (row["close"] - self.avg_entry_price) * direction * abs(self.position) * self.lot_size
            else:
                unrealised = 0.0

            self.peak_equity = max(self.peak_equity, self.equity + unrealised)
            drawdown = max(0, (self.peak_equity - (self.equity + unrealised)) / self.peak_equity)

            self.equity_curve.append(self.equity)
            records.append({
                "timestamp": ts,
                "equity": self.equity,
                "unrealised_pnl": unrealised,
                "total_value": self.equity + unrealised,
                "position": self.position,
                "drawdown": drawdown,
                "holding_bars": self.holding_bars,
            })

        # Force-close at end
        if self.position != 0:
            self._close_position(len(self.bars) - 1, 1.0, "end_of_data")

        self.results_df = pd.DataFrame(records).set_index("timestamp")
        return self.results_df

    # ── Performance reporting ────────────────────────────────────────────────

    def performance_metrics(self) -> dict:
        """Compute comprehensive performance statistics."""
        if not self.trades:
            return {"error": "No trades executed"}

        total_pnl = sum(t.pnl_usd for t in self.trades)
        total_cost = sum(t.commission for t in self.trades)
        winning_trades = [t for t in self.trades if t.pnl_usd > 0]
        losing_trades = [t for t in self.trades if t.pnl_usd < 0]

        returns = self.results_df["equity"].pct_change().dropna()
        sharpe = 0.0
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)  # 1-min bars

        equity_arr = self.results_df["total_value"].values
        rolling_max = np.maximum.accumulate(equity_arr)
        drawdowns = (rolling_max - equity_arr) / (rolling_max + 1e-9)
        max_dd = drawdowns.max()

        avg_bars_held = np.mean([
            (t.exit_time - t.entry_time).total_seconds() / 60
            if t.exit_time else 0 for t in self.trades
        ])

        return {
            "total_return_pct": (self.equity / self.initial_equity - 1) * 100,
            "total_pnl_usd": total_pnl,
            "total_commission_usd": total_cost,
            "net_pnl_usd": total_pnl - total_cost,
            "n_trades": len(self.trades),
            "win_rate_pct": len(winning_trades) / len(self.trades) * 100,
            "avg_win_usd": np.mean([t.pnl_usd for t in winning_trades]) if winning_trades else 0,
            "avg_loss_usd": np.mean([t.pnl_usd for t in losing_trades]) if losing_trades else 0,
            "win_loss_ratio": (
                abs(np.mean([t.pnl_usd for t in winning_trades]))
                / max(abs(np.mean([t.pnl_usd for t in losing_trades])), 0.01)
                if winning_trades and losing_trades else 0
            ),
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "avg_holding_minutes": avg_bars_held,
            "profit_factor": (
                sum(t.pnl_usd for t in winning_trades)
                / max(abs(sum(t.pnl_usd for t in losing_trades)), 0.01)
                if losing_trades else float("inf")
            ),
        }

    def print_performance(self):
        """Print formatted performance report."""
        m = self.performance_metrics()
        print("\n" + "=" * 55)
        print("BACKTEST PERFORMANCE REPORT")
        print("=" * 55)
        for k, v in m.items():
            if k == "error":
                print(f"  ERROR: {v}")
            elif isinstance(v, float):
                print(f"  {k:<30} {v:>10.4f}")
            else:
                print(f"  {k:<30} {v:>10}")
        print("=" * 55)

    def get_equity_curve(self) -> pd.Series:
        """Return the total equity curve (including unrealised P&L)."""
        if hasattr(self, "results_df"):
            return self.results_df["total_value"]
        return pd.Series(self.equity_curve)

    def get_trade_log(self) -> pd.DataFrame:
        """Return all trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "trade_id": t.trade_id,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": "Long" if t.direction > 0 else "Short",
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "lots": t.entry_lots,
            "pnl_pips": t.pnl_pips,
            "pnl_usd": t.pnl_usd,
            "commission": t.commission,
            "exit_reason": t.exit_reason,
            "n_scale_adds": len(t.scale_additions),
        } for t in self.trades])


if __name__ == "__main__":
    # Smoke test with synthetic data
    import sys; sys.path.insert(0, "..")
    from data.data_ingestion import load_or_generate, ForexDataPipeline

    ticks = load_or_generate(n_rows=20_000)
    pipeline = ForexDataPipeline(bar_freq="5min")
    bars = pipeline.run(ticks)

    # Create dummy signals (random strategy for testing)
    rng = np.random.default_rng(42)
    signals = pd.DataFrame(index=bars.index)
    signals["action"] = rng.choice([0, 1, 2, 9], size=len(bars), p=[0.7, 0.1, 0.1, 0.1])
    signals["lots"] = 0.1
    signals["stop_loss"] = bars["close"] - 0.0010
    signals["take_profit"] = bars["close"] + 0.0015

    bt = ForexScalingBacktest(bars=bars, signals=signals, initial_equity=10_000)
    results = bt.run()
    bt.print_performance()

    trades = bt.get_trade_log()
    if len(trades) > 0:
        print(f"\nSample trades:\n{trades.head(5).to_string()}")
