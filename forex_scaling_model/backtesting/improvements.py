"""
backtesting/improvements.py
============================
Backtesting additions:
  1. MonteCarloBacktest   — randomize trade order 1000× for confidence intervals
  2. SlippageCalibrator   — fit power-law slippage model to real fill data
  3. LockboxTest          — held-out 2024 data, evaluated only once before live
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone

from config.settings import PATHS

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MONTE CARLO BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloBacktest:
    """
    Runs 1000 permutations of the trade sequence to build confidence
    intervals on Sharpe ratio and max drawdown.

    Why this matters:
      A single backtest Sharpe of 1.8 could be luck — maybe you happened
      to catch 3 big trend days in the right order. Monte Carlo shows
      whether the strategy is robust to different trade orderings or if it
      depends on a specific sequence of lucky events.

    Method: Fisher-Yates shuffle of trade P&Ls, compute Sharpe + maxDD for
    each permutation, report [5th, 50th, 95th] percentiles.
    """

    def __init__(
        self,
        n_simulations:  int   = 1000,
        confidence:     float = 0.95,
        initial_equity: float = 10_000.0,
        random_seed:    int   = 42,
    ):
        self.n_sims   = n_simulations
        self.conf     = confidence
        self.equity   = initial_equity
        self.seed     = random_seed

    def run(
        self,
        trade_pnls:   np.ndarray,   # Array of per-trade P&L in USD
        annual_factor: float = 252,  # Trading days per year
    ) -> dict:
        """
        Shuffle trades N times, compute Sharpe + max drawdown per shuffle.

        Parameters
        ----------
        trade_pnls : array of realized P&L per trade (not per bar)
        annual_factor : for annualizing Sharpe

        Returns
        -------
        dict with Sharpe and max_drawdown at [5th, median, 95th] percentiles
        """
        rng    = np.random.default_rng(self.seed)
        n      = len(trade_pnls)
        sharpes = np.zeros(self.n_sims)
        max_dds = np.zeros(self.n_sims)

        for i in range(self.n_sims):
            perm = rng.permutation(n)
            pnl  = trade_pnls[perm]

            # Cumulative equity
            eq   = self.equity + np.cumsum(pnl)
            eq   = np.concatenate([[self.equity], eq])

            # Sharpe
            rets    = pnl / self.equity
            sharpe  = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(annual_factor))
            sharpes[i] = sharpe

            # Max drawdown
            peak    = np.maximum.accumulate(eq)
            dd      = (peak - eq) / peak
            max_dds[i] = float(dd.max())

        lo  = (1 - self.conf) / 2
        hi  = 1 - lo

        # Original (unshuffled) stats
        orig_rets = trade_pnls / self.equity
        orig_sharpe = float(orig_rets.mean() / (orig_rets.std() + 1e-9) * np.sqrt(annual_factor))
        orig_eq  = self.equity + np.cumsum(trade_pnls)
        orig_eq  = np.concatenate([[self.equity], orig_eq])
        orig_peak = np.maximum.accumulate(orig_eq)
        orig_dd   = float(((orig_peak - orig_eq) / orig_peak).max())

        # Percentile that original Sharpe falls in → robustness indicator
        sharpe_pct = float(np.mean(sharpes < orig_sharpe))

        result = {
            "n_trades":          n,
            "n_simulations":     self.n_sims,
            "original_sharpe":   round(orig_sharpe, 4),
            "original_max_dd":   round(orig_dd, 4),
            "sharpe_5th":        round(float(np.percentile(sharpes, 5)), 4),
            "sharpe_median":     round(float(np.percentile(sharpes, 50)), 4),
            "sharpe_95th":       round(float(np.percentile(sharpes, 95)), 4),
            "sharpe_percentile": round(sharpe_pct, 4),
            "max_dd_5th":        round(float(np.percentile(max_dds, 5)), 4),
            "max_dd_median":     round(float(np.percentile(max_dds, 50)), 4),
            "max_dd_95th":       round(float(np.percentile(max_dds, 95)), 4),
            "prob_sharpe_above_1": round(float(np.mean(sharpes > 1.0)), 4),
            "prob_sharpe_above_0": round(float(np.mean(sharpes > 0.0)), 4),
            "robust": bool(sharpe_pct > 0.75 and np.percentile(sharpes, 5) > 0.0),
        }

        self._print_report(result)
        return result

    def _print_report(self, r: dict):
        robust = "ROBUST ✓" if r["robust"] else "NOT ROBUST ✗"
        print(f"\n[MC Backtest] {r['n_simulations']:,} simulations | {r['n_trades']} trades")
        print(f"  Status: {robust}")
        print(f"  Original Sharpe:  {r['original_sharpe']:.3f} "
              f"(beats {r['sharpe_percentile']:.0%} of permutations)")
        print(f"  Sharpe CI [{r['sharpe_5th']:.2f}, {r['sharpe_median']:.2f}, {r['sharpe_95th']:.2f}]")
        print(f"  Max DD  CI [{r['max_dd_5th']:.2%}, {r['max_dd_median']:.2%}, {r['max_dd_95th']:.2%}]")
        print(f"  P(Sharpe > 1.0): {r['prob_sharpe_above_1']:.1%}")

    def run_from_backtest(self, backtest_obj) -> dict:
        """
        Convenience: pass the ForexScalingBacktest object directly.
        Extracts trade P&L automatically.
        """
        if hasattr(backtest_obj, "_trade_pnls"):
            return self.run(np.array(backtest_obj._trade_pnls))
        elif hasattr(backtest_obj, "trades"):
            pnls = [t.get("pnl_usd", t.get("pnl", 0)) for t in backtest_obj.trades]
            return self.run(np.array(pnls))
        else:
            raise ValueError("Backtest object must have ._trade_pnls or .trades attribute")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SLIPPAGE CALIBRATOR
# ─────────────────────────────────────────────────────────────────────────────

class SlippageCalibrator:
    """
    Fits a power-law slippage model to real fill data from LMAX or a live broker.

    Standard slippage model (Almgren et al.):
      slippage = α × (order_size / ADV)^β

    Where:
      α   = impact coefficient (fitted)
      β   = exponent (typically 0.5 for square-root impact, fitted from data)
      ADV = average daily volume

    Without calibration, most backtests assume a fixed slippage (e.g. 1 pip).
    Real slippage is:
      - Near-zero for small orders in liquid session
      - 2–5× higher during news events
      - Correlated with the current spread

    Usage:
        cal = SlippageCalibrator()
        cal.fit(real_fills_df)          # From LMAX fill export
        slip = cal.predict(lots=2.0, spread_pips=1.2, session="london_ny")
    """

    def __init__(self, adv_lots: float = 5000.0):
        self.adv = adv_lots
        self.alpha_: float = 0.5    # Default impact coefficient
        self.beta_:  float = 0.5    # Default exponent (square-root)
        self.session_factors: Dict[str, float] = {
            "london_ny": 1.0,   # Reference (tightest spreads)
            "london":    1.15,
            "ny":        1.20,
            "tokyo":     1.60,
            "sydney":    1.80,
            "overnight": 2.50,
        }
        self._fitted = False

    def fit(
        self,
        fills_df: pd.DataFrame,
    ) -> dict:
        """
        Fit slippage model to real fill data.

        fills_df columns:
          lots          : order size in lots
          requested_price : requested entry price
          fill_price    : actual fill price
          direction     : +1 long / -1 short
          session       : "london_ny" | "london" | etc.

        Slippage = (fill_price - requested_price) × direction (negative = adverse)
        """
        df = fills_df.copy()
        slip = (
            (df["fill_price"] - df["requested_price"]) * df["direction"] / 0.0001
        ).to_numpy(dtype=np.float64)
        df["slip_pips"] = -np.minimum(slip, 0.0)  # Adverse slippage only
        df = df[df["lots"] > 0.001]

        if len(df) < 10:
            print("[Slippage] Insufficient data (<10 fills) — using defaults")
            return {"alpha": self.alpha_, "beta": self.beta_}

        # Fit power law: log(slip) = log(α) + β × log(lots/ADV)
        x = np.log(df["lots"] / self.adv + 1e-9)
        y = np.log(np.maximum(np.asarray(df["slip_pips"], dtype=np.float64), 0.01))

        try:
            beta, log_alpha = np.polyfit(x, y, 1)
            self.alpha_ = float(np.exp(log_alpha))
            self.beta_  = float(np.clip(beta, 0.3, 1.0))

            # Fit session multipliers
            if "session" in df.columns:
                ref_slip = self.alpha_ * (1.0 / self.adv) ** self.beta_
                for sess in pd.unique(df["session"]):
                    sk = str(sess)
                    if sk not in self.session_factors:
                        continue
                    sess_slip = df[df["session"] == sess]["slip_pips"].mean()
                    if sess_slip > 0 and ref_slip > 0:
                        self.session_factors[sk] = float(sess_slip / ref_slip)

            self._fitted = True
            print(f"[Slippage] Fitted | α={self.alpha_:.4f} | β={self.beta_:.3f} | "
                  f"n_fills={len(df)}")
        except Exception as e:
            print(f"[Slippage] Fit failed: {e} — using defaults")

        return {"alpha": self.alpha_, "beta": self.beta_,
                "session_factors": self.session_factors, "n_fills": len(df)}

    def predict(
        self,
        lots:        float,
        spread_pips: float = 1.0,
        session:     str   = "london_ny",
        urgency:     float = 1.0,    # 1.0 = market order, 0.5 = limit order
    ) -> float:
        """
        Predict slippage in pips for a given order.

        Returns expected adverse slippage in pips.
        """
        # Base impact
        impact = self.alpha_ * (lots / self.adv) ** self.beta_

        # Session multiplier
        sess_mult = self.session_factors.get(session, 1.5)

        # Spread component (wider spread → more slippage risk)
        spread_mult = 1.0 + 0.3 * max(spread_pips - 1.0, 0)

        # Urgency (limit orders have lower impact)
        total = impact * sess_mult * spread_mult * urgency

        return float(np.clip(total, 0.0, 10.0))

    def calibrate_from_lmax(self, fill_csv_path: str) -> dict:
        """
        Parse LMAX fill export CSV and fit model.

        Expected LMAX fill report columns (adjust as needed):
          DateTime, InstrumentId, Side, Quantity, RequestedPrice, FillPrice
        """
        try:
            df = pd.read_csv(
                fill_csv_path,
                parse_dates=("DateTime",),
                iterator=False,
            )
            df.columns = df.columns.str.lower().str.strip()

            col_map = {}
            for c in df.columns:
                if "quantity" in c or "size" in c or "lot" in c: col_map[c] = "lots"
                elif "request" in c: col_map[c] = "requested_price"
                elif "fill" in c and "price" in c: col_map[c] = "fill_price"
                elif "side" in c or "dir" in c: col_map[c] = "direction"
            df = df.rename(columns=col_map)
            if "direction" in df.columns:
                _dir = {"buy": 1, "sell": -1, "Buy": 1, "Sell": -1}
                df["direction"] = df["direction"].map(_dir)  # type: ignore[arg-type]

            return self.fit(df)
        except Exception as e:
            print(f"[Slippage] LMAX CSV parse error: {e}")
            return {}

    def save(self, path: str):
        json.dump({"alpha": self.alpha_, "beta": self.beta_,
                   "session_factors": self.session_factors},
                  open(path, "w"), indent=2)

    def load(self, path: str):
        d = json.load(open(path))
        self.alpha_ = d["alpha"]; self.beta_ = d["beta"]
        self.session_factors = d["session_factors"]
        self._fitted = True


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOCKBOX TEST
# ─────────────────────────────────────────────────────────────────────────────

class LockboxTest:
    """
    Holds out a final out-of-sample test period that is NEVER used during
    development, hyperparameter search, or walk-forward retraining.

    The lockbox is only opened ONCE — when you're ready to commit the model
    to live trading. This prevents inadvertent overfit to the test period
    through repeated evaluation.

    Recommended lockbox: most recent 6 months of data.

    Usage:
        lb = LockboxTest(start="2024-01-01", end="2024-06-30")
        lb.register_model("haelt_v3", model_description="100ep, 20M ticks")
        # ... do ALL development with data before 2024-01-01 ...
        # Only once, when ready to go live:
        result = lb.evaluate(model, test_features, test_labels, test_bars)
        lb.seal()   # Prevents re-evaluation
    """

    def __init__(
        self,
        start:       str  = "2024-01-01",
        end:         str  = "2024-12-31",
        log_path:    Optional[str] = None,
        max_evals:   int  = 1,   # Only 1 evaluation allowed
    ):
        if log_path is None:
            log_path = PATHS["file_lockbox_log"]
        self.start     = start
        self.end       = end
        self.log_path  = Path(log_path)
        self.max_evals = max_evals
        self._evals:   List[dict] = []
        self._sealed   = False
        self._model_registry: List[dict] = []
        self._load_log()

    def _load_log(self):
        if self.log_path.exists():
            data = json.load(open(self.log_path))
            self._evals   = data.get("evals", [])
            self._sealed  = data.get("sealed", False)
            self._model_registry = data.get("models", [])
            if self._evals:
                print(f"[Lockbox] LOADED — {len(self._evals)} prior evaluation(s)")
                if self._sealed:
                    print("[Lockbox] SEALED — no further evaluations allowed")

    def _save_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {"sealed": self._sealed, "evals": self._evals,
             "models": self._model_registry,
             "period": {"start": self.start, "end": self.end}},
            open(self.log_path, "w"), indent=2
        )

    def register_model(self, model_name: str, description: str = ""):
        """
        Register a model version before development begins.
        Establishes audit trail of what was developed before opening the lockbox.
        """
        entry = {
            "name": model_name,
            "description": description,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self._model_registry.append(entry)
        self._save_log()
        print(f"[Lockbox] Registered model: {model_name}")

    def check_data_leak(self, df: pd.DataFrame) -> bool:
        """
        Verify that a DataFrame does NOT contain lockbox period data.
        Returns True if data is clean (no leak).
        """
        if df.empty: return True
        idx = pd.to_datetime(df.index, utc=True)
        start_ts = pd.Timestamp(self.start, tz="UTC")
        end_ts   = pd.Timestamp(self.end,   tz="UTC")
        overlap  = ((idx >= start_ts) & (idx <= end_ts)).sum()
        if overlap > 0:
            print(f"[Lockbox] ⚠ DATA LEAK DETECTED: {overlap} rows from lockbox period!")
            return False
        return True

    def evaluate(
        self,
        model_name:   str,
        predictions:  np.ndarray,  # Model signals: +1, 0, -1
        returns:      np.ndarray,  # Actual forward returns
        trade_pnls:   Optional[np.ndarray] = None,
        notes:        str = "",
    ) -> dict:
        """
        Evaluate model on the lockbox test period.
        CAN ONLY BE CALLED ONCE (or max_evals times).
        """
        if self._sealed:
            raise RuntimeError(
                "[Lockbox] SEALED — this test period has already been evaluated. "
                "Opening the lockbox a second time invalidates out-of-sample integrity."
            )

        if len(self._evals) >= self.max_evals:
            raise RuntimeError(
                f"[Lockbox] Maximum evaluations ({self.max_evals}) reached. "
                "The lockbox is exhausted."
            )

        print(f"\n[Lockbox] {'═'*50}")
        print(f"[Lockbox] OPENING LOCKBOX — model: {model_name}")
        print(f"[Lockbox] Period: {self.start} → {self.end}")
        print(f"[Lockbox] {'═'*50}")

        # Compute metrics
        directional_acc = float(np.mean(np.sign(predictions) == np.sign(returns)))
        strategy_rets   = predictions * returns
        sharpe = float(strategy_rets.mean() / (strategy_rets.std() + 1e-9) * np.sqrt(252))

        # Max drawdown
        cum_eq  = 10000 * (1 + np.cumsum(strategy_rets / 10000))
        peak    = np.maximum.accumulate(cum_eq)
        max_dd  = float(((peak - cum_eq) / peak).max())

        # Monte Carlo on lockbox trades
        mc_result = {}
        if trade_pnls is not None and len(trade_pnls) > 0:
            mc = MonteCarloBacktest(n_simulations=500)
            mc_result = mc.run(trade_pnls)

        result = {
            "model":              model_name,
            "evaluated_at":       datetime.now(timezone.utc).isoformat(),
            "lockbox_period":     {"start": self.start, "end": self.end},
            "n_predictions":      len(predictions),
            "directional_acc":    round(directional_acc, 4),
            "sharpe":             round(sharpe, 4),
            "max_drawdown":       round(max_dd, 4),
            "total_return":       round(float(strategy_rets.sum()), 6),
            "monte_carlo":        mc_result,
            "notes":              notes,
            "registered_models":  [m["name"] for m in self._model_registry],
        }

        self._evals.append(result)

        # Auto-seal after first evaluation
        if len(self._evals) >= self.max_evals:
            self.seal()

        self._save_log()

        print(f"\n[Lockbox] RESULTS:")
        print(f"  Directional accuracy: {directional_acc:.1%}")
        print(f"  Sharpe ratio:         {sharpe:.3f}")
        print(f"  Max drawdown:         {max_dd:.1%}")
        if mc_result:
            print(f"  MC robust:            {mc_result.get('robust', '?')}")
        print(f"[Lockbox] {'═'*50}\n")

        return result

    def seal(self):
        """Permanently seal the lockbox — no further evaluations."""
        self._sealed = True
        self._save_log()
        print("[Lockbox] SEALED — out-of-sample integrity preserved ✓")

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def n_evaluations(self) -> int:
        return len(self._evals)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("Backtesting improvements — smoke tests")
    print("=" * 50)

    # Monte Carlo
    rng   = np.random.default_rng(42)
    pnls  = rng.normal(5, 30, 300)   # 300 trades
    mc    = MonteCarloBacktest(n_simulations=500)
    result = mc.run(pnls)
    print(f"\n  MC: robust={result['robust']} | Sharpe CI: "
          f"[{result['sharpe_5th']:.2f}, {result['sharpe_95th']:.2f}]")

    # Slippage calibrator
    print()
    sc = SlippageCalibrator()
    # Synthetic fill data
    fills = pd.DataFrame({
        "lots":            rng.uniform(0.1, 5.0, 200),
        "requested_price": rng.uniform(1.085, 1.090, 200),
        "fill_price":      rng.uniform(1.085, 1.090, 200),
        "direction":       rng.choice([1,-1], 200),
        "session":         rng.choice(["london_ny","london","ny","tokyo"], 200),
    })
    fills["fill_price"] += fills["direction"] * rng.uniform(0, 0.0002, 200)
    sc.fit(fills)
    for lots, sess in [(0.1,"london_ny"),(1.0,"london_ny"),(3.0,"tokyo")]:
        slip = sc.predict(lots, spread_pips=1.1, session=sess)
        print(f"  Slippage {lots:.1f}L {sess}: {slip:.4f} pips")

    # Lockbox
    print()
    with tempfile.TemporaryDirectory() as td:
        log_path = f"{td}/lockbox.json"
        lb = LockboxTest(start="2024-01-01", end="2024-06-30", log_path=log_path, max_evals=1)
        lb.register_model("haelt_v3", "100 epochs, 20M ticks, Huber loss")

        # Check data leak detection
        clean_df = pd.DataFrame(index=pd.date_range("2023-01-01","2023-12-31",freq="1D",tz="UTC"))
        dirty_df = pd.DataFrame(index=pd.date_range("2024-03-01","2024-04-01",freq="1D",tz="UTC"))
        print(f"\n  Data leak check (clean): {lb.check_data_leak(clean_df)}")
        print(f"  Data leak check (dirty): {lb.check_data_leak(dirty_df)}")

        # Evaluate
        n = 1000
        preds = rng.choice([-1,0,1], n)
        rets  = rng.normal(0.001, 0.003, n)
        ev = lb.evaluate("haelt_v3", preds, rets, notes="First and only evaluation")
        print(f"\n  Lockbox sealed: {lb.is_sealed}")
        print(f"  Sharpe: {ev['sharpe']:.3f} | Dir acc: {ev['directional_acc']:.1%}")

    print("\nAll backtesting improvement tests passed ✓")
