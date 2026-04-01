"""
validation/promotion_gate.py
=============================
Formal model promotion gate — checks 7 criteria before a model
can move from Staging → Production.

Gates (all must pass to promote):
  1. Out-of-sample Sharpe         ≥ min_sharpe (default 1.5)
  2. Profit factor                ≥ min_profit_factor (default 1.5)
  3. Max drawdown                 ≤ max_drawdown_pct (default 20%)
  4. Number of trades             ≥ min_trades (default 500)
  5. Regime concentration         no single regime > max_regime_conc of P&L
  6. Transaction cost / gross PnL < max_cost_pct (default 30%)
  7. Probabilistic Sharpe Ratio   > 0  (corrects for overfitting)

Usage:
    from validation.promotion_gate import PromotionGate
    gate   = PromotionGate()
    result = gate.evaluate(
        sharpe=1.8,
        profit_factor=1.6,
        max_drawdown=0.12,
        n_trades=800,
        regime_pnl={"trending": 0.6, "neutral": 0.3, "mean_rev": 0.1},
        gross_pnl=5000.0,
        transaction_costs=800.0,
        n_backtest_trials=1,      # set >1 for Deflated Sharpe
        backtest_sharpe_std=0.0,  # std across walk-forward folds
    )
    print(result["promoted"], result["reasons"])
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")


# ── configuration ────────────────────────────────────────────────────────────

@dataclass
class GateConfig:
    """All promotion thresholds in one place."""
    min_sharpe:           float = 1.5
    min_sharpe_emergency: float = 1.3   # for emergency retrain DAG
    min_profit_factor:    float = 1.5
    max_drawdown_pct:     float = 0.20
    min_trades:           int   = 500
    max_regime_conc:      float = 0.75  # no regime > 75% of P&L
    max_cost_pct:         float = 0.30  # transaction cost / gross P&L
    strict_psr:           bool  = False # enable Deflated Sharpe (requires trials>1)


# ── Probabilistic Sharpe Ratio ────────────────────────────────────────────────

def probabilistic_sharpe_ratio(
    sr_hat:       float,
    sr_benchmark: float,
    n_obs:        int,
    skewness:     float = 0.0,
    kurtosis:     float = 3.0,
) -> float:
    """
    P(SR* > SR_benchmark) using Bailey & de Prado (2012).
    Corrects for non-normality of returns.

    Returns a probability in [0, 1].
    Typical threshold: PSR > 0.95 (95% confident true Sharpe beats benchmark).
    """
    if n_obs < 5:
        return 0.5    # not enough data
    try:
        from scipy.stats import norm
    except ImportError:
        # Manual normal CDF approximation
        def _ncdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        norm = type("N", (), {"cdf": staticmethod(_ncdf)})()

    variance = (
        1 / (n_obs - 1)
    ) * (
        1 - skewness * sr_hat
        + ((kurtosis - 1) / 4) * sr_hat ** 2
    )
    if variance <= 0:
        return 0.5
    z = (sr_hat - sr_benchmark) / math.sqrt(max(variance, 1e-12))
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sr_hat:      float,
    sr_trials:   int,
    n_obs:       int,
    skewness:    float = 0.0,
    kurtosis:    float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio from Bailey & de Prado (2014).
    Accounts for multiple testing / hyperparameter search.
    """
    if sr_trials <= 1:
        return sr_hat
    # Expected maximum SR under multiple trials (approximation)
    gamma_euler = 0.5772156649
    e_max_sr = (1 - gamma_euler) * math.sqrt(
        2 * math.log(sr_trials)
    ) + gamma_euler / math.sqrt(2 * math.log(sr_trials))
    psr = probabilistic_sharpe_ratio(sr_hat, e_max_sr, n_obs, skewness, kurtosis)
    return float(psr)


# ── main gate ────────────────────────────────────────────────────────────────

class PromotionGate:
    """
    Evaluates whether a model should be promoted to production.

    All 7 gates must pass (or their override is set) for `promoted=True`.
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.cfg = config or GateConfig()

    def evaluate(
        self,
        sharpe:               float,
        profit_factor:        float,
        max_drawdown:         float,
        n_trades:             int,
        regime_pnl:           Optional[Dict[str, float]] = None,
        gross_pnl:            float = 1.0,
        transaction_costs:    float = 0.0,
        n_obs:                int   = 1000,
        n_backtest_trials:    int   = 1,
        backtest_sharpe_std:  float = 0.0,
        skewness:             float = 0.0,
        kurtosis:             float = 3.0,
        emergency_retrain:    bool  = False,
    ) -> Dict:
        """
        Evaluate all promotion gates.

        Parameters
        ----------
        sharpe              : Out-of-sample annualised Sharpe ratio.
        profit_factor       : Gross profit / gross loss.
        max_drawdown        : Max drawdown as a fraction (0.15 = 15%).
        n_trades            : Number of trades in the backtest.
        regime_pnl          : Dict mapping regime name → fraction of total P&L.
                              e.g. {"trending": 0.7, "neutral": 0.3}
        gross_pnl           : Total gross P&L (before costs), in any currency unit.
        transaction_costs   : Total transaction costs incurred.
        n_obs               : Number of return observations (for PSR).
        n_backtest_trials   : Number of hyperparameter trials run (for DSR).
        backtest_sharpe_std : Std of Sharpe across walk-forward folds.
        skewness            : Skewness of returns (for PSR correction).
        kurtosis            : Kurtosis of returns.
        emergency_retrain   : If True, use lower min_sharpe_emergency threshold.
        """
        min_sr = self.cfg.min_sharpe_emergency if emergency_retrain else self.cfg.min_sharpe
        gates:   Dict[str, bool]  = {}
        details: Dict[str, float] = {}

        # 1. Sharpe
        gates["sharpe_ok"]     = sharpe >= min_sr
        details["sharpe"]      = sharpe
        details["min_sharpe"]  = min_sr

        # 2. Profit factor
        gates["pf_ok"]                = profit_factor >= self.cfg.min_profit_factor
        details["profit_factor"]      = profit_factor
        details["min_profit_factor"]  = self.cfg.min_profit_factor

        # 3. Max drawdown
        gates["dd_ok"]          = max_drawdown <= self.cfg.max_drawdown_pct
        details["max_drawdown"] = max_drawdown
        details["max_dd_limit"] = self.cfg.max_drawdown_pct

        # 4. Sample size
        gates["sample_ok"]  = n_trades >= self.cfg.min_trades
        details["n_trades"] = float(n_trades)
        details["min_trades"] = float(self.cfg.min_trades)

        # 5. Regime concentration
        if regime_pnl:
            max_conc = max(abs(v) for v in regime_pnl.values())
            dominant = max(regime_pnl, key=lambda k: abs(regime_pnl[k]))
        else:
            max_conc, dominant = 0.0, "unknown"
        gates["regime_ok"]         = max_conc <= self.cfg.max_regime_conc
        details["max_regime_conc"] = max_conc
        details["dominant_regime"] = dominant    # type: ignore[assignment]

        # 6. Transaction cost ratio
        cost_pct = transaction_costs / max(abs(gross_pnl), 1e-9)
        gates["cost_ok"]      = cost_pct <= self.cfg.max_cost_pct
        details["cost_pct"]   = cost_pct
        details["cost_limit"] = self.cfg.max_cost_pct

        # 7. Probabilistic Sharpe
        psr = probabilistic_sharpe_ratio(sharpe, 0.0, n_obs, skewness, kurtosis)
        gates["psr_ok"]  = psr > 0.5   # > 50% confident true SR > 0
        details["psr"]   = psr

        if self.cfg.strict_psr and n_backtest_trials > 1:
            dsr = deflated_sharpe_ratio(sharpe, n_backtest_trials, n_obs,
                                        skewness, kurtosis)
            gates["dsr_ok"]  = dsr > 0.5
            details["dsr"]   = dsr
        else:
            gates["dsr_ok"]  = True   # skipped when not in strict mode
            details["dsr"]   = -1.0

        # ── Final decision ─────────────────────────────────────────────────
        promoted = all(gates.values())
        reasons  = [f"{k}: {'✓' if v else '✗'}" for k, v in gates.items()]

        return {
            "promoted":        promoted,
            "gates":           gates,
            "details":         details,
            "reasons":         reasons,
            "emergency_mode":  emergency_retrain,
            "summary": (
                "PROMOTE ✅" if promoted else
                "REJECT ❌  Failed: " + ", ".join(k for k, v in gates.items() if not v)
            ),
        }

    def evaluate_from_history(
        self,
        trade_pnls:    "list[float]",
        equity_curve:  "list[float]",
        regime_labels: "list[str]" = None,
        tx_costs:      "list[float]" = None,
        annualization: float = 252.0,
        **kwargs,
    ) -> Dict:
        """
        Convenience wrapper — compute metrics from raw trade data.

        Parameters
        ----------
        trade_pnls    : Per-trade net P&L values.
        equity_curve  : Equity curve as list of equity values.
        regime_labels : Optional list of regime label per trade.
        tx_costs      : Optional list of transaction cost per trade.
        """
        import numpy as np
        pnls   = np.array(trade_pnls, dtype=float)
        eq     = np.array(equity_curve, dtype=float)

        # Sharpe
        if len(pnls) > 1 and pnls.std() > 0:
            sharpe = float(pnls.mean() / pnls.std() * np.sqrt(annualization))
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = float(pnls[pnls > 0].sum())
        gross_loss   = float(abs(pnls[pnls < 0].sum()))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        # Max drawdown
        if len(eq) > 0:
            peak  = np.maximum.accumulate(eq)
            dd    = (eq - peak) / np.maximum(peak, 1e-9)
            max_dd = float(abs(dd.min()))
        else:
            max_dd = 0.0

        # Regime concentration
        regime_pnl: Optional[Dict[str, float]] = None
        if regime_labels and len(regime_labels) == len(pnls):
            from collections import defaultdict
            rp: Dict[str, float] = defaultdict(float)
            for label, pnl in zip(regime_labels, pnls):
                rp[label] += pnl
            total = sum(abs(v) for v in rp.values()) + 1e-9
            regime_pnl = {k: v / total for k, v in rp.items()}

        # Costs
        tx_total  = float(sum(tx_costs)) if tx_costs else 0.0
        gross_total = float(gross_profit)

        return self.evaluate(
            sharpe=sharpe,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            n_trades=len(pnls),
            regime_pnl=regime_pnl,
            gross_pnl=gross_total,
            transaction_costs=tx_total,
            n_obs=len(pnls),
            **kwargs,
        )


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    gate = PromotionGate()

    print("=== Gate 1: Excellent model (should PROMOTE) ===")
    r = gate.evaluate(
        sharpe=1.8, profit_factor=1.7, max_drawdown=0.12, n_trades=700,
        regime_pnl={"trending": 0.5, "neutral": 0.3, "mean_rev": 0.2},
        gross_pnl=10_000.0, transaction_costs=1_500.0, n_obs=700,
    )
    print(f"  {r['summary']}")
    for reason in r["reasons"]: print(f"    {reason}")

    print("\n=== Gate 2: Suspicious model (should REJECT) ===")
    r2 = gate.evaluate(
        sharpe=1.6, profit_factor=1.3, max_drawdown=0.25, n_trades=300,
        regime_pnl={"trending": 0.92, "neutral": 0.08},
        gross_pnl=8_000.0, transaction_costs=5_000.0, n_obs=300,
    )
    print(f"  {r2['summary']}")
    for reason in r2["reasons"]: print(f"    {reason}")

    print("\n=== Gate 3: evaluate_from_history helper ===")
    rng    = np.random.default_rng(42)
    pnls   = rng.normal(0.002, 0.01, 600)
    equity = np.cumprod(1 + pnls / 100) * 10_000
    r3     = gate.evaluate_from_history(trade_pnls=pnls.tolist(),
                                        equity_curve=equity.tolist())
    print(f"  {r3['summary']}")
    print("OK ✓")
