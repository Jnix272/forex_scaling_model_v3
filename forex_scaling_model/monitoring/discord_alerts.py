"""
monitoring/discord_alerts.py
==============================
Discord webhook alerter for the Forex Scaling Model live engine.

Sends structured Discord embeds for 6 alert types:
  🔴 circuit_breaker  — DrawdownAwareExitManager fires close_all
  🌊 drift_detected   — DriftDetector fires
  🔄 emergency_retrain — Retrain DAG triggered by demotion monitor
  ✅ model_promoted   — PromotionGate.evaluate() passes
  ⬇️  model_demoted   — DemotionMonitor triggers rollback
  💸 tca_breach       — Slippage/cost metrics exceed policy limit

Falls back to print() when DISCORD_WEBHOOK_URL is not set.

Usage:
    from monitoring.discord_alerts import DiscordAlerter
    alerter = DiscordAlerter()

    alerter.send("circuit_breaker", {
        "drawdown": "10.5%",
        "equity":   "$8,950",
        "action":   "close_all",
    })

    alerter.send("model_promoted", {
        "model":   "haelt_v4",
        "sharpe":  "1.82",
        "git":     "a3f9c1d",
    })
"""

import json
import os
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

warnings.filterwarnings("ignore")

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ── alert definitions ────────────────────────────────────────────────────────

ALERT_CONFIG: Dict[str, Dict[str, Any]] = {
    "circuit_breaker": {
        "emoji":       "🔴",
        "title":       "Circuit Breaker Triggered",
        "description": "Drawdown limit breached. All positions closed.",
        "color":       0xFF0000,  # red
    },
    "drift_detected": {
        "emoji":       "🌊",
        "title":       "Feature Drift Detected",
        "description": "Model input distribution has shifted significantly.",
        "color":       0xFF8C00,  # orange
    },
    "emergency_retrain": {
        "emoji":       "🔄",
        "title":       "Emergency Retrain Started",
        "description": "Demotion triggered automatic retraining DAG.",
        "color":       0xFFD700,  # gold
    },
    "model_promoted": {
        "emoji":       "✅",
        "title":       "Model Promoted to Production",
        "description": "New model passed all promotion gates.",
        "color":       0x00CC44,  # green
    },
    "model_demoted": {
        "emoji":       "⬇️",
        "title":       "Model Demoted — Rolling Back",
        "description": "Live performance fell below policy thresholds.",
        "color":       0xAA00FF,  # purple
    },
    "tca_breach": {
        "emoji":       "💸",
        "title":       "TCA Policy Breach",
        "description": "Transaction costs exceeded allowed % of gross P&L.",
        "color":       0xFF4444,  # light red
    },
}


# ── alerter class ─────────────────────────────────────────────────────────────

class DiscordAlerter:
    """
    Discord webhook alerter with rate limiting and fallback to print().

    Parameters
    ----------
    webhook_url    : Discord webhook URL. Falls back to DISCORD_WEBHOOK_URL env var.
    min_interval_s : Minimum seconds between two identical alert types
                     (prevents spam on rapid re-triggers).
    environment    : Tag shown in footers ("production", "staging", etc.).
    verbose        : If True, always prints alerts to stdout as well.
    """

    def __init__(
        self,
        webhook_url:     Optional[str] = None,
        min_interval_s:  float = 300.0,  # 5-minute cooldown per alert type
        environment:     str   = "production",
        verbose:         bool  = True,
    ):
        self._url         = webhook_url or WEBHOOK_URL
        self._min_ivl     = min_interval_s
        self._env         = environment
        self._verbose     = verbose
        self._last_sent:  Dict[str, float] = {}   # alert_type → timestamp

        if not self._url:
            print("[Discord] No webhook URL set — alerts will only print to console. "
                  "Set DISCORD_WEBHOOK_URL env var to enable Discord delivery.")

    # ── public API ──────────────────────────────────────────────────────────

    def send(
        self,
        alert_type: str,
        fields:     Optional[Dict[str, str]] = None,
        force:      bool = False,
    ) -> bool:
        """
        Send a Discord alert embed.

        Parameters
        ----------
        alert_type : One of the 6 alert types (see ALERT_CONFIG).
        fields     : Dict of field_name → value pairs shown in the embed.
        force      : If True, bypass rate-limit cooldown.

        Returns True if the message was sent (or printed), False if throttled.
        """
        if alert_type not in ALERT_CONFIG:
            print(f"[Discord] Unknown alert type: {alert_type}")
            return False

        # Rate limiting
        if not force:
            last = self._last_sent.get(alert_type, 0.0)
            if time.time() - last < self._min_ivl:
                return False

        cfg       = ALERT_CONFIG[alert_type]
        timestamp = datetime.now(timezone.utc).isoformat()
        embed     = self._build_embed(cfg, fields or {}, timestamp)

        printed = self._print_alert(cfg, fields or {}, timestamp)
        sent    = self._post_webhook(embed)
        self._last_sent[alert_type] = time.time()
        return True

    def send_circuit_breaker(self, drawdown: float, equity: float,
                              action: str, pair: str = "EURUSD"):
        self.send("circuit_breaker", {
            "Pair":     pair,
            "Drawdown": f"{drawdown:.2%}",
            "Equity":   f"${equity:,.2f}",
            "Action":   action.upper(),
        })

    def send_drift(self, psi_max: float, reasons: list):
        self.send("drift_detected", {
            "PSI Max":  f"{psi_max:.4f}",
            "Reasons":  ", ".join(str(r) for r in reasons[:3]),
        })

    def send_retrain(self, triggers: list, model: str = "unknown"):
        self.send("emergency_retrain", {
            "Model":    model,
            "Triggers": "\n".join(triggers[:3]),
        })

    def send_promotion(self, model: str, sharpe: float, git_hash: str = ""):
        self.send("model_promoted", {
            "Model":  model,
            "Sharpe": f"{sharpe:.3f}",
            "Git":    git_hash or "—",
        })

    def send_demotion(self, triggers: list, rolled_back: bool):
        self.send("model_demoted", {
            "Triggers":  "\n".join(triggers[:3]),
            "Rollback":  "✓ Previous model restored" if rolled_back else "✗ No backup found",
            "Retrain":   "DAG triggered",
        })

    def send_tca_breach(self, cost_pct: float, limit_pct: float,
                         gross_pnl: float):
        self.send("tca_breach", {
            "Cost %":     f"{cost_pct:.1%}",
            "Policy Limit": f"{limit_pct:.1%}",
            "Gross P&L":  f"${gross_pnl:,.0f}",
        })

    # ── internal ────────────────────────────────────────────────────────────

    def _build_embed(self, cfg: dict, fields: Dict[str, str],
                     timestamp: str) -> dict:
        embed_fields = [
            {"name": k, "value": v, "inline": True}
            for k, v in fields.items()
        ]
        embed_fields.append({"name": "Environment", "value": self._env, "inline": True})
        return {
            "embeds": [{
                "title":       f"{cfg['emoji']}  {cfg['title']}",
                "description": cfg["description"],
                "color":       cfg["color"],
                "fields":      embed_fields,
                "footer":      {"text": f"Forex Scaling Model  •  {timestamp}"},
                "timestamp":   timestamp,
            }]
        }

    def _print_alert(self, cfg: dict, fields: dict, timestamp: str) -> bool:
        if not self._verbose:
            return True
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  {cfg['emoji']}  {cfg['title']}")
        print(f"  {cfg['description']}")
        if fields:
            for k, v in fields.items():
                print(f"    {k}: {v}")
        print(f"  {timestamp[:19]} UTC")
        print(f"{sep}")
        return True

    def _post_webhook(self, payload: dict) -> bool:
        if not self._url:
            return False
        try:
            body = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                self._url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
            return True
        except urllib.error.HTTPError as e:
            print(f"[Discord] HTTP {e.code}: {e.read().decode()[:200]}")
        except Exception as e:
            print(f"[Discord] Send failed: {e}")
        return False


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DiscordAlerter — smoke test (no webhook required)")
    alerter = DiscordAlerter(verbose=True, min_interval_s=0)

    alerter.send_circuit_breaker(0.105, 8_950.0, "close_all")
    alerter.send_drift(0.312, ["PSI > 0.2 on vol_20", "KS p < 0.05 on rsi_14"])
    alerter.send_promotion("haelt_v5", 1.82, "a3f9c1d")
    alerter.send_demotion(
        ["Sharpe 0.42 < floor 0.50", "WinRate 42.1% < floor 45%"],
        rolled_back=True,
    )
    alerter.send_tca_breach(0.36, 0.30, 12_500.0)
    print("\nOK ✓")
