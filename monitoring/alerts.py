"""Alert system for critical trading events.

Delivery methods:
  - Console logging (always)
  - Log file (always, via the structured logger)
  - Email (via smtplib, configurable)
  - Webhook (Slack/Discord, configurable)

Rate limiting: max 1 alert per event type per cooldown period (default 15 min)
to prevent alert fatigue from regime flicker.

Every alert includes full context: what happened, current state, what action
the system took.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert types and levels
# ---------------------------------------------------------------------------

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Predefined alert event types for rate limiting."""

    REGIME_CHANGE = "regime_change"
    CIRCUIT_BREAKER = "circuit_breaker"
    LARGE_TRADE_PNL = "large_trade_pnl"
    DAILY_PNL_THRESHOLD = "daily_pnl_threshold"
    DATA_FEED_DISCONNECT = "data_feed_disconnect"
    API_DISCONNECT = "api_disconnect"
    HMM_RETRAINED = "hmm_retrained"
    FLICKER_THRESHOLD = "flicker_threshold"
    CUSTOM = "custom"


# Default alert type -> level mapping
_DEFAULT_LEVELS = {
    AlertType.REGIME_CHANGE: AlertLevel.WARNING,
    AlertType.CIRCUIT_BREAKER: AlertLevel.CRITICAL,
    AlertType.LARGE_TRADE_PNL: AlertLevel.WARNING,
    AlertType.DAILY_PNL_THRESHOLD: AlertLevel.INFO,
    AlertType.DATA_FEED_DISCONNECT: AlertLevel.ERROR,
    AlertType.API_DISCONNECT: AlertLevel.ERROR,
    AlertType.HMM_RETRAINED: AlertLevel.INFO,
    AlertType.FLICKER_THRESHOLD: AlertLevel.WARNING,
}


@dataclass
class AlertContext:
    """Full context attached to every alert."""

    regime: str = ""
    regime_probability: float = 0.0
    equity: float = 0.0
    daily_pnl_pct: float = 0.0
    positions_count: int = 0
    circuit_breaker: str = "none"
    action_taken: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Alert:
    """A complete alert record."""

    alert_type: AlertType
    level: AlertLevel
    subject: str
    message: str
    context: AlertContext
    timestamp: float = field(default_factory=time.time)
    delivered_to: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------

class AlertManager:
    """Manages alert delivery with rate limiting and multiple channels.

    Parameters
    ----------
    config : dict
        Alert configuration from settings.yaml under 'monitoring.alerts'.
    credentials : dict, optional
        SMTP and webhook credentials. Can also come from env vars.
    """

    def __init__(self, config: dict, credentials: Optional[dict] = None) -> None:
        self._config = config
        self._credentials = credentials or {}
        self._enabled: bool = config.get("enabled", True)
        self._cooldown_seconds: float = config.get("cooldown_seconds", 900)  # 15 min default

        # Rate limiting: alert_type_key -> last_sent_timestamp
        self._last_alert_times: dict[str, float] = {}

        # Alert history
        self._history: list[Alert] = []

        # Email config (from credentials or env vars)
        self._smtp_host = self._credentials.get("smtp", {}).get("host") or os.getenv("SMTP_HOST", "")
        self._smtp_port = int(self._credentials.get("smtp", {}).get("port") or os.getenv("SMTP_PORT", "587"))
        self._smtp_user = self._credentials.get("smtp", {}).get("user") or os.getenv("SMTP_USER", "")
        self._smtp_pass = self._credentials.get("smtp", {}).get("password") or os.getenv("SMTP_PASSWORD", "")
        self._email_to = self._credentials.get("email_to") or os.getenv("ALERT_EMAIL", "")

        # Webhook
        self._webhook_url = self._credentials.get("webhook_url") or os.getenv("ALERT_WEBHOOK_URL", "")

        # Which delivery methods are active
        self._email_enabled = bool(self._smtp_host and self._smtp_user and self._email_to)
        self._webhook_enabled = bool(self._webhook_url)

    @property
    def history(self) -> list[Alert]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Main send method
    # ------------------------------------------------------------------

    def send(
        self,
        alert_type: AlertType,
        subject: str,
        message: str,
        context: Optional[AlertContext] = None,
        level: Optional[AlertLevel] = None,
    ) -> bool:
        """Send an alert if enabled and cooldown has elapsed.

        Parameters
        ----------
        alert_type : AlertType
            Event category for rate limiting.
        subject : str
            Short alert subject line.
        message : str
            Full alert body with context.
        context : AlertContext, optional
            Trading context at time of alert.
        level : AlertLevel, optional
            Override default level for this alert type.

        Returns
        -------
        bool
            True if the alert was delivered, False if rate-limited or disabled.
        """
        if not self._enabled:
            return False

        if not self._check_cooldown(alert_type.value):
            logger.debug("Alert rate-limited: %s — %s", alert_type.value, subject)
            return False

        resolved_level = level or _DEFAULT_LEVELS.get(alert_type, AlertLevel.INFO)
        ctx = context or AlertContext()

        alert = Alert(
            alert_type=alert_type,
            level=resolved_level,
            subject=subject,
            message=message,
            context=ctx,
        )

        # --- Always: log it ---
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }.get(resolved_level, logging.INFO)

        full_message = self._format_log_message(alert)
        logger.log(log_level, "ALERT [%s] %s: %s", resolved_level.value.upper(),
                    subject, full_message)
        alert.delivered_to.append("log")

        # --- Email ---
        if self._email_enabled and resolved_level in (AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL):
            try:
                self._send_email(subject, full_message)
                alert.delivered_to.append("email")
            except Exception as e:
                logger.error("Failed to send email alert: %s", e)

        # --- Webhook ---
        if self._webhook_enabled:
            try:
                self._send_webhook(alert)
                alert.delivered_to.append("webhook")
            except Exception as e:
                logger.error("Failed to send webhook alert: %s", e)

        self._history.append(alert)
        self._last_alert_times[alert_type.value] = time.time()
        return True

    # ------------------------------------------------------------------
    # Convenience methods for common alert types
    # ------------------------------------------------------------------

    def regime_change(self, old_regime: str, new_regime: str,
                      probability: float, context: Optional[AlertContext] = None) -> bool:
        ctx = context or AlertContext()
        ctx.regime = new_regime
        ctx.regime_probability = probability
        ctx.action_taken = f"Strategy switching from {old_regime} to {new_regime}"
        return self.send(
            AlertType.REGIME_CHANGE,
            f"Regime Change: {old_regime} -> {new_regime}",
            f"Regime changed to {new_regime} with probability {probability:.2%}. "
            f"Strategy will adapt to the new market environment.",
            ctx,
        )

    def circuit_breaker(self, breaker_level: str, reason: str,
                        context: Optional[AlertContext] = None) -> bool:
        ctx = context or AlertContext()
        ctx.circuit_breaker = breaker_level
        return self.send(
            AlertType.CIRCUIT_BREAKER,
            f"Circuit Breaker: {breaker_level}",
            reason,
            ctx,
            level=AlertLevel.CRITICAL,
        )

    def large_trade_pnl(self, symbol: str, pnl: float, pnl_pct: float,
                        context: Optional[AlertContext] = None) -> bool:
        ctx = context or AlertContext()
        direction = "profit" if pnl > 0 else "loss"
        return self.send(
            AlertType.LARGE_TRADE_PNL,
            f"Large Trade {direction.title()}: {symbol} {pnl_pct:+.2%}",
            f"{symbol} closed with {direction} of ${abs(pnl):,.2f} ({pnl_pct:+.2%} of portfolio).",
            ctx,
        )

    def daily_pnl_threshold(self, pnl_pct: float,
                            context: Optional[AlertContext] = None) -> bool:
        ctx = context or AlertContext()
        return self.send(
            AlertType.DAILY_PNL_THRESHOLD,
            f"Daily P&L: {pnl_pct:+.2%}",
            f"Daily P&L has reached {pnl_pct:+.2%} of portfolio equity.",
            ctx,
        )

    def data_feed_disconnect(self, seconds_disconnected: float,
                             context: Optional[AlertContext] = None) -> bool:
        ctx = context or AlertContext()
        ctx.action_taken = "Signal generation paused. Existing stops remain active."
        return self.send(
            AlertType.DATA_FEED_DISCONNECT,
            f"Data Feed Disconnected ({seconds_disconnected:.0f}s)",
            f"Market data feed has been disconnected for {seconds_disconnected:.0f} seconds. "
            f"Signal generation is paused but existing positions retain their stops.",
            ctx,
        )

    def api_disconnect(self, context: Optional[AlertContext] = None) -> bool:
        ctx = context or AlertContext()
        ctx.action_taken = "Attempting reconnect with exponential backoff."
        return self.send(
            AlertType.API_DISCONNECT,
            "Alpaca API Connection Lost",
            "Connection to Alpaca API was lost. Attempting automatic reconnection. "
            "Existing orders and positions are managed server-side by Alpaca.",
            ctx,
        )

    def hmm_retrained(self, n_regimes: int, bic: float,
                      context: Optional[AlertContext] = None) -> bool:
        ctx = context or AlertContext()
        return self.send(
            AlertType.HMM_RETRAINED,
            f"HMM Retrained: {n_regimes} regimes",
            f"HMM model retrained via BIC model selection. "
            f"Selected {n_regimes} regimes (BIC={bic:.2f}).",
            ctx,
        )

    def flicker_warning(self, flicker_rate: float, threshold: float,
                        context: Optional[AlertContext] = None) -> bool:
        ctx = context or AlertContext()
        ctx.action_taken = "Forced uncertainty mode — position sizes reduced 50%, leverage 1.0x."
        return self.send(
            AlertType.FLICKER_THRESHOLD,
            f"High Regime Flicker: {flicker_rate:.0f}/{threshold:.0f}",
            f"Regime flicker rate ({flicker_rate:.0f} changes per 20 bars) exceeds "
            f"threshold ({threshold:.0f}). System has entered uncertainty mode.",
            ctx,
        )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_cooldown(self, key: str) -> bool:
        """Return True if enough time has passed since last alert of this type."""
        last = self._last_alert_times.get(key, 0)
        return (time.time() - last) >= self._cooldown_seconds

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_log_message(self, alert: Alert) -> str:
        """Format alert with full context for logging."""
        ctx = alert.context
        parts = [alert.message]
        context_parts = []
        if ctx.regime:
            context_parts.append(f"regime={ctx.regime} ({ctx.regime_probability:.0%})")
        if ctx.equity:
            context_parts.append(f"equity=${ctx.equity:,.2f}")
        if ctx.daily_pnl_pct:
            context_parts.append(f"daily_pnl={ctx.daily_pnl_pct:+.2%}")
        if ctx.positions_count:
            context_parts.append(f"positions={ctx.positions_count}")
        if ctx.circuit_breaker != "none":
            context_parts.append(f"cb={ctx.circuit_breaker}")
        if ctx.action_taken:
            context_parts.append(f"action: {ctx.action_taken}")

        if context_parts:
            parts.append(f"[{', '.join(context_parts)}]")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Email delivery
    # ------------------------------------------------------------------

    def _send_email(self, subject: str, body: str) -> None:
        """Send an email alert via SMTP."""
        msg = MIMEText(body)
        msg["Subject"] = f"[Regime Trader] {subject}"
        msg["From"] = self._smtp_user
        msg["To"] = self._email_to

        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.starttls()
            server.login(self._smtp_user, self._smtp_pass)
            server.sendmail(self._smtp_user, [self._email_to], msg.as_string())

        logger.debug("Email alert sent: %s", subject)

    # ------------------------------------------------------------------
    # Webhook delivery (Slack / Discord)
    # ------------------------------------------------------------------

    def _send_webhook(self, alert: Alert) -> None:
        """Send alert to a webhook URL (Slack/Discord compatible)."""
        level_emoji = {
            AlertLevel.INFO: "info",
            AlertLevel.WARNING: "warning",
            AlertLevel.ERROR: "error",
            AlertLevel.CRITICAL: "rotating_light",
        }
        emoji = level_emoji.get(alert.level, "bell")

        payload = {
            "text": f":{emoji}: *[{alert.level.value.upper()}] {alert.subject}*\n"
                    f"{alert.message}\n"
                    f"_Regime: {alert.context.regime} | "
                    f"Equity: ${alert.context.equity:,.2f} | "
                    f"Daily P&L: {alert.context.daily_pnl_pct:+.2%}_",
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self._webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            urlopen(req, timeout=10)
            logger.debug("Webhook alert sent: %s", alert.subject)
        except URLError as e:
            logger.error("Webhook delivery failed: %s", e)
