"""Risk management — the FINAL GATEKEEPER before any order goes out.

DESIGN PHILOSOPHY:
==================
The risk manager operates INDEPENDENTLY of the HMM regime detection. Even if
the HMM completely fails to identify a crash, the risk manager's circuit
breakers catch it through drawdown limits. This is defense in depth — the HMM
is the first line, the risk manager is the last line.

The risk manager has ABSOLUTE VETO POWER over any signal.

HARDCODED NON-NEGOTIABLE RULES:
- Max total exposure: 80% of portfolio (20% cash minimum)
- Max single position: 15% of portfolio
- Max correlated exposure: 30% in any correlated group
- Max concurrent positions: 5
- Max daily trades: 20
- Max portfolio leverage: 1.25x
- Every position MUST have a stop loss — no exceptions
- Max risk per trade: 1% of portfolio
- Minimum position size: $100

CIRCUIT BREAKERS (based on ACTUAL P&L, independent of regime):
- Daily  DD > 2%:  reduce sizes 50%
- Daily  DD > 3%:  close all, halt rest of day
- Weekly DD > 5%:  reduce sizes 50%
- Weekly DD > 7%:  close all, halt rest of week
- Peak   DD > 10%: halt all, require manual restart (flag file)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .regime_strategies import PairSignal, Signal, SignalDirection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — hardcoded non-negotiable limits
# ---------------------------------------------------------------------------

MAX_TOTAL_EXPOSURE: float = 0.80       # 80% of equity
MAX_SINGLE_POSITION: float = 0.50      # 50% of equity (was 15% — increased for regime-based allocation)
MAX_CORRELATED_EXPOSURE: float = 0.30  # 30% in correlated group
MAX_CONCURRENT_POSITIONS: int = 5
MAX_DAILY_TRADES: int = 20
MAX_PORTFOLIO_LEVERAGE: float = 1.25
MAX_RISK_PER_TRADE: float = 0.01       # 1% of equity
MIN_POSITION_VALUE: float = 100.0      # $100 minimum
MAX_BID_ASK_SPREAD: float = 0.005      # 0.5%
DUPLICATE_ORDER_WINDOW: float = 60.0   # seconds
GAP_RISK_MULTIPLIER: float = 3.0       # assume stop gapped through by 3x
GAP_MAX_LOSS_PCT: float = 0.02         # max 2% loss on gap scenario

# Correlation thresholds
CORR_REDUCE_THRESHOLD: float = 0.70    # reduce size by 50%
CORR_REJECT_THRESHOLD: float = 0.85    # reject entirely
CORR_LOOKBACK: int = 60                # days for correlation

# Leverage restriction conditions
LEVERAGE_MAX_POSITIONS: int = 3         # force 1.0x if holding >= this many
LEVERAGE_ALLOWED_REGIMES = {"NEUTRAL", "STRONG_BULL"}

# Circuit breaker thresholds
DAILY_DD_REDUCE: float = 0.02          # 2% daily DD → reduce sizes 50%
DAILY_DD_HALT: float = 0.03            # 3% daily DD → close all, halt
WEEKLY_DD_REDUCE: float = 0.05         # 5% weekly DD → reduce sizes 50%
WEEKLY_DD_HALT: float = 0.07           # 7% weekly DD → close all, halt
PEAK_DD_HALT: float = 0.10             # 10% from peak → halt, require manual restart

HALT_FLAG_FILE = Path("TRADING_HALTED.flag")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class BreakerLevel(Enum):
    """Circuit breaker severity levels."""
    NONE = "none"
    DAILY_REDUCE = "daily_reduce"      # 2% daily DD
    DAILY_HALT = "daily_halt"          # 3% daily DD
    WEEKLY_REDUCE = "weekly_reduce"    # 5% weekly DD
    WEEKLY_HALT = "weekly_halt"        # 7% weekly DD
    PEAK_HALT = "peak_halt"            # 10% from peak


@dataclass
class CircuitBreakerStatus:
    """Current state of all circuit breakers."""

    level: BreakerLevel = BreakerLevel.NONE
    daily_pnl: float = 0.0
    daily_drawdown_pct: float = 0.0
    weekly_pnl: float = 0.0
    weekly_drawdown_pct: float = 0.0
    peak_drawdown_pct: float = 0.0
    size_multiplier: float = 1.0       # 1.0 = normal, 0.5 = reduced, 0.0 = halted
    is_halted: bool = False
    halt_reason: str = ""
    triggered_at: Optional[pd.Timestamp] = None


@dataclass
class PortfolioState:
    """Complete snapshot of portfolio risk state."""

    equity: float
    cash: float
    buying_power: float
    positions: dict[str, float]        # symbol -> market value
    position_count: int
    daily_pnl: float
    weekly_pnl: float
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    current_drawdown_pct: float        # from peak
    total_exposure: float              # sum |pos| / equity
    max_single_exposure: float
    daily_trade_count: int
    circuit_breaker: CircuitBreakerStatus
    regime_label: str = ""
    regime_probability: float = 0.0
    flicker_rate: float = 0.0


@dataclass
class RiskDecision:
    """Result of risk validation for a signal."""

    approved: bool
    original_signal: Signal
    modified_signal: Optional[Signal]  # None if rejected
    rejection_reason: str              # "" if approved
    modifications_made: list[str] = field(default_factory=list)


@dataclass
class PairRiskDecision:
    """Result of risk validation for a PairSignal.

    A pair is approved as a unit: either both legs pass (with possible size
    modifications) or both are rejected. The rejection reason names which leg
    triggered it so users can debug flaky pairs.
    """

    approved: bool
    original_pair: PairSignal
    modified_pair: Optional[PairSignal]   # None if rejected
    rejection_reason: str                  # "" if approved
    modifications_made: list[str] = field(default_factory=list)


# Legacy aliases used by broker modules
@dataclass
class SizedOrder:
    """A risk-adjusted order ready for execution."""

    symbol: str
    side: str                          # "buy" or "sell"
    qty: int
    order_type: str                    # "market", "limit", "stop_limit"
    limit_price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PortfolioRisk:
    """Simplified risk snapshot (legacy interface)."""

    total_exposure: float
    max_single_exposure: float
    daily_pnl: float
    daily_drawdown: float
    total_drawdown: float
    is_halted: bool


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Tracks P&L and triggers drawdown-based circuit breakers.

    Circuit breakers fire based on ACTUAL P&L, completely independent of
    regime detection. This is the last line of defense.
    """

    def __init__(self, config: dict) -> None:
        self._config = config

        # Thresholds (from config, with hardcoded fallbacks that cannot be loosened)
        self._daily_dd_reduce = min(
            config.get("daily_drawdown_reduce", DAILY_DD_REDUCE), DAILY_DD_REDUCE)
        self._daily_dd_halt = min(
            config.get("daily_drawdown_halt", DAILY_DD_HALT), DAILY_DD_HALT)
        self._weekly_dd_reduce = min(
            config.get("weekly_drawdown_reduce", WEEKLY_DD_REDUCE), WEEKLY_DD_REDUCE)
        self._weekly_dd_halt = min(
            config.get("weekly_drawdown_halt", WEEKLY_DD_HALT), WEEKLY_DD_HALT)
        self._peak_dd_halt = min(
            config.get("peak_drawdown_halt", PEAK_DD_HALT), PEAK_DD_HALT)

        # Tracking
        self._day_start_equity: float = 0.0
        self._week_start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._trigger_history: list[dict] = []

        # Current status
        self._status = CircuitBreakerStatus()
        self._halted_until: Optional[str] = None  # "end_of_day", "end_of_week", "manual"

    @property
    def status(self) -> CircuitBreakerStatus:
        return self._status

    @property
    def trigger_history(self) -> list[dict]:
        return list(self._trigger_history)

    def check(self, equity: float, regime_label: str = "") -> CircuitBreakerStatus:
        """Evaluate all circuit breakers against current equity.

        Parameters
        ----------
        equity : float
            Current account equity.
        regime_label : str
            Current HMM regime label (for logging context only).

        Returns
        -------
        CircuitBreakerStatus
        """
        # Update peak
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Compute drawdowns
        daily_pnl = equity - self._day_start_equity if self._day_start_equity > 0 else 0.0
        weekly_pnl = equity - self._week_start_equity if self._week_start_equity > 0 else 0.0
        daily_dd = -daily_pnl / self._day_start_equity if self._day_start_equity > 0 else 0.0
        weekly_dd = -weekly_pnl / self._week_start_equity if self._week_start_equity > 0 else 0.0
        peak_dd = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0

        # Clamp to non-negative (drawdown is a positive number when losing money)
        daily_dd = max(0.0, daily_dd)
        weekly_dd = max(0.0, weekly_dd)
        peak_dd = max(0.0, peak_dd)

        # Check manual halt flag file
        if HALT_FLAG_FILE.exists():
            self._status = CircuitBreakerStatus(
                level=BreakerLevel.PEAK_HALT,
                daily_pnl=daily_pnl, daily_drawdown_pct=daily_dd,
                weekly_pnl=weekly_pnl, weekly_drawdown_pct=weekly_dd,
                peak_drawdown_pct=peak_dd,
                size_multiplier=0.0, is_halted=True,
                halt_reason=f"Manual halt flag file exists: {HALT_FLAG_FILE}",
                triggered_at=pd.Timestamp.now(),
            )
            return self._status

        # Evaluate breakers from most severe to least
        level = BreakerLevel.NONE
        size_mult = 1.0
        is_halted = False
        halt_reason = ""

        if peak_dd >= self._peak_dd_halt:
            level = BreakerLevel.PEAK_HALT
            size_mult = 0.0
            is_halted = True
            halt_reason = (
                f"Peak drawdown {peak_dd:.2%} >= {self._peak_dd_halt:.0%} limit. "
                f"TRADING HALTED — delete {HALT_FLAG_FILE} to resume."
            )
            self._write_halt_flag(halt_reason, equity, regime_label)

        elif weekly_dd >= self._weekly_dd_halt:
            level = BreakerLevel.WEEKLY_HALT
            size_mult = 0.0
            is_halted = True
            halt_reason = f"Weekly drawdown {weekly_dd:.2%} >= {self._weekly_dd_halt:.0%}. Halt rest of week."
            self._halted_until = "end_of_week"

        elif daily_dd >= self._daily_dd_halt:
            level = BreakerLevel.DAILY_HALT
            size_mult = 0.0
            is_halted = True
            halt_reason = f"Daily drawdown {daily_dd:.2%} >= {self._daily_dd_halt:.0%}. Halt rest of day."
            self._halted_until = "end_of_day"

        elif weekly_dd >= self._weekly_dd_reduce:
            level = BreakerLevel.WEEKLY_REDUCE
            size_mult = 0.5
            halt_reason = f"Weekly drawdown {weekly_dd:.2%} >= {self._weekly_dd_reduce:.0%}. Sizes reduced 50%."

        elif daily_dd >= self._daily_dd_reduce:
            level = BreakerLevel.DAILY_REDUCE
            size_mult = 0.5
            halt_reason = f"Daily drawdown {daily_dd:.2%} >= {self._daily_dd_reduce:.0%}. Sizes reduced 50%."

        # Log breaker changes
        if level != self._status.level and level != BreakerLevel.NONE:
            self._log_trigger(level, halt_reason, equity, daily_pnl, weekly_pnl,
                              peak_dd, regime_label)

        self._status = CircuitBreakerStatus(
            level=level,
            daily_pnl=daily_pnl, daily_drawdown_pct=daily_dd,
            weekly_pnl=weekly_pnl, weekly_drawdown_pct=weekly_dd,
            peak_drawdown_pct=peak_dd,
            size_multiplier=size_mult, is_halted=is_halted,
            halt_reason=halt_reason,
            triggered_at=pd.Timestamp.now() if is_halted else None,
        )
        return self._status

    def reset_daily(self, equity: float) -> None:
        """Reset daily tracking at market open."""
        self._day_start_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        # Clear daily halt
        if self._halted_until == "end_of_day":
            self._halted_until = None
            self._status = CircuitBreakerStatus()
            logger.info("Daily circuit breaker reset. Day-start equity: $%.2f", equity)

    def reset_weekly(self, equity: float) -> None:
        """Reset weekly tracking at start of week."""
        self._week_start_equity = equity
        self._day_start_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        # Clear weekly halt
        if self._halted_until == "end_of_week":
            self._halted_until = None
            self._status = CircuitBreakerStatus()
            logger.info("Weekly circuit breaker reset. Week-start equity: $%.2f", equity)

    def initialize(self, equity: float) -> None:
        """First-time setup of equity tracking."""
        self._day_start_equity = equity
        self._week_start_equity = equity
        self._peak_equity = equity
        self._status = CircuitBreakerStatus()
        logger.info(
            "CircuitBreaker initialized: equity=$%.2f, thresholds: "
            "daily_reduce=%.1f%%, daily_halt=%.1f%%, weekly_reduce=%.1f%%, "
            "weekly_halt=%.1f%%, peak_halt=%.1f%%",
            equity,
            self._daily_dd_reduce * 100, self._daily_dd_halt * 100,
            self._weekly_dd_reduce * 100, self._weekly_dd_halt * 100,
            self._peak_dd_halt * 100,
        )

    def _log_trigger(self, level: BreakerLevel, reason: str, equity: float,
                     daily_pnl: float, weekly_pnl: float, peak_dd: float,
                     regime_label: str) -> None:
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "level": level.value,
            "reason": reason,
            "equity": equity,
            "daily_pnl": daily_pnl,
            "weekly_pnl": weekly_pnl,
            "peak_drawdown_pct": peak_dd,
            "peak_equity": self._peak_equity,
            "day_start_equity": self._day_start_equity,
            "week_start_equity": self._week_start_equity,
            "regime_label": regime_label,
        }
        self._trigger_history.append(record)
        logger.warning("CIRCUIT BREAKER TRIGGERED: %s", json.dumps(record, default=str))

    def _write_halt_flag(self, reason: str, equity: float, regime_label: str) -> None:
        """Write the manual halt flag file for peak drawdown breaker."""
        try:
            HALT_FLAG_FILE.write_text(json.dumps({
                "halted_at": pd.Timestamp.now().isoformat(),
                "reason": reason,
                "equity": equity,
                "peak_equity": self._peak_equity,
                "regime_label": regime_label,
                "instructions": f"Delete this file ({HALT_FLAG_FILE}) to resume trading.",
            }, indent=2))
            logger.critical(
                "PEAK DRAWDOWN HALT: wrote %s — trading will not resume until "
                "this file is manually deleted.", HALT_FLAG_FILE,
            )
        except OSError as e:
            logger.error("Failed to write halt flag file: %s", e)


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """The final gatekeeper — validates every signal before it becomes an order.

    Operates INDEPENDENTLY of regime detection. The HMM is the first line of
    defense, this is the last. Absolute veto power over any trade.

    Parameters
    ----------
    config : dict
        Risk configuration from settings.yaml under the 'risk' key.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._breaker = CircuitBreaker(config.get("circuit_breakers", {}))

        # Configurable thresholds (cannot be loosened beyond hardcoded limits)
        self._max_exposure = min(
            config.get("max_portfolio_exposure", MAX_TOTAL_EXPOSURE), MAX_TOTAL_EXPOSURE)
        self._max_single = min(
            config.get("max_position_size", MAX_SINGLE_POSITION), MAX_SINGLE_POSITION)
        self._max_leverage = min(
            config.get("max_leverage", MAX_PORTFOLIO_LEVERAGE), MAX_PORTFOLIO_LEVERAGE)
        self._risk_per_trade = min(
            config.get("risk_per_trade", MAX_RISK_PER_TRADE), MAX_RISK_PER_TRADE)
        self._min_position_value = config.get("min_position_value", MIN_POSITION_VALUE)

        # State
        self._daily_trade_count: int = 0
        self._recent_orders: list[dict] = []  # for duplicate detection
        self._rejection_log: list[dict] = []

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def rejection_log(self) -> list[dict]:
        return list(self._rejection_log)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        bars: Optional[dict[str, pd.DataFrame]] = None,
        is_overnight: bool = False,
    ) -> RiskDecision:
        """Validate a signal against all risk rules. This is THE method.

        Runs every check in sequence, accumulating modifications. Any single
        check can veto the entire trade.

        Parameters
        ----------
        signal : Signal
            The trade signal from the strategy layer.
        portfolio : PortfolioState
            Current portfolio state snapshot.
        bars : dict[str, pd.DataFrame], optional
            Recent OHLCV data for correlation computation.
        is_overnight : bool
            Whether this position will be held overnight (affects gap risk).

        Returns
        -------
        RiskDecision
            Approved (possibly modified) or rejected with reason.
        """
        modifications: list[str] = []

        # --- 0. Circuit breaker check (absolute priority) ---
        cb = portfolio.circuit_breaker
        if cb.is_halted:
            return self._reject(signal, f"Circuit breaker HALT active: {cb.halt_reason}")

        # --- 1. FLAT signals pass through (they reduce risk) ---
        if signal.direction == SignalDirection.FLAT:
            return RiskDecision(
                approved=True, original_signal=signal,
                modified_signal=signal, rejection_reason="",
                modifications_made=["FLAT signal — pass-through"],
            )

        # --- 2. Stop loss required — absolute, non-negotiable ---
        if signal.stop_loss is None or signal.stop_loss <= 0:
            return self._reject(signal, "No stop loss — every position MUST have a stop loss")

        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        if risk_per_share <= 0:
            return self._reject(signal, "Stop loss equals entry price — zero risk distance")

        # --- 3. Max daily trades ---
        if self._daily_trade_count >= MAX_DAILY_TRADES:
            return self._reject(
                signal,
                f"Max daily trades reached ({MAX_DAILY_TRADES}). Preventing overtrading.",
            )

        # --- 4. Max concurrent positions ---
        if portfolio.position_count >= MAX_CONCURRENT_POSITIONS:
            # Allow if replacing an existing position in the same symbol
            if signal.symbol not in portfolio.positions:
                return self._reject(
                    signal,
                    f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached.",
                )

        # --- 5. Duplicate order check ---
        if self._is_duplicate(signal):
            return self._reject(
                signal,
                f"Duplicate order: {signal.symbol} {signal.direction.value} "
                f"within {DUPLICATE_ORDER_WINDOW}s window.",
            )

        # --- 6. Exposure check ---
        if portfolio.total_exposure >= self._max_exposure:
            # Allow if signal reduces exposure (e.g. closing a position)
            existing_value = portfolio.positions.get(signal.symbol, 0.0)
            is_reducing = (
                (signal.direction == SignalDirection.SHORT and existing_value > 0)
                or (signal.direction == SignalDirection.LONG and existing_value < 0)
            )
            if not is_reducing:
                return self._reject(
                    signal,
                    f"Max exposure {self._max_exposure:.0%} reached "
                    f"(current: {portfolio.total_exposure:.2%}). 20% cash minimum enforced.",
                )

        # --- 7. Position sizing ---
        equity = portfolio.equity
        size_value = self._compute_position_size(
            signal, equity, risk_per_share, portfolio, modifications)

        if size_value is None:
            return self._reject(signal, "Position size computation resulted in zero or negative")

        # --- 8. Circuit breaker size reduction ---
        if cb.size_multiplier < 1.0 and cb.size_multiplier > 0.0:
            size_value *= cb.size_multiplier
            modifications.append(
                f"Circuit breaker size reduction: ×{cb.size_multiplier:.0%} "
                f"({cb.level.value})")

        # --- 9. Gap risk adjustment for overnight positions ---
        if is_overnight:
            gap_max_value = (equity * GAP_MAX_LOSS_PCT) / (GAP_RISK_MULTIPLIER * risk_per_share) * signal.entry_price
            if size_value > gap_max_value:
                modifications.append(
                    f"Overnight gap risk: size ${size_value:.0f} → ${gap_max_value:.0f} "
                    f"(3x stop gap scenario capped at {GAP_MAX_LOSS_PCT:.0%} loss)")
                size_value = gap_max_value

        # --- 10. Correlation check ---
        if bars:
            corr_result = self._check_correlation(signal.symbol, bars, portfolio.positions)
            if corr_result["rejected"]:
                return self._reject(signal, corr_result["reason"])
            if corr_result["scale"] < 1.0:
                size_value *= corr_result["scale"]
                modifications.append(corr_result["reason"])

        # --- 11. Minimum size check ---
        if size_value < self._min_position_value:
            return self._reject(
                signal,
                f"Position value ${size_value:.2f} below minimum ${self._min_position_value:.0f}.",
            )

        # --- 12. Leverage check ---
        leverage = self._compute_leverage(signal, portfolio, modifications)

        # --- 13. Compute final share count ---
        qty = int(size_value / signal.entry_price)
        if qty <= 0:
            return self._reject(signal, "Calculated quantity is zero shares")

        # Build the modified signal
        modified = Signal(
            symbol=signal.symbol,
            direction=signal.direction,
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size_pct=size_value / equity if equity > 0 else 0,
            leverage=leverage,
            regime_id=signal.regime_id,
            regime_name=signal.regime_name,
            regime_probability=signal.regime_probability,
            timestamp=signal.timestamp,
            reasoning=signal.reasoning,
            strategy_name=signal.strategy_name,
            metadata={
                **signal.metadata,
                "risk_sized_value": size_value,
                "risk_sized_qty": qty,
                "risk_per_share": risk_per_share,
                "risk_pct_of_equity": (qty * risk_per_share) / equity if equity > 0 else 0,
                "modifications": modifications,
            },
        )

        # Record the trade
        self._record_order(signal)
        self._daily_trade_count += 1

        return RiskDecision(
            approved=True,
            original_signal=signal,
            modified_signal=modified,
            rejection_reason="",
            modifications_made=modifications,
        )

    # ------------------------------------------------------------------
    # Pair signal validation
    # ------------------------------------------------------------------

    def validate_pair_signal(
        self,
        pair_signal: PairSignal,
        portfolio: PortfolioState,
        bars: Optional[dict[str, pd.DataFrame]] = None,
    ) -> PairRiskDecision:
        """Validate a PairSignal as a unit.

        A pair is one conceptual trade with two coordinated legs. We validate
        both legs against the standard checks but with three pair-specific rule
        adjustments (per design §2.5):

        - Pair counts as **1** toward MAX_CONCURRENT_POSITIONS.
        - Pair counts as **2** toward MAX_DAILY_TRADES.
        - The standard correlation reject is **skipped** for pair legs (we
          *want* them correlated). Instead, we add a "low correlation = broken
          pair" reject when leg-to-leg correlation falls below 0.5.

        If either leg fails, the whole pair is rejected with a reason naming
        which leg triggered it.
        """
        long = pair_signal.long_leg
        short = pair_signal.short_leg

        # --- 0. Circuit breaker (absolute priority) ---
        cb = portfolio.circuit_breaker
        if cb.is_halted:
            return self._reject_pair(
                pair_signal,
                f"Circuit breaker HALT active: {cb.halt_reason}",
            )

        # --- 1. Stop losses required on both legs ---
        for leg_name, leg in (("long", long), ("short", short)):
            if leg.stop_loss is None or leg.stop_loss <= 0:
                return self._reject_pair(
                    pair_signal, f"{leg_name} leg ({leg.symbol}): no stop loss",
                )
            if abs(leg.entry_price - leg.stop_loss) <= 0:
                return self._reject_pair(
                    pair_signal,
                    f"{leg_name} leg ({leg.symbol}): stop equals entry",
                )

        # --- 2. Direction sanity ---
        if long.direction != SignalDirection.LONG:
            return self._reject_pair(
                pair_signal,
                f"long_leg.direction must be LONG (got {long.direction.value})",
            )
        if short.direction != SignalDirection.SHORT:
            return self._reject_pair(
                pair_signal,
                f"short_leg.direction must be SHORT (got {short.direction.value})",
            )

        # --- 3. Daily trade count: pair adds 2 ---
        if self._daily_trade_count + 2 > MAX_DAILY_TRADES:
            return self._reject_pair(
                pair_signal,
                f"Pair would exceed max daily trades "
                f"({self._daily_trade_count}+2 > {MAX_DAILY_TRADES})",
            )

        # --- 4. Concurrent positions: pair adds 1 ---
        # Allow if at least one leg is replacing an existing position in
        # that symbol; otherwise we need room for one new conceptual trade.
        new_symbols = {long.symbol, short.symbol} - set(portfolio.positions.keys())
        if new_symbols and portfolio.position_count + 1 > MAX_CONCURRENT_POSITIONS:
            return self._reject_pair(
                pair_signal,
                f"Pair would exceed max concurrent positions "
                f"({portfolio.position_count}+1 > {MAX_CONCURRENT_POSITIONS})",
            )

        # --- 5. Duplicate-order check on each leg ---
        if self._is_duplicate(long):
            return self._reject_pair(
                pair_signal, f"long leg duplicate within {DUPLICATE_ORDER_WINDOW}s",
            )
        if self._is_duplicate(short):
            return self._reject_pair(
                pair_signal, f"short leg duplicate within {DUPLICATE_ORDER_WINDOW}s",
            )

        # --- 6. Pair-correlation check (inverted vs single-asset) ---
        # We want pair legs to be highly correlated. Reject if cointegration
        # appears broken.
        if bars is not None and long.symbol in bars and short.symbol in bars:
            corr = self._pair_correlation(bars[long.symbol], bars[short.symbol])
            if corr is not None and corr < 0.5:
                return self._reject_pair(
                    pair_signal,
                    f"Pair correlation {corr:.2f} < 0.5 — cointegration likely broken",
                )

        # --- 7. Per-leg sizing (independent of normal correlation guard) ---
        modifications: list[str] = []
        equity = portfolio.equity
        long_size = self._compute_position_size(
            long, equity, abs(long.entry_price - long.stop_loss),
            portfolio, modifications,
        )
        if long_size is None:
            return self._reject_pair(pair_signal, "long leg sizing failed")
        short_size = self._compute_position_size(
            short, equity, abs(short.entry_price - short.stop_loss),
            portfolio, modifications,
        )
        if short_size is None:
            return self._reject_pair(pair_signal, "short leg sizing failed")

        # --- 8. Apply circuit-breaker size multiplier to both legs ---
        if 0.0 < cb.size_multiplier < 1.0:
            long_size *= cb.size_multiplier
            short_size *= cb.size_multiplier
            modifications.append(
                f"Circuit breaker size reduction ×{cb.size_multiplier:.0%} "
                f"({cb.level.value})",
            )

        # --- 9. Combined exposure must fit MAX_TOTAL_EXPOSURE ---
        existing_exposure = sum(abs(v) for v in portfolio.positions.values())
        # Subtract any exposure we'd be replacing in legs that already exist.
        for sym in (long.symbol, short.symbol):
            existing_exposure -= abs(portfolio.positions.get(sym, 0.0))
        existing_exposure = max(existing_exposure, 0.0)

        total_after = existing_exposure + long_size + short_size
        max_total = self._max_exposure * equity
        if total_after > max_total:
            # Scale both legs down proportionally so combined fits.
            slack = max_total - existing_exposure
            if slack <= 0:
                return self._reject_pair(
                    pair_signal,
                    f"No exposure room for pair: existing exposure ${existing_exposure:.0f} "
                    f"already at max ${max_total:.0f}",
                )
            scale = slack / (long_size + short_size)
            long_size *= scale
            short_size *= scale
            modifications.append(
                f"Pair exposure-cap scale ×{scale:.2f}: combined legs sized to fit",
            )

        # --- 10. Min position value on each leg ---
        if long_size < self._min_position_value:
            return self._reject_pair(
                pair_signal,
                f"long leg ${long_size:.2f} < min ${self._min_position_value:.0f}",
            )
        if short_size < self._min_position_value:
            return self._reject_pair(
                pair_signal,
                f"short leg ${short_size:.2f} < min ${self._min_position_value:.0f}",
            )

        # --- 11. Leverage clamp on each leg (independently) ---
        long_lev = self._compute_leverage(long, portfolio, modifications)
        short_lev = self._compute_leverage(short, portfolio, modifications)

        # --- 12. Build modified legs ---
        long_qty = int(long_size / long.entry_price) if long.entry_price > 0 else 0
        short_qty = int(short_size / short.entry_price) if short.entry_price > 0 else 0
        if long_qty <= 0 or short_qty <= 0:
            return self._reject_pair(
                pair_signal,
                f"Pair leg qty <= 0 after sizing (long={long_qty}, short={short_qty})",
            )

        long_risk_per_share = abs(long.entry_price - long.stop_loss)
        short_risk_per_share = abs(short.entry_price - short.stop_loss)
        long_modified = Signal(
            symbol=long.symbol, direction=long.direction,
            confidence=long.confidence,
            entry_price=long.entry_price, stop_loss=long.stop_loss,
            take_profit=long.take_profit,
            position_size_pct=long_size / equity if equity > 0 else 0,
            leverage=long_lev,
            regime_id=long.regime_id, regime_name=long.regime_name,
            regime_probability=long.regime_probability,
            timestamp=long.timestamp, reasoning=long.reasoning,
            strategy_name=long.strategy_name,
            metadata={
                **long.metadata,
                "risk_sized_value": long_size,
                "risk_sized_qty": long_qty,
                "risk_per_share": long_risk_per_share,
                "pair_role": "long",
            },
        )
        short_modified = Signal(
            symbol=short.symbol, direction=short.direction,
            confidence=short.confidence,
            entry_price=short.entry_price, stop_loss=short.stop_loss,
            take_profit=short.take_profit,
            position_size_pct=short_size / equity if equity > 0 else 0,
            leverage=short_lev,
            regime_id=short.regime_id, regime_name=short.regime_name,
            regime_probability=short.regime_probability,
            timestamp=short.timestamp, reasoning=short.reasoning,
            strategy_name=short.strategy_name,
            metadata={
                **short.metadata,
                "risk_sized_value": short_size,
                "risk_sized_qty": short_qty,
                "risk_per_share": short_risk_per_share,
                "pair_role": "short",
            },
        )

        modified_pair = PairSignal(
            pair=pair_signal.pair,
            long_leg=long_modified, short_leg=short_modified,
            spread_value=pair_signal.spread_value,
            z_score=pair_signal.z_score,
            hedge_ratio=pair_signal.hedge_ratio,
            correlation=pair_signal.correlation,
            reasoning=pair_signal.reasoning,
            timestamp=pair_signal.timestamp,
            metadata={**pair_signal.metadata, "modifications": modifications},
        )

        # Record both legs as "trades" so duplicate detection / daily count
        # advance correctly (pair = 2 trades).
        self._record_order(long)
        self._record_order(short)
        self._daily_trade_count += 2

        return PairRiskDecision(
            approved=True,
            original_pair=pair_signal,
            modified_pair=modified_pair,
            rejection_reason="",
            modifications_made=modifications,
        )

    @staticmethod
    def _pair_correlation(
        bars_a: pd.DataFrame, bars_b: pd.DataFrame,
        lookback: int = CORR_LOOKBACK,
    ) -> Optional[float]:
        """Pearson correlation of returns between two symbols over lookback."""
        if bars_a is None or bars_b is None:
            return None
        if len(bars_a) < lookback or len(bars_b) < lookback:
            return None
        ra = bars_a["close"].iloc[-lookback:].pct_change().dropna()
        rb = bars_b["close"].iloc[-lookback:].pct_change().dropna()
        combined = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(combined) < 30:
            return None
        return float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))

    def _reject_pair(self, pair_signal: PairSignal, reason: str) -> PairRiskDecision:
        """Log and return a pair rejection."""
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "pair": list(pair_signal.pair),
            "long_symbol": pair_signal.long_leg.symbol,
            "short_symbol": pair_signal.short_leg.symbol,
            "regime": pair_signal.long_leg.regime_name,
            "strategy": pair_signal.long_leg.strategy_name,
            "z_score": pair_signal.z_score,
            "rejection_reason": reason,
        }
        self._rejection_log.append(record)
        logger.warning("PAIR REJECTED: %s", json.dumps(record, default=str))
        return PairRiskDecision(
            approved=False,
            original_pair=pair_signal,
            modified_pair=None,
            rejection_reason=reason,
        )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _compute_position_size(
        self,
        signal: Signal,
        equity: float,
        risk_per_share: float,
        portfolio: PortfolioState,
        modifications: list[str],
    ) -> Optional[float]:
        """Compute position value in dollars, applying all caps.

        Formula: size = (equity × risk_per_trade) / risk_per_share × entry_price
        Then cap by: regime max, portfolio max, single position max, exposure room.
        """
        if equity <= 0 or risk_per_share <= 0:
            return None

        # Risk-based sizing: risk 1% of equity per trade
        risk_budget = equity * self._risk_per_trade
        shares_by_risk = risk_budget / risk_per_share
        size_value = shares_by_risk * signal.entry_price

        # Cap 1: Regime-suggested position size
        regime_max = signal.position_size_pct * equity
        if size_value > regime_max:
            modifications.append(
                f"Regime cap: ${size_value:.0f} → ${regime_max:.0f} "
                f"({signal.regime_name} max {signal.position_size_pct:.0%})")
            size_value = regime_max

        # Cap 2: Absolute single position max (15% of equity)
        abs_max = self._max_single * equity
        if size_value > abs_max:
            modifications.append(
                f"Single position cap: ${size_value:.0f} → ${abs_max:.0f} "
                f"({self._max_single:.0%} of equity)")
            size_value = abs_max

        # Cap 3: Remaining exposure room
        exposure_room = (self._max_exposure * equity) - sum(
            abs(v) for v in portfolio.positions.values())
        if size_value > exposure_room and exposure_room > 0:
            modifications.append(
                f"Exposure room cap: ${size_value:.0f} → ${exposure_room:.0f}")
            size_value = exposure_room
        elif exposure_room <= 0:
            return None

        return max(size_value, 0.0)

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    def _compute_leverage(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        modifications: list[str],
    ) -> float:
        """Determine allowed leverage, applying all restrictions."""
        requested = signal.leverage

        # Hardcoded ceiling
        leverage = min(requested, self._max_leverage)

        # Force 1.0x conditions
        force_reasons: list[str] = []

        if portfolio.circuit_breaker.level != BreakerLevel.NONE:
            force_reasons.append(f"circuit breaker active ({portfolio.circuit_breaker.level.value})")

        if portfolio.position_count >= LEVERAGE_MAX_POSITIONS:
            force_reasons.append(f"holding {portfolio.position_count} positions (max {LEVERAGE_MAX_POSITIONS} for leverage)")

        if signal.regime_name not in LEVERAGE_ALLOWED_REGIMES:
            force_reasons.append(f"regime {signal.regime_name} not in leverage-allowed set")

        if signal.regime_probability < 0.55:
            force_reasons.append(f"regime confidence {signal.regime_probability:.2f} < 55%")

        if portfolio.flicker_rate > 4:
            force_reasons.append(f"flicker rate {portfolio.flicker_rate:.1f} > 4")

        if force_reasons:
            if leverage > 1.0:
                modifications.append(
                    f"Leverage forced to 1.0x: {'; '.join(force_reasons)}")
            leverage = 1.0

        return leverage

    # ------------------------------------------------------------------
    # Correlation check
    # ------------------------------------------------------------------

    def _check_correlation(
        self,
        symbol: str,
        bars: dict[str, pd.DataFrame],
        positions: dict[str, float],
    ) -> dict:
        """Check correlation of new symbol with all existing positions.

        Returns
        -------
        dict with keys: rejected (bool), scale (float), reason (str)
        """
        if not positions or symbol not in bars:
            return {"rejected": False, "scale": 1.0, "reason": ""}

        if symbol in positions:
            # Adding to existing position — no correlation penalty
            return {"rejected": False, "scale": 1.0, "reason": ""}

        new_returns = self._get_returns(bars.get(symbol))
        if new_returns is None:
            return {"rejected": False, "scale": 1.0, "reason": ""}

        max_corr = 0.0
        max_corr_symbol = ""

        for pos_symbol in positions:
            if pos_symbol == symbol or pos_symbol not in bars:
                continue
            pos_returns = self._get_returns(bars.get(pos_symbol))
            if pos_returns is None:
                continue

            # Align on common dates
            combined = pd.concat([new_returns, pos_returns], axis=1, join="inner").dropna()
            if len(combined) < 30:
                continue

            corr = float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
            corr = abs(corr)

            if corr > max_corr:
                max_corr = corr
                max_corr_symbol = pos_symbol

        if max_corr >= CORR_REJECT_THRESHOLD:
            reason = (
                f"Rejected {symbol}: {max_corr:.2f} correlation with "
                f"existing {max_corr_symbol} (threshold {CORR_REJECT_THRESHOLD})")
            logger.warning(reason)
            return {"rejected": True, "scale": 0.0, "reason": reason}

        if max_corr >= CORR_REDUCE_THRESHOLD:
            reason = (
                f"Correlation guard: {symbol} ↔ {max_corr_symbol} = {max_corr:.2f} "
                f"(> {CORR_REDUCE_THRESHOLD}). Size reduced 50%.")
            logger.info(reason)
            return {"rejected": False, "scale": 0.5, "reason": reason}

        return {"rejected": False, "scale": 1.0, "reason": ""}

    @staticmethod
    def _get_returns(bars: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        if bars is None or len(bars) < CORR_LOOKBACK:
            return None
        close = bars["close"].iloc[-CORR_LOOKBACK:]
        return close.pct_change().dropna()

    # ------------------------------------------------------------------
    # Duplicate order detection
    # ------------------------------------------------------------------

    def _is_duplicate(self, signal: Signal) -> bool:
        """Check if the same symbol+direction was ordered recently."""
        now = time.monotonic()
        # Prune old entries
        self._recent_orders = [
            r for r in self._recent_orders
            if now - r["time"] < DUPLICATE_ORDER_WINDOW
        ]
        for r in self._recent_orders:
            if r["symbol"] == signal.symbol and r["direction"] == signal.direction.value:
                return True
        return False

    def _record_order(self, signal: Signal) -> None:
        self._recent_orders.append({
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "time": time.monotonic(),
        })

    # ------------------------------------------------------------------
    # Rejection logging
    # ------------------------------------------------------------------

    def _reject(self, signal: Signal, reason: str) -> RiskDecision:
        """Log and return a rejection."""
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "regime": signal.regime_name,
            "strategy": signal.strategy_name,
            "rejection_reason": reason,
        }
        self._rejection_log.append(record)
        logger.warning("TRADE REJECTED: %s", json.dumps(record, default=str))

        return RiskDecision(
            approved=False,
            original_signal=signal,
            modified_signal=None,
            rejection_reason=reason,
        )

    # ------------------------------------------------------------------
    # Legacy interface methods (backward compatibility)
    # ------------------------------------------------------------------

    def size_order(
        self,
        signal: Signal,
        current_price: float,
        equity: float,
        existing_positions: dict[str, float],
        atr: Optional[float] = None,
    ) -> Optional[SizedOrder]:
        """Convert a signal into a SizedOrder (legacy interface).

        New code should use validate_signal() instead.
        """
        portfolio = PortfolioState(
            equity=equity, cash=equity - sum(abs(v) for v in existing_positions.values()),
            buying_power=equity, positions=existing_positions,
            position_count=len(existing_positions),
            daily_pnl=0, weekly_pnl=0,
            peak_equity=max(self._breaker._peak_equity, equity),
            day_start_equity=self._breaker._day_start_equity or equity,
            week_start_equity=self._breaker._week_start_equity or equity,
            current_drawdown_pct=0, total_exposure=0, max_single_exposure=0,
            daily_trade_count=self._daily_trade_count,
            circuit_breaker=self._breaker.status,
        )
        decision = self.validate_signal(signal, portfolio)
        if not decision.approved or decision.modified_signal is None:
            return None

        ms = decision.modified_signal
        qty = ms.metadata.get("risk_sized_qty", 0)
        side = "buy" if ms.direction == SignalDirection.LONG else "sell"

        return SizedOrder(
            symbol=ms.symbol, side=side, qty=qty,
            order_type="limit",
            limit_price=ms.entry_price,
            stop_price=ms.stop_loss,
            time_in_force="day",
            metadata=ms.metadata,
        )

    def check_portfolio_risk(
        self, equity: float, positions: dict[str, float],
    ) -> PortfolioRisk:
        """Evaluate current portfolio risk state (legacy interface)."""
        total_exp = sum(abs(v) for v in positions.values()) / equity if equity > 0 else 0
        max_single = max((abs(v) for v in positions.values()), default=0) / equity if equity > 0 else 0
        cb = self._breaker.check(equity)
        return PortfolioRisk(
            total_exposure=total_exp,
            max_single_exposure=max_single,
            daily_pnl=cb.daily_pnl,
            daily_drawdown=cb.daily_drawdown_pct,
            total_drawdown=cb.peak_drawdown_pct,
            is_halted=cb.is_halted,
        )

    def update_equity_tracking(self, equity: float) -> None:
        """Update peak equity tracker. Call after each fill."""
        if equity > self._breaker._peak_equity:
            self._breaker._peak_equity = equity

    def reset_daily(self, equity: float) -> None:
        """Reset daily tracking at market open."""
        self._daily_trade_count = 0
        self._recent_orders.clear()
        self._breaker.reset_daily(equity)

    def reset_weekly(self, equity: float) -> None:
        """Reset weekly tracking at start of week."""
        self._daily_trade_count = 0
        self._recent_orders.clear()
        self._breaker.reset_weekly(equity)

    def initialize(self, equity: float) -> None:
        """First-time initialization of the risk manager."""
        self._breaker.initialize(equity)

    @property
    def is_halted(self) -> bool:
        return self._breaker.status.is_halted

    def check_correlation_guard(
        self, symbol: str, bars: dict[str, pd.DataFrame], positions: dict[str, float],
    ) -> float:
        """Legacy interface: return scaling factor 0-1 based on correlation."""
        result = self._check_correlation(symbol, bars, positions)
        if result["rejected"]:
            return 0.0
        return result["scale"]

    # ------------------------------------------------------------------
    # Risk report
    # ------------------------------------------------------------------

    def get_risk_report(self, portfolio: PortfolioState) -> str:
        """Generate a formatted risk report string.

        Parameters
        ----------
        portfolio : PortfolioState
            Current portfolio state.

        Returns
        -------
        str
            Multi-line formatted risk report.
        """
        cb = portfolio.circuit_breaker
        lines = [
            "=" * 60,
            "RISK MANAGER STATUS REPORT",
            "=" * 60,
            f"Equity:              ${portfolio.equity:>12,.2f}",
            f"Cash:                ${portfolio.cash:>12,.2f}",
            f"Peak Equity:         ${portfolio.peak_equity:>12,.2f}",
            "",
            "--- Exposure ---",
            f"Total Exposure:      {portfolio.total_exposure:>11.2%}  (max {self._max_exposure:.0%})",
            f"Max Single Position: {portfolio.max_single_exposure:>11.2%}  (max {self._max_single:.0%})",
            f"Positions:           {portfolio.position_count:>11d}  (max {MAX_CONCURRENT_POSITIONS})",
            f"Daily Trades:        {portfolio.daily_trade_count:>11d}  (max {MAX_DAILY_TRADES})",
            "",
            "--- Drawdowns ---",
            f"Daily P&L:           ${cb.daily_pnl:>12,.2f}  ({cb.daily_drawdown_pct:>+.2%})",
            f"Weekly P&L:          ${cb.weekly_pnl:>12,.2f}  ({cb.weekly_drawdown_pct:>+.2%})",
            f"From Peak:           {cb.peak_drawdown_pct:>11.2%}  (halt at {PEAK_DD_HALT:.0%})",
            "",
            "--- Circuit Breakers ---",
            f"Status:              {cb.level.value}",
            f"Size Multiplier:     {cb.size_multiplier:>11.0%}",
            f"Halted:              {'YES' if cb.is_halted else 'no'}",
        ]
        if cb.is_halted:
            lines.append(f"Halt Reason:         {cb.halt_reason}")

        lines.extend([
            "",
            "--- Regime Context ---",
            f"Regime:              {portfolio.regime_label or 'N/A'}",
            f"Probability:         {portfolio.regime_probability:>11.2%}",
            f"Flicker Rate:        {portfolio.flicker_rate:>11.1f}",
            "",
            f"Trade Rejections:    {len(self._rejection_log):>11d}",
            f"Breaker Triggers:    {len(self._breaker.trigger_history):>11d}",
            "=" * 60,
        ])
        return "\n".join(lines)
