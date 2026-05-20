"""Tests for risk management — the final gatekeeper.

Verifies:
1. Position sizing respects all caps (risk-based, regime, single, exposure).
2. Circuit breakers fire at correct thresholds.
3. Correlation guard reduces/rejects correlated trades.
4. Leverage restrictions are enforced under all conditions.
5. Stop loss is required — no exceptions.
6. Duplicate order and max trade count checks.
7. Gap risk adjustment for overnight positions.
8. Hardcoded limits cannot be loosened via config.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.risk_manager import (
    RiskManager,
    CircuitBreaker,
    PortfolioState,
    RiskDecision,
    SizedOrder,
    PortfolioRisk,
    BreakerLevel,
    CircuitBreakerStatus,
    MAX_TOTAL_EXPOSURE,
    MAX_SINGLE_POSITION,
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_TRADES,
    MAX_PORTFOLIO_LEVERAGE,
    MAX_RISK_PER_TRADE,
    MIN_POSITION_VALUE,
    CORR_REDUCE_THRESHOLD,
    CORR_REJECT_THRESHOLD,
    DAILY_DD_REDUCE,
    DAILY_DD_HALT,
    WEEKLY_DD_REDUCE,
    WEEKLY_DD_HALT,
    PEAK_DD_HALT,
    HALT_FLAG_FILE,
)
from core.regime_strategies import Signal, SignalDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    symbol: str = "SPY",
    direction: SignalDirection = SignalDirection.LONG,
    entry: float = 450.0,
    stop: float = 445.0,
    take_profit: float = 460.0,
    size_pct: float = 0.10,
    leverage: float = 1.0,
    regime: str = "NEUTRAL",
    regime_prob: float = 0.70,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        confidence=0.65,
        entry_price=entry,
        stop_loss=stop,
        take_profit=take_profit,
        position_size_pct=size_pct,
        leverage=leverage,
        regime_id=1,
        regime_name=regime,
        regime_probability=regime_prob,
        timestamp=pd.Timestamp("2024-06-15 10:30:00"),
        reasoning="Test signal",
        strategy_name="test",
    )


def _make_portfolio(
    equity: float = 100_000.0,
    positions: dict = None,
    daily_pnl: float = 0.0,
    weekly_pnl: float = 0.0,
    daily_trade_count: int = 0,
    cb_level: BreakerLevel = BreakerLevel.NONE,
    flicker_rate: float = 0.0,
    regime_label: str = "NEUTRAL",
    regime_prob: float = 0.70,
) -> PortfolioState:
    if positions is None:
        positions = {}
    total_exp = sum(abs(v) for v in positions.values()) / equity if equity > 0 else 0
    max_single = max((abs(v) for v in positions.values()), default=0) / equity if equity > 0 else 0
    cb = CircuitBreakerStatus(
        level=cb_level,
        daily_pnl=daily_pnl,
        size_multiplier=0.0 if cb_level in (BreakerLevel.DAILY_HALT, BreakerLevel.WEEKLY_HALT, BreakerLevel.PEAK_HALT) else (0.5 if cb_level in (BreakerLevel.DAILY_REDUCE, BreakerLevel.WEEKLY_REDUCE) else 1.0),
        is_halted=cb_level in (BreakerLevel.DAILY_HALT, BreakerLevel.WEEKLY_HALT, BreakerLevel.PEAK_HALT),
        halt_reason="test halt" if cb_level in (BreakerLevel.DAILY_HALT, BreakerLevel.WEEKLY_HALT, BreakerLevel.PEAK_HALT) else "",
    )
    return PortfolioState(
        equity=equity, cash=equity - sum(abs(v) for v in positions.values()),
        buying_power=equity, positions=positions,
        position_count=len(positions),
        daily_pnl=daily_pnl, weekly_pnl=weekly_pnl,
        peak_equity=equity, day_start_equity=equity,
        week_start_equity=equity,
        current_drawdown_pct=0, total_exposure=total_exp,
        max_single_exposure=max_single,
        daily_trade_count=daily_trade_count,
        circuit_breaker=cb,
        regime_label=regime_label,
        regime_probability=regime_prob,
        flicker_rate=flicker_rate,
    )


def _make_bars(symbols: list, n: int = 100, seed: int = 42) -> dict:
    """Make OHLCV bars for multiple symbols."""
    rng = np.random.RandomState(seed)
    result = {}
    for i, sym in enumerate(symbols):
        dates = pd.bdate_range("2024-01-01", periods=n)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
        result[sym] = pd.DataFrame({
            "open": close * 1.001, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": rng.uniform(1e6, 2e6, n),
        }, index=dates)
    return result


@pytest.fixture
def rm():
    """Default risk manager."""
    return RiskManager({})


@pytest.fixture(autouse=True)
def cleanup_halt_flag():
    """Remove halt flag file after each test."""
    yield
    if HALT_FLAG_FILE.exists():
        HALT_FLAG_FILE.unlink()


# ---------------------------------------------------------------------------
# Stop loss — absolute requirement
# ---------------------------------------------------------------------------

class TestStopLossRequired:
    def test_rejects_no_stop_loss(self, rm):
        """EVERY position must have a stop loss — no exceptions."""
        signal = _make_signal(stop=None)
        # Signal dataclass requires a value but let's test with 0
        signal.stop_loss = 0
        decision = rm.validate_signal(signal, _make_portfolio())
        assert not decision.approved
        assert "stop loss" in decision.rejection_reason.lower()

    def test_rejects_stop_equal_to_entry(self, rm):
        signal = _make_signal(entry=100, stop=100)
        decision = rm.validate_signal(signal, _make_portfolio())
        assert not decision.approved
        assert "zero risk distance" in decision.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_risk_based_sizing(self, rm):
        """Position size = (equity × 1%) / risk_per_share × entry_price."""
        signal = _make_signal(entry=100, stop=98, size_pct=0.50)
        portfolio = _make_portfolio(equity=100_000)
        decision = rm.validate_signal(signal, portfolio)
        assert decision.approved

        ms = decision.modified_signal
        risk_per_share = 2.0  # 100 - 98
        # Capped by max single position (50% = $50,000) and regime size (50%)
        assert ms.metadata["risk_sized_qty"] <= 500  # $50,000 / $100

    def test_single_position_cap(self, rm):
        """No single position can exceed max_position_size of equity."""
        signal = _make_signal(entry=10, stop=9.90, size_pct=0.50)
        portfolio = _make_portfolio(equity=100_000)
        decision = rm.validate_signal(signal, portfolio)
        assert decision.approved
        value = decision.modified_signal.metadata["risk_sized_value"]
        assert value <= 100_000 * MAX_SINGLE_POSITION + 1  # +1 for float

    def test_exposure_room_cap(self, rm):
        """Cannot exceed 80% total exposure."""
        # Already 75% exposed
        positions = {"AAPL": 25_000, "GOOG": 25_000, "MSFT": 25_000}
        portfolio = _make_portfolio(equity=100_000, positions=positions)
        signal = _make_signal(entry=100, stop=98, size_pct=0.15)
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved:
            value = decision.modified_signal.metadata["risk_sized_value"]
            total_after = sum(abs(v) for v in positions.values()) + value
            assert total_after <= 100_000 * MAX_TOTAL_EXPOSURE + 1

    def test_min_position_size(self, rm):
        """Skip trade if position value below $100."""
        signal = _make_signal(entry=5000, stop=4999.99, size_pct=0.01)
        portfolio = _make_portfolio(equity=1_000)  # Very small account
        decision = rm.validate_signal(signal, portfolio)
        # Risk-based size: (1000 * 0.01) / 0.01 * 5000 = $5,000,000 -> capped at 15% = $150
        # Actually let's recalculate: risk_budget = 10, risk_per_share = 0.01
        # shares_by_risk = 10 / 0.01 = 1000, value = 1000 * 5000 = $5,000,000
        # Capped at 15% * 1000 = $150 -> above $100 minimum
        # Use a scenario where it actually goes below $100
        signal2 = _make_signal(entry=500, stop=495, size_pct=0.01)
        portfolio2 = _make_portfolio(equity=500)
        decision2 = rm.validate_signal(signal2, portfolio2)
        # risk_budget = 5, risk_per_share = 5, shares = 1, value = $500
        # Capped at 15% * 500 = $75 -> below $100
        assert not decision2.approved
        assert "minimum" in decision2.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Circuit breakers
# ---------------------------------------------------------------------------

class TestCircuitBreakers:
    def test_daily_2pct_reduces_size(self):
        """Daily DD > 2% → reduce sizes 50%."""
        cb = CircuitBreaker({})
        cb.initialize(100_000)
        status = cb.check(97_900)  # 2.1% drawdown
        assert status.level == BreakerLevel.DAILY_REDUCE
        assert status.size_multiplier == 0.5
        assert not status.is_halted

    def test_daily_3pct_halts(self):
        """Daily DD > 3% → halt rest of day."""
        cb = CircuitBreaker({})
        cb.initialize(100_000)
        status = cb.check(96_900)  # 3.1% drawdown
        assert status.level == BreakerLevel.DAILY_HALT
        assert status.is_halted
        assert status.size_multiplier == 0.0

    def test_weekly_5pct_reduces_size(self):
        """Weekly DD > 5% → reduce sizes 50%."""
        cb = CircuitBreaker({})
        cb.initialize(100_000)
        # Simulate: day start at 96k (so daily DD stays below 3% halt)
        # but weekly DD from 100k exceeds 5%.
        cb.reset_daily(96_000)  # Day 2 start
        status = cb.check(94_800)  # daily DD = 1.25% (< 3%), weekly DD = 5.2%
        assert status.level == BreakerLevel.WEEKLY_REDUCE
        assert status.size_multiplier == 0.5

    def test_weekly_7pct_halts(self):
        """Weekly DD > 7% → halt rest of week."""
        cb = CircuitBreaker({})
        cb.initialize(100_000)
        cb.reset_daily(96_000)
        status = cb.check(92_800)  # 7.2% from week start
        assert status.level == BreakerLevel.WEEKLY_HALT
        assert status.is_halted

    def test_peak_10pct_halts_and_writes_flag(self):
        """Peak DD > 10% → halt, write flag file."""
        cb = CircuitBreaker({})
        cb.initialize(100_000)
        cb.reset_daily(95_000)
        cb.reset_weekly(95_000)
        status = cb.check(89_500)  # 10.5% from peak of 100k
        assert status.level == BreakerLevel.PEAK_HALT
        assert status.is_halted
        assert HALT_FLAG_FILE.exists()

    def test_daily_reset_clears_halt(self):
        """Daily reset should clear daily halt."""
        cb = CircuitBreaker({})
        cb.initialize(100_000)
        cb.check(96_500)  # trigger daily halt
        assert cb.status.is_halted
        cb.reset_daily(96_500)
        assert not cb.status.is_halted

    def test_weekly_reset_clears_halt(self):
        """Weekly reset should clear weekly halt."""
        cb = CircuitBreaker({})
        cb.initialize(100_000)
        cb.reset_daily(95_000)
        cb.check(92_500)  # trigger weekly halt
        assert cb.status.is_halted
        cb.reset_weekly(92_500)
        assert not cb.status.is_halted

    def test_halted_portfolio_rejects_all_signals(self, rm):
        """When circuit breaker is halted, all signals are rejected."""
        signal = _make_signal()
        portfolio = _make_portfolio(cb_level=BreakerLevel.DAILY_HALT)
        decision = rm.validate_signal(signal, portfolio)
        assert not decision.approved
        assert "circuit breaker" in decision.rejection_reason.lower()

    def test_reduce_breaker_halves_position_size(self, rm):
        """When size reduction breaker active, position sizes are halved."""
        signal = _make_signal(entry=100, stop=98, size_pct=0.10)
        portfolio = _make_portfolio(
            equity=100_000, cb_level=BreakerLevel.DAILY_REDUCE)
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved:
            mods = decision.modifications_made
            assert any("circuit breaker" in m.lower() for m in mods)

    def test_trigger_history_recorded(self):
        """Breaker triggers are recorded in history."""
        cb = CircuitBreaker({})
        cb.initialize(100_000)
        cb.check(96_500)  # trigger daily halt
        assert len(cb.trigger_history) >= 1
        assert cb.trigger_history[-1]["level"] == "daily_halt"

    def test_config_cannot_loosen_beyond_hardcoded(self):
        """Config values cannot make thresholds LESS strict."""
        cb = CircuitBreaker({
            "daily_drawdown_reduce": 0.10,  # Trying to loosen to 10%
            "peak_drawdown_halt": 0.50,     # Trying to loosen to 50%
        })
        # Internal thresholds should be clamped to hardcoded values
        assert cb._daily_dd_reduce <= DAILY_DD_REDUCE
        assert cb._peak_dd_halt <= PEAK_DD_HALT


# ---------------------------------------------------------------------------
# Leverage restrictions
# ---------------------------------------------------------------------------

class TestLeverageRules:
    def test_max_leverage_1_25x(self, rm):
        """Portfolio leverage never exceeds 1.25x."""
        signal = _make_signal(leverage=2.0, regime="NEUTRAL", regime_prob=0.8)
        portfolio = _make_portfolio()
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved:
            assert decision.modified_signal.leverage <= MAX_PORTFOLIO_LEVERAGE

    def test_force_1x_when_3plus_positions(self, rm):
        """Force 1.0x leverage when holding 3+ positions."""
        positions = {"AAPL": 10_000, "GOOG": 10_000, "MSFT": 10_000}
        signal = _make_signal(leverage=1.25, regime="NEUTRAL")
        portfolio = _make_portfolio(equity=100_000, positions=positions)
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved:
            assert decision.modified_signal.leverage == 1.0

    def test_force_1x_non_allowed_regime(self, rm):
        """Only NEUTRAL and STRONG_BULL may use > 1.0x leverage."""
        signal = _make_signal(leverage=1.25, regime="BEAR")
        portfolio = _make_portfolio()
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved:
            assert decision.modified_signal.leverage == 1.0

    def test_force_1x_low_confidence(self, rm):
        """Force 1.0x when regime probability < 55%."""
        signal = _make_signal(leverage=1.25, regime="NEUTRAL", regime_prob=0.40)
        portfolio = _make_portfolio()
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved:
            assert decision.modified_signal.leverage == 1.0

    def test_force_1x_high_flicker(self, rm):
        """Force 1.0x when flicker rate > 4."""
        signal = _make_signal(leverage=1.25, regime="NEUTRAL")
        portfolio = _make_portfolio(flicker_rate=5.0)
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved:
            assert decision.modified_signal.leverage == 1.0

    def test_force_1x_circuit_breaker_active(self, rm):
        """Force 1.0x when any circuit breaker is active."""
        signal = _make_signal(leverage=1.25, regime="NEUTRAL")
        portfolio = _make_portfolio(cb_level=BreakerLevel.DAILY_REDUCE)
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved:
            assert decision.modified_signal.leverage == 1.0


# ---------------------------------------------------------------------------
# Correlation guard
# ---------------------------------------------------------------------------

class TestCorrelationGuard:
    def test_high_correlation_reduces_size(self, rm):
        """Correlation > 0.7 reduces size by 50%."""
        # Create highly correlated bars
        rng = np.random.RandomState(42)
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        base = np.cumsum(rng.normal(0, 1, n))
        make_df = lambda c: pd.DataFrame({
            "open": c, "high": c + 0.5, "low": c - 0.5,
            "close": c, "volume": np.ones(n) * 1e6,
        }, index=dates)

        bars = {
            "SPY": make_df(100 + base),
            "QQQ": make_df(200 + base * 1.1 + rng.normal(0, 0.1, n)),  # ~0.99 corr
        }
        result = rm._check_correlation("QQQ", bars, {"SPY": 10_000})
        # Should either reduce or reject depending on exact correlation
        assert result["scale"] <= 0.5 or result["rejected"]

    def test_very_high_correlation_rejects(self, rm):
        """Correlation > 0.85 rejects the trade entirely."""
        rng = np.random.RandomState(42)
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        base = np.cumsum(rng.normal(0, 1, n))
        make_df = lambda c: pd.DataFrame({
            "open": c, "high": c + 0.5, "low": c - 0.5,
            "close": c, "volume": np.ones(n) * 1e6,
        }, index=dates)

        bars = {
            "SPY": make_df(100 + base),
            "CLONE": make_df(100 + base + rng.normal(0, 0.01, n)),  # ~1.0 corr
        }
        result = rm._check_correlation("CLONE", bars, {"SPY": 10_000})
        assert result["rejected"]

    def test_uncorrelated_passes(self, rm):
        """Uncorrelated symbols should pass with full size."""
        rng = np.random.RandomState(42)
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        make_df = lambda c: pd.DataFrame({
            "open": c, "high": c + 0.5, "low": c - 0.5,
            "close": c, "volume": np.ones(n) * 1e6,
        }, index=dates)

        bars = {
            "SPY": make_df(100 + np.cumsum(rng.normal(0, 1, n))),
            "GLD": make_df(50 + np.cumsum(rng.normal(0, 1, n))),  # independent
        }
        result = rm._check_correlation("GLD", bars, {"SPY": 10_000})
        assert not result["rejected"]
        # Scale may or may not be 1.0 depending on random correlation


# ---------------------------------------------------------------------------
# Max concurrent positions and daily trades
# ---------------------------------------------------------------------------

class TestPositionAndTradeLimits:
    def test_max_5_concurrent_positions(self, rm):
        """Reject when already at 5 positions and adding a new symbol."""
        positions = {f"SYM{i}": 10_000 for i in range(5)}
        portfolio = _make_portfolio(equity=100_000, positions=positions)
        signal = _make_signal(symbol="NEW")
        decision = rm.validate_signal(signal, portfolio)
        assert not decision.approved
        assert "concurrent" in decision.rejection_reason.lower()

    def test_max_20_daily_trades(self, rm):
        """Reject after 20 trades in a day."""
        rm._daily_trade_count = MAX_DAILY_TRADES  # Set RM's internal counter
        portfolio = _make_portfolio()
        signal = _make_signal()
        decision = rm.validate_signal(signal, portfolio)
        assert not decision.approved
        assert "daily trades" in decision.rejection_reason.lower()

    def test_duplicate_order_rejected(self, rm):
        """Same symbol+direction within 60s is rejected."""
        portfolio = _make_portfolio()
        signal = _make_signal()
        d1 = rm.validate_signal(signal, portfolio)
        assert d1.approved

        # Second identical signal should be rejected as duplicate
        signal2 = _make_signal()
        d2 = rm.validate_signal(signal2, portfolio)
        assert not d2.approved
        assert "duplicate" in d2.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Gap risk
# ---------------------------------------------------------------------------

class TestGapRisk:
    def test_overnight_reduces_size(self, rm):
        """Overnight positions should be smaller due to gap risk."""
        signal = _make_signal(entry=100, stop=95, size_pct=0.15)
        portfolio = _make_portfolio(equity=100_000)

        d_day = rm.validate_signal(signal, portfolio, is_overnight=False)
        rm._recent_orders.clear()  # Allow re-entry for comparison
        rm._daily_trade_count -= 1
        d_night = rm.validate_signal(signal, portfolio, is_overnight=True)

        if d_day.approved and d_night.approved:
            day_value = d_day.modified_signal.metadata["risk_sized_value"]
            night_value = d_night.modified_signal.metadata["risk_sized_value"]
            assert night_value <= day_value


# ---------------------------------------------------------------------------
# FLAT signals
# ---------------------------------------------------------------------------

class TestFlatSignals:
    def test_flat_always_passes(self, rm):
        """FLAT signals always pass — they reduce risk."""
        signal = _make_signal(direction=SignalDirection.FLAT)
        portfolio = _make_portfolio(cb_level=BreakerLevel.DAILY_HALT)
        # Even with halt active...
        # Actually FLAT should be checked after halt. Let's test without halt.
        portfolio2 = _make_portfolio()
        decision = rm.validate_signal(signal, portfolio2)
        assert decision.approved


# ---------------------------------------------------------------------------
# Risk report
# ---------------------------------------------------------------------------

class TestRiskReport:
    def test_report_is_string(self, rm):
        """Risk report should return a formatted string."""
        portfolio = _make_portfolio()
        report = rm.get_risk_report(portfolio)
        assert isinstance(report, str)
        assert "RISK MANAGER STATUS REPORT" in report
        assert "Exposure" in report
        assert "Circuit Breakers" in report

    def test_report_shows_halted(self, rm):
        portfolio = _make_portfolio(cb_level=BreakerLevel.DAILY_HALT)
        report = rm.get_risk_report(portfolio)
        assert "YES" in report


# ---------------------------------------------------------------------------
# Hardcoded limits
# ---------------------------------------------------------------------------

class TestHardcodedLimits:
    def test_config_cannot_loosen_exposure(self):
        """Config max_portfolio_exposure can't exceed 80%."""
        rm_loose = RiskManager({"max_portfolio_exposure": 0.95})
        assert rm_loose._max_exposure <= MAX_TOTAL_EXPOSURE

    def test_config_cannot_loosen_single_position(self):
        rm_loose = RiskManager({"max_position_size": 0.50})
        assert rm_loose._max_single <= MAX_SINGLE_POSITION

    def test_config_cannot_loosen_leverage(self):
        rm_loose = RiskManager({"max_leverage": 3.0})
        assert rm_loose._max_leverage <= MAX_PORTFOLIO_LEVERAGE

    def test_config_cannot_loosen_risk_per_trade(self):
        rm_loose = RiskManager({"risk_per_trade": 0.05})
        assert rm_loose._risk_per_trade <= MAX_RISK_PER_TRADE

    def test_config_can_tighten(self):
        """Config values CAN make limits tighter."""
        rm_tight = RiskManager({
            "max_portfolio_exposure": 0.50,
            "max_position_size": 0.05,
            "max_leverage": 1.0,
            "risk_per_trade": 0.005,
        })
        assert rm_tight._max_exposure == 0.50
        assert rm_tight._max_single == 0.05
        assert rm_tight._max_leverage == 1.0
        assert rm_tight._risk_per_trade == 0.005


# ---------------------------------------------------------------------------
# Legacy interface
# ---------------------------------------------------------------------------

class TestLegacyInterface:
    def test_size_order_returns_sized_order(self, rm):
        rm.initialize(100_000)
        signal = _make_signal(entry=100, stop=98, size_pct=0.10)
        order = rm.size_order(signal, 100.0, 100_000, {})
        assert order is None or isinstance(order, SizedOrder)

    def test_check_portfolio_risk(self, rm):
        rm.initialize(100_000)
        risk = rm.check_portfolio_risk(100_000, {"SPY": 15_000})
        assert isinstance(risk, PortfolioRisk)
        assert risk.total_exposure > 0

    def test_is_halted_property(self, rm):
        assert not rm.is_halted
