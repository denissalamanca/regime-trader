"""Tests for Phase 2 live-safety fixes (A3 calendar resets, A4 leverage cap)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import main as main_module
from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import LowVolBullStrategy, SignalDirection
from main import TradingLoop


def _bars(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-01", periods=n)
    rng = np.random.RandomState(0)
    close = pd.Series(400 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1e6},
        index=idx,
    )


def _regime_info() -> RegimeInfo:
    return RegimeInfo(
        regime_id=0, regime_name="BULL", expected_return=0.5,
        expected_volatility=0.1, recommended_strategy_type="trend_follow",
        max_leverage_allowed=1.0, max_position_size_pct=0.2,
    )


def _regime_state() -> RegimeState:
    return RegimeState(
        label="BULL", state_id=0, probability=0.9,
        state_probabilities=np.array([0.05, 0.05, 0.9]),
        timestamp=pd.Timestamp("2023-06-01"), is_confirmed=True,
        consecutive_bars=5,
    )


# --- A4: LowVolBull leverage cap -------------------------------------------

def test_low_vol_bull_requests_unleveraged():
    """A4: LowVolBull no longer requests 1.25x (it was silently clamped)."""
    strat = LowVolBullStrategy({}, _regime_info())
    sig = strat.generate_signal("SPY", _bars(), _regime_state())
    assert sig is not None
    assert sig.direction == SignalDirection.LONG
    assert sig.leverage == 1.0
    assert sig.position_size_pct == 0.95


# --- A3: daily/weekly reset on calendar rollover ---------------------------

def _loop_with_mock_risk():
    main_module.logger = MagicMock()  # methods log via module-global logger
    risk = MagicMock()
    loop = TradingLoop(
        config={"universe": {"symbols": ["SPY"]}, "schedule": {}, "model": {}},
        alpaca_client=MagicMock(), hmm_engine=MagicMock(),
        feature_engineer=MagicMock(), strategy_manager=MagicMock(),
        risk_manager=risk, order_executor=MagicMock(),
        position_tracker=MagicMock(), market_data=MagicMock(),
        hmm_features=pd.DataFrame(), dry_run=True,
    )
    return loop, risk


def test_first_observation_does_not_reset():
    loop, risk = _loop_with_mock_risk()
    loop._maybe_reset_periods(pd.Timestamp("2024-06-03"), 100_000)  # Monday
    risk.reset_daily.assert_not_called()
    risk.reset_weekly.assert_not_called()


def test_new_day_triggers_daily_reset():
    loop, risk = _loop_with_mock_risk()
    mon = pd.Timestamp("2024-06-03")
    loop._maybe_reset_periods(mon, 100_000)
    loop._maybe_reset_periods(mon + pd.Timedelta(days=1), 100_000)  # Tue, same week
    risk.reset_daily.assert_called_once()
    risk.reset_weekly.assert_not_called()


def test_new_week_triggers_weekly_reset():
    loop, risk = _loop_with_mock_risk()
    mon = pd.Timestamp("2024-06-03")
    loop._maybe_reset_periods(mon, 100_000)
    loop._maybe_reset_periods(mon + pd.Timedelta(days=7), 100_000)  # next Monday
    risk.reset_weekly.assert_called_once()
    risk.reset_daily.assert_not_called()


def test_same_day_is_noop():
    loop, risk = _loop_with_mock_risk()
    d = pd.Timestamp("2024-06-03")
    loop._maybe_reset_periods(d, 100_000)
    loop._maybe_reset_periods(d, 100_000)
    loop._maybe_reset_periods(d, 100_000)
    risk.reset_daily.assert_not_called()
    risk.reset_weekly.assert_not_called()
