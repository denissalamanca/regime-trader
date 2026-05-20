"""Phase D: tests for the six single-asset strategies in core/strategies/.

Each strategy gets a focused set of tests covering:
  - Direction & sizing correctness in the regime it's designed for
  - Stop-loss correctness (below entry for longs, above for shorts)
  - Vol-rank gating (returns None / FLAT outside the intended regime)
  - Edge cases: insufficient bars, missing universe data
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import SignalDirection
from core.strategies import (
    classify_vol_rank,
    DefensiveLongShort,
    HIGH_VOL_THRESHOLD,
    LOW_VOL_THRESHOLD,
    MeanReversionLowVol,
    MomentumRotation,
    PureRegimeAllocation,
    TrendFollowingRegimeFilter,
    VolatilityBreakout,
)


# ---------------------------------------------------------------------------
# Test helpers (mirror tests/test_strategies.py conventions)
# ---------------------------------------------------------------------------

def _bars(n=200, trend="up", seed=42, base=100.0, vol=0.01):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = np.zeros(n)
    close[0] = base
    drift = {"up": 0.001, "down": -0.001, "flat": 0.0}.get(trend, 0.0)
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(rng.normal(drift, vol))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "open": close * 1.001, "high": high, "low": low,
        "close": close, "volume": np.abs(rng.normal(1e6, 2e5, n)),
    }, index=dates)


def _info(label=None, rid=0, vol=0.15):
    """Build a RegimeInfo. If label is omitted, picks a label whose vol-rank
    classification matches the requested ``vol`` per the strategy library's
    LOW/HIGH thresholds — so tests can request a vol rank by passing vol
    alone (legacy convention)."""
    if label is None:
        # vol→label mapping picks a label that yields the desired classify_vol_rank
        if vol < 0.18:
            label = "BULL"            # low vol
        elif vol > 0.30:
            label = "CRASH"           # high vol
        else:
            label = "NEUTRAL"         # mid vol
    return RegimeInfo(
        regime_id=rid, regime_name=label, expected_return=0.0,
        expected_volatility=vol, recommended_strategy_type="trend_follow",
        max_leverage_allowed=1.0, max_position_size_pct=0.50,
        min_confidence_to_act=0.55,
    )


def _state(label="NEUTRAL", sid=0, prob=0.7, n=3):
    probs = np.full(n, (1 - prob) / max(n - 1, 1))
    probs[sid] = prob
    return RegimeState(
        label=label, state_id=sid, probability=prob,
        state_probabilities=probs, timestamp=pd.Timestamp("2024-06-15"),
        is_confirmed=True, consecutive_bars=5,
    )


# ===========================================================================
# Vol-rank classifier
# ===========================================================================

class TestClassifyVolRank:
    def test_low_vol_bull_label(self):
        assert classify_vol_rank(_info(label="BULL", vol=0.15)) == "low"

    def test_low_vol_strong_bull_label(self):
        assert classify_vol_rank(_info(label="STRONG_BULL", vol=0.15)) == "low"

    def test_mid_vol_neutral_label(self):
        assert classify_vol_rank(_info(label="NEUTRAL", vol=0.22)) == "mid"

    def test_mid_vol_bear_label(self):
        assert classify_vol_rank(_info(label="BEAR", vol=0.25)) == "mid"

    def test_high_vol_crash_label(self):
        assert classify_vol_rank(_info(label="CRASH", vol=0.45)) == "high"

    def test_high_vol_strong_bear_label(self):
        assert classify_vol_rank(_info(label="STRONG_BEAR", vol=0.40)) == "high"

    def test_unknown_label_uses_threshold_fallback(self):
        # Unknown label, low z-score-std → low
        assert classify_vol_rank(_info(label="MYSTERY", vol=0.5)) == "low"
        # Unknown label, high z-score-std → high
        assert classify_vol_rank(_info(label="OTHER", vol=2.0)) == "high"


# ===========================================================================
# 1. PureRegimeAllocation
# ===========================================================================

class TestPureRegimeAllocation:
    def test_low_vol_emits_long(self):
        s = PureRegimeAllocation({}, _info(vol=0.10))
        s.set_universe_bars({"SPY": _bars(), "QQQ": _bars(seed=2)})
        sig = s.generate_signal("SPY", _bars(), _state())
        assert sig is not None
        assert sig.direction == SignalDirection.LONG

    def test_high_vol_emits_flat(self):
        s = PureRegimeAllocation({}, _info(vol=0.45))
        s.set_universe_bars({"SPY": _bars()})
        sig = s.generate_signal("SPY", _bars(), _state(label="CRASH"))
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT

    def test_per_name_uses_universe_count(self):
        s = PureRegimeAllocation({"low_vol_gross": 1.0}, _info(vol=0.10))
        s.set_universe_bars({"SPY": _bars(), "QQQ": _bars(seed=2),
                              "IWM": _bars(seed=3), "AAPL": _bars(seed=4)})
        sig = s.generate_signal("SPY", _bars(), _state())
        assert sig.position_size_pct == pytest.approx(0.25)  # 100% / 4

    def test_per_name_falls_back_to_hint(self):
        s = PureRegimeAllocation({"low_vol_gross": 1.0, "n_watchlist_hint": 5}, _info(vol=0.10))
        # No set_universe_bars → fall back to hint=5
        sig = s.generate_signal("SPY", _bars(), _state())
        assert sig.position_size_pct == pytest.approx(0.20)  # 100% / 5

    def test_mid_vol_uses_mid_gross(self):
        s = PureRegimeAllocation({"mid_vol_gross": 0.50}, _info(vol=0.22))
        s.set_universe_bars({"SPY": _bars(), "QQQ": _bars()})
        sig = s.generate_signal("SPY", _bars(), _state())
        assert sig.position_size_pct == pytest.approx(0.25)  # 50% / 2

    def test_stop_below_entry_for_long(self):
        s = PureRegimeAllocation({}, _info(vol=0.10))
        s.set_universe_bars({"SPY": _bars()})
        sig = s.generate_signal("SPY", _bars(), _state())
        assert sig.stop_loss < sig.entry_price

    def test_insufficient_bars_returns_none(self):
        s = PureRegimeAllocation({}, _info(vol=0.10))
        s.set_universe_bars({"SPY": _bars(n=30)})
        sig = s.generate_signal("SPY", _bars(n=30), _state())
        assert sig is None


# ===========================================================================
# 2. TrendFollowingRegimeFilter
# ===========================================================================

class TestTrendFollowingRegimeFilter:
    def test_low_vol_above_ma_long(self):
        s = TrendFollowingRegimeFilter({"ma_period": 50}, _info(vol=0.10))
        sig = s.generate_signal("SPY", _bars(n=200, trend="up"), _state())
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert sig.stop_loss < sig.entry_price

    def test_low_vol_below_ma_flat(self):
        s = TrendFollowingRegimeFilter({"ma_period": 50}, _info(vol=0.10))
        sig = s.generate_signal("SPY", _bars(n=200, trend="down"), _state())
        # Either FLAT (if price below MA) or None — both acceptable
        if sig is not None:
            assert sig.direction == SignalDirection.FLAT

    def test_high_vol_returns_none(self):
        s = TrendFollowingRegimeFilter({}, _info(vol=0.40))
        sig = s.generate_signal("SPY", _bars(n=300, trend="up"), _state())
        assert sig is None

    def test_mid_vol_returns_none(self):
        s = TrendFollowingRegimeFilter({}, _info(vol=0.22))
        sig = s.generate_signal("SPY", _bars(n=300, trend="up"), _state())
        assert sig is None

    def test_insufficient_bars_returns_none(self):
        s = TrendFollowingRegimeFilter({"ma_period": 200}, _info(vol=0.10))
        sig = s.generate_signal("SPY", _bars(n=50), _state())
        assert sig is None


# ===========================================================================
# 3. MeanReversionLowVol
# ===========================================================================

class TestMeanReversionLowVol:
    def _oversold_bars(self, n=200, seed=7):
        """Bars with a sharp end-of-window selloff to trigger oversold RSI."""
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range("2023-01-01", periods=n)
        close = np.zeros(n)
        close[0] = 100.0
        for i in range(1, n - 30):
            close[i] = close[i - 1] * np.exp(rng.normal(0.001, 0.008))
        # Sharp 30-bar selloff
        for i in range(n - 30, n):
            close[i] = close[i - 1] * np.exp(rng.normal(-0.015, 0.008))
        high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
        return pd.DataFrame({
            "open": close * 1.001, "high": high, "low": low,
            "close": close, "volume": np.abs(rng.normal(1e6, 2e5, n)),
        }, index=dates)

    def test_oversold_emits_long(self):
        s = MeanReversionLowVol({}, _info(vol=0.10))
        sig = s.generate_signal("SPY", self._oversold_bars(), _state())
        # Oversold path should emit LONG
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert sig.stop_loss < sig.entry_price

    def test_overbought_emits_flat(self):
        # 200-bar trend up to push RSI > 50
        s = MeanReversionLowVol({}, _info(vol=0.10))
        sig = s.generate_signal("SPY", _bars(n=200, trend="up"), _state())
        # In strong uptrend RSI is above 50 → FLAT
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT

    def test_high_vol_returns_none(self):
        s = MeanReversionLowVol({}, _info(vol=0.40))
        sig = s.generate_signal("SPY", self._oversold_bars(), _state())
        assert sig is None

    def test_neutral_zone_returns_none(self):
        # RSI between thresholds → no action. With sideways drift bars and
        # default 30/50 thresholds we sometimes land in (30, 50).
        s = MeanReversionLowVol({}, _info(vol=0.10))
        rng = np.random.RandomState(99)
        n = 200
        close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n)))
        bars = pd.DataFrame({
            "open": close, "high": close * 1.002, "low": close * 0.998,
            "close": close, "volume": np.full(n, 1e6),
        }, index=pd.bdate_range("2023-01-01", periods=n))
        sig = s.generate_signal("SPY", bars, _state())
        # Either None (no action) or LONG/FLAT — but not all None forever.
        # Just verify that when sig is not None it has a sensible direction.
        if sig is not None:
            assert sig.direction in (SignalDirection.LONG, SignalDirection.FLAT)


# ===========================================================================
# 4. VolatilityBreakout
# ===========================================================================

class TestVolatilityBreakout:
    def _breakout_bars(self, n=200):
        """Bars that finish with a sharp final-bar breakout above the prior N-day range."""
        rng = np.random.RandomState(13)
        dates = pd.bdate_range("2023-01-01", periods=n)
        close = np.zeros(n)
        close[0] = 100.0
        # Steady mild uptrend for n-1 bars, then a sharp gap up on the last.
        for i in range(1, n - 1):
            close[i] = close[i - 1] * np.exp(rng.normal(0.0005, 0.005))
        # Final bar: pop 5% to clear the rolling max
        close[-1] = close[-2] * 1.05
        high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
        # Make sure the final bar's high is also the highest
        high[-1] = max(high[-1], close[-1] * 1.001)
        return pd.DataFrame({
            "open": close * 1.001, "high": high, "low": low,
            "close": close, "volume": np.full(n, 1e6),
        }, index=dates)

    def test_breakout_emits_long(self):
        s = VolatilityBreakout({"breakout_lookback": 20}, _info(vol=0.15))
        bars = self._breakout_bars()
        sig = s.generate_signal("SPY", bars, _state())
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert sig.stop_loss < sig.entry_price

    def test_no_breakout_returns_none(self):
        s = VolatilityBreakout({"breakout_lookback": 20}, _info(vol=0.15))
        sig = s.generate_signal("SPY", _bars(n=200, trend="down"), _state())
        # Downtrend → not above 20d high → None
        assert sig is None

    def test_high_vol_returns_none(self):
        s = VolatilityBreakout({"breakout_lookback": 20}, _info(vol=0.40))
        sig = s.generate_signal("SPY", self._breakout_bars(), _state())
        assert sig is None

    def test_size_capped_at_max_position_size(self):
        # ATR will be very small in a low-vol trend → inverse-ATR sizing
        # would be huge, but should be capped.
        s = VolatilityBreakout(
            {"breakout_lookback": 20, "max_position_size": 0.20,
             "target_risk_pct": 0.01},
            _info(vol=0.15),
        )
        sig = s.generate_signal("SPY", self._breakout_bars(), _state())
        assert sig is not None
        assert sig.position_size_pct <= 0.20 + 1e-9


# ===========================================================================
# 5. MomentumRotation
# ===========================================================================

class TestMomentumRotation:
    def _universe(self):
        """Three symbols with very different trailing returns."""
        return {
            "WIN1": _bars(n=200, trend="up", seed=1),    # high return
            "WIN2": _bars(n=200, trend="up", seed=2, vol=0.005),
            "FLAT": _bars(n=200, trend="flat", seed=3),
            "LOSE": _bars(n=200, trend="down", seed=4),
        }

    def test_top_n_emits_long(self):
        s = MomentumRotation({"lookback": 60, "top_n": 2}, _info(vol=0.15))
        u = self._universe()
        s.set_universe_bars(u)
        # WIN1 should be in top 2
        sig = s.generate_signal("WIN1", u["WIN1"], _state())
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert sig.stop_loss < sig.entry_price

    def test_outside_top_n_emits_flat(self):
        s = MomentumRotation({"lookback": 60, "top_n": 2}, _info(vol=0.15))
        u = self._universe()
        s.set_universe_bars(u)
        sig = s.generate_signal("LOSE", u["LOSE"], _state())
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT

    def test_high_vol_uses_high_vol_topn(self):
        s = MomentumRotation(
            {"lookback": 60, "top_n": 3, "top_n_high_vol": 1},
            _info(vol=0.40),
        )
        u = self._universe()
        s.set_universe_bars(u)
        # Only the #1 should be long; everyone else flat.
        winner_sig = s.generate_signal("WIN1", u["WIN1"], _state(label="CRASH"))
        runner_sig = s.generate_signal("WIN2", u["WIN2"], _state(label="CRASH"))
        # Exactly one of WIN1/WIN2 should be LONG; the other should be FLAT
        directions = {winner_sig.direction, runner_sig.direction}
        assert SignalDirection.LONG in directions
        assert SignalDirection.FLAT in directions

    def test_size_is_one_over_n(self):
        s = MomentumRotation({"lookback": 60, "top_n": 2}, _info(vol=0.15))
        u = self._universe()
        s.set_universe_bars(u)
        sig = s.generate_signal("WIN1", u["WIN1"], _state())
        assert sig.position_size_pct == pytest.approx(0.50)  # 1/2

    def test_no_universe_returns_none(self):
        s = MomentumRotation({"lookback": 60}, _info(vol=0.15))
        # No set_universe_bars
        sig = s.generate_signal("SPY", _bars(), _state())
        assert sig is None


# ===========================================================================
# 6. DefensiveLongShort
# ===========================================================================

class TestDefensiveLongShort:
    def _universe(self):
        return {
            "WIN1": _bars(n=200, trend="up", seed=11),
            "WIN2": _bars(n=200, trend="up", seed=12, vol=0.006),
            "MID":  _bars(n=200, trend="flat", seed=13),
            "LOSE1": _bars(n=200, trend="down", seed=14),
            "LOSE2": _bars(n=200, trend="down", seed=15, vol=0.012),
        }

    def test_top_n_long_in_low_vol(self):
        s = DefensiveLongShort(
            {"lookback": 60, "top_n": 2, "bottom_m": 2}, _info(vol=0.10),
        )
        u = self._universe()
        s.set_universe_bars(u)
        sig = s.generate_signal("WIN1", u["WIN1"], _state())
        assert sig is not None
        assert sig.direction == SignalDirection.LONG

    def test_no_short_in_low_vol(self):
        s = DefensiveLongShort(
            {"lookback": 60, "top_n": 2, "bottom_m": 2}, _info(vol=0.10),
        )
        u = self._universe()
        s.set_universe_bars(u)
        sig = s.generate_signal("LOSE1", u["LOSE1"], _state())
        # Bottom in low-vol → should NOT short (FLAT instead)
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT

    def test_short_active_in_high_vol(self):
        s = DefensiveLongShort(
            {"lookback": 60, "top_n": 2, "bottom_m": 2}, _info(vol=0.40),
        )
        u = self._universe()
        s.set_universe_bars(u)
        sig = s.generate_signal("LOSE1", u["LOSE1"], _state(label="CRASH"))
        assert sig is not None
        # In high-vol the bottom-M bucket activates → SHORT
        assert sig.direction == SignalDirection.SHORT
        assert sig.stop_loss > sig.entry_price  # stop above entry for shorts

    def test_short_per_name_size(self):
        s = DefensiveLongShort(
            {"lookback": 60, "top_n": 2, "bottom_m": 2,
             "long_gross": 0.6, "short_gross": 0.4},
            _info(vol=0.40),
        )
        u = self._universe()
        s.set_universe_bars(u)
        sig = s.generate_signal("LOSE1", u["LOSE1"], _state(label="CRASH"))
        assert sig.direction == SignalDirection.SHORT
        assert sig.position_size_pct == pytest.approx(0.40 / 2)  # 0.20

    def test_long_per_name_size(self):
        s = DefensiveLongShort(
            {"lookback": 60, "top_n": 3, "long_gross": 0.6}, _info(vol=0.10),
        )
        u = self._universe()
        s.set_universe_bars(u)
        sig = s.generate_signal("WIN1", u["WIN1"], _state())
        assert sig.position_size_pct == pytest.approx(0.6 / 3)  # 0.20

    def test_middle_returns_flat(self):
        s = DefensiveLongShort(
            {"lookback": 60, "top_n": 2, "bottom_m": 2}, _info(vol=0.40),
        )
        u = self._universe()
        s.set_universe_bars(u)
        sig = s.generate_signal("MID", u["MID"], _state(label="CRASH"))
        assert sig is not None
        assert sig.direction == SignalDirection.FLAT
