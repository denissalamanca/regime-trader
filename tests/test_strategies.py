"""Tests for regime-adaptive allocation strategies."""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BaseStrategy, Signal, SignalDirection, PairSignal,
    StrategyOrchestrator, RegimeStrategyManager,
    LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy,
    LABEL_TO_STRATEGY, _get_strategy_for_vol_rank,
)


def _make_bars(n=200, trend="up", seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = np.zeros(n); close[0] = 100.0
    for i in range(1, n):
        if trend == "up": close[i] = close[i-1] * np.exp(rng.normal(0.001, 0.01))
        elif trend == "down": close[i] = close[i-1] * np.exp(rng.normal(-0.001, 0.01))
        else: close[i] = close[i-1] * np.exp(rng.normal(0.0, 0.008))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"open": close * 1.001, "high": high, "low": low,
                          "close": close, "volume": np.abs(rng.normal(1e6, 2e5, n))}, index=dates)


def _info(label, rid=0, vol=0.15):
    from core.hmm_engine import _REGIME_DEFAULTS
    d = _REGIME_DEFAULTS.get(label, _REGIME_DEFAULTS["NEUTRAL"])
    return RegimeInfo(regime_id=rid, regime_name=label, expected_return=0,
                      expected_volatility=vol, recommended_strategy_type=d["strategy"],
                      max_leverage_allowed=d["max_leverage"],
                      max_position_size_pct=d["max_position_pct"],
                      min_confidence_to_act=d["min_confidence"])


def _state(label="NEUTRAL", sid=0, prob=0.7, n=3, confirmed=True):
    probs = np.full(n, (1-prob)/(n-1) if n > 1 else 0); probs[sid] = prob
    return RegimeState(label=label, state_id=sid, probability=prob,
                       state_probabilities=probs, timestamp=pd.Timestamp("2024-06-15"),
                       is_confirmed=confirmed, consecutive_bars=5)


class TestLowVolBullStrategy:
    def test_generates_long_with_leverage(self):
        strat = LowVolBullStrategy({}, _info("BULL", vol=0.10))
        sig = strat.generate_signal("SPY", _make_bars(200, "up"), _state("BULL"))
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert sig.position_size_pct == 0.95
        assert sig.leverage == 1.0  # A4: was 1.25 (silently clamped); now unleveraged

    def test_has_stop_below_entry(self):
        strat = LowVolBullStrategy({}, _info("BULL", vol=0.10))
        sig = strat.generate_signal("SPY", _make_bars(200, "up"), _state("BULL"))
        assert sig.stop_loss < sig.entry_price


class TestMidVolCautiousStrategy:
    def test_long_above_ema(self):
        strat = MidVolCautiousStrategy({}, _info("NEUTRAL", vol=0.20))
        sig = strat.generate_signal("SPY", _make_bars(200, "up"), _state("NEUTRAL"))
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert sig.position_size_pct == 0.95  # Above EMA → 95%

    def test_reduced_below_ema(self):
        strat = MidVolCautiousStrategy({}, _info("NEUTRAL", vol=0.20))
        sig = strat.generate_signal("SPY", _make_bars(200, "down"), _state("NEUTRAL"))
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert sig.position_size_pct == 0.60  # Below EMA → reduced


class TestHighVolDefensiveStrategy:
    def test_generates_reduced_long(self):
        strat = HighVolDefensiveStrategy({}, _info("CRASH", vol=0.40))
        sig = strat.generate_signal("SPY", _make_bars(200, "down"), _state("CRASH"))
        assert sig is not None
        assert sig.direction == SignalDirection.LONG
        assert sig.position_size_pct == 0.60  # Reduced but still invested


class TestVolRankMapping:
    def test_3_regimes(self):
        assert _get_strategy_for_vol_rank(0, 3) == LowVolBullStrategy
        assert _get_strategy_for_vol_rank(1, 3) == MidVolCautiousStrategy
        assert _get_strategy_for_vol_rank(2, 3) == HighVolDefensiveStrategy

    def test_all_labels_mapped(self):
        from core.hmm_engine import REGIME_LABEL_SCHEMES
        for n, labels in REGIME_LABEL_SCHEMES.items():
            for label in labels:
                assert label in LABEL_TO_STRATEGY


class TestStrategyOrchestrator:
    @pytest.fixture
    def three(self):
        return [_info("BEAR", 0, 0.35), _info("NEUTRAL", 1, 0.20), _info("BULL", 2, 0.10)]

    def test_3_regimes(self, three):
        orch = StrategyOrchestrator({}, three)
        assert len(orch._strategies) == 3

    def test_low_vol_regime_goes_long(self, three):
        orch = StrategyOrchestrator({}, three)
        sigs, pairs = orch.generate_signals(
            ["SPY"], {"SPY": _make_bars(200, "up")},
            _state("BULL", sid=2, prob=0.8, n=3))
        assert pairs == []
        assert any(s.direction == SignalDirection.LONG for s in sigs)

    def test_high_vol_also_long_but_reduced(self, three):
        orch = StrategyOrchestrator({}, three)
        sigs, pairs = orch.generate_signals(
            ["SPY"], {"SPY": _make_bars(200, "down")},
            _state("BEAR", sid=0, prob=0.8, n=3))
        assert pairs == []
        for s in sigs:
            assert s.direction == SignalDirection.LONG
            assert s.position_size_pct <= 0.65

    def test_uncertainty_halves_size(self, three):
        orch = StrategyOrchestrator({"min_confidence": 0.55}, three)
        sigs, pairs = orch.generate_signals(
            ["SPY"], {"SPY": _make_bars(200, "up")},
            _state("BULL", sid=2, prob=0.40, n=3))
        assert pairs == []
        for s in sigs:
            assert s.position_size_pct <= 0.50


class TestRegimeStrategyManager:
    def test_legacy_interface_returns_list(self):
        """get_signals (legacy) returns a flat list of Signals — no pair tuple."""
        infos = [_info("BEAR", 0, 0.35), _info("NEUTRAL", 1, 0.20), _info("BULL", 2, 0.10)]
        mgr = RegimeStrategyManager({}, regime_infos=infos)
        sigs = mgr.get_signals(_state("NEUTRAL", 1, 0.7, 3), ["SPY"], {"SPY": _make_bars()})
        assert isinstance(sigs, list)
        for s in sigs:
            assert isinstance(s, Signal)

    def test_get_signals_and_pairs_returns_tuple(self):
        """New interface returns (list[Signal], list[PairSignal])."""
        infos = [_info("BEAR", 0, 0.35), _info("NEUTRAL", 1, 0.20), _info("BULL", 2, 0.10)]
        mgr = RegimeStrategyManager({}, regime_infos=infos)
        sigs, pairs = mgr.get_signals_and_pairs(
            _state("NEUTRAL", 1, 0.7, 3), ["SPY"], {"SPY": _make_bars()})
        assert isinstance(sigs, list)
        assert isinstance(pairs, list)
        # Vol-rank strategies are all single-asset → no pair signals
        assert pairs == []

    def test_raises_without_infos(self):
        mgr = RegimeStrategyManager({})
        with pytest.raises(RuntimeError):
            mgr.get_signals(_state(), ["SPY"], {"SPY": _make_bars()})


# ---------------------------------------------------------------------------
# Phase A: PairSignal + BaseStrategy extensions
# ---------------------------------------------------------------------------

class TestPairSignalDataclass:
    def test_pair_signal_constructs(self):
        """PairSignal carries two Signal legs and pair-level metadata."""
        long_leg = Signal(
            symbol="SPY", direction=SignalDirection.LONG, confidence=0.8,
            entry_price=450.0, stop_loss=440.0, take_profit=None,
            position_size_pct=0.10, leverage=1.0,
            regime_id=0, regime_name="BULL", regime_probability=0.8,
            timestamp=pd.Timestamp("2024-06-15"),
            reasoning="long leg", strategy_name="test",
        )
        short_leg = Signal(
            symbol="IWM", direction=SignalDirection.SHORT, confidence=0.8,
            entry_price=200.0, stop_loss=210.0, take_profit=None,
            position_size_pct=0.10, leverage=1.0,
            regime_id=0, regime_name="BULL", regime_probability=0.8,
            timestamp=pd.Timestamp("2024-06-15"),
            reasoning="short leg", strategy_name="test",
        )
        ps = PairSignal(
            pair=("SPY", "IWM"),
            long_leg=long_leg, short_leg=short_leg,
            spread_value=2.25, z_score=2.1, hedge_ratio=2.25,
            correlation=0.85, reasoning="z>2 enter",
        )
        assert ps.pair == ("SPY", "IWM")
        assert ps.long_leg.direction == SignalDirection.LONG
        assert ps.short_leg.direction == SignalDirection.SHORT
        assert ps.z_score == 2.1
        assert ps.hedge_ratio == 2.25
        assert ps.correlation == 0.85


class TestBaseStrategyDefaults:
    """BaseStrategy gained 3 things in Phase A — verify the defaults."""

    @pytest.fixture
    def concrete_strat(self):
        # Use an existing concrete subclass; it should inherit the new defaults.
        return LowVolBullStrategy({}, _info("BULL", vol=0.10))

    def test_is_pair_strategy_default_false(self, concrete_strat):
        assert concrete_strat.is_pair_strategy is False
        assert LowVolBullStrategy.is_pair_strategy is False

    def test_generate_pair_signal_default_returns_none(self, concrete_strat):
        # Single-asset strategies inherit a no-op default
        result = concrete_strat.generate_pair_signal(
            ("SPY", "IWM"), {"SPY": _make_bars(), "IWM": _make_bars()},
            _state("BULL", sid=0, prob=0.8, n=3))
        assert result is None

    def test_set_universe_bars_stores_dict(self, concrete_strat):
        bars = {"SPY": _make_bars(), "QQQ": _make_bars()}
        concrete_strat.set_universe_bars(bars)
        assert concrete_strat._universe_bars == bars

    def test_universe_bars_initialized_empty(self):
        strat = LowVolBullStrategy({}, _info("BULL", vol=0.10))
        assert strat._universe_bars == {}


class TestPairStrategyDispatch:
    """Orchestrator routes is_pair_strategy=True classes to the pair path."""

    class _FakePairStrategy(BaseStrategy):
        strategy_name = "fake_pair"
        is_pair_strategy = True

        def generate_signal(self, symbol, bars, regime_state):
            # Pair strategies don't generate single-asset signals
            return None

        def generate_pair_signal(self, pair, bars, regime_state):
            a, b = pair
            long_leg = Signal(
                symbol=a, direction=SignalDirection.LONG, confidence=0.7,
                entry_price=100.0, stop_loss=95.0, take_profit=None,
                position_size_pct=0.10, leverage=1.0,
                regime_id=regime_state.state_id, regime_name=regime_state.label,
                regime_probability=regime_state.probability,
                timestamp=regime_state.timestamp, reasoning="fake", strategy_name=self.strategy_name,
            )
            short_leg = Signal(
                symbol=b, direction=SignalDirection.SHORT, confidence=0.7,
                entry_price=200.0, stop_loss=210.0, take_profit=None,
                position_size_pct=0.10, leverage=1.0,
                regime_id=regime_state.state_id, regime_name=regime_state.label,
                regime_probability=regime_state.probability,
                timestamp=regime_state.timestamp, reasoning="fake", strategy_name=self.strategy_name,
            )
            return PairSignal(
                pair=pair, long_leg=long_leg, short_leg=short_leg,
                spread_value=0.5, z_score=2.1, hedge_ratio=0.5,
            )

    def test_orchestrator_dispatches_to_pair_method(self, monkeypatch):
        """When the active strategy is_pair_strategy=True, generate_pair_signal is called."""
        # Patch _get_strategy_for_vol_rank so all 3 ranks point at our fake pair strategy
        from core import regime_strategies as rs_mod
        monkeypatch.setattr(rs_mod, "_get_strategy_for_vol_rank",
                            lambda rank, n: self._FakePairStrategy)

        infos = [_info("BEAR", 0, 0.35), _info("NEUTRAL", 1, 0.20), _info("BULL", 2, 0.10)]
        config = {"pairs": [["SPY", "IWM"], ["QQQ", "SPY"]]}
        orch = StrategyOrchestrator(config, infos)

        bars = {"SPY": _make_bars(), "IWM": _make_bars(), "QQQ": _make_bars()}
        sigs, pairs = orch.generate_signals(
            list(bars.keys()), bars, _state("BULL", sid=2, prob=0.8, n=3))

        # Pair strategy → no single-asset signals, two pair signals
        assert sigs == []
        assert len(pairs) == 2
        assert all(isinstance(p, PairSignal) for p in pairs)
        assert {p.pair for p in pairs} == {("SPY", "IWM"), ("QQQ", "SPY")}

    def test_orchestrator_skips_pair_when_symbol_missing(self, monkeypatch):
        """If one leg of a configured pair lacks bars, that pair is skipped."""
        from core import regime_strategies as rs_mod
        monkeypatch.setattr(rs_mod, "_get_strategy_for_vol_rank",
                            lambda rank, n: self._FakePairStrategy)
        infos = [_info("BEAR", 0, 0.35), _info("NEUTRAL", 1, 0.20), _info("BULL", 2, 0.10)]
        config = {"pairs": [["SPY", "IWM"], ["QQQ", "MISSING"]]}
        orch = StrategyOrchestrator(config, infos)
        bars = {"SPY": _make_bars(), "IWM": _make_bars(), "QQQ": _make_bars()}
        sigs, pairs = orch.generate_signals(
            list(bars.keys()), bars, _state("BULL", sid=2, prob=0.8, n=3))
        # Only SPY/IWM passes; QQQ/MISSING is skipped
        assert len(pairs) == 1
        assert pairs[0].pair == ("SPY", "IWM")


class TestSetUniverseBarsCalled:
    """Orchestrator calls set_universe_bars on the dominant strategy each tick."""

    def test_universe_bars_set_before_generate(self):
        infos = [_info("BEAR", 0, 0.35), _info("NEUTRAL", 1, 0.20), _info("BULL", 2, 0.10)]
        orch = StrategyOrchestrator({}, infos)
        bars = {"SPY": _make_bars(), "QQQ": _make_bars()}
        orch.generate_signals(["SPY", "QQQ"], bars,
                              _state("BULL", sid=2, prob=0.8, n=3))
        # The dominant strategy (BULL → vol rank 0 → LowVolBullStrategy) should
        # have received the universe bars
        dominant = orch._strategies[2]  # BULL has regime_id=2
        assert dominant._universe_bars == bars
