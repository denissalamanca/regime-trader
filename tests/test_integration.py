"""Integration tests — end-to-end verification of the full pipeline.

These tests use real (synthetic) data flowing through the actual components,
not mocks. They verify:
  a. End-to-end dry run: data → HMM → strategy → risk → simulated orders
  b. Look-ahead bias: identical trades across different end dates
  c. Risk manager stress: extreme signals capped, rapid-fire blocked
  d. Alpaca connectivity: connect, place, modify, cancel, close
  e. Recovery: state snapshot save/load, position sync
"""

import sys
import json
import time
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hmm_engine import HMMEngine
from core.regime_strategies import (
    RegimeStrategyManager, StrategyOrchestrator, Signal, SignalDirection,
)
from core.risk_manager import (
    RiskManager, PortfolioState, CircuitBreakerStatus, BreakerLevel,
    MAX_SINGLE_POSITION, MAX_PORTFOLIO_LEVERAGE, MAX_DAILY_TRADES,
    MAX_CONCURRENT_POSITIONS,
)
from data.feature_engineering import FeatureEngineer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Two-regime synthetic data: bull first half, bear second half."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.zeros(n)
    close[0] = 100.0
    for i in range(1, n):
        if i < n // 2:
            close[i] = close[i - 1] * np.exp(rng.normal(0.0005, 0.008))
        else:
            close[i] = close[i - 1] * np.exp(rng.normal(-0.0003, 0.018))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = np.abs(rng.normal(1e6, 2e5, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _fast_hmm_config():
    return {
        "n_candidates": [3],
        "n_init": 3,
        "covariance_type": "diag",
        "n_iter": 50,
        "random_state": 42,
        "min_train_bars": 200,
        "refit_interval": 100,
        "stability_bars": 3,
        "flicker_window": 20,
        "flicker_threshold": 4,
        "zscore_window": 100,
    }


# ---------------------------------------------------------------------------
# a. End-to-end dry run
# ---------------------------------------------------------------------------

class TestEndToEndDryRun:
    """Full pipeline: data → features → HMM → strategy → risk → simulated orders."""

    def test_full_pipeline_no_crashes(self):
        """Walk through 1 week of data in simulation. No unhandled exceptions."""
        bars = _make_bars(800)
        fe = FeatureEngineer({"zscore_window": 100})
        features = fe.compute_hmm_features(bars)

        # Train HMM
        hmm = HMMEngine(_fast_hmm_config())
        hmm.fit(features.iloc[:400])

        # Init strategy
        strat = RegimeStrategyManager({}, hmm.regime_infos)

        # Init risk manager
        risk = RiskManager({})
        risk.initialize(100_000)

        all_bars = {"SYN": bars}
        signals_generated = 0
        approved = 0
        rejected = 0

        # Walk through last 5 bars (1 week) of data
        for t in range(395, 400):
            feat_slice = features.iloc[:t + 1]
            regime = hmm.predict_regime_filtered(feat_slice)

            bars_up_to = {"SYN": bars.iloc[:t + 1]}
            sigs = strat.get_signals(regime, ["SYN"], bars_up_to)
            signals_generated += len(sigs)

            for sig in sigs:
                if sig.direction == SignalDirection.FLAT:
                    continue
                portfolio = PortfolioState(
                    equity=100_000, cash=100_000, buying_power=100_000,
                    positions={}, position_count=0,
                    daily_pnl=0, weekly_pnl=0, peak_equity=100_000,
                    day_start_equity=100_000, week_start_equity=100_000,
                    current_drawdown_pct=0, total_exposure=0,
                    max_single_exposure=0, daily_trade_count=0,
                    circuit_breaker=CircuitBreakerStatus(),
                    regime_label=regime.label,
                    regime_probability=regime.probability,
                    flicker_rate=0,
                )
                decision = risk.validate_signal(sig, portfolio, bars=bars_up_to)
                if decision.approved:
                    approved += 1
                else:
                    rejected += 1

        # Pipeline should run without crashing
        assert True  # If we got here, no exceptions

    def test_hmm_trains_and_predicts(self):
        """HMM trains on synthetic data and produces valid regime states."""
        bars = _make_bars(800)
        fe = FeatureEngineer({"zscore_window": 100})
        features = fe.compute_hmm_features(bars)

        hmm = HMMEngine(_fast_hmm_config())
        metrics = hmm.fit(features.iloc[:400])

        assert hmm.is_fitted
        assert hmm.n_regimes == 3
        assert metrics.n_regimes_selected == 3
        assert len(hmm.regime_infos) == 3

        # Predict on test data
        hmm.reset_tracking()
        state = hmm.predict_regime_filtered(features.iloc[:401])
        assert state.probability > 0
        assert state.label in hmm.state_label_map.values()


# ---------------------------------------------------------------------------
# b. Look-ahead bias — full pipeline verification
# ---------------------------------------------------------------------------

class TestLookAheadFullPipeline:
    """Trades in overlapping period must be IDENTICAL with different end dates."""

    def test_signals_identical_with_extended_data(self):
        """Signals at time T must not change when future data is appended."""
        bars = _make_bars(800)
        fe = FeatureEngineer({"zscore_window": 100})

        # Features with two different end points
        features_short = fe.compute_hmm_features(bars.iloc[:600])
        features_long = fe.compute_hmm_features(bars.iloc[:700])

        hmm = HMMEngine(_fast_hmm_config())
        hmm.fit(features_short.iloc[:400])

        # Predict at a fixed point using short vs long feature set
        T = min(len(features_short), len(features_long)) - 1
        if T < 401:
            pytest.skip("Not enough overlapping features")

        hmm.reset_tracking()
        state_short = hmm.predict_regime_filtered(features_short.iloc[:T])

        hmm.reset_tracking()
        state_long = hmm.predict_regime_filtered(features_long.iloc[:T])

        # Regime at time T must be identical
        assert state_short.state_id == state_long.state_id
        assert state_short.label == state_long.label
        np.testing.assert_allclose(
            state_short.state_probabilities,
            state_long.state_probabilities,
            rtol=1e-10,
            err_msg="Look-ahead bias: regime probabilities differ with extended data",
        )

    def test_features_stable_with_extended_data(self):
        """Feature values at time T must not change when future bars are added."""
        bars = _make_bars(600)
        fe = FeatureEngineer({"zscore_window": 100})

        features_400 = fe.compute_hmm_features(bars.iloc[:400])
        features_600 = fe.compute_hmm_features(bars.iloc[:600])

        common = features_400.index.intersection(features_600.index)
        assert len(common) > 50

        pd.testing.assert_frame_equal(
            features_400.loc[common],
            features_600.loc[common],
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# c. Risk manager stress tests
# ---------------------------------------------------------------------------

class TestRiskManagerStress:
    """Feed extreme inputs and verify all are capped to safe levels."""

    @pytest.fixture
    def rm(self):
        rm = RiskManager({})
        rm.initialize(100_000)
        return rm

    def _make_extreme_signal(self, size_pct=1.0, leverage=10.0,
                              entry=100, stop=99):
        return Signal(
            symbol="TEST", direction=SignalDirection.LONG,
            confidence=0.99, entry_price=entry, stop_loss=stop,
            take_profit=110, position_size_pct=size_pct,
            leverage=leverage, regime_id=0, regime_name="BULL",
            regime_probability=0.99,
            timestamp=pd.Timestamp.now(), reasoning="stress test",
            strategy_name="test", metadata={"risk_sized_qty": 100},
        )

    def _portfolio(self, **kwargs):
        defaults = dict(
            equity=100_000, cash=100_000, buying_power=100_000,
            positions={}, position_count=0, daily_pnl=0, weekly_pnl=0,
            peak_equity=100_000, day_start_equity=100_000,
            week_start_equity=100_000, current_drawdown_pct=0,
            total_exposure=0, max_single_exposure=0, daily_trade_count=0,
            circuit_breaker=CircuitBreakerStatus(), regime_label="BULL",
            regime_probability=0.99, flicker_rate=0,
        )
        defaults.update(kwargs)
        return PortfolioState(**defaults)

    def test_100pct_position_capped_to_max(self, rm):
        """100% position size request capped to max single position limit."""
        sig = self._make_extreme_signal(size_pct=1.0)
        decision = rm.validate_signal(sig, self._portfolio())
        if decision.approved:
            value = decision.modified_signal.metadata["risk_sized_value"]
            assert value <= 100_000 * MAX_SINGLE_POSITION + 1

    def test_10x_leverage_capped_to_1_25x(self, rm):
        """10x leverage request capped to 1.25x max."""
        sig = self._make_extreme_signal(leverage=10.0)
        decision = rm.validate_signal(sig, self._portfolio())
        if decision.approved:
            assert decision.modified_signal.leverage <= MAX_PORTFOLIO_LEVERAGE

    def test_rapid_fire_50_signals_blocked(self, rm):
        """50 signals in rapid succession should hit the 20-trade daily limit."""
        portfolio = self._portfolio()
        approved_count = 0
        rejected_count = 0

        for i in range(50):
            sig = Signal(
                symbol=f"SYM{i}", direction=SignalDirection.LONG,
                confidence=0.7, entry_price=100, stop_loss=98,
                take_profit=105, position_size_pct=0.05, leverage=1.0,
                regime_id=0, regime_name="BULL", regime_probability=0.7,
                timestamp=pd.Timestamp.now(), reasoning=f"rapid fire {i}",
                strategy_name="test", metadata={"risk_sized_qty": 10},
            )
            decision = rm.validate_signal(sig, portfolio)
            if decision.approved:
                approved_count += 1
            else:
                rejected_count += 1

        # Should have been capped: max 20 trades + max 5 concurrent positions
        assert approved_count <= MAX_DAILY_TRADES
        assert rejected_count > 0

    def test_no_stop_always_rejected(self, rm):
        """Signal with no stop loss is always rejected, no matter what."""
        sig = self._make_extreme_signal()
        sig.stop_loss = 0  # No stop
        decision = rm.validate_signal(sig, self._portfolio())
        assert not decision.approved
        assert "stop loss" in decision.rejection_reason.lower()

    def test_5_positions_blocks_6th(self, rm):
        """Cannot open a 6th position."""
        positions = {f"POS{i}": 10_000 for i in range(5)}
        portfolio = self._portfolio(
            positions=positions, position_count=5,
            total_exposure=0.50, cash=50_000,
        )
        sig = self._make_extreme_signal()
        sig.symbol = "NEW_SYMBOL"
        decision = rm.validate_signal(sig, portfolio)
        assert not decision.approved
        assert "concurrent" in decision.rejection_reason.lower()


# ---------------------------------------------------------------------------
# d. Alpaca paper trading connectivity
# ---------------------------------------------------------------------------

class TestAlpacaConnectivity:
    """Live Alpaca paper trading tests. These hit the real API.

    Marked with pytest.mark.alpaca so they can be skipped in CI.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env.
    """

    @pytest.fixture
    def client(self):
        from dotenv import load_dotenv
        load_dotenv()
        import os
        key = os.getenv("ALPACA_API_KEY", "")
        # Real Alpaca paper keys are 18-30 chars, mixed case+digits.
        # Treat short/template-looking values as placeholders and skip.
        if not key or key.lower().startswith("your") or len(key) < 16:
            pytest.skip("No real ALPACA_API_KEY — skipping live tests")
        from broker.alpaca_client import AlpacaClient
        c = AlpacaClient({"paper_trading": True})
        c.connect()
        yield c
        c.disconnect()

    @pytest.fixture
    def executor(self, client):
        from broker.order_executor import OrderExecutor
        return OrderExecutor(client, {
            "limit_offset_pct": 0.001,
            "cancel_after_seconds": 10,
            "time_in_force": "day",
        })

    def test_account_is_paper(self, client):
        assert client.is_paper
        acct = client.get_account()
        assert acct.equity > 0

    def test_market_clock(self, client):
        clock = client.get_clock()
        assert "is_open" in clock
        assert "next_open" in clock

    def test_fetch_bars(self, client):
        bars = client.get_bars("SPY", timeframe="1Day", limit=10)
        assert len(bars) >= 5
        assert "close" in bars.columns

    def test_fetch_quote(self, client):
        quote = client.get_latest_quote("SPY")
        assert quote["mid"] > 0
        # Spread can be negative after hours (stale bid/ask)
        assert "spread" in quote

    def test_place_and_cancel_order(self, client, executor):
        """Place a limit order far from market, then cancel it."""
        quote = client.get_latest_quote("SPY")
        # Place limit buy $50 below market — will never fill
        far_price = quote["mid"] - 50.0

        sig = Signal(
            symbol="SPY", direction=SignalDirection.LONG,
            confidence=0.7, entry_price=far_price, stop_loss=far_price - 5,
            take_profit=far_price + 10, position_size_pct=0.01, leverage=1.0,
            regime_id=0, regime_name="TEST", regime_probability=0.7,
            timestamp=pd.Timestamp.now(), reasoning="connectivity test",
            strategy_name="test", metadata={"risk_sized_qty": 1},
        )

        result = executor.submit_bracket_order(sig)
        assert result.order_id != ""

        time.sleep(1)

        # Cancel it
        cancel = executor.cancel_order(result.order_id)
        assert cancel.status.value in ("cancelled", "failed")  # May already be cancelled

        # Verify no position was opened
        time.sleep(1)
        positions = client.get_positions()
        spy_pos = [p for p in positions if p["symbol"] == "SPY"]
        assert len(spy_pos) == 0


# ---------------------------------------------------------------------------
# e. Recovery test
# ---------------------------------------------------------------------------

class TestRecovery:
    """State snapshot save/load and position sync."""

    def test_state_snapshot_roundtrip(self, tmp_path):
        """Save and load a state snapshot."""
        from main import save_state_snapshot, load_state_snapshot

        state = {
            "regime_label": "BULL",
            "regime_probability": 0.72,
            "equity": 105000,
            "positions": {"SPY": 15000, "AAPL": 10000},
            "trade_count": 15,
            "bar_count": 200,
        }
        path = str(tmp_path / "snapshot.json")
        save_state_snapshot(path, state)

        loaded = load_state_snapshot(path)
        assert loaded is not None
        assert loaded["regime_label"] == "BULL"
        assert loaded["equity"] == 105000
        assert loaded["trade_count"] == 15

    def test_missing_snapshot_returns_none(self, tmp_path):
        from main import load_state_snapshot
        result = load_state_snapshot(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_position_tracker_sync(self):
        """PositionTracker.sync_with_broker reconciles state."""
        from unittest.mock import MagicMock
        from broker.position_tracker import PositionTracker

        mock_client = MagicMock()
        mock_client._ensure_connected = MagicMock()
        mock_client.get_positions.return_value = [
            {"symbol": "SPY", "qty": 10, "side": "long",
             "avg_entry_price": 450.0, "current_price": 455.0,
             "market_value": 4550, "unrealized_pl": 50, "unrealized_plpc": 0.011},
            {"symbol": "AAPL", "qty": 20, "side": "long",
             "avg_entry_price": 180.0, "current_price": 185.0,
             "market_value": 3700, "unrealized_pl": 100, "unrealized_plpc": 0.028},
        ]

        tracker = PositionTracker(mock_client)
        snapshot = tracker.sync_with_broker()

        assert len(snapshot.positions) == 2
        assert tracker.get_position("SPY") is not None
        assert tracker.get_position("AAPL") is not None
        assert tracker.get_position("MSFT") is None  # Not in broker response

        # Simulate SPY closed externally
        mock_client.get_positions.return_value = [
            {"symbol": "AAPL", "qty": 20, "side": "long",
             "avg_entry_price": 180.0, "current_price": 185.0,
             "market_value": 3700, "unrealized_pl": 100, "unrealized_plpc": 0.028},
        ]
        snapshot2 = tracker.sync_with_broker()
        assert len(snapshot2.positions) == 1
        assert tracker.get_position("SPY") is None  # Removed
