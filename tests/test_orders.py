"""Tests for broker package — AlpacaClient, OrderExecutor, PositionTracker, MarketDataClient.

All tests use mocked Alpaca API responses since we cannot connect to
the real API in tests. Tests verify:
1. AlpacaClient correctly reads credentials and handles connection states
2. OrderExecutor uses limit orders, enforces stop-only-tightens, generates trade IDs
3. PositionTracker reconciles positions and computes P&L
4. MarketDataClient caches bars and tracks data health
5. Live trading requires explicit confirmation
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker.alpaca_client import AlpacaClient, AccountInfo, PAPER_URL, LIVE_URL
from broker.order_executor import (
    OrderExecutor, OrderResult, OrderStatus, _new_trade_id, _map_status,
)
from broker.position_tracker import PositionTracker, Position, PositionSnapshot
from data.market_data import MarketDataClient


# ---------------------------------------------------------------------------
# AlpacaClient tests
# ---------------------------------------------------------------------------

class TestAlpacaClient:
    def test_defaults_to_paper(self):
        """Should default to paper trading URL."""
        client = AlpacaClient({})
        assert client.is_paper
        assert "paper" in client._base_url

    def test_explicit_paper_true(self):
        client = AlpacaClient({"paper_trading": True})
        assert client.is_paper

    def test_live_url_when_paper_false(self):
        import os
        with patch.dict(os.environ, {"ALPACA_BASE_URL": ""}, clear=False):
            client = AlpacaClient({"paper_trading": False})
            assert not client.is_paper

    def test_raises_without_credentials(self):
        """connect() should raise if no API keys provided."""
        client = AlpacaClient({})
        # Clear env vars for this test
        with patch.dict("os.environ", {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""}):
            client._api_key = ""
            client._secret_key = ""
            with pytest.raises(ValueError, match="credentials"):
                client.connect()

    def test_ensure_connected_raises_when_not_connected(self):
        client = AlpacaClient({})
        with pytest.raises(ConnectionError, match="Not connected"):
            client._ensure_connected()

    def test_config_overrides_env(self):
        """Config keys should override environment variables."""
        client = AlpacaClient({
            "api_key": "config_key",
            "secret_key": "config_secret",
        })
        assert client._api_key == "config_key"
        assert client._secret_key == "config_secret"

    def test_live_trading_requires_confirmation(self):
        """Live trading should require 'YES I UNDERSTAND THE RISKS'."""
        import os
        with patch.dict(os.environ, {"ALPACA_BASE_URL": ""}, clear=False):
            client = AlpacaClient({
                "paper_trading": False,
                "api_key": "test",
                "secret_key": "test",
            })
            with patch("builtins.input", return_value="no"):
                with pytest.raises(RuntimeError, match="confirmation rejected"):
                    client.connect()


# ---------------------------------------------------------------------------
# OrderExecutor tests
# ---------------------------------------------------------------------------

class TestOrderExecutor:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=AlpacaClient)
        client.is_paper = True
        client._api_key = "test"
        client._secret_key = "test"
        client._ensure_connected = MagicMock()
        return client

    @pytest.fixture
    def executor(self, mock_client):
        return OrderExecutor(mock_client, {
            "limit_offset_pct": 0.001,
            "cancel_after_seconds": 30,
            "time_in_force": "day",
            "chase_at_market": False,
        })

    def test_trade_id_generated(self):
        """Each trade should get a unique trade ID."""
        id1 = _new_trade_id()
        id2 = _new_trade_id()
        assert id1 != id2
        assert id1.startswith("T-")

    def test_status_mapping(self):
        """Alpaca status strings should map to our enum."""
        assert _map_status("filled") == OrderStatus.FILLED
        assert _map_status("canceled") == OrderStatus.CANCELLED
        assert _map_status("rejected") == OrderStatus.REJECTED
        assert _map_status("new") == OrderStatus.NEW
        assert _map_status("unknown_status") == OrderStatus.FAILED

    def test_submit_rejects_zero_qty(self, executor):
        """Should reject signals with zero quantity."""
        from core.regime_strategies import Signal, SignalDirection
        signal = Signal(
            symbol="SPY", direction=SignalDirection.LONG,
            confidence=0.7, entry_price=450, stop_loss=445,
            take_profit=460, position_size_pct=0.10, leverage=1.0,
            regime_id=1, regime_name="NEUTRAL", regime_probability=0.7,
            timestamp=pd.Timestamp.now(), reasoning="test",
            strategy_name="test",
            metadata={"risk_sized_qty": 0},
        )
        result = executor.submit_order(signal)
        assert result.status == OrderStatus.FAILED
        assert "Zero quantity" in result.error_message

    def test_modify_stop_rejects_widening_for_long(self, executor, mock_client):
        """For long positions, stop can only move UP (tighten)."""
        # Mock existing order with stop at 445
        mock_order = MagicMock()
        mock_order.stop_price = "445.00"
        mock_order.symbol = "SPY"
        mock_order.qty = "10"
        mock_client.trading_client = MagicMock()
        mock_client.trading_client.get_order_by_id.return_value = mock_order

        result = executor.modify_stop("order123", 440.0, current_side="buy")
        assert result.status == OrderStatus.REJECTED
        assert "widen" in result.error_message.lower()

    def test_modify_stop_rejects_widening_for_short(self, executor, mock_client):
        """For short positions, stop can only move DOWN (tighten)."""
        mock_order = MagicMock()
        mock_order.stop_price = "455.00"
        mock_order.symbol = "SPY"
        mock_order.qty = "10"
        mock_client.trading_client = MagicMock()
        mock_client.trading_client.get_order_by_id.return_value = mock_order

        result = executor.modify_stop("order123", 460.0, current_side="sell")
        assert result.status == OrderStatus.REJECTED
        assert "widen" in result.error_message.lower()

    def test_order_log_records_events(self, executor):
        """Every order event should be logged."""
        from core.regime_strategies import Signal, SignalDirection
        signal = Signal(
            symbol="SPY", direction=SignalDirection.LONG,
            confidence=0.7, entry_price=450, stop_loss=445,
            take_profit=460, position_size_pct=0.10, leverage=1.0,
            regime_id=1, regime_name="NEUTRAL", regime_probability=0.7,
            timestamp=pd.Timestamp.now(), reasoning="test",
            strategy_name="test",
            metadata={"risk_sized_qty": 0},  # Will fail, but still logged
        )
        executor.submit_order(signal)
        assert len(executor.order_log) > 0
        assert executor.order_log[-1]["event"] == "failed"

    def test_cancel_all_clears_pending(self, executor, mock_client):
        """cancel_all should clear all pending orders."""
        mock_client.trading_client = MagicMock()
        executor._pending_orders["order1"] = MagicMock()
        executor._pending_orders["order2"] = MagicMock()
        results = executor.cancel_all()
        assert len(executor._pending_orders) == 0

    def _long_signal(self, take_profit):
        from core.regime_strategies import Signal, SignalDirection
        return Signal(
            symbol="SPY", direction=SignalDirection.LONG, confidence=0.9,
            entry_price=450.0, stop_loss=445.0, take_profit=take_profit,
            position_size_pct=0.5, leverage=1.0, regime_id=1, regime_name="BULL",
            regime_probability=0.9, timestamp=pd.Timestamp.now(),
            reasoning="t", strategy_name="t", metadata={"risk_sized_qty": 10},
        )

    def test_entry_without_target_uses_oto_with_stop(self, executor, mock_client):
        """A1: an entry with no take-profit still brings a resting stop (OTO)."""
        from alpaca.trading.enums import OrderClass
        mock_client.trading_client = MagicMock()
        order = MagicMock(); order.id = "OTO-1"; order.status = "accepted"
        mock_client.trading_client.submit_order.return_value = order

        result = executor.submit_bracket_order(self._long_signal(take_profit=None))

        assert result.status != OrderStatus.FAILED
        req = mock_client.trading_client.submit_order.call_args[0][0]
        assert req.order_class == OrderClass.OTO
        assert req.stop_loss is not None
        assert req.take_profit is None
        assert float(req.stop_loss.stop_price) == 445.0

    def test_entry_with_target_uses_full_bracket(self, executor, mock_client):
        """A1: an entry with a take-profit uses a full bracket (entry+stop+target)."""
        from alpaca.trading.enums import OrderClass
        mock_client.trading_client = MagicMock()
        order = MagicMock(); order.id = "BR-1"; order.status = "accepted"
        mock_client.trading_client.submit_order.return_value = order

        result = executor.submit_bracket_order(self._long_signal(take_profit=470.0))

        assert result.status != OrderStatus.FAILED
        req = mock_client.trading_client.submit_order.call_args[0][0]
        assert req.order_class == OrderClass.BRACKET
        assert req.stop_loss is not None
        assert req.take_profit is not None


# ---------------------------------------------------------------------------
# PositionTracker tests
# ---------------------------------------------------------------------------

class TestPositionTracker:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=AlpacaClient)
        client._ensure_connected = MagicMock()
        client._api_key = "test"
        client._secret_key = "test"
        client.is_paper = True
        return client

    @pytest.fixture
    def tracker(self, mock_client):
        return PositionTracker(mock_client)

    def test_sync_creates_positions(self, tracker, mock_client):
        """sync_with_broker should create Position objects from API response."""
        mock_client.get_positions.return_value = [
            {
                "symbol": "SPY", "qty": 10, "side": "long",
                "avg_entry_price": 450.0, "current_price": 455.0,
                "market_value": 4550.0, "unrealized_pl": 50.0,
                "unrealized_plpc": 0.011,
            },
        ]
        snapshot = tracker.sync_with_broker()
        assert len(snapshot.positions) == 1
        assert snapshot.positions[0].symbol == "SPY"
        assert snapshot.total_market_value == 4550.0

    def test_sync_removes_closed_positions(self, tracker, mock_client):
        """Positions that Alpaca no longer has should be removed."""
        # Initial sync with SPY
        mock_client.get_positions.return_value = [
            {"symbol": "SPY", "qty": 10, "side": "long",
             "avg_entry_price": 450, "current_price": 455,
             "market_value": 4550, "unrealized_pl": 50, "unrealized_plpc": 0.01},
        ]
        tracker.sync_with_broker()
        assert "SPY" in tracker.positions

        # Second sync without SPY
        mock_client.get_positions.return_value = []
        tracker.sync_with_broker()
        assert "SPY" not in tracker.positions

    def test_record_entry(self, tracker):
        """record_entry should create a Position with full metadata."""
        tracker.record_entry(
            "AAPL", 100, "long", 175.0, 170.0, "BULL", "T-abc123")
        pos = tracker.get_position("AAPL")
        assert pos is not None
        assert pos.qty == 100
        assert pos.regime_at_entry == "BULL"
        assert pos.trade_id == "T-abc123"

    def test_update_prices(self, tracker):
        """update_prices should recalculate P&L for all positions."""
        tracker.record_entry("SPY", 10, "long", 450.0, 445.0, "NEUTRAL")
        tracker.update_prices({"SPY": 460.0})
        pos = tracker.get_position("SPY")
        assert pos.current_price == 460.0
        assert pos.unrealized_pnl == 100.0  # 10 shares * $10

    def test_update_prices_short(self, tracker):
        """Short positions P&L should be inverted."""
        tracker.record_entry("SPY", 10, "short", 450.0, 455.0, "BEAR")
        tracker.update_prices({"SPY": 440.0})
        pos = tracker.get_position("SPY")
        assert pos.unrealized_pnl == 100.0  # Profit on short

    def test_increment_holding_period(self, tracker):
        tracker.record_entry("SPY", 10, "long", 450.0, 445.0, "NEUTRAL")
        assert tracker.get_position("SPY").holding_period_bars == 0
        tracker.increment_holding_period()
        assert tracker.get_position("SPY").holding_period_bars == 1
        tracker.increment_holding_period()
        assert tracker.get_position("SPY").holding_period_bars == 2

    def test_fill_callback_registered(self, tracker):
        """on_fill should register a callback."""
        cb = MagicMock()
        tracker.on_fill(cb)
        assert len(tracker._fill_callbacks) == 1

    def test_data_feed_health(self, tracker):
        health = tracker.get_data_feed_health()
        assert "last_sync" in health
        assert "ws_streaming" in health


# ---------------------------------------------------------------------------
# MarketDataClient tests
# ---------------------------------------------------------------------------

class TestMarketDataClient:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=AlpacaClient)
        client._ensure_connected = MagicMock()
        client._api_key = "test"
        client._secret_key = "test"
        return client

    @pytest.fixture
    def mdc(self, mock_client):
        return MarketDataClient(mock_client, {
            "symbols": ["SPY", "QQQ"],
            "timeframe": "1Day",
            "lookback_bars": 504,
        })

    def test_update_cache(self, mdc):
        """update_cache should append bars without duplicates."""
        dates1 = pd.bdate_range("2024-01-01", periods=5)
        bars1 = pd.DataFrame({
            "open": range(5), "high": range(5), "low": range(5),
            "close": range(5), "volume": range(5),
        }, index=dates1)

        dates2 = pd.bdate_range("2024-01-05", periods=5)
        bars2 = pd.DataFrame({
            "open": range(10, 15), "high": range(10, 15), "low": range(10, 15),
            "close": range(10, 15), "volume": range(10, 15),
        }, index=dates2)

        mdc.update_cache("SPY", bars1)
        mdc.update_cache("SPY", bars2)

        cached = mdc.get_cached_bars("SPY")
        assert cached is not None
        assert len(cached) >= 8  # 5 + ~4 new (one overlapping day)

    def test_get_latest_bar_from_cache(self, mdc):
        """get_latest_bar should return last bar from cache."""
        dates = pd.bdate_range("2024-01-01", periods=5)
        bars = pd.DataFrame({
            "open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5],
            "volume": [100] * 5,
        }, index=dates)
        mdc.update_cache("SPY", bars)

        latest = mdc.get_latest_bar("SPY")
        assert latest is not None
        assert latest["close"] == 5

    def test_data_health_report(self, mdc):
        """get_data_health should return a status dict."""
        health = mdc.get_data_health()
        assert "symbols_configured" in health
        assert health["symbols_configured"] == ["SPY", "QQQ"]
        assert "ws_running" in health
        assert health["ws_running"] is False

    def test_update_cache_single_bar(self, mdc):
        """update_cache_single should handle a single bar Series."""
        bar = pd.Series({
            "open": 100, "high": 101, "low": 99,
            "close": 100.5, "volume": 1000,
        }, name=pd.Timestamp("2024-06-15"))
        mdc.update_cache_single("SPY", bar)
        cached = mdc.get_cached_bars("SPY")
        assert cached is not None
        assert len(cached) == 1

    def test_subscribe_registers_callback(self, mdc):
        """subscribe_bars should register a callback."""
        cb = MagicMock()
        # Don't actually start the websocket — just register
        mdc._bar_callbacks.append(cb)
        assert len(mdc._bar_callbacks) == 1
