"""Tests for Phase 2c trailing stops (A2)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

import main as main_module
from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor
from main import TradingLoop


def _flat_bars(price: float = 460.0, n: int = 60) -> pd.DataFrame:
    """Constant-range bars so ATR(14) is deterministically 2.0 (high-low=2 -> TR=2)."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": price, "high": price + 1.0, "low": price - 1.0,
         "close": price, "volume": 1e6},
        index=idx,
    )


def _loop(trail_atr=2.0, tracker=None, executor=None):
    main_module.logger = MagicMock()  # methods log via the module-global logger
    return TradingLoop(
        config={"universe": {"symbols": ["SPY"]}, "schedule": {}, "model": {},
                "execution": {"trail_stop_atr": trail_atr}},
        alpaca_client=MagicMock(), hmm_engine=MagicMock(),
        feature_engineer=MagicMock(), strategy_manager=MagicMock(),
        risk_manager=MagicMock(), order_executor=executor or MagicMock(),
        position_tracker=tracker or MagicMock(), market_data=MagicMock(),
        hmm_features=pd.DataFrame(), dry_run=True,
    )


# --- get_open_stop_order -----------------------------------------------------

def test_get_open_stop_order_finds_stop():
    client = MagicMock(spec=AlpacaClient)
    client._ensure_connected = MagicMock()
    client.trading_client = MagicMock()
    client.trading_client.get_orders.return_value = [
        SimpleNamespace(id="L1", type="limit", stop_price=None, side="buy"),
        SimpleNamespace(id="S1", type="stop", stop_price="445.5", side="sell"),
    ]
    ex = OrderExecutor(client, {})
    assert ex.get_open_stop_order("SPY") == {
        "order_id": "S1", "stop_price": 445.5, "side": "sell"}


def test_get_open_stop_order_none_when_no_stop():
    client = MagicMock(spec=AlpacaClient)
    client._ensure_connected = MagicMock()
    client.trading_client = MagicMock()
    client.trading_client.get_orders.return_value = [
        SimpleNamespace(id="L1", type="limit", stop_price=None, side="buy")]
    ex = OrderExecutor(client, {})
    assert ex.get_open_stop_order("SPY") is None


# --- _trail_stops ------------------------------------------------------------

def test_trail_tightens_long_stop_when_favorable():
    tracker = MagicMock()
    tracker.positions = {"SPY": SimpleNamespace(qty=10, side="long")}
    ex = MagicMock()
    ex.get_open_stop_order.return_value = {"order_id": "S1", "stop_price": 440.0, "side": "sell"}
    ex.modify_stop.return_value = SimpleNamespace(status=SimpleNamespace(value="accepted"))

    loop = _loop(trail_atr=2.0, tracker=tracker, executor=ex)
    loop._trail_stops({"SPY": _flat_bars(460.0)})  # ATR≈2 -> desired 456 > 440

    ex.modify_stop.assert_called_once()
    args, kwargs = ex.modify_stop.call_args
    assert args[0] == "S1"
    assert args[1] == 456.0
    assert kwargs.get("current_side") == "buy"


def test_trail_noop_when_not_tighter():
    tracker = MagicMock()
    tracker.positions = {"SPY": SimpleNamespace(qty=10, side="long")}
    ex = MagicMock()
    ex.get_open_stop_order.return_value = {"order_id": "S1", "stop_price": 458.0, "side": "sell"}

    loop = _loop(trail_atr=2.0, tracker=tracker, executor=ex)
    loop._trail_stops({"SPY": _flat_bars(460.0)})  # desired 456 < existing 458

    ex.modify_stop.assert_not_called()


def test_trail_disabled_when_zero():
    tracker = MagicMock()
    tracker.positions = {"SPY": SimpleNamespace(qty=10, side="long")}
    ex = MagicMock()
    loop = _loop(trail_atr=0.0, tracker=tracker, executor=ex)
    loop._trail_stops({"SPY": _flat_bars(460.0)})
    ex.get_open_stop_order.assert_not_called()
    ex.modify_stop.assert_not_called()
