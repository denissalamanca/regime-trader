"""Phase C: pair-aware risk validation and order execution.

Covers:
- RiskManager.validate_pair_signal: happy path, leg rejections, pair-specific
  rules (count as 1 position / 2 trades), pair-correlation reject, exposure
  scaling.
- OrderExecutor.submit_pair_order: both-fill happy path, one-leg-fill unwind,
  neither-fill cleanup. The executor here uses a fake AlpacaClient so we can
  drive fill outcomes deterministically.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker.order_executor import OrderExecutor, OrderResult, OrderStatus
from core.regime_strategies import PairSignal, Signal, SignalDirection
from core.risk_manager import (
    BreakerLevel,
    CircuitBreakerStatus,
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_TRADES,
    PairRiskDecision,
    PortfolioState,
    RiskManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal(symbol, direction, entry, stop, size_pct=0.10, regime="NEUTRAL"):
    return Signal(
        symbol=symbol, direction=direction,
        confidence=0.7, entry_price=entry, stop_loss=stop, take_profit=None,
        position_size_pct=size_pct, leverage=1.0,
        regime_id=1, regime_name=regime, regime_probability=0.7,
        timestamp=pd.Timestamp("2024-06-15"),
        reasoning="test", strategy_name="pair_test",
    )


def _pair(long_sym="SPY", short_sym="IWM",
          long_entry=400.0, short_entry=200.0,
          long_stop=380.0, short_stop=210.0,
          z=2.5, hedge=1.0):
    return PairSignal(
        pair=(long_sym, short_sym),
        long_leg=_signal(long_sym, SignalDirection.LONG, long_entry, long_stop),
        short_leg=_signal(short_sym, SignalDirection.SHORT, short_entry, short_stop),
        spread_value=0.0, z_score=z, hedge_ratio=hedge,
        reasoning="test pair", timestamp=pd.Timestamp("2024-06-15"),
    )


def _portfolio(equity=100_000.0, positions=None, position_count=None,
               daily_trades=0, cb_level=BreakerLevel.NONE):
    if positions is None:
        positions = {}
    if position_count is None:
        position_count = len(positions)
    cb = CircuitBreakerStatus(
        level=cb_level,
        size_multiplier=(0.5 if cb_level in (BreakerLevel.DAILY_REDUCE, BreakerLevel.WEEKLY_REDUCE) else
                         0.0 if cb_level in (BreakerLevel.DAILY_HALT, BreakerLevel.WEEKLY_HALT, BreakerLevel.PEAK_HALT) else 1.0),
        is_halted=cb_level in (BreakerLevel.DAILY_HALT, BreakerLevel.WEEKLY_HALT, BreakerLevel.PEAK_HALT),
        halt_reason="test halt" if cb_level in (BreakerLevel.DAILY_HALT, BreakerLevel.WEEKLY_HALT, BreakerLevel.PEAK_HALT) else "",
    )
    return PortfolioState(
        equity=equity, cash=equity, buying_power=equity,
        positions=positions, position_count=position_count,
        daily_pnl=0, weekly_pnl=0,
        peak_equity=equity, day_start_equity=equity, week_start_equity=equity,
        current_drawdown_pct=0,
        total_exposure=sum(abs(v) for v in positions.values()) / equity if equity > 0 else 0,
        max_single_exposure=max((abs(v) for v in positions.values()), default=0) / equity if equity > 0 else 0,
        daily_trade_count=daily_trades,
        circuit_breaker=cb,
    )


def _correlated_bars(symbols, n=120, seed=42, drift=0.0005, vol=0.01):
    """Bars with high cross-correlation (shared shocks plus a small idio component)."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    common = rng.normal(drift, vol, n)
    out = {}
    for i, s in enumerate(symbols):
        idio = rng.normal(0, 0.0015, n)
        rets = common + idio
        close = 100.0 * np.exp(np.cumsum(rets))
        out[s] = pd.DataFrame({
            "open": close * 1.001, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": rng.uniform(1e6, 2e6, n),
        }, index=dates)
    return out


def _uncorrelated_bars(symbols, n=120):
    """Bars driven by independent random walks (~ zero correlation)."""
    out = {}
    for i, s in enumerate(symbols):
        rng = np.random.RandomState(7919 + i * 31)
        dates = pd.bdate_range("2024-01-01", periods=n)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
        out[s] = pd.DataFrame({
            "open": close * 1.001, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": rng.uniform(1e6, 2e6, n),
        }, index=dates)
    return out


# ===========================================================================
# RiskManager.validate_pair_signal
# ===========================================================================

class TestValidatePairSignalHappyPath:
    @pytest.fixture
    def rm(self):
        return RiskManager({})

    def test_valid_pair_approved(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        decision = rm.validate_pair_signal(_pair(), _portfolio(), bars=bars)
        assert decision.approved
        assert decision.modified_pair is not None
        assert decision.rejection_reason == ""

    def test_modified_pair_has_qty_set(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        d = rm.validate_pair_signal(_pair(), _portfolio(), bars=bars)
        long_q = d.modified_pair.long_leg.metadata.get("risk_sized_qty", 0)
        short_q = d.modified_pair.short_leg.metadata.get("risk_sized_qty", 0)
        assert long_q > 0
        assert short_q > 0

    def test_pair_role_metadata(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        d = rm.validate_pair_signal(_pair(), _portfolio(), bars=bars)
        assert d.modified_pair.long_leg.metadata["pair_role"] == "long"
        assert d.modified_pair.short_leg.metadata["pair_role"] == "short"

    def test_pair_advances_daily_count_by_2(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        assert rm._daily_trade_count == 0
        rm.validate_pair_signal(_pair(), _portfolio(), bars=bars)
        assert rm._daily_trade_count == 2


class TestValidatePairSignalRejection:
    @pytest.fixture
    def rm(self):
        return RiskManager({})

    def test_rejected_when_circuit_breaker_halt(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        portfolio = _portfolio(cb_level=BreakerLevel.DAILY_HALT)
        d = rm.validate_pair_signal(_pair(), portfolio, bars=bars)
        assert not d.approved
        assert "Circuit breaker" in d.rejection_reason

    def test_rejected_when_long_leg_missing_stop(self, rm):
        ps = _pair()
        ps.long_leg.stop_loss = 0.0
        d = rm.validate_pair_signal(ps, _portfolio())
        assert not d.approved
        assert "long" in d.rejection_reason
        assert "stop" in d.rejection_reason.lower()

    def test_rejected_when_short_leg_missing_stop(self, rm):
        ps = _pair()
        ps.short_leg.stop_loss = 0.0
        d = rm.validate_pair_signal(ps, _portfolio())
        assert not d.approved
        assert "short" in d.rejection_reason

    def test_rejected_when_long_leg_direction_wrong(self, rm):
        ps = _pair()
        ps.long_leg.direction = SignalDirection.SHORT  # bad
        d = rm.validate_pair_signal(ps, _portfolio())
        assert not d.approved
        assert "LONG" in d.rejection_reason

    def test_rejected_when_short_leg_direction_wrong(self, rm):
        ps = _pair()
        ps.short_leg.direction = SignalDirection.LONG  # bad
        d = rm.validate_pair_signal(ps, _portfolio())
        assert not d.approved
        assert "SHORT" in d.rejection_reason

    def test_rejected_when_daily_trade_count_would_exceed(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        # daily trade count already at MAX-1; pair would push it to MAX+1
        rm._daily_trade_count = MAX_DAILY_TRADES - 1
        d = rm.validate_pair_signal(_pair(), _portfolio(daily_trades=MAX_DAILY_TRADES - 1), bars=bars)
        assert not d.approved
        assert "max daily trades" in d.rejection_reason.lower()

    def test_rejected_when_concurrent_positions_full(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        # Already at max with 5 unrelated positions
        positions = {f"X{i}": 5_000 for i in range(MAX_CONCURRENT_POSITIONS)}
        d = rm.validate_pair_signal(_pair(), _portfolio(positions=positions), bars=bars)
        assert not d.approved
        assert "max concurrent" in d.rejection_reason.lower()

    def test_pair_counts_as_one_position_when_room_for_one(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        # 4 unrelated positions; max is 5; pair needs 1 slot — should fit.
        positions = {f"X{i}": 5_000 for i in range(MAX_CONCURRENT_POSITIONS - 1)}
        d = rm.validate_pair_signal(_pair(), _portfolio(positions=positions), bars=bars)
        assert d.approved, d.rejection_reason

    def test_low_correlation_pair_rejected(self, rm):
        # Uncorrelated bars → reject as "cointegration likely broken"
        bars = _uncorrelated_bars(["SPY", "IWM"])
        d = rm.validate_pair_signal(_pair(), _portfolio(), bars=bars)
        assert not d.approved
        assert "correlation" in d.rejection_reason.lower()

    def test_no_bars_skips_correlation_check(self, rm):
        # When bars is None we can't check correlation — should approve on a
        # well-formed pair.
        d = rm.validate_pair_signal(_pair(), _portfolio(), bars=None)
        assert d.approved

    def test_rejected_when_either_leg_has_zero_stop_distance(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        ps = _pair(long_entry=400.0, long_stop=400.0)  # entry == stop
        d = rm.validate_pair_signal(ps, _portfolio(), bars=bars)
        assert not d.approved
        assert "stop" in d.rejection_reason.lower()


class TestValidatePairSignalSizing:
    @pytest.fixture
    def rm(self):
        return RiskManager({})

    def test_circuit_breaker_reduce_scales_both_legs(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        baseline = rm.validate_pair_signal(_pair(), _portfolio(), bars=bars)
        rm2 = RiskManager({})
        reduced = rm2.validate_pair_signal(
            _pair(), _portfolio(cb_level=BreakerLevel.DAILY_REDUCE), bars=bars,
        )
        assert reduced.approved
        # Both leg sizes should be ≤ baseline (often ~0.5x but sizing caps may
        # round; test the inequality).
        assert reduced.modified_pair.long_leg.position_size_pct <= baseline.modified_pair.long_leg.position_size_pct
        assert reduced.modified_pair.short_leg.position_size_pct <= baseline.modified_pair.short_leg.position_size_pct

    def test_combined_legs_dont_blow_max_exposure(self, rm):
        bars = _correlated_bars(["SPY", "IWM"])
        # Existing 70% of equity in unrelated names, max is 80%.
        # Pair sizing must be scaled down so total fits.
        positions = {"AAA": 70_000.0}
        d = rm.validate_pair_signal(_pair(), _portfolio(positions=positions), bars=bars)
        # Either approved with scale-down modification, or rejected with no room
        if d.approved:
            total = (d.modified_pair.long_leg.position_size_pct
                     + d.modified_pair.short_leg.position_size_pct)
            # Existing 70% + total new <= 80% + small float fudge
            assert 70_000.0 + total * 100_000 <= 80_000.0 + 1e-6
        else:
            assert "exposure" in d.rejection_reason.lower() or "min" in d.rejection_reason.lower()


# ===========================================================================
# OrderExecutor.submit_pair_order
# ===========================================================================

class _FakeOrder:
    """Stand-in for an alpaca-py order object."""
    def __init__(self, oid, symbol, side, qty, status="accepted",
                 filled_avg_price=None, filled_qty=None,
                 stop_price=None, type_="limit"):
        self.id = oid
        self.symbol = symbol
        # Use an enum-like .value attribute so _map_status works
        self.side = MagicMock(value=side)
        self.qty = qty
        self.status = MagicMock(value=status)
        self.type = MagicMock(value=type_)
        self.filled_avg_price = filled_avg_price
        self.filled_qty = filled_qty if filled_qty is not None else (qty if status == "filled" else 0)
        self.stop_price = stop_price
        self.submitted_at = "2024-06-15T10:30:00"
        self.filled_at = "2024-06-15T10:30:05" if status == "filled" else None


class _FakeAlpacaClient:
    """Fake AlpacaClient that lets each test program order outcomes.

    Each call to submit_order returns the next pre-programmed order. Each call
    to get_order_by_id returns a programmed status (so the monitor thread can
    drive fill outcomes).
    """

    def __init__(self):
        self.trading_client = self
        self._next_orders: list = []
        self._statuses: dict = {}    # order_id -> sequence of statuses
        self.cancelled: list = []
        self._auto_id = 0
        self._submitted: list = []

    def queue_submit(self, *orders: _FakeOrder):
        self._next_orders.extend(orders)

    def set_statuses(self, oid, statuses: list):
        self._statuses[oid] = list(statuses)

    def submit_order(self, req):
        if not self._next_orders:
            raise RuntimeError("No more queued orders for FakeAlpacaClient")
        o = self._next_orders.pop(0)
        self._submitted.append((o.symbol, getattr(req, "side", None), o.qty))
        return o

    def get_order_by_id(self, oid):
        seq = self._statuses.get(oid, [])
        if seq:
            return seq.pop(0)
        # Default: return a "filled" snapshot if nothing programmed
        return _FakeOrder(oid, "?", "buy", 0, status="filled", filled_avg_price=100.0)

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)

    def cancel_orders(self):
        pass

    def close_position(self, symbol):
        pass

    def close_all_positions(self, **kwargs):
        pass

    def _ensure_connected(self):
        pass


def _make_pair_signal_for_executor():
    """A pair signal where each leg has metadata.risk_sized_qty (mimicking
    a post-risk-manager modified pair)."""
    long = _signal("SPY", SignalDirection.LONG, 400.0, 380.0)
    long.metadata = {"risk_sized_qty": 10}
    short = _signal("IWM", SignalDirection.SHORT, 200.0, 210.0)
    short.metadata = {"risk_sized_qty": 20}
    return PairSignal(
        pair=("SPY", "IWM"),
        long_leg=long, short_leg=short,
        spread_value=0.0, z_score=2.5, hedge_ratio=1.0,
        timestamp=pd.Timestamp("2024-06-15"),
    )


class TestSubmitPairOrder:
    @pytest.fixture
    def fake_client(self):
        return _FakeAlpacaClient()

    @pytest.fixture
    def executor(self, fake_client):
        # Short cancel window so monitoring threads finish quickly in tests
        return OrderExecutor(fake_client, {"cancel_after_seconds": 1})

    def test_zero_qty_returns_failure_pair(self, executor):
        ps = _make_pair_signal_for_executor()
        ps.long_leg.metadata["risk_sized_qty"] = 0
        long_res, short_res = executor.submit_pair_order(ps)
        assert long_res.status == OrderStatus.FAILED
        assert short_res.status == OrderStatus.FAILED

    def test_both_legs_submitted_with_shared_pair_id(self, fake_client, executor):
        fake_client.queue_submit(
            _FakeOrder("L-1", "SPY", "buy", 10, status="accepted"),
            _FakeOrder("S-1", "IWM", "sell", 20, status="accepted"),
        )
        ps = _make_pair_signal_for_executor()
        long_res, short_res = executor.submit_pair_order(ps, trade_id="PAIR-test1")
        assert long_res.trade_id == "PAIR-test1"
        assert short_res.trade_id == "PAIR-test1"
        assert long_res.symbol == "SPY"
        assert short_res.symbol == "IWM"
        assert long_res.qty == 10
        assert short_res.qty == 20

    def test_both_legs_use_different_sides(self, fake_client, executor):
        fake_client.queue_submit(
            _FakeOrder("L-1", "SPY", "buy", 10, status="accepted"),
            _FakeOrder("S-1", "IWM", "sell", 20, status="accepted"),
        )
        ps = _make_pair_signal_for_executor()
        long_res, short_res = executor.submit_pair_order(ps)
        assert long_res.side == "buy"
        assert short_res.side == "sell"

    def test_both_legs_fill_within_window(self, fake_client, executor):
        long_filled = _FakeOrder("L-1", "SPY", "buy", 10, status="filled", filled_avg_price=400.5)
        short_filled = _FakeOrder("S-1", "IWM", "sell", 20, status="filled", filled_avg_price=199.5)
        fake_client.queue_submit(
            _FakeOrder("L-1", "SPY", "buy", 10, status="accepted"),
            _FakeOrder("S-1", "IWM", "sell", 20, status="accepted"),
        )
        # Status polls return filled immediately
        fake_client.set_statuses("L-1", [long_filled, long_filled])
        fake_client.set_statuses("S-1", [short_filled, short_filled])
        ps = _make_pair_signal_for_executor()
        executor.submit_pair_order(ps, trade_id="PAIR-bothfill")
        # Wait for monitor thread to finish
        time.sleep(2.5)
        # No unwind log
        unwind_events = [e for e in executor.order_log if "unwind" in e["event"].lower()]
        assert unwind_events == []

    def test_only_long_fills_triggers_unwind(self, fake_client, executor):
        # Submission accepts both. Then status polls report long filled, short
        # never filled within window → cancel short, market-close long.
        long_filled = _FakeOrder("L-1", "SPY", "buy", 10, status="filled", filled_avg_price=400.5)
        short_pending = _FakeOrder("S-1", "IWM", "sell", 20, status="accepted")
        market_close = _FakeOrder("M-1", "SPY", "sell", 10, status="accepted", type_="market")
        fake_client.queue_submit(
            _FakeOrder("L-1", "SPY", "buy", 10, status="accepted"),
            _FakeOrder("S-1", "IWM", "sell", 20, status="accepted"),
            market_close,  # the unwind market order
        )
        fake_client.set_statuses("L-1", [long_filled, long_filled, long_filled])
        fake_client.set_statuses("S-1", [short_pending, short_pending, short_pending])
        ps = _make_pair_signal_for_executor()
        executor.submit_pair_order(ps, trade_id="PAIR-unwindshort")
        # Wait for monitor thread to fire (cancel_after_seconds=1)
        time.sleep(2.5)
        # Short should have been cancelled and an unwind market close logged
        assert "S-1" in fake_client.cancelled
        unwind_events = [e for e in executor.order_log if e["event"] == "pair_unwind_close"]
        assert len(unwind_events) == 1
        assert unwind_events[0]["symbol"] == "SPY"
        assert unwind_events[0]["side"] == "sell"  # closing the long

    def test_only_short_fills_triggers_unwind(self, fake_client, executor):
        long_pending = _FakeOrder("L-1", "SPY", "buy", 10, status="accepted")
        short_filled = _FakeOrder("S-1", "IWM", "sell", 20, status="filled", filled_avg_price=199.5)
        market_close = _FakeOrder("M-1", "IWM", "buy", 20, status="accepted", type_="market")
        fake_client.queue_submit(
            _FakeOrder("L-1", "SPY", "buy", 10, status="accepted"),
            _FakeOrder("S-1", "IWM", "sell", 20, status="accepted"),
            market_close,
        )
        fake_client.set_statuses("L-1", [long_pending, long_pending, long_pending])
        fake_client.set_statuses("S-1", [short_filled, short_filled, short_filled])
        ps = _make_pair_signal_for_executor()
        executor.submit_pair_order(ps, trade_id="PAIR-unwindlong")
        time.sleep(2.5)
        assert "L-1" in fake_client.cancelled
        unwind_events = [e for e in executor.order_log if e["event"] == "pair_unwind_close"]
        assert len(unwind_events) == 1
        assert unwind_events[0]["symbol"] == "IWM"
        assert unwind_events[0]["side"] == "buy"  # closing the short

    def test_neither_fills_cancels_both(self, fake_client, executor):
        long_pending = _FakeOrder("L-1", "SPY", "buy", 10, status="accepted")
        short_pending = _FakeOrder("S-1", "IWM", "sell", 20, status="accepted")
        fake_client.queue_submit(long_pending, short_pending)
        fake_client.set_statuses("L-1", [long_pending] * 5)
        fake_client.set_statuses("S-1", [short_pending] * 5)
        ps = _make_pair_signal_for_executor()
        executor.submit_pair_order(ps, trade_id="PAIR-neither")
        time.sleep(2.5)
        # Both orders should have been cancelled
        assert "L-1" in fake_client.cancelled
        assert "S-1" in fake_client.cancelled
        # No unwind close (nothing was filled)
        unwind_events = [e for e in executor.order_log if e["event"] == "pair_unwind_close"]
        assert unwind_events == []

    def test_submission_failure_on_one_leg_unwinds_other(self, fake_client, executor):
        """If the broker rejects one leg outright at submission, we cancel the
        accepted leg immediately rather than waiting for the monitor."""
        fake_client.queue_submit(
            _FakeOrder("L-1", "SPY", "buy", 10, status="accepted"),
        )
        # Second submit_order raises → short leg fails
        original = fake_client.submit_order

        call = {"n": 0}
        def fail_second(req):
            call["n"] += 1
            if call["n"] == 1:
                return original(req)
            raise RuntimeError("broker rejected short leg")
        fake_client.submit_order = fail_second
        # The unwind tries to market-close the long leg
        market_close_long = _FakeOrder("M-1", "SPY", "sell", 10, status="accepted", type_="market")
        # Need to keep submit_order call count, but allow the unwind market close.
        # Reroute: after the first submit fails, the unwind path also calls
        # submit_order via _place_market — make the third call succeed.
        def smart_submit(req):
            call["n"] += 1
            if call["n"] == 1:
                return original(req)
            if call["n"] == 2:
                raise RuntimeError("broker rejected short leg")
            # 3rd call: the market close from _unwind_filled_leg
            return market_close_long
        fake_client.submit_order = smart_submit
        # Reset call counter
        call["n"] = 0
        # Re-queue (since the previous submit_order pop'd one already)
        fake_client._next_orders = [
            _FakeOrder("L-1", "SPY", "buy", 10, status="accepted"),
        ]

        ps = _make_pair_signal_for_executor()
        long_res, short_res = executor.submit_pair_order(ps, trade_id="PAIR-failshort")
        assert long_res.status in (OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.SUBMITTED)
        assert short_res.status == OrderStatus.FAILED
        # Long order should have been cancelled
        assert "L-1" in fake_client.cancelled
