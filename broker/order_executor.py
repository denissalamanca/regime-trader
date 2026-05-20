"""Order placement, modification, and cancellation via Alpaca.

Every order goes through here. The executor:
- Uses LIMIT orders by default (not market) to control slippage
- Supports bracket orders (entry + stop + take-profit as OCO)
- Tracks every order with a unique trade_id linking signal → risk → order → fill
- Logs EVERY submission, fill, cancellation, and rejection
- Enforces stop-only-tightens (never widen a stop)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Thread
from typing import Optional

import pandas as pd

from .alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    NEW = "new"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class OrderResult:
    """Result of an order submission or query."""

    order_id: str
    trade_id: str                  # Links signal → risk → order → fill
    symbol: str
    side: str
    qty: int
    order_type: str
    status: OrderStatus
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_qty: int = 0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    error_message: Optional[str] = None
    slippage: Optional[float] = None  # Filled price vs intended price
    metadata: dict = field(default_factory=dict)


def _new_trade_id() -> str:
    """Generate a unique trade ID for order chain tracking."""
    return f"T-{uuid.uuid4().hex[:12]}"


def _map_status(alpaca_status: str) -> OrderStatus:
    """Map Alpaca order status string to our enum."""
    mapping = {
        "new": OrderStatus.NEW,
        "accepted": OrderStatus.ACCEPTED,
        "pending_new": OrderStatus.PENDING,
        "partially_filled": OrderStatus.PARTIAL,
        "filled": OrderStatus.FILLED,
        "done_for_day": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.EXPIRED,
        "replaced": OrderStatus.SUBMITTED,
        "rejected": OrderStatus.REJECTED,
        "pending_cancel": OrderStatus.CANCELLED,
        "pending_replace": OrderStatus.SUBMITTED,
    }
    return mapping.get(alpaca_status, OrderStatus.FAILED)


class OrderExecutor:
    """Submits and manages orders through Alpaca's API.

    Parameters
    ----------
    client : AlpacaClient
        Connected Alpaca client.
    config : dict
        Execution configuration from settings.yaml under 'execution'.
    """

    def __init__(self, client: AlpacaClient, config: dict) -> None:
        self._client = client
        self._config = config

        # Config
        self._limit_offset_pct: float = config.get("limit_offset_pct", 0.001)
        self._cancel_after_s: int = config.get("cancel_after_seconds", 30)
        self._retry_attempts: int = config.get("retry_attempts", 1)
        self._retry_delay: float = config.get("retry_delay_seconds", 2.0)
        self._chase_at_market: bool = config.get("chase_at_market", False)
        self._time_in_force: str = config.get("time_in_force", "day")

        # Tracking
        self._pending_orders: dict[str, OrderResult] = {}  # order_id -> result
        self._order_log: list[dict] = []

    @property
    def order_log(self) -> list[dict]:
        return list(self._order_log)

    # ------------------------------------------------------------------
    # Submit limit order
    # ------------------------------------------------------------------

    def submit_order(
        self,
        signal,
        trade_id: Optional[str] = None,
    ) -> OrderResult:
        """Submit a limit order from a validated signal.

        Uses limit price = entry_price +/- limit_offset_pct depending on side.
        If not filled within cancel_after_seconds, cancels and optionally
        retries at market (if chase_at_market is True).

        Parameters
        ----------
        signal : Signal
            Validated signal with entry_price, stop_loss, direction, etc.
            Must have metadata["risk_sized_qty"] from the risk manager.
        trade_id : str, optional
            Override trade ID. Generated if not provided.

        Returns
        -------
        OrderResult
        """
        from core.regime_strategies import SignalDirection

        tid = trade_id or _new_trade_id()
        qty = signal.metadata.get("risk_sized_qty", 0)
        if qty <= 0:
            return self._fail_result(tid, signal.symbol, "Zero quantity", signal)

        side = "buy" if signal.direction == SignalDirection.LONG else "sell"
        # Limit price: slightly aggressive to improve fill probability
        offset = signal.entry_price * self._limit_offset_pct
        if side == "buy":
            limit_price = round(signal.entry_price + offset, 2)
        else:
            limit_price = round(signal.entry_price - offset, 2)

        self._log_event("submit", tid, signal.symbol, side, qty,
                        limit_price=limit_price, signal_price=signal.entry_price)

        result = self._place_limit(tid, signal.symbol, side, qty, limit_price)

        if result.status in (OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.SUBMITTED):
            self._pending_orders[result.order_id] = result
            # Monitor for fill in background
            self._monitor_fill(result, signal)

        return result

    # ------------------------------------------------------------------
    # Submit pair order (two coordinated legs)
    # ------------------------------------------------------------------

    def submit_pair_order(
        self,
        pair_signal,
        trade_id: Optional[str] = None,
    ) -> tuple[OrderResult, OrderResult]:
        """Submit two coordinated leg orders sharing a pair_id.

        Both legs go out as limit orders simultaneously. Within
        ``cancel_after_seconds`` we monitor both fills:
          - If both fill → success (logged at INFO).
          - If neither fills → both cancelled at the deadline (no exposure).
          - If exactly one fills → cancel the unfilled leg, then **market-close
            the filled leg** to avoid carrying naked directional risk. Logged
            at WARNING with a ``PAIR_UNWOUND`` prefix so flaky pairs are loud.

        The pair_id (``PAIR-<uuid>``) is shared between both ``OrderResult``
        records' ``trade_id`` fields so downstream tracking can correlate them.
        """
        from core.regime_strategies import SignalDirection

        pair_id = trade_id or f"PAIR-{uuid.uuid4().hex[:12]}"
        long = pair_signal.long_leg
        short = pair_signal.short_leg

        long_qty = long.metadata.get("risk_sized_qty", 0)
        short_qty = short.metadata.get("risk_sized_qty", 0)
        if long_qty <= 0 or short_qty <= 0:
            failure = self._fail_result(
                pair_id, f"{long.symbol}/{short.symbol}",
                f"Pair leg qty<=0 (long={long_qty}, short={short_qty})",
            )
            return failure, failure

        # Limit prices: aggressive offset on each side (matches submit_order).
        long_offset = long.entry_price * self._limit_offset_pct
        short_offset = short.entry_price * self._limit_offset_pct
        long_limit = round(long.entry_price + long_offset, 2)
        short_limit = round(short.entry_price - short_offset, 2)

        self._log_event(
            "submit_pair", pair_id, f"{long.symbol}/{short.symbol}",
            "buy/sell", long_qty + short_qty,
            long_limit=long_limit, short_limit=short_limit,
            z_score=getattr(pair_signal, "z_score", None),
        )

        long_result = self._place_limit(pair_id, long.symbol, "buy", long_qty, long_limit)
        short_result = self._place_limit(pair_id, short.symbol, "sell", short_qty, short_limit)

        # If either leg failed at submission (broker rejected), unwind the
        # surviving submission. We can't have one leg accepted while the
        # other was outright refused.
        long_accepted = long_result.status in (
            OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL, OrderStatus.FILLED,
        )
        short_accepted = short_result.status in (
            OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL, OrderStatus.FILLED,
        )
        if long_accepted and not short_accepted:
            logger.warning(
                "PAIR_UNWIND pair_id=%s: short %s submission failed (%s); "
                "cancelling long %s and closing position if filled.",
                pair_id, short.symbol, short_result.error_message or short_result.status.value,
                long.symbol,
            )
            self.cancel_order(long_result.order_id)
            self._unwind_filled_leg(pair_id, long.symbol, "buy", long_qty)
            return long_result, short_result
        if short_accepted and not long_accepted:
            logger.warning(
                "PAIR_UNWIND pair_id=%s: long %s submission failed (%s); "
                "cancelling short %s and closing position if filled.",
                pair_id, long.symbol, long_result.error_message or long_result.status.value,
                short.symbol,
            )
            self.cancel_order(short_result.order_id)
            self._unwind_filled_leg(pair_id, short.symbol, "sell", short_qty)
            return long_result, short_result
        if not (long_accepted or short_accepted):
            logger.warning(
                "PAIR_UNWIND pair_id=%s: both legs failed at submission. "
                "long=%s, short=%s",
                pair_id, long_result.error_message, short_result.error_message,
            )
            return long_result, short_result

        # Both legs are live — track them and start the fill-coordination
        # watcher that handles partial-fill atomicity.
        if long_result.order_id:
            self._pending_orders[long_result.order_id] = long_result
        if short_result.order_id:
            self._pending_orders[short_result.order_id] = short_result
        self._monitor_pair_fills(long_result, short_result, long, short, pair_id)

        return long_result, short_result

    def _monitor_pair_fills(
        self, long_result: "OrderResult", short_result: "OrderResult",
        long_signal, short_signal, pair_id: str,
    ) -> None:
        """Background watcher: enforce both-or-neither pair fill semantics.

        Polls both leg orders until ``cancel_after_seconds`` elapses. If only
        one leg fills, cancels the unfilled leg and market-closes the filled
        one to avoid carrying naked directional exposure. Logs every unwind
        loudly so users notice flaky pairs.
        """
        def _watch():
            deadline = time.monotonic() + self._cancel_after_s
            long_filled = False
            short_filled = False
            while time.monotonic() < deadline:
                time.sleep(2)
                try:
                    if not long_filled and long_result.order_id:
                        s = self.get_order_status(long_result.order_id)
                        if s.status == OrderStatus.FILLED:
                            long_filled = True
                            self._log_event(
                                "pair_leg_filled", pair_id, long_signal.symbol,
                                "buy", long_result.qty,
                                fill_price=s.filled_avg_price,
                                signal_price=long_signal.entry_price,
                            )
                    if not short_filled and short_result.order_id:
                        s = self.get_order_status(short_result.order_id)
                        if s.status == OrderStatus.FILLED:
                            short_filled = True
                            self._log_event(
                                "pair_leg_filled", pair_id, short_signal.symbol,
                                "sell", short_result.qty,
                                fill_price=s.filled_avg_price,
                                signal_price=short_signal.entry_price,
                            )
                except Exception:
                    pass
                if long_filled and short_filled:
                    self._pending_orders.pop(long_result.order_id, None)
                    self._pending_orders.pop(short_result.order_id, None)
                    logger.info(
                        "PAIR_FILLED pair_id=%s: both legs filled (%s long, %s short)",
                        pair_id, long_signal.symbol, short_signal.symbol,
                    )
                    return

            # Deadline reached. Determine outcome.
            if long_filled and not short_filled:
                logger.warning(
                    "PAIR_UNWIND pair_id=%s: long %s filled but short %s did not. "
                    "Cancelling short and market-closing long. "
                    "Frequent unwinds suggest a liquidity-asymmetric pair.",
                    pair_id, long_signal.symbol, short_signal.symbol,
                )
                if short_result.order_id:
                    self.cancel_order(short_result.order_id)
                self._unwind_filled_leg(pair_id, long_signal.symbol, "buy", long_result.qty)
            elif short_filled and not long_filled:
                logger.warning(
                    "PAIR_UNWIND pair_id=%s: short %s filled but long %s did not. "
                    "Cancelling long and market-closing short. "
                    "Frequent unwinds suggest a liquidity-asymmetric pair.",
                    pair_id, short_signal.symbol, long_signal.symbol,
                )
                if long_result.order_id:
                    self.cancel_order(long_result.order_id)
                self._unwind_filled_leg(pair_id, short_signal.symbol, "sell", short_result.qty)
            else:
                # Neither leg filled — clean cancel both.
                logger.info(
                    "PAIR_DROPPED pair_id=%s: neither leg filled within %ds; cancelling both.",
                    pair_id, self._cancel_after_s,
                )
                if long_result.order_id:
                    self.cancel_order(long_result.order_id)
                if short_result.order_id:
                    self.cancel_order(short_result.order_id)

            self._pending_orders.pop(long_result.order_id, None)
            self._pending_orders.pop(short_result.order_id, None)

        thread = Thread(
            target=_watch, daemon=True,
            name=f"pair-monitor-{pair_id[:12]}",
        )
        thread.start()

    def _unwind_filled_leg(
        self, pair_id: str, symbol: str, original_side: str, qty: int,
    ) -> Optional[OrderResult]:
        """Market-close a filled leg whose pair partner failed to fill.

        ``original_side`` is the side that originally OPENED the position. We
        send the OPPOSITE side to close it.
        """
        close_side = "sell" if original_side == "buy" else "buy"
        try:
            result = self._place_market(pair_id, symbol, close_side, qty)
            self._log_event(
                "pair_unwind_close", pair_id, symbol, close_side, qty,
                reason="partner leg unfilled",
            )
            return result
        except Exception as e:
            logger.error(
                "PAIR_UNWIND pair_id=%s: failed to market-close %s: %s. "
                "MANUAL INTERVENTION REQUIRED.",
                pair_id, symbol, e,
            )
            return None

    # ------------------------------------------------------------------
    # Submit bracket order (entry + stop + take-profit)
    # ------------------------------------------------------------------

    def submit_bracket_order(
        self,
        signal,
        trade_id: Optional[str] = None,
    ) -> OrderResult:
        """Submit an entry that ALWAYS carries a protective stop at the broker.

        With a take-profit this is a full Alpaca BRACKET (entry + stop + target).
        Without one it is an OTO ("one-triggers-other") so the entry still brings
        a resting stop-loss. A bare entry with no broker-side stop is never used
        for live trading — closing that gap is the point of this method (A1).

        Parameters
        ----------
        signal : Signal
            Must have entry_price and stop_loss; take_profit is optional.
        trade_id : str, optional

        Returns
        -------
        OrderResult
        """
        from alpaca.trading.requests import (
            LimitOrderRequest, OrderClass, TakeProfitRequest, StopLossRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce
        from core.regime_strategies import SignalDirection

        tid = trade_id or _new_trade_id()
        qty = signal.metadata.get("risk_sized_qty", 0)
        if qty <= 0:
            return self._fail_result(tid, signal.symbol, "Zero quantity", signal)

        side = OrderSide.BUY if signal.direction == SignalDirection.LONG else OrderSide.SELL
        offset = signal.entry_price * self._limit_offset_pct
        limit_price = round(
            signal.entry_price + (offset if side == OrderSide.BUY else -offset), 2)

        stop_price = round(signal.stop_loss, 2)

        # BRACKET when a target exists, otherwise OTO (entry-triggers-stop) so a
        # protective stop always rests at the broker even with no take-profit.
        has_target = signal.take_profit is not None
        order_class = OrderClass.BRACKET if has_target else OrderClass.OTO

        kwargs = {
            "symbol": signal.symbol,
            "qty": qty,
            "side": side,
            "type": "limit",
            "time_in_force": TimeInForce.DAY,
            "limit_price": limit_price,
            "order_class": order_class,
            "stop_loss": StopLossRequest(stop_price=stop_price),
        }
        if has_target:
            kwargs["take_profit"] = TakeProfitRequest(
                limit_price=round(signal.take_profit, 2))

        self._log_event("submit_bracket", tid, signal.symbol,
                        side.value, qty,
                        order_class=order_class.value,
                        limit_price=limit_price,
                        stop_price=stop_price,
                        take_profit=signal.take_profit)

        try:
            self._client._ensure_connected()
            t0 = time.monotonic()
            order_req = LimitOrderRequest(**kwargs)
            order = self._client.trading_client.submit_order(order_req)
            elapsed = time.monotonic() - t0

            status_str = order.status.value if hasattr(order.status, "value") else str(order.status)
            result = OrderResult(
                order_id=str(order.id),
                trade_id=tid,
                symbol=signal.symbol,
                side=side.value,
                qty=qty,
                order_type="bracket" if has_target else "oto",
                status=_map_status(status_str),
                limit_price=limit_price,
                stop_price=stop_price,
            )

            self._log_event("bracket_accepted", tid, signal.symbol,
                            side.value, qty, order_id=str(order.id),
                            elapsed=elapsed)
            self._pending_orders[result.order_id] = result
            return result

        except Exception as e:
            logger.error("Bracket order failed for %s: %s", signal.symbol, e)
            return self._fail_result(tid, signal.symbol, str(e), signal)

    # ------------------------------------------------------------------
    # Modify stop (only tighten, never widen)
    # ------------------------------------------------------------------

    def modify_stop(
        self,
        order_id: str,
        new_stop_price: float,
        current_side: str = "buy",
    ) -> OrderResult:
        """Modify a stop order price. Only moves stop in the favorable direction.

        For longs: stop can only move UP. For shorts: stop can only move DOWN.

        Parameters
        ----------
        order_id : str
            The stop order ID to modify.
        new_stop_price : float
            New stop price.
        current_side : str
            "buy" for long positions, "sell" for short positions.

        Returns
        -------
        OrderResult
        """
        self._client._ensure_connected()

        try:
            existing = self._client.trading_client.get_order_by_id(order_id)
            old_stop = float(existing.stop_price) if existing.stop_price else 0

            # Enforce stop-only-tightens
            if current_side == "buy" and new_stop_price < old_stop:
                logger.warning(
                    "Refusing to WIDEN stop for long: %.2f → %.2f (order %s)",
                    old_stop, new_stop_price, order_id)
                return OrderResult(
                    order_id=order_id, trade_id="", symbol=existing.symbol,
                    side="sell", qty=int(existing.qty), order_type="stop",
                    status=OrderStatus.REJECTED,
                    error_message=f"Cannot widen stop: {old_stop:.2f} → {new_stop_price:.2f}",
                )
            if current_side == "sell" and new_stop_price > old_stop:
                logger.warning(
                    "Refusing to WIDEN stop for short: %.2f → %.2f (order %s)",
                    old_stop, new_stop_price, order_id)
                return OrderResult(
                    order_id=order_id, trade_id="", symbol=existing.symbol,
                    side="buy", qty=int(existing.qty), order_type="stop",
                    status=OrderStatus.REJECTED,
                    error_message=f"Cannot widen stop: {old_stop:.2f} → {new_stop_price:.2f}",
                )

            self._log_event("modify_stop", "", existing.symbol, current_side,
                            int(existing.qty),
                            old_stop=old_stop, new_stop=new_stop_price)

            from alpaca.trading.requests import ReplaceOrderRequest
            replace_req = ReplaceOrderRequest(
                stop_price=round(new_stop_price, 2),
            )
            replaced = self._client.trading_client.replace_order_by_id(
                order_id, replace_req)

            status_str = replaced.status.value if hasattr(replaced.status, "value") else str(replaced.status)
            return OrderResult(
                order_id=str(replaced.id),
                trade_id="",
                symbol=existing.symbol,
                side=current_side,
                qty=int(existing.qty),
                order_type="stop",
                status=_map_status(status_str),
                stop_price=new_stop_price,
            )

        except Exception as e:
            logger.error("modify_stop failed for order %s: %s", order_id, e)
            return OrderResult(
                order_id=order_id, trade_id="", symbol="", side=current_side,
                qty=0, order_type="stop", status=OrderStatus.FAILED,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel a pending order."""
        self._client._ensure_connected()
        try:
            self._client.trading_client.cancel_order_by_id(order_id)
            self._pending_orders.pop(order_id, None)

            self._log_event("cancel", "", "", "", 0, order_id=order_id)
            return OrderResult(
                order_id=order_id, trade_id="", symbol="", side="",
                qty=0, order_type="", status=OrderStatus.CANCELLED,
            )
        except Exception as e:
            logger.error("cancel_order failed for %s: %s", order_id, e)
            return OrderResult(
                order_id=order_id, trade_id="", symbol="", side="",
                qty=0, order_type="", status=OrderStatus.FAILED,
                error_message=str(e),
            )

    def cancel_all(self) -> list[OrderResult]:
        """Cancel all open orders."""
        self._client._ensure_connected()
        results = []
        try:
            self._client.trading_client.cancel_orders()
            for oid in list(self._pending_orders):
                results.append(OrderResult(
                    order_id=oid, trade_id="", symbol="", side="",
                    qty=0, order_type="", status=OrderStatus.CANCELLED,
                ))
            self._pending_orders.clear()
            self._log_event("cancel_all", "", "", "", 0,
                            count=len(results))
        except Exception as e:
            logger.error("cancel_all failed: %s", e)
        return results

    # ------------------------------------------------------------------
    # Close positions
    # ------------------------------------------------------------------

    def close_position(self, symbol: str) -> OrderResult:
        """Close a single position at market."""
        self._client._ensure_connected()
        tid = _new_trade_id()
        try:
            self._client.trading_client.close_position(symbol)
            self._log_event("close_position", tid, symbol, "market_close", 0)
            return OrderResult(
                order_id="", trade_id=tid, symbol=symbol, side="close",
                qty=0, order_type="market", status=OrderStatus.SUBMITTED,
            )
        except Exception as e:
            logger.error("close_position(%s) failed: %s", symbol, e)
            return self._fail_result(tid, symbol, str(e))

    def close_all_positions(self) -> list[OrderResult]:
        """Emergency exit: close ALL positions at market."""
        self._client._ensure_connected()
        results = []
        logger.warning("EMERGENCY CLOSE ALL POSITIONS")
        try:
            self._client.trading_client.close_all_positions(cancel_orders=True)
            self._pending_orders.clear()
            self._log_event("emergency_close_all", "", "", "", 0)
            results.append(OrderResult(
                order_id="", trade_id=_new_trade_id(), symbol="ALL",
                side="close", qty=0, order_type="market",
                status=OrderStatus.SUBMITTED,
            ))
        except Exception as e:
            logger.error("close_all_positions failed: %s", e)
        return results

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_order_status(self, order_id: str) -> OrderResult:
        """Query the current status of an order."""
        self._client._ensure_connected()
        try:
            order = self._client.trading_client.get_order_by_id(order_id)
            status_str = order.status.value if hasattr(order.status, "value") else str(order.status)
            filled_avg = float(order.filled_avg_price) if order.filled_avg_price else None

            return OrderResult(
                order_id=str(order.id),
                trade_id=self._pending_orders.get(str(order.id), OrderResult(
                    order_id="", trade_id="", symbol="", side="", qty=0,
                    order_type="", status=OrderStatus.PENDING)).trade_id,
                symbol=order.symbol,
                side=order.side.value if hasattr(order.side, "value") else str(order.side),
                qty=int(order.qty),
                order_type=order.type.value if hasattr(order.type, "value") else str(order.type),
                status=_map_status(status_str),
                filled_qty=int(order.filled_qty) if order.filled_qty else 0,
                filled_avg_price=filled_avg,
                submitted_at=str(order.submitted_at) if order.submitted_at else None,
                filled_at=str(order.filled_at) if order.filled_at else None,
            )
        except Exception as e:
            return OrderResult(
                order_id=order_id, trade_id="", symbol="", side="",
                qty=0, order_type="", status=OrderStatus.FAILED,
                error_message=str(e),
            )

    def get_open_stop_order(self, symbol: str) -> Optional[dict]:
        """Return the open protective stop for ``symbol`` as a dict, or None.

        Finds the live stop placed by an OTO/bracket entry by querying the
        broker (the source of truth, so it survives restarts and external
        changes). Used by trailing-stop management to locate the order to
        tighten.

        Returns
        -------
        dict or None
            ``{"order_id": str, "stop_price": float, "side": str}`` for the
            first open stop / stop-limit order on the symbol, else None.
        """
        self._client._ensure_connected()
        from alpaca.trading.requests import GetOrdersRequest
        try:
            req = GetOrdersRequest(status="open", symbols=[symbol], nested=False)
            orders = self._client.trading_client.get_orders(filter=req)
        except Exception as e:
            logger.warning("get_open_stop_order(%s) failed: %s", symbol, e)
            return None

        for o in orders:
            otype = o.type.value if hasattr(o.type, "value") else str(o.type)
            if "stop" in otype and getattr(o, "stop_price", None):
                side = o.side.value if hasattr(o.side, "value") else str(o.side)
                return {
                    "order_id": str(o.id),
                    "stop_price": float(o.stop_price),
                    "side": side,
                }
        return None

    def sweep_stale_orders(self) -> list[OrderResult]:
        """Cancel orders pending longer than cancel_after_seconds."""
        if self._cancel_after_s <= 0:
            return []

        stale = []
        now = time.monotonic()
        for oid, result in list(self._pending_orders.items()):
            submitted = result.metadata.get("submit_time", now)
            if now - submitted > self._cancel_after_s:
                cancelled = self.cancel_order(oid)
                stale.append(cancelled)
                logger.info("Swept stale order %s (%s) after %ds",
                            oid, result.symbol, self._cancel_after_s)
        return stale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_limit(self, trade_id: str, symbol: str, side: str,
                     qty: int, limit_price: float) -> OrderResult:
        """Place a single limit order via alpaca-py."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        tif = tif_map.get(self._time_in_force, TimeInForce.DAY)

        try:
            self._client._ensure_connected()
            t0 = time.monotonic()
            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=tif,
                limit_price=round(limit_price, 2),
            )
            order = self._client.trading_client.submit_order(order_req)
            elapsed = time.monotonic() - t0

            status_str = order.status.value if hasattr(order.status, "value") else str(order.status)
            result = OrderResult(
                order_id=str(order.id),
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="limit",
                status=_map_status(status_str),
                limit_price=limit_price,
                submitted_at=str(order.submitted_at) if order.submitted_at else None,
                metadata={"submit_time": time.monotonic()},
            )

            self._log_event("order_accepted", trade_id, symbol, side, qty,
                            order_id=str(order.id), elapsed=elapsed)
            return result

        except Exception as e:
            logger.error("Limit order failed for %s: %s", symbol, e)
            return self._fail_result(trade_id, symbol, str(e))

    def _monitor_fill(self, result: OrderResult, signal) -> None:
        """Start a background thread to monitor order fill status.

        If unfilled after cancel_after_seconds, cancels and optionally retries
        at market.
        """
        def _watch():
            deadline = time.monotonic() + self._cancel_after_s
            while time.monotonic() < deadline:
                time.sleep(2)
                try:
                    status = self.get_order_status(result.order_id)
                    if status.status == OrderStatus.FILLED:
                        slippage = 0.0
                        if status.filled_avg_price and signal.entry_price > 0:
                            slippage = (status.filled_avg_price - signal.entry_price) / signal.entry_price
                        self._log_event("filled", result.trade_id, result.symbol,
                                        result.side, result.qty,
                                        fill_price=status.filled_avg_price,
                                        signal_price=signal.entry_price,
                                        slippage=slippage)
                        self._pending_orders.pop(result.order_id, None)
                        return
                    if status.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED,
                                         OrderStatus.EXPIRED):
                        self._pending_orders.pop(result.order_id, None)
                        return
                except Exception:
                    pass

            # Not filled — cancel
            logger.info("Order %s not filled after %ds, cancelling",
                        result.order_id, self._cancel_after_s)
            self.cancel_order(result.order_id)

            # Optionally chase at market
            if self._chase_at_market and signal.metadata.get("allow_chase", True):
                logger.info("Chasing %s at market for trade %s",
                            result.symbol, result.trade_id)
                self._place_market(result.trade_id, result.symbol,
                                   result.side, result.qty)

        thread = Thread(target=_watch, daemon=True,
                        name=f"fill-monitor-{result.order_id[:8]}")
        thread.start()

    def _place_market(self, trade_id: str, symbol: str, side: str, qty: int) -> OrderResult:
        """Place a market order (used for chase-at-market fallback)."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        try:
            self._client._ensure_connected()
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.trading_client.submit_order(order_req)
            status_str = order.status.value if hasattr(order.status, "value") else str(order.status)

            self._log_event("market_chase", trade_id, symbol, side, qty,
                            order_id=str(order.id))
            return OrderResult(
                order_id=str(order.id), trade_id=trade_id, symbol=symbol,
                side=side, qty=qty, order_type="market",
                status=_map_status(status_str),
            )
        except Exception as e:
            logger.error("Market chase failed for %s: %s", symbol, e)
            return self._fail_result(trade_id, symbol, str(e))

    def _fail_result(self, trade_id: str, symbol: str, reason: str,
                     signal=None) -> OrderResult:
        self._log_event("failed", trade_id, symbol, "", 0, error=reason)
        return OrderResult(
            order_id="", trade_id=trade_id, symbol=symbol, side="",
            qty=0, order_type="", status=OrderStatus.FAILED,
            error_message=reason,
        )

    def _log_event(self, event: str, trade_id: str, symbol: str,
                   side: str, qty: int, **kwargs) -> None:
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "event": event,
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            **kwargs,
        }
        self._order_log.append(record)
        logger.info("ORDER %s: %s %s %d %s | %s",
                     event.upper(), symbol, side, qty, trade_id,
                     " ".join(f"{k}={v}" for k, v in kwargs.items()))
