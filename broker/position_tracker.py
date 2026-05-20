"""Track open positions, unrealized P&L, and sync with Alpaca.

Responsibilities:
- Real-time position tracking via Alpaca WebSocket trade updates
- Per-position metrics: entry time, holding period, regime at entry, distance to stop
- Continuous unrealized P&L from latest prices
- Startup reconciliation: sync our state with Alpaca's actual positions
- Fill processing: update PortfolioState and CircuitBreaker on every fill
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Thread, Lock
from typing import Optional, Callable

import pandas as pd

from .alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single open position with detailed tracking."""

    symbol: str
    qty: int
    side: str                          # "long" or "short"
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    # Extended tracking
    entry_time: Optional[pd.Timestamp] = None
    current_stop_price: Optional[float] = None
    distance_to_stop_pct: Optional[float] = None
    holding_period_bars: int = 0
    regime_at_entry: str = ""
    current_regime: str = ""
    trade_id: str = ""


@dataclass
class PositionSnapshot:
    """Complete view of all positions at a point in time."""

    positions: list[Position]
    total_market_value: float
    total_unrealized_pnl: float
    timestamp: pd.Timestamp


class PositionTracker:
    """Tracks live positions, P&L, and provides portfolio state.

    On startup, reconciles internal state with Alpaca's actual positions.
    Subscribes to WebSocket trade updates for instant fill notifications.

    Parameters
    ----------
    client : AlpacaClient
        Connected Alpaca client.
    config : dict
        Optional configuration.
    """

    def __init__(self, client: AlpacaClient, config: dict = None) -> None:
        self._client = client
        self._config = config or {}
        self._positions: dict[str, Position] = {}
        self._position_history: list[PositionSnapshot] = []
        self._lock = Lock()
        self._fill_callbacks: list[Callable] = []
        self._ws_thread: Optional[Thread] = None
        self._ws_running: bool = False
        self._last_sync: Optional[pd.Timestamp] = None

    @property
    def positions(self) -> dict[str, Position]:
        with self._lock:
            return dict(self._positions)

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def sync_with_broker(self) -> PositionSnapshot:
        """Reconcile our tracked positions with Alpaca's actual positions.

        Call this on startup and periodically to ensure consistency.

        Returns
        -------
        PositionSnapshot
            Current position state after sync.
        """
        self._client._ensure_connected()
        raw = self._client.get_positions()

        with self._lock:
            broker_symbols = set()
            for p in raw:
                sym = p["symbol"]
                broker_symbols.add(sym)

                if sym in self._positions:
                    # Update existing
                    pos = self._positions[sym]
                    pos.qty = p["qty"]
                    pos.avg_entry_price = p["avg_entry_price"]
                    pos.current_price = p["current_price"]
                    pos.market_value = p["market_value"]
                    pos.unrealized_pnl = p["unrealized_pl"]
                    pos.unrealized_pnl_pct = p["unrealized_plpc"]
                else:
                    # New position we didn't know about
                    side = "long" if p["qty"] > 0 else "short"
                    self._positions[sym] = Position(
                        symbol=sym,
                        qty=abs(p["qty"]),
                        side=side,
                        avg_entry_price=p["avg_entry_price"],
                        current_price=p["current_price"],
                        market_value=p["market_value"],
                        unrealized_pnl=p["unrealized_pl"],
                        unrealized_pnl_pct=p["unrealized_plpc"],
                    )
                    logger.info("Discovered position during sync: %s %d %s @ $%.2f",
                                sym, abs(p["qty"]), side, p["avg_entry_price"])

            # Remove positions that Alpaca doesn't have (closed externally)
            for sym in list(self._positions):
                if sym not in broker_symbols:
                    logger.info("Position %s closed externally, removing from tracker", sym)
                    del self._positions[sym]

        self._last_sync = pd.Timestamp.now()
        snapshot = self._take_snapshot()
        logger.info(
            "Position sync complete: %d positions, total value $%.2f, "
            "unrealized P&L $%.2f",
            len(snapshot.positions), snapshot.total_market_value,
            snapshot.total_unrealized_pnl,
        )
        return snapshot

    # ------------------------------------------------------------------
    # Real-time updates via WebSocket
    # ------------------------------------------------------------------

    def start_streaming(self) -> None:
        """Subscribe to Alpaca's WebSocket trade updates stream."""
        if self._ws_running:
            return

        self._ws_running = True
        self._ws_thread = Thread(
            target=self._run_trade_stream, daemon=True,
            name="position-tracker-ws",
        )
        self._ws_thread.start()
        logger.info("Position tracker WebSocket stream started.")

    def stop_streaming(self) -> None:
        """Stop the WebSocket stream."""
        self._ws_running = False
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)
        logger.info("Position tracker WebSocket stream stopped.")

    def _run_trade_stream(self) -> None:
        """WebSocket listener for trade update events."""
        try:
            from alpaca.trading.stream import TradingStream

            stream = TradingStream(
                api_key=self._client._api_key,
                secret_key=self._client._secret_key,
                paper=self._client.is_paper,
            )

            @stream.on("trade_updates")
            async def on_trade_update(data):
                self._handle_trade_update(data)

            # This blocks until the stream is closed
            while self._ws_running:
                try:
                    stream.run()
                except Exception as e:
                    if self._ws_running:
                        logger.warning("Trade stream disconnected: %s. Reconnecting in 5s...", e)
                        time.sleep(5)

        except ImportError:
            logger.warning(
                "alpaca TradingStream not available. "
                "Position updates will rely on polling via sync_with_broker()."
            )
        except Exception as e:
            logger.error("Trade stream fatal error: %s", e)

    def _handle_trade_update(self, data) -> None:
        """Process a single trade update event from WebSocket."""
        try:
            event = data.event if hasattr(data, "event") else str(data.get("event", ""))
            order = data.order if hasattr(data, "order") else data.get("order", {})

            symbol = order.symbol if hasattr(order, "symbol") else order.get("symbol", "")
            if not symbol:
                return

            if event == "fill":
                filled_price = float(order.filled_avg_price) if hasattr(order, "filled_avg_price") else 0
                filled_qty = int(order.filled_qty) if hasattr(order, "filled_qty") else 0
                side = order.side if hasattr(order, "side") else order.get("side", "")
                side_str = side.value if hasattr(side, "value") else str(side)

                logger.info(
                    "FILL: %s %s %d @ $%.2f",
                    symbol, side_str, filled_qty, filled_price,
                )

                # Refresh position from broker to get accurate state
                self.sync_with_broker()

                # Notify callbacks
                for cb in self._fill_callbacks:
                    try:
                        cb(symbol, side_str, filled_qty, filled_price)
                    except Exception as e:
                        logger.error("Fill callback error: %s", e)

            elif event in ("canceled", "expired", "rejected"):
                logger.info("Order %s for %s: %s", event, symbol,
                            order.id if hasattr(order, "id") else "")

        except Exception as e:
            logger.error("Error handling trade update: %s", e)

    def on_fill(self, callback: Callable) -> None:
        """Register a callback for fill events.

        Callback signature: (symbol, side, qty, fill_price) -> None
        """
        self._fill_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def refresh(self) -> PositionSnapshot:
        """Fetch current positions from Alpaca and update internal state."""
        return self.sync_with_broker()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get the current position for a specific symbol."""
        with self._lock:
            return self._positions.get(symbol)

    def get_position_values(self) -> dict[str, float]:
        """Get symbol -> market value map for all open positions."""
        with self._lock:
            return {sym: pos.market_value for sym, pos in self._positions.items()}

    def get_position_count(self) -> int:
        with self._lock:
            return len(self._positions)

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def record_entry(
        self,
        symbol: str,
        qty: int,
        side: str,
        entry_price: float,
        stop_price: Optional[float],
        regime_label: str,
        trade_id: str = "",
    ) -> None:
        """Record a new position entry (called when our order fills)."""
        with self._lock:
            self._positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                side=side,
                avg_entry_price=entry_price,
                current_price=entry_price,
                market_value=qty * entry_price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                entry_time=pd.Timestamp.now(),
                current_stop_price=stop_price,
                distance_to_stop_pct=(
                    abs(entry_price - stop_price) / entry_price
                    if stop_price and entry_price > 0 else None
                ),
                regime_at_entry=regime_label,
                current_regime=regime_label,
                trade_id=trade_id,
            )
        logger.info(
            "Position opened: %s %s %d @ $%.2f, stop=$%.2f, regime=%s, id=%s",
            side, symbol, qty, entry_price, stop_price or 0, regime_label, trade_id,
        )

    def update_prices(self, latest_prices: dict[str, float]) -> None:
        """Update current prices and recalculate P&L for all positions."""
        with self._lock:
            for sym, pos in self._positions.items():
                if sym in latest_prices:
                    price = latest_prices[sym]
                    pos.current_price = price
                    pos.market_value = abs(pos.qty) * price

                    if pos.side == "long":
                        pos.unrealized_pnl = (price - pos.avg_entry_price) * abs(pos.qty)
                    else:
                        pos.unrealized_pnl = (pos.avg_entry_price - price) * abs(pos.qty)

                    if pos.avg_entry_price > 0:
                        pos.unrealized_pnl_pct = pos.unrealized_pnl / (pos.avg_entry_price * abs(pos.qty))

                    if pos.current_stop_price and price > 0:
                        pos.distance_to_stop_pct = abs(price - pos.current_stop_price) / price

    def update_regime(self, regime_label: str) -> None:
        """Update the current regime for all positions."""
        with self._lock:
            for pos in self._positions.values():
                pos.current_regime = regime_label

    def increment_holding_period(self) -> None:
        """Increment holding period bar count for all positions. Call on each bar."""
        with self._lock:
            for pos in self._positions.values():
                pos.holding_period_bars += 1

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close_position(self, symbol: str) -> None:
        """Submit a market order to close a position."""
        self._client._ensure_connected()
        try:
            self._client.trading_client.close_position(symbol)
            logger.info("Close position submitted: %s", symbol)
        except Exception as e:
            logger.error("Failed to close position %s: %s", symbol, e)

    def close_all(self) -> None:
        """Close all open positions."""
        self._client._ensure_connected()
        try:
            self._client.trading_client.close_all_positions(cancel_orders=True)
            with self._lock:
                self._positions.clear()
            logger.warning("All positions closed.")
        except Exception as e:
            logger.error("Failed to close all positions: %s", e)

    # ------------------------------------------------------------------
    # History and snapshots
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> PositionSnapshot:
        with self._lock:
            pos_list = list(self._positions.values())
        total_val = sum(abs(p.market_value) for p in pos_list)
        total_pnl = sum(p.unrealized_pnl for p in pos_list)
        snapshot = PositionSnapshot(
            positions=pos_list,
            total_market_value=total_val,
            total_unrealized_pnl=total_pnl,
            timestamp=pd.Timestamp.now(),
        )
        self._position_history.append(snapshot)
        return snapshot

    def get_history(self) -> pd.DataFrame:
        """Return position snapshot history as a DataFrame."""
        if not self._position_history:
            return pd.DataFrame()
        records = []
        for snap in self._position_history:
            records.append({
                "timestamp": snap.timestamp,
                "position_count": len(snap.positions),
                "total_market_value": snap.total_market_value,
                "total_unrealized_pnl": snap.total_unrealized_pnl,
            })
        return pd.DataFrame(records).set_index("timestamp")

    def get_data_feed_health(self) -> dict:
        """Report data feed and sync health."""
        return {
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "positions_tracked": len(self._positions),
            "ws_streaming": self._ws_running,
            "fill_callbacks_registered": len(self._fill_callbacks),
        }
