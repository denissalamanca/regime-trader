"""Alpaca API wrapper using the alpaca-py SDK.

Handles authentication, account queries, market clock, and historical data.
Credentials are read from environment variables (via .env) — NEVER hardcoded.

Paper vs live trading is controlled solely by the base URL:
  - Paper: https://paper-api.alpaca.markets  (DEFAULT)
  - Live:  https://api.alpaca.markets
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# URLs
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
PAPER_DATA_URL = "https://data.alpaca.markets"

# Retry config
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds


@dataclass
class AccountInfo:
    """Snapshot of the Alpaca account state."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    currency: str
    is_trading_blocked: bool
    is_pattern_day_trader: bool
    initial_margin: float
    maintenance_margin: float
    last_equity: float
    multiplier: float              # Margin multiplier (2 = 2x margin)


@dataclass
class MarginInfo:
    """Margin utilization details."""

    available_margin: float
    used_margin: float
    margin_multiplier: float
    buying_power: float
    regt_buying_power: float       # Reg-T buying power
    daytrading_buying_power: float


class AlpacaClient:
    """Manages the Alpaca API connection via the alpaca-py SDK.

    Reads credentials from environment variables:
      ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

    Parameters
    ----------
    config : dict
        Configuration dict. Keys:
        - paper_trading (bool): default True — use paper URL
        - api_key (str): override env var
        - secret_key (str): override env var
        - base_url (str): override env var
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._api_key: str = config.get("api_key", "") or os.getenv("ALPACA_API_KEY", "")
        self._secret_key: str = config.get("secret_key", "") or os.getenv("ALPACA_SECRET_KEY", "")

        # Determine base URL — paper by default
        paper = config.get("paper_trading", True)
        default_url = PAPER_URL if paper else LIVE_URL
        self._base_url: str = config.get("base_url", "") or os.getenv("ALPACA_BASE_URL", default_url)

        self._trading_client = None  # alpaca.trading.TradingClient
        self._data_client = None     # alpaca.data.StockHistoricalDataClient
        self._connected: bool = False
        self._is_paper: bool = "paper" in self._base_url

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    @property
    def trading_client(self):
        """Raw alpaca-py TradingClient for advanced operations."""
        return self._trading_client

    @property
    def data_client(self):
        """Raw alpaca-py StockHistoricalDataClient."""
        return self._data_client

    def connect(self) -> None:
        """Establish connection to Alpaca and verify credentials.

        For live trading, requires explicit user confirmation.

        Raises
        ------
        ConnectionError
            If authentication fails or API is unreachable.
        ValueError
            If credentials are missing.
        """
        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or provide in config."
            )

        # Live trading confirmation
        if not self._is_paper:
            self._confirm_live_trading()

        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.monotonic()

                self._trading_client = TradingClient(
                    api_key=self._api_key,
                    secret_key=self._secret_key,
                    paper=self._is_paper,
                )
                self._data_client = StockHistoricalDataClient(
                    api_key=self._api_key,
                    secret_key=self._secret_key,
                )

                # Health check — fetch account to verify credentials
                acct = self._trading_client.get_account()
                elapsed = time.monotonic() - t0

                self._connected = True
                mode = "PAPER" if self._is_paper else "LIVE"
                logger.info(
                    "Connected to Alpaca (%s) in %.2fs. Equity: $%s, "
                    "Buying Power: $%s, Status: %s",
                    mode, elapsed, acct.equity, acct.buying_power, acct.status,
                )
                return

            except Exception as e:
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Connection attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt, MAX_RETRIES, e, backoff,
                )
                if attempt == MAX_RETRIES:
                    raise ConnectionError(
                        f"Failed to connect to Alpaca after {MAX_RETRIES} attempts: {e}"
                    ) from e
                time.sleep(backoff)

    def disconnect(self) -> None:
        """Clean up API connections."""
        self._trading_client = None
        self._data_client = None
        self._connected = False
        logger.info("Disconnected from Alpaca.")

    def _confirm_live_trading(self) -> None:
        """Require explicit confirmation for live trading."""
        print("\n" + "=" * 60)
        print("  LIVE TRADING MODE. Real money will be at risk.")
        print("=" * 60)
        response = input("Type 'YES I UNDERSTAND THE RISKS' to confirm: ")
        if response.strip() != "YES I UNDERSTAND THE RISKS":
            raise RuntimeError("Live trading confirmation rejected. Exiting.")
        logger.warning("LIVE TRADING confirmed by user.")

    def _ensure_connected(self) -> None:
        if not self._connected or self._trading_client is None:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        """Fetch current account information."""
        self._ensure_connected()
        t0 = time.monotonic()
        acct = self._trading_client.get_account()
        elapsed = time.monotonic() - t0
        logger.debug("get_account: %.3fs", elapsed)

        return AccountInfo(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            portfolio_value=float(acct.portfolio_value),
            currency=acct.currency,
            is_trading_blocked=acct.trading_blocked,
            is_pattern_day_trader=acct.pattern_day_trader,
            initial_margin=float(acct.initial_margin),
            maintenance_margin=float(acct.maintenance_margin),
            last_equity=float(acct.last_equity),
            multiplier=float(acct.multiplier),
        )

    def get_available_margin(self) -> MarginInfo:
        """Fetch margin utilization details."""
        self._ensure_connected()
        acct = self._trading_client.get_account()
        return MarginInfo(
            available_margin=float(acct.buying_power) - float(acct.initial_margin),
            used_margin=float(acct.initial_margin),
            margin_multiplier=float(acct.multiplier),
            buying_power=float(acct.buying_power),
            regt_buying_power=float(acct.regt_buying_power),
            daytrading_buying_power=float(acct.daytrading_buying_power),
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Fetch all open positions with unrealized P&L.

        Returns
        -------
        list[dict]
            Each dict has: symbol, qty, side, avg_entry_price, current_price,
            market_value, unrealized_pl, unrealized_plpc.
        """
        self._ensure_connected()
        t0 = time.monotonic()
        raw = self._trading_client.get_all_positions()
        elapsed = time.monotonic() - t0
        logger.debug("get_positions: %.3fs, %d positions", elapsed, len(raw))

        positions = []
        for p in raw:
            positions.append({
                "symbol": p.symbol,
                "qty": int(p.qty),
                "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            })
        return positions

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def get_order_history(self, limit: int = 100) -> list[dict]:
        """Fetch recent orders with fill details.

        Parameters
        ----------
        limit : int
            Maximum number of orders to return.

        Returns
        -------
        list[dict]
            Order details including fill info.
        """
        self._ensure_connected()
        from alpaca.trading.requests import GetOrdersRequest

        t0 = time.monotonic()
        request = GetOrdersRequest(status="all", limit=limit)
        raw = self._trading_client.get_orders(filter=request)
        elapsed = time.monotonic() - t0
        logger.debug("get_order_history: %.3fs, %d orders", elapsed, len(raw))

        orders = []
        for o in raw:
            orders.append({
                "id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                "qty": str(o.qty),
                "filled_qty": str(o.filled_qty),
                "type": o.type.value if hasattr(o.type, "value") else str(o.type),
                "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else None,
                "submitted_at": str(o.submitted_at) if o.submitted_at else None,
                "filled_at": str(o.filled_at) if o.filled_at else None,
                "created_at": str(o.created_at) if o.created_at else None,
            })
        return orders

    # ------------------------------------------------------------------
    # Market clock
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Check whether the market is currently open."""
        self._ensure_connected()
        clock = self._trading_client.get_clock()
        return clock.is_open

    def get_clock(self) -> dict:
        """Get market clock info.

        Returns
        -------
        dict
            Keys: is_open, next_open, next_close, timestamp.
        """
        self._ensure_connected()
        t0 = time.monotonic()
        clock = self._trading_client.get_clock()
        elapsed = time.monotonic() - t0
        logger.debug("get_clock: %.3fs", elapsed)

        return {
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
            "timestamp": str(clock.timestamp),
        }

    # ------------------------------------------------------------------
    # Historical bars
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 504,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars via alpaca-py data client.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str
            Bar timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day".
        limit : int
            Maximum bars to return (used when start/end not specified).
        start : str, optional
            Start date (YYYY-MM-DD). If None, computed from limit.
        end : str, optional
            End date (YYYY-MM-DD). Defaults to today.

        Returns
        -------
        pd.DataFrame
            OHLCV data with DatetimeIndex. Columns: open, high, low, close, volume.
        """
        self._ensure_connected()
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min") if hasattr(TimeFrame, "__init__") else TimeFrame.Minute,
            "15Min": TimeFrame(15, "Min") if hasattr(TimeFrame, "__init__") else TimeFrame.Minute,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        if start is None:
            # Estimate start date from limit
            days_back = limit * 2 if timeframe == "1Day" else limit
            start_dt = datetime.now() - timedelta(days=days_back)
            start = start_dt.strftime("%Y-%m-%d")

        t0 = time.monotonic()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )
        bars = self._data_client.get_stock_bars(request)
        elapsed = time.monotonic() - t0

        # Convert to DataFrame
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            # Multi-symbol response: select the symbol level
            df = df.xs(symbol, level="symbol")

        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        })
        # Ensure DatetimeIndex
        if df.index.tz is not None:
            df.index = df.index.tz_convert("US/Eastern").tz_localize(None)

        # Keep only OHLCV columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        logger.debug(
            "get_bars(%s, %s): %.3fs, %d bars from %s to %s",
            symbol, timeframe, elapsed, len(df),
            df.index[0] if len(df) > 0 else "N/A",
            df.index[-1] if len(df) > 0 else "N/A",
        )
        return df

    # ------------------------------------------------------------------
    # Quotes
    # ------------------------------------------------------------------

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest quote for a symbol.

        Returns
        -------
        dict
            Keys: bid, ask, mid, spread, spread_pct, timestamp.
        """
        self._ensure_connected()
        from alpaca.data.requests import StockLatestQuoteRequest

        t0 = time.monotonic()
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self._data_client.get_stock_latest_quote(request)
        elapsed = time.monotonic() - t0

        if isinstance(quote, dict):
            q = quote.get(symbol, quote)
        else:
            q = quote

        bid = float(q.bid_price) if hasattr(q, "bid_price") else 0.0
        ask = float(q.ask_price) if hasattr(q, "ask_price") else 0.0
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)
        spread = ask - bid
        spread_pct = spread / mid if mid > 0 else 0.0

        logger.debug(
            "get_latest_quote(%s): %.3fs, bid=%.2f, ask=%.2f, spread=%.4f%%",
            symbol, elapsed, bid, ask, spread_pct * 100,
        )
        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "spread_pct": spread_pct,
            "timestamp": str(q.timestamp) if hasattr(q, "timestamp") else None,
        }

    def get_snapshot(self, symbol: str) -> dict:
        """Get combined quote + latest bar + latest trade info.

        Returns
        -------
        dict
            Keys: quote (dict), latest_trade_price (float), daily_bar (dict).
        """
        self._ensure_connected()
        from alpaca.data.requests import StockSnapshotRequest

        t0 = time.monotonic()
        request = StockSnapshotRequest(symbol_or_symbols=symbol)
        snap = self._data_client.get_stock_snapshot(request)
        elapsed = time.monotonic() - t0

        if isinstance(snap, dict):
            s = snap.get(symbol, snap)
        else:
            s = snap

        result = {
            "latest_trade_price": float(s.latest_trade.price) if s.latest_trade else None,
            "quote": {
                "bid": float(s.latest_quote.bid_price) if s.latest_quote else None,
                "ask": float(s.latest_quote.ask_price) if s.latest_quote else None,
            },
            "daily_bar": {
                "open": float(s.daily_bar.open) if s.daily_bar else None,
                "high": float(s.daily_bar.high) if s.daily_bar else None,
                "low": float(s.daily_bar.low) if s.daily_bar else None,
                "close": float(s.daily_bar.close) if s.daily_bar else None,
                "volume": float(s.daily_bar.volume) if s.daily_bar else None,
            },
        }
        logger.debug("get_snapshot(%s): %.3fs", symbol, elapsed)
        return result
