"""Real-time and historical market data fetching via Alpaca.

Provides a unified interface for:
- Historical bars (for HMM training and backtesting)
- Real-time bars (via WebSocket subscription)
- Real-time quotes (for spread checking before order placement)
- Rolling DataFrame buffering with automatic feature computation
- Data feed health monitoring

Handles market data gaps (weekends, holidays, halts) gracefully.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Thread, Lock
from typing import Callable, Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk cache for historical bars
# ---------------------------------------------------------------------------
#
# Backtests are run repeatedly and shouldn't need a network round-trip every
# time. The cache stores one CSV per (symbol, timeframe) pair. It's keyed
# only by symbol+timeframe, NOT by date range — when a slice of a cached
# series is requested, we filter in-memory.
#
# Cache misses (no file) trigger an Alpaca fetch and write the result to disk.
# Cache hits are silent. On hit, we still slice by start/end/limit before
# returning so the caller gets exactly the window they asked for.

DEFAULT_CACHE_DIR = Path("data_cache")


def _cache_path(cache_dir: Path, symbol: str, timeframe: str) -> Path:
    """Resolve the on-disk cache file for a (symbol, timeframe) pair."""
    return cache_dir / f"{symbol.upper()}_{timeframe}.csv"


def _load_from_cache(path: Path) -> Optional[pd.DataFrame]:
    """Load a cached bars CSV. Returns None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        logger.warning("Cache file %s unreadable (%s); will refetch.", path, e)
        return None


def _save_to_cache(path: Path, bars: pd.DataFrame) -> None:
    """Write bars to disk. Creates parent dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(path)


def _slice_bars(
    bars: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    limit: Optional[int],
) -> pd.DataFrame:
    """Filter a cached bars DataFrame by start/end/limit (callers' window)."""
    out = bars
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.index <= pd.Timestamp(end)]
    if limit is not None and len(out) > limit:
        out = out.iloc[-limit:]
    return out


class MarketDataClient:
    """Fetches and caches market data from Alpaca.

    Maintains a rolling cache of bars per symbol and supports real-time
    streaming updates via WebSocket.

    Parameters
    ----------
    client : AlpacaClient
        Connected Alpaca client.
    config : dict
        Universe configuration from settings.yaml under 'universe'.
    """

    def __init__(self, client: AlpacaClient, config: dict) -> None:
        self._client = client
        self._config = config

        self._symbols: list[str] = config.get("symbols", [])
        self._timeframe: str = config.get("timeframe", "1Day")
        self._lookback: int = config.get("lookback_bars", 504)

        # Disk cache for historical bars. Default: data_cache/ at project root.
        # Disable by setting cache_dir to None or "" (passes through to Alpaca every time).
        cache_dir_cfg = config.get("cache_dir", str(DEFAULT_CACHE_DIR))
        self._cache_dir: Optional[Path] = Path(cache_dir_cfg) if cache_dir_cfg else None

        self._bar_cache: dict[str, pd.DataFrame] = {}
        self._quote_cache: dict[str, dict] = {}
        self._lock = Lock()

        # Streaming
        self._bar_callbacks: list[Callable] = []
        self._quote_callbacks: list[Callable] = []
        self._ws_thread: Optional[Thread] = None
        self._ws_running: bool = False

        # Health monitoring
        self._last_bar_time: dict[str, pd.Timestamp] = {}
        self._last_quote_time: dict[str, pd.Timestamp] = {}
        self._gap_count: int = 0

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch historical bars for a single symbol with range-aware caching.

        Resolution order:
          1. Load `data_cache/<SYMBOL>_<timeframe>.csv` if present.
          2. If the cache covers the requested [start, end] window, slice and
             return. Silent on hit.
          3. If the cache exists but the request extends beyond it (start
             earlier than first cached bar, OR end later than last cached
             bar), refetch the union range from Alpaca, merge into the
             cache (Alpaca's values win on overlapping dates), persist, and
             return the requested slice. Logs INFO on refetch.
          4. If the cache doesn't exist, fetch the requested range, persist,
             and return. Logs INFO on miss.

        Set `cache_dir` to None in the config to bypass the disk cache entirely.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str, optional
            Override default timeframe.
        start : str, optional
            Start date (YYYY-MM-DD).
        end : str, optional
            End date (YYYY-MM-DD).
        limit : int, optional
            Max bars to return (applied AFTER start/end filtering).

        Returns
        -------
        pd.DataFrame
            OHLCV data with DatetimeIndex.
        """
        tf = timeframe or self._timeframe
        lim = limit or self._lookback

        cache_file: Optional[Path] = None
        cached: Optional[pd.DataFrame] = None
        if self._cache_dir is not None:
            cache_file = _cache_path(self._cache_dir, symbol, tf)
            cached = _load_from_cache(cache_file)

        bars: Optional[pd.DataFrame] = None

        if cached is not None and len(cached) > 0:
            # We have something on disk — decide if it covers the request.
            # Treat None bounds as "no extension demanded" against the cache.
            cache_first = cached.index[0]
            cache_last = cached.index[-1]
            req_start = pd.Timestamp(start) if start else cache_first
            req_end = pd.Timestamp(end) if end else cache_last

            extends_earlier = req_start < cache_first
            extends_later = req_end > cache_last

            if not extends_earlier and not extends_later:
                # Pure cache hit — slice and return silently
                bars = _slice_bars(cached, start, end, lim)
            else:
                # Cache exists but request extends beyond it — refetch the
                # union range, then merge with what we already have.
                why = []
                if extends_earlier:
                    why.append(f"start {req_start.date()} < cached {cache_first.date()}")
                if extends_later:
                    why.append(f"end {req_end.date()} > cached {cache_last.date()}")
                logger.info(
                    "Cache extension for %s %s — refetching: %s",
                    symbol, tf, "; ".join(why),
                )

                # Fetch the broader window directly from Alpaca. If this
                # fails (no creds, network down, API error), fall back to
                # whatever the cache already holds — better to serve stale
                # data than to crash the backtest.
                union_start = min(req_start, cache_first).strftime("%Y-%m-%d")
                union_end = max(req_end, cache_last).strftime("%Y-%m-%d")
                try:
                    fetched = self._client.get_bars(
                        symbol=symbol, timeframe=tf, limit=10_000,
                        start=union_start, end=union_end,
                    )
                except Exception as e:
                    logger.warning(
                        "Refetch failed for %s (%s); serving cached data only.",
                        symbol, e,
                    )
                    fetched = None

                if fetched is not None and len(fetched) > 0:
                    # Merge: cache + fetched, prefer fetched on overlap.
                    # `keep="last"` after concat([cache, fetched]) means
                    # fetched (the second frame) wins for duplicate dates.
                    merged = pd.concat([cached, fetched])
                    merged = merged[~merged.index.duplicated(keep="last")]
                    merged = merged.sort_index()

                    if cache_file is not None:
                        try:
                            _save_to_cache(cache_file, merged)
                            logger.info(
                                "Cache updated: %d bars in %s (was %d)",
                                len(merged), cache_file, len(cached),
                            )
                        except Exception as e:
                            logger.warning("Failed to write cache %s: %s", cache_file, e)

                    bars = _slice_bars(merged, start, end, lim)
                else:
                    # Refetch failed — fall back to whatever the cache had
                    logger.warning(
                        "Refetch returned no data for %s; serving stale cache.",
                        symbol,
                    )
                    bars = _slice_bars(cached, start, end, lim)

        if bars is None:
            # No cache file at all — full miss
            logger.info(
                "Cache miss for %s %s — fetching from Alpaca", symbol, tf,
            )
            fetched = self._client.get_bars(
                symbol=symbol, timeframe=tf, limit=lim,
                start=start, end=end,
            )
            bars = fetched

            if cache_file is not None and fetched is not None and len(fetched) > 0:
                try:
                    _save_to_cache(cache_file, fetched)
                    logger.info(
                        "Cached %d bars to %s", len(fetched), cache_file,
                    )
                except Exception as e:
                    logger.warning("Failed to write cache %s: %s", cache_file, e)

        # Update in-memory cache
        with self._lock:
            self._bar_cache[symbol] = bars
            if bars is not None and len(bars) > 0:
                self._last_bar_time[symbol] = bars.index[-1]

        return bars

    def fetch_bars(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch bars (alias for get_historical_bars for backward compatibility)."""
        return self.get_historical_bars(symbol, timeframe=timeframe, limit=limit)

    def fetch_all_bars(self) -> dict[str, pd.DataFrame]:
        """Fetch bars for all symbols in the universe.

        Returns
        -------
        dict[str, pd.DataFrame]
            OHLCV data keyed by symbol.
        """
        result = {}
        for sym in self._symbols:
            try:
                result[sym] = self.get_historical_bars(sym)
                logger.info("Fetched %d bars for %s", len(result[sym]), sym)
            except Exception as e:
                logger.error("Failed to fetch bars for %s: %s", sym, e)
        return result

    # ------------------------------------------------------------------
    # Real-time quotes
    # ------------------------------------------------------------------

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Get the most recent OHLCV bar from cache.

        Returns
        -------
        pd.Series or None
            Most recent bar, or None if no data cached.
        """
        with self._lock:
            if symbol in self._bar_cache and len(self._bar_cache[symbol]) > 0:
                return self._bar_cache[symbol].iloc[-1]
        return None

    def get_latest_quote(self, symbol: str) -> dict:
        """Get the current bid/ask/last for a symbol.

        Fetches from Alpaca if not in cache.

        Returns
        -------
        dict
            Keys: bid, ask, mid, spread, spread_pct, timestamp.
        """
        quote = self._client.get_latest_quote(symbol)
        with self._lock:
            self._quote_cache[symbol] = quote
            self._last_quote_time[symbol] = pd.Timestamp.now()
        return quote

    def get_latest_prices(self) -> dict[str, float]:
        """Get latest mid price for all symbols in the universe.

        Returns
        -------
        dict[str, float]
            Mid prices keyed by symbol.
        """
        prices = {}
        for sym in self._symbols:
            try:
                quote = self.get_latest_quote(sym)
                prices[sym] = quote["mid"]
            except Exception as e:
                logger.warning("Failed to get price for %s: %s", sym, e)
                # Fall back to last cached bar close
                with self._lock:
                    if sym in self._bar_cache and len(self._bar_cache[sym]) > 0:
                        prices[sym] = float(self._bar_cache[sym]["close"].iloc[-1])
        return prices

    def get_snapshot(self, symbol: str) -> dict:
        """Get combined quote + bar + trade info for a symbol."""
        return self._client.get_snapshot(symbol)

    # ------------------------------------------------------------------
    # Real-time streaming
    # ------------------------------------------------------------------

    def subscribe_bars(self, callback: Callable[[str, pd.Series], None]) -> None:
        """Register a callback for real-time bar updates.

        Parameters
        ----------
        callback : callable
            Called with (symbol, bar_series) on each new bar.
        """
        self._bar_callbacks.append(callback)
        if not self._ws_running:
            self._start_data_stream()

    def subscribe_quotes(self, callback: Callable[[str, dict], None]) -> None:
        """Register a callback for real-time quote updates.

        Parameters
        ----------
        callback : callable
            Called with (symbol, quote_dict) on each new quote.
        """
        self._quote_callbacks.append(callback)
        if not self._ws_running:
            self._start_data_stream()

    def unsubscribe(self) -> None:
        """Disconnect from real-time data streams."""
        self._ws_running = False
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)
        self._bar_callbacks.clear()
        self._quote_callbacks.clear()
        logger.info("Unsubscribed from all data streams.")

    def _start_data_stream(self) -> None:
        """Start the WebSocket data stream in a background thread."""
        if self._ws_running:
            return
        self._ws_running = True
        self._ws_thread = Thread(
            target=self._run_data_stream, daemon=True,
            name="market-data-ws",
        )
        self._ws_thread.start()
        logger.info("Market data WebSocket stream started for %s", self._symbols)

    def _run_data_stream(self) -> None:
        """WebSocket listener for market data."""
        try:
            from alpaca.data.live import StockDataStream

            stream = StockDataStream(
                api_key=self._client._api_key,
                secret_key=self._client._secret_key,
            )

            @stream.on_bar(*self._symbols)
            async def on_bar(bar):
                self._handle_bar(bar)

            @stream.on_quote(*self._symbols)
            async def on_quote(quote):
                self._handle_quote(quote)

            while self._ws_running:
                try:
                    stream.run()
                except Exception as e:
                    if self._ws_running:
                        logger.warning("Data stream disconnected: %s. Reconnecting in 5s...", e)
                        time.sleep(5)

        except ImportError:
            logger.warning(
                "alpaca StockDataStream not available. "
                "Real-time updates will rely on polling."
            )
        except Exception as e:
            logger.error("Data stream fatal error: %s", e)

    def _handle_bar(self, bar) -> None:
        """Process an incoming bar from the WebSocket."""
        try:
            symbol = bar.symbol
            bar_data = pd.Series({
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }, name=bar.timestamp)

            # Update cache
            self.update_cache_single(symbol, bar_data)

            # Notify callbacks
            for cb in self._bar_callbacks:
                try:
                    cb(symbol, bar_data)
                except Exception as e:
                    logger.error("Bar callback error for %s: %s", symbol, e)

        except Exception as e:
            logger.error("Error handling bar update: %s", e)

    def _handle_quote(self, quote) -> None:
        """Process an incoming quote from the WebSocket."""
        try:
            symbol = quote.symbol
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)

            quote_dict = {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": ask - bid,
                "spread_pct": (ask - bid) / mid if mid > 0 else 0,
                "timestamp": str(quote.timestamp),
            }

            with self._lock:
                self._quote_cache[symbol] = quote_dict
                self._last_quote_time[symbol] = pd.Timestamp.now()

            for cb in self._quote_callbacks:
                try:
                    cb(symbol, quote_dict)
                except Exception as e:
                    logger.error("Quote callback error for %s: %s", symbol, e)

        except Exception as e:
            logger.error("Error handling quote update: %s", e)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def update_cache(self, symbol: str, new_bars: pd.DataFrame) -> None:
        """Append new bars to the local cache."""
        with self._lock:
            if symbol in self._bar_cache:
                existing = self._bar_cache[symbol]
                combined = pd.concat([existing, new_bars])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                # Keep only the most recent lookback bars
                self._bar_cache[symbol] = combined.iloc[-self._lookback:]
            else:
                self._bar_cache[symbol] = new_bars.iloc[-self._lookback:]

            if len(self._bar_cache[symbol]) > 0:
                self._last_bar_time[symbol] = self._bar_cache[symbol].index[-1]

    def update_cache_single(self, symbol: str, bar: pd.Series) -> None:
        """Append a single bar (pd.Series) to the cache."""
        bar_df = bar.to_frame().T
        if hasattr(bar, "name") and bar.name is not None:
            bar_df.index = [bar.name]
        self.update_cache(symbol, bar_df)

    def get_cached_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached bars for a symbol without fetching."""
        with self._lock:
            return self._bar_cache.get(symbol)

    # ------------------------------------------------------------------
    # Data health monitoring
    # ------------------------------------------------------------------

    def get_data_health(self) -> dict:
        """Report data feed health status.

        Returns
        -------
        dict
            Keys: symbols_streaming, last_bar_times, last_quote_times,
            gap_count, ws_running.
        """
        with self._lock:
            bar_times = {s: t.isoformat() for s, t in self._last_bar_time.items()}
            quote_times = {s: t.isoformat() for s, t in self._last_quote_time.items()}
            cached_sizes = {s: len(df) for s, df in self._bar_cache.items()}

        now = pd.Timestamp.now()
        stale_symbols = []
        for sym, t in self._last_bar_time.items():
            if (now - t).total_seconds() > 86400 * 3:  # 3 days stale
                stale_symbols.append(sym)

        if stale_symbols:
            logger.warning("Stale data detected for: %s", stale_symbols)

        return {
            "symbols_configured": self._symbols,
            "symbols_cached": list(self._bar_cache.keys()),
            "cached_bar_counts": cached_sizes,
            "last_bar_times": bar_times,
            "last_quote_times": quote_times,
            "stale_symbols": stale_symbols,
            "gap_count": self._gap_count,
            "ws_running": self._ws_running,
            "bar_callbacks": len(self._bar_callbacks),
            "quote_callbacks": len(self._quote_callbacks),
        }
