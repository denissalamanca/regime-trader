"""Structured logging with context injection, rotating files, and rich console.

Log files:
  logs/main.log     — everything (rotating, 10MB, 30 files)
  logs/trades.log   — order submissions, fills, cancellations only
  logs/alerts.log   — WARNING and above only
  logs/regime.log   — regime changes and HMM retraining only

Every log entry is optionally enriched with trading context (regime,
equity, positions, circuit breaker) via a TradingContextFilter.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 30               # Keep 30 rotated files

CONSOLE_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
JSON_FORMAT_KEYS = [
    "timestamp", "level", "logger", "message",
    "regime", "regime_prob", "equity", "positions",
    "daily_pnl_pct", "circuit_breaker",
]
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Trading context — injected into every log record
# ---------------------------------------------------------------------------

@dataclass
class TradingContext:
    """Mutable trading state attached to log records."""

    regime: str = ""
    regime_probability: float = 0.0
    equity: float = 0.0
    open_positions: int = 0
    daily_pnl_pct: float = 0.0
    circuit_breaker: str = "none"


# Singleton context updated by the main loop
_context = TradingContext()


def update_log_context(
    regime: str = "",
    regime_probability: float = 0.0,
    equity: float = 0.0,
    open_positions: int = 0,
    daily_pnl_pct: float = 0.0,
    circuit_breaker: str = "none",
) -> None:
    """Update the global trading context injected into log records."""
    if regime:
        _context.regime = regime
    if regime_probability:
        _context.regime_probability = regime_probability
    if equity:
        _context.equity = equity
    _context.open_positions = open_positions
    _context.daily_pnl_pct = daily_pnl_pct
    if circuit_breaker:
        _context.circuit_breaker = circuit_breaker


class TradingContextFilter(logging.Filter):
    """Injects trading context fields into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.regime = _context.regime
        record.regime_prob = _context.regime_probability
        record.equity = _context.equity
        record.positions = _context.open_positions
        record.daily_pnl_pct = _context.daily_pnl_pct
        record.cb_status = _context.circuit_breaker
        return True


# ---------------------------------------------------------------------------
# Filters for specialized log files
# ---------------------------------------------------------------------------

class TradeEventFilter(logging.Filter):
    """Only pass log records related to order/trade events."""

    _KEYWORDS = {"ORDER", "FILL", "submit", "cancel", "close_position",
                 "bracket", "modify_stop", "TRADE REJECTED"}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return any(kw in msg for kw in self._KEYWORDS)


class RegimeEventFilter(logging.Filter):
    """Only pass log records related to regime changes and HMM retraining."""

    _KEYWORDS = {"REGIME", "regime", "HMM", "Regime", "refit", "BIC",
                 "n_regimes", "trained", "model selection"}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return any(kw in msg for kw in self._KEYWORDS)


# ---------------------------------------------------------------------------
# JSON formatter for structured file logging
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON with trading context."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "regime": getattr(record, "regime", ""),
            "regime_prob": getattr(record, "regime_prob", 0),
            "equity": getattr(record, "equity", 0),
            "positions": getattr(record, "positions", 0),
            "daily_pnl_pct": getattr(record, "daily_pnl_pct", 0),
            "circuit_breaker": getattr(record, "cb_status", "none"),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Rich console handler (colored, human-readable)
# ---------------------------------------------------------------------------

def _make_rich_handler(level: int) -> logging.Handler:
    """Create a Rich-based console handler with colors.

    Falls back to a plain StreamHandler if rich is not installed.
    """
    try:
        from rich.logging import RichHandler
        handler = RichHandler(
            level=level,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        return handler
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
        return handler


# ---------------------------------------------------------------------------
# Rotating file handler helper
# ---------------------------------------------------------------------------

def _make_rotating_handler(
    path: str,
    level: int = logging.DEBUG,
    use_json: bool = True,
    extra_filter: Optional[logging.Filter] = None,
) -> logging.Handler:
    """Create a rotating file handler."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        p, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
    )
    handler.setLevel(level)

    if use_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))

    if extra_filter:
        handler.addFilter(extra_filter)

    return handler


# ---------------------------------------------------------------------------
# Main setup function
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "regime_trader",
    log_file: Optional[str] = None,
    level: str = "INFO",
    log_dir: str = "logs",
    json_files: bool = True,
) -> logging.Logger:
    """Configure the full logging stack.

    Sets up:
    - Rich console handler (colored, human-readable)
    - main.log: rotating, all levels, JSON structured
    - trades.log: rotating, trade events only
    - alerts.log: rotating, WARNING+ only
    - regime.log: rotating, regime/HMM events only
    - Trading context filter on all handlers

    Parameters
    ----------
    name : str
        Root logger name.
    log_file : str, optional
        Override path for main log file.
    level : str
        Minimum log level.
    log_dir : str
        Directory for all log files.
    json_files : bool
        Use JSON format for file logs.

    Returns
    -------
    logging.Logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Root captures all; handlers filter
    root.handlers.clear()

    ctx_filter = TradingContextFilter()

    # --- Console: rich colored output ---
    console = _make_rich_handler(log_level)
    console.addFilter(ctx_filter)
    root.addHandler(console)

    # --- Main log file ---
    main_path = log_file or f"{log_dir}/main.log"
    main_handler = _make_rotating_handler(main_path, logging.DEBUG, json_files)
    main_handler.addFilter(ctx_filter)
    root.addHandler(main_handler)

    # --- Trades log ---
    trades_handler = _make_rotating_handler(
        f"{log_dir}/trades.log", logging.DEBUG, json_files, TradeEventFilter())
    trades_handler.addFilter(ctx_filter)
    root.addHandler(trades_handler)

    # --- Alerts log (WARNING+) ---
    alerts_handler = _make_rotating_handler(
        f"{log_dir}/alerts.log", logging.WARNING, json_files)
    alerts_handler.addFilter(ctx_filter)
    root.addHandler(alerts_handler)

    # --- Regime log ---
    regime_handler = _make_rotating_handler(
        f"{log_dir}/regime.log", logging.DEBUG, json_files, RegimeEventFilter())
    regime_handler.addFilter(ctx_filter)
    root.addHandler(regime_handler)

    result = logging.getLogger(name)
    result.info("Logger initialized: level=%s, log_dir=%s, json=%s", level, log_dir, json_files)
    return result
