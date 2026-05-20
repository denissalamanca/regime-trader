"""Regime-adaptive allocation strategy.

DESIGN: The HMM detects VOLATILITY REGIMES, not bull/bear markets.
The strategy uses this to its advantage:

  - Low-vol regime  → be fully invested (calm markets trend up)
  - Mid-vol regime  → reduce exposure (uncertain, preserve capital)
  - High-vol regime → go short (high vol usually means selling/crisis)

This is simple but effective because:
  1. Stocks go up ~70% of the time in low-vol environments
  2. The worst drawdowns happen during high-vol spikes
  3. Avoiding the big drawdowns is worth more than catching every rally

The strategy scales to any number of regimes (3-7) by normalizing each
regime's position in the sorted order to a [-1, +1] allocation scale.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .hmm_engine import RegimeInfo, RegimeState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """Complete trade signal with regime context."""

    symbol: str
    direction: SignalDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size_pct: float
    leverage: float
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: pd.Timestamp
    reasoning: str
    strategy_name: str
    metadata: dict = field(default_factory=dict)


RawSignal = Signal


@dataclass
class PairSignal:
    """Two coordinated leg Signals that must execute (or be rejected) together.

    Used by pair-trading strategies that take a long position in one symbol
    and a short position in another, sized to be dollar-neutral via a hedge
    ratio computed from the lookback window.

    Both legs share a logical lifecycle: enter together, exit together. The
    risk manager validates them as a unit (both pass or neither does), and
    the order executor submits them with linked trade_ids and unwinds the
    filled leg if the other one doesn't fill within the cancel window.
    """

    pair: tuple[str, str]              # (symbol_a, symbol_b) — config order
    long_leg: Signal                   # always direction == LONG
    short_leg: Signal                  # always direction == SHORT
    spread_value: float                # current spread (ratio or log-spread)
    z_score: float                     # spread's z-score over lookback
    hedge_ratio: float                 # short_qty_value / long_qty_value
    correlation: Optional[float] = None  # optional, for correlation-gated strategies
    reasoning: str = ""
    timestamp: Optional[pd.Timestamp] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    strategy_name: str = "base"
    # Set True on subclasses that produce PairSignals via generate_pair_signal.
    # Single-asset strategies leave this False and implement generate_signal.
    is_pair_strategy: bool = False

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        self._config = config
        self._regime_info = regime_info
        # Optional: cached snapshot of all watchlist bars for the current bar.
        # Strategies that need cross-symbol info (e.g. momentum rotation, pair
        # spreads) read self._universe_bars after the orchestrator calls
        # set_universe_bars(). Single-asset strategies can ignore it.
        self._universe_bars: dict[str, pd.DataFrame] = {}

    @abstractmethod
    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState,
    ) -> Optional[Signal]:
        """Per-symbol signal generation. Implemented by every strategy.

        Pair strategies should return None here and override generate_pair_signal.
        """

    def generate_pair_signal(
        self,
        pair: tuple[str, str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
    ) -> Optional[PairSignal]:
        """Per-pair signal generation. Override in pair strategies.

        Default returns None — single-asset strategies don't generate pair signals.
        """
        return None

    def set_universe_bars(self, bars: dict[str, pd.DataFrame]) -> None:
        """Store a snapshot of all watchlist bars for the current bar.

        Called by the orchestrator before the per-symbol generate_signal loop.
        Strategies that need cross-symbol data (momentum rankings, spread
        calculations) read self._universe_bars inside generate_signal /
        generate_pair_signal.

        Default: store and ignore. Override only if you need to do work
        once per bar instead of once per symbol.
        """
        self._universe_bars = bars

    def _make_signal(
        self, symbol: str, direction: SignalDirection, entry_price: float,
        stop_loss: float, regime_state: RegimeState, reasoning: str,
        take_profit: Optional[float] = None, position_size_pct: Optional[float] = None,
        leverage: Optional[float] = None, confidence_boost: float = 0.0,
        **extra_metadata,
    ) -> Signal:
        ri = self._regime_info
        conf = min(1.0, regime_state.probability + confidence_boost)
        return Signal(
            symbol=symbol, direction=direction, confidence=conf,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            position_size_pct=position_size_pct or ri.max_position_size_pct,
            leverage=leverage or ri.max_leverage_allowed,
            regime_id=regime_state.state_id, regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=regime_state.timestamp, reasoning=reasoning,
            strategy_name=self.strategy_name, metadata=extra_metadata,
        )


# ---------------------------------------------------------------------------
# Regime-sorted allocation strategies
# ---------------------------------------------------------------------------

class LowVolBullStrategy(BaseStrategy):
    """Low-volatility regime → go long with large allocation.

    Calm markets trend upward. Be fully invested (95%), unleveraged, with a
    stop based on the 50 EMA as floor.

    NOTE: leverage is intentionally 1.0x (A4). The previous 1.25x request was
    effectively dead — the risk manager only permits >1.0x leverage in the
    NEUTRAL/STRONG_BULL regimes, which this low-vol archetype rarely maps to,
    so it was silently clamped. Leveraging the calmest regimes also runs
    counter to the system's drawdown-first philosophy. To run leveraged, raise
    ``risk.max_leverage`` in config deliberately rather than here.
    """
    strategy_name = "low_vol_bull"

    def generate_signal(self, symbol: str, bars: pd.DataFrame,
                        regime_state: RegimeState) -> Optional[Signal]:
        if len(bars) < 60:
            return None
        close = bars["close"]
        price = float(close.iloc[-1])
        atr = float(_compute_atr(bars).iloc[-1])
        ema50 = float(_ema(close, 50).iloc[-1])

        stop = max(price - 3.0 * atr, ema50 - 0.5 * atr)

        return self._make_signal(
            symbol, SignalDirection.LONG, price, stop, regime_state,
            f"Low-vol regime ({regime_state.label}): 95% long (1.0x), "
            f"stop at max(3ATR, 50EMA)={stop:.2f}",
            position_size_pct=0.95,
            leverage=1.0,
        )


class MidVolCautiousStrategy(BaseStrategy):
    """Mid-volatility regime → small long or flat.

    Uncertain environment. Stay invested only if trend structure is intact
    (price above 50 EMA). Otherwise go flat.
    """
    strategy_name = "mid_vol_cautious"

    def generate_signal(self, symbol: str, bars: pd.DataFrame,
                        regime_state: RegimeState) -> Optional[Signal]:
        if len(bars) < 60:
            return None
        close = bars["close"]
        price = float(close.iloc[-1])
        atr = float(_compute_atr(bars).iloc[-1])
        ema50 = float(_ema(close, 50).iloc[-1])

        if price > ema50:
            stop = ema50 - 0.5 * atr
            return self._make_signal(
                symbol, SignalDirection.LONG, price, stop, regime_state,
                f"Mid-vol regime ({regime_state.label}): 95% long, "
                f"price above 50 EMA ({ema50:.2f})",
                position_size_pct=0.95,
                leverage=1.0,
            )
        else:
            stop = price - 2.0 * atr
            return self._make_signal(
                symbol, SignalDirection.LONG, price, stop, regime_state,
                f"Mid-vol regime ({regime_state.label}): reduced 60% long, "
                f"price below 50 EMA ({ema50:.2f})",
                position_size_pct=0.60,
                leverage=1.0,
            )


class HighVolDefensiveStrategy(BaseStrategy):
    """High-volatility regime → reduce but stay invested (60%).

    High vol is dangerous, but going fully to cash means missing
    V-shaped recoveries. Stay 60% invested to capture rebounds
    while limiting drawdown exposure.
    """
    strategy_name = "high_vol_defensive"

    def generate_signal(self, symbol: str, bars: pd.DataFrame,
                        regime_state: RegimeState) -> Optional[Signal]:
        if len(bars) < 60:
            return None
        close = bars["close"]
        price = float(close.iloc[-1])
        atr = float(_compute_atr(bars).iloc[-1])
        stop = price - 2.0 * atr

        return self._make_signal(
            symbol, SignalDirection.LONG, price, stop, regime_state,
            f"High-vol regime ({regime_state.label}): reduced 60% long, preserving capital",
            position_size_pct=0.60,
            leverage=1.0,
        )


# ---------------------------------------------------------------------------
# Strategy mapping by volatility rank
# ---------------------------------------------------------------------------

# For any number of regimes, we map each sorted position to an archetype:
# Bottom third (low vol) → LowVolBullStrategy
# Middle third → MidVolCautiousStrategy
# Top third (high vol) → HighVolDefensiveStrategy

def _get_strategy_for_vol_rank(rank: int, n_regimes: int) -> type[BaseStrategy]:
    """Map a regime's volatility rank to a strategy class.

    rank 0 = lowest vol, rank n-1 = highest vol.
    Bottom third → long, middle → cautious, top third → short.
    """
    if n_regimes <= 1:
        return MidVolCautiousStrategy

    position = rank / (n_regimes - 1)  # 0.0 to 1.0

    if position <= 0.33:
        return LowVolBullStrategy
    elif position >= 0.67:
        return HighVolDefensiveStrategy
    else:
        return MidVolCautiousStrategy


# Also keep the old label mapping for backward compat with tests
LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "CRASH":       HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "BEAR":        HighVolDefensiveStrategy,
    "WEAK_BEAR":   MidVolCautiousStrategy,
    "NEUTRAL":     MidVolCautiousStrategy,
    "WEAK_BULL":   MidVolCautiousStrategy,
    "BULL":        LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA":    LowVolBullStrategy,
}

# Aliases for test backward compat
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
DistributionStrategy = MidVolCautiousStrategy
MeanReversionStrategy = MidVolCautiousStrategy
AccumulationStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy


# ---------------------------------------------------------------------------
# Strategy Orchestrator
# ---------------------------------------------------------------------------

class StrategyOrchestrator:
    """Selects strategy based on regime volatility rank and blends sizes.

    Instead of mapping by label name (which can be misleading since the HMM
    detects vol clusters not direction), this maps by VOLATILITY RANK:
    regimes are sorted by their learned variance, and position is allocated
    based on where in the vol spectrum the current regime sits.

    Parameters
    ----------
    config : dict
        Strategy config from settings.yaml under 'strategy'.
    regime_infos : list[RegimeInfo]
        Regime metadata from HMM engine (sorted by mean return).
    """

    def __init__(self, config: dict, regime_infos: list[RegimeInfo]) -> None:
        self._config = config
        self._regime_infos = regime_infos
        self._min_confidence = config.get("min_confidence", 0.55)
        self._uncertainty_size_mult = config.get("uncertainty_size_mult", 0.50)
        # Configured pairs (list of [a, b] symbol tuples). Used by pair strategies.
        # Single-asset strategies ignore this.
        raw_pairs = config.get("pairs", [])
        self._pairs: list[tuple[str, str]] = [
            (p[0], p[1]) for p in raw_pairs if len(p) == 2
        ]

        # Sort regimes by expected VOLATILITY (ascending)
        vol_sorted = sorted(regime_infos, key=lambda r: r.expected_volatility)
        self._vol_rank: dict[int, int] = {}  # regime_id -> vol_rank (0=lowest)
        for rank, info in enumerate(vol_sorted):
            self._vol_rank[info.regime_id] = rank

        # Create strategies based on vol rank, OR use a single override class
        # for every regime if config["strategy_class"] is set. The override is
        # for benchmarking / library-strategy backtests; production use should
        # rely on the vol-rank mapping.
        self._strategies: dict[int, BaseStrategy] = {}
        n = len(regime_infos)
        override_cls: Optional[type[BaseStrategy]] = config.get("strategy_class")
        for info in regime_infos:
            if override_cls is not None:
                strat_cls = override_cls
            else:
                rank = self._vol_rank[info.regime_id]
                strat_cls = _get_strategy_for_vol_rank(rank, n)
            self._strategies[info.regime_id] = strat_cls(config, info)

        logger.info(
            "StrategyOrchestrator: %d regimes, vol-ranked strategies: %s",
            n,
            {info.regime_name: (f"rank={self._vol_rank[info.regime_id]}, "
                                f"vol={info.expected_volatility:.3f}, "
                                f"strategy={self._strategies[info.regime_id].strategy_name}")
             for info in regime_infos},
        )

    def generate_signals(
        self, symbols: list[str], bars: dict[str, pd.DataFrame],
        regime_state: RegimeState, is_flickering: bool = False,
    ) -> tuple[list[Signal], list[PairSignal]]:
        """Run the dominant strategy and return both single-asset and pair signals.

        Returns a tuple `(signals, pair_signals)`. Single-asset strategies
        produce only `signals` (pair_signals will be empty). Pair strategies
        produce only `pair_signals` (signals will be empty).

        Callers that don't care about pairs can do:
            signals, _ = orchestrator.generate_signals(...)
        """
        probs = regime_state.state_probabilities
        dominant_id = regime_state.state_id

        # Uncertainty mode: only when genuinely uncertain (low probability)
        # Don't penalize for unconfirmed regimes — those are just transitions
        in_uncertainty = (
            is_flickering
            or regime_state.probability < self._min_confidence
        )

        dominant_strategy = self._strategies[dominant_id]

        # Snapshot the universe bars for any strategy that needs cross-symbol
        # data (momentum ranking, pair spreads, etc). No-op for strategies
        # that don't override set_universe_bars.
        dominant_strategy.set_universe_bars(bars)

        signals: list[Signal] = []
        pair_signals: list[PairSignal] = []

        if dominant_strategy.is_pair_strategy:
            # Pair-strategy path: iterate configured pairs, not symbols
            for pair in self._pairs:
                a, b = pair
                if a not in bars or b not in bars:
                    logger.debug("Skipping pair %s/%s: missing bars", a, b)
                    continue
                if len(bars[a]) < 50 or len(bars[b]) < 50:
                    continue

                ps = dominant_strategy.generate_pair_signal(pair, bars, regime_state)
                if ps is None:
                    continue

                if in_uncertainty:
                    ps.long_leg.position_size_pct *= self._uncertainty_size_mult
                    ps.short_leg.position_size_pct *= self._uncertainty_size_mult
                    ps.long_leg.leverage = 1.0
                    ps.short_leg.leverage = 1.0
                    ps.reasoning = (ps.reasoning or "") + " [UNCERTAINTY — size halved]"

                pair_signals.append(ps)
        else:
            # Single-asset path (unchanged behavior)
            for symbol in symbols:
                if symbol not in bars or len(bars[symbol]) < 50:
                    continue

                raw = dominant_strategy.generate_signal(symbol, bars[symbol], regime_state)
                if raw is None:
                    continue

                if in_uncertainty:
                    raw.position_size_pct *= self._uncertainty_size_mult
                    raw.leverage = 1.0
                    raw.reasoning += " [UNCERTAINTY — size halved]"

                signals.append(raw)

        return signals, pair_signals

    def update_regime_infos(self, regime_infos: list[RegimeInfo]) -> None:
        self.__init__(self._config, regime_infos)


# ---------------------------------------------------------------------------
# RegimeStrategyManager — wrapper for backward compat
# ---------------------------------------------------------------------------

class RegimeStrategyManager:
    def __init__(self, config: dict, regime_infos: Optional[list[RegimeInfo]] = None) -> None:
        self._config = config
        self._orchestrator: Optional[StrategyOrchestrator] = None
        if regime_infos:
            self._orchestrator = StrategyOrchestrator(config, regime_infos)

    def set_regime_infos(self, regime_infos: list[RegimeInfo]) -> None:
        self._orchestrator = StrategyOrchestrator(self._config, regime_infos)

    def get_signals(
        self, regime_state: RegimeState, symbols: list[str],
        bars: dict[str, pd.DataFrame], is_flickering: bool = False,
    ) -> list[Signal]:
        """Legacy interface: returns single-asset signals only.

        Pair signals (if any) are dropped silently. New callers that want
        both should use get_signals_and_pairs().
        """
        signals, _pair_signals = self.get_signals_and_pairs(
            regime_state, symbols, bars, is_flickering)
        return signals

    def get_signals_and_pairs(
        self, regime_state: RegimeState, symbols: list[str],
        bars: dict[str, pd.DataFrame], is_flickering: bool = False,
    ) -> tuple[list[Signal], list[PairSignal]]:
        """New interface: returns both single-asset and pair signals."""
        if self._orchestrator is None:
            raise RuntimeError("Must call set_regime_infos() before generating signals.")
        return self._orchestrator.generate_signals(symbols, bars, regime_state, is_flickering)
