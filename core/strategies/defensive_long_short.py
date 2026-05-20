"""DefensiveLongShort — long top-momentum + short bottom-momentum in high-vol.

LONG every symbol in top-N by trailing return (always).
SHORT every symbol in bottom-M by trailing return (only in high-vol).
FLAT for everyone else.

Each leg is signaled independently — this is *not* a pair strategy, just a
strategy that sometimes returns SHORT direction. Sized so longs total
``long_gross`` and shorts total ``short_gross`` of equity.

Stateless rebalance, same pattern as MomentumRotation.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BaseStrategy, Signal, SignalDirection, _compute_atr,
)
from . import classify_vol_rank


class DefensiveLongShort(BaseStrategy):
    """Long top-momentum + short bottom-momentum (shorts gated to high-vol).

    Parameters (config)
    -------------------
    lookback : int = 60
    top_n : int = 3
        Number of long names.
    bottom_m : int = 2
        Number of short names (only fires in high-vol).
    short_in_regimes : list[str] = ["high"]
        Vol ranks where shorts are allowed.
    long_gross : float = 0.6
        Total long-side gross exposure (split equal-weight across top-N).
    short_gross : float = 0.4
        Total short-side gross exposure (split equal-weight across bottom-M).
    stop_atr : float = 2.0
    """

    strategy_name = "defensive_long_short"

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        super().__init__(config, regime_info)
        self._lookback = int(config.get("lookback", 60))
        self._top_n = int(config.get("top_n", 3))
        self._bottom_m = int(config.get("bottom_m", 2))
        self._short_in_regimes = list(config.get("short_in_regimes", ["high"]))
        self._long_gross = float(config.get("long_gross", 0.6))
        self._short_gross = float(config.get("short_gross", 0.4))
        self._stop_atr = float(config.get("stop_atr", 2.0))

    def _rank_universe(self) -> list[tuple[str, float]]:
        ranked: list[tuple[str, float]] = []
        for sym, bars in self._universe_bars.items():
            if bars is None or len(bars) < self._lookback + 1:
                continue
            close = bars["close"]
            r = float(close.iloc[-1] / close.iloc[-self._lookback - 1] - 1)
            ranked.append((sym, r))
        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState,
    ) -> Optional[Signal]:
        if len(bars) < max(60, self._lookback + 1):
            return None
        if not self._universe_bars:
            return None

        ranked = self._rank_universe()
        if not ranked:
            return None

        rank = classify_vol_rank(self._regime_info)
        shorts_active = rank in self._short_in_regimes

        top_n = max(self._top_n, 1)
        bottom_m = max(self._bottom_m, 1)
        top_symbols = {s for s, _ in ranked[:top_n]}
        bottom_symbols = {s for s, _ in ranked[-bottom_m:]} if shorts_active else set()

        price = float(bars["close"].iloc[-1])
        atr = float(_compute_atr(bars).iloc[-1])

        if symbol in top_symbols:
            stop = price - self._stop_atr * atr
            per_name = self._long_gross / top_n
            return self._make_signal(
                symbol, SignalDirection.LONG, price, stop, regime_state,
                f"DefensiveLongShort: {symbol} top-{top_n} long ({rank}-vol), "
                f"size={per_name:.1%}",
                position_size_pct=per_name, leverage=1.0,
            )
        if symbol in bottom_symbols:
            stop = price + self._stop_atr * atr
            per_name = self._short_gross / bottom_m
            return self._make_signal(
                symbol, SignalDirection.SHORT, price, stop, regime_state,
                f"DefensiveLongShort: {symbol} bottom-{bottom_m} short "
                f"({rank}-vol), size={per_name:.1%}",
                position_size_pct=per_name, leverage=1.0,
            )

        # Not in either basket — emit FLAT to close anything that was open.
        return self._make_signal(
            symbol, SignalDirection.FLAT, price, 0.0, regime_state,
            f"DefensiveLongShort: {symbol} not in top/bottom, exit",
            position_size_pct=0.0, leverage=1.0,
        )
