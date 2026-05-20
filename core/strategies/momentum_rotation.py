"""MomentumRotation — equal-weight top-N momentum rotation.

LONG every symbol that's in the current top-N ranked by trailing return.
FLAT every symbol that's outside top-N.

Stateless: the top-N composition is recomputed every bar from
``self._universe_bars``, so add/drop signals fire automatically as
ranking shifts. No date tracking, no persisted state — restarts cleanly.

In high-vol regimes, ``top_n_high_vol`` shrinks the basket so we hold
fewer concurrent positions and keep more cash.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BaseStrategy, Signal, SignalDirection, _compute_atr,
)
from . import classify_vol_rank


class MomentumRotation(BaseStrategy):
    """Top-N momentum rotation, stateless rebalance.

    Parameters (config)
    -------------------
    lookback : int = 60
        Trading-day lookback for the trailing-return ranking.
    top_n : int = 3
        Number of names to hold in low/mid-vol regimes.
    top_n_high_vol : int = 1
        Number of names to hold in high-vol regimes.
    stop_atr : float = 2.0
    """

    strategy_name = "momentum_rotation"

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        super().__init__(config, regime_info)
        self._lookback = int(config.get("lookback", 60))
        self._top_n = int(config.get("top_n", 3))
        self._top_n_high_vol = int(config.get("top_n_high_vol", 1))
        self._stop_atr = float(config.get("stop_atr", 2.0))

    def _rank_universe(self) -> list[tuple[str, float]]:
        """Return list of (symbol, return) sorted by return descending.

        Drops symbols that don't have enough bars for the lookback.
        """
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
        n_target = self._top_n_high_vol if rank == "high" else self._top_n
        n_target = max(n_target, 1)

        top_symbols = {s for s, _ in ranked[:n_target]}
        price = float(bars["close"].iloc[-1])

        if symbol in top_symbols:
            atr = float(_compute_atr(bars).iloc[-1])
            stop = price - self._stop_atr * atr
            per_name = 1.0 / n_target
            return self._make_signal(
                symbol, SignalDirection.LONG, price, stop, regime_state,
                f"MomentumRotation: {symbol} in top-{n_target} ({rank}-vol), "
                f"size={per_name:.1%}",
                position_size_pct=per_name, leverage=1.0,
                top_n=n_target,
            )
        # Out of top-N → FLAT (close any position)
        return self._make_signal(
            symbol, SignalDirection.FLAT, price, 0.0, regime_state,
            f"MomentumRotation: {symbol} not in top-{n_target}, exit",
            position_size_pct=0.0, leverage=1.0,
        )
