"""PureRegimeAllocation — passive equal-weight allocator.

LONG every watchlist symbol when in low/mid vol; FLAT in high vol. This is
not an entry strategy — it's a passive allocator that switches between
risk-on and risk-off based purely on the regime.

The strategy reads the current watchlist size from `set_universe_bars()`
(populated by the orchestrator) so each signal sizes itself as
``gross_exposure / N_watchlist``. If the universe wasn't snapshot, falls
back to ``n_watchlist_hint`` from config.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BaseStrategy, Signal, SignalDirection, _compute_atr,
)
from . import classify_vol_rank


class PureRegimeAllocation(BaseStrategy):
    """Equal-weight passive allocator gated by regime vol rank.

    Parameters (config)
    -------------------
    low_vol_gross : float = 1.0
        Total gross exposure in low-vol regimes (split across N watchlist).
    mid_vol_gross : float = 0.50
        Total gross exposure in mid-vol regimes.
    high_vol_gross : float = 0.0
        Total gross exposure in high-vol regimes (close all).
    n_watchlist_hint : int = 6
        Fallback watchlist size when set_universe_bars() hasn't been called.
    stop_atr : float = 3.0
        Stop distance in ATRs (loose — passive allocator, not an entry edge).
    """

    strategy_name = "pure_regime_allocation"

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        super().__init__(config, regime_info)
        self._low_vol_gross = float(config.get("low_vol_gross", 1.0))
        self._mid_vol_gross = float(config.get("mid_vol_gross", 0.50))
        self._high_vol_gross = float(config.get("high_vol_gross", 0.0))
        self._n_hint = int(config.get("n_watchlist_hint", 6))
        self._stop_atr = float(config.get("stop_atr", 3.0))
        # Optional leverage by vol rank — defaults to 1.0 (unleveraged).
        self._low_vol_leverage = float(config.get("low_vol_leverage", 1.0))
        self._mid_vol_leverage = float(config.get("mid_vol_leverage", 1.0))

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState,
    ) -> Optional[Signal]:
        if len(bars) < 60:
            return None

        rank = classify_vol_rank(self._regime_info)
        if rank == "low":
            gross = self._low_vol_gross
            leverage = self._low_vol_leverage
        elif rank == "mid":
            gross = self._mid_vol_gross
            leverage = self._mid_vol_leverage
        else:
            gross = self._high_vol_gross
            leverage = 1.0

        # Determine universe size: prefer the live snapshot, fall back to hint.
        n = len(self._universe_bars) if self._universe_bars else self._n_hint
        n = max(n, 1)
        per_name = gross / n

        price = float(bars["close"].iloc[-1])

        if per_name <= 0:
            # High-vol or zero allocation → emit FLAT to close any position.
            return self._make_signal(
                symbol, SignalDirection.FLAT, price, 0.0, regime_state,
                f"PureRegimeAllocation: {rank}-vol regime, close position",
                position_size_pct=0.0, leverage=1.0,
            )

        atr = float(_compute_atr(bars).iloc[-1])
        stop = price - self._stop_atr * atr

        return self._make_signal(
            symbol, SignalDirection.LONG, price, stop, regime_state,
            f"PureRegimeAllocation: {rank}-vol gross={gross:.0%} "
            f"lev={leverage:.2f}x, per_name={per_name:.2%} (N={n})",
            position_size_pct=per_name, leverage=leverage,
        )
