"""VolatilityBreakout — N-day high breakout with inverse-ATR sizing.

LONG when price breaks above the rolling N-day high AND vol-rank != high.
Position size scales inversely with ATR — calm markets get a larger share,
volatile markets get a smaller share. Stop trails on ATR.

Sized to risk a fixed fraction of equity per trade so a stop-out costs
``target_risk_pct`` regardless of vol regime.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BaseStrategy, Signal, SignalDirection, _compute_atr,
)
from . import classify_vol_rank


class VolatilityBreakout(BaseStrategy):
    """Donchian-style breakout with inverse-vol sizing.

    Parameters (config)
    -------------------
    breakout_lookback : int = 20
    stop_atr : float = 2.0
    target_risk_pct : float = 0.01
        Fraction of equity to risk per trade. With ATR-based stops this
        determines the size implicitly.
    max_position_size : float = 0.30
        Hard cap on position size, even when ATR is tiny.
    """

    strategy_name = "volatility_breakout"

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        super().__init__(config, regime_info)
        self._lookback = int(config.get("breakout_lookback", 20))
        self._stop_atr = float(config.get("stop_atr", 2.0))
        self._target_risk_pct = float(config.get("target_risk_pct", 0.01))
        self._max_position_size = float(config.get("max_position_size", 0.30))

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState,
    ) -> Optional[Signal]:
        if len(bars) < max(60, self._lookback + 5):
            return None

        if classify_vol_rank(self._regime_info) == "high":
            return None

        # Use highs prior to today (no look-ahead).
        high = bars["high"]
        rolling_max = high.iloc[-self._lookback - 1:-1].max()
        price = float(bars["close"].iloc[-1])
        if pd.isna(rolling_max) or price <= float(rolling_max):
            return None

        atr = float(_compute_atr(bars).iloc[-1])
        if atr <= 0 or pd.isna(atr):
            return None

        stop_distance = self._stop_atr * atr
        stop = price - stop_distance

        # Inverse-ATR sizing: target_risk_pct of equity / (ATR/price) → fraction.
        # Capped at max_position_size.
        atr_pct = atr / price
        if atr_pct <= 0:
            return None
        target_size = min(
            self._target_risk_pct / atr_pct,
            self._max_position_size,
        )

        return self._make_signal(
            symbol, SignalDirection.LONG, price, stop, regime_state,
            f"VolatilityBreakout: price ${price:.2f} > {self._lookback}d-high "
            f"${rolling_max:.2f}, ATR%={atr_pct:.2%}, size={target_size:.1%}",
            position_size_pct=target_size, leverage=1.0,
            atr_pct=atr_pct, breakout_high=float(rolling_max),
        )
