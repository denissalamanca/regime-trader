"""MeanReversionLowVol — RSI dip-buyer in low-vol regimes.

LONG when RSI < oversold threshold AND vol-rank == low.
FLAT when RSI > exit threshold (mean-reverted).

Mean reversion in trending bull markets gets cooked — the low-vol gate
keeps this strategy out of crisis chop where dips keep dipping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BaseStrategy, Signal, SignalDirection, _compute_atr,
)
from . import classify_vol_rank


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class MeanReversionLowVol(BaseStrategy):
    """RSI mean-reversion, low-vol gated.

    Parameters (config)
    -------------------
    rsi_period : int = 14
    rsi_oversold : float = 30
        Entry: RSI <= this triggers LONG.
    rsi_exit : float = 50
        Exit: RSI >= this triggers FLAT.
    stop_atr : float = 2.0
    position_size_pct : float = 0.50
    """

    strategy_name = "mean_reversion_low_vol"

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        super().__init__(config, regime_info)
        self._rsi_period = int(config.get("rsi_period", 14))
        self._rsi_oversold = float(config.get("rsi_oversold", 30))
        self._rsi_exit = float(config.get("rsi_exit", 50))
        self._stop_atr = float(config.get("stop_atr", 2.0))
        self._position_size_pct = float(config.get("position_size_pct", 0.50))

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState,
    ) -> Optional[Signal]:
        if len(bars) < max(60, self._rsi_period * 3):
            return None

        if classify_vol_rank(self._regime_info) != "low":
            return None

        close = bars["close"]
        rsi = _rsi(close, self._rsi_period)
        cur_rsi = float(rsi.iloc[-1])
        if pd.isna(cur_rsi):
            return None
        price = float(close.iloc[-1])

        if cur_rsi <= self._rsi_oversold:
            atr = float(_compute_atr(bars).iloc[-1])
            stop = price - self._stop_atr * atr
            return self._make_signal(
                symbol, SignalDirection.LONG, price, stop, regime_state,
                f"MeanReversion: RSI={cur_rsi:.1f} <= {self._rsi_oversold:.0f}, "
                f"low-vol entry",
                position_size_pct=self._position_size_pct, leverage=1.0,
            )
        if cur_rsi >= self._rsi_exit:
            return self._make_signal(
                symbol, SignalDirection.FLAT, price, 0.0, regime_state,
                f"MeanReversion: RSI={cur_rsi:.1f} >= {self._rsi_exit:.0f}, exit",
                position_size_pct=0.0, leverage=1.0,
            )
        # In between → no signal (don't add and don't close)
        return None
