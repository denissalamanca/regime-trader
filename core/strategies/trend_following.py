"""TrendFollowingRegimeFilter — long-only with MA filter, low-vol gated.

LONG when:
  - vol-rank == low (gated)
  - price > N-day moving average

FLAT (close) when price drops below the MA.

The MA filter ensures we only chase confirmed trends — not chop. Coupled
with the low-vol gate, this strategy stays out of choppy/crisis markets
entirely.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BaseStrategy, Signal, SignalDirection, _compute_atr, _sma,
)
from . import classify_vol_rank


class TrendFollowingRegimeFilter(BaseStrategy):
    """Trend-follow above MA, low-vol regime only.

    Parameters (config)
    -------------------
    ma_period : int = 200
        Moving-average lookback. 200 = the classic long-trend filter.
    stop_atr : float = 2.0
        ATR multiplier for the disaster stop.
    position_size_pct : float = 0.80
        Equity allocation when in low-vol AND above MA.
    """

    strategy_name = "trend_following_regime_filter"

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        super().__init__(config, regime_info)
        self._ma_period = int(config.get("ma_period", 200))
        self._stop_atr = float(config.get("stop_atr", 2.0))
        self._position_size_pct = float(config.get("position_size_pct", 0.80))

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState,
    ) -> Optional[Signal]:
        if len(bars) < max(60, self._ma_period + 5):
            return None

        rank = classify_vol_rank(self._regime_info)
        if rank != "low":
            # Defensive: orchestrator should only place this on low-vol, but
            # if a user wires it elsewhere, do nothing.
            return None

        close = bars["close"]
        price = float(close.iloc[-1])
        ma = float(_sma(close, self._ma_period).iloc[-1])
        if pd.isna(ma):
            return None

        if price <= ma:
            # Trend broken → FLAT (close any open position)
            return self._make_signal(
                symbol, SignalDirection.FLAT, price, 0.0, regime_state,
                f"TrendFollowing: price ${price:.2f} <= MA ${ma:.2f} — flat",
                position_size_pct=0.0, leverage=1.0,
            )

        atr = float(_compute_atr(bars).iloc[-1])
        # Tighter of (ATR stop, MA-buffer stop) — never widen below MA-0.5*ATR.
        stop = max(price - self._stop_atr * atr, ma - 0.5 * atr)

        return self._make_signal(
            symbol, SignalDirection.LONG, price, stop, regime_state,
            f"TrendFollowing: low-vol + price ${price:.2f} > MA ${ma:.2f}, "
            f"stop=${stop:.2f}",
            position_size_pct=self._position_size_pct, leverage=1.0,
        )
