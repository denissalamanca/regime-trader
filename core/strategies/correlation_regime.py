"""CorrelationRegimeAllocation — pair fade gated by realized correlation.

A relative of StaticPairsZScore where the position size *scales with
correlation*. High correlation → full size. Correlation drifting toward
``exit_corr`` → shrinking size. Below ``exit_corr`` → no signal.

Spread / z-score / hedge-ratio computation matches StaticPairsZScore,
but the entry threshold is looser (``entry_z=1.5`` default) since the
correlation gate is doing additional work.

Low-vol regime gated.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BaseStrategy, PairSignal, Signal, SignalDirection, _compute_atr,
)
from . import classify_vol_rank


class CorrelationRegimeAllocation(BaseStrategy):
    """Correlation-gated pair fade with continuous size scaling.

    Parameters (config)
    -------------------
    corr_lookback : int = 60
    entry_corr : float = 0.7
        Realized correlation must exceed this to open.
    exit_corr : float = 0.5
        Below this, no signal (size scales linearly to zero between entry/exit).
    entry_z : float = 1.5
        |z| threshold to fire (looser than StaticPairsZScore since correlation
        gates already filter).
    pair_pct : float = 0.20
        Maximum equity allocated when correlation == 1.0.
    stop_atr : float = 5.0
        Per-leg disaster stop in ATRs.
    """

    is_pair_strategy = True
    strategy_name = "correlation_regime_allocation"

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        super().__init__(config, regime_info)
        self._corr_lookback = int(config.get("corr_lookback", 60))
        self._entry_corr = float(config.get("entry_corr", 0.7))
        self._exit_corr = float(config.get("exit_corr", 0.5))
        self._entry_z = float(config.get("entry_z", 1.5))
        self._pair_pct = float(config.get("pair_pct", 0.20))
        self._stop_atr = float(config.get("stop_atr", 5.0))

    def generate_signal(self, symbol, bars, regime_state):  # noqa: ARG002
        return None

    def _compute_spread(
        self, close_a: pd.Series, close_b: pd.Series, lookback: int,
    ) -> Optional[tuple[float, float, float]]:
        if len(close_a) < lookback + 1 or len(close_b) < lookback + 1:
            return None
        a = close_a.iloc[-lookback:].astype(float)
        b = close_b.iloc[-lookback:].astype(float)
        if (a <= 0).any() or (b <= 0).any():
            return None
        log_a = np.log(a.values)
        log_b = np.log(b.values)
        try:
            slope, _ = np.polyfit(log_b, log_a, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        hedge_ratio = float(slope)
        spread = log_a - hedge_ratio * log_b
        sigma = float(spread.std(ddof=0))
        if sigma <= 1e-12:
            return None
        z = float((spread[-1] - spread.mean()) / sigma)
        return z, float(spread[-1]), hedge_ratio

    @staticmethod
    def _correlation(a: pd.Series, b: pd.Series, lookback: int) -> Optional[float]:
        if len(a) < lookback + 1 or len(b) < lookback + 1:
            return None
        ra = a.iloc[-lookback:].pct_change().dropna()
        rb = b.iloc[-lookback:].pct_change().dropna()
        combined = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(combined) < 30:
            return None
        return float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))

    def generate_pair_signal(
        self,
        pair: tuple[str, str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
    ) -> Optional[PairSignal]:
        if classify_vol_rank(self._regime_info) != "low":
            return None

        a, b = pair
        if a not in bars or b not in bars:
            return None
        bars_a, bars_b = bars[a], bars[b]
        if len(bars_a) < self._corr_lookback + 5 or len(bars_b) < self._corr_lookback + 5:
            return None

        corr = self._correlation(bars_a["close"], bars_b["close"], self._corr_lookback)
        if corr is None or corr < self._exit_corr:
            return None

        result = self._compute_spread(
            bars_a["close"], bars_b["close"], self._corr_lookback,
        )
        if result is None:
            return None
        z, spread, hedge_ratio = result

        if abs(z) < self._entry_z:
            return None

        # Linear size ramp: full at corr=1.0, zero at corr=exit_corr.
        denom = max(1.0 - self._exit_corr, 1e-6)
        size_factor = max(0.0, min(1.0, (corr - self._exit_corr) / denom))
        if size_factor <= 0:
            return None
        per_leg_pct = (self._pair_pct / 2.0) * size_factor
        if per_leg_pct <= 0:
            return None

        price_a = float(bars_a["close"].iloc[-1])
        price_b = float(bars_b["close"].iloc[-1])
        atr_a = float(_compute_atr(bars_a).iloc[-1])
        atr_b = float(_compute_atr(bars_b).iloc[-1])

        if z > 0:
            long_sym, short_sym = b, a
            long_price, short_price = price_b, price_a
            long_atr, short_atr = atr_b, atr_a
        else:
            long_sym, short_sym = a, b
            long_price, short_price = price_a, price_b
            long_atr, short_atr = atr_a, atr_b

        long_stop = long_price - self._stop_atr * long_atr
        short_stop = short_price + self._stop_atr * short_atr

        long_leg = self._make_signal(
            long_sym, SignalDirection.LONG, long_price, long_stop, regime_state,
            f"CorrelationRegime: z={z:.2f}, corr={corr:.2f}, size={size_factor:.0%} long {long_sym}",
            position_size_pct=per_leg_pct, leverage=1.0,
        )
        short_leg = self._make_signal(
            short_sym, SignalDirection.SHORT, short_price, short_stop, regime_state,
            f"CorrelationRegime: z={z:.2f}, corr={corr:.2f}, size={size_factor:.0%} short {short_sym}",
            position_size_pct=per_leg_pct, leverage=1.0,
        )

        return PairSignal(
            pair=pair, long_leg=long_leg, short_leg=short_leg,
            spread_value=spread, z_score=z, hedge_ratio=hedge_ratio,
            correlation=corr,
            reasoning=f"corr={corr:.2f} (>{self._exit_corr}), z={z:.2f}, size×{size_factor:.0%}",
            timestamp=regime_state.timestamp,
        )
