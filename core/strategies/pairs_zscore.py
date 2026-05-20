"""StaticPairsZScore — z-score mean-reversion on a cointegrated pair.

Spread definition (per design §3.7):
    spread_t = log(price_a_t) - hedge_ratio * log(price_b_t)
    hedge_ratio = OLS slope of log(a) ~ log(b) over the lookback window.
    z_t       = (spread_t - mean(spread, lookback)) / std(spread, lookback)

Entry rules:
    z >  entry_z   → spread is rich → SHORT A, LONG B
    z < -entry_z   → spread is cheap → LONG A, SHORT B
    otherwise      → no entry (return None)

This strategy emits ENTRY signals only. Exits are handled by the
backtester / live executor via the leg-level disaster stops (5×ATR),
which fire when cointegration breaks (|z| ≫ entry_z) or when prices
move dramatically. A future revision could add explicit z-revert exits;
for now, position management is downstream.

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


class StaticPairsZScore(BaseStrategy):
    """Z-score pair trade with OLS hedge ratio.

    Parameters (config)
    -------------------
    lookback : int = 60
        Bars used for hedge-ratio regression and z-score window.
    entry_z : float = 2.0
        |z| threshold to fire an entry.
    exit_z : float = 0.5
        |z| threshold below which the pair is considered mean-reverted.
        (Currently informational — exits ride leg stops.)
    stop_z : float = 4.0
        |z| threshold above which the pair is considered broken.
        (Currently informational — exits ride leg stops.)
    pair_pct : float = 0.20
        Total equity allocated to the pair (split pair_pct/2 per leg).
    stop_atr : float = 5.0
        Per-leg disaster stop in ATRs. Loose by design — leg stops are
        for catastrophe, the pair logic handles graceful exits.
    """

    is_pair_strategy = True
    strategy_name = "static_pairs_zscore"

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        super().__init__(config, regime_info)
        self._lookback = int(config.get("lookback", 60))
        self._entry_z = float(config.get("entry_z", 2.0))
        self._exit_z = float(config.get("exit_z", 0.5))
        self._stop_z = float(config.get("stop_z", 4.0))
        self._pair_pct = float(config.get("pair_pct", 0.20))
        self._stop_atr = float(config.get("stop_atr", 5.0))

    # generate_signal is required by the abstract base; pair strategies stub it.
    def generate_signal(self, symbol, bars, regime_state):  # noqa: ARG002
        return None

    def _compute_spread(
        self, close_a: pd.Series, close_b: pd.Series,
    ) -> Optional[tuple[float, float, float]]:
        """Return (z_score, spread_value, hedge_ratio) or None on failure."""
        if len(close_a) < self._lookback + 1 or len(close_b) < self._lookback + 1:
            return None
        a = close_a.iloc[-self._lookback:].astype(float)
        b = close_b.iloc[-self._lookback:].astype(float)
        if (a <= 0).any() or (b <= 0).any():
            return None
        log_a = np.log(a.values)
        log_b = np.log(b.values)
        # OLS slope of log(a) ~ log(b)
        try:
            slope, _intercept = np.polyfit(log_b, log_a, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        hedge_ratio = float(slope)
        spread = log_a - hedge_ratio * log_b
        mu = float(spread.mean())
        sigma = float(spread.std(ddof=0))
        if sigma <= 1e-12:
            return None
        z = float((spread[-1] - mu) / sigma)
        return z, float(spread[-1]), hedge_ratio

    def generate_pair_signal(
        self,
        pair: tuple[str, str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
    ) -> Optional[PairSignal]:
        # Vol gating
        if classify_vol_rank(self._regime_info) != "low":
            return None

        a, b = pair
        if a not in bars or b not in bars:
            return None
        bars_a, bars_b = bars[a], bars[b]
        if len(bars_a) < self._lookback + 5 or len(bars_b) < self._lookback + 5:
            return None

        result = self._compute_spread(bars_a["close"], bars_b["close"])
        if result is None:
            return None
        z, spread, hedge_ratio = result

        if abs(z) < self._entry_z:
            return None
        if abs(z) > self._stop_z:
            # Cointegration likely broken — don't open new exposure.
            return None

        price_a = float(bars_a["close"].iloc[-1])
        price_b = float(bars_b["close"].iloc[-1])
        atr_a = float(_compute_atr(bars_a).iloc[-1])
        atr_b = float(_compute_atr(bars_b).iloc[-1])

        per_leg_pct = self._pair_pct / 2.0

        if z > 0:
            # spread rich → short A, long B
            long_sym, short_sym = b, a
            long_price, short_price = price_b, price_a
            long_atr, short_atr = atr_b, atr_a
        else:
            # spread cheap → long A, short B
            long_sym, short_sym = a, b
            long_price, short_price = price_a, price_b
            long_atr, short_atr = atr_a, atr_b

        long_stop = long_price - self._stop_atr * long_atr
        short_stop = short_price + self._stop_atr * short_atr

        long_leg = self._make_signal(
            long_sym, SignalDirection.LONG, long_price, long_stop, regime_state,
            f"StaticPairsZScore: z={z:.2f} → long {long_sym}",
            position_size_pct=per_leg_pct, leverage=1.0,
        )
        short_leg = self._make_signal(
            short_sym, SignalDirection.SHORT, short_price, short_stop, regime_state,
            f"StaticPairsZScore: z={z:.2f} → short {short_sym}",
            position_size_pct=per_leg_pct, leverage=1.0,
        )

        return PairSignal(
            pair=pair, long_leg=long_leg, short_leg=short_leg,
            spread_value=spread, z_score=z, hedge_ratio=hedge_ratio,
            reasoning=f"z={z:.2f} (entry @ |z|>{self._entry_z}), hedge_ratio={hedge_ratio:.3f}",
            timestamp=regime_state.timestamp,
        )
