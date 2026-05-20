"""Optional strategy library shipped alongside the regime trader template.

These eight strategies are independent of the three built-in archetypes
(`LowVolBullStrategy`, `MidVolCautiousStrategy`, `HighVolDefensiveStrategy`).
Users wire them into the orchestrator by overriding `_get_strategy_for_vol_rank`
or by direct instantiation.

All strategies are stateless — they recompute their decision from current
bars on every call so restarts are clean.

Single-asset (`generate_signal`):
    1. PureRegimeAllocation
    2. TrendFollowingRegimeFilter
    3. MeanReversionLowVol
    4. VolatilityBreakout
    5. MomentumRotation
    6. DefensiveLongShort

Pair (`generate_pair_signal`, `is_pair_strategy=True`):
    7. StaticPairsZScore
    8. CorrelationRegimeAllocation
"""

from __future__ import annotations

from core.hmm_engine import RegimeInfo


# Note: the HMM's RegimeInfo.expected_volatility is the std of standardized
# z-score features, NOT annualized return vol. We classify primarily by the
# regime LABEL (which is stable and known per scheme). Falls back to the
# expected_volatility field only when the label is unfamiliar.
#
# Bull regimes are typically calm; bear/crisis regimes are typically volatile.
LOW_VOL_LABELS = {"BULL", "STRONG_BULL", "EUPHORIA", "WEAK_BULL"}
HIGH_VOL_LABELS = {"CRASH", "STRONG_BEAR"}
# Z-score-space std fallbacks (used only when label isn't in either set).
LOW_VOL_THRESHOLD = 0.8       # below this → low (in z-score-space)
HIGH_VOL_THRESHOLD = 1.5      # above this → high


def classify_vol_rank(info: RegimeInfo) -> str:
    """Classify a regime's vol rank as 'low', 'mid', or 'high'.

    Strategies use this to defensively gate themselves to a vol regime in
    case a user wires them outside the orchestrator's natural placement.
    Classification is by regime LABEL when known; falls back to the
    standardized vol field for unknown labels.
    """
    label = (info.regime_name or "").upper()
    if label in LOW_VOL_LABELS:
        return "low"
    if label in HIGH_VOL_LABELS:
        return "high"
    if label:
        # Known scheme labels not in either set are mid (NEUTRAL, BEAR, WEAK_BEAR).
        if label in {"NEUTRAL", "BEAR", "WEAK_BEAR"}:
            return "mid"
    # Unknown label → fall back to z-score-std fallback thresholds.
    v = info.expected_volatility
    if v < LOW_VOL_THRESHOLD:
        return "low"
    if v > HIGH_VOL_THRESHOLD:
        return "high"
    return "mid"


from .pure_regime_allocation import PureRegimeAllocation
from .trend_following import TrendFollowingRegimeFilter
from .mean_reversion import MeanReversionLowVol
from .volatility_breakout import VolatilityBreakout
from .momentum_rotation import MomentumRotation
from .defensive_long_short import DefensiveLongShort
from .pairs_zscore import StaticPairsZScore
from .correlation_regime import CorrelationRegimeAllocation

__all__ = [
    "classify_vol_rank",
    "LOW_VOL_THRESHOLD",
    "HIGH_VOL_THRESHOLD",
    "PureRegimeAllocation",
    "TrendFollowingRegimeFilter",
    "MeanReversionLowVol",
    "VolatilityBreakout",
    "MomentumRotation",
    "DefensiveLongShort",
    "StaticPairsZScore",
    "CorrelationRegimeAllocation",
]
