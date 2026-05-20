"""Technical indicator computation and feature engineering for regime classification.

IMPORTANT DESIGN NOTE:
=======================
These features feed the HMM regime classifier. They characterize the MARKET
ENVIRONMENT — they are NOT standalone entry/exit signals.

The difference is fundamental:
  - "RSI crossed 30 so buy" = using an indicator as a signal (NOT what we do here)
  - "RSI + vol + volume + trend + 10 other features together characterize whether
    we're in a low-vol bull, high-vol crash, or transition regime" = regime classification

Every function in this module produces a feature for the regime classifier.
Strategy-level entry/exit signals are computed separately in regime_strategies.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling z-score standardization
# ---------------------------------------------------------------------------

def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Standardize a series with a rolling z-score to maintain stationarity.

    Parameters
    ----------
    series : pd.Series
        Raw feature values.
    window : int
        Lookback window for mean/std computation. Default 252 (~1 year daily).

    Returns
    -------
    pd.Series
        Z-scored values. NaN where insufficient history.
    """
    roll_mean = series.rolling(window=window, min_periods=max(window // 2, 20)).mean()
    roll_std = series.rolling(window=window, min_periods=max(window // 2, 20)).std()
    # Avoid division by zero — if std is 0, the feature has no variance
    roll_std = roll_std.replace(0.0, np.nan)
    return (series - roll_mean) / roll_std


# ---------------------------------------------------------------------------
# Individual feature functions — pure, stateless, operating on DataFrames
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Compute log returns over the given period.

    This is a REGIME CLASSIFICATION feature — it captures the magnitude and
    direction of price moves as one input to the HMM, not a trading signal.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        Number of periods for the return calculation.

    Returns
    -------
    pd.Series
        Log returns.
    """
    return np.log(close / close.shift(period))


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling realized volatility (annualized std of log returns).

    This is a REGIME CLASSIFICATION feature — it characterizes the current
    volatility environment, not a signal to trade vol directly.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    window : int
        Rolling window for std computation.

    Returns
    -------
    pd.Series
        Annualized rolling volatility.
    """
    ret = np.log(close / close.shift(1))
    return ret.rolling(window=window, min_periods=window).std() * np.sqrt(252)


def volatility_ratio(close: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """Compute the ratio of short-term to long-term realized volatility.

    Ratio > 1 means vol is expanding (recent moves larger than average).
    Ratio < 1 means vol is compressing. This helps the HMM detect regime
    transitions where vol structure is changing.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    short_window : int
        Short vol lookback.
    long_window : int
        Long vol lookback.

    Returns
    -------
    pd.Series
        Short vol / long vol ratio.
    """
    ret = np.log(close / close.shift(1))
    short_vol = ret.rolling(window=short_window, min_periods=short_window).std()
    long_vol = ret.rolling(window=long_window, min_periods=long_window).std()
    long_vol = long_vol.replace(0.0, np.nan)
    return short_vol / long_vol


def normalized_volume(volume: pd.Series, window: int = 50) -> pd.Series:
    """Compute z-score of volume relative to rolling mean.

    This is a REGIME CLASSIFICATION feature — abnormal volume often
    accompanies regime changes. It tells the HMM about participation
    levels, not whether to buy or sell.

    Parameters
    ----------
    volume : pd.Series
        Volume data.
    window : int
        Rolling window for mean/std.

    Returns
    -------
    pd.Series
        Volume z-scores.
    """
    roll_mean = volume.rolling(window=window, min_periods=window).mean()
    roll_std = volume.rolling(window=window, min_periods=window).std()
    roll_std = roll_std.replace(0.0, np.nan)
    return (volume - roll_mean) / roll_std


def volume_trend(volume: pd.Series, window: int = 10) -> pd.Series:
    """Compute slope of the volume SMA (via linear regression).

    Positive slope = volume increasing over time. This helps the HMM
    detect accumulation/distribution phases across regimes.

    Parameters
    ----------
    volume : pd.Series
        Volume data.
    window : int
        SMA lookback for slope computation.

    Returns
    -------
    pd.Series
        Slope of volume SMA (units: volume per period).
    """
    sma = volume.rolling(window=window, min_periods=window).mean()
    # Slope via rolling linear regression: cov(x, y) / var(x)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(vals: np.ndarray) -> float:
        if len(vals) < window or np.any(np.isnan(vals)):
            return np.nan
        return np.sum((x - x_mean) * (vals - vals.mean())) / x_var

    return sma.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX).

    ADX measures trend STRENGTH regardless of direction. High ADX = strong
    trend (bullish or bearish). Low ADX = range-bound. This is a REGIME
    CLASSIFICATION feature — it tells the HMM whether the market is trending,
    not which direction to trade.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with 'high', 'low', 'close' columns.
    period : int
        ADX lookback period.

    Returns
    -------
    pd.Series
        ADX values (0-100 scale).
    """
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = _true_range(high, low, close)

    # Wilder's smoothing (exponential with alpha = 1/period)
    alpha = 1.0 / period
    atr_smooth = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_smooth
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_smooth

    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum

    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def sma_slope(close: pd.Series, sma_period: int = 50, slope_window: int = 10) -> pd.Series:
    """Compute the slope of a simple moving average.

    Positive slope = uptrend. This is a REGIME CLASSIFICATION feature —
    it tells the HMM about the prevailing trend structure, not whether
    to enter a position.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    sma_period : int
        SMA lookback.
    slope_window : int
        Window for slope computation on the SMA.

    Returns
    -------
    pd.Series
        SMA slope (normalized by SMA level to be scale-invariant).
    """
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
    # Percentage change of SMA over slope_window
    return sma.pct_change(periods=slope_window)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index.

    As a REGIME CLASSIFICATION feature, RSI helps the HMM characterize
    mean-reversion tendency. Persistently high RSI = momentum regime.
    Oscillating RSI = mean-reversion regime. This is NOT used as
    "RSI < 30 = buy signal".

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        RSI lookback period.

    Returns
    -------
    pd.Series
        RSI values (0-100 scale).
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def distance_from_sma(close: pd.Series, sma_period: int = 200) -> pd.Series:
    """Compute distance from SMA as a percentage of SMA.

    This is a REGIME CLASSIFICATION feature — large positive distance
    = overextended bull, large negative = overextended bear. It helps
    the HMM distinguish euphoria from normal bull and crash from normal bear.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    sma_period : int
        SMA lookback (200 is the long-term trend benchmark).

    Returns
    -------
    pd.Series
        (close - SMA) / SMA as a fraction.
    """
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
    return (close - sma) / sma


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    """Compute Rate of Change (ROC) as percentage.

    ROC captures momentum at a specific timescale. Used as a REGIME
    CLASSIFICATION feature — the HMM uses ROC across multiple timescales
    to characterize momentum structure.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        ROC lookback.

    Returns
    -------
    pd.Series
        (close - close[t-period]) / close[t-period].
    """
    return close.pct_change(periods=period)


def normalized_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR as a percentage of close price.

    Normalizing by close makes ATR comparable across price levels and
    assets. This is a REGIME CLASSIFICATION feature for characterizing
    the current range/volatility environment.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with 'high', 'low', 'close' columns.
    period : int
        ATR lookback.

    Returns
    -------
    pd.Series
        ATR / close (fraction).
    """
    tr = _true_range(bars["high"], bars["low"], bars["close"])
    atr_val = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr_val / bars["close"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute True Range."""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV data with 'high', 'low', 'close' columns.
    period : int
        ATR lookback period.

    Returns
    -------
    pd.Series
        ATR values.
    """
    tr = _true_range(bars["high"], bars["low"], bars["close"])
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """Computes feature matrices for the HMM regime classifier.

    All features are standardized with rolling z-scores (252-period lookback)
    to maintain stationarity. The HMM requires stationary inputs — raw price
    levels or un-normalized indicators would violate this assumption.

    Parameters
    ----------
    config : dict
        Configuration dict. Expected keys:
        - zscore_window (int): lookback for rolling z-score (default 252)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._zscore_window: int = config.get("zscore_window", 252)

    def compute_hmm_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute the full feature matrix for HMM regime classification.

        Features (all rolling z-scored):
        - ret_1, ret_5, ret_20: log returns at 1, 5, 20 periods
        - rvol_20: 20-period realized volatility
        - vol_ratio: 5-period vol / 20-period vol
        - vol_norm: normalized volume (z-score vs 50-period mean)
        - vol_trend: slope of 10-period volume SMA
        - adx_14: Average Directional Index
        - sma50_slope: slope of 50-period SMA
        - rsi14_zscore: RSI(14) z-scored
        - dist_sma200: distance from 200 SMA as pct
        - roc_10, roc_20: rate of change 10 and 20 period
        - natr: normalized ATR (ATR / close)

        Parameters
        ----------
        bars : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume.
            Index should be a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Feature matrix. Rows with NaN (from warmup periods) are dropped.
        """
        close = bars["close"]
        volume = bars["volume"]
        w = self._zscore_window

        features = pd.DataFrame(index=bars.index)

        # --- Returns at multiple horizons ---
        features["ret_1"] = rolling_zscore(log_returns(close, 1), w)
        features["ret_5"] = rolling_zscore(log_returns(close, 5), w)
        features["ret_20"] = rolling_zscore(log_returns(close, 20), w)

        # --- Volatility structure ---
        features["rvol_20"] = rolling_zscore(realized_volatility(close, 20), w)
        features["vol_ratio"] = rolling_zscore(volatility_ratio(close, 5, 20), w)

        # --- Volume dynamics ---
        features["vol_norm"] = rolling_zscore(normalized_volume(volume, 50), w)
        features["vol_trend"] = rolling_zscore(volume_trend(volume, 10), w)

        # --- Trend strength ---
        features["adx_14"] = rolling_zscore(adx(bars, 14), w)
        features["sma50_slope"] = rolling_zscore(sma_slope(close, 50, 10), w)

        # --- Mean reversion signals (as regime classifiers, NOT trade signals) ---
        features["rsi14_zscore"] = rolling_zscore(rsi(close, 14), w)
        features["dist_sma200"] = rolling_zscore(distance_from_sma(close, 200), w)

        # --- Momentum ---
        features["roc_10"] = rolling_zscore(rate_of_change(close, 10), w)
        features["roc_20"] = rolling_zscore(rate_of_change(close, 20), w)

        # --- Range ---
        features["natr"] = rolling_zscore(normalized_atr(bars, 14), w)

        return features.dropna()

    def compute_strategy_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute features needed by strategy modules (not z-scored).

        These are raw indicator values for strategy-level entry/exit logic,
        separate from the z-scored HMM features above.

        Parameters
        ----------
        bars : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Strategy feature matrix with raw indicator values.
        """
        close = bars["close"]
        features = pd.DataFrame(index=bars.index)

        features["sma_10"] = close.rolling(10).mean()
        features["sma_50"] = close.rolling(50).mean()
        features["sma_200"] = close.rolling(200).mean()
        features["rsi_14"] = rsi(close, 14)
        features["atr_14"] = atr(bars, 14)
        features["bb_mid"] = close.rolling(20).mean()
        features["bb_std"] = close.rolling(20).std()
        features["bb_upper"] = features["bb_mid"] + 2.0 * features["bb_std"]
        features["bb_lower"] = features["bb_mid"] - 2.0 * features["bb_std"]

        return features
