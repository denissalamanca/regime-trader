"""Hidden Markov Model engine for market regime detection.

DESIGN PHILOSOPHY:
==================
The HMM is a REGIME CLASSIFIER, not a price predictor. It characterizes the
current market environment (bull, bear, crash, euphoria, transition, etc.)
so the system can adapt risk parameters and strategy selection accordingly.
It does NOT generate buy/sell signals directly.

CRITICAL — NO LOOK-AHEAD BIAS:
===============================
This module uses FILTERED INFERENCE (forward algorithm only) for real-time
regime detection. The Viterbi algorithm (model.predict()) processes the entire
sequence and revises past state assignments based on future data — that is
look-ahead bias and will make backtests look artificially good.

The only method that should be used for live trading and backtesting is
predict_regime_filtered(), which computes P(state_t | observations_1:t)
using ONLY past and present data.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime label schemes — keyed by number of regimes
# ---------------------------------------------------------------------------

REGIME_LABEL_SCHEMES: dict[int, list[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}

# Default strategy/risk mappings per regime label
_REGIME_DEFAULTS: dict[str, dict] = {
    "CRASH":       {"strategy": "defensive",    "max_leverage": 0.25, "max_position_pct": 0.05, "min_confidence": 0.50},
    "STRONG_BEAR": {"strategy": "defensive",    "max_leverage": 0.40, "max_position_pct": 0.08, "min_confidence": 0.55},
    "BEAR":        {"strategy": "mean_revert",  "max_leverage": 0.50, "max_position_pct": 0.10, "min_confidence": 0.55},
    "WEAK_BEAR":   {"strategy": "mean_revert",  "max_leverage": 0.60, "max_position_pct": 0.12, "min_confidence": 0.55},
    "NEUTRAL":     {"strategy": "trend_follow", "max_leverage": 0.80, "max_position_pct": 0.15, "min_confidence": 0.55},
    "WEAK_BULL":   {"strategy": "trend_follow", "max_leverage": 0.90, "max_position_pct": 0.18, "min_confidence": 0.55},
    "BULL":        {"strategy": "trend_follow", "max_leverage": 1.00, "max_position_pct": 0.20, "min_confidence": 0.55},
    "STRONG_BULL": {"strategy": "aggressive",   "max_leverage": 1.00, "max_position_pct": 0.20, "min_confidence": 0.60},
    "EUPHORIA":    {"strategy": "aggressive",   "max_leverage": 0.70, "max_position_pct": 0.12, "min_confidence": 0.65},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeInfo:
    """Metadata for a single regime learned by the HMM."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float = 0.55


@dataclass
class RegimeState:
    """Current regime detection result from filtered inference."""

    label: str                       # Human-readable label (e.g. "BULL", "CRASH")
    state_id: int                    # Sorted regime index
    probability: float               # Posterior probability of the assigned state
    state_probabilities: np.ndarray  # Full posterior distribution over all states
    timestamp: pd.Timestamp
    is_confirmed: bool = True        # False during the stability-filter transition period
    consecutive_bars: int = 1        # How many bars the current regime has persisted


@dataclass
class TrainingMetrics:
    """Diagnostics from a training run."""

    n_regimes_selected: int
    bic_scores: dict[int, float]
    log_likelihood: float
    converged: bool
    n_iterations: int
    n_samples: int
    n_features: int
    training_time_seconds: float
    timestamp: pd.Timestamp


# ---------------------------------------------------------------------------
# HMM Engine
# ---------------------------------------------------------------------------

class HMMEngine:
    """Fits and predicts market regimes using a Gaussian HMM with model selection.

    Key design decisions:
    - Model selection via BIC over [3..7] components avoids hardcoding regime count.
    - Regimes are labeled post-training by sorting on mean return.
    - Prediction uses filtered (forward-only) inference to prevent look-ahead bias.
    - A stability filter requires 3 consecutive bars before confirming regime changes.

    Parameters
    ----------
    config : dict
        HMM configuration. Expected keys:
        - covariance_type (str): default "full"
        - n_iter (int): max EM iterations, default 200
        - random_state (int): RNG seed, default 42
        - min_train_bars (int): minimum training samples, default 504
        - refit_interval (int): bars between refits, default 5 (weekly)
        - n_candidates (list[int]): regime counts to test, default [3,4,5,6,7]
        - n_init (int): random restarts per candidate, default 10
        - stability_bars (int): bars to confirm regime change, default 3
        - flicker_window (int): lookback for flicker rate, default 20
        - flicker_threshold (int): max changes before forcing uncertainty, default 4
    """

    def __init__(self, config: dict) -> None:
        self._config = config

        # Model selection parameters
        self._n_candidates: list[int] = config.get("n_candidates", [3, 4, 5, 6, 7])
        self._n_init: int = config.get("n_init", 10)
        self._cov_type: str = config.get("covariance_type", "full")
        self._n_iter: int = config.get("n_iter", 200)
        self._random_state: int = config.get("random_state", 42)
        self._min_train_bars: int = config.get("min_train_bars", 504)
        self._refit_interval: int = config.get("refit_interval", 5)

        # Stability filter
        self._stability_bars: int = config.get("stability_bars", 3)
        self._flicker_window: int = config.get("flicker_window", 20)
        self._flicker_threshold: int = config.get("flicker_threshold", 4)

        # Internal state
        self._model: Optional[GaussianHMM] = None
        self._n_regimes: int = 0
        self._regime_infos: list[RegimeInfo] = []
        self._state_label_map: dict[int, str] = {}
        self._sorted_state_indices: np.ndarray = np.array([])  # maps sorted_idx -> original_idx
        self._bars_since_refit: int = 0
        self._is_fitted: bool = False
        self._last_training_metrics: Optional[TrainingMetrics] = None

        # Regime tracking for stability filter
        self._current_regime_id: Optional[int] = None
        self._confirmed_regime_id: Optional[int] = None
        self._consecutive_bars: int = 0
        self._regime_history: list[int] = []

        # Forward algorithm cache
        self._cached_log_alpha: Optional[np.ndarray] = None  # log forward variables
        self._cached_length: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted at least once."""
        return self._is_fitted

    @property
    def n_regimes(self) -> int:
        """Number of regimes selected by BIC."""
        return self._n_regimes

    @property
    def state_label_map(self) -> dict[int, str]:
        """Mapping from sorted regime index to human-readable label."""
        return dict(self._state_label_map)

    @property
    def regime_infos(self) -> list[RegimeInfo]:
        """Metadata for each regime, sorted by mean return."""
        return list(self._regime_infos)

    @property
    def last_training_metrics(self) -> Optional[TrainingMetrics]:
        return self._last_training_metrics

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> TrainingMetrics:
        """Fit the HMM with automatic model selection via BIC.

        Tests n_components in self._n_candidates, selects the model with
        the lowest BIC. For each candidate, runs n_init random restarts
        and keeps the best.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (already z-scored). Rows are time steps, columns
            are features. Index should be a DatetimeIndex.

        Returns
        -------
        TrainingMetrics
            Diagnostics from the selected model.

        Raises
        ------
        ValueError
            If fewer than min_train_bars rows are provided.
        """
        n_samples, n_features = features.shape
        if n_samples < self._min_train_bars:
            raise ValueError(
                f"Need at least {self._min_train_bars} bars for training, "
                f"got {n_samples}"
            )

        X = features.values
        t0 = time.monotonic()

        logger.info(
            "Starting HMM model selection: candidates=%s, n_init=%d, "
            "n_samples=%d, n_features=%d",
            self._n_candidates, self._n_init, n_samples, n_features,
        )

        best_bic = np.inf
        best_model: Optional[GaussianHMM] = None
        best_n: int = self._n_candidates[0]
        bic_scores: dict[int, float] = {}

        for n_comp in self._n_candidates:
            model, ll = self._fit_single(X, n_comp)
            if model is None:
                logger.warning("All %d inits failed for n_components=%d", self._n_init, n_comp)
                bic_scores[n_comp] = np.inf
                continue

            n_params = self._count_params(n_comp, n_features)
            bic = -2.0 * ll + n_params * np.log(n_samples)
            bic_scores[n_comp] = bic

            logger.info(
                "  n_components=%d: BIC=%.2f, LL=%.2f, n_params=%d",
                n_comp, bic, ll, n_params,
            )

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n_comp

        if best_model is None:
            raise RuntimeError("All model candidates failed to converge.")

        self._model = best_model
        self._n_regimes = best_n

        logger.info("Selected model: n_regimes=%d, BIC=%.2f", best_n, best_bic)

        # Sort regimes by mean return and assign labels
        self._sort_and_label_regimes(features)

        self._is_fitted = True
        self._bars_since_refit = 0

        # Reset forward algorithm cache on refit
        self._cached_log_alpha = None
        self._cached_length = 0

        elapsed = time.monotonic() - t0
        metrics = TrainingMetrics(
            n_regimes_selected=best_n,
            bic_scores=bic_scores,
            log_likelihood=best_model.score(X),
            converged=best_model.monitor_.converged,
            n_iterations=best_model.monitor_.iter,
            n_samples=n_samples,
            n_features=n_features,
            training_time_seconds=elapsed,
            timestamp=features.index[-1] if isinstance(features.index, pd.DatetimeIndex) else pd.Timestamp.now(),
        )
        self._last_training_metrics = metrics

        logger.info(
            "Training complete in %.1fs: converged=%s, iterations=%d, LL=%.2f",
            elapsed, metrics.converged, metrics.n_iterations, metrics.log_likelihood,
        )

        return metrics

    def _fit_single(self, X: np.ndarray, n_components: int) -> tuple[Optional[GaussianHMM], float]:
        """Fit a single HMM with multiple random restarts, return best.

        Returns
        -------
        tuple
            (best_model, best_log_likelihood) or (None, -inf) if all fail.
        """
        best_model: Optional[GaussianHMM] = None
        best_ll = -np.inf

        for i in range(self._n_init):
            seed = self._random_state + i * 1000 + n_components
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=self._cov_type,
                n_iter=self._n_iter,
                random_state=seed,
                verbose=False,
            )
            try:
                model.fit(X)
                ll = model.score(X)
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception as e:
                logger.debug(
                    "Init %d/%d failed for n_components=%d: %s",
                    i + 1, self._n_init, n_components, e,
                )
                continue

        return best_model, best_ll

    def _count_params(self, n_components: int, n_features: int) -> int:
        """Count free parameters for BIC computation.

        Parameters for GaussianHMM:
        - Initial state probabilities: n_components - 1
        - Transition matrix: n_components * (n_components - 1)
        - Means: n_components * n_features
        - Covariances: depends on covariance_type
        """
        n = n_components
        k = n_features

        n_params = (n - 1)  # startprob
        n_params += n * (n - 1)  # transmat

        n_params += n * k  # means

        if self._cov_type == "full":
            n_params += n * k * (k + 1) // 2
        elif self._cov_type == "diag":
            n_params += n * k
        elif self._cov_type == "spherical":
            n_params += n
        elif self._cov_type == "tied":
            n_params += k * (k + 1) // 2

        return n_params

    def _sort_and_label_regimes(self, features: pd.DataFrame) -> None:
        """Sort HMM states by mean return and assign human-readable labels.

        The first feature column is assumed to be short-term returns (ret_1).
        States are sorted ascending by their mean on this feature so that:
        - Index 0 = lowest mean return (CRASH / BEAR)
        - Index N-1 = highest mean return (EUPHORIA / BULL)
        """
        means = self._model.means_  # shape: (n_components, n_features)
        # Sort by mean of the first feature (ret_1 — short-term returns z-score).
        # Labels reflect return characteristics (BEAR = low return, BULL = high).
        # The StrategyOrchestrator separately sorts by volatility for allocation.
        return_means = means[:, 0]
        self._sorted_state_indices = np.argsort(return_means)  # ascending

        # Get label scheme
        n = self._n_regimes
        if n in REGIME_LABEL_SCHEMES:
            labels = REGIME_LABEL_SCHEMES[n]
        else:
            # Fallback: generate numbered labels sorted by return
            labels = [f"REGIME_{i}" for i in range(n)]

        self._state_label_map = {}
        self._regime_infos = []

        for sorted_idx, original_idx in enumerate(self._sorted_state_indices):
            label = labels[sorted_idx]
            self._state_label_map[sorted_idx] = label

            # Extract expected return and volatility from learned parameters
            mean_vec = means[original_idx]
            # hmmlearn 0.3+ stores covars_ as (n_components, n_features, n_features)
            # for full, diag, and tied types. Access variance of first feature.
            covars = self._model.covars_
            if covars.ndim == 3 and covars.shape[0] >= n:
                var_ret = float(covars[original_idx][0, 0])
            elif covars.ndim == 2:
                var_ret = float(covars[0, 0])
            else:
                var_ret = float(covars[original_idx]) if covars.ndim == 1 else 1.0

            defaults = _REGIME_DEFAULTS.get(label, _REGIME_DEFAULTS["NEUTRAL"])

            info = RegimeInfo(
                regime_id=sorted_idx,
                regime_name=label,
                expected_return=float(mean_vec[0]),
                expected_volatility=float(np.sqrt(var_ret)),
                recommended_strategy_type=defaults["strategy"],
                max_leverage_allowed=defaults["max_leverage"],
                max_position_size_pct=defaults["max_position_pct"],
                min_confidence_to_act=defaults["min_confidence"],
            )
            self._regime_infos.append(info)

        logger.info("Regime labels assigned: %s", self._state_label_map)
        for info in self._regime_infos:
            logger.info(
                "  %s: E[ret_zscore]=%.3f, E[vol]=%.3f, strategy=%s",
                info.regime_name, info.expected_return,
                info.expected_volatility, info.recommended_strategy_type,
            )

    # ------------------------------------------------------------------
    # Regime detection — FILTERED INFERENCE (no look-ahead bias)
    # ------------------------------------------------------------------

    def predict_regime_filtered(self, features: pd.DataFrame) -> RegimeState:
        """Detect the current regime using forward-only (filtered) inference.

        *** THIS IS THE ONLY METHOD THAT SHOULD BE USED FOR LIVE TRADING
        AND BACKTESTING. ***

        Computes P(state_t | observations_1:t) using the forward algorithm.
        At time T, only observations [0:T] are used. No future data.

        For efficiency, the forward pass is computed incrementally: if we
        previously computed alpha for observations [0:T-1], we extend it
        by one step rather than recomputing from scratch.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix up to (and including) the current time step.
            Must have the same columns used during training.

        Returns
        -------
        RegimeState
            Current regime with filtered posterior probabilities.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = features.values
        T = X.shape[0]

        # Compute per-observation log likelihoods for the full sequence
        # using the model's emission parameters
        framelogprob = self._compute_emission_logprob(X)

        # Run or extend the forward algorithm
        log_alpha = self._forward_pass(framelogprob, T)

        # Extract filtered posterior at the last time step
        log_posterior_last = log_alpha[-1] - logsumexp(log_alpha[-1])
        posterior = np.exp(log_posterior_last)

        # Map from original HMM state indices to sorted regime indices
        sorted_posterior = self._to_sorted_posterior(posterior)

        sorted_state_id = int(np.argmax(sorted_posterior))
        label = self._state_label_map[sorted_state_id]
        prob = float(sorted_posterior[sorted_state_id])

        timestamp = (
            features.index[-1]
            if isinstance(features.index, pd.DatetimeIndex)
            else pd.Timestamp.now()
        )

        # Update stability tracking
        self._bars_since_refit += 1
        is_confirmed, consecutive = self._update_stability(sorted_state_id)

        state = RegimeState(
            label=label,
            state_id=sorted_state_id,
            probability=prob,
            state_probabilities=sorted_posterior,
            timestamp=timestamp,
            is_confirmed=is_confirmed,
            consecutive_bars=consecutive,
        )

        if not is_confirmed:
            logger.info(
                "Regime UNCONFIRMED at %s: raw=%s (prob=%.3f), confirmed=%s, "
                "consecutive=%d/%d",
                timestamp, label, prob,
                self._state_label_map.get(self._confirmed_regime_id, "NONE"),
                consecutive, self._stability_bars,
            )
        elif self._consecutive_bars == self._stability_bars:
            # Just became confirmed
            logger.warning(
                "REGIME CHANGE CONFIRMED at %s: %s (prob=%.3f) after %d bars",
                timestamp, label, prob, self._stability_bars,
            )

        return state

    def predict_regime_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return the filtered posterior probability distribution over all regimes.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix up to current time.

        Returns
        -------
        np.ndarray
            Shape (n_regimes,) probability distribution (sorted regime order).
        """
        state = self.predict_regime_filtered(features)
        return state.state_probabilities

    def predict_regime_viterbi(self, features: pd.DataFrame) -> np.ndarray:
        """*** POST-HOC ANALYSIS ONLY — DO NOT USE FOR LIVE TRADING OR BACKTESTING ***

        Runs the Viterbi algorithm on the full sequence to find the most likely
        state path. This uses future data to revise past assignments and is
        ONLY appropriate for after-the-fact analysis of historical regimes.

        Parameters
        ----------
        features : pd.DataFrame
            Complete feature matrix.

        Returns
        -------
        np.ndarray
            Array of sorted regime IDs for each time step.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        raw_states = self._model.predict(features.values)
        # Map raw states to sorted indices
        raw_to_sorted = self._build_raw_to_sorted_map()
        return np.array([raw_to_sorted[s] for s in raw_states])

    def _compute_emission_logprob(self, X: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities for each observation under each state.

        Parameters
        ----------
        X : np.ndarray
            Shape (T, n_features) observation matrix.

        Returns
        -------
        np.ndarray
            Shape (T, n_components) log P(x_t | state_k).
        """
        return self._model._compute_log_likelihood(X)

    def _forward_pass(self, framelogprob: np.ndarray, T: int) -> np.ndarray:
        """Run or incrementally extend the forward algorithm.

        If we have cached forward variables for [0:T_prev] and the new
        sequence is an extension (T > T_prev), we only compute the new steps.

        Parameters
        ----------
        framelogprob : np.ndarray
            Shape (T, n_components) log emission probabilities.
        T : int
            Length of the current sequence.

        Returns
        -------
        np.ndarray
            Shape (T, n_components) log forward variables.
        """
        log_startprob = np.log(self._model.startprob_ + 1e-300)
        log_transmat = np.log(self._model.transmat_ + 1e-300)
        n = self._n_regimes

        # Check if we can extend the cache
        if (
            self._cached_log_alpha is not None
            and self._cached_length > 0
            and self._cached_length <= T
        ):
            # Extend from cached position
            start = self._cached_length
            log_alpha = np.full((T, n), -np.inf)
            log_alpha[:start] = self._cached_log_alpha[:start]

            for t in range(start, T):
                for j in range(n):
                    log_alpha[t, j] = (
                        logsumexp(log_alpha[t - 1] + log_transmat[:, j])
                        + framelogprob[t, j]
                    )
        else:
            # Full forward pass from scratch
            log_alpha = np.full((T, n), -np.inf)
            log_alpha[0] = log_startprob + framelogprob[0]

            for t in range(1, T):
                for j in range(n):
                    log_alpha[t, j] = (
                        logsumexp(log_alpha[t - 1] + log_transmat[:, j])
                        + framelogprob[t, j]
                    )

        # Update cache
        self._cached_log_alpha = log_alpha
        self._cached_length = T

        return log_alpha

    def _to_sorted_posterior(self, posterior: np.ndarray) -> np.ndarray:
        """Map posterior from raw HMM state order to sorted regime order.

        Parameters
        ----------
        posterior : np.ndarray
            Shape (n_components,) in raw HMM state order.

        Returns
        -------
        np.ndarray
            Shape (n_components,) in sorted regime order.
        """
        sorted_posterior = np.zeros(self._n_regimes)
        for sorted_idx, original_idx in enumerate(self._sorted_state_indices):
            sorted_posterior[sorted_idx] = posterior[original_idx]
        return sorted_posterior

    def _build_raw_to_sorted_map(self) -> dict[int, int]:
        """Build a mapping from raw HMM state index to sorted regime index."""
        return {
            int(orig): sorted_idx
            for sorted_idx, orig in enumerate(self._sorted_state_indices)
        }

    # ------------------------------------------------------------------
    # Stability filter
    # ------------------------------------------------------------------

    def _update_stability(self, raw_regime_id: int) -> tuple[bool, int]:
        """Update the regime stability tracker.

        A regime change is only "confirmed" after the new regime persists
        for stability_bars consecutive observations.

        Parameters
        ----------
        raw_regime_id : int
            The sorted regime ID detected at the current time step.

        Returns
        -------
        tuple[bool, int]
            (is_confirmed, consecutive_bars_in_current_raw_regime)
        """
        self._regime_history.append(raw_regime_id)

        if raw_regime_id == self._current_regime_id:
            self._consecutive_bars += 1
        else:
            if self._current_regime_id is not None:
                logger.info(
                    "Regime change detected (UNCONFIRMED): %s -> %s",
                    self._state_label_map.get(self._current_regime_id, "NONE"),
                    self._state_label_map.get(raw_regime_id, "NONE"),
                )
            self._current_regime_id = raw_regime_id
            self._consecutive_bars = 1

        if self._consecutive_bars >= self._stability_bars:
            self._confirmed_regime_id = self._current_regime_id
            return True, self._consecutive_bars

        # Not yet confirmed — keep previous confirmed regime
        if self._confirmed_regime_id is None:
            # First ever detection — confirm immediately
            self._confirmed_regime_id = raw_regime_id
            return True, self._consecutive_bars

        return False, self._consecutive_bars

    def get_confirmed_regime(self) -> Optional[int]:
        """Return the current confirmed (stable) regime ID, or None."""
        return self._confirmed_regime_id

    def get_confirmed_regime_label(self) -> Optional[str]:
        """Return the confirmed regime label, or None if not yet set."""
        if self._confirmed_regime_id is None:
            return None
        return self._state_label_map.get(self._confirmed_regime_id)

    def get_regime_stability(self) -> int:
        """Return how many consecutive bars the current raw regime has persisted."""
        return self._consecutive_bars

    def detect_regime_change(self) -> bool:
        """Return True if the regime just became confirmed (stability threshold met).

        A regime change is detected when:
        1. The raw regime differs from the previously confirmed regime, AND
        2. It has persisted for exactly stability_bars consecutive bars.
        """
        return (
            self._consecutive_bars == self._stability_bars
            and self._confirmed_regime_id == self._current_regime_id
            and len(self._regime_history) > self._stability_bars
        )

    def get_regime_flicker_rate(self) -> float:
        """Compute the regime change rate over the last flicker_window periods.

        High flicker rate = HMM is uncertain about the regime = signal to
        reduce exposure.

        Returns
        -------
        float
            Number of regime changes in the last flicker_window periods.
        """
        window = self._flicker_window
        if len(self._regime_history) < 2:
            return 0.0

        recent = self._regime_history[-window:]
        changes = sum(
            1 for i in range(1, len(recent)) if recent[i] != recent[i - 1]
        )
        return float(changes)

    def is_flickering(self) -> bool:
        """Return True if flicker rate exceeds the threshold.

        When flickering, the system should force uncertainty mode and
        reduce exposure regardless of individual regime probabilities.
        """
        return self.get_regime_flicker_rate() >= self._flicker_threshold

    # ------------------------------------------------------------------
    # Refit scheduling
    # ------------------------------------------------------------------

    def should_refit(self) -> bool:
        """Check whether the model should be refitted based on bar count."""
        return self._bars_since_refit >= self._refit_interval

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def get_transition_matrix(self) -> np.ndarray:
        """Return the learned transition matrix in sorted regime order.

        Returns
        -------
        np.ndarray
            Shape (n_regimes, n_regimes). Entry [i, j] = P(regime_j | regime_i).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing transition matrix.")

        raw = self._model.transmat_
        n = self._n_regimes
        sorted_mat = np.zeros((n, n))

        for si in range(n):
            for sj in range(n):
                sorted_mat[si, sj] = raw[
                    self._sorted_state_indices[si],
                    self._sorted_state_indices[sj],
                ]

        return sorted_mat

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the fitted model and metadata to disk.

        Parameters
        ----------
        path : str or Path
            File path for the pickle.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        state = {
            "model": self._model,
            "n_regimes": self._n_regimes,
            "sorted_state_indices": self._sorted_state_indices,
            "state_label_map": self._state_label_map,
            "regime_infos": self._regime_infos,
            "config": self._config,
            "training_metrics": self._last_training_metrics,
            "timestamp": pd.Timestamp.now(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved model.

        Parameters
        ----------
        path : str or Path
            File path to the pickle.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self._model = state["model"]
        self._n_regimes = state["n_regimes"]
        self._sorted_state_indices = state["sorted_state_indices"]
        self._state_label_map = state["state_label_map"]
        self._regime_infos = state["regime_infos"]
        self._last_training_metrics = state.get("training_metrics")
        self._is_fitted = True

        # Reset caches
        self._cached_log_alpha = None
        self._cached_length = 0

        logger.info(
            "Model loaded from %s: n_regimes=%d, labels=%s",
            path, self._n_regimes, self._state_label_map,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_regime_info(self, regime_id: int) -> RegimeInfo:
        """Get metadata for a specific regime.

        Parameters
        ----------
        regime_id : int
            Sorted regime index.

        Returns
        -------
        RegimeInfo
        """
        return self._regime_infos[regime_id]

    def get_regime_info_by_label(self, label: str) -> Optional[RegimeInfo]:
        """Look up regime info by its label name.

        Parameters
        ----------
        label : str
            Regime label (e.g. "BULL").

        Returns
        -------
        RegimeInfo or None
        """
        for info in self._regime_infos:
            if info.regime_name == label:
                return info
        return None

    def reset_tracking(self) -> None:
        """Reset regime tracking state (for backtesting new windows)."""
        self._current_regime_id = None
        self._confirmed_regime_id = None
        self._consecutive_bars = 0
        self._regime_history = []
        self._cached_log_alpha = None
        self._cached_length = 0
        self._bars_since_refit = 0
