"""Benchmark every strategy against SPY 2018-2026 and print a comparison.

Each strategy runs through the same walk-forward backtester with the same
HMM regime predictions and the same risk manager. The only thing that
changes between runs is which strategy class is bound to every regime
(via the StrategyOrchestrator's ``strategy_class`` config override).

Single-asset strategies are tested on SPY only.
Multi-asset strategies (MomentumRotation, DefensiveLongShort) get the full
6-symbol universe.
Pair strategies (StaticPairsZScore, CorrelationRegimeAllocation) are tested
on SPY/IWM with HMM features computed on SPY.

Results are printed as a markdown-style comparison table, sortable by total
return.

Run from the project root:
    python3 scripts/bench_strategies.py
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.backtester import WalkForwardBacktester
from backtest.performance import PerformanceAnalyzer
from core.hmm_engine import HMMEngine
from core.regime_strategies import (
    BaseStrategy, HighVolDefensiveStrategy, LowVolBullStrategy,
    MidVolCautiousStrategy,
)
from core.strategies import (
    CorrelationRegimeAllocation, DefensiveLongShort, MeanReversionLowVol,
    MomentumRotation, PureRegimeAllocation, StaticPairsZScore,
    TrendFollowingRegimeFilter, VolatilityBreakout,
)
from data.feature_engineering import FeatureEngineer

# Quiet the noisier loggers — we only want the comparison output.
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Bench config
# ---------------------------------------------------------------------------

BACKTEST_CONFIG = {
    "hmm": {
        "n_candidates": [3, 4, 5],
        "zscore_window": 126,
        "min_train_bars": 252,
    },
    "backtest": {
        "initial_capital": 100_000,
        "slippage_pct": 0.0005,
        "fill_delay_bars": 1,
        "risk_free_rate": 0.045,
        "walk_forward": {
            "train_window": 252,
            "test_window": 63,
            "step_size": 126,
        },
    },
    "risk": {},
    "strategy": {},
}

START = "2018-01-01"
END = "2026-04-27"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
SINGLE_SYMBOLS = ["SPY"]
MULTI_SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL"]
PAIR_SYMBOLS = ["SPY", "IWM"]


@dataclass
class BenchResult:
    name: str
    universe: str
    total_return: float
    annualized: float
    sharpe: float
    max_dd: float
    n_trades: int
    win_rate: float
    elapsed_s: float


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Load bars from the project's pre-populated cache."""
    path = CACHE_DIR / f"{symbol}_1Day.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]
    return df if len(df) >= 50 else None


def load_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load bars for a list of symbols. Skips any that aren't cached."""
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        bars = load_bars(s)
        if bars is not None:
            out[s] = bars
    return out


# ---------------------------------------------------------------------------
# Run a single strategy through the backtester
# ---------------------------------------------------------------------------

def run_one(
    name: str,
    strategy_cls: Optional[type[BaseStrategy]],
    symbols: list[str],
    universe_label: str,
    pairs: Optional[list[list[str]]] = None,
    strategy_params: Optional[dict] = None,
) -> Optional[BenchResult]:
    """Run a backtest with `strategy_cls` bound to every regime.

    Returns None if any required data is missing.
    """
    bars = load_universe(symbols)
    if not bars or symbols[0] not in bars:
        return None

    # Compute HMM features on the reference symbol (first in list).
    fe = FeatureEngineer(BACKTEST_CONFIG["hmm"])
    hmm_features = fe.compute_hmm_features(bars[symbols[0]])
    if len(hmm_features) < 300:
        return None

    # Build a config copy with this strategy as the orchestrator override.
    cfg = {**BACKTEST_CONFIG, "strategy": {**(strategy_params or {})}}
    if strategy_cls is not None:
        cfg["strategy"]["strategy_class"] = strategy_cls
    if pairs:
        cfg["strategy"]["pairs"] = pairs

    bt = WalkForwardBacktester(cfg)
    t0 = time.monotonic()
    result = bt.run(bars, hmm_features)
    elapsed = time.monotonic() - t0

    m = result.metrics
    return BenchResult(
        name=name, universe=universe_label,
        total_return=m.total_return, annualized=m.annualized_return,
        sharpe=m.sharpe_ratio, max_dd=m.max_drawdown,
        n_trades=m.total_trades, win_rate=m.win_rate, elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Main bench
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\nBenchmarking strategies on {START} → {END}")
    print(f"Single-asset universe:  {SINGLE_SYMBOLS}")
    print(f"Multi-asset universe:   {MULTI_SYMBOLS}")
    print(f"Pair universe:          {PAIR_SYMBOLS}\n")

    results: list[BenchResult] = []

    # --- Baseline: the original 3 archetypes (default vol-rank mapping) ---
    print("Running: baseline (3-archetype vol-rank mapping) ...")
    r = run_one(
        "BASELINE (vol-rank)", strategy_cls=None,
        symbols=SINGLE_SYMBOLS, universe_label="SPY",
    )
    if r:
        results.append(r)

    # --- Single-asset library strategies on SPY ---
    single_asset_strategies: list[tuple[str, type[BaseStrategy], dict]] = [
        ("PureRegimeAllocation",        PureRegimeAllocation,        {"n_watchlist_hint": 1, "low_vol_gross": 1.0, "mid_vol_gross": 0.50}),
        ("PureRegimeAllocation (tuned)", PureRegimeAllocation,       {"n_watchlist_hint": 1, "low_vol_gross": 1.0, "mid_vol_gross": 0.85, "low_vol_leverage": 1.25, "mid_vol_leverage": 1.0}),
        ("TrendFollowingRegimeFilter",  TrendFollowingRegimeFilter,  {"ma_period": 200}),
        ("MeanReversionLowVol",         MeanReversionLowVol,         {}),
        ("VolatilityBreakout",          VolatilityBreakout,          {}),
        ("VolatilityBreakout (tuned)",  VolatilityBreakout,          {"breakout_lookback": 10, "max_position_size": 0.50, "target_risk_pct": 0.015}),
        # Re-run the 3 archetypes individually (only-vol-rank) for reference
        ("LowVolBull (forced)",         LowVolBullStrategy,          {}),
        ("MidVolCautious (forced)",     MidVolCautiousStrategy,      {}),
        ("HighVolDefensive (forced)",   HighVolDefensiveStrategy,    {}),
    ]
    for name, cls, params in single_asset_strategies:
        print(f"Running: {name} on SPY ...")
        r = run_one(
            name, strategy_cls=cls,
            symbols=SINGLE_SYMBOLS, universe_label="SPY",
            strategy_params=params,
        )
        if r:
            results.append(r)

    # --- Multi-asset strategies on the 6-symbol universe ---
    multi_asset_strategies: list[tuple[str, type[BaseStrategy], dict]] = [
        ("MomentumRotation",            MomentumRotation,            {"top_n": 3, "top_n_high_vol": 1}),
        ("DefensiveLongShort",          DefensiveLongShort,          {"top_n": 2, "bottom_m": 2}),
        ("PureRegimeAllocation (multi)", PureRegimeAllocation,       {"n_watchlist_hint": 6}),
    ]
    for name, cls, params in multi_asset_strategies:
        print(f"Running: {name} on multi-asset universe ...")
        r = run_one(
            name, strategy_cls=cls,
            symbols=MULTI_SYMBOLS, universe_label="6-symbol",
            strategy_params=params,
        )
        if r:
            results.append(r)

    # --- Pair strategies on SPY/IWM ---
    pair_strategies: list[tuple[str, type[BaseStrategy], dict]] = [
        ("StaticPairsZScore (SPY/IWM)",       StaticPairsZScore,        {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "pair_pct": 0.30}),
        ("CorrelationRegime (SPY/IWM)",       CorrelationRegimeAllocation, {"entry_corr": 0.5, "exit_corr": 0.3, "entry_z": 1.5, "pair_pct": 0.30}),
    ]
    for name, cls, params in pair_strategies:
        print(f"Running: {name} ...")
        r = run_one(
            name, strategy_cls=cls,
            symbols=PAIR_SYMBOLS, universe_label="SPY/IWM",
            pairs=[["SPY", "IWM"]],
            strategy_params=params,
        )
        if r:
            results.append(r)

    # --- Print comparison table ---
    print("\n" + "=" * 110)
    print(f"{'STRATEGY':<35} {'UNIVERSE':<12} "
          f"{'TOTAL':>8} {'ANN':>7} {'SHARPE':>7} "
          f"{'MAX_DD':>8} {'TRADES':>7} {'WIN%':>6} {'TIME':>6}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: x.total_return, reverse=True):
        print(f"{r.name:<35} {r.universe:<12} "
              f"{r.total_return:>7.1%} {r.annualized:>6.1%} {r.sharpe:>7.2f} "
              f"{r.max_dd:>7.1%} {r.n_trades:>7d} "
              f"{r.win_rate:>5.0%} {r.elapsed_s:>5.1f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()
