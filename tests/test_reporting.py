"""Tests for the Phase 1 reporting/metrics fixes (C1-C5).

These cover the benchmark-comparison and metric-labeling corrections:
  C1 - comparison serializes to a tidy frame (so it can be written to CSV)
  C2 - benchmarks are computed over the strategy's ACTUAL traded span
  C3 - the random-entry benchmark is a real median-outcome path (not a
       volatility-cancelling mean of many paths)
  C4 - passive benchmark rows render "-" for trades, with a footnote
  C5 - regime/confidence "PnL/sigma" use one consistent mean/std formula
       (not two different sqrt(252) variants, and not a real Sharpe)
"""

import numpy as np
import pandas as pd
import pytest

from backtest.performance import PerformanceAnalyzer


_EMPTY_TRADES = pd.DataFrame(columns=[
    "pnl", "entry_date", "exit_date", "entry_price", "exit_price",
    "qty", "regime", "confidence",
])


def _make_bars(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    rng = np.random.RandomState(seed)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx))), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def _strategy_equity(start: str, periods: int) -> pd.Series:
    idx = pd.bdate_range(start, periods=periods)
    return pd.Series(np.linspace(100_000, 120_000, periods), index=idx, name="equity")


def test_compare_benchmarks_aligns_to_strategy_span():
    """C2: benchmarks must reflect only the strategy's traded window."""
    bars = _make_bars("2018-01-01", "2024-12-31")
    strat_idx = pd.bdate_range("2020-01-15", "2022-12-31")
    strat_eq = pd.Series(
        np.linspace(100_000, 120_000, len(strat_idx)), index=strat_idx, name="equity")

    comp = PerformanceAnalyzer().compare_benchmarks(
        strat_eq, _EMPTY_TRADES, bars, 100_000, n_random_runs=5)

    close = bars["close"]
    bh_full = close.iloc[-1] / close.iloc[0] - 1
    span = close[(close.index >= strat_idx[0]) & (close.index <= strat_idx[-1])]
    bh_span = span.iloc[-1] / span.iloc[0] - 1

    # Buy & hold matches the strategy-span return, not the full-history return.
    assert comp.buy_and_hold.total_return == pytest.approx(bh_span, rel=1e-6)
    assert abs(comp.buy_and_hold.total_return - bh_full) > 1e-3


def test_random_entry_is_a_real_path_with_volatility():
    """C3: random benchmark is a single median-outcome path, not a flat mean."""
    bars = _make_bars("2020-01-01", "2021-12-31", seed=1)
    eq = PerformanceAnalyzer()._random_entry_equity(
        bars, 100_000, pd.DataFrame({"pnl": [1] * 12}), n_runs=21)

    rets = eq.pct_change().dropna()
    assert len(eq) == len(bars)
    assert rets.std() > 0.0  # not a volatility-cancelled flat line
    sharpe = (rets - 0.045 / 252).mean() / rets.std() * np.sqrt(252)
    assert abs(sharpe) < 25  # no absurd artifact Sharpe (old mean-of-paths gave ~-25)


def test_comparison_to_frame_shape_and_columns():
    """C1: a ComparisonResult serializes to a 4-row tidy frame."""
    bars = _make_bars("2020-01-01", "2021-12-31")
    strat_eq = _strategy_equity("2020-06-01", 100)
    a = PerformanceAnalyzer()
    comp = a.compare_benchmarks(strat_eq, _EMPTY_TRADES, bars, 100_000, n_random_runs=5)
    df = a.comparison_to_frame(comp)

    assert list(df["series"]) == ["strategy", "buy_and_hold", "sma_trend", "random_entry"]
    for col in ("total_return", "annualized_return", "sharpe_ratio",
                "max_drawdown", "total_trades"):
        assert col in df.columns
    assert len(df) == 4


def test_format_comparison_marks_benchmark_trades_na():
    """C4: passive benchmark rows are flagged (footnote present)."""
    bars = _make_bars("2020-01-01", "2021-12-31")
    strat_eq = _strategy_equity("2020-06-01", 100)
    a = PerformanceAnalyzer()
    comp = a.compare_benchmarks(strat_eq, _EMPTY_TRADES, bars, 100_000, n_random_runs=5)
    out = a.format_comparison(comp)
    assert "(no trades)" in out  # footnote unique to the C4 fix


def test_regime_and_confidence_signal_to_noise_are_consistent():
    """C5: both tables use the same mean/std per-trade formula (no sqrt(252))."""
    trades = pd.DataFrame({
        "regime": ["BULL"] * 4,
        "pnl": [100.0, -50.0, 200.0, -30.0],
        "entry_price": [100.0] * 4,
        "qty": [10] * 4,
        "confidence": [0.95] * 4,
        "entry_date": pd.to_datetime(["2020-01-01"] * 4),
        "exit_date": pd.to_datetime(["2020-01-05"] * 4),
    })
    a = PerformanceAnalyzer()

    tr = trades["pnl"] / (trades["entry_price"] * trades["qty"].abs())
    expected = float(tr.mean() / tr.std())  # plain mean/std, no annualization

    rb = a.regime_breakdown(trades, pd.Series([100_000.0, 101_000.0]))
    assert rb[0].sharpe == pytest.approx(expected, rel=1e-9)

    buckets = a.confidence_buckets(trades)
    bucket_70plus = next(b for b in buckets if b.trade_count == 4)
    assert bucket_70plus.sharpe == pytest.approx(expected, rel=1e-9)
