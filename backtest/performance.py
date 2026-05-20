"""Performance analytics: Sharpe, drawdown, win rate, regime-specific metrics,
confidence bucketing, and benchmark comparisons.

All metrics are computed from equity curves and trade logs produced by the
walk-forward backtester. No lookahead — everything here operates on
already-generated results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Core performance statistics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    avg_holding_period_days: float
    expectancy: float
    avg_trades_per_week: float
    max_consecutive_losses: int
    worst_day: float
    worst_week: float
    worst_month: float


@dataclass
class RegimeMetrics:
    """Performance breakdown by regime."""

    regime_name: str
    trade_count: int
    pnl_contribution: float             # $ P&L from this regime
    return_contribution_pct: float      # % of total return
    win_rate: float
    avg_pnl: float
    sharpe: float


@dataclass
class ConfidenceBucket:
    """Performance by regime confidence level."""

    bucket_label: str                   # e.g. "70%+"
    min_confidence: float
    max_confidence: float
    trade_count: int
    total_pnl: float
    win_rate: float
    sharpe: float


@dataclass
class PairMetrics:
    """Performance breakdown for a single pair (long/short symbol combo).

    One row per (long_symbol, short_symbol). Aggregates over all round-trip
    pair_ids of that pair. Win rate is over pair-trades (combined leg PnL > 0),
    not per leg.
    """

    pair_label: str                     # e.g. "SPY/IWM"
    long_symbol: str
    short_symbol: str
    n_pair_trades: int                  # number of round-trips
    total_pnl: float                    # sum across both legs of all round-trips
    avg_pnl_per_pair: float
    win_rate: float                     # fraction of round-trips with combined PnL > 0
    avg_holding_period_days: float
    best_pair_pnl: float
    worst_pair_pnl: float


@dataclass
class ComparisonResult:
    """Side-by-side comparison of strategy vs benchmarks."""

    strategy: PerformanceMetrics
    buy_and_hold: PerformanceMetrics
    sma_trend: PerformanceMetrics
    random_entry: PerformanceMetrics


@dataclass
class FullReport:
    """Complete backtest report."""

    core: PerformanceMetrics
    regime_breakdown: list[RegimeMetrics]
    confidence_buckets: list[ConfidenceBucket]
    comparison: Optional[ComparisonResult] = None


# ---------------------------------------------------------------------------
# Static metric functions
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.045,
                 periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods
    return float(excess.mean() / returns.std() * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.045,
                  periods: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods
    downside = returns[returns < 0]
    if len(downside) < 1 or downside.std() == 0:
        return float(excess.mean() / 1e-10 * np.sqrt(periods))  # All positive
    return float(excess.mean() / downside.std() * np.sqrt(periods))


def max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """Maximum drawdown fraction and duration in trading days."""
    if len(equity_curve) < 2:
        return 0.0, 0
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    max_dd = float(dd.min())  # Most negative = worst drawdown

    # Duration: longest period below previous peak
    is_dd = equity_curve < peak
    if not is_dd.any():
        return 0.0, 0

    groups = (~is_dd).cumsum()
    dd_lengths = is_dd.groupby(groups).sum()
    max_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    return abs(max_dd), max_duration


def _max_consecutive(series: pd.Series, condition: bool = True) -> int:
    """Count max consecutive True values in a boolean series."""
    if len(series) == 0:
        return 0
    groups = (series != condition).cumsum()
    streaks = series[series == condition].groupby(groups).count()
    return int(streaks.max()) if len(streaks) > 0 else 0


# ---------------------------------------------------------------------------
# Performance Analyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """Computes comprehensive performance metrics from backtest results.

    Parameters
    ----------
    risk_free_rate : float
        Annualized risk-free rate (default 4.5%).
    """

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self._rf = risk_free_rate

    def analyze(self, equity_curve: pd.Series,
                trades: pd.DataFrame) -> PerformanceMetrics:
        """Compute full performance metrics.

        Parameters
        ----------
        equity_curve : pd.Series
            Daily equity values with DatetimeIndex.
        trades : pd.DataFrame
            Trade log. Required columns: symbol, entry_date, exit_date,
            entry_price, exit_price, qty, pnl, regime, confidence.

        Returns
        -------
        PerformanceMetrics
        """
        returns = equity_curve.pct_change().dropna()
        n_days = len(returns)

        if n_days < 2:
            return self._empty_metrics()

        # Core return metrics
        total_ret = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)
        ann_ret = float((1 + total_ret) ** (252 / n_days) - 1) if n_days > 0 else 0
        ann_vol = float(returns.std() * np.sqrt(252))
        sr = sharpe_ratio(returns, self._rf)
        so = sortino_ratio(returns, self._rf)
        mdd, mdd_dur = max_drawdown(equity_curve)
        calmar = ann_ret / mdd if mdd > 0 else 0

        # Trade metrics
        n_trades = len(trades)
        if n_trades > 0:
            winners = trades[trades["pnl"] > 0]
            losers = trades[trades["pnl"] < 0]
            win_rate = len(winners) / n_trades
            avg_win = float(winners["pnl"].mean()) if len(winners) > 0 else 0
            avg_loss = float(losers["pnl"].mean()) if len(losers) > 0 else 0
            gross_profit = float(winners["pnl"].sum()) if len(winners) > 0 else 0
            gross_loss = abs(float(losers["pnl"].sum())) if len(losers) > 0 else 0
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

            # Holding period
            if "entry_date" in trades.columns and "exit_date" in trades.columns:
                durations = (pd.to_datetime(trades["exit_date"])
                             - pd.to_datetime(trades["entry_date"])).dt.days
                avg_hold = float(durations.mean())
            else:
                avg_hold = 0

            # Consecutive losses
            is_loss = trades["pnl"] < 0
            max_consec = _max_consecutive(is_loss, True)

            # Trades per week
            weeks = n_days / 5
            tpw = n_trades / weeks if weeks > 0 else 0
        else:
            win_rate = avg_win = avg_loss = pf = expectancy = avg_hold = tpw = 0
            max_consec = 0

        # Worst periods
        worst_day = float(returns.min()) if len(returns) > 0 else 0
        weekly = returns.resample("W").sum() if hasattr(returns.index, "freq") or isinstance(returns.index, pd.DatetimeIndex) else returns.rolling(5).sum()
        worst_week = float(weekly.min()) if len(weekly) > 0 else 0
        monthly = returns.resample("ME").sum() if isinstance(returns.index, pd.DatetimeIndex) else returns.rolling(21).sum()
        worst_month = float(monthly.min()) if len(monthly) > 0 else 0

        return PerformanceMetrics(
            total_return=total_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sr,
            sortino_ratio=so,
            max_drawdown=mdd,
            max_drawdown_duration_days=mdd_dur,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=pf,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=n_trades,
            avg_holding_period_days=avg_hold,
            expectancy=expectancy,
            avg_trades_per_week=tpw,
            max_consecutive_losses=max_consec,
            worst_day=worst_day,
            worst_week=worst_week,
            worst_month=worst_month,
        )

    # ------------------------------------------------------------------
    # Regime-specific breakdown
    # ------------------------------------------------------------------

    def regime_breakdown(self, trades: pd.DataFrame,
                         equity_curve: pd.Series) -> list[RegimeMetrics]:
        """Break down performance by regime at time of entry."""
        if len(trades) == 0 or "regime" not in trades.columns:
            return []

        results = []
        total_pnl = float(trades["pnl"].sum())

        for regime, group in trades.groupby("regime"):
            n = len(group)
            pnl = float(group["pnl"].sum())
            winners = group[group["pnl"] > 0]
            wr = len(winners) / n if n > 0 else 0

            # Per-trade signal-to-noise: mean trade return / std of trade returns.
            # NOT an annualized Sharpe — these are per-trade stats on small,
            # uneven samples, so treat them as a rough quality gauge only. Kept
            # consistent with confidence_buckets (same formula, no sqrt(252)).
            if n > 1:
                trade_rets = group["pnl"] / (group["entry_price"] * group["qty"].abs())
                sr = float(trade_rets.mean() / trade_rets.std()) if trade_rets.std() > 0 else 0.0
            else:
                sr = 0.0

            results.append(RegimeMetrics(
                regime_name=str(regime),
                trade_count=n,
                pnl_contribution=pnl,
                return_contribution_pct=pnl / total_pnl * 100 if total_pnl != 0 else 0,
                win_rate=wr,
                avg_pnl=float(group["pnl"].mean()),
                sharpe=sr,
            ))

        return sorted(results, key=lambda r: r.pnl_contribution, reverse=True)

    # ------------------------------------------------------------------
    # Pair breakdown
    # ------------------------------------------------------------------

    def pair_breakdown(self, trades: pd.DataFrame) -> list[PairMetrics]:
        """Break down performance by pair (long/short symbol combo).

        Groups trades by ``pair_id`` (one round-trip per id, two legs each),
        then aggregates round-trips by their long/short symbol pair. Returns
        one row per pair, sorted by total P&L descending. Trades with a null
        ``pair_id`` (non-pair trades) are ignored.
        """
        if len(trades) == 0 or "pair_id" not in trades.columns:
            return []

        paired = trades[trades["pair_id"].notna()]
        if len(paired) == 0:
            return []

        # Roll up legs into round-trips keyed by pair_id.
        round_trips: dict[str, dict] = {}
        for _, row in paired.iterrows():
            pid = row["pair_id"]
            rt = round_trips.setdefault(pid, {
                "long_symbol": None, "short_symbol": None,
                "combined_pnl": 0.0,
                "entry_date": None, "exit_date": None,
            })
            rt["combined_pnl"] += float(row["pnl"])
            if str(row["direction"]).lower() == "long":
                rt["long_symbol"] = str(row["symbol"])
            else:
                rt["short_symbol"] = str(row["symbol"])
            entry = pd.to_datetime(row["entry_date"])
            exit_ = pd.to_datetime(row["exit_date"])
            rt["entry_date"] = entry if rt["entry_date"] is None else min(rt["entry_date"], entry)
            rt["exit_date"] = exit_ if rt["exit_date"] is None else max(rt["exit_date"], exit_)

        # Group round-trips by (long_symbol, short_symbol).
        by_pair: dict[tuple[str, str], list[dict]] = {}
        for rt in round_trips.values():
            if rt["long_symbol"] is None or rt["short_symbol"] is None:
                # Orphan or single-leg artifact — skip from aggregate row.
                continue
            key = (rt["long_symbol"], rt["short_symbol"])
            by_pair.setdefault(key, []).append(rt)

        results: list[PairMetrics] = []
        for (long_sym, short_sym), rts in by_pair.items():
            n = len(rts)
            pnls = [r["combined_pnl"] for r in rts]
            total = float(sum(pnls))
            avg = total / n
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / n
            holds = [(r["exit_date"] - r["entry_date"]).days for r in rts]
            avg_hold = float(sum(holds) / n) if n > 0 else 0.0
            results.append(PairMetrics(
                pair_label=f"{long_sym}/{short_sym}",
                long_symbol=long_sym,
                short_symbol=short_sym,
                n_pair_trades=n,
                total_pnl=total,
                avg_pnl_per_pair=avg,
                win_rate=wr,
                avg_holding_period_days=avg_hold,
                best_pair_pnl=float(max(pnls)),
                worst_pair_pnl=float(min(pnls)),
            ))

        return sorted(results, key=lambda r: r.total_pnl, reverse=True)

    # ------------------------------------------------------------------
    # Confidence bucketing
    # ------------------------------------------------------------------

    def confidence_buckets(self, trades: pd.DataFrame) -> list[ConfidenceBucket]:
        """Bucket trades by regime probability at entry."""
        if len(trades) == 0 or "confidence" not in trades.columns:
            return []

        buckets_def = [
            ("< 50%", 0.0, 0.50),
            ("50-60%", 0.50, 0.60),
            ("60-70%", 0.60, 0.70),
            ("70%+", 0.70, 1.01),
        ]
        results = []
        for label, lo, hi in buckets_def:
            mask = (trades["confidence"] >= lo) & (trades["confidence"] < hi)
            group = trades[mask]
            n = len(group)
            if n == 0:
                results.append(ConfidenceBucket(label, lo, hi, 0, 0, 0, 0))
                continue

            pnl = float(group["pnl"].sum())
            winners = group[group["pnl"] > 0]
            wr = len(winners) / n

            # Per-trade signal-to-noise (mean/std), consistent with
            # regime_breakdown. NOT an annualized Sharpe.
            if n > 1:
                trade_rets = group["pnl"] / (group["entry_price"] * group["qty"].abs())
                sr = float(trade_rets.mean() / trade_rets.std()) if trade_rets.std() > 0 else 0.0
            else:
                sr = 0.0

            results.append(ConfidenceBucket(label, lo, hi, n, pnl, wr, sr))

        return results

    # ------------------------------------------------------------------
    # Benchmark comparisons
    # ------------------------------------------------------------------

    def compare_benchmarks(
        self,
        strategy_equity: pd.Series,
        strategy_trades: pd.DataFrame,
        bars: pd.DataFrame,
        initial_capital: float = 100_000,
        n_random_runs: int = 50,
    ) -> ComparisonResult:
        """Compare strategy against buy-and-hold, SMA trend, and random entry.

        Parameters
        ----------
        strategy_equity : pd.Series
            Strategy equity curve.
        strategy_trades : pd.DataFrame
            Strategy trade log.
        bars : pd.DataFrame
            OHLCV data for the reference symbol (e.g. SPY).
        initial_capital : float
        n_random_runs : int
            Number of random-entry simulations; the median-outcome path is used.
        """
        strat_metrics = self.analyze(strategy_equity, strategy_trades)

        # Align benchmarks to the strategy's ACTUAL traded span so the
        # comparison is like-for-like. The strategy equity curve only covers the
        # out-of-sample period (after walk-forward warm-up + first-train window),
        # which is typically ~2 years shorter than the full bar history. Without
        # this, buy-and-hold is credited with returns from years the strategy
        # never traded.
        if len(strategy_equity) > 0:
            span_start, span_end = strategy_equity.index[0], strategy_equity.index[-1]
            bars = bars.loc[(bars.index >= span_start) & (bars.index <= span_end)]

        # Buy and hold
        bh_equity = self._buy_and_hold_equity(bars, initial_capital)
        bh_metrics = self.analyze(bh_equity, pd.DataFrame(columns=["pnl", "entry_date", "exit_date", "entry_price", "exit_price", "qty", "regime", "confidence"]))

        # 200 SMA trend following
        sma_equity = self._sma_trend_equity(bars, initial_capital)
        sma_metrics = self.analyze(sma_equity, pd.DataFrame(columns=["pnl", "entry_date", "exit_date", "entry_price", "exit_price", "qty", "regime", "confidence"]))

        # Random entry (median-outcome path across n runs)
        rand_equity = self._random_entry_equity(bars, initial_capital, strategy_trades, n_random_runs)
        rand_metrics = self.analyze(rand_equity, pd.DataFrame(columns=["pnl", "entry_date", "exit_date", "entry_price", "exit_price", "qty", "regime", "confidence"]))

        return ComparisonResult(
            strategy=strat_metrics,
            buy_and_hold=bh_metrics,
            sma_trend=sma_metrics,
            random_entry=rand_metrics,
        )

    def _buy_and_hold_equity(self, bars: pd.DataFrame, capital: float) -> pd.Series:
        """Simple buy-and-hold equity curve."""
        close = bars["close"]
        shares = capital / close.iloc[0]
        return close * shares

    def _sma_trend_equity(self, bars: pd.DataFrame, capital: float,
                          sma_period: int = 200) -> pd.Series:
        """200 SMA trend-following: long above SMA, cash below."""
        close = bars["close"]
        sma = close.rolling(sma_period).mean()
        position = (close > sma).astype(float)
        # Fill NaN period with 0 (cash)
        position = position.fillna(0)
        returns = close.pct_change().fillna(0) * position.shift(1).fillna(0)
        equity = capital * (1 + returns).cumprod()
        return equity

    def _random_entry_equity(self, bars: pd.DataFrame, capital: float,
                             strategy_trades: pd.DataFrame,
                             n_runs: int = 50) -> pd.Series:
        """Random entries with same frequency and risk management as strategy."""
        close = bars["close"]
        n_bars = len(close)
        n_trades = len(strategy_trades) if len(strategy_trades) > 0 else 10
        # Approximate trade frequency
        trade_prob = n_trades / n_bars if n_bars > 0 else 0.02

        all_equities = []
        for seed in range(n_runs):
            rng = np.random.RandomState(seed)
            position = 0.0
            equity = capital
            equities = [equity]

            for i in range(1, n_bars):
                ret = float(close.iloc[i] / close.iloc[i - 1] - 1)
                # Random entry/exit
                if position == 0 and rng.random() < trade_prob:
                    direction = rng.choice([-1, 1])
                    position = direction * 0.10  # 10% position
                elif position != 0 and rng.random() < trade_prob:
                    position = 0
                # Apply 1% stop loss
                if abs(ret * position) > 0.01:
                    equity += equity * (-0.01 * np.sign(position))
                    position = 0
                else:
                    equity += equity * ret * position
                equities.append(equity)

            all_equities.append(pd.Series(equities, index=close.index))

        # Return the MEDIAN-OUTCOME path (by final equity), not the mean of all
        # paths. Averaging many random paths cancels their volatility and yields
        # a near-flat curve whose Sharpe/drawdown are meaningless artifacts. The
        # median run is a single, real random path that preserves realistic vol.
        finals = [float(s.iloc[-1]) for s in all_equities]
        median_run = all_equities[int(np.argsort(finals)[len(finals) // 2])]
        return median_run

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            max_drawdown_duration_days=0, calmar_ratio=0, win_rate=0,
            profit_factor=0, avg_win=0, avg_loss=0, total_trades=0,
            avg_holding_period_days=0, expectancy=0, avg_trades_per_week=0,
            max_consecutive_losses=0, worst_day=0, worst_week=0, worst_month=0,
        )

    @staticmethod
    def comparison_to_frame(comp: "ComparisonResult") -> pd.DataFrame:
        """Serialize a ComparisonResult to a tidy DataFrame (one row per series).

        Written to ``results/comparison.csv`` so the benchmark comparison is
        persisted, not just printed. ``total_trades`` is only meaningful for the
        strategy row; benchmark rows are passive curves (0 trades).
        """
        rows = []
        for series, m in (
            ("strategy", comp.strategy),
            ("buy_and_hold", comp.buy_and_hold),
            ("sma_trend", comp.sma_trend),
            ("random_entry", comp.random_entry),
        ):
            rows.append({
                "series": series,
                "total_return": m.total_return,
                "annualized_return": m.annualized_return,
                "annualized_volatility": m.annualized_volatility,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "max_drawdown": m.max_drawdown,
                "calmar_ratio": m.calmar_ratio,
                "total_trades": m.total_trades,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Formatted output
    # ------------------------------------------------------------------

    def format_summary(self, metrics: PerformanceMetrics) -> str:
        """Format metrics as a readable table string."""
        return (
            f"\n{'=' * 55}\n"
            f"{'PERFORMANCE SUMMARY':^55}\n"
            f"{'=' * 55}\n"
            f"  Total Return:          {metrics.total_return:>10.2%}\n"
            f"  Annualized Return:     {metrics.annualized_return:>10.2%}\n"
            f"  Annualized Volatility: {metrics.annualized_volatility:>10.2%}\n"
            f"  Sharpe Ratio:          {metrics.sharpe_ratio:>10.2f}\n"
            f"  Sortino Ratio:         {metrics.sortino_ratio:>10.2f}\n"
            f"  Max Drawdown:          {metrics.max_drawdown:>10.2%}\n"
            f"  Max DD Duration:       {metrics.max_drawdown_duration_days:>10d} days\n"
            f"  Calmar Ratio:          {metrics.calmar_ratio:>10.2f}\n"
            f"{'-' * 55}\n"
            f"  Total Trades:          {metrics.total_trades:>10d}\n"
            f"  Win Rate:              {metrics.win_rate:>10.2%}\n"
            f"  Profit Factor:         {metrics.profit_factor:>10.2f}\n"
            f"  Avg Win:              ${metrics.avg_win:>10.2f}\n"
            f"  Avg Loss:             ${metrics.avg_loss:>10.2f}\n"
            f"  Expectancy:           ${metrics.expectancy:>10.2f}\n"
            f"  Avg Hold Period:       {metrics.avg_holding_period_days:>10.1f} days\n"
            f"  Avg Trades/Week:       {metrics.avg_trades_per_week:>10.1f}\n"
            f"{'-' * 55}\n"
            f"  Worst Day:             {metrics.worst_day:>10.2%}\n"
            f"  Worst Week:            {metrics.worst_week:>10.2%}\n"
            f"  Worst Month:           {metrics.worst_month:>10.2%}\n"
            f"  Max Consec. Losses:    {metrics.max_consecutive_losses:>10d}\n"
            f"{'=' * 55}"
        )

    def format_comparison(self, comp: ComparisonResult) -> str:
        """Format comparison table."""
        def _row(name, m):
            # Benchmarks are passive equity curves with no trade log, so their
            # trade-derived stats are not meaningful — show "-" instead of 0.
            trades = f"{m.total_trades:>6d}" if m.total_trades else f"{'-':>6}"
            return (f"  {name:<16} {m.total_return:>8.2%}  {m.annualized_return:>8.2%}  "
                    f"{m.sharpe_ratio:>6.2f}  {m.max_drawdown:>8.2%}  {trades}")

        header = f"  {'':16} {'TotRet':>8}  {'AnnRet':>8}  {'Sharpe':>6}  {'MaxDD':>8}  {'Trades':>6}"
        return (
            f"\n{'=' * 70}\n"
            f"{'BENCHMARK COMPARISON':^70}\n"
            f"{'=' * 70}\n"
            f"{header}\n"
            f"  {'-' * 64}\n"
            f"{_row('Strategy', comp.strategy)}\n"
            f"{_row('Buy & Hold', comp.buy_and_hold)}\n"
            f"{_row('200 SMA Trend', comp.sma_trend)}\n"
            f"{_row('Random Entry', comp.random_entry)}\n"
            f"  {'-' * 64}\n"
            f"  benchmarks span the strategy's traded window; '-' = passive (no trades)\n"
            f"{'=' * 70}"
        )
