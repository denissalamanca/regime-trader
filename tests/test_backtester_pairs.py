"""Phase B: backtester multi-symbol and pair-trading tests.

Verifies the new behaviors added in the Phase B rewrite:
  - SimPosition lifecycle (lazy init, qty signed, direction property)
  - Trade dataclass extensions (pair_id, pair_pnl)
  - Multi-symbol equity calculation
  - Pair fill atomicity (both fill / neither fills / one fills + unwind)
  - Orphan-leg detection and force-close
  - Conditional CSV column emission (no extra cols when no pair trades)
  - Pair P&L attribution after both legs close

These tests exercise the backtester mechanics directly, without going through
the full HMM training loop, by injecting fake strategies and synthetic bars.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.backtester import (
    SimPosition, Trade, PendingPairFill, WalkForwardBacktester,
)
from backtest.performance import PerformanceAnalyzer, PairMetrics


# ---------------------------------------------------------------------------
# Synthetic bar helpers (mirrors test_strategies.py pattern)
# ---------------------------------------------------------------------------

def _make_bars(n=200, start_price=100.0, drift=0.001, vol=0.01, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = np.zeros(n)
    close[0] = start_price
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(rng.normal(drift, vol))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "open": close * 1.001, "high": high, "low": low, "close": close,
        "volume": np.abs(rng.normal(1e6, 2e5, n)),
    }, index=dates)


# ---------------------------------------------------------------------------
# SimPosition
# ---------------------------------------------------------------------------

class TestSimPosition:
    def test_default_is_flat(self):
        p = SimPosition(symbol="SPY")
        assert p.qty == 0
        assert p.is_flat
        assert p.direction == "flat"
        assert p.pair_id is None
        assert p.prev_alloc == 0.0

    def test_long_direction(self):
        p = SimPosition(symbol="SPY", qty=100)
        assert p.direction == "long"
        assert not p.is_flat

    def test_short_direction(self):
        p = SimPosition(symbol="SPY", qty=-100)
        assert p.direction == "short"
        assert not p.is_flat


# ---------------------------------------------------------------------------
# Trade extensions
# ---------------------------------------------------------------------------

class TestTradeExtensions:
    def test_pair_id_and_pair_pnl_default_none(self):
        t = Trade(
            symbol="SPY", direction="long",
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-05"),
            entry_price=400.0, exit_price=410.0,
            qty=10, pnl=100.0, pnl_pct=0.025,
            regime="BULL", confidence=0.8, strategy="test",
        )
        assert t.pair_id is None
        assert t.pair_pnl is None

    def test_pair_id_and_pair_pnl_settable(self):
        t = Trade(
            symbol="SPY", direction="long",
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-05"),
            entry_price=400.0, exit_price=410.0,
            qty=10, pnl=100.0, pnl_pct=0.025,
            regime="BULL", confidence=0.8, strategy="test",
            pair_id="PAIR-abc123", pair_pnl=250.0,
        )
        assert t.pair_id == "PAIR-abc123"
        assert t.pair_pnl == 250.0


# ---------------------------------------------------------------------------
# Multi-symbol equity calculation
# ---------------------------------------------------------------------------

class TestMultiSymbolMarkToMarket:
    def test_single_position_long(self):
        positions = {"SPY": SimPosition(symbol="SPY", qty=100, entry_price=400.0)}
        bars = {"SPY": _make_bars(n=50, start_price=400.0)}
        date = bars["SPY"].index[-1]
        last_close = float(bars["SPY"].loc[date, "close"])

        equity = WalkForwardBacktester._mark_to_market(50_000, positions, bars, date)
        assert equity == pytest.approx(50_000 + 100 * last_close)

    def test_single_position_short(self):
        # Short 100 shares at $400, then price goes up to whatever close says
        positions = {"SPY": SimPosition(symbol="SPY", qty=-100, entry_price=400.0)}
        bars = {"SPY": _make_bars(n=50, start_price=400.0)}
        date = bars["SPY"].index[-1]
        last_close = float(bars["SPY"].loc[date, "close"])

        # Short proceeds were already in cash; equity = cash + qty * close (qty negative)
        equity = WalkForwardBacktester._mark_to_market(50_000, positions, bars, date)
        assert equity == pytest.approx(50_000 + (-100) * last_close)

    def test_two_positions_one_long_one_short(self):
        positions = {
            "SPY": SimPosition(symbol="SPY", qty=50, entry_price=400.0),
            "IWM": SimPosition(symbol="IWM", qty=-30, entry_price=200.0),
        }
        bars = {
            "SPY": _make_bars(n=50, start_price=400.0, seed=1),
            "IWM": _make_bars(n=50, start_price=200.0, seed=2),
        }
        # Both should have data on the last common date
        date = bars["SPY"].index[-1]

        equity = WalkForwardBacktester._mark_to_market(100_000, positions, bars, date)
        spy_close = float(bars["SPY"].loc[date, "close"])
        iwm_close = float(bars["IWM"].loc[date, "close"])
        expected = 100_000 + 50 * spy_close + (-30) * iwm_close
        assert equity == pytest.approx(expected)

    def test_flat_position_does_not_contribute(self):
        positions = {"SPY": SimPosition(symbol="SPY", qty=0)}
        bars = {"SPY": _make_bars()}
        date = bars["SPY"].index[-1]
        equity = WalkForwardBacktester._mark_to_market(100_000, positions, bars, date)
        assert equity == 100_000

    def test_missing_bar_uses_prior_close(self):
        positions = {"SPY": SimPosition(symbol="SPY", qty=10, entry_price=100.0)}
        spy_bars = _make_bars(n=10, start_price=100.0)
        bars = {"SPY": spy_bars}
        # Ask for a date AFTER the bars end → should fall back to last prior close
        date_after = spy_bars.index[-1] + pd.Timedelta(days=10)
        last_close = float(spy_bars["close"].iloc[-1])
        equity = WalkForwardBacktester._mark_to_market(50_000, positions, bars, date_after)
        assert equity == pytest.approx(50_000 + 10 * last_close)


# ---------------------------------------------------------------------------
# Pair fill atomicity
# ---------------------------------------------------------------------------

class TestPairFillAtomicity:
    @pytest.fixture
    def bt(self):
        return WalkForwardBacktester({"backtest": {"slippage_pct": 0.0}})  # no slippage for clean math

    @pytest.fixture
    def both_fillable_bars(self):
        # Both bars have ranges that include their respective signal prices
        spy = pd.DataFrame({
            "open": [400.0], "high": [405.0], "low": [395.0],
            "close": [402.0], "volume": [1e6],
        }, index=[pd.Timestamp("2024-06-15")])
        iwm = pd.DataFrame({
            "open": [200.0], "high": [205.0], "low": [195.0],
            "close": [201.0], "volume": [1e6],
        }, index=[pd.Timestamp("2024-06-15")])
        return {"SPY": spy, "IWM": iwm}

    @pytest.fixture
    def long_unfillable_bars(self, both_fillable_bars):
        # Long limit at 400, but SPY's bar low is 401 (above limit) → won't fill
        b = both_fillable_bars
        b["SPY"] = b["SPY"].copy()
        b["SPY"].loc[pd.Timestamp("2024-06-15"), "low"] = 401.0
        b["SPY"].loc[pd.Timestamp("2024-06-15"), "open"] = 401.5
        return b

    def _pending(self, pair_id="P1"):
        return PendingPairFill(
            pair_id=pair_id,
            long_symbol="SPY", short_symbol="IWM",
            long_entry=400.0, short_entry=200.0,
            long_qty=10, short_qty=20,
            long_stop=380.0, short_stop=210.0,
            regime="BULL", confidence=0.8, strategy_name="test_pair",
            queued_at=pd.Timestamp("2024-06-14"),
        )

    def test_both_legs_fill_creates_paired_positions(self, bt, both_fillable_bars):
        positions: dict[str, SimPosition] = {}
        result, trades = bt._attempt_pair_fill(
            self._pending(), both_fillable_bars,
            pd.Timestamp("2024-06-15"), positions, regime=None,
        )
        cash_delta, link = result
        assert link == ("SPY", "IWM")
        assert "SPY" in positions and positions["SPY"].qty == 10
        assert "IWM" in positions and positions["IWM"].qty == -20
        assert positions["SPY"].pair_id == "P1"
        assert positions["IWM"].pair_id == "P1"
        # No unwind trades on a clean open
        assert trades == []

    def test_neither_leg_fills_returns_none(self, bt):
        # SPY long at 400: needs low<=400 to fill; bar low is 490 → no fill
        # IWM short at 200: needs high>=200 to fill; bar high is 110 → no fill
        spy = pd.DataFrame({
            "open": [500.0], "high": [510.0], "low": [490.0],
            "close": [502.0], "volume": [1e6],
        }, index=[pd.Timestamp("2024-06-15")])
        iwm = pd.DataFrame({
            "open": [105.0], "high": [110.0], "low": [100.0],
            "close": [108.0], "volume": [1e6],
        }, index=[pd.Timestamp("2024-06-15")])
        positions: dict[str, SimPosition] = {}
        result, trades = bt._attempt_pair_fill(
            self._pending(), {"SPY": spy, "IWM": iwm},
            pd.Timestamp("2024-06-15"), positions, regime=None,
        )
        assert result is None
        assert trades == []
        assert positions == {}

    def test_one_leg_fills_unwinds_with_doubled_slippage(self):
        # Use slippage so the unwind cost is visible
        bt = WalkForwardBacktester({"backtest": {"slippage_pct": 0.001}})

        # SPY long fills (low <= 400), IWM short doesn't fill (high < 200)
        date = pd.Timestamp("2024-06-15")
        next_date = pd.Timestamp("2024-06-16")
        spy = pd.DataFrame({
            "open": [399.0, 400.0], "high": [402.0, 401.0],
            "low": [395.0, 399.0], "close": [400.0, 400.0], "volume": [1e6, 1e6],
        }, index=[date, next_date])
        iwm = pd.DataFrame({
            "open": [180.0, 181.0], "high": [185.0, 184.0],   # high < short_entry=200 → no fill
            "low": [175.0, 178.0], "close": [182.0, 183.0], "volume": [1e6, 1e6],
        }, index=[date, next_date])

        positions: dict[str, SimPosition] = {}
        result, trades = bt._attempt_pair_fill(
            PendingPairFill(
                pair_id="P1",
                long_symbol="SPY", short_symbol="IWM",
                long_entry=400.0, short_entry=200.0,
                long_qty=10, short_qty=20,
                long_stop=380.0, short_stop=210.0,
                regime="BULL", confidence=0.8, strategy_name="test_pair",
                queued_at=date,
            ),
            {"SPY": spy, "IWM": iwm}, date, positions, regime=None,
        )
        cash_delta, link = result
        assert link is None              # not a real pair — only one filled
        assert positions == {}            # filled leg got immediately unwound, no position open
        # One unwind trade was logged
        assert len(trades) == 1
        assert trades[0].symbol == "SPY"
        assert trades[0].pair_id == "P1"
        assert "_unwind" in trades[0].strategy
        # PnL should be slightly negative due to doubled-slippage unwind
        assert trades[0].pnl < 0


# ---------------------------------------------------------------------------
# Conditional CSV columns
# ---------------------------------------------------------------------------

class TestConditionalCsvColumns:
    def test_no_pair_trades_uses_legacy_columns(self):
        trades = [
            Trade(
                symbol="SPY", direction="long",
                entry_date=pd.Timestamp("2024-01-01"),
                exit_date=pd.Timestamp("2024-01-05"),
                entry_price=400.0, exit_price=410.0,
                qty=10, pnl=100.0, pnl_pct=0.025,
                regime="BULL", confidence=0.8, strategy="test",
            ),
        ]
        df = WalkForwardBacktester._trades_to_df(trades)
        # Legacy 13-column format
        assert list(df.columns) == [
            "symbol", "direction", "entry_date", "exit_date",
            "entry_price", "exit_price", "qty", "pnl", "pnl_pct",
            "regime", "confidence", "strategy", "stop_price",
        ]

    def test_pair_trades_add_pair_columns(self):
        trades = [
            Trade(
                symbol="SPY", direction="long",
                entry_date=pd.Timestamp("2024-01-01"),
                exit_date=pd.Timestamp("2024-01-05"),
                entry_price=400.0, exit_price=410.0,
                qty=10, pnl=100.0, pnl_pct=0.025,
                regime="BULL", confidence=0.8, strategy="test",
                pair_id="P1", pair_pnl=250.0,
            ),
        ]
        df = WalkForwardBacktester._trades_to_df(trades)
        assert "pair_id" in df.columns
        assert "pair_pnl" in df.columns
        assert df.loc[0, "pair_id"] == "P1"
        assert df.loc[0, "pair_pnl"] == 250.0


# ---------------------------------------------------------------------------
# Pair P&L attribution
# ---------------------------------------------------------------------------

class TestPairPnlAttribution:
    def test_pair_pnl_filled_in_for_both_legs(self):
        long_leg = Trade(
            symbol="SPY", direction="long",
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-05"),
            entry_price=400.0, exit_price=410.0,
            qty=10, pnl=100.0, pnl_pct=0.025,
            regime="BULL", confidence=0.8, strategy="pair",
            pair_id="P1",
        )
        short_leg = Trade(
            symbol="IWM", direction="short",
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-05"),
            entry_price=200.0, exit_price=205.0,
            qty=20, pnl=-100.0, pnl_pct=-0.025,
            regime="BULL", confidence=0.8, strategy="pair",
            pair_id="P1",
        )
        WalkForwardBacktester._attribute_pair_pnl([long_leg, short_leg])
        assert long_leg.pair_pnl == 0.0
        assert short_leg.pair_pnl == 0.0

    def test_unpaired_trades_unaffected(self):
        solo = Trade(
            symbol="SPY", direction="long",
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-05"),
            entry_price=400.0, exit_price=410.0,
            qty=10, pnl=100.0, pnl_pct=0.025,
            regime="BULL", confidence=0.8, strategy="test",
        )
        WalkForwardBacktester._attribute_pair_pnl([solo])
        assert solo.pair_pnl is None


# ---------------------------------------------------------------------------
# Orphan-leg iteration (the detection helper)
# ---------------------------------------------------------------------------

class TestOrphanDetection:
    def test_paired_with_both_open_no_orphan(self):
        bt = WalkForwardBacktester({"backtest": {}})
        positions = {
            "SPY": SimPosition(symbol="SPY", qty=10, pair_id="P1"),
            "IWM": SimPosition(symbol="IWM", qty=-20, pair_id="P1"),
        }
        pair_links = {"P1": ("SPY", "IWM")}
        orphans = list(bt._iter_orphans(pair_links, positions))
        assert orphans == []

    def test_long_open_short_closed_yields_orphan(self):
        bt = WalkForwardBacktester({"backtest": {}})
        positions = {
            "SPY": SimPosition(symbol="SPY", qty=10, pair_id="P1"),
            "IWM": SimPosition(symbol="IWM", qty=0, pair_id="P1"),  # closed
        }
        pair_links = {"P1": ("SPY", "IWM")}
        orphans = list(bt._iter_orphans(pair_links, positions))
        assert orphans == [("SPY", "P1")]

    def test_short_open_long_missing_yields_orphan(self):
        bt = WalkForwardBacktester({"backtest": {}})
        positions = {
            # No SPY entry at all
            "IWM": SimPosition(symbol="IWM", qty=-20, pair_id="P1"),
        }
        pair_links = {"P1": ("SPY", "IWM")}
        orphans = list(bt._iter_orphans(pair_links, positions))
        assert orphans == [("IWM", "P1")]


# ---------------------------------------------------------------------------
# PerformanceAnalyzer.pair_breakdown
# ---------------------------------------------------------------------------

class TestPairBreakdown:
    @pytest.fixture
    def analyzer(self):
        return PerformanceAnalyzer()

    def _trades(self, rows):
        """rows: list of dicts → DataFrame."""
        return pd.DataFrame(rows)

    def test_empty_trades_returns_empty(self, analyzer):
        assert analyzer.pair_breakdown(pd.DataFrame()) == []

    def test_no_pair_id_column_returns_empty(self, analyzer):
        df = self._trades([
            {"symbol": "SPY", "direction": "long", "pnl": 100,
             "entry_date": "2024-01-01", "exit_date": "2024-01-05"},
        ])
        assert analyzer.pair_breakdown(df) == []

    def test_only_solo_trades_returns_empty(self, analyzer):
        df = self._trades([
            {"symbol": "SPY", "direction": "long", "pnl": 100,
             "entry_date": "2024-01-01", "exit_date": "2024-01-05",
             "pair_id": None},
        ])
        assert analyzer.pair_breakdown(df) == []

    def test_single_pair_round_trip_aggregates_legs(self, analyzer):
        df = self._trades([
            {"symbol": "SPY", "direction": "long", "pnl": 150,
             "entry_date": "2024-01-01", "exit_date": "2024-01-05",
             "pair_id": "P1"},
            {"symbol": "IWM", "direction": "short", "pnl": -50,
             "entry_date": "2024-01-01", "exit_date": "2024-01-05",
             "pair_id": "P1"},
        ])
        results = analyzer.pair_breakdown(df)
        assert len(results) == 1
        m = results[0]
        assert m.pair_label == "SPY/IWM"
        assert m.long_symbol == "SPY"
        assert m.short_symbol == "IWM"
        assert m.n_pair_trades == 1
        assert m.total_pnl == 100.0  # 150 + -50
        assert m.avg_pnl_per_pair == 100.0
        assert m.win_rate == 1.0
        assert m.avg_holding_period_days == 4.0
        assert m.best_pair_pnl == 100.0
        assert m.worst_pair_pnl == 100.0

    def test_multiple_round_trips_same_pair(self, analyzer):
        df = self._trades([
            # Round trip 1: combined +100 (winner)
            {"symbol": "SPY", "direction": "long", "pnl": 150,
             "entry_date": "2024-01-01", "exit_date": "2024-01-05", "pair_id": "P1"},
            {"symbol": "IWM", "direction": "short", "pnl": -50,
             "entry_date": "2024-01-01", "exit_date": "2024-01-05", "pair_id": "P1"},
            # Round trip 2: combined -30 (loser)
            {"symbol": "SPY", "direction": "long", "pnl": -80,
             "entry_date": "2024-02-01", "exit_date": "2024-02-10", "pair_id": "P2"},
            {"symbol": "IWM", "direction": "short", "pnl": 50,
             "entry_date": "2024-02-01", "exit_date": "2024-02-10", "pair_id": "P2"},
        ])
        results = analyzer.pair_breakdown(df)
        assert len(results) == 1
        m = results[0]
        assert m.n_pair_trades == 2
        assert m.total_pnl == 70.0  # 100 + -30
        assert m.avg_pnl_per_pair == 35.0
        assert m.win_rate == 0.5
        assert m.best_pair_pnl == 100.0
        assert m.worst_pair_pnl == -30.0

    def test_multiple_distinct_pairs_sorted_by_total_pnl(self, analyzer):
        df = self._trades([
            # SPY/IWM: +100
            {"symbol": "SPY", "direction": "long", "pnl": 150,
             "entry_date": "2024-01-01", "exit_date": "2024-01-05", "pair_id": "P1"},
            {"symbol": "IWM", "direction": "short", "pnl": -50,
             "entry_date": "2024-01-01", "exit_date": "2024-01-05", "pair_id": "P1"},
            # QQQ/AAPL: +250
            {"symbol": "QQQ", "direction": "long", "pnl": 300,
             "entry_date": "2024-03-01", "exit_date": "2024-03-04", "pair_id": "P2"},
            {"symbol": "AAPL", "direction": "short", "pnl": -50,
             "entry_date": "2024-03-01", "exit_date": "2024-03-04", "pair_id": "P2"},
        ])
        results = analyzer.pair_breakdown(df)
        assert len(results) == 2
        # Sorted by total_pnl descending
        assert results[0].pair_label == "QQQ/AAPL"
        assert results[1].pair_label == "SPY/IWM"

    def test_orphan_pair_id_one_leg_only_skipped(self, analyzer):
        df = self._trades([
            # Only the long leg; short never opened (orphan/unwound)
            {"symbol": "SPY", "direction": "long", "pnl": -10,
             "entry_date": "2024-01-01", "exit_date": "2024-01-01", "pair_id": "P1"},
        ])
        results = analyzer.pair_breakdown(df)
        assert results == []  # one-legged "pair" is excluded
