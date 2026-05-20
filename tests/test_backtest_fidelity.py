"""Tests for backtest fidelity: B2 (slippage), B4 (stop_price), B1 (risk-manager mode)."""

from pathlib import Path

import pandas as pd
import yaml

from backtest.backtester import WalkForwardBacktester, _slip_fill
from data.feature_engineering import FeatureEngineer

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "baselines" / "regression_SPY_bars.csv"


def test_slip_fill_direction():
    """B2 mechanic: buys fill above the price, sells below; 0 slippage is a no-op."""
    assert _slip_fill(100.0, buying=True, slippage=0.001) == 100.1
    assert _slip_fill(100.0, buying=False, slippage=0.001) == 99.9
    assert _slip_fill(100.0, buying=True, slippage=0.0) == 100.0
    assert _slip_fill(100.0, buying=False, slippage=0.0) == 100.0


def _run(slippage: float, end: str = "2020-12-31"):
    """Run the backtester on a short slice of the fixture (fast: ~2 windows)."""
    config = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    config.setdefault("backtest", {})["slippage_pct"] = slippage
    bars = pd.read_csv(BARS, index_col=0, parse_dates=True)
    bars = bars[bars.index <= pd.Timestamp(end)]
    feats = FeatureEngineer(config["hmm"]).compute_hmm_features(bars)
    return WalkForwardBacktester(config).run({"SPY": bars}, feats)


def test_stop_price_recorded_and_slippage_costs_money():
    """B4: trades carry a non-zero stop_price. B2: slippage drags ending equity down."""
    base = _run(slippage=0.0)
    slipped = _run(slippage=0.01)  # exaggerated so the cost is unambiguous

    assert len(slipped.trades) > 0
    # B4: the trade log now records the protective stop (was all zeros before).
    assert (slipped.trades["stop_price"] != 0).any()
    # B2: on the same regime path, slippage strictly reduces ending equity.
    assert slipped.equity_curve.iloc[-1] < base.equity_curve.iloc[-1]


def test_apply_risk_manager_is_more_conservative():
    """B1: routing sizing through the RiskManager (1%-risk sizing) cuts exposure,
    so the risk-managed backtest takes no larger a drawdown than the idealized
    one and follows a different equity path."""
    config = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    bars = pd.read_csv(BARS, index_col=0, parse_dates=True)
    bars = bars[bars.index <= pd.Timestamp("2020-12-31")]
    feats = FeatureEngineer(config["hmm"]).compute_hmm_features(bars)

    config["backtest"]["apply_risk_manager"] = False
    idealized = WalkForwardBacktester(config).run({"SPY": bars}, feats)
    config["backtest"]["apply_risk_manager"] = True
    managed = WalkForwardBacktester(config).run({"SPY": bars}, feats)

    assert len(managed.trades) > 0
    assert idealized.equity_curve.iloc[-1] != managed.equity_curve.iloc[-1]
    # 1%-risk sizing is far smaller than the archetypes' ~95%, so the
    # risk-managed path takes no larger a drawdown.
    assert managed.metrics.max_drawdown <= idealized.metrics.max_drawdown
