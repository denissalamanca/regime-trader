"""Capture / regenerate the backtest regression baseline (refactor safety).

Runs the SPY walk-forward backtest over a FIXED, committed input
(``baselines/regression_SPY_bars.csv``) and writes:

  - ``baselines/regression_equity_curve.csv`` — the equity series (cents)
  - ``baselines/regression_baseline.json``    — config snapshot + equity SHA-256
                                                 + key metrics

The committed input plus the HMM's fixed random seeds make the equity curve
deterministic, so ``tests/test_baseline_regression.py`` can assert the code
still reproduces it. The equity is rounded to cents before hashing so the
guardrail is robust to sub-cent float-ordering noise across library versions.

Re-run this and commit the result whenever an intentional change to the
backtester or strategies alters the curve:

    python scripts/capture_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.backtester import WalkForwardBacktester  # noqa: E402
from data.feature_engineering import FeatureEngineer  # noqa: E402

BARS_CSV = ROOT / "baselines" / "regression_SPY_bars.csv"
EQUITY_CSV = ROOT / "baselines" / "regression_equity_curve.csv"
BASELINE_JSON = ROOT / "baselines" / "regression_baseline.json"
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def equity_hash(equity: pd.Series) -> str:
    """SHA-256 of the cents-rounded equity curve (robust to sub-cent drift)."""
    rounded = equity.round(2)
    payload = "\n".join(f"{d.date()},{v:.2f}" for d, v in rounded.items())
    return hashlib.sha256(payload.encode()).hexdigest()


def run_baseline() -> dict:
    """Run the fixed SPY backtest and return {result, equity, summary}."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    bars = pd.read_csv(BARS_CSV, index_col=0, parse_dates=True)
    fe = FeatureEngineer(config.get("hmm", {}))
    feats = fe.compute_hmm_features(bars)
    result = WalkForwardBacktester(config).run({"SPY": bars}, feats)
    eq = result.equity_curve
    bt = config.get("backtest", {})
    summary = {
        "symbol": "SPY",
        "input_fixture": BARS_CSV.name,
        "n_input_bars": int(len(bars)),
        "input_range": [str(bars.index[0].date()), str(bars.index[-1].date())],
        "config": {
            "hmm.n_candidates": config["hmm"]["n_candidates"],
            "hmm.zscore_window": config["hmm"]["zscore_window"],
            "hmm.min_train_bars": config["hmm"]["min_train_bars"],
            "hmm.random_state": config["hmm"].get("random_state", 42),
            "backtest.walk_forward": bt.get("walk_forward"),
            "backtest.slippage_pct": bt.get("slippage_pct"),
            "backtest.apply_risk_manager": bt.get("apply_risk_manager", False),
            "risk.max_leverage": config["risk"]["max_leverage"],
        },
        "equity_curve": {
            "first_date": str(eq.index[0].date()),
            "last_date": str(eq.index[-1].date()),
            "n_bars": int(len(eq)),
            "starting_equity": round(float(eq.iloc[0]), 2),
            "ending_equity": round(float(eq.iloc[-1]), 2),
            "sha256": equity_hash(eq),
        },
        "metrics": {
            "total_return": round(float(result.metrics.total_return), 8),
            "sharpe_ratio": round(float(result.metrics.sharpe_ratio), 6),
            "max_drawdown": round(float(result.metrics.max_drawdown), 8),
            "total_trades": int(result.metrics.total_trades),
        },
    }
    return {"result": result, "equity": eq, "summary": summary}


def main() -> None:
    out = run_baseline()
    out["equity"].round(2).to_csv(EQUITY_CSV, header=True)
    BASELINE_JSON.write_text(json.dumps(out["summary"], indent=2) + "\n")

    s = out["summary"]
    print(
        f"Captured baseline: {s['equity_curve']['n_bars']} bars "
        f"({s['equity_curve']['first_date']} -> {s['equity_curve']['last_date']}), "
        f"ending ${s['equity_curve']['ending_equity']:,.2f}, "
        f"total_return={s['metrics']['total_return']:.4%}, "
        f"trades={s['metrics']['total_trades']}"
    )
    print(f"equity sha256: {s['equity_curve']['sha256']}")
    print(f"wrote {BASELINE_JSON.name} and {EQUITY_CSV.name}")


if __name__ == "__main__":
    main()
