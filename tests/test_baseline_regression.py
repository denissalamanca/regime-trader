"""Regression guardrail: the SPY backtest must reproduce the committed baseline.

Reruns the walk-forward backtest over the committed fixed input
(``baselines/regression_SPY_bars.csv``) and asserts the equity curve still
matches ``baselines/regression_baseline.json``. This is the regression check
the project's hand-captured baselines never actually enforced.

If a change to the backtester/strategies alters the curve *intentionally*,
regenerate with ``python scripts/capture_baseline.py`` and commit the result.

Note: this runs a real (offline) backtest, so it's slower than the unit tests
(~10-20s); that's the cost of an end-to-end guardrail.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_JSON = ROOT / "baselines" / "regression_baseline.json"


@pytest.mark.skipif(
    not BASELINE_JSON.exists(),
    reason="no committed regression baseline (run scripts/capture_baseline.py)",
)
def test_backtest_matches_committed_baseline():
    from scripts.capture_baseline import run_baseline  # reuse the exact pipeline

    committed = json.loads(BASELINE_JSON.read_text())["equity_curve"]
    actual = run_baseline()["summary"]["equity_curve"]

    # Cheaper, clearer diagnostics first.
    assert actual["n_bars"] == committed["n_bars"], "equity-curve length changed"
    assert actual["ending_equity"] == committed["ending_equity"], "ending equity changed"
    # Primary guardrail: the full equity curve is byte-identical (cents-rounded).
    assert actual["sha256"] == committed["sha256"], (
        "Backtest equity curve changed vs the committed baseline. If this is an "
        "intentional change, regenerate with `python scripts/capture_baseline.py` "
        "and commit baselines/regression_baseline.json + regression_equity_curve.csv."
    )
