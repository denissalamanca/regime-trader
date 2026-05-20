# Getting Started with Regime Trader

A step-by-step walkthrough from cloning the repo to running your first backtest, picking a strategy, and (if you want) wiring it up to a paper-trading account.

If you just want the conceptual overview, read the [README](../README.md) first. This guide is the hands-on tutorial.

---

## What You're Getting

Regime Trader is a **template** for building a regime-aware trading bot. It's not a black box — you fork it, configure it, optionally write your own strategy, and run it.

Out of the box you get:

- **HMM regime detection** — Gaussian HMM with BIC model selection (3-7 regimes), forward-algorithm filtering (no look-ahead bias), 3-bar stability filter
- **Walk-forward backtester** — multi-symbol, supports pair trading, realistic slippage and fill delays, equity-curve hashing for regression tests
- **A library of 11 strategies** to start with:
  - 3 built-in archetypes: `LowVolBullStrategy`, `MidVolCautiousStrategy`, `HighVolDefensiveStrategy`
  - 6 single-asset library strategies: `PureRegimeAllocation`, `TrendFollowingRegimeFilter`, `MeanReversionLowVol`, `VolatilityBreakout`, `MomentumRotation`, `DefensiveLongShort`
  - 2 pair strategies: `StaticPairsZScore`, `CorrelationRegimeAllocation`
- **Risk manager with circuit breakers** — drawdown halts, position caps, correlation guards, leverage clamps
- **Alpaca integration** for both paper and live trading
- **216 unit tests + 6 connectivity tests** — backtester, strategies, risk manager, executor, look-ahead checks

---

## Prerequisites

- **Python 3.9 or newer**
- **An Alpaca account** ([free paper trading](https://alpaca.markets) — no funding required)
- **Git** to clone the repo
- **~10 minutes** for first backtest, ~30 seconds for re-runs (HMM training is the slow part)

---

## Step 1 — Clone & Install

```bash
git clone <your-fork-url>
cd regime-trader
pip install -r requirements.txt
```

The dependencies are: `pandas`, `numpy`, `hmmlearn`, `alpaca-py`, `pytest`, `python-dotenv`, `pyyaml`. No GPU or heavy ML libs.

---

## Step 2 — Set Up Your API Keys

```bash
cp .env.example .env
```

Open `.env` and fill in:
```
ALPACA_API_KEY=your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```

You can leave the email/SMTP variables as placeholders unless you want trade alerts.

> **Backtests work without API keys** — the project ships with a pre-populated `data_cache/` containing daily bars for SPY, QQQ, IWM, AAPL, MSFT, and GOOGL through 2026-04-24. If you want to backtest other symbols or extend the date range, you'll need real keys.

---

## Step 3 — Run the Tests

Verify your install before doing anything else:

```bash
python3 -m pytest -q
```

Expected: **216 passed, 6 skipped** (the 6 skipped tests need real Alpaca keys to run; that's normal).

If anything fails here, fix the install before continuing. The most common cause is an outdated `numpy` or `pandas` — try `pip install --upgrade -r requirements.txt`.

---

## Step 4 — Your First Backtest

```bash
python3 main.py backtest --symbols SPY --start 2018-01-01 --end 2026-04-27
```

This runs an 8-year walk-forward backtest on SPY using the default 3-archetype strategy mix. First run takes ~45-60 seconds because it trains a fresh HMM at every walk-forward window.

**What you'll see at the end:**

```
PERFORMANCE SUMMARY
═══════════════════════════════════════════════════════
  Total Return:               89.58%
  Annualized Return:          10.78%
  Sharpe Ratio:               0.48
  Max Drawdown:               17.51%
  Total Trades:               158
  Win Rate:                   62.66%
  Profit Factor:              1.80
  ...
```

Plus a per-regime breakdown showing which regimes generated the most P&L, a confidence-bucket breakdown, and a walk-forward window summary.

**Output files** land in `results/`:
- `equity_curve.csv` — daily equity values
- `trade_log.csv` — every entry/exit with regime context
- `regime_history.csv` — daily regime predictions

---

## Step 5 — Compare All the Strategies

The fastest way to see what each strategy does on your data is the bundled benchmark:

```bash
python3 scripts/bench_strategies.py
```

Takes ~10 minutes (one walk-forward backtest per strategy). When it finishes you'll see a table like this:

```
STRATEGY                          UNIVERSE     TOTAL    ANN  SHARPE  MAX_DD  TRADES  WIN%
─────────────────────────────────────────────────────────────────────────────────────────
PureRegimeAllocation (tuned)      SPY         100.0%  11.7%   0.47   21.1%     105   62%
LowVolBull (forced)               SPY          92.1%  11.0%   0.38   30.1%      40   70%
BASELINE (vol-rank)               SPY          89.6%  10.8%   0.48   17.5%     158   63%
PureRegimeAllocation (multi)      6-symbol     86.7%  10.5%   0.42   20.3%      60   42%
PureRegimeAllocation              SPY          84.2%  10.3%   0.49   14.8%     113   62%
... (lower-performing strategies)
```

A few takeaways from running this on the cached SPY data:

- **`PureRegimeAllocation` (tuned)** is the highest-return library strategy on SPY-only — 100% return over 8 years with 21% max DD. The "tuned" version uses 1.25× leverage in low-vol regimes and 85% gross in mid-vol.
- **`PureRegimeAllocation` (default, untuned)** is the best risk-adjusted: Sharpe 0.49, max DD 14.8%.
- **`MomentumRotation` and `DefensiveLongShort`** are the multi-asset strategies — they need an universe of 4+ correlated symbols to do their work.
- **`MeanReversionLowVol`** and **`VolatilityBreakout` (default)** rarely fire on SPY — mean reversion and breakouts are conceptually rare in the same regime they're gated to. Tune the parameters or apply them to a more volatile underlying.
- **The pair strategies (`StaticPairsZScore`, `CorrelationRegimeAllocation`)** lose money on SPY/IWM in this template — they don't yet have an explicit z-revert exit signal, so positions ride the 5×ATR disaster stop. They're useful as a starting point but need exit work before you'd deploy them.

---

## Step 6 — Picking a Strategy

The default backtester uses the 3-archetype vol-rank mix from `core/regime_strategies.py`. To run a backtest with one of the library strategies instead, use the `strategy.strategy_class` config override.

The cleanest way to do this is to make a copy of `config/settings.yaml`, edit it, and pass `--config`:

```yaml
# config/my_settings.yaml
strategy:
  strategy_class: pure_regime_allocation_tuned   # not yet wired to YAML — see below
  ...
```

YAML override is on the roadmap. **For now**, see the next section for a code-based approach.

### Code-based override (today's path)

The benchmark script ([scripts/bench_strategies.py](../scripts/bench_strategies.py)) is the worked example. To run a single strategy in your own script:

```python
from backtest.backtester import WalkForwardBacktester
from core.strategies import PureRegimeAllocation
from data.feature_engineering import FeatureEngineer
import pandas as pd

bars = {"SPY": pd.read_csv("data_cache/SPY_1Day.csv", index_col=0, parse_dates=True)}
fe = FeatureEngineer({"zscore_window": 126})
hmm_features = fe.compute_hmm_features(bars["SPY"])

config = {
    "hmm": {"n_candidates": [3, 4, 5], "zscore_window": 126, "min_train_bars": 252},
    "backtest": {
        "initial_capital": 100_000,
        "slippage_pct": 0.0005,
        "walk_forward": {"train_window": 252, "test_window": 63, "step_size": 126},
    },
    "strategy": {
        "strategy_class": PureRegimeAllocation,
        "low_vol_gross": 1.0,
        "mid_vol_gross": 0.85,
        "low_vol_leverage": 1.25,
    },
}

bt = WalkForwardBacktester(config)
result = bt.run(bars, hmm_features)
print(f"Return: {result.metrics.total_return:.1%}")
```

---

## Step 7 — Writing Your Own Strategy

Custom strategies are the whole point of the template. Two paths:

1. **Single-asset** (most strategies): override `generate_signal(symbol, bars, regime_state)` and return a `Signal`. Bind to one regime via `_get_strategy_for_vol_rank()` or pass via `strategy_class`.
2. **Pair / multi-asset**: set `is_pair_strategy = True` and override `generate_pair_signal(pair, bars, regime_state)` returning a `PairSignal`. The orchestrator will iterate configured pairs instead of symbols.

The full step-by-step is in [CUSTOM_STRATEGIES.md](CUSTOM_STRATEGIES.md). A 30-second sketch:

```python
from core.regime_strategies import BaseStrategy, Signal, SignalDirection, _compute_atr

class MyStrategy(BaseStrategy):
    strategy_name = "my_strategy"

    def generate_signal(self, symbol, bars, regime_state):
        if len(bars) < 60:
            return None
        price = float(bars["close"].iloc[-1])
        atr = float(_compute_atr(bars).iloc[-1])
        stop = price - 2.0 * atr  # required — risk manager rejects no-stop signals

        return self._make_signal(
            symbol, SignalDirection.LONG, price, stop, regime_state,
            reasoning="My logic fired",
            position_size_pct=0.50,
            leverage=1.0,
        )
```

The risk manager will further clamp position size and leverage. You can never request more than the hardcoded ceilings allow.

---

## Step 8 — Paper Trading (Dry Run)

Before placing real (paper) orders, do a dry run. It runs the full pipeline — fetches live bars, predicts regimes, generates signals — but logs what it *would* trade instead of placing orders:

```bash
python3 main.py live --dry-run
```

You'll see lines like:
```
DRY RUN — would place: long SPY 130 shares @ $665.43 (stop=$648.10, regime=BULL)
```

Let it run for a session or two. Watch the logs in `logs/`:
- `app.log` — high-level events
- `trades.log` — every signal, validation, and (would-be) order
- `errors.log` — anything that failed
- `regime.log` — regime detections + transitions

---

## Step 9 — Paper Trading (Real Orders, Fake Money)

```bash
python3 main.py live
```

This is full paper trading. Orders go to Alpaca, fills come back via WebSocket, P&L is tracked. The dollars are pretend, so you can leave it running for weeks and learn how the system behaves.

**What to watch for:**
- **Trades getting rejected**. `logs/trades.log` will tell you why. Most common: max exposure (80%) reached, max concurrent positions (5), correlation > 0.85, position size below $100 minimum.
- **Circuit breakers tripping**. They're in `logs/app.log` with `CIRCUIT_BREAKER_TRIGGERED` prefix. Daily DD > 2% reduces sizes 50%; DD > 3% halts the day.
- **Regime flickering**. Some periods produce rapid regime transitions. The 3-bar stability filter dampens this; if you still see noise, increase `hmm.stability_bars` in the config.

---

## Step 10 — Going to Real Money

**Don't, until you've paper-traded for at least one full month including a drawdown.** Watch the system handle a losing week. Watch how it recovers. Read the logs. Make sure rejection reasons make sense.

When you're ready:

```yaml
# config/settings.yaml
broker:
  paper_trading: false
```

On startup the system will prompt you to type `YES I UNDERSTAND THE RISKS` before placing a single live order. This is intentional friction. If you skip the prompt, no orders go out.

The code path is **identical** between paper and live — only the Alpaca API URL changes. So if it works in paper, it works in live.

---

## Common Errors & Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `Refetch failed for SPY (Call connect() first)` during backtest | Placeholder API keys + bars cache was empty for the requested range | Fill in `.env` with real Alpaca keys, OR pick a date range covered by `data_cache/*.csv` |
| `0 trades` for a strategy you expected to fire | Strategy is gated to a vol rank that didn't appear in the regime mix | Print `regime_state.label` per bar; check `core/strategies/__init__.py` for `LOW_VOL_LABELS` / `HIGH_VOL_LABELS` |
| Test suite fails with `ImportError: cannot import name 'PairSignal'` | Stale Python bytecode | `find . -name '__pycache__' -exec rm -rf {} +` then re-run |
| `Insufficient data: need 315, got N` from the backtester | Too short a date range for the walk-forward window | Use at least ~2 years (you need 252 train + 63 test) |
| `pip install hmmlearn` fails with a compile error | hmmlearn needs a C compiler | macOS: `xcode-select --install`. Linux: `apt install build-essential`. Windows: install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

---

## Project Layout — What Lives Where

```
regime-trader/
├── core/
│   ├── hmm_engine.py            # Gaussian HMM, BIC selection, forward inference
│   ├── regime_strategies.py     # BaseStrategy, Signal, PairSignal, orchestrator
│   ├── risk_manager.py          # Validation, circuit breakers, sizing
│   └── strategies/              # 8 library strategies (Phase D + E)
│       ├── pure_regime_allocation.py
│       ├── trend_following.py
│       ├── mean_reversion.py
│       ├── volatility_breakout.py
│       ├── momentum_rotation.py
│       ├── defensive_long_short.py
│       ├── pairs_zscore.py
│       └── correlation_regime.py
├── broker/
│   ├── alpaca_client.py         # alpaca-py wrapper
│   ├── order_executor.py        # Limit, bracket, pair orders + unwind logic
│   └── position_tracker.py      # WebSocket fills, P&L
├── data/
│   ├── market_data.py           # Bars cache + fetch with range-aware refetch
│   └── feature_engineering.py   # 14 z-scored HMM features
├── backtest/
│   ├── backtester.py            # Walk-forward, multi-symbol, pair-aware
│   ├── performance.py           # Metrics, regime/pair breakdowns, benchmarks
│   └── stress_test.py           # Crash injection, gap simulation
├── monitoring/                  # Structured logs, alerts
├── tests/                       # 216 unit + 6 connectivity tests
├── data_cache/                  # Pre-populated daily bars 2018→2026
├── baselines/                   # SPY regression hashes for refactor safety
├── scripts/bench_strategies.py  # Compare all strategies on your data
├── docs/
│   ├── GETTING_STARTED.md       # ← you are here
│   ├── CUSTOM_STRATEGIES.md     # how to write your own strategy
│   └── STRATEGY_LIBRARY_DESIGN.md  # design notes for the strategy library
├── config/settings.yaml         # All knobs, all in one place
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Where to Go From Here

1. **Read [CUSTOM_STRATEGIES.md](CUSTOM_STRATEGIES.md)** — write a strategy with a hypothesis you actually believe in.
2. **Tweak `config/settings.yaml`** — change watchlist, walk-forward windows, circuit breaker thresholds. Inline comments explain every field.
3. **Run `scripts/bench_strategies.py` after each change** — make sure your edits didn't regress the strategies you cared about.
4. **Look at `baselines/baseline_phaseA_SPY_2018-2026.json`** — that's the SHA-256-hashed regression target that proves changes to `core/` haven't altered single-asset arithmetic. If you refactor and the hash drifts, the verification artifact at `baselines/phase_b_verification.json` shows the tolerance criteria fallback.
5. **Don't trust the backtest blindly.** Run a dry-run live for at least a week. Then a paper-trading month. Then maybe live, with money you can afford to lose.

---

## Disclaimer

This software is for **educational purposes only**. Trading involves substantial risk of financial loss. Past backtest performance does not guarantee future results. The walk-forward methodology reduces but does not eliminate overfitting risk.

Always paper trade extensively before considering live deployment. Start with small position sizes and monitor the system closely. The authors assume no responsibility for trading losses.
