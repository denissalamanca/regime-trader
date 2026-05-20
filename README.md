# Regime Trader

An HMM-based market regime detection trading system with Alpaca integration.

**Core philosophy: risk management matters more than signal generation.**
The system's value comes from adapting position sizing, leverage, and strategy selection to the detected market environment — not from predicting price direction.

> **New here?** Start with the hands-on walkthrough: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). This README is the conceptual reference.

## What This Is and Isn't

**What it is:**
- A regime-aware trading bot template using a Gaussian HMM
- Walk-forward backtesting with realistic fills and slippage
- Conservative, defense-in-depth risk management
- A starting point you fork and customize

**What it isn't:**
- A blank-slate framework — it has opinions about how to do regime trading
- A high-frequency or intraday system
- An options bot (separate template coming)
- Financial advice. Past backtest performance doesn't predict future results.

## Architecture

```
Market Data (Alpaca)
       |
       v
  Feature Engineering ──── 14 z-scored features (returns, vol, volume, trend, momentum)
       |
       v
  HMM Regime Detection ── Forward algorithm only (no look-ahead bias)
       |                   BIC model selection: 3-7 regimes
       |                   Stability filter: 3-bar persistence required
       v
  Strategy Orchestrator ── HMM still labels regimes (CRASH/BEAR/...) but the
       |                   orchestrator picks a strategy by VOLATILITY RANK,
       |                   not by label. Three archetypes: low-vol / mid-vol / high-vol.
       v
  Risk Manager ─────────── ABSOLUTE VETO POWER over any trade
       |                   Circuit breakers independent of HMM
       |                   Hardcoded limits cannot be loosened via config
       v
  Order Executor ──────── Limit orders via Alpaca (bracket: entry + stop + target)
       |                  Stop-only-tightens enforcement
       v
  Position Tracker ────── WebSocket fills, real-time P&L, regime-at-entry tracking
```

### Directory Structure

```
regime-trader/
├── core/                       # Brain of the system
│   ├── hmm_engine.py           # Gaussian HMM with BIC model selection
│   ├── regime_strategies.py    # BaseStrategy, Signal, PairSignal, orchestrator
│   ├── risk_manager.py         # Circuit breakers, position sizing, veto power
│   └── strategies/             # Optional library of 8 strategies
│       ├── pure_regime_allocation.py
│       ├── trend_following.py
│       ├── mean_reversion.py
│       ├── volatility_breakout.py
│       ├── momentum_rotation.py
│       ├── defensive_long_short.py
│       ├── pairs_zscore.py
│       └── correlation_regime.py
├── broker/                     # Alpaca integration
│   ├── alpaca_client.py        # alpaca-py SDK wrapper, paper/live
│   ├── order_executor.py       # Limit/bracket orders, pair orders, unwind logic
│   └── position_tracker.py     # WebSocket fills, P&L tracking, sync
├── data/                       # Market data pipeline
│   ├── market_data.py          # Historical + real-time bars/quotes (range-aware cache)
│   └── feature_engineering.py  # 14 HMM features + strategy indicators
├── monitoring/                 # Observability
│   ├── logger.py               # Structured JSON logs, 4 rotating files
│   └── alerts.py               # Email/webhook alerts with rate limiting
├── backtest/                   # Walk-forward backtesting
│   ├── backtester.py           # Multi-symbol WFO engine, pair-aware
│   ├── performance.py          # Sharpe/Sortino, regime + pair breakdowns
│   └── stress_test.py          # Crash injection, gap sim, Monte Carlo
├── data_cache/                 # Pre-populated daily bars (SPY/QQQ/IWM/AAPL/MSFT/GOOGL)
├── baselines/                  # SHA-256 regression hashes for refactor safety
├── scripts/bench_strategies.py # Compare every strategy on your data
├── tests/                      # 216 unit tests + 6 connectivity tests
├── config/settings.yaml        # All configurable parameters
├── main.py                     # Entry point and main trading loop
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a free Alpaca paper trading account at [alpaca.markets](https://alpaca.markets), generate API keys, then:

```bash
cp .env.example .env
# Edit .env with your ALPACA_API_KEY and ALPACA_SECRET_KEY
```

### 3. Train the HMM Model

```bash
python main.py train-only
```

This fetches ~2 years of daily SPY data, runs BIC model selection across 3-7 regimes, trains the best model, and saves it to `models/`.

### 4. Run a Backtest

```bash
python main.py backtest --symbols SPY --start 2020-01-01 --end 2024-12-31 --compare
```

### 5. Paper Trading (Dry Run)

```bash
python main.py live --dry-run
```

Runs the full pipeline but logs what it *would* trade instead of placing orders.

### 6. Paper Trading (Real Orders, Fake Money)

```bash
python main.py live
```

### 7. Run Tests

```bash
python -m pytest tests/ -v
```

## How It Works

### Regime Detection

The system uses a Gaussian HMM to classify the current market environment. The number of regimes (`N`) is picked automatically by BIC over the candidate range `[3, 4, 5, 6, 7]` — the count that explains the data with the best complexity tradeoff wins. After training, regimes are labeled by sorting on **mean return** ascending, using the schemes in `core/hmm_engine.py`:

| `N` | Labels (lowest to highest mean return) |
|---|---|
| 3 | BEAR, NEUTRAL, BULL |
| 4 | CRASH, BEAR, BULL, EUPHORIA |
| 5 | CRASH, BEAR, NEUTRAL, BULL, EUPHORIA |
| 6 | CRASH, STRONG_BEAR, WEAK_BEAR, WEAK_BULL, STRONG_BULL, EUPHORIA |
| 7 | CRASH, STRONG_BEAR, WEAK_BEAR, NEUTRAL, WEAK_BULL, STRONG_BULL, EUPHORIA |

**Critical**: The HMM uses the **forward algorithm** (filtered inference), not Viterbi. Viterbi processes the entire sequence and can revise past states using future data — that's look-ahead bias. The forward algorithm computes `P(state_t | observations_1:t)` using only past and present data. Verified by `tests/test_look_ahead.py`.

### Strategy Selection by Volatility Rank

Regime labels (CRASH, BULL, etc.) are useful diagnostics, but the actual strategy is picked by **volatility rank**. The `StrategyOrchestrator` independently re-sorts regimes by `expected_volatility` ascending, then maps each rank into thirds:

```
rank percentile = rank / (N - 1)        # 0.0 to 1.0

  percentile ≤ 0.33   →  LowVolBullStrategy
  0.33 < pct < 0.67   →  MidVolCautiousStrategy
  percentile ≥ 0.67   →  HighVolDefensiveStrategy
```

This is in `_get_strategy_for_vol_rank()` in `core/regime_strategies.py`. It works for any `N` from 3 to 7.

The three strategies hardcode their own position size and leverage — they do **not** read these from `RegimeInfo` defaults:

| Strategy | Direction | Position Size | Leverage | Stop |
|---|---|---|---|---|
| `LowVolBullStrategy` | LONG | 95% | 1.25x | `max(price − 3·ATR, 50EMA − 0.5·ATR)` |
| `MidVolCautiousStrategy` (price > 50 EMA) | LONG | 95% | 1.0x | `50EMA − 0.5·ATR` |
| `MidVolCautiousStrategy` (price < 50 EMA) | LONG | 60% | 1.0x | `price − 2·ATR` |
| `HighVolDefensiveStrategy` | LONG | 60% | 1.0x | `price − 2·ATR` |

`RegimeInfo` (defined in `core/hmm_engine.py`) carries per-label `max_leverage_allowed`, `max_position_size_pct`, and `min_confidence_to_act` populated from the `_REGIME_DEFAULTS` table, but the three vol-rank strategies pass their own explicit values to `_make_signal()`. The `RegimeInfo` defaults only kick in if a custom strategy omits those arguments.

**Final word goes to the risk manager.** Whatever the strategy requests, the values get clamped by hardcoded ceilings (50% max single position, 1.25x max leverage, 1% max risk per trade). Strategies cannot override these — only the risk manager can.

### Risk Management

The risk manager operates **independently** of regime detection. Even if the HMM fails completely, circuit breakers catch drawdowns based on actual P&L:

| Trigger | Action |
|---------|--------|
| Daily DD > 2% | Reduce all sizes 50% |
| Daily DD > 3% | Close all, halt rest of day |
| Weekly DD > 5% | Reduce all sizes 50% |
| Weekly DD > 7% | Close all, halt rest of week |
| Peak DD > 10% | Halt all trading, require manual restart |

**Hardcoded limits** (cannot be loosened via config):
- Max exposure: 80% (20% cash minimum)
- Max single position: 50% *(raised from 15% to allow regime-based allocation)*
- Max leverage: 1.25x
- Max risk per trade: 1%
- Max concurrent positions: 5
- Max daily trades: 20

### Walk-Forward Backtesting

The backtester uses rolling in-sample / out-of-sample windows:

```
|--- 2yr Train ---|--- 6mo Test ---|
                     |--- 2yr Train ---|--- 6mo Test ---|
                                          |--- 2yr Train ---|--- 6mo Test ---|
```

Each test window uses a freshly trained HMM. Single-asset fills include slippage; the pair path also models 1-bar delay and overnight gap-through.

**Sizing modes.** By default the backtest applies each strategy's raw target allocation (the *idealized* run). Set `backtest.apply_risk_manager: true` to size through the live risk manager instead (1%-risk sizing + exposure/leverage caps) — a more conservative "what risk management would have done" view (typically much lower drawdown *and* return).

**Reading the outputs.** The **equity curve** (`results/equity_curve.csv`, guarded by a SHA-256 regression test) is the ground truth for performance. The **trade log** (`results/trade_log.csv`) records rebalance/close *events* via a gate, not clean entry→exit round-trips, so its summed P&L is an approximation that may not tie exactly to the equity curve (~7% on the SPY baseline) — use it for direction/regime attribution, not exact P&L.

## CLI Reference

```bash
# Live trading
python main.py live                  # Paper trading (default)
python main.py live --dry-run        # Full pipeline, no orders placed

# HMM training
python main.py train-only            # Train model and exit

# Backtesting
python main.py backtest              # Use settings.yaml defaults
python main.py backtest --start 2020-01-01 --end 2024-12-31
python main.py backtest --symbols SPY QQQ AAPL
python main.py backtest --compare    # vs buy-hold, SMA, random
python main.py backtest --stress-test # Crash injection, gap sim

# Options
python main.py <mode> --config path/to/settings.yaml
```

## Configuration

All parameters are in `config/settings.yaml`. See that file for inline documentation of every setting. Key sections:

- **broker**: paper/live toggle, watchlist symbols
- **hmm**: regime count candidates, training window, stability filter
- **strategy**: confidence thresholds, per-regime tuning parameters
- **risk**: position limits, circuit breaker thresholds
- **execution**: order type, slippage tolerance, fill timeout
- **monitoring**: log levels, alert cooldowns
- **backtest**: window sizes, slippage model, output paths

### Strategy Library

Beyond the 3 built-in archetypes, the template ships with 8 additional strategies in `core/strategies/`:

| Strategy | Type | Vol gate | Notes |
|---|---|---|---|
| `PureRegimeAllocation` | Single-asset | All | Passive equal-weight allocator, gross varies by regime |
| `TrendFollowingRegimeFilter` | Single-asset | Low only | Long above MA, FLAT below |
| `MeanReversionLowVol` | Single-asset | Low only | RSI dip-buyer |
| `VolatilityBreakout` | Single-asset | Low/Mid | N-day high breakout, inverse-ATR sizing |
| `MomentumRotation` | Multi-asset | All | Top-N by trailing return, stateless rebalance |
| `DefensiveLongShort` | Multi-asset | All | Long top-N always, short bottom-M in high-vol |
| `StaticPairsZScore` | Pair | Low only | OLS hedge ratio, z-score fade |
| `CorrelationRegimeAllocation` | Pair | Low only | Correlation-gated, size scales with corr |

To use one, set `strategy.strategy_class` in your config (or pass it programmatically). See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for a worked example.

To benchmark every strategy on your data:

```bash
python3 scripts/bench_strategies.py
```

### Customizing the Strategy

To swap in your own logic, write a class that inherits from `BaseStrategy` and override `generate_signal()` (or `generate_pair_signal()` for pair strategies).

See [docs/CUSTOM_STRATEGIES.md](docs/CUSTOM_STRATEGIES.md) for a step-by-step guide with a worked example.

## FAQ

**Why forward algorithm instead of Viterbi?**
Viterbi runs on the entire observation sequence and revises past state assignments based on future data. In a backtest, this means the regime at time T could be influenced by data at T+100 — that's look-ahead bias. The forward algorithm computes P(state_t | obs_1:t), using only data available at time t. This is verified by `test_look_ahead.py`.

**Why is my regime count different from 7?**
The HMM tests 3, 4, 5, 6, and 7 regimes and selects the model with the lowest BIC (Bayesian Information Criterion). BIC penalizes model complexity, so it finds the simplest model that adequately explains the data. For some market periods, 3 regimes suffice; for others, 5+ are needed.

**Why did the system reject my trade?**
Check `logs/trades.log` for structured JSON entries containing the `rejection_reason`. Common reasons:
- Max exposure (80%) reached
- Max concurrent positions (5)
- Correlation > 0.85 with existing position
- Daily trade limit (20) reached
- Circuit breaker active
- Position size below $100 minimum

**How do I switch to live trading?**
In `config/settings.yaml`, set `broker.paper_trading: false`. On startup, you'll be prompted to type `YES I UNDERSTAND THE RISKS` to confirm. The code is identical between paper and live — only the Alpaca API URL changes.

**How do I add or remove symbols?**
Edit the `universe.symbols` list in `settings.yaml`. The system adapts automatically.

## Disclaimer

This software is for **educational purposes only**. Trading involves substantial risk of financial loss. Past backtest performance does not guarantee future results. The walk-forward methodology reduces but does not eliminate overfitting risk.

Always paper trade extensively before considering live deployment. Start with small position sizes and monitor the system closely. The authors assume no responsibility for trading losses.
