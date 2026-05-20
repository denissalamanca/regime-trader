# Writing Custom Strategies

This guide walks you through replacing or extending the built-in strategies in `core/regime_strategies.py`. By the end you'll have a custom strategy class wired into the system that fires when its regime is detected.

The reader is assumed to have cloned the repo, run `setup.sh`, and trained the HMM at least once with `python main.py train-only`.

---

## How the strategy layer works

The HMM doesn't know about bull or bear markets directly — it learns volatility clusters from your features. The job of the strategy layer is to translate "we're in regime X" into "buy/sell/hold this much of this symbol."

The flow on every bar:

1. `HMMEngine` runs the forward algorithm and returns a `RegimeState`.
2. `StrategyOrchestrator` looks up which strategy class is mapped to that regime.
3. The strategy's `generate_signal()` method runs against the latest OHLCV bars and produces a `Signal` (or `None` if no entry condition is met).
4. The risk manager validates the `Signal` and either executes or vetoes it.

You only have to write step 3. Everything else is plumbing.

---

## The `BaseStrategy` abstract class

Every strategy inherits from `BaseStrategy` (in `core/regime_strategies.py`). The minimum contract:

```python
from core.regime_strategies import BaseStrategy, Signal, SignalDirection

class MyStrategy(BaseStrategy):
    strategy_name = "my_strategy"   # Required: identifies your strategy in logs

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,           # OHLCV with columns: open, high, low, close, volume
        regime_state: RegimeState,    # Current regime info (label, probability, state_id)
    ) -> Optional[Signal]:
        # Return a Signal if conditions are met, otherwise None
        ...
```

That's the whole interface. `bars` is the symbol's recent price history; `regime_state` is the HMM's current view of the world. Return `None` to do nothing.

Use the helper `self._make_signal(...)` to build a `Signal` — it auto-fills the regime fields and respects `RegimeInfo` defaults.

---

## The `Signal` dataclass

Every field, what it means:

| Field | Meaning |
|---|---|
| `symbol` | Ticker (e.g. `"SPY"`). |
| `direction` | `SignalDirection.LONG`, `SHORT`, or `FLAT`. `FLAT` closes existing positions in this symbol. |
| `confidence` | 0–1. Usually `regime_state.probability`. The risk manager downweights low-confidence trades. |
| `entry_price` | Where you intend to enter. Used as the limit price; actual fill includes slippage. |
| `stop_loss` | **Required and non-negotiable.** Risk manager rejects any signal without one. For longs: below entry. For shorts: above. |
| `take_profit` | Optional. `None` means "trail stop only, no fixed target." |
| `position_size_pct` | Suggested fraction of equity (e.g. `0.10` = 10%). Capped by `MAX_SINGLE_POSITION` (50%). |
| `leverage` | 1.0–1.25. Capped by `MAX_PORTFOLIO_LEVERAGE`. |
| `regime_id` / `regime_name` / `regime_probability` | Auto-filled by `_make_signal`. |
| `timestamp` | Auto-filled. |
| `reasoning` | Free-text explanation that appears in logs. Be specific — future-you will thank you. |
| `strategy_name` | The class attribute. Auto-filled. |
| `metadata` | Dict for anything strategy-specific (e.g. `{"trail_atr_mult": 2.5}`). |

---

## How `StrategyOrchestrator` picks a strategy

The orchestrator does **not** look at regime labels (BULL, BEAR, etc.) — those names come from sorting regimes by mean return and can be misleading. Instead it sorts regimes by their learned **volatility** and assigns a strategy by rank:

```
regimes sorted by expected_volatility ascending
   ↓
rank 0 (lowest vol)   → LowVolBullStrategy        (be invested, leverage allowed)
   ↓
rank middle           → MidVolCautiousStrategy    (long if above 50 EMA, else flat)
   ↓
rank N-1 (highest vol) → HighVolDefensiveStrategy (reduced exposure, capital preservation)
```

The mapping from rank to class lives in `_get_strategy_for_vol_rank()` in `core/regime_strategies.py`. The thresholds are 33% and 67% of the vol-rank percentile, so the system works with any number of regimes (3–7).

**This means:** when you replace a strategy, you're replacing its behavior for everywhere in the volatility spectrum the orchestrator points to it.

---

## Step-by-step: adding a custom strategy

### 1. Open `core/regime_strategies.py` and define your class

Put it next to the existing strategies (around line 220, after `HighVolDefensiveStrategy`).

### 2. Wire it into `_get_strategy_for_vol_rank()`

If you want your new strategy to handle a particular vol rank, edit that function. If you want it to *replace* one of the existing three, just swap the class it returns.

Or for finer control, edit `LABEL_TO_STRATEGY` to map specific regime labels to your class.

### 3. Run the tests

```bash
python -m pytest tests/test_strategies.py -v
```

The test file checks that every regime label in `REGIME_LABEL_SCHEMES` has a strategy mapping. If you add a new strategy class, this should still pass automatically as long as `_get_strategy_for_vol_rank` covers all ranks.

### 4. Backtest it

```bash
python main.py backtest --symbols SPY --start 2020-01-01 --end 2024-12-31 --compare
```

Look at the regime breakdown table to see how your strategy performed in each regime.

---

## Worked example: a 50 EMA + low-vol strategy

Goal: go long when price is above the 50 EMA **and** the HMM is in a low-vol regime, with a 2 ATR stop.

Add this class to `core/regime_strategies.py`:

```python
class FiftyEmaLowVolStrategy(BaseStrategy):
    """Long when price is above the 50 EMA in a low-vol regime.

    Stop: 2 ATR below entry.
    Position size: 80% of equity.
    """
    strategy_name = "fifty_ema_low_vol"

    def generate_signal(self, symbol, bars, regime_state):
        if len(bars) < 60:
            return None

        close = bars["close"]
        price = float(close.iloc[-1])

        # Compute 50 EMA and ATR
        ema50 = float(_ema(close, 50).iloc[-1])
        atr = float(_compute_atr(bars).iloc[-1])

        # Entry condition: price above 50 EMA
        if price <= ema50:
            return None

        # Stop: 2 ATR below entry, but never below the EMA itself
        stop = max(price - 2.0 * atr, ema50 - 0.5 * atr)

        return self._make_signal(
            symbol=symbol,
            direction=SignalDirection.LONG,
            entry_price=price,
            stop_loss=stop,
            regime_state=regime_state,
            reasoning=(
                f"Low-vol + price above 50 EMA: "
                f"price={price:.2f}, 50EMA={ema50:.2f}, ATR={atr:.2f}"
            ),
            position_size_pct=0.80,
            leverage=1.0,
        )
```

Now wire it in by editing `_get_strategy_for_vol_rank()`:

```python
def _get_strategy_for_vol_rank(rank, n_regimes):
    if n_regimes <= 1:
        return MidVolCautiousStrategy
    position = rank / (n_regimes - 1)
    if position <= 0.33:
        return FiftyEmaLowVolStrategy   # ← was LowVolBullStrategy
    elif position >= 0.67:
        return HighVolDefensiveStrategy
    else:
        return MidVolCautiousStrategy
```

That's it. The orchestrator will instantiate your class for every low-vol regime detected, run `generate_signal()` on each bar, and pass surviving `Signal`s to the risk manager.

---

## Things to remember

- **Stop loss is mandatory.** Returning a `Signal` with `stop_loss=None` or `stop_loss=entry_price` will get vetoed by the risk manager 100% of the time. There is no override.
- **The risk manager has the final say.** Even if you return `position_size_pct=1.0`, the risk manager caps it at `MAX_SINGLE_POSITION` (50%) and `MAX_PORTFOLIO_LEVERAGE` (1.25x). These are hardcoded — config can only tighten them, never loosen.
- **`generate_signal()` is called on every bar.** It needs to be idempotent: returning the same signal on consecutive bars while a position is already open is fine — the orchestrator and backtester deduplicate.
- **No look-ahead.** You receive `bars` sliced up to the current bar inclusive. Don't use `bars.shift(-1)` or anything that peeks forward. The look-ahead-bias tests will catch you (`tests/test_look_ahead.py`).
- **Helpers are in the same file.** `_compute_atr`, `_ema`, `_sma` are at the top of `core/regime_strategies.py`. For more elaborate features, see `data/feature_engineering.py`.
- **Test it.** Add a test in `tests/test_strategies.py` that constructs your strategy with synthetic bars and asserts the signal direction/size you expect. The existing tests are good templates.
