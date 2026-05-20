# Strategy Library — Design Document

This is a design review and plan for shipping 8 strategies alongside the regime trader template, including the architectural changes needed to support pairs trading.

**Status:** design only. No code is written yet.

---

## Step 1: Current System Documentation

### 1.1 `BaseStrategy` interface

Defined in `core/regime_strategies.py:92`. The contract is minimal:

```python
class BaseStrategy(ABC):
    strategy_name: str = "base"        # class attribute, identifies in logs

    def __init__(self, config: dict, regime_info: RegimeInfo) -> None:
        self._config = config           # passed-in strategy config dict
        self._regime_info = regime_info # one regime's RegimeInfo (NOT a list)

    @abstractmethod
    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState,
    ) -> Optional[Signal]:
        ...

    def _make_signal(self, symbol, direction, entry_price, stop_loss,
                     regime_state, reasoning, take_profit=None,
                     position_size_pct=None, leverage=None,
                     confidence_boost=0.0, **extra_metadata) -> Signal:
        # Helper that auto-fills regime fields. Falls back to RegimeInfo
        # defaults if position_size_pct or leverage are None.
```

Key facts:
- One strategy instance is bound to **one regime** (one `RegimeInfo` per instance). The orchestrator creates one instance per regime, all from the same class.
- `generate_signal` is called per-symbol per-bar by the orchestrator.
- Returning `None` means "do nothing." Returning a `Signal` with `direction=FLAT` means "close any existing position in this symbol."

### 1.2 `Signal` dataclass — every field

Defined at `core/regime_strategies.py:45`. All fields:

| Field | Type | Purpose | Set by |
|---|---|---|---|
| `symbol` | `str` | Ticker | Strategy |
| `direction` | `SignalDirection` | LONG / SHORT / FLAT | Strategy |
| `confidence` | `float` (0–1) | Usually `regime_state.probability + confidence_boost` | `_make_signal` |
| `entry_price` | `float` | Limit price the executor uses (with slippage offset) | Strategy |
| `stop_loss` | `float` | **Required, non-negotiable**. Long: below entry. Short: above. | Strategy |
| `take_profit` | `Optional[float]` | `None` = trail-stop only | Strategy |
| `position_size_pct` | `float` (0–1) | Suggested fraction of equity. Capped by risk manager. | Strategy or `RegimeInfo` default |
| `leverage` | `float` (1.0–1.25) | Capped by risk manager. | Strategy or `RegimeInfo` default |
| `regime_id` | `int` | Which regime is active | Auto-filled |
| `regime_name` | `str` | Label like "BULL", "CRASH" | Auto-filled |
| `regime_probability` | `float` | HMM posterior | Auto-filled |
| `timestamp` | `pd.Timestamp` | Bar timestamp | Auto-filled |
| `reasoning` | `str` | Free-text, appears in logs | Strategy |
| `strategy_name` | `str` | Class attribute | Auto-filled |
| `metadata` | `dict` | Strategy-specific extras (e.g. `{"trail_atr_mult": 2.5}`) | `**extra_metadata` |

### 1.3 `StrategyOrchestrator` — how strategies are picked

At `core/regime_strategies.py:276`. Flow:

1. On construction, takes the list of `RegimeInfo` from the trained HMM.
2. Sorts regimes by `expected_volatility` ascending, builds `_vol_rank: {regime_id: rank}`.
3. For each regime, calls `_get_strategy_for_vol_rank(rank, n_regimes)` to pick a class, then instantiates: `strat_cls(config, regime_info)`. One instance per regime, stored in `_strategies: {regime_id: BaseStrategy}`.
4. On each bar, `generate_signals(symbols, bars, regime_state, is_flickering)`:
   - Identifies the dominant regime (highest probability) and looks up its strategy.
   - Calls `dominant_strategy.generate_signal(symbol, bars[symbol], regime_state)` for each symbol.
   - If `is_flickering` or `regime_state.probability < min_confidence`, halves `position_size_pct` and forces `leverage=1.0` ("uncertainty mode").

So the orchestrator is **per-regime, single-symbol-loop**. It doesn't carry any concept of multi-asset coordination.

### 1.4 `docs/CUSTOM_STRATEGIES.md` — drift check

Read line by line against the current code. Findings:

- The flow description (HMM → orchestrator → generate_signal → risk manager) — **accurate.**
- The `BaseStrategy` skeleton example — **accurate.**
- The `Signal` field table — **accurate.**
- The orchestrator description (sort by vol, map to thirds) — **accurate.**
- The worked example (`FiftyEmaLowVolStrategy`) — **accurate.** Reads `_ema` and `_compute_atr` (both still exist), uses `_make_signal` with the right kwargs.
- The "Things to remember" section — **accurate** (stop-loss mandatory, risk manager final say, idempotent, no look-ahead, helpers in same file).

**No drift.** The doc still matches the code. We will need to update it when pairs land — specifically, add a "Writing pair strategies" section.

### 1.5 `core/risk_manager.py` — what strategies CAN'T do

Hardcoded ceilings in `core/risk_manager.py:53-64`. Any value a strategy puts in a `Signal` is clamped down to these — **the strategy can never request more, only less:**

| Limit | Constant | Value |
|---|---|---|
| Max single position | `MAX_SINGLE_POSITION` | 50% of equity |
| Max portfolio leverage | `MAX_PORTFOLIO_LEVERAGE` | 1.25x |
| Max risk per trade | `MAX_RISK_PER_TRADE` | 1% of equity |
| Max total exposure | `MAX_TOTAL_EXPOSURE` | 80% (20% cash floor) |
| Max correlated exposure | `MAX_CORRELATED_EXPOSURE` | 30% of correlated group |
| Max concurrent positions | `MAX_CONCURRENT_POSITIONS` | 5 |
| Max daily trades | `MAX_DAILY_TRADES` | 20 |
| Min position value | `MIN_POSITION_VALUE` | $100 |

**Validation steps inside `validate_signal()` (in order, any one rejects):**

0. Circuit breaker active → reject all.
1. `direction=FLAT` → pass-through (closes always allowed).
2. **Stop loss required.** `stop_loss is None or stop_loss <= 0` → reject. `abs(entry - stop) == 0` → reject.
3. Daily trade count >= 20 → reject.
4. Concurrent positions >= 5 (and signal opens a new symbol) → reject.
5. Duplicate same-symbol same-direction order within 60s → reject.
6. Total exposure >= 80% (and signal isn't reducing) → reject.
7. **Position sizing applied** (the strategy's `position_size_pct` is one input; `MAX_RISK_PER_TRADE` and stop distance another; `MAX_SINGLE_POSITION` is the cap; available cash is another bound). Final dollar size = `min(strategy_request, equity*max_single, risk_budget/risk_per_share*price, cash)`.
8. Circuit breaker size multiplier applied (e.g. 0.5x in DAILY_REDUCE).
9. Gap risk for overnight: capped at the size where a 3x stop gap would lose ≤ 2% of equity.
10. Correlation check vs existing positions:
    - corr >= 0.85 → reject
    - corr >= 0.70 → halve size
11. Min position value $100 → reject if below.
12. Leverage clamped: if no circuit breaker, regime is in `LEVERAGE_ALLOWED_REGIMES = {"NEUTRAL", "STRONG_BULL"}`, fewer than 3 concurrent positions, regime_probability ≥ 0.55, flicker rate ≤ 4 → leverage stays at the strategy's request, capped at 1.25x. Otherwise → forced to 1.0x.

**Strategy implications (what a strategy CAN'T fight):**

- Cannot ship a signal without a stop loss.
- Cannot get >50% of equity in a single name regardless of `position_size_pct=1.0`.
- Cannot use leverage outside NEUTRAL or STRONG_BULL regimes.
- Cannot open >5 concurrent positions (orchestrator generating 7 signals in one tick → only 5 fill).
- Cannot pile into correlated names — the 6th SPY-correlated long gets either reduced or rejected.
- Cannot trade more than 20 times per day.
- Cannot recover from `MIN_POSITION_VALUE` rejection — if equity*size_pct*price calculation is below $100, signal dies.

Strategies should size themselves under these caps so most signals pass without modification. Otherwise the risk manager's "modifications" pile up in logs.

### 1.6 Test patterns (`tests/test_strategies.py`)

Conventions:

1. **Synthetic bar generator** `_make_bars(n=200, trend="up"|"down"|"sideways", seed=42)` produces an OHLCV DataFrame. Use this — don't fetch live data.
2. **`_info(label, rid=0, vol=0.15)`** builds a `RegimeInfo` from `_REGIME_DEFAULTS`. Pass the vol explicitly because that's what determines orchestrator placement.
3. **`_state(label, sid=0, prob=0.7, n=3, confirmed=True)`** builds a `RegimeState` with the right `state_probabilities` array.
4. One `TestXxxStrategy` class per strategy, with focused tests:
   - `test_generates_xxx`: assert direction, position_size_pct, leverage match expectations
   - `test_has_stop_below_entry` (or above for shorts)
   - Edge cases: insufficient bars (< 60), trend mismatched to regime
5. `TestStrategyOrchestrator`: tests the orchestrator picks the right class given a vol-sorted RegimeInfo list and produces signals end-to-end.
6. Tests are fast — no Alpaca calls, no HMM training. Synthetic everything.

---

## Step 2: Multi-Asset Extension Design

### 2.1 Core idea

Add a second optional method on `BaseStrategy`:

```python
class BaseStrategy(ABC):
    strategy_name: str = "base"
    is_pair_strategy: bool = False     # NEW class attribute

    @abstractmethod
    def generate_signal(self, symbol, bars, regime_state) -> Optional[Signal]:
        """Single-asset method. Pair strategies stub this to return None."""
        ...

    def generate_pair_signal(
        self, pair: tuple[str, str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
    ) -> Optional["PairSignal"]:
        """Multi-asset method. Single-asset strategies don't override."""
        return None
```

`is_pair_strategy = True` on pair-strategy subclasses tells the orchestrator to call `generate_pair_signal` for configured pairs instead of `generate_signal` for each watchlist symbol.

### 2.2 `PairSignal` dataclass — proposed location

Lives in `core/regime_strategies.py` next to `Signal`. Two coordinated `Signal` objects bundled with pair-level metadata:

```python
@dataclass
class PairSignal:
    """Two coordinated leg Signals that must execute or be rejected together."""

    pair: tuple[str, str]              # ("SPY", "IWM")
    long_leg: Signal                    # one leg goes LONG
    short_leg: Signal                   # the other goes SHORT
    spread_value: float                 # current ratio or log-spread
    z_score: float                      # spread's z-score vs lookback
    hedge_ratio: float                  # short_qty / long_qty for $-neutrality
    correlation: Optional[float] = None # optional, for CorrelationRegimeAllocation
    reasoning: str = ""
    timestamp: Optional[pd.Timestamp] = None
    metadata: dict = field(default_factory=dict)
```

The two `Signal` objects retain all their normal fields (each leg has its own stop, target, size). `PairSignal` adds the pair-level context.

**Naming convention for legs:** `long_leg.symbol` and `short_leg.symbol` are always the actual long and short. So if z > 2 (spread is high → short the rich one, long the cheap one), `long_leg.symbol` is the cheap one.

### 2.3 Hedge ratio and dollar neutrality

For pair sizing we need a hedge ratio so the two legs are dollar-neutral (or beta-neutral). Two options:

- **Static dollar-neutral**: `long_qty * long_price ≈ short_qty * short_price`. Simple, works for high-correlation pairs like SPY/IWM.
- **OLS hedge ratio**: regress `log(price_a) ~ log(price_b)` on the lookback window, use the slope as the ratio. More accurate for less-correlated pairs.

For v1 I'd ship the OLS version as default with a config flag to fall back to static.

### 2.4 Atomicity — what if one leg fills and the other doesn't?

This is the hardest part. Alpaca does **not** support OCO/atomic execution across two different symbols — the existing `submit_bracket_order` is single-symbol only (entry + stop + take-profit on one ticker). We have to implement coordination ourselves.

**Failure modes to handle:**

1. **Leg A fills, leg B rejected by Alpaca** (e.g. insufficient buying power, halted symbol). We're naked long with no hedge. Bad.
2. **Leg A fills immediately, leg B partial fill or slow fill.** Temporary exposure mismatch.
3. **Leg A fills, leg B fills at much worse price than expected.** Spread is no longer at its z-score; entry edge gone.

**Proposed solution** — three-stage submit:

1. **Pre-check** both legs against the risk manager as a unit (see 2.5). If either fails, reject the pair entirely. No order goes to Alpaca.
2. **Submit both legs simultaneously** as paired limit orders with linked `trade_id` (already in `OrderResult`). Use `time_in_force=DAY` so unfilled legs don't haunt us.
3. **Within `cancel_after_seconds` (default 30s), monitor both fills.** If only one fills:
   - **If the unfilled leg's limit price is no longer reachable** (price moved away by more than 0.3% from limit): immediately submit a market order to **close the filled leg**. Better to take a small slippage cost than carry a naked directional position. Log as `pair_unwound`.
   - **If both eventually fill within the window**: log success.

This is best-effort, not true atomicity. We accept the slippage risk on the unwind. Worst case a pair entry ends with a small loss equal to the spread on the filled leg. Acceptable.

Add a new `OrderExecutor.submit_pair_order(pair_signal) -> tuple[OrderResult, OrderResult]` method that orchestrates this.

### 2.5 Risk manager: evaluating a `PairSignal` as a unit

Add `validate_pair_signal(pair_signal, portfolio, bars=None) -> RiskDecision`. Logic:

1. Run **both** legs through the existing single-leg checks (stops, regime, size caps, daily trade count, etc).
2. **If either leg is rejected, the whole pair is rejected.** Return one `RiskDecision` carrying the failure reason from whichever leg failed.
3. **Special pair-level rules:**
   - Both legs combined must fit within `MAX_TOTAL_EXPOSURE`.
   - Pair counts as **one** position toward `MAX_CONCURRENT_POSITIONS` (this is debatable — alternative is to count as 2). I'd argue 1 because it's a single conceptual trade.
   - Pair counts as **two** trades toward `MAX_DAILY_TRADES` (each leg is a real order).
   - The correlation check is **inverted**: for a pair, we *want* high correlation. Skip the standard correlation reject. But add a new check: if the two legs of the pair have correlation < 0.5 over the lookback, reject as "cointegration likely broken."
4. Return `RiskDecision` with both modified legs sized to the dollar-neutral hedge ratio.

### 2.6 Orchestrator changes

`StrategyOrchestrator.__init__` accepts an optional `pairs` config:

```python
def __init__(self, config, regime_infos):
    ...
    self._pairs: list[tuple[str, str]] = config.get("pairs", [])
```

`generate_signals` becomes:

```python
def generate_signals(self, symbols, bars, regime_state, is_flickering=False):
    dominant_strategy = self._strategies[regime_state.state_id]
    signals = []
    pair_signals = []

    if dominant_strategy.is_pair_strategy:
        for pair in self._pairs:
            if pair[0] not in bars or pair[1] not in bars:
                continue
            ps = dominant_strategy.generate_pair_signal(pair, bars, regime_state)
            if ps:
                pair_signals.append(ps)
    else:
        for symbol in symbols:
            ...  # unchanged single-asset path

    return signals, pair_signals  # Tuple instead of single list
```

This **breaks the orchestrator's return type signature.** Callers (main.py, backtester) need to handle `(signals, pair_signals)`. Discussed below in 2.7.

Alternative: a single mixed-type list. I prefer the tuple — explicit is better, and it lets callers decide whether they support pairs at all.

### 2.7 Integration points — where changes are needed

| File | Change |
|---|---|
| `core/regime_strategies.py` | Add `PairSignal` dataclass. Add `is_pair_strategy` class attr and `generate_pair_signal()` no-op default to `BaseStrategy`. Update `StrategyOrchestrator.generate_signals` to return `(signals, pair_signals)`. |
| `core/risk_manager.py` | Add `validate_pair_signal()` method. Optional helper `_check_pair_correlation()`. |
| `broker/order_executor.py` | Add `submit_pair_order(pair_signal)` that submits two limit orders with linked trade_id and unwinds the filled leg if the other leg doesn't fill within the window. |
| `main.py` (`TradingLoop._tick`) | Unpack `signals, pair_signals = orchestrator.generate_signals(...)`. Iterate `pair_signals` separately, calling `risk.validate_pair_signal()` then `executor.submit_pair_order()`. |
| `backtest/backtester.py` | Same unpack. Backtester also needs to track pair positions as a unit — it currently tracks a single `shares` int and `entry_price`. Multi-symbol position tracking needs to be added. **This is non-trivial.** |
| `broker/position_tracker.py` | Add a concept of "linked positions" so a closed leg automatically signals the pair as broken. Or: simpler — treat each leg as an independent position and let the strategy's exit logic close the other leg when its conditions trigger. |
| `tests/test_strategies.py` | Add a `_make_pair_bars(symbols, ...)` helper. Add a `TestPairSignal` class. |
| `tests/test_risk.py` | Add tests for `validate_pair_signal`. |
| `config/settings.yaml` | Add `pairs:` list under `strategy:`. |

### 2.8 Backtester rewrite — REQUIRED, not optional

> **DECISION (locked in):** Pair strategies must work in backtest. No "live only" cop-out. Don't ship anything that can't be backtested.

The current backtester (`backtest/backtester.py`) is **single-symbol** — it tracks one `shares: int`, one `entry_price`, one `entry_regime`. We rewrite it to multi-symbol with linked-position support.

**Refactor plan:**

```python
@dataclass
class SimPosition:
    symbol: str
    direction: str               # "long" or "short"
    qty: int                     # always positive; direction tracks sign
    entry_price: float
    entry_date: pd.Timestamp
    entry_regime: str
    stop_price: float
    target_price: Optional[float]
    pair_id: Optional[str] = None  # links sibling legs of a pair
    strategy_name: str = ""
    confidence: float = 0.0
```

Backtester state becomes:

```python
positions: dict[str, SimPosition] = {}  # keyed by symbol
pair_links: dict[str, str] = {}          # pair_id -> "SYMBOL_A,SYMBOL_B" for fast lookup
```

**Pair fill atomicity in backtest** — mirror live logic from §2.4:

1. On bar `t`, strategy emits a `PairSignal`. Both legs go into `pending_fills` queue tagged with the same `pair_id`.
2. On bar `t+1` (`fill_delay_bars=1`), attempt to fill both legs at the open with slippage.
3. **Both legs check `bar.high >= limit_price` (long leg) and `bar.low <= limit_price` (short leg) for fillability.** Pair limit orders fill only if the bar's range covers the limit price.
4. Three outcomes:
   - **Both fill:** create two `SimPosition`s with the shared `pair_id`. Normal pair entry.
   - **Neither fills:** drop both, no position opened. Same as a regular limit-order miss.
   - **One fills, one doesn't:** unwind the filled leg at the next bar's open with extra slippage (market-close penalty: `2x slippage_pct`). Log a `pair_unwind` trade with `pnl_pct` reflecting the slippage cost. **Loud warning log so the user notices a flaky pair.**

**Linked-position close coordination:**

When a strategy emits a `PairSignal` to close a pair (both legs FLAT), close both at the same bar at market. When one leg's stop is hit individually (disaster ATR stop fires on bar shock), the orchestrator should detect the orphan leg on the next bar and close it. To make this clean:

- After the stop check loop, scan `positions` for any `pair_id` where only one symbol remains. Force-close the orphan at the next bar's open with `2x slippage_pct` penalty. Log as `orphan_close`.

**Pair-level P&L attribution:**

Each leg writes its own `Trade` record (so single-symbol stats still work), but Trade gets a new field:

```python
@dataclass
class Trade:
    ...                                    # existing fields
    pair_id: Optional[str] = None          # NEW
    pair_pnl: Optional[float] = None       # NEW: aggregate pnl of both legs (filled in on second leg's close)
```

Performance reporting (`performance.py`) gets a `pair_breakdown(trades)` method analogous to `regime_breakdown`. Pairs show up as a single row with combined P&L, win rate over pair-trades (not leg-trades), avg holding period, etc.

**Equity-curve accuracy with multiple positions:**

The current backtester computes `equity = cash + shares * price` (one symbol). Multi-position version:

```python
equity = cash + sum(
    pos.qty * (1 if pos.direction == "long" else -1) * float(bars[pos.symbol].loc[date, "close"])
    for pos in positions.values()
)
```

For shorts, `cash` increases when entering (received from short sale) and decreases when covering. Standard accounting — no special case for pair shorts.

**Rebalancing logic:**

The current backtester does a delta-rebalance gate (`size_change > 10% AND alloc_change > 5%`). For multi-symbol, this gate applies per-symbol independently. Each `Signal` triggers its own rebalance check.

**Estimated work:** ~500 lines of changes in `backtest/backtester.py`, including a roughly proportional set of test additions. This is genuine work but not architecturally hard — the data structures are clear.

### 2.9 Design problems and locked-in resolutions

| Problem | Resolution |
|---|---|
| One-leg fills, other doesn't (live) | Best-effort unwind with market order on the filled leg. **Loud warning log on every unwind** — frequent unwinds signal a liquidity-asymmetric pair the user should reconsider. |
| One-leg fills, other doesn't (backtest) | Same as live: unwind at next bar's open with `2x slippage_pct` penalty. Log `pair_unwind`. |
| Pair strategy on a multi-asset orchestrator that doesn't have all pair symbols in `bars` | Skip that pair silently for the bar; log at DEBUG |
| Pair count toward `MAX_CONCURRENT_POSITIONS` | **1** (one conceptual trade) |
| Pair count toward `MAX_DAILY_TRADES` | **2** (each leg is a real order) |
| Correlation check on pair legs | Skip the standard `_check_correlation`. Add a separate "low correlation = break pair" reject inside `validate_pair_signal` that fires when the pair's own correlation < 0.5 over its lookback. |
| Backtester pair support | **Build it.** §2.8 covers the rewrite. Multi-symbol position tracking, linked positions, fill atomicity sim, pair P&L attribution. |
| Two pair strategies want to trade the same pair simultaneously | One strategy active per regime, orchestrator picks one. Cross-regime conflict possible if user maps two pair strategies to different vol ranks but same pair — document and don't try to merge. |

### 2.10 Backwards compatibility

All existing single-asset strategy code keeps working unchanged:
- `BaseStrategy.is_pair_strategy = False` is the default.
- `BaseStrategy.generate_pair_signal` returns `None` by default.
- `StrategyOrchestrator.generate_signals` returns a tuple `(signals, pair_signals)` — **this is a breaking change for callers.** Migration: callers do `signals, pair_signals = orch.generate_signals(...)` and ignore `pair_signals` if they don't care.

To avoid breaking the legacy `RegimeStrategyManager.get_signals()` call site:
- Keep `get_signals()` returning `list[Signal]` (signals only, no pairs).
- Add a new `get_signals_and_pairs()` that returns the tuple.
- Old callers stay happy. New callers (main.py, backtester) use the new method.

---

## Step 3: Per-Strategy Design

### Single-asset strategies (`generate_signal`)

#### 1. `TrendFollowingRegimeFilter`

| Aspect | Spec |
|---|---|
| Method | `generate_signal` |
| Direction | LONG only; FLAT in mid/high vol |
| Entry | `price > N-day MA` AND vol-rank == low (bottom third) |
| Exit | `price < N-day MA` (signal returns FLAT) |
| Data needed | `bars["close"]` for last `ma_period + 5` rows. Min 60 bars. |
| Stop | `max(price - stop_atr * ATR, MA_value - 0.5*ATR)` |
| Position size | 80% in low-vol, 0% in mid/high |
| Parameters | `ma_period: int = 200`, `stop_atr: float = 2.0` |
| Regime gating | The orchestrator *already* gates by vol rank — this strategy gets instantiated only for low-vol regime (if mapped via `_get_strategy_for_vol_rank`). It still defensive-checks vol rank inside in case of mid/high mapping. |
| Concerns | Basically a simpler `LowVolBullStrategy` with an MA filter. Similar enough that we should ship it as an alternative for users who want an explicit trend filter. |

#### 2. `MeanReversionLowVol`

| Aspect | Spec |
|---|---|
| Method | `generate_signal` |
| Direction | LONG on RSI < 30; FLAT (close) when RSI > 50; otherwise None |
| Entry | RSI(14) < `rsi_oversold` AND vol-rank == low |
| Exit | RSI(14) > `rsi_exit` |
| Data needed | `bars["close"]` for last 30+ rows. Min 60 bars. |
| Stop | `price - stop_atr * ATR` |
| Position size | 50% in low-vol, 0% otherwise |
| Parameters | `rsi_period: int = 14`, `rsi_oversold: int = 30`, `rsi_exit: int = 50`, `stop_atr: float = 2.0`, `position_size_pct: float = 0.50` |
| Regime gating | Returns None (i.e. do nothing) if not low-vol rank. |
| Concerns | RSI mean-reversion in a trending bull market loses money. The low-vol gate is doing real work — if the orchestrator wires this to a non-low regime, it just sits idle. That's fine but document. |

#### 3. `VolatilityBreakout`

| Aspect | Spec |
|---|---|
| Method | `generate_signal` |
| Direction | LONG on breakout above N-day high; FLAT in high-vol |
| Entry | `price > rolling_max(high, lookback)` AND vol-rank != high |
| Exit | None (trail stop only) |
| Data needed | `bars["high"]`, `bars["close"]`, ATR. Min 60 bars. |
| Stop | `price - stop_atr * ATR` (initial); ATR-trailing in subsequent bars |
| Position size | **Inversely scaled by ATR**: `target_risk_pct / (ATR / price)`. Higher vol → smaller position. Capped at `max_position_size`. |
| Parameters | `breakout_lookback: int = 20`, `stop_atr: float = 2.0`, `target_risk_pct: float = 0.01`, `max_position_size: float = 0.30` |
| Regime gating | Skip in high-vol rank (returns None). |
| Concerns | Inverse-ATR sizing means in calm markets the position is huge. Need to clamp to `max_position_size`. The risk manager will clamp further but document the intent. |

#### 4. `MomentumRotation`

| Aspect | Spec |
|---|---|
| Method | `generate_signal` (single-symbol method, but strategy holds full universe via `set_universe_bars`) |
| Direction | LONG for top-N performers, FLAT for everyone else |
| Entry | Symbol is in the current top-N ranked by `lookback`-day return |
| Exit | Symbol falls out of top-N — strategy emits FLAT for that symbol |
| Data needed | All watchlist bars (via `set_universe_bars`). Plus the per-symbol bars passed to `generate_signal`. |
| Stop | `price - 2*ATR` per leg |
| Position size | Equal-weight across top-N. `1/N` (e.g. top_n=3 → ~33% per name). The `MAX_SINGLE_POSITION` cap (50%) is well above this so risk manager won't clamp. |
| Parameters | `lookback: int = 60`, `top_n: int = 3`, `top_n_high_vol: int = 1`, `stop_atr: float = 2.0` |
| Regime gating | `top_n` shrinks in high-vol rank → fewer concurrent positions, more diversified cash. |
| **Stateless rebalance** | **No persisted state. No date tracking.** On every call: (1) Rank universe by trailing return. (2) Pick top-N (using `top_n` or `top_n_high_vol` based on current regime's vol rank). (3) For the symbol being asked about: if it's in top-N → emit LONG (or None if already held); if it's NOT in top-N → emit FLAT (so the orchestrator/risk manager closes the position if open). Survives restarts cleanly because behavior is purely a function of current data + current regime. |
| Implementation note | The orchestrator calls `generate_signal(symbol_X)` per symbol per bar. The strategy uses `self._universe_bars` (set via `set_universe_bars` at start of each bar) to compute the ranking. Ranking is recomputed every bar — that's fine, it's cheap (sort N closes). |
| Concerns | **No more "rebalance every 5 days" gate.** That means signals fire whenever ranking changes. To avoid churn, the risk manager's existing duplicate-order window (60s) handles intraday churn for live; for backtest the rebalance gate (`size_change > 10% AND alloc_change > 5%`) absorbs noise. If churn is still excessive in backtest, add a `min_rank_change` parameter (e.g. only signal a swap if the rank change is >=2 positions). Default off. |

#### 5. `DefensiveLongShort`

| Aspect | Spec |
|---|---|
| Method | `generate_signal` (single-symbol method, full universe via `set_universe_bars`) |
| Direction | LONG for top-momentum names always; SHORT for bottom-momentum names only in high-vol |
| Entry (long leg) | Top-N by `lookback`-day return |
| Entry (short leg) | Bottom-M by `lookback`-day return AND vol-rank == high |
| Exit | Symbol falls out of its bucket → emit FLAT for that symbol (stateless, same pattern as MomentumRotation) |
| Data needed | All watchlist bars (via `set_universe_bars`). |
| Stop | `price - 2*ATR` for longs, `price + 2*ATR` for shorts |
| Position size | Longs: `1/N * 0.6`. Shorts: `1/M * 0.4`. |
| Parameters | `lookback: int = 60`, `top_n: int = 3`, `bottom_m: int = 2`, `short_in_regimes: list = ["high"]`, `stop_atr: float = 2.0` |
| Regime gating | Shorts only fire in high-vol; longs always |
| **Stateless rebalance** | Same as MomentumRotation. No date tracking. Ranking recomputed every bar from `self._universe_bars`. If symbol is in top-N → LONG. If symbol is in bottom-M and high-vol → SHORT. Else → FLAT. |
| Concerns | Same as MomentumRotation — needs `set_universe_bars()`. Also: **shorts in non-NEUTRAL/STRONG_BULL regimes will get leverage forced to 1.0x.** That's fine since shorts here are unleveraged hedges. Document. Also: this isn't a pair strategy — each long and short is independently signaled, the user said so. So it goes through `generate_signal`, just sometimes returning SHORT direction. |

#### 6. `PureRegimeAllocation`

| Aspect | Spec |
|---|---|
| Method | `generate_signal` |
| Direction | LONG always (or FLAT in high-vol); never SHORT |
| Entry | Always for every watchlist symbol when invested; FLAT when high-vol |
| Exit | High-vol regime |
| Data needed | One symbol's bars (just to read close price). Min 60 bars for stop calc. |
| Stop | `price - 3*ATR` (loose stop — this is a passive allocator, not an entry strategy) |
| Position size | `gross_exposure / N_watchlist`. Low-vol → 100%/N. Mid → 50%/N. High → 0. |
| Parameters | `low_vol_gross: float = 1.0`, `mid_vol_gross: float = 0.50`, `high_vol_gross: float = 0.0`, `n_watchlist_hint: int = 6` |
| Regime gating | The whole strategy IS regime-gated. |
| Concerns | Needs to know `N` watchlist size to divide. Either pass via config (`n_watchlist_hint`) or via `set_universe_bars()`. **Same multi-asset interface issue.** Recommend using `set_universe_bars()` to grab the count dynamically. |

### Pair strategies (`generate_pair_signal`)

#### 7. `StaticPairsZScore`

| Aspect | Spec |
|---|---|
| Method | `generate_pair_signal` |
| `is_pair_strategy` | `True` |
| Spread definition | `log(price_a) - hedge_ratio * log(price_b)`. Hedge ratio = OLS slope of log(a) vs log(b) on lookback window. |
| Z-score | `(spread - mean(spread, lookback)) / std(spread, lookback)` |
| Entry | `|z| > entry_z`. If z > entry_z (spread is high), short A, long B. If z < -entry_z, long A, short B. |
| Exit | `|z| < exit_z` (mean-reverted). Generates a `PairSignal` with both legs FLAT. |
| Hard stop | `|z| > stop_z` (cointegration broke) — close at market |
| Data needed | Both symbols' close series, min `lookback + 5` bars |
| Stop loss per leg | `entry_price ± stop_atr * ATR` (each leg has its own ATR-based stop, so risk manager accepts) |
| Position size per leg | `pair_pct / 2` (half on each leg, dollar-neutral via hedge ratio) |
| Parameters | `lookback: int = 60`, `entry_z: float = 2.0`, `exit_z: float = 0.5`, `stop_z: float = 4.0`, `pair_pct: float = 0.20` |
| Regime gating | Only generates signals in low-vol rank. Returns None otherwise. |
| Concerns | **The risk manager requires a stop_loss on every Signal.** The pair-level z-score stop doesn't translate to a price-level stop trivially. Resolution: each leg gets a price-level stop derived from the z=4 condition (compute what price would push the spread to z=4 given the other leg holds), OR each leg gets a generic ATR-based stop and the strategy itself monitors z and emits a FLAT pair_signal when z exceeds 4. Latter is simpler and cleaner — the leg-level stop is just disaster protection (e.g. 5*ATR). |

#### 8. `CorrelationRegimeAllocation`

| Aspect | Spec |
|---|---|
| Method | `generate_pair_signal` |
| `is_pair_strategy` | `True` |
| Trigger | Rolling 60-day correlation > `entry_corr` |
| Direction | Same as StaticPairsZScore — fade the spread when |z| > 1.5 |
| Sizing | Position size scales with correlation: `pair_pct * (corr - exit_corr) / (1 - exit_corr)`. At corr=1.0 → full size. At corr=`exit_corr` → zero. |
| Exit | Correlation falls below `exit_corr` |
| Data needed | Both symbols' close series, min 70 bars |
| Stop loss per leg | ATR-based (5*ATR — same disaster-protection logic as #7) |
| Parameters | `corr_lookback: int = 60`, `entry_corr: float = 0.7`, `exit_corr: float = 0.5`, `entry_z: float = 1.5`, `pair_pct: float = 0.20` |
| Regime gating | Low-vol only (same as #7) |
| Concerns | Position size *changes* over time as correlation degrades. The orchestrator currently calls `generate_pair_signal` per bar — strategy should re-emit a `PairSignal` with new sizing each time correlation crosses a meaningful threshold. The risk manager needs to handle rebalancing, not just open/close. **Open question:** does the executor support resizing existing positions, or only entry/exit? Looking at `broker/order_executor.py`: `modify_stop` exists but no `resize_position`. Resolution: emit FLAT pair-signal first, then a new sized entry. Two trades, but clean. Document the pattern. |

### Default pairs config (settings.yaml)

```yaml
strategy:
  pairs:
    - [SPY, IWM]
    - [XLF, KRE]
    - [GLD, SLV]
    - [QQQ, SPY]
```

User can override. Document that pairs should be cointegrated and have similar liquidity.

---

## Step 4: Summary and Decisions Needed

**Concrete file plan:**

| File | New / Modified | Purpose |
|---|---|---|
| `core/regime_strategies.py` | Modified | Add `PairSignal`, `is_pair_strategy` attr, `generate_pair_signal()` default, `set_universe_bars()` default. Update `StrategyOrchestrator.generate_signals` return type. |
| `core/strategies/__init__.py` | New | Export the 8 new strategy classes |
| `core/strategies/trend_following.py` | New | `TrendFollowingRegimeFilter` |
| `core/strategies/mean_reversion.py` | New | `MeanReversionLowVol` |
| `core/strategies/volatility_breakout.py` | New | `VolatilityBreakout` |
| `core/strategies/momentum_rotation.py` | New | `MomentumRotation` |
| `core/strategies/defensive_long_short.py` | New | `DefensiveLongShort` |
| `core/strategies/pure_regime_allocation.py` | New | `PureRegimeAllocation` |
| `core/strategies/pairs_zscore.py` | New | `StaticPairsZScore` |
| `core/strategies/correlation_regime.py` | New | `CorrelationRegimeAllocation` |
| `core/risk_manager.py` | Modified | Add `validate_pair_signal()` |
| `broker/order_executor.py` | Modified | Add `submit_pair_order()` with unwind logic |
| `main.py` | Modified | Update `_tick` to handle pair signals |
| `backtest/backtester.py` | Major rewrite | Multi-symbol position tracking, linked positions, pair fill atomicity, pair P&L (§2.8) |
| `tests/test_strategies/` | New directory | One test file per strategy + `test_pair_signal.py` |
| `tests/test_risk.py` | Modified | Add `validate_pair_signal` tests |
| `docs/CUSTOM_STRATEGIES.md` | Modified | Add "Writing pair strategies" section |
| `docs/STRATEGY_LIBRARY.md` | New | User-facing reference for the 8 strategies |
| `config/settings.yaml` | Modified | Add `strategy.pairs` list |

**Locked-in decisions (approved 2026-04-27):**

1. **Pair counts:** 1 toward `MAX_CONCURRENT_POSITIONS`, 2 toward `MAX_DAILY_TRADES`. ✓
2. **Backtester pair support:** **Build it.** Multi-symbol position tracking, linked positions, atomicity sim, pair P&L attribution. **Don't ship anything that can't be backtested.** §2.8 has the design. ✓
3. **Orchestrator return type:** tuple `(signals, pair_signals)`. ✓
4. **Multi-asset data hook:** `set_universe_bars(bars: dict)` on `BaseStrategy`. ✓
5. **Pair leg stops:** ATR-based disaster stops (5*ATR). Strategy monitors z-score itself for graceful exit. ✓
6. **Pair unwind on partial fill:** market-close the filled leg. **Loud warning log on every unwind event** so users notice flaky pairs. ✓
7. **Strategy file layout:** `core/strategies/*.py`, one file per strategy. ✓
8. **`MomentumRotation` (and `DefensiveLongShort`):** stateless rebalance — compare current top-N composition against current holdings each bar, signal adds/drops accordingly. No date tracking, no persisted state, survives restarts cleanly. ✓

---

## Implementation Plan — Phased Rollout

This is too large for one PR. Proposed phases, each ending in a green test suite and a logical checkpoint:

### Phase A — Foundation (no new strategies yet)
**Why first:** Lays the rails everything else rides on. Ships behind a compatible interface, single-asset path keeps working unchanged.

- Add `PairSignal` dataclass to `core/regime_strategies.py`
- Add `is_pair_strategy: bool = False` and `generate_pair_signal() -> None` default to `BaseStrategy`
- Add `set_universe_bars(bars: dict) -> None` default to `BaseStrategy` (store and ignore)
- Update `StrategyOrchestrator.generate_signals()` return type to `tuple[list[Signal], list[PairSignal]]`
- Keep `RegimeStrategyManager.get_signals()` returning `list[Signal]` (legacy) + add `get_signals_and_pairs()` for new callers
- Update `main.py` `TradingLoop._tick` to unpack tuple, iterate pair signals (still no-op since no pair strategies exist yet)
- Update existing single-asset tests to match new return type
- **Ship checkpoint:** Test suite green; behavior identical for existing users.

### Phase B — Backtester rewrite (no new strategies yet)
**Why next:** Foundation is in but pairs can't be backtested without this. Single-asset backtesting must keep working identically (regression test the current numbers).

- Refactor `backtest/backtester.py` to multi-symbol position tracking (`positions: dict[str, SimPosition]`)
- Add linked-positions support (`pair_id` on `SimPosition` and `Trade`)
- Add pair fill atomicity simulation
- Add orphan-leg detection and force-close
- Add `pair_pnl` attribution in trade log
- Add `PerformanceAnalyzer.pair_breakdown()`
- Backtester regression test: re-run SPY backtest, confirm metrics unchanged (or document any drift)
- **Ship checkpoint:** Existing single-asset backtest produces same numbers. Backtester ready to support pairs.

### Phase C — Risk manager + executor pair support
**Why next:** All the plumbing for pairs is in. Now the validation/execution path.

- Add `RiskManager.validate_pair_signal()` (§2.5)
- Add `OrderExecutor.submit_pair_order()` with unwind logic (§2.4) and loud unwind logging
- Update `main.py` to call these for each `PairSignal`
- Tests for both new methods (mocked executor)
- **Ship checkpoint:** Pair pipeline complete end-to-end. Still no real pair strategies, but you could ship one and it would route correctly.

### Phase D — Single-asset strategies (1-6)
Order them so the simplest land first and validate the foundation:
1. `PureRegimeAllocation` (validates `set_universe_bars` works end-to-end)
2. `TrendFollowingRegimeFilter`
3. `MeanReversionLowVol`
4. `VolatilityBreakout`
5. `MomentumRotation` (stateless)
6. `DefensiveLongShort` (stateless, both directions)

Each strategy lands as: source file + tests + brief docs entry. ~200-300 lines per PR.

### Phase E — Pair strategies (7-8)
7. `StaticPairsZScore`
8. `CorrelationRegimeAllocation`

Each: source file + backtester end-to-end test (now that B is done) + docs.

### Phase F — Documentation + config
- Update `docs/CUSTOM_STRATEGIES.md` with "Writing pair strategies" section
- New `docs/STRATEGY_LIBRARY.md` with usage and parameters for all 8
- Update `config/settings.yaml` with example pair config and strategy mapping

### Recommendation

Start with **Phase A**. It's a small, contained change that unblocks everything else and lets us land the new types and orchestrator return signature without dragging strategies and backtester rewrites along. After Phase A is reviewed and merged, Phase B (backtester) is the next big lift.

If you want to go faster: A and B can land together since they're both "foundation" work that doesn't ship user-visible features. C-E are clearly separable.

Tell me which phase to start with — A alone, A+B together, or jump elsewhere — and I'll begin.