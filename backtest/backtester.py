"""Walk-forward optimization (WFO) backtesting engine — multi-symbol.

Phase B: rewritten to track multiple positions simultaneously via
``positions: dict[str, SimPosition]``. Single-asset behavior is preserved
arithmetically; the SPY 2018→today baseline produces the same equity-curve
SHA-256 as Phase A.

Pair strategies are handled via a separate path:
- A ``PairSignal`` queues both legs into ``pending_pair_fills`` for the next bar.
- On the next bar, both legs attempt to fill at the open with slippage.
- If both fill: create two ``SimPosition`` records sharing a ``pair_id``.
- If neither: drop both, no position opened.
- If exactly one fills: unwind the filled leg at the next bar's open with
  doubled slippage, log loud ``pair_unwind`` warning.
- If a single leg of an existing pair gets force-closed (orphan), close the
  surviving leg at the next bar's open with doubled slippage.

Single-asset signals continue to rebalance at the close on the same bar
without slippage — this preserves the Phase A baseline numerics.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .performance import PerformanceAnalyzer, PerformanceMetrics

logger = logging.getLogger(__name__)


def _compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _slip_fill(price: float, buying: bool, slippage: float) -> float:
    """Apply slippage to a fill price (B2): buys fill above, sells below."""
    return price * (1.0 + slippage) if buying else price * (1.0 - slippage)


# ---------------------------------------------------------------------------
# Per-symbol simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimPosition:
    """Per-symbol state for the multi-symbol backtester.

    A SimPosition is created lazily on first signal for a symbol. It persists
    across bars even when ``qty == 0`` (flat) so per-symbol metadata like
    ``prev_alloc`` is preserved between rebalances.

    Fields prefixed with ``entry_`` carry the *most recent* rebalance metadata
    — they're overwritten each time the position is resized in the same
    direction. The ``entry_confidence`` field is also updated on every bar
    where a signal for this symbol exists, even without rebalancing — this
    matches the legacy single-asset accounting.
    """

    symbol: str
    qty: int = 0                                # signed: + long, - short, 0 flat
    entry_price: float = 0.0
    entry_date: Optional[pd.Timestamp] = None
    entry_regime: str = ""
    entry_confidence: float = 0.0
    prev_alloc: float = 0.0                     # last applied target allocation (for rebalance gate)
    stop_price: float = 0.0
    target_price: Optional[float] = None
    pair_id: Optional[str] = None               # links sibling legs of a pair trade
    strategy_name: str = ""

    @property
    def direction(self) -> str:
        if self.qty > 0:
            return "long"
        if self.qty < 0:
            return "short"
        return "flat"

    @property
    def is_flat(self) -> bool:
        return self.qty == 0


@dataclass
class PendingPairFill:
    """A pair signal queued for next-bar fill simulation."""

    pair_id: str
    long_symbol: str
    short_symbol: str
    long_entry: float                           # signal's intended entry price (for limit check)
    short_entry: float
    long_qty: int                               # share counts (signed appropriately at fill time)
    short_qty: int
    long_stop: float
    short_stop: float
    regime: str
    confidence: float
    strategy_name: str
    queued_at: pd.Timestamp


# ---------------------------------------------------------------------------
# Trade record — extended for pairs
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A completed trade. Extended in Phase B with pair_id/pair_pnl.

    Both new fields default to None so single-asset trades and the trade-log
    CSV format remain unchanged when no pair signals are active.
    """

    symbol: str
    direction: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    pnl_pct: float
    regime: str
    confidence: float
    strategy: str
    stop_price: float = 0.0
    trade_id: str = ""
    pair_id: Optional[str] = None               # NEW: shared between sibling legs
    pair_pnl: Optional[float] = None            # NEW: aggregate PnL of both legs (set on second leg's close)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    regime_history: pd.DataFrame
    metrics: PerformanceMetrics
    walk_forward_windows: list[dict]
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fill simulator (used by the pair path)
# ---------------------------------------------------------------------------

class FillSimulator:
    def __init__(self, slippage_pct=0.0005, commission=0.0, fill_delay_bars=1):
        self._slippage = slippage_pct
        self._commission = commission
        self._fill_delay = fill_delay_bars

    def get_fill_price(self, signal_price, direction, fill_bar):
        bar_open = float(fill_bar["open"])
        if direction == "long":
            return bar_open * (1 + self._slippage)
        else:
            return bar_open * (1 - self._slippage)

    def check_stop(self, stop_price, direction, bar):
        bar_open, bar_low, bar_high = float(bar["open"]), float(bar["low"]), float(bar["high"])
        if direction == "long":
            if bar_open <= stop_price:
                return bar_open * (1 - self._slippage)
            if bar_low <= stop_price:
                return stop_price * (1 - self._slippage)
        else:
            if bar_open >= stop_price:
                return bar_open * (1 + self._slippage)
            if bar_high >= stop_price:
                return stop_price * (1 + self._slippage)
        return None

    def check_target(self, target_price, direction, bar):
        if direction == "long" and float(bar["high"]) >= target_price:
            return target_price * (1 - self._slippage)
        if direction == "short" and float(bar["low"]) <= target_price:
            return target_price * (1 + self._slippage)
        return None

    @property
    def fill_delay(self):
        return self._fill_delay

    @property
    def commission(self):
        return self._commission


# ---------------------------------------------------------------------------
# Walk-forward backtester — multi-symbol
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """Regime-adaptive allocation backtester with multi-symbol support.

    The single-asset code path remains numerically equivalent to the Phase A
    baseline. The multi-symbol path runs the same per-symbol rebalance logic
    independently for each signaled symbol. The pair path uses queue-based
    next-bar atomic fills with leg-unwind on partial fills.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        bt = config.get("backtest", {})
        wf = bt.get("walk_forward", {})

        self._train_window = wf.get("train_window", 252)
        self._test_window = wf.get("test_window", 63)
        self._step_size = wf.get("step_size", 63)
        self._initial_capital = bt.get("initial_capital", 100_000)
        self._slippage = bt.get("slippage_pct", 0.0005)
        self._commission = bt.get("commission", 0.0)
        self._rf = bt.get("risk_free_rate", 0.045)

        self._fill_sim = FillSimulator(self._slippage, self._commission)

    # ------------------------------------------------------------------
    # Equity helper — works for single-symbol (preserves SPY hash) and
    # multi-symbol (sums across positions).
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_to_market(
        cash: float,
        positions: dict[str, "SimPosition"],
        bars: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> float:
        """Equity = cash + sum(qty * close) across all open positions.

        For symbols with no bar at the current date, fall back to the most
        recent prior close. For SPY-only single-asset backtests this branch
        is never hit and the arithmetic reduces to ``cash + qty * close``
        identical to the Phase A code.
        """
        equity = cash
        for sym, pos in positions.items():
            if pos.qty == 0:
                continue
            sym_bars = bars.get(sym)
            if sym_bars is None:
                # No bars for this symbol at all — use entry price (no MTM change)
                equity += pos.qty * pos.entry_price
                continue
            if date in sym_bars.index:
                equity += pos.qty * float(sym_bars.loc[date, "close"])
            else:
                # Use most recent prior close
                prior = sym_bars[sym_bars.index <= date]
                if len(prior) > 0:
                    equity += pos.qty * float(prior["close"].iloc[-1])
                else:
                    equity += pos.qty * pos.entry_price
        return equity

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, bars: dict[str, pd.DataFrame],
            hmm_features: pd.DataFrame) -> BacktestResult:
        from core.hmm_engine import HMMEngine
        from core.regime_strategies import StrategyOrchestrator, SignalDirection, PairSignal

        index = hmm_features.index
        windows = self._generate_windows(index)
        if not windows:
            raise ValueError(f"Insufficient data: need {self._train_window + self._test_window}, got {len(index)}")

        logger.info(
            "Walk-forward: %d windows, train=%d, test=%d, step=%d, bars=%d",
            len(windows), self._train_window, self._test_window, self._step_size, len(index),
        )

        oos_start = windows[0]["test_start_idx"]
        oos_end = windows[-1]["test_end_idx"]
        retrain_points = {w["test_start_idx"]: w for w in windows}

        hmm = HMMEngine(self._config.get("hmm", {}))
        orch = None
        hmm_fitted = False

        # ref_sym is used for HMM regime detection (always same single symbol's bars
        # produce the features); per-symbol price lookups go through `bars[sym]`.
        ref_sym = list(bars.keys())[0]
        ref_bars = bars[ref_sym]

        # --- New multi-symbol state ---
        cash: float = float(self._initial_capital)
        positions: dict[str, SimPosition] = {}
        pending_pair_fills: list[PendingPairFill] = []

        # Per-pair tracking: pair_id -> (long_symbol, short_symbol)
        # Used to detect orphans (one leg closed, the other still open).
        pair_links: dict[str, tuple[str, str]] = {}

        equity: float = cash
        trades: list[Trade] = []
        equity_points: list[tuple] = []
        regime_history: list[dict] = []
        window_results: list[dict] = []
        win_id = 0
        win_start_eq = equity
        win_trades = 0

        for t in range(oos_start, oos_end):
            date = index[t]

            # --- Retrain at window boundary ---
            if t in retrain_points:
                win = retrain_points[t]
                train_data = hmm_features.iloc[win["train_start_idx"]:win["train_end_idx"]]
                try:
                    hmm.fit(train_data)
                    hmm.reset_tracking()
                    orch = StrategyOrchestrator(self._config.get("strategy", {}), hmm.regime_infos)
                    hmm_fitted = True
                    logger.info("Window %d: retrained HMM (%d regimes)", win_id, hmm.n_regimes)
                except Exception as e:
                    logger.warning("HMM train failed window %d: %s", win_id, e)

                if win_id > 0:
                    window_results.append({
                        "window_id": win_id - 1, "n_trades": win_trades,
                        "pnl": equity - win_start_eq,
                        "return_pct": (equity / win_start_eq - 1) if win_start_eq > 0 else 0,
                        "n_regimes": hmm.n_regimes if hmm_fitted else 0,
                        # Attach the dates of the window being CLOSED (win_id-1),
                        # not `win` (the one we're now retraining) — using `win`
                        # was an off-by-one that mislabeled every window's range.
                        **windows[win_id - 1],
                    })
                win_start_eq = equity
                win_trades = 0
                win_id += 1

            if not hmm_fitted or orch is None:
                equity_points.append((date, equity))
                continue

            if date not in ref_bars.index:
                equity_points.append((date, equity))
                continue

            # --- HMM prediction ---
            features_now = hmm_features.iloc[:t + 1]
            try:
                regime = hmm.predict_regime_filtered(features_now)
            except Exception:
                equity_points.append((date, equity))
                continue

            regime_history.append({
                "date": date, "regime": regime.label,
                "probability": regime.probability,
                "confirmed": regime.is_confirmed, "state_id": regime.state_id,
            })

            # --- Process pending pair fills queued from prior bar ---
            if pending_pair_fills:
                still_pending: list[PendingPairFill] = []
                for pending in pending_pair_fills:
                    pair_filled, pair_trades = self._attempt_pair_fill(
                        pending, bars, date, positions, regime,
                    )
                    if pair_filled is None:
                        # Both legs unfillable on this bar — drop (limit window of 1 bar)
                        logger.debug(
                            "Pair %s: neither leg fillable on %s, dropping",
                            pending.pair_id, date,
                        )
                        continue
                    cash_delta, link = pair_filled
                    cash += cash_delta
                    if link is not None:
                        pair_links[pending.pair_id] = link
                    win_trades += len(pair_trades)
                    trades.extend(pair_trades)
                pending_pair_fills = still_pending

            # --- Generate signals via orchestrator ---
            # Pass all available bars so multi-asset strategies can see the universe.
            bars_now: dict[str, pd.DataFrame] = {}
            for sym, sym_bars in bars.items():
                sliced = sym_bars[sym_bars.index <= date]
                if len(sliced) >= 50:
                    bars_now[sym] = sliced
            if ref_sym not in bars_now:
                equity_points.append((date, equity))
                continue

            signals, pair_signals = orch.generate_signals(
                list(bars_now.keys()), bars_now, regime,
                is_flickering=hmm.is_flickering(),
            )

            # --- Single-asset signal handling ---
            for sig in signals:
                # Get price for this signal's symbol
                if sig.symbol not in bars_now:
                    continue
                if date not in bars[sig.symbol].index:
                    continue
                price = float(bars[sig.symbol].loc[date, "close"])

                # Get/create per-symbol state
                pos = positions.setdefault(sig.symbol, SimPosition(symbol=sig.symbol))

                # --- Compute target allocation from this signal ---
                if sig.direction == SignalDirection.LONG:
                    target_alloc = sig.position_size_pct * sig.leverage
                elif sig.direction == SignalDirection.SHORT:
                    target_alloc = -sig.position_size_pct * sig.leverage
                else:
                    target_alloc = 0.0

                # Update per-bar metadata that legacy code set on every signal,
                # not just on rebalance bars (entry_confidence in Phase A).
                pos.entry_confidence = sig.confidence
                pos.strategy_name = sig.strategy_name

                # Mark to market BEFORE sizing (matches Phase A line 261).
                # For SPY-only this is `cash + qty * price`.
                equity = self._mark_to_market(cash, positions, bars, date)

                target_shares_new = int(equity * target_alloc / price) if price > 0 else 0
                size_change = abs(target_shares_new - pos.qty)

                # Rebalance gate (matches Phase A line 268)
                if size_change > abs(pos.qty) * 0.10 + 1 and abs(target_alloc - pos.prev_alloc) > 0.05:
                    # Trade-record gate: only record when closing/flipping an existing position
                    if pos.qty != 0 and abs(target_alloc - pos.prev_alloc) > 0.15:
                        # Exit fill with slippage (B2): closing a long sells
                        # (fills below), closing a short buys (fills above).
                        exit_price = _slip_fill(price, buying=(pos.qty < 0), slippage=self._slippage)
                        pnl = (exit_price - pos.entry_price) * pos.qty
                        denom = abs(pos.entry_price * pos.qty)
                        trades.append(Trade(
                            symbol=sig.symbol,
                            direction="long" if pos.qty > 0 else "short",
                            entry_date=pos.entry_date or date,
                            exit_date=date,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            qty=abs(pos.qty),
                            pnl=pnl,
                            pnl_pct=pnl / denom if denom > 0 else 0,
                            regime=pos.entry_regime,
                            confidence=pos.entry_confidence,
                            strategy="regime_alloc",
                            stop_price=pos.stop_price,  # B4: record the stop in effect
                        ))
                        win_trades += 1

                    # Apply rebalance with slippage on the fill (B2): buys fill
                    # above the close, sells below.
                    delta = target_shares_new - pos.qty
                    fill_price = _slip_fill(price, buying=(delta > 0), slippage=self._slippage)
                    cash -= delta * fill_price
                    pos.qty = target_shares_new
                    pos.entry_price = fill_price
                    pos.stop_price = sig.stop_loss      # B4: capture the signal's stop
                    pos.entry_date = date
                    pos.entry_regime = regime.label
                    pos.prev_alloc = target_alloc

            # --- Pair signal handling (Phase B new path) ---
            for ps in pair_signals:
                self._enqueue_pair_fill(ps, bars_now, date, equity, pending_pair_fills)

            # --- Orphan-leg closure: if one leg of a pair is open and its
            # sibling isn't, force-close the orphan at the next bar's open with
            # doubled slippage.
            for orphan_sym, link_pair_id in list(self._iter_orphans(pair_links, positions)):
                pos = positions.get(orphan_sym)
                if pos is None or pos.qty == 0:
                    continue
                # Use next bar's open if available, else current close
                exit_price_raw = self._next_bar_open(bars.get(orphan_sym), date)
                if exit_price_raw is None and date in bars.get(orphan_sym, pd.DataFrame()).index:
                    exit_price_raw = float(bars[orphan_sym].loc[date, "close"])
                if exit_price_raw is None:
                    continue
                # Doubled slippage on the unwind
                slip_dir = -1 if pos.qty > 0 else 1
                exit_price = exit_price_raw * (1 + slip_dir * 2 * self._slippage)
                pnl = (exit_price - pos.entry_price) * pos.qty
                denom = abs(pos.entry_price * pos.qty)
                logger.warning(
                    "ORPHAN_CLOSE pair_id=%s symbol=%s qty=%d entry=%.2f exit=%.2f pnl=%.2f",
                    link_pair_id, orphan_sym, pos.qty, pos.entry_price, exit_price, pnl,
                )
                trades.append(Trade(
                    symbol=orphan_sym,
                    direction="long" if pos.qty > 0 else "short",
                    entry_date=pos.entry_date or date,
                    exit_date=date,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    qty=abs(pos.qty),
                    pnl=pnl,
                    pnl_pct=pnl / denom if denom > 0 else 0,
                    regime=pos.entry_regime,
                    confidence=pos.entry_confidence,
                    strategy=pos.strategy_name + "_orphan",
                    pair_id=link_pair_id,
                ))
                cash += pos.qty * exit_price
                pos.qty = 0
                pos.pair_id = None
                win_trades += 1
                # Remove the link since both legs are now closed
                pair_links.pop(link_pair_id, None)

            # --- End-of-bar mark-to-market and equity record ---
            equity = self._mark_to_market(cash, positions, bars, date)
            equity_points.append((date, equity))

        # --- Close any remaining open positions at the last bar ---
        last_date = index[min(oos_end - 1, len(index) - 1)]
        for sym, pos in list(positions.items()):
            if pos.qty == 0:
                continue
            sym_bars = bars.get(sym)
            if sym_bars is not None and last_date in sym_bars.index:
                close_px = float(sym_bars.loc[last_date, "close"])
                # Slippage on the final close for single-asset positions (B2);
                # pair positions keep their existing fill handling.
                if pos.pair_id is None:
                    exit_price = _slip_fill(close_px, buying=(pos.qty < 0), slippage=self._slippage)
                else:
                    exit_price = close_px
            else:
                exit_price = pos.entry_price
            pnl = (exit_price - pos.entry_price) * pos.qty
            denom = abs(pos.entry_price * pos.qty)
            trades.append(Trade(
                symbol=sym,
                direction="long" if pos.qty > 0 else "short",
                entry_date=pos.entry_date or last_date,
                exit_date=last_date,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                qty=abs(pos.qty),
                pnl=pnl,
                pnl_pct=pnl / denom if denom > 0 else 0,
                regime=pos.entry_regime,
                confidence=pos.entry_confidence,
                # Single-asset rebalance trades and the end-of-loop close
                # both use the generic "regime_alloc" label to match the
                # Phase A baseline. Pair-related trades carry the actual
                # strategy_name (set elsewhere).
                strategy="regime_alloc" if pos.pair_id is None else (pos.strategy_name or "regime_alloc"),
                stop_price=pos.stop_price,
                pair_id=pos.pair_id,
            ))
            cash += pos.qty * exit_price
            pos.qty = 0

        # --- Pair PnL attribution: for each closed pair, fill in pair_pnl on both legs ---
        self._attribute_pair_pnl(trades)

        # --- Final window record ---
        final_rec = {
            "window_id": win_id - 1, "n_trades": win_trades,
            "pnl": equity - win_start_eq,
            "return_pct": (equity / win_start_eq - 1) if win_start_eq > 0 else 0,
            "n_regimes": hmm.n_regimes if hmm_fitted else 0,
        }
        # Include the final window's dates (previously omitted, so the last
        # window printed with no date range).
        if 0 <= win_id - 1 < len(windows):
            final_rec.update(windows[win_id - 1])
        window_results.append(final_rec)

        # --- Build output ---
        eq_series = pd.Series({d: v for d, v in equity_points}, name="equity").sort_index()
        eq_series = eq_series[~eq_series.index.duplicated(keep="last")]
        trades_df = self._trades_to_df(trades)
        regime_df = pd.DataFrame(regime_history)

        analyzer = PerformanceAnalyzer(self._rf)
        metrics = analyzer.analyze(eq_series, trades_df)

        logger.info(
            "Backtest complete: %d windows, %d trades, return=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
            len(windows), len(trades), metrics.total_return * 100,
            metrics.sharpe_ratio, metrics.max_drawdown * 100,
        )

        return BacktestResult(
            equity_curve=eq_series, trades=trades_df,
            regime_history=regime_df, metrics=metrics,
            walk_forward_windows=window_results,
            config=self._config.get("backtest", {}),
        )

    # ------------------------------------------------------------------
    # Pair-fill helpers
    # ------------------------------------------------------------------

    def _enqueue_pair_fill(
        self,
        pair_signal,
        bars_now: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        equity: float,
        queue: list[PendingPairFill],
    ) -> None:
        """Convert a PairSignal to a PendingPairFill and queue it for next bar."""
        long_leg = pair_signal.long_leg
        short_leg = pair_signal.short_leg

        if long_leg.symbol not in bars_now or short_leg.symbol not in bars_now:
            return

        # Size each leg using equity * position_size_pct / leg_price.
        # The PairSignal carries the hedge ratio implicitly through each leg's
        # position_size_pct (the strategy is expected to split sizing already).
        long_price = long_leg.entry_price
        short_price = short_leg.entry_price
        if long_price <= 0 or short_price <= 0:
            return

        long_qty = max(1, int(equity * long_leg.position_size_pct * long_leg.leverage / long_price))
        short_qty = max(1, int(equity * short_leg.position_size_pct * short_leg.leverage / short_price))

        pair_id = f"PAIR-{uuid.uuid4().hex[:8]}"
        queue.append(PendingPairFill(
            pair_id=pair_id,
            long_symbol=long_leg.symbol,
            short_symbol=short_leg.symbol,
            long_entry=long_price,
            short_entry=short_price,
            long_qty=long_qty,
            short_qty=short_qty,
            long_stop=long_leg.stop_loss,
            short_stop=short_leg.stop_loss,
            regime=long_leg.regime_name,
            confidence=long_leg.confidence,
            strategy_name=long_leg.strategy_name,
            queued_at=date,
        ))
        logger.info(
            "Pair queued: %s long %d @ ~$%.2f / short %s %d @ ~$%.2f (pair_id=%s)",
            long_leg.symbol, long_qty, long_price,
            short_leg.symbol, short_qty, short_price, pair_id,
        )

    def _attempt_pair_fill(
        self,
        pending: PendingPairFill,
        bars: dict[str, pd.DataFrame],
        fill_date: pd.Timestamp,
        positions: dict[str, SimPosition],
        regime,
    ) -> tuple[Optional[tuple[float, Optional[tuple[str, str]]]], list[Trade]]:
        """Try to fill both legs of a pending pair on `fill_date`.

        Returns:
          (None, []) — both legs unfillable; caller drops the pending fill.
          ((cash_delta, link_or_None), trades) — at least one leg attempted.
            cash_delta is the net cash change for whatever filled.
            link is (long_sym, short_sym) if both filled (a real pair was opened),
              or None if only one filled (it was unwound).
            trades is the list of Trade records produced by this attempt
              (currently empty for the open path; unwind produces 1 record).
        """
        long_bar = self._bar_at(bars.get(pending.long_symbol), fill_date)
        short_bar = self._bar_at(bars.get(pending.short_symbol), fill_date)

        if long_bar is None or short_bar is None:
            return None, []

        # Limit-order fillability: long leg fills if its limit is within the bar's range
        long_fill = self._try_fill_leg(pending.long_entry, "long", long_bar)
        short_fill = self._try_fill_leg(pending.short_entry, "short", short_bar)

        cash_delta = 0.0
        unwind_trades: list[Trade] = []

        if long_fill is not None and short_fill is not None:
            # Both legs fill → open coordinated position
            positions[pending.long_symbol] = SimPosition(
                symbol=pending.long_symbol,
                qty=pending.long_qty,
                entry_price=long_fill,
                entry_date=fill_date,
                entry_regime=pending.regime,
                entry_confidence=pending.confidence,
                stop_price=pending.long_stop,
                pair_id=pending.pair_id,
                strategy_name=pending.strategy_name,
            )
            positions[pending.short_symbol] = SimPosition(
                symbol=pending.short_symbol,
                qty=-pending.short_qty,
                entry_price=short_fill,
                entry_date=fill_date,
                entry_regime=pending.regime,
                entry_confidence=pending.confidence,
                stop_price=pending.short_stop,
                pair_id=pending.pair_id,
                strategy_name=pending.strategy_name,
            )
            cash_delta -= pending.long_qty * long_fill          # paid for longs
            cash_delta += pending.short_qty * short_fill        # received for shorts
            return (cash_delta, (pending.long_symbol, pending.short_symbol)), []

        if long_fill is not None and short_fill is None:
            # Long filled, short didn't → unwind the long at next-bar open with 2x slippage
            unwind_price, unwind_pnl = self._unwind_filled_leg(
                pending.long_symbol, "long", pending.long_qty,
                long_fill, bars, fill_date,
            )
            cash_delta -= pending.long_qty * long_fill          # paid for it
            cash_delta += pending.long_qty * unwind_price       # got it back
            logger.warning(
                "PAIR_UNWIND pair_id=%s — long %s filled @ $%.2f but short %s didn't fill. "
                "Unwound long at $%.2f (PnL=$%.2f). Frequent unwinds suggest a liquidity-asymmetric pair.",
                pending.pair_id, pending.long_symbol, long_fill,
                pending.short_symbol, unwind_price, unwind_pnl,
            )
            unwind_trades.append(Trade(
                symbol=pending.long_symbol, direction="long",
                entry_date=fill_date, exit_date=fill_date,
                entry_price=long_fill, exit_price=unwind_price,
                qty=pending.long_qty, pnl=unwind_pnl,
                pnl_pct=unwind_pnl / (long_fill * pending.long_qty) if long_fill > 0 else 0,
                regime=pending.regime, confidence=pending.confidence,
                strategy=pending.strategy_name + "_unwind",
                pair_id=pending.pair_id,
            ))
            return (cash_delta, None), unwind_trades

        if short_fill is not None and long_fill is None:
            # Short filled, long didn't → unwind the short at next-bar open with 2x slippage
            unwind_price, unwind_pnl = self._unwind_filled_leg(
                pending.short_symbol, "short", pending.short_qty,
                short_fill, bars, fill_date,
            )
            cash_delta += pending.short_qty * short_fill        # received from short sale
            cash_delta -= pending.short_qty * unwind_price      # paid to cover
            logger.warning(
                "PAIR_UNWIND pair_id=%s — short %s filled @ $%.2f but long %s didn't fill. "
                "Unwound short at $%.2f (PnL=$%.2f). Frequent unwinds suggest a liquidity-asymmetric pair.",
                pending.pair_id, pending.short_symbol, short_fill,
                pending.long_symbol, unwind_price, unwind_pnl,
            )
            unwind_trades.append(Trade(
                symbol=pending.short_symbol, direction="short",
                entry_date=fill_date, exit_date=fill_date,
                entry_price=short_fill, exit_price=unwind_price,
                qty=pending.short_qty, pnl=unwind_pnl,
                pnl_pct=unwind_pnl / (short_fill * pending.short_qty) if short_fill > 0 else 0,
                regime=pending.regime, confidence=pending.confidence,
                strategy=pending.strategy_name + "_unwind",
                pair_id=pending.pair_id,
            ))
            return (cash_delta, None), unwind_trades

        # Neither fills
        return None, []

    def _try_fill_leg(self, limit_price: float, direction: str, bar: pd.Series) -> Optional[float]:
        """Decide if a limit order at `limit_price` fills on this bar."""
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        bar_open = float(bar["open"])
        if direction == "long":
            # Long limit fills if bar low touches or goes below the limit.
            # Filled price = max(limit, open) with slippage.
            if bar_low <= limit_price:
                fill = max(limit_price, bar_open)
                return fill * (1 + self._slippage)
            return None
        else:
            # Short limit fills if bar high touches or exceeds the limit.
            if bar_high >= limit_price:
                fill = min(limit_price, bar_open)
                return fill * (1 - self._slippage)
            return None

    def _unwind_filled_leg(
        self,
        symbol: str,
        direction: str,
        qty: int,
        fill_price: float,
        bars: dict[str, pd.DataFrame],
        fill_date: pd.Timestamp,
    ) -> tuple[float, float]:
        """Market-close an orphaned filled leg at next bar's open (2x slippage).

        Returns (unwind_price, pnl).
        """
        sym_bars = bars.get(symbol)
        next_open = self._next_bar_open(sym_bars, fill_date)
        if next_open is None:
            # No next bar — unwind at current bar's close with doubled slippage
            if sym_bars is not None and fill_date in sym_bars.index:
                next_open = float(sym_bars.loc[fill_date, "close"])
            else:
                next_open = fill_price                            # fallback: no slippage
        # Doubled slippage in the worst-case direction
        if direction == "long":
            unwind_price = next_open * (1 - 2 * self._slippage)
            pnl = (unwind_price - fill_price) * qty
        else:
            unwind_price = next_open * (1 + 2 * self._slippage)
            pnl = (fill_price - unwind_price) * qty
        return unwind_price, pnl

    @staticmethod
    def _bar_at(sym_bars: Optional[pd.DataFrame], date: pd.Timestamp) -> Optional[pd.Series]:
        if sym_bars is None or date not in sym_bars.index:
            return None
        return sym_bars.loc[date]

    @staticmethod
    def _next_bar_open(sym_bars: Optional[pd.DataFrame], date: pd.Timestamp) -> Optional[float]:
        if sym_bars is None:
            return None
        future = sym_bars[sym_bars.index > date]
        if len(future) == 0:
            return None
        return float(future["open"].iloc[0])

    # ------------------------------------------------------------------
    # Orphan detection — pair member closed independently while sibling open
    # ------------------------------------------------------------------

    def _iter_orphans(
        self,
        pair_links: dict[str, tuple[str, str]],
        positions: dict[str, SimPosition],
    ):
        """Yield (orphan_symbol, pair_id) for any pair where exactly one leg is open."""
        for pair_id, (long_sym, short_sym) in pair_links.items():
            long_open = (long_sym in positions and positions[long_sym].qty != 0
                         and positions[long_sym].pair_id == pair_id)
            short_open = (short_sym in positions and positions[short_sym].qty != 0
                          and positions[short_sym].pair_id == pair_id)
            if long_open and not short_open:
                yield long_sym, pair_id
            elif short_open and not long_open:
                yield short_sym, pair_id

    # ------------------------------------------------------------------
    # Pair PnL attribution
    # ------------------------------------------------------------------

    @staticmethod
    def _attribute_pair_pnl(trades: list[Trade]) -> None:
        """For each pair_id with two legs in the trade log, compute aggregate
        P&L and write it onto both legs' ``pair_pnl`` field."""
        from collections import defaultdict
        groups: dict[str, list[Trade]] = defaultdict(list)
        for tr in trades:
            if tr.pair_id is not None:
                groups[tr.pair_id].append(tr)
        for pair_id, legs in groups.items():
            total = sum(leg.pnl for leg in legs)
            for leg in legs:
                leg.pair_pnl = total

    # ------------------------------------------------------------------
    # Window generation (unchanged)
    # ------------------------------------------------------------------

    def _generate_windows(self, index):
        n = len(index)
        min_total = self._train_window + self._test_window
        if n < min_total:
            return []
        windows = []
        start = 0
        while start + min_total <= n:
            train_end = start + self._train_window
            test_end = min(train_end + self._test_window, n)
            windows.append({
                "train_start": index[start], "train_end": index[train_end - 1],
                "test_start": index[train_end], "test_end": index[test_end - 1],
                "train_start_idx": start, "train_end_idx": train_end,
                "test_start_idx": train_end, "test_end_idx": test_end,
            })
            start += self._step_size
            if test_end >= n:
                break
        return windows

    # ------------------------------------------------------------------
    # Trade DataFrame — conditional pair columns to preserve baseline hash
    # ------------------------------------------------------------------

    @staticmethod
    def _trades_to_df(trades):
        """Build the trade-log DataFrame.

        For backwards compatibility (and to preserve the Phase A baseline
        SHA-256), the legacy 13 columns are emitted when no pair trades exist.
        When at least one trade has a non-null ``pair_id``, two extra columns
        (``pair_id``, ``pair_pnl``) are appended.
        """
        legacy_cols = [
            "symbol", "direction", "entry_date", "exit_date",
            "entry_price", "exit_price", "qty", "pnl", "pnl_pct",
            "regime", "confidence", "strategy", "stop_price",
        ]
        if not trades:
            return pd.DataFrame(columns=legacy_cols)

        any_pair = any(t.pair_id is not None for t in trades)

        def _row(t):
            base = {
                "symbol": t.symbol, "direction": t.direction,
                "entry_date": t.entry_date, "exit_date": t.exit_date,
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "qty": t.qty, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
                "regime": t.regime, "confidence": t.confidence,
                "strategy": t.strategy, "stop_price": t.stop_price,
            }
            if any_pair:
                base["pair_id"] = t.pair_id
                base["pair_pnl"] = t.pair_pnl
            return base

        return pd.DataFrame([_row(t) for t in trades])

    def save_results(self, result, output_config):
        for key in ["equity_curve_csv", "trade_log_csv", "regime_history_csv"]:
            path = output_config.get(key)
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        eq_path = output_config.get("equity_curve_csv")
        if eq_path:
            result.equity_curve.to_csv(eq_path, header=True)
        trade_path = output_config.get("trade_log_csv")
        if trade_path:
            result.trades.to_csv(trade_path, index=False)
        regime_path = output_config.get("regime_history_csv")
        if regime_path:
            result.regime_history.to_csv(regime_path, index=False)
