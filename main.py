"""Regime Trader — Entry point and main trading loop.

Ties together every component: HMM regime detection, strategy selection,
risk management, order execution, and position tracking.

STARTUP: config → Alpaca connection → HMM load/train → risk init → sync positions → loop
LOOP:    new bar → features → HMM predict → strategy signals → risk validate → execute
SHUTDOWN: save state → close streams → log summary (positions stay open with stops)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal as signal_mod
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

logger: Optional[logging.Logger] = None

# ---------------------------------------------------------------------------
# State snapshot paths
# ---------------------------------------------------------------------------

CRASH_LOG_DIR = Path("logs")
MODEL_DIR = Path("models")


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_credentials(cred_path: str = "config/credentials.yaml") -> dict:
    path = Path(cred_path)
    if not path.exists():
        # Credentials can also come purely from .env, so this is a soft warning
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def init_logging(config: dict) -> logging.Logger:
    from monitoring.logger import setup_logger
    mon = config.get("monitoring", {})
    return setup_logger(
        name="regime_trader",
        log_file=None,  # main.log path is derived from log_dir
        level=mon.get("log_level", "INFO"),
        log_dir=mon.get("log_dir", "logs"),
        json_files=mon.get("json_log_files", True),
    )


# ---------------------------------------------------------------------------
# Crash dump
# ---------------------------------------------------------------------------

def write_crash_dump(error: Exception, system_state: dict) -> Path:
    """Write full system state to a crash log file."""
    CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CRASH_LOG_DIR / f"crash_{ts}.json"
    dump = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "system_state": _serialize_state(system_state),
    }
    try:
        path.write_text(json.dumps(dump, indent=2, default=str))
    except Exception:
        pass
    return path


def _serialize_state(state: dict) -> dict:
    """Best-effort JSON serialization of system state."""
    out = {}
    for k, v in state.items():
        try:
            json.dumps(v, default=str)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# State snapshot for recovery
# ---------------------------------------------------------------------------

def save_state_snapshot(path: str, state: dict) -> None:
    try:
        Path(path).write_text(json.dumps(_serialize_state(state), indent=2, default=str))
        if logger:
            logger.info("State snapshot saved to %s", path)
    except Exception as e:
        if logger:
            logger.error("Failed to save state snapshot: %s", e)


def load_state_snapshot(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if logger:
            logger.info("Loaded state snapshot from %s", path)
        return data
    except Exception as e:
        if logger:
            logger.warning("Failed to load state snapshot: %s", e)
        return None


# ---------------------------------------------------------------------------
# HMM model management
# ---------------------------------------------------------------------------

def load_or_train_hmm(
    config: dict,
    market_data,
    feature_engineer,
    dry_run: bool = False,
):
    """Load a recent saved model or train a new one.

    Returns (hmm_engine, hmm_features_df).
    """
    from core.hmm_engine import HMMEngine

    hmm_config = config.get("hmm", {})
    model_config = config.get("model", {})
    model_dir = Path(model_config.get("save_dir", "models"))
    max_age = model_config.get("max_age_days", 7)
    model_path = model_dir / "hmm_latest.pkl"

    engine = HMMEngine(hmm_config)

    # Try loading existing model
    if model_path.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)).days
        if age_days <= max_age:
            try:
                engine.load(model_path)
                logger.info(
                    "Loaded saved HMM model (%d days old, %d regimes): %s",
                    age_days, engine.n_regimes, engine.state_label_map,
                )
                # Still need features for prediction — fetch and compute
                features = _fetch_and_compute_features(config, market_data, feature_engineer)
                return engine, features
            except Exception as e:
                logger.warning("Failed to load saved model, will retrain: %s", e)
        else:
            logger.info("Saved model is %d days old (max %d), retraining.", age_days, max_age)

    # Train new model
    logger.info("Training new HMM model...")
    features = _fetch_and_compute_features(config, market_data, feature_engineer)

    if features is None or len(features) < hmm_config.get("min_train_bars", 504):
        raise RuntimeError(
            f"Insufficient data for HMM training: got {len(features) if features is not None else 0} "
            f"bars, need {hmm_config.get('min_train_bars', 504)}"
        )

    metrics = engine.fit(features)

    logger.info(
        "HMM trained: %d regimes selected by BIC. Regime breakdown:",
        metrics.n_regimes_selected,
    )
    for info in engine.regime_infos:
        logger.info(
            "  %s (id=%d): E[ret]=%.3f, E[vol]=%.3f, strategy=%s, max_lev=%.2f",
            info.regime_name, info.regime_id, info.expected_return,
            info.expected_volatility, info.recommended_strategy_type,
            info.max_leverage_allowed,
        )
    logger.info("BIC scores: %s", metrics.bic_scores)

    # Save model
    if not dry_run:
        try:
            engine.save(model_path)
        except Exception as e:
            logger.warning("Failed to save model: %s", e)

    return engine, features


def _fetch_and_compute_features(config, market_data, feature_engineer) -> Optional[pd.DataFrame]:
    """Fetch daily bars for the reference symbol and compute HMM features."""
    symbols = config.get("universe", {}).get("symbols", ["SPY"])
    ref_symbol = symbols[0]  # Use first symbol as regime reference
    logger.info("Fetching daily bars for %s (HMM reference symbol)...", ref_symbol)

    bars = market_data.get_historical_bars(
        ref_symbol,
        timeframe="1Day",
        limit=config.get("universe", {}).get("lookback_bars", 504),
    )
    if bars is None or len(bars) == 0:
        return None

    logger.info("Fetched %d daily bars for %s (%s to %s)",
                len(bars), ref_symbol, bars.index[0], bars.index[-1])

    features = feature_engineer.compute_hmm_features(bars)
    logger.info("Computed %d feature rows (%d columns)", len(features), features.shape[1])
    return features


# ---------------------------------------------------------------------------
# Main trading loop
# ---------------------------------------------------------------------------

class TradingLoop:
    """Orchestrates the main bar-by-bar trading loop.

    On each bar:
    1. Compute features (rolling, no future data)
    2. HMM filtered prediction → regime probabilities
    3. Stability filter check
    4. Strategy signals via StrategyOrchestrator
    5. Risk validation for each signal
    6. Order execution for approved signals
    7. Trailing stop updates
    8. Circuit breaker check
    """

    def __init__(
        self,
        config: dict,
        alpaca_client,
        hmm_engine,
        feature_engineer,
        strategy_manager,
        risk_manager,
        order_executor,
        position_tracker,
        market_data,
        hmm_features: pd.DataFrame,
        dry_run: bool = False,
    ) -> None:
        self._config = config
        self._client = alpaca_client
        self._hmm = hmm_engine
        self._fe = feature_engineer
        self._strategies = strategy_manager
        self._risk = risk_manager
        self._executor = order_executor
        self._tracker = position_tracker
        self._market_data = market_data
        self._hmm_features = hmm_features
        self._dry_run = dry_run

        self._symbols = config.get("universe", {}).get("symbols", ["SPY"])
        self._ref_symbol = self._symbols[0]
        self._signal_tf = config.get("universe", {}).get("signal_timeframe", "5Min")
        self._loop_interval = config.get("schedule", {}).get("loop_interval_seconds", 60)
        self._state_path = config.get("model", {}).get("state_snapshot", "state_snapshot.json")

        self._running = False
        self._session_start = pd.Timestamp.now()
        self._trade_count = 0
        self._regime_change_count = 0
        self._last_regime_label: Optional[str] = None
        self._bar_count = 0
        # Trading-day tracker for daily/weekly risk-window resets (A3); None
        # until the first tick observes a date.
        self._last_trading_date: Optional[pd.Timestamp] = None
        # Trailing-stop ATR multiple (A2): each bar the broker stop is tightened
        # toward price -/+ mult*ATR. <= 0 disables trailing.
        self._trail_atr_mult: float = config.get("execution", {}).get("trail_stop_atr", 2.0)

    def run(self) -> None:
        """Run the trading loop until interrupted."""
        self._running = True
        logger.info("Entering main trading loop (interval=%ds, dry_run=%s)",
                     self._loop_interval, self._dry_run)

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Error in trading loop tick: %s", e, exc_info=True)
                self._handle_loop_error(e)

            # Wait for next tick
            if self._running:
                time.sleep(self._loop_interval)

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._running = False

    def _maybe_reset_periods(self, today: pd.Timestamp, equity: float) -> None:
        """Reset daily/weekly risk windows when the calendar rolls over (A3).

        The circuit breaker measures daily/weekly drawdown from the day/week
        start equity, and the risk manager caps trades per day — both drift if
        never reset (e.g. "daily" DD measured from process start, daily trade
        count never clearing). Called once per tick; a no-op until the date
        actually advances. ``today`` is a normalized (midnight) trading date.
        """
        if self._last_trading_date is None:
            self._last_trading_date = today
            return
        if today <= self._last_trading_date:
            return
        prev = self._last_trading_date
        self._last_trading_date = today
        # New ISO week → weekly reset (which also re-bases the daily window);
        # otherwise a new day → daily reset.
        if today.isocalendar().week != prev.isocalendar().week or (today - prev).days >= 7:
            logger.info("New trading week (%s -> %s): resetting weekly risk window.",
                        prev.date(), today.date())
            self._risk.reset_weekly(equity)
        else:
            logger.info("New trading day (%s -> %s): resetting daily risk window.",
                        prev.date(), today.date())
            self._risk.reset_daily(equity)

    def _trail_stops(self, all_bars: dict) -> None:
        """Trail the broker-side protective stop for each open position (A2).

        For each position with a live stop at the broker, compute a candidate
        stop at ``price -/+ trail_atr_mult * ATR`` and tighten the resting stop
        toward it. modify_stop() enforces stop-only-tightens; we also gate here
        to skip a redundant API call when the candidate isn't tighter. No-op
        when trailing is disabled (trail_stop_atr <= 0).
        """
        if self._trail_atr_mult <= 0:
            return
        from data.feature_engineering import atr as _atr

        for sym, pos in self._tracker.positions.items():
            if getattr(pos, "qty", 0) == 0:
                continue
            bars = all_bars.get(sym)
            if bars is None or len(bars) < 20:
                continue
            stop_info = self._executor.get_open_stop_order(sym)
            if not stop_info:
                continue
            try:
                price = float(bars["close"].iloc[-1])
                atr_val = float(_atr(bars).iloc[-1])
            except Exception:
                continue
            if atr_val <= 0 or pd.isna(atr_val):
                continue

            cur_stop = stop_info["stop_price"]
            if pos.side == "long":
                desired = round(price - self._trail_atr_mult * atr_val, 2)
                tighter = desired > cur_stop
                pos_side = "buy"
            else:
                desired = round(price + self._trail_atr_mult * atr_val, 2)
                tighter = desired < cur_stop
                pos_side = "sell"
            if not tighter:
                continue

            res = self._executor.modify_stop(
                stop_info["order_id"], desired, current_side=pos_side)
            logger.info("Trailed stop %s: %.2f -> %.2f (%s)",
                        sym, cur_stop, desired, res.status.value)

    def _tick(self) -> None:
        """Execute one iteration of the trading loop."""
        self._bar_count += 1

        # --- 1. Check market status ---
        try:
            if not self._client.is_market_open():
                if self._bar_count % 10 == 1:  # Log every 10th tick when closed
                    clock = self._client.get_clock()
                    logger.info("Market closed. Next open: %s", clock.get("next_open"))
                return
        except Exception as e:
            logger.warning("Failed to check market status: %s", e)

        # --- 2. Get account and position state ---
        try:
            account = self._client.get_account()
            equity = account.equity
        except Exception as e:
            logger.error("Failed to get account info: %s. Skipping tick.", e)
            return

        self._risk.update_equity_tracking(equity)

        # Roll over daily/weekly risk windows when the calendar advances (A3).
        today_et = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        self._maybe_reset_periods(today_et, equity)

        # Sync positions
        try:
            self._tracker.sync_with_broker()
        except Exception as e:
            logger.warning("Position sync failed: %s", e)

        positions = self._tracker.get_position_values()

        # --- 3. Fetch latest bars and compute features ---
        try:
            ref_bars = self._market_data.get_historical_bars(
                self._ref_symbol, timeframe="1Day")
            if ref_bars is None or len(ref_bars) < 50:
                logger.warning("Insufficient bars for %s, skipping tick", self._ref_symbol)
                return
            hmm_features = self._fe.compute_hmm_features(ref_bars)
            if len(hmm_features) == 0:
                logger.warning("Feature computation returned empty, skipping tick")
                return
            self._hmm_features = hmm_features
        except Exception as e:
            logger.error("Feature computation failed: %s. Holding current regime.", e)
            return

        # --- 4. HMM filtered prediction (forward algorithm ONLY) ---
        try:
            regime_state = self._hmm.predict_regime_filtered(self._hmm_features)
        except Exception as e:
            logger.error("HMM prediction failed: %s. Holding current regime.", e)
            return

        # Track regime changes
        if self._last_regime_label and regime_state.label != self._last_regime_label:
            if regime_state.is_confirmed:
                self._regime_change_count += 1
        self._last_regime_label = regime_state.label

        is_flickering = self._hmm.is_flickering()

        if self._bar_count % 5 == 1:
            logger.info(
                "Regime: %s (prob=%.2f, confirmed=%s, stability=%d, flicker=%.0f) | "
                "Equity: $%.2f | Positions: %d",
                regime_state.label, regime_state.probability,
                regime_state.is_confirmed, regime_state.consecutive_bars,
                self._hmm.get_regime_flicker_rate(),
                equity, len(positions),
            )

        # --- 5. Check HMM refit ---
        if self._hmm.should_refit():
            logger.info("HMM refit triggered after %d bars since last fit",
                        self._hmm._bars_since_refit)
            try:
                self._hmm.fit(self._hmm_features)
                self._strategies.set_regime_infos(self._hmm.regime_infos)
                logger.info("HMM refitted: %d regimes", self._hmm.n_regimes)
            except Exception as e:
                logger.error("HMM refit failed: %s. Continuing with current model.", e)

        # --- 6. Fetch signal-timeframe bars for all symbols ---
        all_bars: dict[str, pd.DataFrame] = {}
        for sym in self._symbols:
            try:
                bars = self._market_data.get_historical_bars(
                    sym, timeframe=self._signal_tf,
                    limit=self._config.get("universe", {}).get("signal_lookback_bars", 200))
                if bars is not None and len(bars) >= 50:
                    all_bars[sym] = bars
            except Exception as e:
                logger.debug("Failed to fetch bars for %s: %s", sym, e)

        if not all_bars:
            logger.warning("No bars available for any symbol, skipping tick")
            return

        # --- 7. Generate signals via StrategyOrchestrator ---
        # Returns (single-asset signals, pair signals). In Phase A we collect
        # both but pair_signals will always be empty until Phase C wires up
        # validate_pair_signal / submit_pair_order.
        try:
            signals, pair_signals = self._strategies.get_signals_and_pairs(
                regime_state, list(all_bars.keys()), all_bars,
                is_flickering=is_flickering)
        except Exception as e:
            logger.error("Strategy signal generation failed: %s", e)
            signals, pair_signals = [], []


        # --- 8. Build portfolio state for risk manager ---
        from core.risk_manager import PortfolioState
        cb_status = self._risk.circuit_breaker.check(equity, regime_state.label)

        portfolio = PortfolioState(
            equity=equity,
            cash=account.cash,
            buying_power=account.buying_power,
            positions=positions,
            position_count=len(positions),
            daily_pnl=cb_status.daily_pnl,
            weekly_pnl=cb_status.weekly_pnl,
            peak_equity=self._risk.circuit_breaker._peak_equity,
            day_start_equity=self._risk.circuit_breaker._day_start_equity,
            week_start_equity=self._risk.circuit_breaker._week_start_equity,
            current_drawdown_pct=cb_status.peak_drawdown_pct,
            total_exposure=sum(abs(v) for v in positions.values()) / equity if equity > 0 else 0,
            max_single_exposure=max((abs(v) for v in positions.values()), default=0) / equity if equity > 0 else 0,
            daily_trade_count=self._risk._daily_trade_count,
            circuit_breaker=cb_status,
            regime_label=regime_state.label,
            regime_probability=regime_state.probability,
            flicker_rate=self._hmm.get_regime_flicker_rate(),
        )

        # --- 9. Circuit breaker actions ---
        if cb_status.is_halted:
            logger.warning("CIRCUIT BREAKER HALT: %s", cb_status.halt_reason)
            if not self._dry_run:
                self._executor.close_all_positions()
                self._tracker.close_all()
            return

        # --- 10. Validate and execute each signal ---
        for sig in signals:
            from core.regime_strategies import SignalDirection
            if sig.direction == SignalDirection.FLAT:
                continue

            decision = self._risk.validate_signal(
                sig, portfolio, bars=all_bars,
                is_overnight=not self._client.is_market_open())

            if not decision.approved:
                logger.info("Signal rejected [%s %s]: %s",
                            sig.symbol, sig.direction.value, decision.rejection_reason)
                continue

            if decision.modifications_made:
                logger.info("Signal modified [%s %s]: %s",
                            sig.symbol, sig.direction.value,
                            "; ".join(decision.modifications_made))

            # Execute
            if self._dry_run:
                logger.info(
                    "DRY RUN — would place: %s %s %d shares @ $%.2f "
                    "(stop=$%.2f, regime=%s)",
                    sig.direction.value, sig.symbol,
                    decision.modified_signal.metadata.get("risk_sized_qty", 0),
                    sig.entry_price, sig.stop_loss, sig.regime_name,
                )
            else:
                try:
                    # Always attach a protective stop at the broker (BRACKET when
                    # a take-profit exists, else OTO). Never submit a bare entry.
                    result = self._executor.submit_bracket_order(decision.modified_signal)
                    self._trade_count += 1
                    logger.info("Order submitted: %s %s (trade_id=%s, status=%s)",
                                sig.symbol, sig.direction.value,
                                result.trade_id, result.status.value)
                except Exception as e:
                    logger.error("Order execution failed for %s: %s", sig.symbol, e)

        # --- 10b. Validate and execute each pair signal ---
        for ps in pair_signals:
            pair_decision = self._risk.validate_pair_signal(
                ps, portfolio, bars=all_bars,
            )
            if not pair_decision.approved:
                logger.info(
                    "Pair rejected [%s/%s]: %s",
                    ps.long_leg.symbol, ps.short_leg.symbol,
                    pair_decision.rejection_reason,
                )
                continue
            if pair_decision.modifications_made:
                logger.info(
                    "Pair modified [%s/%s]: %s",
                    ps.long_leg.symbol, ps.short_leg.symbol,
                    "; ".join(pair_decision.modifications_made),
                )

            modified = pair_decision.modified_pair
            if self._dry_run:
                logger.info(
                    "DRY RUN — would place pair: LONG %s %d @ $%.2f / SHORT %s %d @ $%.2f "
                    "(z=%.2f, hedge_ratio=%.3f)",
                    modified.long_leg.symbol,
                    modified.long_leg.metadata.get("risk_sized_qty", 0),
                    modified.long_leg.entry_price,
                    modified.short_leg.symbol,
                    modified.short_leg.metadata.get("risk_sized_qty", 0),
                    modified.short_leg.entry_price,
                    modified.z_score, modified.hedge_ratio,
                )
            else:
                try:
                    long_res, short_res = self._executor.submit_pair_order(modified)
                    self._trade_count += 2
                    logger.info(
                        "Pair submitted: %s/%s (long_status=%s, short_status=%s)",
                        modified.long_leg.symbol, modified.short_leg.symbol,
                        long_res.status.value, short_res.status.value,
                    )
                except Exception as e:
                    logger.error(
                        "Pair execution failed for %s/%s: %s",
                        modified.long_leg.symbol, modified.short_leg.symbol, e,
                    )

        # --- 11. Update trailing stops for existing positions ---
        if not self._dry_run:
            try:
                self._trail_stops(all_bars)
            except Exception as e:
                logger.warning("Trailing-stop update failed: %s", e)
        self._tracker.increment_holding_period()
        self._tracker.update_regime(regime_state.label)

        # --- 12. Periodic state save ---
        if self._bar_count % 20 == 0:
            self._save_state(regime_state, equity)

    def _handle_loop_error(self, error: Exception) -> None:
        """Handle non-fatal errors in the loop."""
        dump_path = write_crash_dump(error, {
            "bar_count": self._bar_count,
            "trade_count": self._trade_count,
            "regime": self._last_regime_label,
            "running": self._running,
        })
        logger.error("Error logged to %s", dump_path)

    def _save_state(self, regime_state, equity: float) -> None:
        state = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "regime_label": regime_state.label,
            "regime_probability": float(regime_state.probability),
            "regime_confirmed": regime_state.is_confirmed,
            "equity": equity,
            "positions": self._tracker.get_position_values(),
            "bar_count": self._bar_count,
            "trade_count": self._trade_count,
            "regime_change_count": self._regime_change_count,
        }
        save_state_snapshot(self._state_path, state)

    def get_session_summary(self) -> str:
        elapsed = pd.Timestamp.now() - self._session_start
        return (
            f"\n{'=' * 60}\n"
            f"SESSION SUMMARY\n"
            f"{'=' * 60}\n"
            f"Duration:         {elapsed}\n"
            f"Bars processed:   {self._bar_count}\n"
            f"Trades executed:  {self._trade_count}\n"
            f"Regime changes:   {self._regime_change_count}\n"
            f"Last regime:      {self._last_regime_label}\n"
            f"Dry run:          {self._dry_run}\n"
            f"{'=' * 60}"
        )


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_live(config: dict, dry_run: bool = False) -> None:
    """Initialize all components and run the live trading loop."""
    global logger
    logger = init_logging(config)

    from broker.alpaca_client import AlpacaClient
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from data.market_data import MarketDataClient
    from data.feature_engineering import FeatureEngineer
    from core.regime_strategies import RegimeStrategyManager
    from core.risk_manager import RiskManager

    # --- 1. Alpaca connection ---
    broker_config = config.get("broker", {})
    client = AlpacaClient(broker_config)

    logger.info("Connecting to Alpaca (%s)...",
                "paper" if broker_config.get("paper_trading", True) else "LIVE")
    client.connect()

    account = client.get_account()
    logger.info(
        "Account: equity=$%.2f, cash=$%.2f, buying_power=$%.2f, "
        "PDT=%s, blocked=%s",
        account.equity, account.cash, account.buying_power,
        account.is_pattern_day_trader, account.is_trading_blocked,
    )

    if account.is_trading_blocked:
        logger.error("Trading is blocked on this account. Exiting.")
        sys.exit(1)

    # --- 2. Check market status ---
    clock = client.get_clock()
    if not clock["is_open"]:
        logger.info("Market is closed. Next open: %s", clock["next_open"])
        logger.info("System will wait for market open or process will idle.")

    # --- 3. Initialize components ---
    feature_engineer = FeatureEngineer(config.get("hmm", {}))
    market_data = MarketDataClient(client, config.get("universe", {}))

    # --- 4. Load or train HMM ---
    hmm_engine, hmm_features = load_or_train_hmm(
        config, market_data, feature_engineer, dry_run=dry_run)

    # --- 5. Initialize strategy manager ---
    strategy_config = config.get("strategy", {})
    strategy_manager = RegimeStrategyManager(strategy_config, hmm_engine.regime_infos)

    # --- 6. Initialize risk manager ---
    risk_manager = RiskManager(config.get("risk", {}))
    risk_manager.initialize(account.equity)

    # --- 7. Initialize order executor ---
    order_executor = OrderExecutor(client, config.get("execution", {}))

    # --- 8. Initialize position tracker and sync ---
    position_tracker = PositionTracker(client)
    snapshot = position_tracker.sync_with_broker()
    logger.info("Synced %d positions from Alpaca", len(snapshot.positions))
    for pos in snapshot.positions:
        logger.info("  %s: %d %s @ $%.2f, P&L=$%.2f",
                     pos.symbol, pos.qty, pos.side,
                     pos.avg_entry_price, pos.unrealized_pnl)

    # --- 9. Start WebSocket streams ---
    if not dry_run:
        try:
            position_tracker.start_streaming()
        except Exception as e:
            logger.warning("Failed to start position streaming: %s", e)

    # --- 10. Load recovery state if available ---
    state_path = config.get("model", {}).get("state_snapshot", "state_snapshot.json")
    prev_state = load_state_snapshot(state_path)
    if prev_state:
        logger.info("Recovered previous state: regime=%s, trades=%s",
                     prev_state.get("regime_label"),
                     prev_state.get("trade_count"))

    # --- 11. Print system state ---
    regime_state = hmm_engine.predict_regime_filtered(hmm_features)
    logger.info(
        "System initialized. Current regime: %s (prob=%.2f). "
        "Entering main loop.",
        regime_state.label, regime_state.probability,
    )

    # --- 12. Create and run the trading loop ---
    loop = TradingLoop(
        config=config,
        alpaca_client=client,
        hmm_engine=hmm_engine,
        feature_engineer=feature_engineer,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        order_executor=order_executor,
        position_tracker=position_tracker,
        market_data=market_data,
        hmm_features=hmm_features,
        dry_run=dry_run,
    )

    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received (signal %d)", signum)
        loop.stop()

    signal_mod.signal(signal_mod.SIGINT, shutdown_handler)
    signal_mod.signal(signal_mod.SIGTERM, shutdown_handler)

    try:
        loop.run()
    except Exception as e:
        dump = write_crash_dump(e, {
            "bar_count": loop._bar_count,
            "trade_count": loop._trade_count,
        })
        logger.critical("Fatal error: %s. Crash dump: %s", e, dump, exc_info=True)
    finally:
        # --- Shutdown ---
        logger.info("Shutdown initiated.")

        # Save state
        try:
            final_regime = hmm_engine.predict_regime_filtered(hmm_features)
            loop._save_state(final_regime, account.equity)
        except Exception:
            pass

        # Close streams
        try:
            position_tracker.stop_streaming()
        except Exception:
            pass

        try:
            client.disconnect()
        except Exception:
            pass

        # Session summary
        logger.info(loop.get_session_summary())
        logger.info("System shut down cleanly. Positions remain open with stops.")


def run_train_only(config: dict) -> None:
    """Train the HMM model and exit."""
    global logger
    logger = init_logging(config)

    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketDataClient
    from data.feature_engineering import FeatureEngineer

    broker_config = config.get("broker", {})
    client = AlpacaClient(broker_config)
    client.connect()

    feature_engineer = FeatureEngineer(config.get("hmm", {}))
    market_data = MarketDataClient(client, config.get("universe", {}))

    hmm_engine, features = load_or_train_hmm(config, market_data, feature_engineer)

    # Print regime details
    print(f"\nHMM Model: {hmm_engine.n_regimes} regimes selected by BIC")
    print(f"Training metrics: {hmm_engine.last_training_metrics}")
    print(f"Transition matrix:\n{hmm_engine.get_transition_matrix()}")
    print("\nRegime details:")
    for info in hmm_engine.regime_infos:
        print(f"  {info.regime_name}: E[ret]={info.expected_return:.3f}, "
              f"E[vol]={info.expected_volatility:.3f}, "
              f"strategy={info.recommended_strategy_type}")

    client.disconnect()
    print("\nModel saved. Exiting.")


def run_backtest(config: dict, symbols: Optional[list] = None,
                 start: Optional[str] = None, end: Optional[str] = None,
                 stress_test: bool = False, compare: bool = False) -> None:
    """Run a walk-forward backtest with optional stress testing and comparison."""
    global logger
    logger = init_logging(config)

    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketDataClient
    from data.feature_engineering import FeatureEngineer
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer
    from backtest.stress_test import StressTester

    bt_config = config.get("backtest", {})
    if start:
        bt_config["start_date"] = start
    if end:
        bt_config["end_date"] = end
    config["backtest"] = bt_config

    syms = symbols or config.get("universe", {}).get("symbols", ["SPY"])

    # Connect to Alpaca to fetch historical data. If credentials are
    # placeholders or missing, fall back to disk cache only — backtests
    # work fully offline once data_cache/ is populated.
    broker_config = config.get("broker", {})
    client = AlpacaClient(broker_config)
    try:
        client.connect()
    except (ValueError, ConnectionError) as e:
        logger.warning(
            "Alpaca connect failed (%s) — running backtest from data_cache/ only. "
            "If a needed CSV is missing, the fetch will fail.", e,
        )

    market_data = MarketDataClient(client, config.get("universe", {}))
    fe = FeatureEngineer(config.get("hmm", {}))

    # Fetch bars for all symbols
    logger.info("Fetching historical data for %s (%s to %s)...",
                syms, bt_config.get("start_date"), bt_config.get("end_date"))
    all_bars: dict[str, pd.DataFrame] = {}
    for sym in syms:
        bars = market_data.get_historical_bars(
            sym, timeframe="1Day",
            start=bt_config.get("start_date"),
            end=bt_config.get("end_date"),
            limit=5000,
        )
        if bars is not None and len(bars) > 0:
            all_bars[sym] = bars
            logger.info("  %s: %d bars (%s to %s)", sym, len(bars),
                        bars.index[0].date(), bars.index[-1].date())

    if not all_bars:
        logger.error("No data fetched. Exiting.")
        client.disconnect()
        return

    # Compute HMM features from reference symbol
    ref_sym = syms[0]
    hmm_features = fe.compute_hmm_features(all_bars[ref_sym])
    logger.info("HMM features: %d rows, %d columns", *hmm_features.shape)

    # Run walk-forward backtest
    bt = WalkForwardBacktester(config)
    result = bt.run(all_bars, hmm_features)

    # Effective traded span: walk-forward drops the warm-up + first training
    # window at the start and any trailing partial window at the end, so the
    # OOS track record is typically shorter than the requested date range.
    eq = result.equity_curve
    if len(eq) > 0:
        logger.info(
            "Effective OOS traded span: %s -> %s (%d bars). Requested %s -> %s; "
            "the gap is walk-forward warm-up/first-train + any trailing partial window.",
            eq.index[0].date(), eq.index[-1].date(), len(eq),
            bt_config.get("start_date"), bt_config.get("end_date"),
        )

    # Display results
    analyzer = PerformanceAnalyzer(bt_config.get("risk_free_rate", 0.045))
    print(analyzer.format_summary(result.metrics))

    # Regime breakdown
    regime_breakdown = analyzer.regime_breakdown(result.trades, result.equity_curve)
    if regime_breakdown:
        print(f"\n{'REGIME BREAKDOWN':^55}")
        print(f"{'=' * 55}")
        for r in regime_breakdown:
            print(f"  {r.regime_name:<16} trades={r.trade_count:>3}  "
                  f"P&L=${r.pnl_contribution:>10,.2f}  "
                  f"WR={r.win_rate:.0%}  PnL/σ={r.sharpe:>5.2f}")

    # Confidence buckets
    buckets = analyzer.confidence_buckets(result.trades)
    if buckets:
        print(f"\n{'CONFIDENCE BUCKETS':^55}")
        print(f"{'=' * 55}")
        for b in buckets:
            print(f"  {b.bucket_label:<10} trades={b.trade_count:>3}  "
                  f"P&L=${b.total_pnl:>10,.2f}  "
                  f"WR={b.win_rate:.0%}  PnL/σ={b.sharpe:>5.2f}")

    # Benchmark comparison
    if compare and ref_sym in all_bars:
        comp = analyzer.compare_benchmarks(
            result.equity_curve, result.trades,
            all_bars[ref_sym], bt_config.get("initial_capital", 100_000))
        print(analyzer.format_comparison(comp))

        # Persist the comparison (previously computed but never written to disk).
        comp_path = bt_config.get("output", {}).get("comparison_csv")
        if comp_path:
            Path(comp_path).parent.mkdir(parents=True, exist_ok=True)
            analyzer.comparison_to_frame(comp).to_csv(comp_path, index=False)
            logger.info("Comparison written to %s", comp_path)

    # Stress test
    if stress_test:
        logger.info("Running stress tests...")
        st = StressTester(config)
        scenarios = st.run_all_scenarios(all_bars, hmm_features)
        print(StressTester.format_report(scenarios, baseline=result))

    # Save results
    output_config = bt_config.get("output", {})
    if output_config:
        bt.save_results(result, output_config)

    # Window summary
    print(f"\nWalk-forward windows: {len(result.walk_forward_windows)}")
    for w in result.walk_forward_windows:
        ts, te = w.get("train_start"), w.get("test_end")
        rng = f"{ts:%Y-%m-%d} → {te:%Y-%m-%d}  " if ts is not None and te is not None else ""
        print(f"  Window {w['window_id']}: {rng}trades={w['n_trades']}  return={w['return_pct']:.2%}")

    client.disconnect()
    logger.info("Backtest complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the appropriate run mode."""
    parser = argparse.ArgumentParser(
        description="Regime Trader — HMM-based market regime detection trading system",
    )
    parser.add_argument(
        "mode",
        choices=["live", "backtest", "train-only"],
        help=(
            "Run mode: 'live' for paper/live trading, "
            "'backtest' for historical testing, "
            "'train-only' to train the HMM model and exit."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run everything including HMM and signal generation, "
            "but do NOT place any orders. Logs what it would have done."
        ),
    )
    # Backtest-specific arguments
    parser.add_argument("--start", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument(
        "--symbols", nargs="+",
        help="Symbols to backtest (e.g. SPY QQQ AAPL).",
    )
    parser.add_argument(
        "--stress-test", action="store_true",
        help="Run stress tests (crash injection, gap sim, vol spikes).",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare strategy vs buy-and-hold, SMA trend, and random entry.",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    if args.mode == "live":
        run_live(config, dry_run=args.dry_run)
    elif args.mode == "train-only":
        run_train_only(config)
    elif args.mode == "backtest":
        run_backtest(
            config,
            symbols=args.symbols,
            start=args.start,
            end=args.end,
            stress_test=args.stress_test,
            compare=args.compare,
        )


if __name__ == "__main__":
    main()
