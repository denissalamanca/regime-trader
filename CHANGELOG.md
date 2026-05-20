# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-04-26

Initial public release of the Regime Trader template.

### Included
- Gaussian HMM regime detection with BIC model selection (3–7 regimes)
- Forward-algorithm filtered inference (no look-ahead bias)
- Volatility-rank strategy orchestrator with low/mid/high vol archetypes
- Walk-forward backtester with realistic slippage, fill delay, and gap handling
- Risk manager with hardcoded limits and drawdown circuit breakers
- Alpaca paper/live trading integration via `alpaca-py`
- Performance analytics (Sharpe, Sortino, Calmar, regime breakdown, benchmark comparison)
- Stress testing utilities (crash injection, gap simulation, Monte Carlo)
- Structured JSON logging across 4 rotating files
- Email and webhook alerts with rate limiting
- 216 unit and integration tests (plus 6 live Alpaca-connectivity tests, skipped without API keys)
