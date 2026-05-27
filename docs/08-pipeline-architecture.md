# 08 — Pipeline Architecture

The system has two halves: a submission engine that must run inside the exchange's constraints, and an offline research-and-validation pipeline that has no such limits.

## Submission engine (`trader.py`)

A single self-contained file. The exchange disallows third-party libraries, so every model is implemented in pure Python and the engine must serialize all state into a sub-90KB string each tick.

### Strategy dispatch
Each product is routed to one of nine strategy handlers, either by an explicit per-product configuration or by a runtime classifier when an unconfigured product appears:

- `pegged` — stable-value market making: sweep all profitable resting liquidity, then post improved bid/ask one tick inside the book; throttle as inventory approaches the position limit.
- `ar_olivia` — AR(p) regression on returns plus a Kalman-filtered fair value, with an informed-counterparty detector biasing direction.
- `olivia_follow` — directional signal-following with no resting quotes (for products where passive quoting is adversely selected).
- `wide_spread` — EMA-anchored wide two-sided quoting for high-volatility products.
- `basket_arb` — z-score entry on the basket-minus-NAV premium with partial component hedging.
- `conversion_arb` — cross-market arbitrage incorporating environmental/observation factors.
- `options` — implied-vol scalping against a fitted vol smile, with a per-product kill switch.
- `pairs_arb` — cointegrated pairs trading.
- `generic_mm` / `do_nothing` — fallback market making, or an explicit sit-out for unclassifiable products.

All nine archetypes exist in the engine, developed across the practice round and prior-year reference data. The live Prosperity 4 rounds exercised a subset: `pegged` on the Round 1–4 cash products (ASH_COATED_OSMIUM, INTARIAN_PEPPER_ROOT, HYDROGEL_PACK, VELVETFRUIT_EXTRACT), `options` on the Round 3–4 VEV vouchers, and mean-reversion on the Round 5 product set. The remaining archetypes (basket, conversion, pairs, AR/signal-following) were built and validated against reference data but did not map to a live Round 1–5 product.

### Core mathematical components (all stdlib)
- **Black-Scholes library** — normal CDF, call/put pricing, delta, vega, and Newton-style implied-volatility inversion.
- **AR(p) estimation** — ordinary least squares via a Cholesky solve of the normal equations.
- **Kalman filter** — per-product-type process/measurement noise (Q/R) tuning.
- **Bayesian Online Changepoint Detection** — monitors for regime shifts and halves position sizing when a changepoint is detected.
- **Markout tracking** — measures post-fill price movement to detect adverse selection and scale sizing down when fills are consistently unfavorable.

### Risk and safety
- **Trim-based position-limit enforcement** — when an order pack would breach a limit, the widest passive quotes are trimmed while aggressive takes are preserved, rather than nuking the entire pack.
- **Kill switches** — per-product PnL floors (e.g. −5K on options) that force inventory unwind.
- **State hygiene** — `traderData` is pruned to stay under the size limit; rolling buffers are capped while preserving enough history for each model's window.

## Offline research and validation pipeline

Standard scientific Python is permitted offline (it never ships), so the research tooling uses numpy/pandas/matplotlib freely. The pipeline's job is to decide what is real before it reaches the engine.

### Backtesting
- **Integrated backtester** — reconstructs the order book tick by tick from historical price data, runs the actual `Trader` class against it, simulates fills through book depth, and reports per-product PnL. This is the ground-truth check.
- **Per-round backtester** — replays specific rounds for regression testing.

### Validation
- **Walk-forward validation** — selects a configuration on a training subset of days, then evaluates it on a held-out day. A signal that only survives in-sample is rejected.
- **Regression gauntlet** — replays prior rounds against the current engine and aborts the ship if any product regresses beyond a tolerance versus a blessed baseline.
- **Quality gates** — pre-submission checks: file size under 100KB, valid JSON state, position-limit safety, graceful handling of empty order books.

### Signal research
A toolkit for separating alpha from noise, used heavily in Round 5's 50-product search:
- Cross-asset and within-category return-correlation matrices.
- Lead-lag detection via cross-correlation at multiple offsets.
- Order-book-imbalance studies.
- Hurst-exponent and half-life estimation for mean-reversion vs trending classification.
- Time-of-day PnL attribution.
- Monte-Carlo PnL distribution modeling for the manual rounds.

## Design principle

The engine is deliberately conservative — explicit sit-outs, kill switches, trim-not-nuke order handling — because the dominant failure mode in this competition was not missing upside but taking unhedged or overfit risk into an unseen market. The validation pipeline exists to enforce that a strategy earns its place across multiple regimes before it is trusted with capital. The one round where that discipline was bypassed (Round 4) is documented in [Overfitting Analysis](09-overfitting-lessons.md).
