# 04. Round 3: Options & Volatility

**Result:** Algo +65,477 · Manual +67,713 · Cumulative 133,190 (Phase 2 reset, position 581)

Phase 2 opened with the leaderboard reset to zero, and the entire product set changed. Round 3 introduced an underlying plus a chain of **option/voucher products**, the first round where derivatives pricing became central.

## Algorithmic side

### Products
- `VELVETFRUIT_EXTRACT`: the underlying, traded with the `pegged` market-making core.
- `HYDROGEL_PACK`: a second market-making product.
- `VEV_4000` … `VEV_6500`: a chain of voucher (call-option) products on VELVETFRUIT_EXTRACT, spanning strikes from deep in-the-money to far out-of-the-money.

### Options: the `options` strategy
The VEV vouchers are options on VELVETFRUIT_EXTRACT. Our approach drew on top-team analysis (notably the 2nd-place "Frankfurt" team's IV-scalping method):

- **Black-Scholes in pure Python**: `_norm_cdf`, `bs_call`, `bs_delta`, `bs_vega`, plus Newton-style **implied-volatility inversion**. All stdlib.
- **IV scalping**: compute implied vol per strike, fit a **volatility smile** (polynomial across moneyness), and trade strikes whose IV deviates from the smile fit (rich vol → sell, cheap vol → buy).
- **Mean-reversion** on the option price with a position cap.
- **Risk controls**: a hard **−5K kill switch** per product (force-flatten if a product's PnL craters), because options PnL is the highest-variance in the engine.

The options strategy was historically the most volatile component: in prior-year analysis it ranged from **+134K to −59K** on the same code in different markets. That volatility foreshadowed the R4 problem.

### Validation infrastructure built here
R3 is where we hardened the pipeline:
- **Gauntlet**: a regression harness that replays prior rounds against the current `trader.py` and aborts the ship if any product regresses beyond a threshold.
- **Quality gates**: pre-submission validation (size < 100KB, valid JSON state, position-limit safety, empty-book handling).
- **First-log intel**: first-tick analysis of a freshly-opened round to classify products before committing strategy.

## Manual side

The R3 manual was a quantitative puzzle; we scored +67,713 (rank 541). Solid but middle-of-pack, a sign the manual edge was strongest on game-theoretic (R1/R2) rounds.

## Takeaways

- **Implementing Black-Scholes + IV inversion in pure stdlib** was a real engineering exercise, and reusable infrastructure for R4's exotic options.
- **Vol-smile fitting** is the right frame for options market-making: trade *relative* mispricing across the smile, not absolute price.
- The validation gauntlet became mandatory pre-ship from here on, though, as R4 would show, a gauntlet on *past* data doesn't protect you from *overfitting to a single future day*.

→ Next: [Round 4](05-round4.md)
