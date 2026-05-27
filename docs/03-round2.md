# 03 — Round 2: Qualifying + Game Theory

**Result:** Algo +73,268 · Manual +187,694 (rank 128) · Cumulative 444,349 — **qualified for Phase 2**

Round 2 closed out the qualifier phase. Since Phase 1 only required ~200K cumulative XIRECs to advance (and we were at 183K after R1), our explicit strategy was **"clear the bar safely, don't over-optimize."** The leaderboard would reset for Phase 2, so burning effort on R2 micro-edges was wasted capacity.

## Algorithmic side

R2 extended the R1 product set with additional market-making and basket products. Key engineering work:

- **Basket arbitrage** (`basket_arb`) — index products priced against a weighted basket of components. We used **z-score entry** on the premium (basket price − NAV) rather than absolute thresholds. A key finding from backtesting: *absolute* premium thresholds lose money (you trade noise), while *z-score* thresholds adapt to volatility regimes. Half-sized component hedging on each entry.
- **Adverse-selection discovery** — we learned the backtest fill model was ~5× more generous than the real server. On the server, fills only happen on ticks where a market trade occurs (~4% of ticks). This recalibrated all our PnL expectations downward and made us distrust backtest-only validation.

A documented limitation: basket arbitrage was only marginally profitable because the exchange gave us no passive fills — we paid adverse selection on every take. We kept it small.

## Manual side — game-theoretic

The R2 manual was **explicitly game-theoretic**: the payoff for your choice depended on what *everyone else* chose (a fee/crowding mechanic). This is fundamentally different from a fixed-answer puzzle — you have to model the population's behavior and find an equilibrium.

We scored +187,694 (rank 128) — strong by absolute number, but our weakest *relative* manual finish of the rounds where we competed seriously. The lesson logged at the time: **don't shortcut to "looks like last round."** Each manual prompt must be re-derived from scratch — the payoff function changes, and last round's solver model is often wrong for this round.

## Qualifier secured

At 444,349 cumulative XIRECs we cleared the 200K bar with large margin and advanced to Phase 2 ranked 1078th. From here, the real tournament — and the leaderboard reset — began.

## Takeaways

- **Resource triage matters.** Recognizing that R2 was a filter (not a final) let us conserve effort for the rounds that actually decided placement.
- **Backtest ≠ reality.** The 5× fill-model gap discovered here informed every later "is this signal real?" decision.
- **Game-theoretic manual rounds** require population modeling, not just optimal-for-me reasoning.

→ Next: [Round 3](04-round3.md)
