# 03 — Round 2: Qualifying

**Result:** Algo +73,268 · Manual +187,694 (rank 128) · Cumulative 444,349 — **qualified for Phase 2**

Round 2 closed out the qualifier phase. It continued with the same two market-making products as Round 1 (`ASH_COATED_OSMIUM` and `INTARIAN_PEPPER_ROOT`), so the algorithmic work was refinement rather than new strategy.

Phase 1 only required ~200K cumulative XIRECs to advance, and we were at 183K after Round 1. The explicit strategy was therefore **"clear the bar safely, don't over-optimize"** — the leaderboard resets for Phase 2, so effort spent squeezing micro-edges out of R2 would have been wasted capacity better saved for the rounds that actually decide placement.

## Algorithmic side

Refinement of the Round 1 `pegged` strategies on the same two products:
- Tuning the inventory thresholds and quote offsets on the stable product (ASH).
- Refining the drift handling on INTARIAN_PEPPER_ROOT.

A more important discovery here was methodological: backtesting revealed the simulated fill model was substantially more generous than the live server, where fills only occur on the minority of ticks that carry a market trade. This recalibrated our PnL expectations downward and seeded a lasting distrust of backtest-only validation — the discipline that paid off in the Phase-2 rounds.

## Manual side — game-theoretic

The Round 2 manual was explicitly game-theoretic: the payoff for a choice depended on what everyone else chose (a crowding/fee mechanic). This is fundamentally different from a fixed-answer puzzle — you have to model the population's behavior and reason about the equilibrium, not just optimize for yourself in isolation.

We scored +187,694 (rank 128) — strong by absolute number, though our weakest *relative* manual finish among the rounds we contested seriously. The lesson logged at the time: **never shortcut to "looks like last round."** Each manual prompt has its own payoff function and must be re-derived from scratch.

## Qualifier secured

At 444,349 cumulative XIRECs we cleared the 200K threshold with large margin and advanced to Phase 2. From here the leaderboard reset and the real tournament — Rounds 3, 4, and 5 — began.

## Takeaways

- **Resource triage matters.** Recognizing R2 as a filter rather than a final let us conserve effort for the rounds that counted.
- **Backtest is not reality.** The fill-model gap discovered here informed every later "is this signal real?" decision.
- **Game-theoretic manual rounds** require population modeling, not just optimal-for-me reasoning.

→ Next: [Round 3](04-round3.md)
