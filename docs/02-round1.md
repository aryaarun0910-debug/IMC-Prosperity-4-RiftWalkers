# 02 — Round 1: Foundations + a World-Class Manual

**Result:** Algo +95,490 · Manual +87,897 (**6th in the world**) · Cumulative 183,387

Round 1 introduced the core market-making products and set up the qualifier. Our manual result here was the single best showing of the entire competition.

## Algorithmic side

Three products, three archetypes — the foundation the whole engine was built on:

### RAINFOREST_RESIN — `pegged`
A stable-value product anchored near a fixed fair value (10,000). The optimal play is aggressive two-sided market making:
- **Sweep all profitable liquidity** — take any bid above fair or ask below fair.
- **Overbid / undercut** the rest of the book by one tick to capture the spread.
- Inventory-aware throttling: pull quotes as position approaches the limit.

This "sweep + peg" pattern (inspired by top-team analysis of prior years) was the most reliable PnL source in the engine — low variance, high fill rate on a product that barely moves.

### KELP — `ar_olivia`
A trending product. We modeled it as:
- **AR(p) regression on returns** (not raw prices — returns are stationary, which makes the regression well-behaved). Refit every ~25 ticks.
- **Kalman filter** for a smoothed fair-value estimate (tuned Q/R per product type).
- **Informed-trader detection** ("Olivia" — see below) layered on top to bias direction when a known-informed counterparty was active.

### SQUID_INK — `olivia_follow`
A product where passive market-making lost money (we'd get adversely selected). The winning approach was pure **signal-following**: no resting quotes, only directional takes when the informed-trader signal fired.

### The "Olivia" signal
Across Prosperity, a recurring bot named "Olivia" (and analogues) traded **rarely but in large size** — the classic fingerprint of an informed trader. We built a Bayesian detector: rare + large + directionally-consistent trades update a posterior that this counterparty knows something. When the posterior crossed a threshold, we followed their direction. This became one of the most-connected concepts in the whole codebase.

## Manual side — 6th globally

The R1 manual was a **multi-party currency-exchange puzzle**: convert through a sequence of currencies to maximize final holdings, where the conversion rates and others' behavior interact.

We solved it as a **game / optimal-path problem**:
- Enumerated conversion sequences (bounded search over the rate matrix).
- Reasoned about the equilibrium — where everyone's conversions push rates, the optimal path isn't the naive greedy one.
- Landed within **~0.1% of the theoretical maximum** (max payout ≈ 87,995; we scored 87,897).

**6th out of ~18,000 teams.** This set the tone: disciplined, from-first-principles puzzle solving was our edge.

## Takeaways

- The `pegged` sweep-and-peg pattern was the backbone of algo PnL for the whole competition.
- Modeling returns (stationary) rather than prices for the AR model was a small decision with outsized payoff.
- Manual rounds are not a side-quest — they're a pure-skill scoreboard where a good solver is a real edge. We treated them that way from R1 onward.

→ Next: [Round 2](03-round2.md)
