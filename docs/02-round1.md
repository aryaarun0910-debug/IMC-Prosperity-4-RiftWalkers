# 02 — Round 1: Foundations + a World-Class Manual

**Result:** Algo +95,490 · Manual +87,897 (**6th in the world**) · Cumulative 183,387

Round 1 introduced the market-making products and opened the qualifier phase. Our manual result here was the single best showing of the entire competition.

## Algorithmic side

Two products, both market-making, handled by the engine's `pegged` archetype:

### ASH_COATED_OSMIUM — stable-value market making
A product anchored near a fixed fair value. The optimal play is aggressive two-sided market making:
- **Sweep all profitable liquidity** — take any bid above fair or ask below fair.
- **Overbid / undercut** the rest of the book by one tick to capture the spread.
- **Inventory-aware throttling** — pull quotes as position approaches the limit.

This "sweep + peg" pattern was the most reliable PnL source in the engine throughout the competition: low variance, high fill rate on a product that barely moves.

### INTARIAN_PEPPER_ROOT — pegged with drift
The same market-making core, but this product carried a slow **upward fair-value drift** rather than sitting flat. The fair-value estimate adds a drift term, and the quoting skews to lean into the trend — bidding slightly more aggressively and holding inventory on the favorable side rather than flattening immediately.

### Engine foundation
These two products established the infrastructure the rest of the competition was built on: the fair-value estimator, trim-based position-limit enforcement, and markout-driven sizing that scales down when fills are being adversely selected. The engine also carries an informed-trader detector (a Bayesian filter for rare, large, directionally-consistent counterparty trades) that became central later when an exploitable counterparty was introduced in Round 4.

## Manual side — 6th globally

The Round 1 manual was a game-theoretic optimization, solved via equilibrium reasoning rather than a naive greedy approach. We modeled the payoff structure precisely, reasoned about where aggregate participant behavior would settle, and optimized against that — landing within roughly 0.1% of the theoretical maximum payout (≈ 87,995; we scored 87,897).

**6th out of ~18,000 teams.** This set the tone: disciplined, from-first-principles puzzle solving was our edge, and it stayed our strongest track for the whole competition.

## Takeaways

- The `pegged` sweep-and-peg pattern was the backbone of algorithmic PnL across every round.
- Adding a drift term to the fair value (INTARIAN_PEPPER_ROOT) rather than treating every product as a flat peg was a small modeling decision with real payoff.
- Manual rounds are not a side-quest — they are a pure-skill scoreboard where a good solver is a genuine edge. We treated them that way from Round 1 onward.

→ Next: [Round 2](03-round2.md)
