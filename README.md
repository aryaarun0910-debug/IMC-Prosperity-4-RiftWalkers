# IMC Prosperity 4 — Team RiftWalkers

A complete record of a two-person team's run through IMC Trading's **Prosperity 4**, a five-round global algorithmic and manual trading competition with 18,803 participating teams.

**Final standing: 700th of 18,803 (top 3.7%).**
**Manual Round 1: 6th globally.**

This repository documents each round end to end — the products, the strategies, the underlying mathematics, the results, and the analysis of what worked and what did not. The algorithmic side is a single-file, standard-library-only Python trading engine; the manual side is a set of from-first-principles solutions to game-theoretic and quantitative puzzles. Supporting both is a custom backtesting, validation, and signal-research pipeline.

---

## Results

| Round | Phase | Cumulative XIRECs | Algo PnL | Algo Rank | Manual PnL | Manual Rank |
|-------|-------|-------------------|----------|-----------|------------|-------------|
| R1 | Qualifier | 183,387 | +95,490 | 1642 | +87,897 | **6** |
| R2 | Qualifier | 444,349 | +73,268 | 3407 | +187,694 | 128 |
| R3 | Final | 133,190 | +65,477 | 504 | +67,713 | 541 |
| R4 | Final | 176,419 | +19,664 | 1265 | +23,566 | 699 |
| R5 | Final | 272,456 | +6,849 | 896 | +89,187 | 411 |

*Phase 2 (R3–R5) began with a full leaderboard reset; Phase-1 PnL did not carry forward.*

![Cumulative score across rounds](docs/assets/cumulative_progression.png)

The manual track consistently outperformed the algorithmic track in relative ranking. The contrast is the throughline of the competition and is examined directly in the round writeups.

![Algorithmic vs manual PnL and rank](docs/assets/algo_vs_manual.png)

---

## Contents

| Document | Focus |
|----------|-------|
| [Competition Overview](docs/01-competition-overview.md) | Exchange model, rules, two-phase structure, scoring |
| [Round 1](docs/02-round1.md) | Market-making foundations; AR + informed-trader signal; Manual R1 |
| [Round 2](docs/03-round2.md) | Qualifier strategy; game-theoretic manual round |
| [Round 3](docs/04-round3.md) | Options and volatility — implied-vol scalping, vol-smile fitting |
| [Round 4](docs/05-round4.md) | Exotic options pricing; the algorithmic overfitting failure |
| [Round 5](docs/06-round5.md) | 50 new products and signal discovery; news-driven manual portfolio |
| [Manual Rounds](docs/07-manual-rounds.md) | The mathematics behind the strongest results |
| [Pipeline Architecture](docs/08-pipeline-architecture.md) | Trading engine and research/validation tooling |
| [Overfitting Analysis](docs/09-overfitting-lessons.md) | The central technical finding, with data |
| [Retrospective](docs/10-retrospective.md) | What worked, what did not, and where the remaining edge is |

---

## The trading engine

A single self-contained `trader.py` (standard library only — numpy, pandas, and scipy are disallowed in submissions) implementing nine strategy archetypes, auto-dispatched per product:

| Strategy | Application |
|----------|-------------|
| `pegged` | Stable-value products — sweep-all liquidity plus overbid/undercut market making |
| `ar_olivia` | Trending products — AR(p) on returns, Kalman filtering, informed-trader detection |
| `olivia_follow` | Signal-following with no passive quoting |
| `wide_spread` | High-volatility products — EMA-based wide market making |
| `basket_arb` | Index/component arbitrage — z-score entry with partial hedging |
| `conversion_arb` | Cross-market arbitrage with environmental factors |
| `options` | Implied-vol scalping and mean-reversion with kill-switch controls |
| `pairs_arb` | Cointegrated pairs trading |
| `generic_mm` | Fallback adaptive market making |

Supporting infrastructure includes Bayesian Online Changepoint Detection for regime shifts, trim-based position-limit enforcement, markout-driven adverse-selection sizing, and a runtime product classifier. Every model — Black-Scholes, implied-volatility inversion, Kalman filtering, AR regression, changepoint detection — is implemented in pure Python to satisfy the submission constraints. Details in [Pipeline Architecture](docs/08-pipeline-architecture.md).

---

## Methods

Statistical arbitrage · market microstructure · options pricing (Black-Scholes, implied-volatility inversion) · Kalman filtering · autoregressive time-series modeling · Bayesian changepoint detection · game theory and equilibrium analysis · constrained optimization (Lagrangian, Kelly-style sizing) · Monte-Carlo simulation · walk-forward validation.

---

## A note on the analysis

This repository is a retrospective compiled after the competition closed. All figures are the official final results. The writeups are deliberately candid about failures — particularly the Round 4 algorithmic loss and its root cause — because the analysis of those failures is the most transferable part of the work.
