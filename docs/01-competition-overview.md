# 01 — Competition Overview

## What is IMC Prosperity?

[IMC Prosperity](https://prosperity.imc.com/) is a global algorithmic trading competition run by [IMC Trading](https://www.imc.com/), a leading market-making firm. Prosperity 4 (2025) ran for **5 rounds** and drew **18,803 teams** worldwide — students, quants, and professionals.

Each round has two independent components:

1. **Algorithmic trading** — you submit a single Python program that trades on a simulated exchange. It runs autonomously against bot counterparties over a fixed number of discrete time-steps ("ticks"). PnL is measured in the competition currency (XIRECs / seashells).

2. **Manual trading** — a one-shot puzzle (often game-theoretic or requiring fundamental/quantitative analysis) where you submit a portfolio or set of decisions. Scored against an optimal/anchor solution, frequently with a "wisdom of the crowd" correction.

Your round score = algo PnL + manual PnL. Cumulative XIRECs determine standing.

## The exchange model (algorithmic side)

The simulated exchange has specific mechanics that shape every strategy decision:

- **Discrete-tick single-shot auction.** Orders don't persist — every order is cancelled at the end of each tick. You must resubmit your entire book every tick.
- **No queue priority.** Fills are determined by price-crossing against the visible book and the bots' market trades.
- **Atomic position limits.** If any single order would breach a product's position limit, the exchange can reject the entire order pack. Limits are typically 10–80 per product depending on the round.
- **Processing order:** deep makers → takers → your bot → other bots.
- **State persistence** via a `traderData` string (must stay under ~90KB), serialized/deserialized each tick. This is the only memory between ticks.
- **stdlib only.** No numpy, pandas, or scipy in submissions. Every model — Black-Scholes, Kalman filters, AR regression, changepoint detection — had to be implemented in pure Python.

These constraints reward *correctness under tight limits* over raw modeling firepower. A clever signal is worthless if it can't be computed in pure Python within the tick budget and serialized into 90KB.

## The two-phase structure

Prosperity 4 was split into two phases (confirmed from the in-game "On-Board Advisor" guidance):

- **Phase 1 = Rounds 1 + 2 (Qualifier).** Teams needed ~200K cumulative XIRECs to advance. We closed R1 at 183,387 and qualified comfortably in R2.
- **Phase 2 = Rounds 3 + 4 + 5 (Final).** The leaderboard **resets** — Phase-1 PnL does not carry forward. R3–R5 were explicitly "significantly harder and more time-consuming." The final tournament was decided here.

This shaped our resource allocation: we treated R2 as "clear the bar safely, don't over-optimize" and saved capacity for the high-leverage Phase-2 rounds.

## How we worked

A two-person team split between:
- **Algo side** — building and iterating the trading engine, backtesting, signal research.
- **Manual side** — solving each round's puzzle from first principles (game theory, options pricing, constrained optimization).

Every round followed a loop: read the freshly-released wiki → extract the exact payoff/market model → research signals on the provided data → validate across multiple days → ship → postmortem.

The hardest-won lesson — and the one most worth reading — is in [Overfitting Lessons](09-overfitting-lessons.md).
