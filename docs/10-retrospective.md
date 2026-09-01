# 10. Retrospective

Final standing: 700 of 18,803 teams (top 3.7%). This document summarizes what drove the result and where the remaining edge is.

## What worked

**Manual problem-solving.** The strongest and most consistent track. Ranks of 6 (R1), 128 (R2), and 411 (R5) on the manual scoreboard came from treating each puzzle as a modeling exercise: extract the exact payoff function from the prompt, solve it from first principles, and cross-check against a hand calculation. The R1 currency-exchange optimization landed within 0.1% of the theoretical maximum; the R5 news portfolio was solved in closed form under quadratic fees and then Monte-Carlo'd for the distribution.

**The pegged market-making core.** The sweep-and-peg pattern on stable-value products was the most reliable algorithmic PnL source across every round: low variance, high fill rate.

**Pure-stdlib model engineering.** Implementing Black-Scholes with implied-volatility inversion, AR(p) via Cholesky, Kalman filtering, and Bayesian changepoint detection inside the no-dependencies constraint was a substantial engineering effort and provided reusable infrastructure across rounds.

**The validation pipeline, once it existed.** The integrated backtester and walk-forward validator built in Rounds 3–5 caught at least one shipping-grade overfit before it cost real PnL (see [Overfitting Analysis](09-overfitting-lessons.md)).

## What did not work

**The algorithmic track relative to the field.** Algo ranks (1642, 3407, 504, 1265, 896) trailed manual ranks in every round. The two clearest causes:

- **Overfitting to single-day data** (Round 4, and nearly Round 5). Calibrating to one day's drift produced models that inverted on the scoring day.
- **A thin validation jury.** Three development days is not enough to certify a signal as robust. Even validated mean-reversion gave back most of its edge on the Round 5 live day because that day trended rather than oscillated.

**Manual execution discipline in Round 4.** Beyond the algo loss, the R4 manual left roughly $155K on the table: partly from skipping positive-expectancy trades judged "too small," and partly from a direction transcription error on one exotic that alone cost about $18K. Both were process failures, not modeling failures, and both were corrected by Round 5 (size every edge; derive direction from the model, not by hand).

## Where the remaining edge is

To move from top 4% toward top 1%, the work is almost entirely on the algorithmic side:

1. **More robust signal certification.** Cross-validate against more days and synthetic regime perturbations, not just the handful of provided days. Penalize signals whose performance is concentrated in one regime.
2. **Regime-aware strategy switching.** Mean-reversion and momentum fail in each other's regimes. A detector that sizes each down when the current regime is unfavorable (the momentum filter built in R5 was a first step) would reduce the give-back on adverse days.
3. **Adverse-selection-aware market making.** The largest structural drag on the algo side was paying adverse selection on takes with no compensating passive fills. Better quote placement and markout-driven sizing would recover some of this.
4. **Earlier, faster research loops.** The strongest manual results came from time spent modeling the problem precisely. The algo side would benefit from the same: more time on what the signal *is* and less on parameter tuning, which the server calibration showed matters least.

## Closing

The competition rewarded clear problem formulation and punished fitting to noise. Our results reflect both: a manual track that consistently modeled problems from first principles and finished near the top of its scoreboard, and an algorithmic track that learned, sometimes expensively, that a signal which only works on the day you measured it is not a signal at all.
