# 06. Round 5: 50 Products, Mean-Reversion, and a News-Driven Manual

**Result:** Algo +6,849 · Manual +89,187 (rank 411) · Final cumulative 272,456 · **Final position 700 / 18,803**

The final round. The algorithmic challenge ("Cherry Picking Winners") replaced all prior products with **50 brand-new goods in 10 categories of 5**, each with a position limit of 10. The manual challenge was a **news-driven portfolio** with quadratic transaction fees.

## Algorithmic side: the alpha hunt

With 50 unknown products and 3 days of historical data, the On-Board Advisor's hint was explicit: *don't evaluate products individually, find cluster-level structure.* So we ran a systematic signal search:

### What we tested (and what was real)

| Signal hypothesis | Result |
|---|---|
| **Within-category return correlation** | Tradable: SnackPacks showed ±0.92 return correlations (cross-day stable); Pebbles showed PEBBLES_XL anti-correlated −0.5 with every other size |
| **Single-asset z-score mean-reversion** | Tradable: a subset of products mean-revert robustly across all three days |
| Cross-category correlation (50×50) | No edge: zero stable cross-category pairs |
| Order-book imbalance (OBI) | No edge: the book is mirrored by design (bid volume = ask volume per level), so OBI is structurally zero |
| Trade-flow / informed-counterparty | No edge: all counterparty IDs anonymized in R5 |
| Long-lag autocorrelation (5–500 ticks) | No edge: no cross-day-stable signal |
| Robots intraday momentum | No edge: early-tick direction predicted final direction only 33% of the time |

The discipline here was the R4 lesson applied: **a signal only counts if it's stable across all 3 days.** We built an integrated backtester and a **walk-forward validator** (train config on days 2+3, test on held-out day 4) to enforce this.

### The mean-reversion engine
The shippable alpha was **single-asset z-score mean reversion with a momentum filter**:
- Rolling z-score on each product's mid-price.
- Enter when |z| exceeds a threshold; exit near the mean.
- **Momentum filter**: skip new entries when the product is in a strong directional trend (mean-reversion bleeds in trends; this was the key robustness fix).
- All configs validated to be positive across all 3 days *and* on the held-out walk-forward day.

Our validated engine backtested at +285K XIRECs across 3 days (all days positive, min day +74K XIRECs).

### The overfitting trap: caught this time
Our teammate also produced higher-backtest candidates that **hardcoded LONG/SHORT directional bets calibrated to one observed day** (one literally said in its docstring *"Strategy is calibrated to this known market"*). On the 1-day aesthetic test these scored up to 174K XIRECs. But run through our 3-day integrated backtester:

| Candidate | 1-day aesthetic | 3-day backtest | Worst day |
|---|---|---|---|
| Overfit (hardcoded direction) | **174K XIRECs** | +44K XIRECs | **−33K XIRECs (loses money)** |
| Signal-based (cross-day validated) | 34K XIRECs | **+616K XIRECs** | **+143K XIRECs** |

This is the R4 lesson in miniature, caught before shipping. The candidate that looked best on the single-day test was the one that lost money on a different day. We flagged it and recommended the cross-day-validated version instead.

![Overfit vs signal-based validation](assets/overfitting_comparison.png)

### Why the algo still only made +6,849 XIRECs live
The honest outcome: the R5 algo underperformed (+6,849 XIRECs, rank 896). The live scoring day was a genuinely difficult, different regime, and even validated mean-reversion gives back PnL when the day trends rather than oscillates. **The algo side remained our weak link to the end.** The lesson is that 3 days of data is a thin jury, and pure mean-reversion has a real failure mode (trending days) that no amount of in-sample validation fully removes.

## Manual side: news + quadratic fees (+89,187 XIRECs, rank 411)

The R5 manual gave 9 tradable goods on a foreign exchange, a news feed ("Ashflow Alpha"), a 1,000,000 budget, and a **quadratic fee**: `fee = (allocation%)² × budget`.

We solved it cleanly:

1. **Read each news item for direction + magnitude** (e.g., a food-safety sales-halt = strong SELL; a 2.7× user-growth report = strong BUY; an influencer "pump" call = fade).
2. **Closed-form optimal sizing.** With quadratic fees, the PnL per product is `p·10000·|r| − 100·p²`. Taking the derivative gives the optimal allocation: **`p* = 50 × |expected return|`**. The quadratic fee naturally prevents over-concentration.
3. **Lagrangian** when allocations summed past 100% of budget.
4. **Monte-Carlo** (50,000 draws) over our return-estimate uncertainty to get a PnL distribution, not just a point estimate.
5. **Behavioral / game-theory layer**: IMC's game-makers are active traders, so we read the deliberately-suspicious "pump" articles as fade-tests and shorted them small.

Projected ~155K XIRECs (range 120K–190K XIRECs). **Delivered 89,187 XIRECs**: the "wisdom of the crowd" correction pulled returns toward the middle of each range, consistent with the moderator's stated grading mechanic. **Rank 411 / 18,803: top 2.2% on the manual.**

## Final standing

**272,456 cumulative XIRECs → 700th / 18,803 teams (top 3.7%).** The manual carried the round; the algo was the anchor.

→ Next: [Manual Rounds Deep Dive](07-manual-rounds.md) · [Overfitting Lessons](09-overfitting-lessons.md)
