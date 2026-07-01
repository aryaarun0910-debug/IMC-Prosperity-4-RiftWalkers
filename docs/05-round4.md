# 05 — Round 4: Exotic Options & the Algo Overfitting Disaster

**Result:** Algo +19,664 · Manual +23,566 · Cumulative 176,419 (position 917)

Round 4 was our hardest lesson. The product set was the same underlying as R3 (`VELVETFRUIT_EXTRACT`) plus a chain of voucher/option strikes (`VEV_4000` … `VEV_6500`) and **exotic options** on the manual side. Both sides under-delivered relative to projections — and the *why* became the central technical lesson of the competition.

## Algorithmic side — projected $130K, delivered $19.7K

We shipped a version (internally "v11") that added **deep-in-the-money short option positions** (selling VEV_4000/4500/5000 calls) on top of the spot market-making. The thesis: these options would decay toward intrinsic value as the underlying drifted, and the short premium would be collected.

In backtesting against the available day's data, this looked excellent — projected ~$130K. **It delivered $19,664.**

### What went wrong (the root cause)

The fatal flaw: **we calibrated a directional thesis to one observed day's price trajectory and assumed the live (unseen) scoring day would behave the same way.**

- On the day we tuned against, the underlying drifted down monotonically (−$63 over the window). The deep-ITM short calls profited as the options tracked intrinsic value down.
- On the **live scoring day**, the underlying barely moved net (−$9) but **spiked intraday** to near our stop-loss levels.
- The short option book ate massive intraday mark-to-market drawdowns (one strike peaked at +$7,633 then troughed at −$20,991 before recovering to −$5,781).
- The market-making products that were reliable in backtest also delivered only ~50% of their backtested PnL — single-day variance on identical config.

![R4 VEV_4000 intraday drawdown](assets/r4_drawdown.png)

That strike is VEV_4000 — the real, tick-by-tick mark-to-market PnL from the actual live-scored submission (`PIPELINE/data/r4/Round 4 Full 10k tick run/545097/545097.json`, total algo profit $19,663.66, matching the $19,664 reported above). Two full round trips through a ~$28K swing before the position ever nets out — this is what "the underlying spiked intraday" looks like in practice, not just in prose.

**The mistake had a name we'd use repeatedly afterward: overfitting to a single observed day.** A strategy that depends on the *direction* of a specific day's drift is not a signal — it's a bet that tomorrow looks like today. See the full analysis in [Overfitting Lessons](09-overfitting-lessons.md).

## Manual side — the optimal we missed

The R4 manual was an **exotic options pricing** puzzle: trade an underlying (AC) plus vanilla puts/calls, a **chooser option**, a **binary put**, and a **knock-out put** — each requiring its own pricing model.

The theoretically optimal trade (which we reverse-engineered afterward) was a clean structure:
- **Max long the underlying** (directional)
- **Buy every cheap put** (cheap downside insurance under the simulated GBM with flat vol)
- **Sell every overpriced exotic** (chooser, binary put, knock-out — all priced rich vs theory)

Optimal payout: **177,980.** We scored **23,566** — leaving ~$155K on the table.

Our errors:
1. **Skipped trades** because individual edges looked small — but at max volume × contract-size multiplier, even a "small" edge is thousands.
2. **A direction error** on the knock-out put (our submitted side disagreed with our own rationale text — a transcription mistake that alone cost ~$18K).
3. **Didn't compute Monte-Carlo theory** for the path-dependent exotics; relied on intuition where a simulation was needed.

## Takeaways (these directly shaped R5)

- **A directional bet calibrated to one day is not alpha.** This is the thesis of the whole competition for us.
- **Don't skip positive-EV trades.** Small × scale = significant.
- **Derive Buy/Sell from the model, not by hand** — eliminate transcription error.
- **Path-dependent exotics need Monte-Carlo pricing**, not intuition.

R4 hurt. But it produced the discipline that made our R5 manual (+$89K, rank 411) and our analysis of the R5 algo candidates genuinely rigorous.

→ Next: [Round 5](06-round5.md)
