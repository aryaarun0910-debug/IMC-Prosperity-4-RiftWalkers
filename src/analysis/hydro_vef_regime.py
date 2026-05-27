"""HYDROGEL / VELVETFRUIT regime profiler.

Computes:
  - EMA(window) deviation distribution at multiple windows
  - Mean-reversion half-life from AR(1) on returns
  - Oracle-PnL upper bound assuming perfect deviation-sized timing
  - Per-window suggested deviation threshold (75/90/95th percentile of |dev|)

Usage:
  py -3.12 analysis/hydro_vef_regime.py
"""
import math
import os
import statistics
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "r3", "prices")


def wall_mid(row):
    """Wall mid: largest-volume bid/ask midpoint."""
    bids = []
    asks = []
    for i in range(1, 4):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv > 0:
            bids.append((float(bp), float(bv)))
        if pd.notna(ap) and pd.notna(av) and av > 0:
            asks.append((float(ap), float(av)))
    if not bids or not asks:
        return None
    bmax = max(bids, key=lambda x: x[1])[0]
    amax = max(asks, key=lambda x: x[1])[0]
    return 0.5 * (bmax + amax)


def ema_series(values, window):
    """Exponential moving average with alpha = 2/(window+1)."""
    alpha = 2.0 / (window + 1.0)
    out = []
    e = values[0]
    for v in values:
        e = alpha * v + (1 - alpha) * e
        out.append(e)
    return out


def ar1_half_life(returns):
    """Half-life of mean reversion from AR(1) phi: t = ln(2) / -ln(phi)."""
    if len(returns) < 100:
        return None
    n = len(returns)
    mean = sum(returns) / n
    cov = sum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, n)) / (n - 1)
    var = sum((r - mean) ** 2 for r in returns) / n
    if var == 0:
        return None
    phi = cov / var
    if phi <= 0 or phi >= 1:
        return None
    return math.log(2.0) / -math.log(phi)


def oracle_pnl(mids, ema, threshold, limit):
    """PnL upper bound under deviation-sized targeting.

    target_pos = -clip(deviation/threshold * limit, -limit, limit)
    rebalance instantly each tick (no spread cost — pure direction).
    """
    pos = 0.0
    pnl = 0.0
    for i in range(1, len(mids)):
        dev = mids[i - 1] - ema[i - 1]
        target = max(-limit, min(limit, -dev / threshold * limit))
        pos = target  # instant rebalance
        pnl += pos * (mids[i] - mids[i - 1])
    return pnl


def analyse(name, df):
    df = df[df["product"] == name].sort_values("timestamp").reset_index(drop=True)
    print(f"\n=== {name}  (n={len(df)}) ===")
    mids = []
    for _, row in df.iterrows():
        m = wall_mid(row)
        if m is None:
            m = float(row.get("mid_price", 0)) if pd.notna(row.get("mid_price")) else None
        if m is not None:
            mids.append(m)
    print(f"  valid_mids={len(mids)}  mid_mean={statistics.mean(mids):.2f}  "
          f"mid_std={statistics.stdev(mids):.2f}  range=[{min(mids):.0f}, {max(mids):.0f}]")

    rets = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    print(f"  return_std={statistics.stdev(rets):.3f}  abs_mean={statistics.mean(abs(r) for r in rets):.3f}")

    hl = ar1_half_life(rets)
    print(f"  AR(1) half-life of returns: {hl}")

    print(f"\n  Window | dev_p50 | dev_p90 | dev_p99 | oracle@thr=p90 (limit=200)")
    print(f"  -------+---------+---------+---------+-------------------------")
    for window in [5, 10, 20, 30, 50, 100]:
        ema = ema_series(mids, window)
        devs = [abs(mids[i] - ema[i]) for i in range(window, len(mids))]
        p50 = statistics.quantiles(devs, n=100)[49]
        p90 = statistics.quantiles(devs, n=100)[89]
        p99 = statistics.quantiles(devs, n=100)[98]
        oracle = oracle_pnl(mids, ema, p90, 200)
        print(f"  {window:>6} | {p50:>7.2f} | {p90:>7.2f} | {p99:>7.2f} | {oracle:>+10,.0f}")


def main():
    days = []
    for d in [0, 1, 2]:
        path = os.path.join(DATA, f"prices_round_3_day_{d}.csv")
        df = pd.read_csv(path, sep=";")
        df.columns = [c.strip().lower() for c in df.columns]
        days.append(df)

    full = pd.concat(days, ignore_index=True)
    for prod in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
        analyse(prod, full)


if __name__ == "__main__":
    main()
