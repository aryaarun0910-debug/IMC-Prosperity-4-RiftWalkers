"""
K2 — Monte Carlo PnL distribution.

Given a historical per-tick PnL series (from gauntlet or backtest.py output),
bootstrap-resample blocks to estimate the 5/50/95 percentiles and worst-case
drawdown under reshuffled day orderings. Useful for sizing the final R5
position bet and judging whether a backtest win is a regime fluke or
actually robust to permutation.

Usage:
    py -3.12 analysis/monte_carlo_pnl.py --pnl data/r1/pnl_per_tick.csv
    py -3.12 analysis/monte_carlo_pnl.py --synthetic  # runs on fake data
"""

import argparse
import csv
import math
import os
import random
import sys


def load_pnl_series(path):
    series = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                series.append(float(row[-1]))
            except ValueError:
                continue
    return series


def block_bootstrap(series, block_size, n_iters):
    n = len(series)
    if n < block_size:
        return []
    num_blocks = n // block_size
    finals = []
    max_dds = []
    for _ in range(n_iters):
        order = list(range(num_blocks))
        random.shuffle(order)
        seq = []
        for idx in order:
            seq.extend(series[idx * block_size: (idx + 1) * block_size])
        cum = [0]
        for v in seq:
            cum.append(cum[-1] + v)
        finals.append(cum[-1])
        peak = cum[0]
        mdd = 0.0
        for c in cum:
            if c > peak:
                peak = c
            dd = peak - c
            if dd > mdd:
                mdd = dd
        max_dds.append(mdd)
    return finals, max_dds


def pctile(sorted_arr, q):
    n = len(sorted_arr)
    if n == 0:
        return 0.0
    idx = min(n - 1, max(0, int(q * (n - 1))))
    return sorted_arr[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnl", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    if args.synthetic:
        series = [random.gauss(0.05, 2.0) for _ in range(30000)]
        print(f"Synthetic series: n={len(series)}, mean={sum(series)/len(series):.3f}")
    elif args.pnl and os.path.exists(args.pnl):
        series = load_pnl_series(args.pnl)
        print(f"Loaded {len(series)} pnl ticks from {args.pnl}")
    else:
        print("ERROR: pass --pnl <file> or --synthetic")
        sys.exit(1)

    finals, max_dds = block_bootstrap(series, args.block, args.iters)
    if not finals:
        print("not enough data")
        sys.exit(1)
    finals.sort()
    max_dds.sort()

    print(f"\nBootstrap ({args.iters} iters, block={args.block}):")
    print(f"  Final PnL: p05={pctile(finals,0.05):+,.0f}  p50={pctile(finals,0.50):+,.0f}  p95={pctile(finals,0.95):+,.0f}")
    print(f"  Max DD:    p05={pctile(max_dds,0.05):,.0f}  p50={pctile(max_dds,0.50):,.0f}  p95={pctile(max_dds,0.95):,.0f}")
    baseline = sum(series)
    better = sum(1 for f in finals if f >= baseline)
    print(f"  Baseline final: {baseline:+,.0f}  (bootstrap >= baseline: {100*better/len(finals):.1f}%)")


if __name__ == "__main__":
    main()
