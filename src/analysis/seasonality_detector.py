"""
K5 — Day/tick seasonality detector.

For each product, bucket mid-price changes by (ticks since day start / bucket_size)
and report mean return, std, and fill-volume per bucket. Reveals time-of-day
patterns in volatility or directional drift — e.g., if bucket[0..50] has mean
+0.3 vs bucket[900..950] mean -0.3, we ship a time-of-day sizing override.

Usage:
    py -3.12 analysis/seasonality_detector.py --round 3
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


def load_product_returns(data_dir):
    prices_by_product_by_day = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("prices_") or not fname.endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    d = int(row["day"])
                    ts = int(row["timestamp"])
                    mid = float(row["mid_price"])
                except (KeyError, ValueError):
                    continue
                prices_by_product_by_day[row["product"]][d].append((ts, mid))
    return prices_by_product_by_day


def analyze_product(day_data, bucket_size):
    bucket_returns = defaultdict(list)
    for d, series in day_data.items():
        series.sort()
        for i in range(1, len(series)):
            ts, mid = series[i]
            prev_mid = series[i - 1][1]
            if prev_mid == 0:
                continue
            ret = (mid - prev_mid) / prev_mid
            bucket = int(ts / bucket_size)
            bucket_returns[bucket].append(ret)
    return bucket_returns


def summarize(bucket_returns, top_k=5):
    stats = []
    for b, rets in bucket_returns.items():
        if len(rets) < 20:
            continue
        m = sum(rets) / len(rets)
        s = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
        stats.append((b, m, s, len(rets)))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--bucket", type=int, default=50000)  # 50K ticks per bucket
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if args.dir is None and args.round is not None:
        args.dir = f"reference/p3_data/Round {args.round} Data"
    if args.dir is None or not os.path.isdir(args.dir):
        print("ERROR: pass --dir or --round")
        sys.exit(1)

    print(f"Loading: {args.dir}  (bucket={args.bucket})")
    data = load_product_returns(args.dir)

    print(f"\n{'product':<35} {'top_bucket':>10} {'top_mean':>12} {'bot_bucket':>10} {'bot_mean':>12} {'n_buckets':>10}")
    for product, day_data in sorted(data.items()):
        stats = summarize(analyze_product(day_data, args.bucket))
        if len(stats) < 3:
            continue
        stats.sort(key=lambda s: s[1])
        bot = stats[0]
        top = stats[-1]
        print(f"{product:<35} {top[0]:>10} {top[1]:>12.6f} {bot[0]:>10} {bot[1]:>12.6f} {len(stats):>10}")


if __name__ == "__main__":
    main()
