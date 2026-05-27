"""
K1 — Cointegration auto-discovery.

For every pair of products in a round, fit a linear relationship
mid_A = alpha + beta * mid_B and test stationarity of residuals
via an Engle-Granger style ADF-light (autocorrelation-based proxy,
no scipy). Surfaces candidate pairs for pairs_arb and basket_arb.

Usage:
    py -3.12 analysis/cointegration_discovery.py --round 3
    py -3.12 analysis/cointegration_discovery.py --dir "reference/p3_data/Round 3 Data"
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


def load_mids(data_dir):
    mids = defaultdict(list)
    ts_by_product = defaultdict(list)
    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("prices_") or not fname.endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    m = float(row["mid_price"])
                except (KeyError, ValueError):
                    continue
                p = row["product"]
                try:
                    ts = int(row["timestamp"])
                except (KeyError, ValueError):
                    continue
                mids[p].append(m)
                ts_by_product[p].append(ts)
    return mids, ts_by_product


def ols_alpha_beta(y, x):
    n = len(y)
    if n < 10:
        return None, None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    if den <= 1e-12:
        return None, None
    beta = num / den
    alpha = my - beta * mx
    return alpha, beta


def acf1(series):
    n = len(series)
    if n < 5:
        return 0.0
    m = sum(series) / n
    num = sum((series[i] - m) * (series[i - 1] - m) for i in range(1, n))
    den = sum((series[i] - m) ** 2 for i in range(n))
    if den <= 1e-12:
        return 0.0
    return num / den


def half_life(rho):
    if rho <= 0 or rho >= 1:
        return float("inf")
    return -math.log(2) / math.log(rho)


def test_pair(yseries, xseries):
    n = min(len(yseries), len(xseries))
    if n < 200:
        return None
    y = yseries[:n]
    x = xseries[:n]
    alpha, beta = ols_alpha_beta(y, x)
    if alpha is None:
        return None
    resid = [y[i] - (alpha + beta * x[i]) for i in range(n)]
    rho = acf1(resid)
    s = (sum((r - sum(resid) / n) ** 2 for r in resid) / n) ** 0.5
    hl = half_life(rho)
    return {
        "n": n,
        "alpha": alpha,
        "beta": beta,
        "resid_std": s,
        "acf1": rho,
        "half_life": hl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    if args.dir is None and args.round is not None:
        args.dir = f"reference/p3_data/Round {args.round} Data"
    if args.dir is None or not os.path.isdir(args.dir):
        print("ERROR: pass --dir or --round")
        sys.exit(1)

    print(f"Loading mids from: {args.dir}")
    mids, _ = load_mids(args.dir)
    products = [p for p, m in mids.items() if len(m) >= 500]
    print(f"Products with >=500 mid samples: {len(products)}")

    results = []
    for i, pa in enumerate(products):
        for pb in products[i + 1:]:
            r = test_pair(mids[pa], mids[pb])
            if r is None:
                continue
            r["a"] = pa
            r["b"] = pb
            r["score"] = (1 - abs(r["acf1"])) * 100 + (1.0 / max(r["half_life"], 1) * 50)
            if r["acf1"] < 0.95 and r["half_life"] < 500:
                results.append(r)

    results.sort(key=lambda r: r["half_life"])
    print(f"\nTop {args.top} cointegrated pairs (shortest half-life first):")
    print(f"{'A':<35} {'B':<35} {'beta':>8} {'hl':>8} {'acf1':>6} {'std':>8}")
    for r in results[: args.top]:
        print(f"{r['a']:<35} {r['b']:<35} {r['beta']:>8.3f} {r['half_life']:>8.1f} {r['acf1']:>6.3f} {r['resid_std']:>8.2f}")


if __name__ == "__main__":
    main()
