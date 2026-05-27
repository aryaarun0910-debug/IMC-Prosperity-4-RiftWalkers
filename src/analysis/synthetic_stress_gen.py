"""
K7 — Synthetic stress regime generator.

Generates three synthetic price regimes (trending, mean-reverting, crash)
and writes them as prices CSVs compatible with backtest.py. Used to probe
how trader.py strategies respond to extreme regimes we don't see in P3/P4
historical data.

Outputs to: data/synthetic/<regime>/prices_round_0_day_X.csv

Usage:
    py -3.12 analysis/synthetic_stress_gen.py --regime trending --product ASH_COATED_OSMIUM
    py -3.12 analysis/synthetic_stress_gen.py --regime all
"""

import argparse
import csv
import math
import os
import random


REGIMES = ["trending", "mean_revert", "crash", "high_vol", "low_vol", "regime_switch"]


def gen_prices(regime, n_ticks, seed, start_price=10000.0):
    rng = random.Random(seed)
    prices = [start_price]
    for i in range(n_ticks - 1):
        p = prices[-1]
        if regime == "trending":
            drift = 0.02
            shock = rng.gauss(0, 0.5)
        elif regime == "mean_revert":
            drift = -0.01 * (p - start_price) / start_price * 100
            shock = rng.gauss(0, 0.5)
        elif regime == "crash":
            if i == n_ticks // 2:
                p *= 0.85
            drift = 0.0
            shock = rng.gauss(0, 0.5)
        elif regime == "high_vol":
            drift = 0.0
            shock = rng.gauss(0, 3.0)
        elif regime == "low_vol":
            drift = 0.0
            shock = rng.gauss(0, 0.1)
        elif regime == "regime_switch":
            phase = (i // (n_ticks // 4)) % 2
            drift = 0.03 if phase == 0 else -0.03
            shock = rng.gauss(0, 0.5)
        else:
            drift = 0
            shock = rng.gauss(0, 0.5)
        p = max(1.0, p + drift + shock)
        prices.append(p)
    return prices


def write_csv(out_dir, regime, product, n_ticks, n_days=3, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    header = ["day", "timestamp", "product",
              "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
              "bid_price_3", "bid_volume_3",
              "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2",
              "ask_price_3", "ask_volume_3", "mid_price", "profit_and_loss"]
    for d in range(n_days):
        prices = gen_prices(regime, n_ticks, seed=seed + d)
        path = os.path.join(out_dir, f"prices_round_0_day_{d}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)
            for i, p in enumerate(prices):
                bid = int(p) - 1
                ask = int(p) + 1
                writer.writerow([d, i * 100, product,
                                 bid, 20, bid - 1, 15, bid - 2, 10,
                                 ask, 20, ask + 1, 15, ask + 2, 10,
                                 (bid + ask) / 2, 0.0])
        print(f"  wrote {path}  ({n_ticks} ticks, regime={regime})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="all", choices=REGIMES + ["all"])
    ap.add_argument("--product", default="SYNTHETIC_PRODUCT")
    ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()

    regimes = REGIMES if args.regime == "all" else [args.regime]
    for r in regimes:
        out = os.path.join(args.out, r)
        print(f"\nGenerating regime={r} -> {out}")
        write_csv(out, r, args.product, args.ticks, args.days, args.seed)


if __name__ == "__main__":
    main()
