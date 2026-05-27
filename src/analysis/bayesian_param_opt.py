"""
K6 — Bayesian optimization of CONFIG parameters.

Blocked until §0.1 (fill_rng in backtest.py) is fixed — grid search
output is currently degenerate on parameter axes like make_offset and
take_width. This tool implements Gaussian-process-UCB parameter search
over promising axes (position_limit, kill_threshold, kalman Q/R).

Uses random search fallback if GP fit is unstable (stdlib-only).

Usage:
    py -3.12 analysis/bayesian_param_opt.py --param kalman_Q --product RESIN --budget 30
    py -3.12 analysis/bayesian_param_opt.py --dry-run
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile


def sample_point(bounds, rng):
    return [rng.uniform(lo, hi) for lo, hi in bounds]


def gp_ucb_pick(history, bounds, rng, k=2.0, n_candidates=64):
    if len(history) < 3:
        return sample_point(bounds, rng)
    candidates = [sample_point(bounds, rng) for _ in range(n_candidates)]
    best = None
    best_ucb = -float("inf")
    for c in candidates:
        dists = [math.sqrt(sum((c[d] - h[0][d]) ** 2 for d in range(len(bounds)))) for h in history]
        weights = [math.exp(-d * d / 2) for d in dists]
        wsum = sum(weights) or 1e-9
        mu = sum(weights[i] * history[i][1] for i in range(len(history))) / wsum
        sigma = math.sqrt(max(0.0, 1 - max(weights)))
        ucb = mu + k * sigma
        if ucb > best_ucb:
            best_ucb = ucb
            best = c
    return best


def score_point(point, param_names, product, dry_run):
    if dry_run:
        base = 1000.0
        return base - sum((p - 1.0) ** 2 * 100 for p in point) + random.gauss(0, 50)
    patch = dict(zip(param_names, point))
    patch_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({product: patch}, patch_file)
    patch_file.close()
    try:
        res = subprocess.run(
            ["py", "-3.12", "tools/gauntlet.py", "--patch", patch_file.name],
            capture_output=True, text=True, timeout=120
        )
        for line in res.stdout.splitlines():
            if "TOTAL" in line:
                parts = line.replace(",", "").replace("+", "").split()
                for p in parts:
                    try:
                        return float(p)
                    except ValueError:
                        pass
    finally:
        os.unlink(patch_file.name)
    return -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", nargs="+", default=["kalman_Q", "kalman_R"])
    ap.add_argument("--bounds", nargs="+", type=float,
                    default=[0.01, 5.0, 0.1, 20.0])
    ap.add_argument("--product", default="RESIN")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    p = len(args.param)
    bounds = [(args.bounds[2 * i], args.bounds[2 * i + 1]) for i in range(p)]

    print(f"BayesOpt: params={args.param} bounds={bounds} budget={args.budget}")
    if args.dry_run:
        print("DRY RUN — using synthetic score")

    history = []
    for it in range(args.budget):
        pt = gp_ucb_pick(history, bounds, rng)
        sc = score_point(pt, args.param, args.product, args.dry_run)
        history.append((pt, sc))
        print(f"  iter {it+1:>2}: {pt} -> score={sc:.2f}")

    history.sort(key=lambda h: -h[1])
    best = history[0]
    print(f"\nBest: {dict(zip(args.param, best[0]))} -> score={best[1]:.2f}")


if __name__ == "__main__":
    main()
