"""
K3 — Adversarial replay (executable harness).

Perturbs p4r1 trade/price CSVs in-place, runs backtest, then restores originals.
Outputs per-perturbation per-product PnL delta vs blessed baseline.

Usage:
    py -3.12 analysis/adversarial_replay.py --round p4r1 --perturb olivia_flip
    py -3.12 analysis/adversarial_replay.py --round p4r1 --perturb all
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PERTURBATIONS = [
    "olivia_flip",
    "size_double",
    "size_half",
    "timing_jitter",
    "price_spike_2pct",
]

ROOT = Path(__file__).resolve().parent.parent
ROUND_DATA = {
    "p4r1": [
        ROOT / "data/r1/prices/prices_round_1_day_-2.csv",
        ROOT / "data/r1/prices/prices_round_1_day_-1.csv",
        ROOT / "data/r1/prices/prices_round_1_day_0.csv",
        ROOT / "data/r1/trades/trades_round_1_day_-2.csv",
        ROOT / "data/r1/trades/trades_round_1_day_-1.csv",
        ROOT / "data/r1/trades/trades_round_1_day_0.csv",
    ],
}

_PNL_LINE = re.compile(r"^([A-Z_0-9]+)\s+([+-]?\d+)\s+([+-]?\d+)\s+([\d.]+)\s*$")


def perturb_row(row, kind, rng):
    try:
        if kind == "olivia_flip" and "buyer" in row:
            if row.get("buyer") == "Olivia":
                row["buyer"], row["seller"] = row.get("seller", ""), "Olivia"
            elif row.get("seller") == "Olivia":
                row["buyer"], row["seller"] = "Olivia", row.get("buyer", "")
        elif kind == "size_double" and "quantity" in row:
            row["quantity"] = str(int(float(row["quantity"])) * 2)
        elif kind == "size_half" and "quantity" in row:
            q = int(float(row["quantity"]))
            row["quantity"] = str(max(1, q // 2))
        elif kind == "timing_jitter" and "timestamp" in row:
            ts = int(row["timestamp"])
            row["timestamp"] = str(max(0, ts + rng.randint(-100, 100)))
        elif kind.startswith("price_spike") and "price" in row:
            pct = float(kind.split("_")[-1].replace("pct", "")) / 100.0
            try:
                p = float(row["price"])
                if rng.random() < 0.05:  # only 5% of rows perturbed
                    row["price"] = str(p * (1 + rng.choice([-1, 1]) * pct))
            except Exception:
                pass
    except Exception:
        pass
    return row


def perturb_csv(path, kind, rng):
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = [perturb_row(r, kind, rng) for r in reader]
        fields = reader.fieldnames
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_backtest(rnd):
    cmd = ["py", "-3.12", "backtest.py", "--round", rnd, "--seeds", "42"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"Backtest failed: {proc.stderr[-300:]}")
        return {}
    pnls = {}
    for line in proc.stdout.splitlines():
        m = _PNL_LINE.match(line.strip())
        if m and m.group(1) not in ("TOTAL", "Product"):
            pnls[m.group(1)] = int(m.group(2))
    return pnls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="p4r1")
    ap.add_argument("--perturb", default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.round not in ROUND_DATA:
        print(f"Unknown round {args.round}")
        return 2

    perturbs = PERTURBATIONS if args.perturb == "all" else [args.perturb]

    # Load blessed baseline
    baseline_path = ROOT / "tools/gauntlet_baseline.json"
    with open(baseline_path) as f:
        baseline = json.load(f)
    base_pnls = baseline["rounds"].get(args.round, {})

    files = [p for p in ROUND_DATA[args.round] if p.exists()]
    if not files:
        print("No data files exist for this round.")
        return 2

    # Backup originals
    backup_dir = Path(tempfile.mkdtemp(prefix="adv_replay_"))
    print(f"Backing up {len(files)} files to {backup_dir}")
    for p in files:
        shutil.copy2(p, backup_dir / p.name)

    results = {}
    try:
        for kind in perturbs:
            print(f"\n--- Perturbation: {kind} ---")
            rng = random.Random(args.seed)
            for p in files:
                # restore from backup before each perturbation
                shutil.copy2(backup_dir / p.name, p)
                perturb_csv(p, kind, rng)
            t0 = time.time()
            pnls = run_backtest(args.round)
            elapsed = time.time() - t0
            results[kind] = pnls
            print(f"  Completed in {elapsed:.1f}s")
            for prod in sorted(set(base_pnls) | set(pnls)):
                b = base_pnls.get(prod, 0)
                c = pnls.get(prod, 0)
                d = c - b
                marker = ""
                if abs(d) > 2000:
                    marker = "  <-- LARGE"
                elif abs(d) > 500:
                    marker = "  warn"
                print(f"  {prod:<32} base={b:>+8d}  perturbed={c:>+8d}  delta={d:>+8d}{marker}")
    finally:
        # Restore originals always
        print("\nRestoring originals...")
        for p in files:
            shutil.copy2(backup_dir / p.name, p)
        shutil.rmtree(backup_dir, ignore_errors=True)

    # Save results
    out = ROOT / "analysis/findings/adversarial_replay_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"round": args.round, "baseline": base_pnls, "perturbed": results}, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
