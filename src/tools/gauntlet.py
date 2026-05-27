"""
gauntlet.py — Regression gate for trader.py edits.

Runs `backtest.py --round <r>` and compares per-product PnL to a saved baseline.
Fails (exit 1) if any product regresses by more than --threshold (default 2000).

First run: --bless saves the current PnL as the new baseline.
Subsequent runs: --check (default) compares against the saved baseline.

Usage:
    py -3.12 tools/gauntlet.py --bless                # capture current as baseline
    py -3.12 tools/gauntlet.py                        # check vs baseline
    py -3.12 tools/gauntlet.py --rounds p4r1 r3 r5    # multi-round
    py -3.12 tools/gauntlet.py --threshold 1000       # tighter gate
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
BASELINE_PATH = os.path.join(_ROOT_DIR, "tools", "gauntlet_baseline.json")
DEFAULT_ROUNDS = ["p4r1"]
DEFAULT_THRESHOLD = 2000

_PNL_LINE = re.compile(r"^([A-Z_0-9]+)\s+([+-]?\d+)\s+([+-]?\d+)\s+([\d.]+)\s*$")


def run_backtest(rnd):
    """Invoke backtest.py for one round, return per-product PnL dict."""
    cmd = ["py", "-3.12", "backtest.py", "--round", rnd, "--seeds", "42"]
    proc = subprocess.run(cmd, cwd=_ROOT_DIR, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"backtest.py exited {proc.returncode}\nSTDERR: {proc.stderr[-500:]}")
    pnls = {}
    for line in proc.stdout.splitlines():
        m = _PNL_LINE.match(line.strip())
        if m and m.group(1) not in ("TOTAL", "Product"):
            pnls[m.group(1)] = int(m.group(2))
    if not pnls:
        raise RuntimeError(f"No PnL rows parsed from backtest output for {rnd}")
    return pnls


def bless(rounds):
    baseline = {"timestamp": int(time.time()), "rounds": {}}
    for r in rounds:
        print(f"Running {r}...")
        pnls = run_backtest(r)
        baseline["rounds"][r] = pnls
        print(f"  {r}: {pnls}, total {sum(pnls.values()):+,d}")
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nBaseline written: {BASELINE_PATH}")


def check(rounds, threshold):
    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: no baseline at {BASELINE_PATH}. Run with --bless first.")
        return 2
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    failed = False
    print(f"=== GAUNTLET (threshold: regression > {threshold}) ===\n")
    for r in rounds:
        if r not in baseline["rounds"]:
            print(f"WARNING: {r} not in baseline; skipping")
            continue
        print(f"Running {r}...")
        try:
            current = run_backtest(r)
        except Exception as e:
            print(f"  FAIL: {e}")
            failed = True
            continue
        base = baseline["rounds"][r]
        all_products = sorted(set(base) | set(current))
        print(f"  {'Product':<32} {'Baseline':>12} {'Current':>12} {'Delta':>12}  Status")
        print(f"  {'-'*32} {'-'*12} {'-'*12} {'-'*12}  ------")
        for p in all_products:
            b = base.get(p, 0)
            c = current.get(p, 0)
            d = c - b
            if d < -threshold:
                status = "FAIL"
                failed = True
            elif d < -500:
                status = "warn"
            elif d > 500:
                status = "+"
            else:
                status = "ok"
            print(f"  {p:<32} {b:>+12,d} {c:>+12,d} {d:>+12,d}  {status}")
        bt = sum(base.values())
        ct = sum(current.values())
        print(f"  {'TOTAL':<32} {bt:>+12,d} {ct:>+12,d} {ct-bt:>+12,d}\n")
    if failed:
        print("GAUNTLET FAILED — do not ship.")
        return 1
    print("GAUNTLET PASSED.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bless", action="store_true", help="Save current PnL as new baseline")
    ap.add_argument("--rounds", nargs="+", default=DEFAULT_ROUNDS, help=f"Rounds to test (default: {DEFAULT_ROUNDS})")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help=f"Regression failure threshold (default: {DEFAULT_THRESHOLD})")
    args = ap.parse_args()
    if args.bless:
        bless(args.rounds)
        return 0
    return check(args.rounds, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
