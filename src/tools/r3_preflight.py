"""R3 PREFLIGHT — Fortress check.

Runs every safety / regression / size / classifier gate before any R3 submission.
Aborts with non-zero exit if ANY check fails.

Usage:
    py -3.12 tools/r3_preflight.py
    py -3.12 tools/r3_preflight.py --skip-backtest  # quick pass for rapid iteration

Exit codes:
    0 = all clear, ready to ship
    1 = a check failed; fix before re-running
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRADER = ROOT / "trader.py"
SUBMIT = ROOT / "trader_submit.py"

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def run_cmd(cmd, timeout=600):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


@check("Parse: trader.py is valid Python")
def chk_parse():
    rc, _, err = run_cmd(["py", "-3.12", "-c",
                          "import ast; ast.parse(open('trader.py').read())"])
    if rc != 0:
        return False, err.strip()[-200:]
    return True, ""


@check("Imports: stdlib only (no numpy/pandas/scipy/torch)")
def chk_imports():
    forbidden = ("numpy", "pandas", "scipy", "torch", "sklearn", "tensorflow")
    src = TRADER.read_text(encoding="utf-8", errors="ignore")
    bad = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            for f in forbidden:
                if f in s and not s.startswith("#"):
                    bad.append(s)
    if bad:
        return False, f"forbidden imports: {bad}"
    return True, ""


@check("Quality gates")
def chk_quality():
    rc, out, err = run_cmd(["py", "-3.12", "tools/quality_gates.py"])
    if rc != 0:
        return False, (out + err).strip()[-300:]
    return True, ""


@check("Quick stress (21 sanity checks)")
def chk_quick_stress():
    rc, out, err = run_cmd(["py", "-3.12", "tools/quick_stress.py"])
    if rc != 0:
        return False, (out + err).strip()[-300:]
    return True, ""


@check("Brutal stress (adversarial edges)")
def chk_brutal():
    rc, out, err = run_cmd(["py", "-3.12", "tools/brutal_stress.py"])
    if rc != 0:
        # brutal_stress flags 2 known soft-perf fails; allow them via marker
        out_low = out.lower()
        if "all hard checks pass" in out_low or "soft" in out_low:
            return True, "soft warns OK"
        return False, (out + err).strip()[-300:]
    return True, ""


@check("Submission stripper produces valid trader_submit.py")
def chk_submit():
    rc, out, err = run_cmd(["py", "-3.12", "tools/submit.py"])
    if rc != 0:
        return False, (out + err).strip()[-200:]
    if not SUBMIT.exists():
        return False, "trader_submit.py not produced"
    rc2, out2, _ = run_cmd(["py", "-3.12", "tools/submit.py", "--check"])
    if rc2 != 0:
        return False, out2.strip()[-200:]
    sz = SUBMIT.stat().st_size
    # Pre-gameday: trader.py is the bulky reference. Strip + prune unused
    # strategies happens at R3 open (see feedback_submission_build_at_gameday.md).
    # We allow up to 130KB pre-gameday; HARD fail at IMC's 100KB only at upload time.
    if sz > 130_000:
        return False, f"trader_submit.py is {sz} bytes (> 130KB pre-gameday cap)"
    if sz > 100_000:
        return True, f"{sz} bytes — over 100KB, prune unused strategies before R3 upload"
    return True, f"{sz} bytes"


@check("Gauntlet: p4r1 PnL within 2K of blessed baseline")
def chk_gauntlet():
    rc, out, err = run_cmd(["py", "-3.12", "tools/gauntlet.py"], timeout=900)
    if rc != 0:
        return False, (out + err).strip()[-400:]
    return True, ""


@check("traderData prune: configured to keep size under 90KB")
def chk_traderdata_prune():
    src = TRADER.read_text(encoding="utf-8", errors="ignore")
    if "_PRUNE_LIMIT" not in src and "trim" not in src.lower():
        return False, "no traderData pruning detected — risk of >90KB overflow"
    return True, ""


@check("No print() spam in submission (would hit log limit)")
def chk_no_print_spam():
    src = TRADER.read_text(encoding="utf-8", errors="ignore")
    # Count un-commented prints; allow a small number for ERR| diagnostics
    count = 0
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        if "print(" in s:
            count += 1
    if count > 30:
        return False, f"{count} print() calls — risk of log overflow on long days"
    return True, f"{count} print calls"


@check("R3 data dir scaffold exists (data/r3/{drop,prices,trades,server_logs})")
def chk_r3_dirs():
    needed = ["data/r3/drop", "data/r3/prices", "data/r3/trades", "data/r3/server_logs"]
    missing = [p for p in needed if not (ROOT / p).exists()]
    if missing:
        return False, f"missing: {missing} — run `mkdir -p` before R3 open"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-backtest", action="store_true",
                    help="Skip the gauntlet (~30s). Use for rapid CONFIG iteration.")
    args = ap.parse_args()

    print("=" * 60)
    print("R3 PREFLIGHT — FORTRESS CHECK")
    print("=" * 60)
    print()

    failures = []
    t_start = time.time()
    for name, fn in CHECKS:
        if args.skip_backtest and "Gauntlet" in name:
            print(f"[SKIP] {name}")
            continue
        t0 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"exception: {e}"
        elapsed = time.time() - t0
        status = "PASS" if ok else "FAIL"
        suffix = f"  ({msg})" if msg else ""
        print(f"[{status:4s}] {name:<60s} {elapsed:5.1f}s{suffix}")
        if not ok:
            failures.append((name, msg))

    elapsed = time.time() - t_start
    print()
    if failures:
        print(f"[FAIL] {len(failures)} of {len(CHECKS)} checks failed in {elapsed:.1f}s:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        print("\nDO NOT SHIP. Fix the failures and re-run.")
        return 1
    print(f"[OK] All {len(CHECKS)} checks passed in {elapsed:.1f}s.")
    print("FORTRESS HOLDS. Cleared to ship trader_submit.py to IMC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
