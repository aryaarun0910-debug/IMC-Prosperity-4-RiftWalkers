"""R3 gameday orchestrator (2026-04-24).

One command from the PIPELINE/ directory after dropping new files into
data/r3/drop/:

    py -3.12 tools/r3_kickoff.py

Pipeline:
  1. sort_drops --round 3            (fans drop/ into prices/trades/server_logs/wiki/intel)
  2. first_log_intel on any new server logs
  3. r3_preflight                    (10-check fortress gate)
  4. gauntlet                        (regression vs blessed p4r1 baseline)
  5. summary report

Each step is independent; failures in steps 2-4 are logged but the script keeps
going so you see the full picture.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3_DIR = os.path.join(PIPELINE_DIR, "data", "r3")


def run(cmd, label, optional=False):
    print(f"\n{'='*70}\n[STEP] {label}\n  $ {cmd}\n{'='*70}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, shell=True, cwd=PIPELINE_DIR,
                                capture_output=False)
        dt = time.time() - t0
        ok = (result.returncode == 0)
        status = "OK" if ok else ("WARN" if optional else "FAIL")
        print(f"\n[STEP] {label}: {status} ({dt:.1f}s, exit={result.returncode})")
        return ok
    except Exception as e:
        print(f"\n[STEP] {label}: EXCEPTION {e}")
        return False


def list_logs():
    sl = os.path.join(R3_DIR, "server_logs")
    if not os.path.isdir(sl):
        return []
    return sorted(os.path.join(sl, f) for f in os.listdir(sl)
                  if f.endswith(".log") or f.endswith(".json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gauntlet", action="store_true",
                    help="Skip the gauntlet regression check")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="Skip the r3_preflight fortress gate")
    args = ap.parse_args()

    print(f"\nR3 KICKOFF — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PIPELINE_DIR={PIPELINE_DIR}")
    print(f"R3_DIR={R3_DIR}")

    if not os.path.isdir(R3_DIR):
        print(f"ERROR: R3 folder missing at {R3_DIR}. Create it first.")
        sys.exit(1)

    results = {}

    logs_before = set(list_logs())
    results["sort_drops"] = run("py -3.12 tools/sort_drops.py --round 3",
                                "Sort drop/ into r3/ subfolders")
    logs_after = set(list_logs())
    new_logs = sorted(logs_after - logs_before)

    if new_logs:
        for log in new_logs:
            results[f"first_log_intel:{os.path.basename(log)}"] = run(
                f"py -3.12 tools/first_log_intel.py {shlex.quote(log)}",
                f"First-log intel on {os.path.basename(log)}", optional=True
            )
    else:
        print("\n[INFO] No new server logs detected — skipping first_log_intel.")

    if not args.skip_preflight:
        results["r3_preflight"] = run("py -3.12 tools/r3_preflight.py",
                                      "r3_preflight fortress gate")

    if not args.skip_gauntlet:
        results["gauntlet"] = run("py -3.12 tools/gauntlet.py",
                                  "Gauntlet regression vs blessed baseline")

    print(f"\n{'='*70}\nR3 KICKOFF SUMMARY\n{'='*70}")
    longest = max((len(k) for k in results), default=20)
    for k, v in results.items():
        flag = "OK  " if v else "FAIL"
        print(f"  [{flag}] {k}")

    failed = [k for k, v in results.items() if not v
              and not k.startswith("first_log_intel")]
    if failed:
        print(f"\n{len(failed)} required step(s) failed: {failed}")
        sys.exit(2)
    print("\nAll required steps passed.")


if __name__ == "__main__":
    main()
