"""
Iteration Loop — Rapid deployment orchestrator for IMC Prosperity rounds.

Usage (when R1 data drops):
    python iteration_loop.py --round 1 --prices data/r1/*.csv --trades data/r1/trades_*.csv

Workflow:
    1. deploy_round.py → auto-classify products, generate CONFIG
    2. Patch trader.py with new CONFIG
    3. Syntax check + local backtest (sanity only)
    4. Print: "Ready to submit. Upload trader.py to IMC."
    5. User drops server log path
    6. server_log_analyzer.py → parse, compare, suggest
    7. Apply suggestions → loop back to step 2

Tracks iteration count, PnL trend, and every CONFIG variant with its result.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import shutil
from datetime import datetime

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(_TOOLS_DIR)  # parent = PIPELINE root
PYTHON = sys.executable
HISTORY_DIR = os.path.join(PIPELINE_DIR, "iteration_history")
TRADER_PATH = os.path.join(PIPELINE_DIR, "trader.py")


def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def get_iteration_count():
    """Count existing iterations."""
    if not os.path.isdir(HISTORY_DIR):
        return 0
    return len([f for f in os.listdir(HISTORY_DIR) if f.startswith("iter_")])


def save_iteration(iteration, config, pnl=None, log_path=None, notes=""):
    """Save iteration snapshot."""
    ensure_dirs()
    iter_dir = os.path.join(HISTORY_DIR, f"iter_{iteration:03d}")
    os.makedirs(iter_dir, exist_ok=True)

    # Save config
    with open(os.path.join(iter_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save trader.py snapshot
    shutil.copy2(TRADER_PATH, os.path.join(iter_dir, "trader.py"))

    # Save metadata
    meta = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "pnl": pnl,
        "log_path": log_path,
        "notes": notes,
    }
    with open(os.path.join(iter_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return iter_dir


def load_history():
    """Load all iteration results for trend analysis."""
    ensure_dirs()
    results = []
    for d in sorted(os.listdir(HISTORY_DIR)):
        if not d.startswith("iter_"):
            continue
        meta_path = os.path.join(HISTORY_DIR, d, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                results.append(json.load(f))
    return results


def print_trend(history):
    """Print PnL trend across iterations."""
    if not history:
        print("  No iteration history yet.")
        return

    print(f"\n{'=' * 50}")
    print(f"  ITERATION TREND ({len(history)} runs)")
    print(f"{'=' * 50}")
    print(f"{'Iter':<6} {'PnL':>10} {'Time':>20} {'Notes'}")
    print(f"{'-' * 60}")

    pnls = []
    for h in history:
        pnl_str = f"{h['pnl']:>+10.1f}" if h['pnl'] is not None else "   pending"
        ts = h.get('timestamp', '')[:19]
        notes = h.get('notes', '')[:30]
        print(f"{h['iteration']:<6} {pnl_str} {ts:>20} {notes}")
        if h['pnl'] is not None:
            pnls.append(h['pnl'])

    if len(pnls) >= 2:
        print(f"\n  Best:  {max(pnls):+.1f}")
        print(f"  Worst: {min(pnls):+.1f}")
        print(f"  Last:  {pnls[-1]:+.1f}")
        # Convergence check
        if len(pnls) >= 3:
            recent = pnls[-3:]
            spread = max(recent) - min(recent)
            if spread < 50:
                print(f"  ** CONVERGED ** (last 3 within {spread:.0f} PnL)")
            else:
                print(f"  Trend: last 3 spread = {spread:.0f}")


def syntax_check():
    """Verify trader.py compiles."""
    result = subprocess.run(
        [PYTHON, "-c", "import importlib.util; spec = importlib.util.spec_from_file_location('t', 'trader.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"],
        cwd=PIPELINE_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  COMPILE ERROR:\n{result.stderr}")
        return False
    print(f"  Syntax check: PASS")
    return True


def run_backtest(price_csvs, trade_csvs):
    """Run backtest for sanity (numbers are directional only)."""
    cmd = [PYTHON, "backtest.py", "--seeds", "42"]
    if price_csvs:
        # backtest.py uses hardcoded paths, but we can still run it for sanity
        pass
    result = subprocess.run(cmd, cwd=PIPELINE_DIR, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  Backtest ERROR:\n{result.stderr[-500:]}")
        return None

    # Extract total PnL from output
    for line in result.stdout.split('\n'):
        if 'TOTAL' in line and '+' in line:
            parts = line.split()
            for p in parts:
                try:
                    return float(p.replace('+', '').replace(',', ''))
                except ValueError:
                    continue
    print(result.stdout[-1000:])
    return None


def analyze_log(log_path, compare_path=None):
    """Run server_log_analyzer on a log file."""
    cmd = [PYTHON, os.path.join(_TOOLS_DIR, "server_log_analyzer.py"), log_path]
    if compare_path:
        cmd.extend(["--compare", compare_path])
    result = subprocess.run(cmd, cwd=PIPELINE_DIR, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"  [WARN] {result.stderr[:200]}")

    # Extract total PnL
    for line in result.stdout.split('\n'):
        if 'TOTAL' in line:
            parts = line.split()
            for p in parts:
                try:
                    val = float(p.replace('+', '').replace(',', ''))
                    if val > 0:
                        return val
                except ValueError:
                    continue
    return None


def interactive_loop(args):
    """Interactive iteration loop."""
    ensure_dirs()
    iteration = get_iteration_count()

    # Show history
    history = load_history()
    if history:
        print_trend(history)

    best_log = None
    # Find best log from history
    for h in sorted(history, key=lambda x: x.get('pnl') or 0, reverse=True):
        if h.get('log_path') and os.path.exists(h['log_path']):
            best_log = h['log_path']
            break

    while True:
        iteration += 1
        print(f"\n{'#' * 60}")
        print(f"  ITERATION {iteration}")
        print(f"{'#' * 60}")

        # Step 1: Syntax check
        if not syntax_check():
            print("  Fix trader.py and press Enter to retry...")
            input()
            iteration -= 1
            continue

        # Step 2: Quick backtest
        print("\n  Running sanity backtest...")
        bt_pnl = run_backtest(args.prices, args.trades)
        if bt_pnl:
            print(f"  Backtest PnL: {bt_pnl:+.0f} (directional only, don't trust absolute)")

        # Step 3: Save pre-submission snapshot
        # Read current CONFIG from trader.py
        config = {}
        try:
            result = subprocess.run(
                [PYTHON, "-c", "import importlib.util; spec = importlib.util.spec_from_file_location('t', 'trader.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); import json; print(json.dumps(m.Trader()._config))"],
                cwd=PIPELINE_DIR, capture_output=True, text=True
            )
            if result.returncode == 0:
                config = json.loads(result.stdout.strip())
        except Exception:
            pass

        iter_dir = save_iteration(iteration, config, notes="pre-submit")
        print(f"  Snapshot saved to {iter_dir}")

        print(f"\n  {'*' * 40}")
        print(f"  READY TO SUBMIT")
        print(f"  Upload trader.py to IMC Prosperity")
        print(f"  {'*' * 40}")

        # Step 4: Wait for log
        print(f"\n  Paste server log path (or 'q' to quit, 't' for trend, 's' to skip):")
        user_input = input("  > ").strip()

        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 't':
            print_trend(load_history())
            iteration -= 1
            continue
        elif user_input.lower() == 's':
            continue

        log_path = user_input.strip('"').strip("'")
        if not os.path.exists(log_path):
            # Try as directory
            if os.path.isdir(log_path):
                logs = [f for f in os.listdir(log_path) if f.endswith('.log')]
                if logs:
                    log_path = os.path.join(log_path, logs[0])

        if not os.path.exists(log_path):
            print(f"  File not found: {log_path}")
            iteration -= 1
            continue

        # Step 5: Analyze
        print(f"\n  Analyzing {log_path}...")
        server_pnl = analyze_log(log_path, best_log)

        # Update iteration with result
        save_iteration(iteration, config, pnl=server_pnl, log_path=log_path, notes=f"server_pnl={server_pnl}")

        # Update best log if this is better
        if server_pnl and (best_log is None or server_pnl > max(h.get('pnl') or 0 for h in history)):
            best_log = log_path
            print(f"  NEW BEST! {server_pnl:+.1f}")

        # Show trend
        print_trend(load_history())

        print(f"\n  Make changes to trader.py, then press Enter for next iteration...")
        input()


def main():
    parser = argparse.ArgumentParser(description="Rapid iteration loop for IMC Prosperity")
    parser.add_argument("--round", type=int, default=None, help="Round number")
    parser.add_argument("--prices", nargs="+", default=None, help="Price CSV paths")
    parser.add_argument("--trades", nargs="+", default=None, help="Trade CSV paths")
    parser.add_argument("--trend", action="store_true", help="Show iteration trend only")
    args = parser.parse_args()

    if args.trend:
        print_trend(load_history())
        return

    # Expand globs
    if args.prices:
        expanded = []
        for p in args.prices:
            expanded.extend(sorted(glob.glob(p)) or [p])
        args.prices = expanded

    if args.trades:
        expanded = []
        for t in args.trades:
            expanded.extend(sorted(glob.glob(t)) or [t])
        args.trades = expanded

    interactive_loop(args)


if __name__ == "__main__":
    main()
