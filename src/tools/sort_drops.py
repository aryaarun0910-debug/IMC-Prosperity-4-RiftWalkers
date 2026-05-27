"""
Drop Sorter — Auto-sorts files from data/r1/drop/ into the right folders.

Usage:
    py -3.12 tools/sort_drops.py              # Sort all files in drop/
    py -3.12 tools/sort_drops.py --round 2    # Sort into data/r2/ instead
    py -3.12 tools/sort_drops.py --watch      # Watch for new files continuously

Sorts by extension and content:
    .log / .json with activityLogs  -> server_logs/
    .csv with prices_               -> prices/
    .csv with trades_               -> trades/
    .py                             -> submissions/ (numbered)
    anything else                   -> stays in drop/ with warning
"""

import argparse
import json
import os
import shutil
import sys
import time

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_round_dir(round_num):
    return os.path.join(PIPELINE_DIR, "data", f"r{round_num}")


def ensure_structure(round_dir):
    for sub in ["drop", "server_logs", "prices", "trades", "observations",
                "submissions", "wiki", "intel"]:
        os.makedirs(os.path.join(round_dir, sub), exist_ok=True)


def is_server_log(filepath):
    """Check if a file is a server log (JSON with activityLogs or sandboxLog)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            first_chunk = f.read(2000)
        # Quick heuristic before full parse
        if "activityLogs" in first_chunk or "sandboxLog" in first_chunk:
            return True
        # Try full JSON parse for small files
        if len(first_chunk) < 2000:
            data = json.loads(first_chunk)
            return isinstance(data, dict) and ("activityLogs" in data or "sandboxLog" in data)
        return False
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False


def classify_file(filepath):
    """Classify a file and return its destination subfolder."""
    name = os.path.basename(filepath).lower()
    ext = os.path.splitext(name)[1]

    # Wiki / round info: text files, screenshots
    if ext in (".txt", ".md") and ("wiki" in name or "round" in name or "info" in name):
        return "wiki"
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"):
        return "wiki"

    # Intel JSONs (manual solver, ceilings) -> intel/
    if ext == ".json" and ("intel" in name or "ceiling" in name or "manual" in name
                            or "fingerprint" in name):
        return "intel"

    # Python files -> submissions
    if ext == ".py":
        return "submissions"

    # CSV files -> prices, trades, or observations (R4+ macarons)
    if ext == ".csv":
        if "observation" in name:
            return "observations"
        elif "price" in name:
            return "prices"
        elif "trade" in name:
            return "trades"
        else:
            # Peek at headers
            try:
                with open(filepath, "r") as f:
                    header = f.readline().lower()
                if "mid_price" in header or "bid_price" in header:
                    return "prices"
                elif "buyer" in header or "seller" in header:
                    return "trades"
            except OSError:
                pass
            return "prices"  # default for CSV

    # Log / JSON files -> check if server log
    if ext in (".log", ".json"):
        if is_server_log(filepath):
            return "server_logs"
        # Plain .log or .json that isn't a server log
        return "server_logs"  # still put in server_logs, user can move if wrong

    return None  # unknown


def next_submission_number(submissions_dir):
    """Get next submission number."""
    existing = [f for f in os.listdir(submissions_dir) if f.startswith("v")]
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.split("_")[0].replace("v", "")))
        except (ValueError, IndexError):
            pass
    return max(nums, default=0) + 1


def sort_drops(round_dir):
    """Sort all files in drop/ into appropriate folders."""
    drop_dir = os.path.join(round_dir, "drop")
    if not os.path.isdir(drop_dir):
        print(f"  No drop folder at {drop_dir}")
        return 0

    files = [f for f in os.listdir(drop_dir)
             if os.path.isfile(os.path.join(drop_dir, f)) and not f.startswith(".")]

    if not files:
        print("  Drop folder is empty. Nothing to sort.")
        return 0

    sorted_count = 0
    for fname in files:
        src = os.path.join(drop_dir, fname)
        dest_folder = classify_file(src)

        if dest_folder is None:
            print(f"  [SKIP] {fname} — unknown file type, leaving in drop/")
            continue

        dest_dir = os.path.join(round_dir, dest_folder)

        # Special naming for submissions
        if dest_folder == "submissions":
            n = next_submission_number(dest_dir)
            dest_name = f"v{n:03d}_{fname}"
        else:
            dest_name = fname

        dest_path = os.path.join(dest_dir, dest_name)

        # Don't overwrite
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(dest_name)
            dest_name = f"{base}_{int(time.time())}{ext}"
            dest_path = os.path.join(dest_dir, dest_name)

        shutil.move(src, dest_path)
        print(f"  [SORTED] {fname} -> {dest_folder}/{dest_name}")
        sorted_count += 1

    return sorted_count


def watch_drops(round_dir, interval=5):
    """Watch drop folder for new files."""
    drop_dir = os.path.join(round_dir, "drop")
    print(f"  Watching {drop_dir} for new files (Ctrl+C to stop)...")
    seen = set()
    try:
        while True:
            if os.path.isdir(drop_dir):
                current = set(os.listdir(drop_dir))
                new_files = current - seen
                if new_files:
                    print(f"\n  New files detected: {', '.join(new_files)}")
                    count = sort_drops(round_dir)
                    if count:
                        print(f"  Sorted {count} file(s).\n")
                seen = set(os.listdir(drop_dir)) if os.path.isdir(drop_dir) else set()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Stopped watching.")


def main():
    parser = argparse.ArgumentParser(description="Sort dropped files into R1 data folders")
    parser.add_argument("--round", type=int, default=1, help="Round number (default: 1)")
    parser.add_argument("--watch", action="store_true", help="Watch for new files continuously")
    args = parser.parse_args()

    round_dir = get_round_dir(args.round)
    ensure_structure(round_dir)

    print(f"  Round {args.round} data folder: {round_dir}")

    if args.watch:
        watch_drops(round_dir)
    else:
        count = sort_drops(round_dir)
        print(f"\n  Done. Sorted {count} file(s).")

        # Show current state
        print(f"\n  Current R{args.round} data:")
        for sub in ["server_logs", "prices", "trades", "observations",
                    "submissions", "wiki", "intel"]:
            sub_dir = os.path.join(round_dir, sub)
            files = os.listdir(sub_dir) if os.path.isdir(sub_dir) else []
            print(f"    {sub}/: {len(files)} file(s)")


if __name__ == "__main__":
    main()
