"""R3-only trimmer.

Produces a trader.py variant with unused strategy functions removed so the
stripped submission fits under 100KB. Only pegged, generic_mm and do_nothing
dispatch paths are reachable from the R3 CONFIG.

Usage:
  py -3.12 tools/r3_trim.py                  # writes trader_r3_src.py
  py -3.12 tools/submit.py --source trader_r3_src.py --output trader_submit.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "trader.py")
DST = os.path.join(ROOT, "trader_r3_src.py")

# Exact function names (module-level `def NAME(`) to delete.
# v4 (2026-04-24): skew_mm + delta-hedge agg added. Preserve BS helpers +
# _smile_theo + run_skew_mm + run_skew_delta_agg. run_options still deleted
# (self-ref EMA produces 0 fills on calibrated smile — replaced by skew_mm).
DELETE_FUNCS = [
    "run_ar_olivia",
    "run_olivia_follow",
    "run_pairs_arb",
    "run_wide_spread",
    "run_basket_arb",
    "_discover_basket_components",
    "run_conversion_arb",
    "run_options",
    "_implied_vol",
    "_auto_fit_smile",
    "_get_wall_mid",
    "fit_svi",
    "svi_total_var",
]


def find_def_ranges(src, names):
    """Return list of (start_line, end_line_exclusive) for top-level defs matching names."""
    lines = src.splitlines(keepends=True)
    ranges = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]", line)
        if m and m.group(2) in names:
            # Find end: next top-level def/class (not indented).
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if re.match(r"^(def|class)\s+", nxt) or re.match(r"^[A-Za-z_]", nxt) and "=" in nxt.split("\n", 1)[0] and not nxt.startswith(" "):
                    break
                j += 1
            ranges.append((i, j))
            i = j
        else:
            i += 1
    return ranges


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        src = f.read()

    ranges = find_def_ranges(src, set(DELETE_FUNCS))
    if not ranges:
        print("No matching functions found — aborting.")
        sys.exit(1)

    lines = src.splitlines(keepends=True)
    keep = [True] * len(lines)
    total_deleted = 0
    for (s, e) in ranges:
        for k in range(s, e):
            keep[k] = False
        total_deleted += (e - s)
        fname = re.match(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", lines[s]).group(1)
        print(f"  cut {fname}: lines {s+1}..{e} ({e - s} lines)")

    trimmed = "".join(l for l, k in zip(lines, keep) if k)

    # Replace dispatcher elif branches for removed strategy types with no-op.
    # v2 (options re-cut 2026-04-24): "options" branch back to no-op.
    dead_types = ["ar_olivia", "olivia_follow", "wide_spread", "pairs_arb",
                  "basket_arb", "conversion_arb", "options"]

    patterns = [
        (r'elif ptype == "ar_olivia":\s*\n\s*orders, mem = run_ar_olivia\([^\n]*\n',
         'elif ptype == "ar_olivia":\n                    orders = []\n'),
        (r'elif ptype == "olivia_follow":\s*\n\s*orders, mem = run_olivia_follow\([^\n]*\n',
         'elif ptype == "olivia_follow":\n                    orders = []\n'),
        (r'elif ptype == "wide_spread":\s*\n\s*orders, mem = run_wide_spread\([^\n]*\n',
         'elif ptype == "wide_spread":\n                    orders = []\n'),
        (r'elif ptype == "pairs_arb":\s*\n\s*orders, mem, hedge_orders = run_pairs_arb\([^\n]*\n',
         'elif ptype == "pairs_arb":\n                    orders = []\n'),
        (r'elif ptype == "basket_arb":\s*\n\s*orders, mem, hedge_orders = run_basket_arb\([^\n]*\n',
         'elif ptype == "basket_arb":\n                    orders = []\n'),
        (r'elif ptype == "conversion_arb":\s*\n\s*orders, mem, conv = run_conversion_arb\([^\n]*\n\s*total_conversions \+= conv\s*\n',
         'elif ptype == "conversion_arb":\n                    orders = []\n'),
        (r'elif ptype == "options":\s*\n\s*orders, mem, hedge_orders = run_options\([^\n]*\n',
         'elif ptype == "options":\n                    orders = []\n'),
    ]

    for pat, rep in patterns:
        new_trimmed, n = re.subn(pat, rep, trimmed)
        if n == 0:
            print(f"  WARN: dispatcher pattern not matched: {pat[:60]}...")
        trimmed = new_trimmed

    # IMC submission rule: `import os` is forbidden. Strip the import and
    # neutralise every os.* call site so the runtime no-ops on the server.
    os_subs = [
        (r"^import os\n", ""),
        (r'os\.environ\.get\("IMC_TRADE_LOG_PATH"\)', "None"),
        (r"os\.path\.join\(os\.path\.dirname\(__file__\) or \".\",\s*"
         r"\"analysis\", \"fingerprints\", \"olivia_cascade\.json\"\)",
         '""'),
        (r"os\.path\.join\(os\.path\.dirname\(__file__\) or \".\",\s*"
         r"\"analysis\", \"fingerprints\", \"signal_stack\.json\"\)",
         '""'),
        (r"os\.path\.exists\(_cascade_path\)", "False"),
        (r"os\.path\.exists\(_stack_path\)", "False"),
    ]
    for pat, rep in os_subs:
        new_trimmed, n = re.subn(pat, rep, trimmed, flags=re.MULTILINE)
        if n == 0:
            print(f"  WARN: os-strip pattern not matched: {pat[:60]}...")
        trimmed = new_trimmed

    # Sanity: no surviving `import os` or bare `os.` references.
    if re.search(r"^import\s+os\b", trimmed, flags=re.MULTILINE):
        print("  ERROR: `import os` still present after strip — aborting.")
        sys.exit(2)
    leftover_os = re.findall(r"\bos\.\w+", trimmed)
    if leftover_os:
        print(f"  ERROR: leftover os.* refs: {set(leftover_os)} — aborting.")
        sys.exit(2)

    with open(DST, "w", encoding="utf-8", newline="\n") as f:
        f.write(trimmed)

    orig_size = os.path.getsize(SRC)
    new_size = os.path.getsize(DST)
    print(f"\nOriginal: {orig_size:,} bytes ({orig_size/1024:.1f} KB)")
    print(f"Trimmed:  {new_size:,} bytes ({new_size/1024:.1f} KB)")
    print(f"Saved:    {orig_size - new_size:,} bytes ({(orig_size - new_size)/1024:.1f} KB)")
    print(f"Deleted:  {total_deleted} lines across {len(ranges)} defs")
    print(f"\nOutput: {DST}")


if __name__ == "__main__":
    main()
