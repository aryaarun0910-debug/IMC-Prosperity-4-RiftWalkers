"""
make_variants.py — Generate 6 signal-variant submission files for A/B testing.

Each variant is a full stripped copy of trader.py with one targeted change.
Output: data/r1/variants/v_*.py  (all < 100KB, LF line endings)
Also creates: data/r1/variant_logs/  (drop server logs here after submitting)

Variants:
  v_bp3.py       — penny-jump +3 from book instead of +2 (more exclusive queue)
  v_tighttake.py — take threshold wall_mid-2 instead of wall_mid-1 (more selective)
  v_notake.py    — disable aggressive takes entirely (pure passive MM)
  v_ofi60.py     — OFI thresholds 0.6/0.4 instead of 0.5/0.3 (less reactive)
  v_wmid75.py    — wall_mid weight 75/25 instead of 50/50 (trust dynamic more)
  v_combo.py     — combined: bp+3 + take threshold wall_mid-2

Usage:
    py -3.12 tools/make_variants.py
"""

import os
import sys
import ast

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _TOOLS_DIR)

# Import strip logic from submit.py
from submit import strip_trader, validate_stripped

TRADER_PATH = os.path.join(_ROOT_DIR, "trader.py")
VARIANTS_DIR = os.path.join(_ROOT_DIR, "data", "r1", "variants")
VARIANT_LOGS_DIR = os.path.join(_ROOT_DIR, "data", "r1", "variant_logs")
MAX_SIZE_KB = 100

# ── Exact strings to find/replace (verified against trader.py line numbers) ──
# bp+2 block (lines 1461-1473)
BP2_COND_BID   = "if bv > 1 and bp + 2 < wall_mid:"
BP2_TAKE_BID   = "our_bid = max(our_bid, bp + 2)   # +2 not +1: exclusive queue"
BP2_COND_ASK   = "if av > 1 and ap - 2 > wall_mid:"
BP2_TAKE_ASK   = "our_ask = min(our_ask, ap - 2)   # -2 not -1: exclusive queue"

# take threshold (line 1408)
TAKE_THRESH    = "if ask_p <= wall_mid - 1:"

# OFI block (lines 1481-1489)
OFI_POS_HI     = "if imb > 0.5:"
OFI_POS_LO     = "elif imb > 0.3:"
OFI_NEG_HI     = "elif imb < -0.5:"
OFI_NEG_LO     = "elif imb < -0.3:"

# wall_mid 50/50 blend (line 1374)
WMID_BLEND     = "wall_mid = 0.50 * wall_mid + 0.50 * ps.wall_mid"


def apply_patches(source: str, patches: list[tuple[str, str]]) -> str:
    """Apply multiple find/replace patches. Raises if any find-string is missing."""
    for old, new in patches:
        if old not in source:
            raise ValueError(f"Patch target not found in source:\n  >>> {old!r}")
        source = source.replace(old, new, 1)
    return source


VARIANTS = {
    "v_bp3": {
        "desc": "penny-jump +3 from book (tighter, more exclusive queue position)",
        "patches": [
            (BP2_COND_BID, "if bv > 1 and bp + 3 < wall_mid:"),
            (BP2_TAKE_BID, "our_bid = max(our_bid, bp + 3)   # +3: tighter exclusive queue"),
            (BP2_COND_ASK, "if av > 1 and ap - 3 > wall_mid:"),
            (BP2_TAKE_ASK, "our_ask = min(our_ask, ap - 3)   # -3: tighter exclusive queue"),
        ],
    },
    "v_tighttake": {
        "desc": "aggressive take only when price is >= 2 ticks below FV (wall_mid-2)",
        "patches": [
            (TAKE_THRESH, "if ask_p <= wall_mid - 2:"),
        ],
    },
    "v_notake": {
        "desc": "disable aggressive takes entirely — pure passive MM, no crossing spread",
        "patches": [
            (TAKE_THRESH, "if ask_p <= wall_mid - 50:"),
        ],
    },
    "v_ofi60": {
        "desc": "OFI thresholds 0.6/0.4 (less reactive, only acts on stronger imbalance)",
        "patches": [
            (OFI_POS_HI, "if imb > 0.6:"),
            (OFI_POS_LO, "elif imb > 0.4:"),
            (OFI_NEG_HI, "elif imb < -0.6:"),
            (OFI_NEG_LO, "elif imb < -0.4:"),
        ],
    },
    "v_wmid75": {
        "desc": "wall_mid 75/25 blend (trusts dynamic mid more, less anchored to static FV)",
        "patches": [
            (WMID_BLEND, "wall_mid = 0.75 * wall_mid + 0.25 * ps.wall_mid"),
        ],
    },
    "v_combo": {
        "desc": "combined: bp+3 penny-jump AND take threshold wall_mid-2",
        "patches": [
            (BP2_COND_BID, "if bv > 1 and bp + 3 < wall_mid:"),
            (BP2_TAKE_BID, "our_bid = max(our_bid, bp + 3)   # +3: tighter exclusive queue"),
            (BP2_COND_ASK, "if av > 1 and ap - 3 > wall_mid:"),
            (BP2_TAKE_ASK, "our_ask = min(our_ask, ap - 3)   # -3: tighter exclusive queue"),
            (TAKE_THRESH,  "if ask_p <= wall_mid - 2:"),
        ],
    },
}


def main():
    os.makedirs(VARIANTS_DIR, exist_ok=True)
    os.makedirs(VARIANT_LOGS_DIR, exist_ok=True)
    print(f"Created: {VARIANTS_DIR}")
    print(f"Created: {VARIANT_LOGS_DIR}")
    print()

    with open(TRADER_PATH, 'r', encoding='utf-8') as f:
        base_source = f.read()

    results = []

    for name, spec in VARIANTS.items():
        print(f"--- {name}: {spec['desc']}")
        try:
            patched = apply_patches(base_source, spec["patches"])
        except ValueError as e:
            print(f"  FAILED: {e}")
            results.append((name, False, "patch failed"))
            continue

        # Strip (removes comments/docstrings, compresses whitespace)
        import tempfile, io
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          encoding='utf-8', delete=False,
                                          newline='\n')
        tmp.write(patched)
        tmp.close()
        try:
            stripped = strip_trader(tmp.name)
        except SystemExit:
            print(f"  FAILED: strip_trader exited")
            results.append((name, False, "strip failed"))
            os.unlink(tmp.name)
            continue
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

        size = len(stripped.encode('utf-8'))
        ok, msg = validate_stripped(stripped)

        if not ok:
            print(f"  FAILED: {msg}")
            results.append((name, False, msg))
            continue

        if size > MAX_SIZE_KB * 1024:
            over = size - MAX_SIZE_KB * 1024
            print(f"  WARNING: {size/1024:.1f} KB — OVER limit by {over/1024:.1f} KB")
        else:
            headroom = MAX_SIZE_KB * 1024 - size
            print(f"  Size: {size/1024:.1f} KB ({headroom/1024:.1f} KB headroom)")

        out_path = os.path.join(VARIANTS_DIR, f"{name}.py")
        with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(stripped)
        print(f"  Written: {out_path}")
        results.append((name, True, f"{size/1024:.1f} KB"))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = 0
    for name, ok, info in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:<18} {info}")
        if ok:
            passed += 1

    print()
    print(f"{passed}/{len(VARIANTS)} variants generated successfully.")
    print()
    print("Next steps:")
    print(f"  1. Submit each .py in {VARIANTS_DIR} to the IMC portal")
    print(f"  2. Drop all server logs into:  {VARIANT_LOGS_DIR}")
    print(f"  3. Run:  py -3.12 tools/analyze_variants.py")
    print(f"     (will compare all variant + baseline logs -> final submission)")


if __name__ == "__main__":
    main()
