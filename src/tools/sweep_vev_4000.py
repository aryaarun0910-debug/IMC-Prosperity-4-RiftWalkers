"""Param sweep for VEV_4000 voucher_intrinsic_mm.

Sweeps make_size x make_edge x take_edge x inv_skew_ticks and ranks by total PnL.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADER = os.path.join(ROOT, "trader.py")

CONFIG_BLOCK = (
    '"VEV_4000": {{\n'
    '        "type": "voucher_intrinsic_mm", "position_limit": 300,\n'
    '        "underlying": "VELVETFRUIT_EXTRACT", "strike": 4000,\n'
    '        "make_size": {ms}, "make_edge": {me}, "take_edge": {te},\n'
    '        "max_pos_frac": 0.6, "inv_skew_ticks": {sk},\n'
    '    }},'
)

PATTERN = re.compile(
    r'"VEV_4000": \{\s*"type": "voucher_intrinsic_mm", "position_limit": 300,\s*'
    r'"underlying": "VELVETFRUIT_EXTRACT", "strike": 4000,\s*'
    r'"make_size": \d+, "make_edge": \d+, "take_edge": \d+,\s*'
    r'"max_pos_frac": [\d.]+, "inv_skew_ticks": \d+,\s*\},'
)


def patch_config(make_size, make_edge, take_edge, skew):
    with open(TRADER, "r", encoding="utf-8") as f:
        src = f.read()
    new_block = CONFIG_BLOCK.format(ms=make_size, me=make_edge, te=take_edge, sk=skew)
    new_src, n = PATTERN.subn(new_block, src)
    if n != 1:
        print(f"PATCH FAILED: matched {n} blocks")
        sys.exit(1)
    with open(TRADER, "w", encoding="utf-8") as f:
        f.write(new_src)


def run_backtest():
    p = subprocess.run(
        ["py", "-3.12", "backtest.py", "--round", "p4r3"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    out = p.stdout
    m = re.search(r"TOTAL\s+\+(\d+)", out)
    total = int(m.group(1)) if m else None
    m = re.search(r"VEV_4000\s+([+-]\d+)", out)
    vev = int(m.group(1)) if m else None
    m = re.search(r"VEV_4000\s+[+-]?\d+\s+[+-]?\d+\s+[+-]?\d+\s+(\d+)\s+(\d+)", out)
    takes, makes = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    return total, vev, takes, makes


def main():
    grid = []
    # Coarse sweep around current best (15, 10, 8, 4)
    for ms in [10, 15, 20]:
        for me in [8, 10, 12]:
            for te in [6, 8, 10]:
                grid.append((ms, me, te, 4))
    # Skew variations at best edge/size
    for sk in [2, 6, 8]:
        grid.append((15, 10, 8, sk))

    # Save results to file as we go (avoid stdout buffering loss)
    out_path = os.path.join(ROOT, "tools", "sweep_vev_4000_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Running {len(grid)} configs...\n")

    print(f"Running {len(grid)} configs...")
    print(f"{'ms':>3} {'me':>3} {'te':>3} {'sk':>3}  {'total':>7}  {'vev':>7}  {'tk':>4}  {'mk':>4}")
    print("-" * 60)

    results = []
    for (ms, me, te, sk) in grid:
        patch_config(ms, me, te, sk)
        t, v, tk, mk = run_backtest()
        results.append((ms, me, te, sk, t, v, tk, mk))
        line = (f"{ms:>3} {me:>3} {te:>3} {sk:>3}  {t or 0:>7}  {v or 0:>+7}  "
                f"{tk or 0:>4}  {mk or 0:>4}")
        with open(out_path, "a") as f:
            f.write(line + "\n")

    with open(out_path, "a") as f:
        f.write("\n=== TOP 10 by TOTAL ===\n")
        for r in sorted(results, key=lambda x: -(x[4] or 0))[:10]:
            f.write(f"  ms={r[0]} me={r[1]} te={r[2]} sk={r[3]}: total={r[4]} vev={r[5]:+}\n")


if __name__ == "__main__":
    main()
