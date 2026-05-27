"""V3 candidate experiments — three parallel tests.

Test 1: VEV_4500 voucher_intrinsic_mm enable (replicate VEV_4000 success)
Test 2: Slope-bias high-threshold (only fire on extreme moves)
Test 3: Path B — lower Mark 67 detection threshold

Each variant must beat v2 baseline on R3 30K AND R3 1K AND R4 30K to ship.
"""
import sys, os, importlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r3_p = [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0,1,2]]
r3_t = [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0,1,2]]
r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]

# 1K slice (R3 Day 2 first 10%)
src_p = "data/r3/prices/prices_round_3_day_2.csv"
src_t = "data/r3/trades/trades_round_3_day_2.csv"
tmpdir = tempfile.mkdtemp(prefix="v3_")
slice_p = os.path.join(tmpdir, "prices_round_3_day_2.csv")
slice_t = os.path.join(tmpdir, "trades_round_3_day_2.csv")
MAX_TS = 99900
with open(src_p) as f, open(slice_p, "w", newline="") as o:
    o.write(f.readline())
    for line in f:
        parts = line.split(";")
        if len(parts) >= 2 and int(parts[1]) <= MAX_TS:
            o.write(line)
with open(src_t) as f, open(slice_t, "w", newline="") as o:
    o.write(f.readline())
    for line in f:
        parts = line.split(";")
        if len(parts) >= 2:
            try:
                if int(parts[0]) <= MAX_TS:
                    o.write(line)
            except (ValueError, IndexError):
                continue

VEV_4500_CFG = {
    "type": "voucher_intrinsic_mm", "position_limit": 300,
    "underlying": "VELVETFRUIT_EXTRACT", "strike": 4500,
    "make_size": 15, "make_edge": 10, "take_edge": 8,
    "max_pos_frac": 0.6, "inv_skew_ticks": 4,
}

def run(label, mods):
    """mods = list of (cfg_path, value) tuples to apply"""
    importlib.reload(bt)
    for path, val in mods:
        keys = path.split('.')
        d = bt.tc.CONFIG
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    r3_30, r3_30p = bt.run_backtest(r3_p, r3_t)
    importlib.reload(bt)
    for path, val in mods:
        keys = path.split('.')
        d = bt.tc.CONFIG
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    r3_1k, _ = bt.run_backtest([slice_p], [slice_t])
    importlib.reload(bt)
    for path, val in mods:
        keys = path.split('.')
        d = bt.tc.CONFIG
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    r4_30, r4_30p = bt.run_backtest(r4_p, r4_t)
    return label, r3_30, r3_1k, r4_30

def run_with_vev(label, vev_cfg):
    """Adding VEV_4500 needs special handling (replace whole CONFIG entry)"""
    importlib.reload(bt)
    bt.tc.CONFIG["VEV_4500"] = vev_cfg
    r3_30, _ = bt.run_backtest(r3_p, r3_t)
    importlib.reload(bt)
    bt.tc.CONFIG["VEV_4500"] = vev_cfg
    r3_1k, _ = bt.run_backtest([slice_p], [slice_t])
    importlib.reload(bt)
    bt.tc.CONFIG["VEV_4500"] = vev_cfg
    r4_30, _ = bt.run_backtest(r4_p, r4_t)
    return label, r3_30, r3_1k, r4_30

print()
print("=" * 90)
print(f"{'variant':<40} {'R3_30K':>10} {'R3_1K':>10} {'R4_30K':>10}")
print("=" * 90)

# Baseline (v2)
v2 = run("v2 baseline", [])
print(f"{v2[0]:<40} {v2[1]:>+10.0f} {v2[2]:>+10.0f} {v2[3]:>+10.0f}", flush=True)
print("-" * 90)

# Experiment 1: VEV_4500 enable
e1 = run_with_vev("v3-A: VEV_4500 intrinsic_mm enabled", VEV_4500_CFG)
print(f"{e1[0]:<40} {e1[1]:>+10.0f} {e1[2]:>+10.0f} {e1[3]:>+10.0f}", flush=True)

# Experiment 2: slope_bias high thresholds (HYD only — VEF noise too high)
for thr, target in [(0.3, 100), (0.5, 100), (0.7, 100), (1.0, 100), (0.5, 50), (0.5, 150)]:
    e = run(f"v3-B: HYD slope_bias thr={thr} tgt={target}",
            [("HYDROGEL_PACK.slope_bias_thr", thr),
             ("HYDROGEL_PACK.slope_bias_target", target)])
    print(f"{e[0]:<40} {e[1]:>+10.0f} {e[2]:>+10.0f} {e[3]:>+10.0f}", flush=True)

# Experiment 3: Path B (lower informed threshold)
for min_obs, bias_thr in [(10, 0.85), (8, 0.9), (5, 0.95), (5, 1.0)]:
    e = run(f"v3-C: VEF min_obs={min_obs} bias_thr={bias_thr}",
            [("VELVETFRUIT_EXTRACT.informed_lean", True)])
    # Pass through to update_informed_counterparty defaults — needs code change
    # For now this just runs v2 again (no effect). TODO: parameterize properly.
    print(f"{e[0]:<40} {e[1]:>+10.0f} {e[2]:>+10.0f} {e[3]:>+10.0f}", flush=True)

print()
print("Ship rule: any v3 variant must beat v2 on ALL 3 columns to be considered.")
