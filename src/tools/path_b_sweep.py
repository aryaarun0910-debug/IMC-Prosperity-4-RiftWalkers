"""Path B sweep: lower informed-counterparty detection threshold to fire Mark 67
sooner on the 1K visible slice. Strict gates: must beat v2 on R3+R4."""
import sys, os, importlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r3_p = [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0,1,2]]
r3_t = [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0,1,2]]
r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]
src_p = "data/r3/prices/prices_round_3_day_2.csv"
src_t = "data/r3/trades/trades_round_3_day_2.csv"
tmpdir = tempfile.mkdtemp(prefix="pb_")
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

# Also build R4 1K slice (Day 2 first 10%)
src_p4 = "data/r4/prices/prices_round_4_day_2.csv"
src_t4 = "data/r4/trades/trades_round_4_day_2.csv"
slice_p4 = os.path.join(tmpdir, "prices_round_4_day_2.csv")
slice_t4 = os.path.join(tmpdir, "trades_round_4_day_2.csv")
with open(src_p4) as f, open(slice_p4, "w", newline="") as o:
    o.write(f.readline())
    for line in f:
        parts = line.split(";")
        if len(parts) >= 2 and int(parts[1]) <= MAX_TS:
            o.write(line)
with open(src_t4) as f, open(slice_t4, "w", newline="") as o:
    o.write(f.readline())
    for line in f:
        parts = line.split(";")
        if len(parts) >= 2:
            try:
                if int(parts[0]) <= MAX_TS:
                    o.write(line)
            except (ValueError, IndexError):
                continue

def setvef(min_obs, bias_thr):
    bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["informed_min_obs"] = min_obs
    bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["informed_bias_thr"] = bias_thr

variants = [
    ("v2 baseline (20, 0.7)", 20, 0.7),
    ("Path B-1: (10, 0.85)",  10, 0.85),
    ("Path B-2: (8, 0.9)",     8, 0.9),
    ("Path B-3: (5, 0.95)",    5, 0.95),
    ("Path B-4: (5, 1.0)",     5, 1.0),
    ("Path B-5: (3, 1.0)",     3, 1.0),
    ("Path B-6: (12, 0.9)",   12, 0.9),
]

print()
print("=" * 100)
print(f"{'variant':<32} {'R3_30K':>8} {'R3_1K':>8} {'R4_30K':>8} {'R4_1K':>8}")
print("=" * 100)
for label, min_obs, bias_thr in variants:
    importlib.reload(bt); setvef(min_obs, bias_thr)
    r3_30, _ = bt.run_backtest(r3_p, r3_t)
    importlib.reload(bt); setvef(min_obs, bias_thr)
    r3_1k, _ = bt.run_backtest([slice_p], [slice_t])
    importlib.reload(bt); setvef(min_obs, bias_thr)
    r4_30, _ = bt.run_backtest(r4_p, r4_t)
    importlib.reload(bt); setvef(min_obs, bias_thr)
    r4_1k, _ = bt.run_backtest([slice_p4], [slice_t4])
    print(f"{label:<32} {r3_30:>+8.0f} {r3_1k:>+8.0f} {r4_30:>+8.0f} {r4_1k:>+8.0f}", flush=True)
print()
print("Ship rule: variant must beat v2 on ALL 4 columns.")
