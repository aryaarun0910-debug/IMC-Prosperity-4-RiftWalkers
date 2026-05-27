"""Sweep HYD slope_defensive_thr and slope_defensive_inv_gate.
Find sweet spot that protects R4 live without regressing R3 backtest.
"""
import sys, os, importlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r3_p = [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0,1,2]]
r3_t = [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0,1,2]]
r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]

# 1K slice (R3 day 2 first 10%)
src_p = "data/r3/prices/prices_round_3_day_2.csv"
src_t = "data/r3/trades/trades_round_3_day_2.csv"
tmpdir = tempfile.mkdtemp(prefix="path_a_")
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

variants = [
    ("v8 baseline (thr 0.2 gate 160)",  0.2, 160),
    ("v2a thr 0.1 gate 100",            0.1, 100),
    ("v2b thr 0.1 gate 130",            0.1, 130),
    ("v2c thr 0.15 gate 120",          0.15, 120),
    ("v2d thr 0.15 gate 100",          0.15, 100),
    ("v2e thr 0.2 gate 100",            0.2, 100),
    ("v2f thr 0.12 gate 110",          0.12, 110),
]

print()
print("=" * 110)
print(f"{'variant':<32} {'R3_30K':>8} {'R3_1K':>8} {'R4_30K':>8} {'R3_HYD30':>9} {'R3_HYD1K':>9} {'R4_HYD30':>9}")
print("=" * 110)
for label, thr, gate in variants:
    importlib.reload(bt)
    bt.tc.CONFIG["HYDROGEL_PACK"]["slope_defensive_thr"] = thr
    bt.tc.CONFIG["HYDROGEL_PACK"]["slope_defensive_inv_gate"] = gate
    r3_30, r3_30_per = bt.run_backtest(r3_p, r3_t)
    importlib.reload(bt)
    bt.tc.CONFIG["HYDROGEL_PACK"]["slope_defensive_thr"] = thr
    bt.tc.CONFIG["HYDROGEL_PACK"]["slope_defensive_inv_gate"] = gate
    r3_1k, r3_1k_per = bt.run_backtest([slice_p], [slice_t])
    importlib.reload(bt)
    bt.tc.CONFIG["HYDROGEL_PACK"]["slope_defensive_thr"] = thr
    bt.tc.CONFIG["HYDROGEL_PACK"]["slope_defensive_inv_gate"] = gate
    r4_30, r4_30_per = bt.run_backtest(r4_p, r4_t)
    print(f"{label:<32} {r3_30:>+8.0f} {r3_1k:>+8.0f} {r4_30:>+8.0f} "
          f"{r3_30_per.get('HYDROGEL_PACK',0):>+9.0f} {r3_1k_per.get('HYDROGEL_PACK',0):>+9.0f} "
          f"{r4_30_per.get('HYDROGEL_PACK',0):>+9.0f}", flush=True)
print()
print("Goal: maintain R3_30K >= 233K AND R3_1K >= 12.9K AND maximize R4_30K.")
