"""HYD fair_value sweep — test 9990, 9995, 10000, 10005 on R3+R4."""
import sys, os, importlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r3_p = [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0,1,2]]
r3_t = [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0,1,2]]
r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]
src_p = "data/r3/prices/prices_round_3_day_2.csv"
src_t = "data/r3/trades/trades_round_3_day_2.csv"
tmpdir = tempfile.mkdtemp(prefix="hyd_fv_")
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

# Also build R4 1K slice
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

variants = [9990, 9994, 9998, 9999, 10000, 10002, 10005]

print()
print("=" * 100)
print(f"{'FV':>6} | {'R3_30K':>10} {'R3_1K':>10} {'R4_30K':>10} {'R4_1K':>10} | {'R3_HYD30':>10} {'R4_HYD30':>10}")
print("=" * 100)
for fv in variants:
    importlib.reload(bt); bt.tc.CONFIG["HYDROGEL_PACK"]["fair_value"] = fv
    r3_30, r3_30p = bt.run_backtest(r3_p, r3_t)
    importlib.reload(bt); bt.tc.CONFIG["HYDROGEL_PACK"]["fair_value"] = fv
    r3_1k, _ = bt.run_backtest([slice_p], [slice_t])
    importlib.reload(bt); bt.tc.CONFIG["HYDROGEL_PACK"]["fair_value"] = fv
    r4_30, r4_30p = bt.run_backtest(r4_p, r4_t)
    importlib.reload(bt); bt.tc.CONFIG["HYDROGEL_PACK"]["fair_value"] = fv
    r4_1k, _ = bt.run_backtest([slice_p4], [slice_t4])
    print(f"{fv:>6} | {r3_30:>+10.0f} {r3_1k:>+10.0f} {r4_30:>+10.0f} {r4_1k:>+10.0f} | "
          f"{r3_30p.get('HYDROGEL_PACK',0):>+10.0f} {r4_30p.get('HYDROGEL_PACK',0):>+10.0f}", flush=True)
print()
print("Ship: must beat v3 baseline (FV=9990) on all 4 columns.")
