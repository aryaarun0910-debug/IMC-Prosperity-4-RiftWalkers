"""V3 take-rate-cap sweep on HYD. Test max_take_per_tick on R3 + R4."""
import sys, os, importlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r3_p = [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0,1,2]]
r3_t = [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0,1,2]]
r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]
src_p = "data/r3/prices/prices_round_3_day_2.csv"
src_t = "data/r3/trades/trades_round_3_day_2.csv"
tmpdir = tempfile.mkdtemp(prefix="tcap_")
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
    ("v2 (no cap)",         None, None),
    ("HYD cap=20",            20, None),
    ("HYD cap=30",            30, None),
    ("HYD cap=40",            40, None),
    ("HYD cap=50",            50, None),
    ("HYD cap=80",            80, None),
    ("BOTH cap=30",           30,   30),
    ("BOTH cap=50",           50,   50),
]

print()
print("=" * 90)
print(f"{'variant':<22} {'R3_30K':>10} {'R3_1K':>10} {'R4_30K':>10}")
print("=" * 90)
for label, hyd_cap, vef_cap in variants:
    importlib.reload(bt)
    if hyd_cap is not None:
        bt.tc.CONFIG["HYDROGEL_PACK"]["max_take_per_tick"] = hyd_cap
    if vef_cap is not None:
        bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["max_take_per_tick"] = vef_cap
    r3_30, _ = bt.run_backtest(r3_p, r3_t)

    importlib.reload(bt)
    if hyd_cap is not None:
        bt.tc.CONFIG["HYDROGEL_PACK"]["max_take_per_tick"] = hyd_cap
    if vef_cap is not None:
        bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["max_take_per_tick"] = vef_cap
    r3_1k, _ = bt.run_backtest([slice_p], [slice_t])

    importlib.reload(bt)
    if hyd_cap is not None:
        bt.tc.CONFIG["HYDROGEL_PACK"]["max_take_per_tick"] = hyd_cap
    if vef_cap is not None:
        bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["max_take_per_tick"] = vef_cap
    r4_30, _ = bt.run_backtest(r4_p, r4_t)

    print(f"{label:<22} {r3_30:>+10.0f} {r3_1k:>+10.0f} {r4_30:>+10.0f}", flush=True)
print()
print("Ship: must beat v2 (235740 / 14484 / 224684) on all 3.")
