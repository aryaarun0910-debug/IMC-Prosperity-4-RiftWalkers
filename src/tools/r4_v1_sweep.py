"""R4 v1 sweep: relax_mult and max_relax tuning on p4r4 30K data + 1K slice."""
import sys, os, importlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

prices_full = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1, 2, 3]]
trades_full = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1, 2, 3]]

# Build day_2 first 10% slice (mirror of R3 visible slice convention)
src_p = "data/r4/prices/prices_round_4_day_2.csv"
src_t = "data/r4/trades/trades_round_4_day_2.csv"
tmpdir = tempfile.mkdtemp(prefix="r4_v1_")
slice_p = os.path.join(tmpdir, "prices_round_4_day_2.csv")
slice_t = os.path.join(tmpdir, "trades_round_4_day_2.csv")
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
    ("baseline OFF",        False, 0,  0),
    ("v1 mult=4 cap=8",     True,  4,  8),
    ("v1 mult=5 cap=10",    True,  5, 10),
    ("v1 mult=6 cap=12",    True,  6, 12),
    ("v1 mult=8 cap=15",    True,  8, 15),
    ("v1 mult=4 cap=15",    True,  4, 15),
    ("v1 mult=10 cap=20",   True, 10, 20),
]

print()
print("=" * 100)
print(f"{'variant':<22} {'30K_TOTAL':>10} {'30K_VEF':>10} {'30K_dlt':>9}  {'1K_TOTAL':>9} {'1K_VEF':>9} {'1K_dlt':>9}")
print("=" * 100)
base30, base1k = None, None
for label, lean, mult, cap in variants:
    importlib.reload(bt)
    cfg = bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]
    cfg["informed_lean"] = lean
    cfg["informed_relax_mult"] = mult
    cfg["informed_max_relax"] = cap
    total30, per30 = bt.run_backtest(prices_full, trades_full)
    vef30 = per30.get("VELVETFRUIT_EXTRACT", 0)

    importlib.reload(bt)
    cfg = bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]
    cfg["informed_lean"] = lean
    cfg["informed_relax_mult"] = mult
    cfg["informed_max_relax"] = cap
    total1k, per1k = bt.run_backtest([slice_p], [slice_t])
    vef1k = per1k.get("VELVETFRUIT_EXTRACT", 0)

    if base30 is None: base30 = total30
    if base1k is None: base1k = total1k
    d30 = total30 - base30
    d1k = total1k - base1k
    print(f"{label:<22} {total30:>+10.0f} {vef30:>+10.0f} {d30:>+9.0f}  {total1k:>+9.0f} {vef1k:>+9.0f} {d1k:>+9.0f}", flush=True)
print()
print("Goal: large 30K_dlt AND positive 1K_dlt = ship-ready.")
