"""R4 v1 A/B test: informed_lean ON vs OFF on p4r4 data.

Reports per-product PnL delta to isolate the informed-lean alpha.
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

prices = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1, 2, 3]]
trades = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1, 2, 3]]

variants = [
    ("baseline (lean OFF)", False),
    ("v1 informed_lean ON", True),
]

results = []
for label, lean in variants:
    importlib.reload(bt)
    bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["informed_lean"] = lean
    total, per = bt.run_backtest(prices, trades)
    results.append((label, total, per))

print()
print("=" * 90)
print(f"{'variant':<28} {'TOTAL':>10} {'HYD':>10} {'VEF':>10} {'VEV_4000':>10}  {'delta':>10}")
print("=" * 90)
base = results[0][1]
for label, total, per in results:
    hyd = per.get("HYDROGEL_PACK", 0)
    vef = per.get("VELVETFRUIT_EXTRACT", 0)
    v40 = per.get("VEV_4000", 0)
    delta = total - base
    print(f"{label:<28} {total:>+10.0f} {hyd:>+10.0f} {vef:>+10.0f} {v40:>+10.0f}  {delta:>+10.0f}")
print()
print("Goal: 'v1' delta vs baseline >> 0 to ship.")
