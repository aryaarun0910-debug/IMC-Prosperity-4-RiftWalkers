"""Sweep VEV_4000 voucher_intrinsic_mm parameters on 30K p4r3 backtest."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

variants = [
    ("baseline (me=10, te=8, make_side=T)", {}),
    ("me=5, te=8", {"make_edge": 5}),
    ("me=7, te=8", {"make_edge": 7}),
    ("me=15, te=8", {"make_edge": 15}),
    ("me=10, te=4", {"take_edge": 4}),
    ("me=10, te=12", {"take_edge": 12}),
    ("make_side=False (take only)", {"make_side": False}),
    ("me=5, make_size=5", {"make_edge": 5, "make_size": 5}),
    ("me=15, make_size=30", {"make_edge": 15, "make_size": 30}),
]

print(f"{'variant':<40} {'TOTAL':>10} {'VEV_4000':>10} {'delta_vs_baseline':>18}")
print("-" * 80)
baseline_total = None
baseline_v4k = None
for desc, overrides in variants:
    importlib.reload(bt)
    for k, v in overrides.items():
        bt.tc.CONFIG['VEV_4000'][k] = v
    total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
    v4k = per_prod.get('VEV_4000', 0)
    if baseline_total is None:
        baseline_total = total
        baseline_v4k = v4k
    print(f"{desc:<40} {total:>+10.0f} {v4k:>+10.0f} {total-baseline_total:>+15.0f}")
