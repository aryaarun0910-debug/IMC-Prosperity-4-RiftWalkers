"""Test friend's suggestion: enable VEV_4500 voucher_intrinsic_mm.

Memory + line 875 of trader.py both flag VEV_4500 as having ~1 trade in 30K ticks
of p4r3 data. Hypothesis: zero counterparties = zero PnL. Test definitively.
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

VEV_4500_CFG = {
    "type": "voucher_intrinsic_mm", "position_limit": 300,
    "underlying": "VELVETFRUIT_EXTRACT", "strike": 4500,
    "make_size": 15, "make_edge": 10, "take_edge": 8,
    "max_pos_frac": 0.6, "inv_skew_ticks": 4,
}

variants = [
    ("baseline (v7 shipped)", {}),
    ("VEV_4500 enabled (friend rec)", {"VEV_4500": VEV_4500_CFG}),
]

print(f"{'variant':<35} {'TOTAL':>10} {'V4500':>10} {'V4000':>10} {'VEF':>10} {'HYD':>10} {'delta':>10}")
print("-" * 100)
baseline_total = None
for desc, overrides in variants:
    importlib.reload(bt)
    if overrides:
        for prod, patches in overrides.items():
            bt.tc.CONFIG[prod] = patches
    total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
    v4500 = per_prod.get('VEV_4500', 0)
    v4000 = per_prod.get('VEV_4000', 0)
    vef = per_prod.get('VELVETFRUIT_EXTRACT', 0)
    hyd = per_prod.get('HYDROGEL_PACK', 0)
    if baseline_total is None:
        baseline_total = total
    delta = total - baseline_total
    print(f"{desc:<35} {total:>+10.0f} {v4500:>+10.0f} {v4000:>+10.0f} {vef:>+10.0f} {hyd:>+10.0f} {delta:>+10.0f}", flush=True)
