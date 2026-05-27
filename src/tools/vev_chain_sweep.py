"""Sweep VEV_4500 + VEV_5000 voucher_intrinsic_mm parameters on 30K p4r3 backtest."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

# Restore baseline VEV_4000 = me=10, te=8
def reset_vev_4000():
    bt.tc.CONFIG['VEV_4000'].update({"make_edge": 10, "take_edge": 8, "make_size": 15})

variants = [
    ("baseline (4500/5000 do_nothing)", "baseline"),
    # VEV_4500 enable
    ("VEV_4500 me=8, te=6", "vim_4500", {"make_edge": 8, "take_edge": 6, "make_size": 15}),
    ("VEV_4500 me=5, te=4", "vim_4500", {"make_edge": 5, "take_edge": 4, "make_size": 15}),
    ("VEV_4500 me=10, te=8 (mirror 4000)", "vim_4500", {"make_edge": 10, "take_edge": 8, "make_size": 15}),
    ("VEV_4500 me=6, te=5, ms=10", "vim_4500", {"make_edge": 6, "take_edge": 5, "make_size": 10}),
    # VEV_5000 enable
    ("VEV_5000 me=3, te=3", "vim_5000", {"make_edge": 3, "take_edge": 3, "make_size": 10}),
    ("VEV_5000 me=2, te=2", "vim_5000", {"make_edge": 2, "take_edge": 2, "make_size": 10}),
    ("VEV_5000 me=5, te=4", "vim_5000", {"make_edge": 5, "take_edge": 4, "make_size": 10}),
    # Combined best (placeholder — fill after individual)
]

print(f"{'variant':<45} {'TOTAL':>10} {'V4500':>10} {'V5000':>10} {'delta':>10}")
print("-" * 90)
baseline_total = None
for spec in variants:
    desc = spec[0]
    importlib.reload(bt)
    reset_vev_4000()
    if spec[1] == "vim_4500":
        params = spec[2]
        bt.tc.CONFIG['VEV_4500'] = {
            "type": "voucher_intrinsic_mm", "position_limit": 300,
            "underlying": "VELVETFRUIT_EXTRACT", "strike": 4500,
            "max_pos_frac": 0.6, "inv_skew_ticks": 4,
            **params
        }
    elif spec[1] == "vim_5000":
        params = spec[2]
        bt.tc.CONFIG['VEV_5000'] = {
            "type": "voucher_intrinsic_mm", "position_limit": 300,
            "underlying": "VELVETFRUIT_EXTRACT", "strike": 5000,
            "max_pos_frac": 0.6, "inv_skew_ticks": 4,
            **params
        }
    total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
    v4500 = per_prod.get('VEV_4500', 0)
    v5000 = per_prod.get('VEV_5000', 0)
    if baseline_total is None:
        baseline_total = total
    delta = total - baseline_total
    print(f"{desc:<45} {total:>+10.0f} {v4500:>+10.0f} {v5000:>+10.0f} {delta:>+10.0f}")
