"""Test slope_bias activation on HYD/VEF — directional momentum entry.

Code already wired at trader.py:2627. Comment claims '50K-parallel chart confirmed
top traders ride directional drift on HYD/VEF'. Switch never flipped.

Hypothesis A (code comment): RIGHT — drift is real, slope-bias adds 10-30K
Hypothesis B (v6 RW thesis): WRONG — HYD/VEF are RW, slope-following = buy tops
                              sell bottoms = systematic loser (like voucher_chain_dir)

This test will resolve A vs B definitively on the 30K backtest."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

def sb_cfg(thr, target, window=50):
    return {"slope_bias_thr": thr, "slope_bias_target": target, "slope_bias_window": window}

variants = [
    ("baseline (slope OFF)", {}),
    ("HYD thr=0.1 t=30 w=50",  {"HYDROGEL_PACK": sb_cfg(0.1, 30)}),
    ("HYD thr=0.2 t=50 w=50",  {"HYDROGEL_PACK": sb_cfg(0.2, 50)}),
    ("HYD thr=0.1 t=50 w=100", {"HYDROGEL_PACK": sb_cfg(0.1, 50, 100)}),
    ("VEF thr=0.05 t=30 w=50", {"VELVETFRUIT_EXTRACT": sb_cfg(0.05, 30)}),
    ("VEF thr=0.1 t=50 w=50",  {"VELVETFRUIT_EXTRACT": sb_cfg(0.1, 50)}),
    ("VEF thr=0.05 t=50 w=100",{"VELVETFRUIT_EXTRACT": sb_cfg(0.05, 50, 100)}),
    ("BOTH thr=0.1 t=30",      {"HYDROGEL_PACK": sb_cfg(0.1, 30), "VELVETFRUIT_EXTRACT": sb_cfg(0.05, 30)}),
    ("BOTH thr=0.1 t=50",      {"HYDROGEL_PACK": sb_cfg(0.1, 50), "VELVETFRUIT_EXTRACT": sb_cfg(0.05, 50)}),
]

print(f"{'variant':<32} {'TOTAL':>10} {'HYD':>10} {'VEF':>10} {'delta':>10}")
print("-" * 80)
baseline_total = None
for desc, overrides in variants:
    importlib.reload(bt)
    if overrides:
        for prod, patches in overrides.items():
            existing = bt.tc.CONFIG.get(prod, {})
            merged = {**existing, **patches}
            bt.tc.CONFIG[prod] = merged
    total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
    hyd = per_prod.get('HYDROGEL_PACK', 0)
    vef = per_prod.get('VELVETFRUIT_EXTRACT', 0)
    if baseline_total is None:
        baseline_total = total
    delta = total - baseline_total
    print(f"{desc:<32} {total:>+10.0f} {hyd:>+10.0f} {vef:>+10.0f} {delta:>+10.0f}", flush=True)
