"""Sweep slope_bias_thr + slope_bias_target on HYD/VEF on 30K p4r3 backtest.

Hypothesis (v8.6): top-team 50K-PnL chart shows directional position-trading on HYD/VEF.
Active slope_bias crosses spread when 50-tick OLS slope > thr to reach target_pos.
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

# (desc, product_overrides_dict)
variants = [
    ("baseline (slope_bias OFF)", {}),
    # HYD only, thr sweep at target=60
    ("HYD thr=0.05, t=60", {"HYDROGEL_PACK": {"slope_bias_thr": 0.05, "slope_bias_target": 60}}),
    ("HYD thr=0.1, t=60",  {"HYDROGEL_PACK": {"slope_bias_thr": 0.10, "slope_bias_target": 60}}),
    ("HYD thr=0.2, t=60",  {"HYDROGEL_PACK": {"slope_bias_thr": 0.20, "slope_bias_target": 60}}),
    # HYD target sweep at thr=0.1
    ("HYD thr=0.1, t=30",  {"HYDROGEL_PACK": {"slope_bias_thr": 0.10, "slope_bias_target": 30}}),
    ("HYD thr=0.1, t=100", {"HYDROGEL_PACK": {"slope_bias_thr": 0.10, "slope_bias_target": 100}}),
    ("HYD thr=0.1, t=150", {"HYDROGEL_PACK": {"slope_bias_thr": 0.10, "slope_bias_target": 150}}),
    # VEF only
    ("VEF thr=0.1, t=60",  {"VELVETFRUIT_EXTRACT": {"slope_bias_thr": 0.10, "slope_bias_target": 60}}),
    ("VEF thr=0.05, t=100",{"VELVETFRUIT_EXTRACT": {"slope_bias_thr": 0.05, "slope_bias_target": 100}}),
    # combined
    ("HYD+VEF thr=0.1, t=60", {
        "HYDROGEL_PACK": {"slope_bias_thr": 0.10, "slope_bias_target": 60},
        "VELVETFRUIT_EXTRACT": {"slope_bias_thr": 0.10, "slope_bias_target": 60},
    }),
    ("HYD+VEF thr=0.1, t=100", {
        "HYDROGEL_PACK": {"slope_bias_thr": 0.10, "slope_bias_target": 100},
        "VELVETFRUIT_EXTRACT": {"slope_bias_thr": 0.10, "slope_bias_target": 100},
    }),
]

print(f"{'variant':<40} {'TOTAL':>10} {'HYD':>10} {'VEF':>10} {'Final HYD':>10} {'Final VEF':>10} {'delta':>10}")
print("-" * 105)
baseline_total = None
for desc, overrides in variants:
    importlib.reload(bt)
    for prod, params in overrides.items():
        for k, v in params.items():
            bt.tc.CONFIG[prod][k] = v
    total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
    hyd = per_prod.get('HYDROGEL_PACK', 0)
    vef = per_prod.get('VELVETFRUIT_EXTRACT', 0)
    # also report final positions (already printed by backtest, but capture)
    if baseline_total is None:
        baseline_total = total
    delta = total - baseline_total
    print(f"{desc:<40} {total:>+10.0f} {hyd:>+10.0f} {vef:>+10.0f} {'-':>10} {'-':>10} {delta:>+10.0f}")
