"""Test markout_sizedown=True on VEF, HYD, and both — next candidate from p4_r3_ship memory.

Hypothesis: markout-driven size reduction filters adverse selection.
Memory says: 'currently OFF, code already exists at trader.py:2919'.
27K alpha gap to competitor on VEF (we 85K, them 109K). Markout sizedown is
the single most plausible remaining lever per the v6 RW-thesis findings."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

variants = [
    ("baseline (v7 shipped)", {}),
    ("VEF markout_sizedown",  {"VELVETFRUIT_EXTRACT": {"markout_sizedown": True}}),
    ("HYD markout_sizedown",  {"HYDROGEL_PACK": {"markout_sizedown": True}}),
    ("BOTH markout_sizedown", {
        "VELVETFRUIT_EXTRACT": {"markout_sizedown": True},
        "HYDROGEL_PACK": {"markout_sizedown": True},
    }),
]

print(f"{'variant':<30} {'TOTAL':>10} {'HYD':>10} {'VEF':>10} {'VEV4000':>8} {'delta':>10}")
print("-" * 85)
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
    v4 = per_prod.get('VEV_4000', 0)
    if baseline_total is None:
        baseline_total = total
    delta = total - baseline_total
    print(f"{desc:<30} {total:>+10.0f} {hyd:>+10.0f} {vef:>+10.0f} {v4:>+8.0f} {delta:>+10.0f}", flush=True)
