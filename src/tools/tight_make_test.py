"""Test tighter make_offset on HYD/VEF — only NEW lever in user's MM proposal.

Current make_offset=2 for both. Test {1, 2, 3} to bracket.
v6 RW thesis predicts ±300 noise — but make_edge wasn't explicitly swept.

Hypothesis: tighter = more fills + more pinning = backtest-positive but live-toxic.
This will likely confirm v7's "less aggressive, more survivable" thesis."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

variants = [
    ("baseline (mo=2)", {}),
    ("HYD mo=1",  {"HYDROGEL_PACK": {"make_offset": 1}}),
    ("HYD mo=3",  {"HYDROGEL_PACK": {"make_offset": 3}}),
    ("VEF mo=1",  {"VELVETFRUIT_EXTRACT": {"make_offset": 1}}),
    ("VEF mo=3",  {"VELVETFRUIT_EXTRACT": {"make_offset": 3}}),
    ("BOTH mo=1", {"HYDROGEL_PACK": {"make_offset": 1}, "VELVETFRUIT_EXTRACT": {"make_offset": 1}}),
    ("BOTH mo=3", {"HYDROGEL_PACK": {"make_offset": 3}, "VELVETFRUIT_EXTRACT": {"make_offset": 3}}),
]

print(f"{'variant':<22} {'TOTAL':>10} {'HYD':>10} {'VEF':>10} {'delta':>10}")
print("-" * 70)
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
    print(f"{desc:<22} {total:>+10.0f} {hyd:>+10.0f} {vef:>+10.0f} {delta:>+10.0f}", flush=True)
