"""Test enabling voucher_intrinsic_mm on the WHOLE VEV chain (not just 4000).

Hypothesis: competitor's +83K per VEV strike comes from passive intrinsic-MM
with large position limit, not from directional taking.

VEV_4000 already at voucher_intrinsic_mm = +9.4K backtest. Try same on 5000-5400."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

def intrinsic_cfg(strike, pos_limit=300, make_size=15, make_edge=10, take_edge=8):
    return {"type": "voucher_intrinsic_mm", "position_limit": pos_limit,
            "underlying": "VELVETFRUIT_EXTRACT", "strike": strike,
            "make_size": make_size, "make_edge": make_edge, "take_edge": take_edge,
            "max_pos_frac": 0.6, "inv_skew_ticks": 4}

variants = [
    ("baseline", {}),
    ("VEV_5000 intrinsic_mm pl=300", {"VEV_5000": intrinsic_cfg(5000)}),
    ("VEV_5000+5100 intrinsic_mm", {"VEV_5000": intrinsic_cfg(5000), "VEV_5100": intrinsic_cfg(5100, pos_limit=50)}),
    ("CHAIN 5000-5400 intrinsic_mm", {
        "VEV_5000": intrinsic_cfg(5000),
        "VEV_5100": intrinsic_cfg(5100, pos_limit=50),
        "VEV_5300": intrinsic_cfg(5300, pos_limit=50),
        "VEV_5400": intrinsic_cfg(5400, pos_limit=50),
    }),
    ("CHAIN 5000-5500 intrinsic_mm", {
        "VEV_5000": intrinsic_cfg(5000),
        "VEV_5100": intrinsic_cfg(5100, pos_limit=50),
        "VEV_5300": intrinsic_cfg(5300, pos_limit=50),
        "VEV_5400": intrinsic_cfg(5400, pos_limit=50),
        "VEV_5500": intrinsic_cfg(5500, pos_limit=50),
    }),
    # Try tighter MM (smaller make_edge)
    ("CHAIN tight make_edge=3", {
        "VEV_5000": intrinsic_cfg(5000, make_edge=3),
        "VEV_5100": intrinsic_cfg(5100, pos_limit=50, make_edge=3),
        "VEV_5300": intrinsic_cfg(5300, pos_limit=50, make_edge=3),
        "VEV_5400": intrinsic_cfg(5400, pos_limit=50, make_edge=3),
    }),
]

print(f"{'variant':<35} {'TOTAL':>10} {'V5000':>8} {'V5100':>8} {'V5300':>8} {'V5400':>8} {'V5500':>8} {'delta':>10}")
print("-" * 105)
baseline_total = None
for desc, overrides in variants:
    importlib.reload(bt)
    if overrides:
        for prod, params in overrides.items():
            bt.tc.CONFIG[prod] = params
    total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
    v5000 = per_prod.get('VEV_5000', 0)
    v5100 = per_prod.get('VEV_5100', 0)
    v5300 = per_prod.get('VEV_5300', 0)
    v5400 = per_prod.get('VEV_5400', 0)
    v5500 = per_prod.get('VEV_5500', 0)
    if baseline_total is None:
        baseline_total = total
    delta = total - baseline_total
    print(f"{desc:<35} {total:>+10.0f} {v5000:>+8.0f} {v5100:>+8.0f} {v5300:>+8.0f} {v5400:>+8.0f} {v5500:>+8.0f} {delta:>+10.0f}", flush=True)
