"""Sweep voucher_chain_dir on VEV chain — directional VEF momentum trade.

Hypothesis: top teams ride VEF drift via leveraged voucher chain. 30K backtest
target: per-strike +50K range (vs our skew_mm cap +740). Sweep window x thr."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

# Only test VEV_5000 first (lowest spread, deepest ITM). Sweep window+thr.
strikes_under_test = ["VEV_5000"]

_chain_strikes = [5000, 5100, 5300, 5400]  # Skip 5200 (smile bias short) and 5500 (low signal)

def chain_override(window, thr, size=50):
    """Build override applying same params to all chain strikes."""
    return {f"VEV_{k}": {"dir_window": window, "dir_thr": thr, "dir_target_size": size}
            for k in _chain_strikes}

variants = [
    ("baseline (chain do_nothing)", {}),
    # Tighter thresholds — VEF intra-day drift is ~0.001-0.003/tick
    ("5000 w=50 thr=0.0005", {"VEV_5000": {"dir_window": 50, "dir_thr": 0.0005}}),
    ("5000 w=100 thr=0.0005", {"VEV_5000": {"dir_window": 100, "dir_thr": 0.0005}}),
    ("5000 w=200 thr=0.0001", {"VEV_5000": {"dir_window": 200, "dir_thr": 0.0001}}),
    ("5000 w=200 thr=0.0005", {"VEV_5000": {"dir_window": 200, "dir_thr": 0.0005}}),
    ("5000 w=500 thr=0.0005", {"VEV_5000": {"dir_window": 500, "dir_thr": 0.0005}}),
    # Full chain — competitor's signature shows multi-strike P&L
    ("CHAIN w=200 thr=0.0005 sz=50", chain_override(200, 0.0005, 50)),
    ("CHAIN w=500 thr=0.0005 sz=50", chain_override(500, 0.0005, 50)),
    ("CHAIN w=200 thr=0.001 sz=50",  chain_override(200, 0.001, 50)),
    ("CHAIN w=100 thr=0.0005 sz=50", chain_override(100, 0.0005, 50)),
]

print(f"{'variant':<40} {'TOTAL':>10} {'VEV_5000':>10} {'VEF':>10} {'delta':>10}")
print("-" * 85)
baseline_total = None
for desc, overrides in variants:
    importlib.reload(bt)
    if overrides:
        for prod, params in overrides.items():
            base = {"type": "voucher_chain_dir", "position_limit": 300,
                    "underlying": "VELVETFRUIT_EXTRACT", "strike": int(prod.split("_")[1]),
                    "dir_target_size": 50}
            base.update(params)
            bt.tc.CONFIG[prod] = base
    total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
    v5000 = per_prod.get('VEV_5000', 0)
    vef = per_prod.get('VELVETFRUIT_EXTRACT', 0)
    if baseline_total is None:
        baseline_total = total
    delta = total - baseline_total
    print(f"{desc:<40} {total:>+10.0f} {v5000:>+10.0f} {vef:>+10.0f} {delta:>+10.0f}")
