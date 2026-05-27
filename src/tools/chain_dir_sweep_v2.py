"""Calibrated voucher_chain_dir sweep — thresholds based on actual VEF slope dist.

VEF slope stats (30K data):
  w=200:  median |slope| 0.049, p90 0.115
  w=500:  median 0.029, p90 0.063
  w=1000: median 0.016, p90 0.040
  w=2000: median 0.007, p90 0.022

Strategy is "trend follow": only enter when |slope| > thr (ie above noise).
Set thr at p70-p90 to filter noise."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

_chain_strikes = [5000, 5100, 5300, 5400]

def chain_override(window, thr, size=50, strikes=None):
    s = strikes or _chain_strikes
    return {f"VEV_{k}": {"type": "voucher_chain_dir", "position_limit": 50,
                          "underlying": "VELVETFRUIT_EXTRACT", "strike": k,
                          "dir_window": window, "dir_thr": thr, "dir_target_size": size}
            for k in s}

# Need to also handle VEV_5000 with its 300-pos-limit
def with_5000_limit(override, size_5000=50):
    if "VEV_5000" in override:
        override["VEV_5000"]["position_limit"] = 300
    return override

variants = [
    ("baseline", {}),
    # Calibrated single-strike VEV_5000 — sweep p50 to p90 thr
    ("5000 w=200 thr=0.05",  with_5000_limit(chain_override(200, 0.05, 50, [5000]))),
    ("5000 w=200 thr=0.10",  with_5000_limit(chain_override(200, 0.10, 50, [5000]))),
    ("5000 w=500 thr=0.03",  with_5000_limit(chain_override(500, 0.03, 50, [5000]))),
    ("5000 w=500 thr=0.06",  with_5000_limit(chain_override(500, 0.06, 50, [5000]))),
    ("5000 w=1000 thr=0.02", with_5000_limit(chain_override(1000, 0.02, 50, [5000]))),
    ("5000 w=1000 thr=0.04", with_5000_limit(chain_override(1000, 0.04, 50, [5000]))),
    ("5000 w=2000 thr=0.01", with_5000_limit(chain_override(2000, 0.01, 50, [5000]))),
    # Full chain — 4 strikes, calibrated thr
    ("CHAIN w=500 thr=0.03",  with_5000_limit(chain_override(500, 0.03, 50))),
    ("CHAIN w=1000 thr=0.02", with_5000_limit(chain_override(1000, 0.02, 50))),
    ("CHAIN w=1000 thr=0.04", with_5000_limit(chain_override(1000, 0.04, 50))),
    ("CHAIN w=2000 thr=0.01", with_5000_limit(chain_override(2000, 0.01, 50))),
]

print(f"{'variant':<32} {'TOTAL':>10} {'V5000':>8} {'V5100':>8} {'V5300':>8} {'V5400':>8} {'VEF':>8} {'delta':>10}")
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
    vef = per_prod.get('VELVETFRUIT_EXTRACT', 0)
    if baseline_total is None:
        baseline_total = total
    delta = total - baseline_total
    print(f"{desc:<32} {total:>+10.0f} {v5000:>+8.0f} {v5100:>+8.0f} {v5300:>+8.0f} {v5400:>+8.0f} {vef:>+8.0f} {delta:>+10.0f}", flush=True)
