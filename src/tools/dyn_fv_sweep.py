"""Sweep dyn_fv_alpha + static_anchor on HYD+VEF for v8 dynamic-FV restructure."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest as bt
from glob import glob

# Round paths (p4r3)
price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))
print(f"Loading {len(price_csvs)} price CSVs, {len(trade_csvs)} trade CSVs")

# Sweep grid
sweep = [
    (0.0,  0.0,  "baseline (legacy static fv)"),
    (0.005, 0.3, "slow EMA + 0.3 static"),
    (0.01,  0.3, "med EMA + 0.3 static"),
    (0.02,  0.3, "fast EMA + 0.3 static"),
    (0.01,  0.0, "med EMA, no static"),
    (0.01,  0.5, "med EMA + 0.5 static"),
    (0.005, 0.0, "slow EMA, no static"),
    (0.005, 0.5, "slow EMA + 0.5 static"),
]

print(f"\n{'alpha':>6} {'anchor':>6}  {'TOTAL':>10}  {'HYD':>10} {'pos':>5}  {'VEF':>10} {'pos':>5}  desc")
for alpha, anchor, desc in sweep:
    # Reload trader so CONFIG dict is fresh
    if 'trader' in sys.modules:
        importlib.reload(sys.modules['trader'])
    importlib.reload(bt)
    # Patch via Trader instantiation hack: monkey-patch the CONFIG reference inside the module
    bt.tc.CONFIG['HYDROGEL_PACK']['dyn_fv_alpha'] = alpha
    bt.tc.CONFIG['HYDROGEL_PACK']['static_anchor'] = anchor
    bt.tc.CONFIG['VELVETFRUIT_EXTRACT']['dyn_fv_alpha'] = alpha
    bt.tc.CONFIG['VELVETFRUIT_EXTRACT']['static_anchor'] = anchor

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
    out = buf.getvalue()
    # Parse final positions from output
    hyd_pnl = per_prod.get('HYDROGEL_PACK', 0)
    vef_pnl = per_prod.get('VELVETFRUIT_EXTRACT', 0)
    hyd_pos = vef_pos = 0
    for line in out.split('\n'):
        if 'HYDROGEL_PACK' in line and '+' in line:
            parts = line.split()
            try:
                hyd_pos = int(parts[2])
            except Exception:
                pass
        if 'VELVETFRUIT_EXTRACT' in line and '+' in line:
            parts = line.split()
            try:
                vef_pos = int(parts[2])
            except Exception:
                pass
    print(f"{alpha:>6.3f} {anchor:>6.2f}  {total:>+10.0f}  {hyd_pnl:>+10.0f} {hyd_pos:>5}  {vef_pnl:>+10.0f} {vef_pos:>5}  {desc}")
