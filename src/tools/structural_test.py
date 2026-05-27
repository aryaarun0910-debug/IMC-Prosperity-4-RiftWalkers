"""Quick test: full 30K + Day 2 first 10% slice for current trader.py state.

Reports both numbers compactly for ship-gate evaluation."""
import sys, os, importlib, csv, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

src_prices = 'data/r3/prices/prices_round_3_day_2.csv'
src_trades = 'data/r3/trades/trades_round_3_day_2.csv'
tmpdir = tempfile.mkdtemp(prefix='struct_test_')
slice_prices = os.path.join(tmpdir, 'prices_round_3_day_2.csv')
slice_trades = os.path.join(tmpdir, 'trades_round_3_day_2.csv')
MAX_TS = 99900

with open(src_prices) as f, open(slice_prices, 'w', newline='') as o:
    o.write(f.readline())
    for line in f:
        parts = line.split(';')
        if len(parts) >= 2 and int(parts[1]) <= MAX_TS:
            o.write(line)
with open(src_trades) as f, open(slice_trades, 'w', newline='') as o:
    o.write(f.readline())
    for line in f:
        parts = line.split(';')
        if len(parts) >= 2:
            try:
                if int(parts[0]) <= MAX_TS:
                    o.write(line)
            except (ValueError, IndexError):
                continue

full_prices = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
full_trades = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

importlib.reload(bt)
total30, per30 = bt.run_backtest(full_prices, full_trades)

importlib.reload(bt)
total1k, per1k = bt.run_backtest([slice_prices], [slice_trades])

print()
print("=" * 80)
print(f"30K_TOTAL: {total30:>+10.0f}  (gate: >= +233,000)")
print(f"  HYD: {per30.get('HYDROGEL_PACK', 0):>+10.0f}")
print(f"  VEF: {per30.get('VELVETFRUIT_EXTRACT', 0):>+10.0f}")
print(f"  V40: {per30.get('VEV_4000', 0):>+10.0f}")
print()
print(f"1K_TOTAL:  {total1k:>+10.0f}  (gate: >= +12,891)")
print(f"  HYD: {per1k.get('HYDROGEL_PACK', 0):>+10.0f}")
print(f"  VEF: {per1k.get('VELVETFRUIT_EXTRACT', 0):>+10.0f}")
print(f"  V40: {per1k.get('VEV_4000', 0):>+10.0f}")
print("=" * 80)
