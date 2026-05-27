"""Quick: 30K backtest with HYD te=10, VEF te=8 vs current te=15/5."""
import sys, os, importlib, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

price_csvs = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
trade_csvs = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

# baseline
total, per_prod = bt.run_backtest(price_csvs, trade_csvs)
print(f"baseline (HYD te=15, VEF te=5): TOTAL={total:+.0f}  HYD={per_prod['HYDROGEL_PACK']:+.0f}  VEF={per_prod['VELVETFRUIT_EXTRACT']:+.0f}")

# te=10/8
importlib.reload(bt)
bt.tc.CONFIG['HYDROGEL_PACK']['take_edge_pegged'] = 10
bt.tc.CONFIG['VELVETFRUIT_EXTRACT']['take_edge_pegged'] = 8
total2, per_prod2 = bt.run_backtest(price_csvs, trade_csvs)
print(f"te=10/8: TOTAL={total2:+.0f}  HYD={per_prod2['HYDROGEL_PACK']:+.0f}  VEF={per_prod2['VELVETFRUIT_EXTRACT']:+.0f}")
print(f"DELTA: {total2-total:+.0f}")
