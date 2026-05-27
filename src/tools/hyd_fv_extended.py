"""Extended HYD FV sweep — test 9994 to 10015."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r3_p = [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0,1,2]]
r3_t = [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0,1,2]]
r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]

print()
print("=" * 60)
print(f"{'FV':>6} {'R3_30K':>10} {'R4_30K':>10} {'R3_HYD':>10} {'R4_HYD':>10}")
print("=" * 60)
for fv in [9994, 10000, 10005, 10008, 10010, 10012, 10015]:
    importlib.reload(bt)
    bt.tc.CONFIG['HYDROGEL_PACK']['fair_value'] = fv
    r3, r3p = bt.run_backtest(r3_p, r3_t)
    importlib.reload(bt)
    bt.tc.CONFIG['HYDROGEL_PACK']['fair_value'] = fv
    r4, r4p = bt.run_backtest(r4_p, r4_t)
    print(f"{fv:>6} {r3:>+10.0f} {r4:>+10.0f} {r3p.get('HYDROGEL_PACK',0):>+10.0f} {r4p.get('HYDROGEL_PACK',0):>+10.0f}", flush=True)
print()
