"""Debug: did informed_lean signal actually fire in the backtest?"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

prices = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1, 2, 3]]
trades = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1, 2, 3]]

importlib.reload(bt)
bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["informed_lean"] = True
total, per = bt.run_backtest(prices, trades)
trader = bt._last_trader
mem = trader._mem
print()
print(f'TOTAL: {total}')
print(f'_dbg_inf_calls: {mem.get("_dbg_inf_calls", 0)}')
print(f'_dbg_inf_fires: {mem.get("_dbg_inf_fires", 0)}')
print(f'last signal:    {mem.get("_inf_sig_VELVETFRUIT_EXTRACT", "NONE")}')
cp = mem.get("_cp_VELVETFRUIT_EXTRACT", {"marks": {}})
print(f'\nDetected marks ({len(cp["marks"])}):')
for name, s in cp["marks"].items():
    nb, ns = s["nb"], s["ns"]
    fb_avg = s["fb_sum"]/max(s["fb_n"], 1)
    fs_avg = s["fs_sum"]/max(s["fs_n"], 1)
    print(f'  {name:12s}  nb={nb:4d}  ns={ns:4d}  fb_n={s["fb_n"]:4d}  fb_avg={fb_avg:+.2f}  fs_n={s["fs_n"]:4d}  fs_avg={fs_avg:+.2f}')
