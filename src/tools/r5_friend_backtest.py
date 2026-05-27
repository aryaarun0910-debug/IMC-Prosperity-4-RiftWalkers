"""Test all 7 of the friend's algos against our 3-day backtest data (days 2/3/4).

Critical question: do the algos work consistently across days, or are they overfit to one day?
If overfit, days other than the calibration day will show negative or massively reduced PnL.
"""
import sys, os, importlib.util, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r5_backtest import load_prices, build_state, simulate_fills

BASE = "C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/Round 5 - Algo side final check"

def find_algo_pys():
    """Returns dict of version -> py path."""
    out = {}
    for d in sorted(os.listdir(BASE)):
        sub = os.path.join(BASE, d)
        if not os.path.isdir(sub): continue
        pys = glob.glob(os.path.join(sub, "*", "*.py"))
        if pys:
            out[d] = pys[0]
    return out

def load_trader(py_path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trader()

def run_backtest_quiet(trader, prices):
    from collections import defaultdict
    timestamps = sorted(prices.keys())
    position = {}
    cash_pnl = defaultdict(float)
    traderData = ""
    for ts in timestamps:
        books = prices[ts]
        state = build_state(ts, books, position, traderData)
        try:
            result, conv, traderData = trader.run(state)
        except Exception as e:
            print(f"  ERROR at ts={ts}: {type(e).__name__}: {e}")
            return None
        filled, position, pnl_d = simulate_fills(result, books, position)
        for k, v in pnl_d.items():
            cash_pnl[k] += v
    last_books = prices[timestamps[-1]]
    final_pnl = {}
    for prod, p in position.items():
        rec = last_books.get(prod)
        if not rec: continue
        final_pnl[prod] = cash_pnl[prod] + p * rec["mid"]
    for prod in cash_pnl:
        if prod not in final_pnl:
            final_pnl[prod] = cash_pnl[prod]
    return sum(final_pnl.values()), final_pnl

def main():
    algos = find_algo_pys()
    print(f"Found {len(algos)} algos to test\n")

    # Pre-load all 3 days
    days_data = {}
    for d in [2,3,4]:
        path = f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv"
        days_data[d] = load_prices(path)

    print(f"{'Algo':<22} {'Day 2':>10} {'Day 3':>10} {'Day 4':>10} {'TOTAL':>12} {'Min day':>10}")
    print("-" * 85)

    for algo_name, py_path in algos.items():
        try:
            trader = load_trader(py_path, f"alg_{algo_name.replace(' ', '_')}")
        except Exception as e:
            print(f"{algo_name:<22}  LOAD ERROR: {type(e).__name__}: {e}")
            continue
        per_day = {}
        for d in [2,3,4]:
            try:
                trader = load_trader(py_path, f"alg_{algo_name.replace(' ', '_')}_d{d}")
                pnl, _ = run_backtest_quiet(trader, days_data[d])
                per_day[d] = pnl
            except Exception as e:
                per_day[d] = None
                print(f"  {algo_name} day {d} error: {e}")
        if all(per_day[d] is not None for d in [2,3,4]):
            total = sum(per_day.values())
            min_day = min(per_day.values())
            print(f"{algo_name:<22} {per_day[2]:>+10.0f} {per_day[3]:>+10.0f} {per_day[4]:>+10.0f} {total:>+12.0f} {min_day:>+10.0f}")
        else:
            print(f"{algo_name:<22}  PARTIAL FAILURE")

if __name__ == "__main__":
    main()
