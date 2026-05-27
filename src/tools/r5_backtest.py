"""Minimal R5 backtester. Loads price CSVs, constructs TradingState per tick, simulates fills,
tracks per-product P&L. Aimed at fast iteration of trader_r5.py.

Fill model: any aggressive order at or beyond best bid/ask gets filled (simple — assumes infinite
counterparty). No queue. Position-limit enforced.
"""
import sys, os, math, json, importlib.util
from collections import defaultdict, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load datamodel + trader_r5
import datamodel as dm
spec = importlib.util.spec_from_file_location('trader_r5', 'trader_r5.py')
tr5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr5)


def load_prices(path):
    """Returns OrderedDict[ts -> {product -> dict(bid_px, bid_vol, ask_px, ask_vol, mid)}].

    Each product's book has up to 3 bid/ask levels.
    """
    out = OrderedDict()
    with open(path, "r") as f:
        h = f.readline().strip().split(";")
        idx = {c: i for i, c in enumerate(h)}
        for ln in f:
            p = ln.strip().split(";")
            if len(p) < len(h): continue
            try:
                ts = int(p[idx["timestamp"]])
                prod = p[idx["product"]]
                rec = {"bids": [], "asks": [], "mid": float(p[idx["mid_price"]])}
                for k in [1, 2, 3]:
                    bp = p[idx[f"bid_price_{k}"]]
                    bv = p[idx[f"bid_volume_{k}"]]
                    ap = p[idx[f"ask_price_{k}"]]
                    av = p[idx[f"ask_volume_{k}"]]
                    if bp and bv:
                        rec["bids"].append((int(bp), int(bv)))
                    if ap and av:
                        rec["asks"].append((int(ap), int(av)))
                out.setdefault(ts, {})[prod] = rec
            except Exception:
                pass
    return out


def build_state(ts, books, position, traderData):
    od_map = {}
    for prod, rec in books.items():
        od = dm.OrderDepth()
        od.buy_orders = {p: v for p, v in rec["bids"]}
        od.sell_orders = {p: -v for p, v in rec["asks"]}
        od_map[prod] = od
    state = dm.TradingState.__new__(dm.TradingState)
    state.traderData = traderData
    state.timestamp = ts
    state.listings = {}
    state.order_depths = od_map
    state.own_trades = {}
    state.market_trades = {}
    state.position = position
    state.observations = None
    return state


def simulate_fills(orders_by_prod, books, position, pos_limit=10):
    """Returns (filled_orders_by_prod, new_position, realized_pnl_delta).
    Fill model: any buy >= best ask up to volume; any sell <= best bid up to volume.
    """
    filled = defaultdict(list)
    new_pos = dict(position)
    pnl_delta = defaultdict(float)
    for prod, orders in orders_by_prod.items():
        if prod not in books: continue
        rec = books[prod]
        bids = sorted(rec["bids"], key=lambda x: -x[0])  # high to low
        asks = sorted(rec["asks"], key=lambda x:  x[0])  # low to high
        cur = new_pos.get(prod, 0)
        for o in orders:
            qty = o.quantity
            if qty == 0: continue
            if qty > 0:
                # buy: fill against asks
                remaining = qty
                # also enforce position limit
                room = pos_limit - cur
                fill_qty = min(remaining, room)
                if fill_qty <= 0: continue
                vol_filled = 0
                cost = 0
                for ask_p, ask_v in asks:
                    if o.price < ask_p: break
                    take = min(fill_qty - vol_filled, ask_v)
                    if take <= 0: break
                    vol_filled += take
                    cost += take * ask_p
                    if vol_filled >= fill_qty: break
                if vol_filled > 0:
                    cur += vol_filled
                    pnl_delta[prod] -= cost
                    filled[prod].append((o.price, vol_filled))
            else:
                # sell: fill against bids
                remaining = -qty
                room = pos_limit + cur
                fill_qty = min(remaining, room)
                if fill_qty <= 0: continue
                vol_filled = 0
                proceeds = 0
                for bid_p, bid_v in bids:
                    if o.price > bid_p: break
                    take = min(fill_qty - vol_filled, bid_v)
                    if take <= 0: break
                    vol_filled += take
                    proceeds += take * bid_p
                    if vol_filled >= fill_qty: break
                if vol_filled > 0:
                    cur -= vol_filled
                    pnl_delta[prod] += proceeds
                    filled[prod].append((o.price, -vol_filled))
        new_pos[prod] = cur
    return filled, new_pos, pnl_delta


def run_backtest(price_path, label):
    print(f"\n{'='*100}\nBACKTEST: {label}\n{'='*100}")
    prices = load_prices(price_path)
    timestamps = sorted(prices.keys())

    trader = tr5.Trader()
    position = {}
    cash_pnl = defaultdict(float)  # cash flow per product
    fill_counts = defaultdict(int)
    traderData = ""

    for i, ts in enumerate(timestamps):
        books = prices[ts]
        state = build_state(ts, books, position, traderData)
        result, conversions, traderData = trader.run(state)
        filled, position, pnl_d = simulate_fills(result, books, position)
        for k, v in pnl_d.items():
            cash_pnl[k] += v
        for k, fs in filled.items():
            fill_counts[k] += len(fs)

    # Mark to market at last tick
    last_books = prices[timestamps[-1]]
    final_pnl = {}
    for prod, p in position.items():
        rec = last_books.get(prod)
        if not rec: continue
        mid = rec["mid"]
        mtm = p * mid
        final_pnl[prod] = cash_pnl[prod] + mtm

    # Add zero-position products too
    for prod in cash_pnl:
        if prod not in final_pnl:
            final_pnl[prod] = cash_pnl[prod]

    total = sum(final_pnl.values())
    print(f"\nTOTAL: {total:+.0f}  (across {len(timestamps)} ticks)")
    print(f"\nPer-product PnL:")
    for prod, p in sorted(final_pnl.items(), key=lambda kv: -kv[1]):
        pos = position.get(prod, 0)
        fc = fill_counts.get(prod, 0)
        print(f"  {prod:<32} pnl={p:+8.0f}  end_pos={pos:+3d}  fills={fc}")
    return total


if __name__ == "__main__":
    totals = {}
    for d in [2, 3, 4]:
        path = f"data/r5/prices/prices_round_5_day_{d}.csv"
        totals[d] = run_backtest(path, f"R5 Day {d}")
    print(f"\n{'='*100}\nSUMMARY\n{'='*100}")
    for d, t in totals.items():
        print(f"  Day {d}: {t:+.0f}")
    print(f"  3-day total: {sum(totals.values()):+.0f}")
