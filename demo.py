"""Runnable demonstration of the Round 5 mean-reversion engine.

The real competition data is not included in this repository (it is large and
gitignored). This demo generates a synthetic, mean-reverting price series with
the same order-book schema the engine expects, runs the actual `trader_r5`
engine against it tick by tick, simulates fills, and reports PnL.

It exists to show the engine runs end to end and that the mean-reversion logic
behaves as designed on a series it is built to trade. Run:

    pip install -r requirements.txt   # (numpy only, for the synthetic generator)
    python demo.py
"""
import os
import sys
import random

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "engine")
sys.path.insert(0, ENGINE)

import datamodel as dm
import trader_r5 as tr5


def synth_mean_reverting_series(n_ticks, anchor, half_life, vol, seed):
    """Ornstein-Uhlenbeck-style discrete mean-reverting mid prices around `anchor`."""
    rng = random.Random(seed)
    theta = 1.0 - 0.5 ** (1.0 / half_life)   # reversion speed from half-life
    mids = []
    x = anchor
    for _ in range(n_ticks):
        x += theta * (anchor - x) + rng.gauss(0, vol)
        mids.append(round(x * 2) / 2.0)       # half-tick grid, like the real data
    return mids


def build_book(product, mid, spread=2, depth=30):
    od = dm.OrderDepth()
    bb, ba = mid - spread / 2, mid + spread / 2
    od.buy_orders = {int(bb): depth, int(bb) - 1: depth * 2}
    od.sell_orders = {int(ba): -depth, int(ba) + 1: -depth * 2}
    return od


def make_state(ts, books, position, traderData):
    s = dm.TradingState.__new__(dm.TradingState)
    s.traderData = traderData
    s.timestamp = ts
    s.listings = {}
    s.order_depths = books
    s.own_trades = {}
    s.market_trades = {}
    s.position = position
    s.observations = None
    return s


def simulate_fills(orders_by_prod, books, position, pos_limit=10):
    new_pos = dict(position)
    cash = {}
    for prod, orders in orders_by_prod.items():
        if prod not in books:
            continue
        bids = sorted(books[prod].buy_orders.items(), key=lambda kv: -kv[0])
        asks = sorted(((p, -v) for p, v in books[prod].sell_orders.items()), key=lambda kv: kv[0])
        cur = new_pos.get(prod, 0)
        c = 0.0
        for o in orders:
            if o.quantity > 0:
                room = pos_limit - cur
                fill = min(o.quantity, room)
                for ap, av in asks:
                    if o.price < ap or fill <= 0:
                        break
                    take = min(fill, av)
                    cur += take
                    c -= take * ap
                    fill -= take
            elif o.quantity < 0:
                room = pos_limit + cur
                fill = min(-o.quantity, room)
                for bp, bv in bids:
                    if o.price > bp or fill <= 0:
                        break
                    take = min(fill, bv)
                    cur -= take
                    c += take * bp
                    fill -= take
        new_pos[prod] = cur
        cash[prod] = cash.get(prod, 0.0) + c
    return new_pos, cash


def run():
    # Products the R5 engine actively mean-reverts, with synthetic series tuned
    # to oscillate (the regime the strategy is designed for).
    products = {
        "PEBBLES_XL":             dict(anchor=10000, half_life=120, vol=18, seed=1),
        "MICROCHIP_SQUARE":       dict(anchor=10000, half_life=90,  vol=15, seed=2),
        "TRANSLATOR_VOID_BLUE":   dict(anchor=10000, half_life=110, vol=12, seed=3),
        "TRANSLATOR_ASTRO_BLACK": dict(anchor=10000, half_life=100, vol=14, seed=4),
        "PEBBLES_S":              dict(anchor=10000, half_life=130, vol=16, seed=5),
    }
    N = 3000
    series = {p: synth_mean_reverting_series(N, **cfg) for p, cfg in products.items()}

    trader = tr5.Trader()
    position = {}
    cash_total = {p: 0.0 for p in products}
    traderData = ""

    for t in range(N):
        books = {p: build_book(p, series[p][t]) for p in products}
        state = make_state(t * 100, books, position, traderData)
        result, _, traderData = trader.run(state)
        position, cash = simulate_fills(result, books, position)
        for p, c in cash.items():
            cash_total[p] += c

    # Mark to market at the final mid
    print(f"{'Product':<26} {'PnL':>10}  {'end_pos':>7}")
    print("-" * 48)
    total = 0.0
    for p in products:
        mtm = position.get(p, 0) * series[p][-1]
        pnl = cash_total[p] + mtm
        total += pnl
        print(f"{p:<26} {pnl:>+10.0f}  {position.get(p, 0):>+7d}")
    print("-" * 48)
    print(f"{'TOTAL (synthetic, ' + str(N) + ' ticks)':<26} {total:>+10.0f}")
    print()
    print("Note: synthetic data for demonstration only. Real competition results")
    print("are in the writeup (700th / 18,803 teams). The point here is that the")
    print("engine runs end to end and the mean-reversion logic profits on the")
    print("oscillating regime it is designed for.")
    return total


if __name__ == "__main__":
    run()
