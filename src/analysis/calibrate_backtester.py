"""
Backtester Calibration — Match backtester fill behavior to server reality.

Uses server logs as ground truth:
  - Extract price/trade CSVs from server activitiesLog
  - Run backtester at many fill rates
  - Find the fill rate where backtester fills ≈ server fills
  - Report calibrated fill rate + PnL comparison

Usage:
    python calibrate_backtester.py
    python calibrate_backtester.py --server-log data/server_logs/18442.log
"""

import csv
import io
import json
import os
import sys
import tempfile
import numpy as np
from collections import defaultdict

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_ANALYSIS_DIR)
sys.path.insert(0, _ROOT_DIR)
os.chdir(_ROOT_DIR)

import importlib.util
spec = importlib.util.spec_from_file_location('tc', os.path.join(_ROOT_DIR, 'trader.py'))
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def extract_data_from_server_log(log_path):
    """Extract price and trade data from a server log JSON."""
    with open(log_path) as f:
        data = json.load(f)

    # Parse activitiesLog (semicolon-separated CSV)
    activities = data['activitiesLog']
    reader = csv.DictReader(io.StringIO(activities), delimiter=';')
    rows = []
    for row in reader:
        cleaned = {k.strip(): v.strip() for k, v in row.items()}
        rows.append(cleaned)

    # Build price DataFrame-like structure grouped by (day, timestamp)
    price_data = defaultdict(list)
    for row in rows:
        day = int(row['day'])
        ts = int(row['timestamp'])
        price_data[(day, ts)].append(row)

    # Parse tradeHistory — ALL trades including non-SUBMISSION
    trade_map = {}
    for t in data.get('tradeHistory', []):
        day = -1  # tutorial is single day
        ts = int(t['timestamp'])
        sym = t['symbol']
        key = (day, ts, sym)
        if key not in trade_map:
            trade_map[key] = []
        trade_map[key].append({
            "price": int(t["price"]),
            "quantity": int(t["quantity"]),
            "buyer": t.get("buyer", ""),
            "seller": t.get("seller", ""),
        })

    # Count server fills
    our_trades = [t for t in data.get('tradeHistory', [])
                  if t.get('buyer') == 'SUBMISSION' or t.get('seller') == 'SUBMISSION']
    server_fills = len(our_trades)
    server_qty = sum(t['quantity'] for t in our_trades)

    # Server PnL from last activities entry per product
    server_pnl = {}
    for row in rows:
        product = row['product']
        pnl = float(row['profit_and_loss'])
        server_pnl[product] = pnl  # last one wins (final PnL)

    return price_data, trade_map, server_fills, server_qty, server_pnl


def build_state_from_log(ts_rows, day, timestamp, trade_map, positions, trader_data):
    """Build TradingState from server log rows for a single timestamp."""
    order_depths = {}
    for row in ts_rows:
        product = row['product']
        od = tc.OrderDepth()
        for i in range(1, 4):
            bp = row.get(f'bid_price_{i}', '')
            bv = row.get(f'bid_volume_{i}', '')
            ap = row.get(f'ask_price_{i}', '')
            av = row.get(f'ask_volume_{i}', '')
            if bp and bp.strip():
                od.buy_orders[int(float(bp))] = int(float(bv))
            if ap and ap.strip():
                od.sell_orders[int(float(ap))] = int(float(av))
        order_depths[product] = od

    market_trades = {}
    own_trades = {}
    for product in order_depths:
        key = (day, timestamp, product)
        raw = trade_map.get(key, [])
        trades = []
        for t in raw:
            trades.append(tc.Trade(
                symbol=product,
                price=t["price"],
                quantity=t["quantity"],
                buyer=t.get("buyer", ""),
                seller=t.get("seller", ""),
                timestamp=timestamp,
            ))
        market_trades[product] = trades
        own_trades[product] = []

    state = tc.TradingState(
        timestamp=timestamp,
        traderData=trader_data,
        listings={},
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=market_trades,
        position=dict(positions),
        observations=tc.Observation(),
    )
    return state


def match_orders(orders, od, pos, limit, fill_rng, fill_rate, agg_fill_rate=1.0):
    """Match orders against the book with configurable aggressive + passive fill rates.
    agg_fill_rate: probability that a spread-crossing order actually fills (default 1.0 = guaranteed)."""
    fills = []

    total_buy = sum(o.quantity for o in orders if o.quantity > 0)
    total_sell = sum(abs(o.quantity) for o in orders if o.quantity < 0)
    if pos + total_buy > limit or pos - total_sell < -limit:
        return [], pos

    for order in orders:
        if order.quantity > 0:  # BUY
            for ask_p in sorted(od.sell_orders.keys()):
                if order.price >= ask_p:
                    # Aggressive fill: subject to agg_fill_rate
                    if fill_rng.random() >= agg_fill_rate:
                        continue
                    avail = abs(od.sell_orders[ask_p])
                    fill_qty = min(order.quantity, avail, limit - pos)
                    if fill_qty > 0:
                        fills.append((order.symbol, ask_p, fill_qty))
                        pos += fill_qty
                        order = tc.Order(order.symbol, order.price, order.quantity - fill_qty)
                        if order.quantity <= 0:
                            break
            if order.quantity > 0:
                if fill_rng.random() < fill_rate:
                    fill_qty = min(order.quantity, limit - pos)
                    if fill_qty > 0:
                        fills.append((order.symbol, order.price, fill_qty))
                        pos += fill_qty

        elif order.quantity < 0:  # SELL
            qty = abs(order.quantity)
            for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                if order.price <= bid_p:
                    # Aggressive fill: subject to agg_fill_rate
                    if fill_rng.random() >= agg_fill_rate:
                        continue
                    avail = od.buy_orders[bid_p]
                    fill_qty = min(qty, avail, limit + pos)
                    if fill_qty > 0:
                        fills.append((order.symbol, bid_p, -fill_qty))
                        pos -= fill_qty
                        qty -= fill_qty
                        if qty <= 0:
                            break
            if qty > 0:
                if fill_rng.random() < fill_rate:
                    fill_qty = min(qty, limit + pos)
                    if fill_qty > 0:
                        fills.append((order.symbol, order.price, -fill_qty))
                        pos -= fill_qty

    return fills, pos


def run_sim(price_data, trade_map, fill_rate, seed=42, agg_fill_rate=1.0):
    """Run simulation on extracted server data with given fill_rate and agg_fill_rate."""
    trader = tc.Trader()
    positions = defaultdict(int)
    cash = defaultdict(float)
    trader_data = ""
    fill_rng = np.random.RandomState(seed)

    total_fills = 0
    total_orders = 0

    sorted_keys = sorted(price_data.keys())
    products = set()

    for (day, ts) in sorted_keys:
        rows = price_data[(day, ts)]
        for r in rows:
            products.add(r['product'])

        state = build_state_from_log(rows, day, ts, trade_map, positions, trader_data)
        result, conversions, trader_data = trader.run(state)

        for product, orders in result.items():
            if not orders:
                continue
            od = state.order_depths.get(product)
            if od is None:
                continue
            limit = trader._config.get(product, {}).get("position_limit", 50)
            total_orders += len(orders)

            fills, new_pos = match_orders(orders, od, positions[product], limit, fill_rng, fill_rate, agg_fill_rate)
            positions[product] = new_pos

            for sym, price, qty in fills:
                cash[sym] -= price * qty
                total_fills += 1

    # Mark to market
    pnl = {}
    for product in products:
        # Find last mid price
        last_mid = None
        for (day, ts) in reversed(sorted_keys):
            for r in price_data[(day, ts)]:
                if r['product'] == product:
                    bid1 = r.get('bid_price_1', '')
                    ask1 = r.get('ask_price_1', '')
                    if bid1 and ask1:
                        last_mid = (float(bid1) + float(ask1)) / 2.0
                        break
            if last_mid is not None:
                break

        if last_mid is None:
            last_mid = 0
        mtm = positions[product] * last_mid
        pnl[product] = cash[product] + mtm

    total_pnl = sum(pnl.values())
    return total_pnl, total_fills, total_orders, pnl, dict(positions)


def calibrate(log_path, seeds=[42, 123, 456]):
    """Find the fill rate that best matches server reality."""
    print(f"Loading server log: {log_path}")
    price_data, trade_map, server_fills, server_qty, server_pnl = extract_data_from_server_log(log_path)

    total_server_pnl = sum(server_pnl.values())
    print(f"\n--- SERVER GROUND TRUTH ---")
    print(f"  Ticks: {len(price_data)}")
    print(f"  Our fills: {server_fills}")
    print(f"  Total qty: {server_qty}")
    for p, pnl in sorted(server_pnl.items()):
        print(f"  {p}: PnL={pnl:+.1f}")
    print(f"  TOTAL PnL: {total_server_pnl:+.1f}")

    # 2D grid search: passive fill rate x aggressive fill rate
    passive_rates = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12]
    agg_rates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]

    print(f"\n--- 2D FILL RATE GRID SEARCH ---")
    print(f"  Target: fills={server_fills}, PnL={total_server_pnl:+.1f}")
    print(f"{'Passive':>8} {'Aggr':>6} {'AvgPnL':>10} {'AvgFills':>9} {'PnL_err%':>9} {'Fill_err%':>10} {'Score':>8}")
    print(f"{'-' * 65}")

    best_score = float('inf')
    best_combo = (0.015, 0.20)
    all_results = []

    for pfr in passive_rates:
        for afr in agg_rates:
            pnls = []
            fills_list = []
            for seed in seeds:
                total_pnl, total_fills, _, _, _ = run_sim(
                    price_data, trade_map, pfr, seed, agg_fill_rate=afr)
                pnls.append(total_pnl)
                fills_list.append(total_fills)

            avg_pnl = sum(pnls) / len(pnls)
            avg_fills = sum(fills_list) / len(fills_list)

            pnl_err = abs(avg_pnl - total_server_pnl) / total_server_pnl * 100 if total_server_pnl else 0
            fill_err = abs(avg_fills - server_fills) / server_fills * 100 if server_fills else 0
            # Combined score: weight PnL match more (it's what matters)
            score = pnl_err * 2 + fill_err

            all_results.append((pfr, afr, avg_pnl, avg_fills, pnl_err, fill_err, score))

            if score < best_score:
                best_score = score
                best_combo = (pfr, afr)

    # Sort by score and show top 15
    all_results.sort(key=lambda x: x[6])
    for pfr, afr, avg_pnl, avg_fills, pnl_err, fill_err, score in all_results[:15]:
        marker = " <-- BEST" if (pfr, afr) == best_combo else ""
        print(f"{pfr:>8.3f} {afr:>6.2f} {avg_pnl:>+10.1f} {avg_fills:>9.0f} {pnl_err:>8.1f}% {fill_err:>9.1f}% {score:>8.1f}{marker}")

    print(f"\n{'=' * 65}")
    print(f"CALIBRATION RESULTS")
    print(f"{'=' * 65}")
    print(f"  Best combo:  passive_fill_rate = {best_combo[0]:.3f}, agg_fill_rate = {best_combo[1]:.2f}")
    print(f"  Server: fills={server_fills}, PnL={total_server_pnl:+.1f}")

    # Detailed report at best combo
    print(f"\n--- DETAILED: passive={best_combo[0]:.3f}, agg={best_combo[1]:.2f} ---")
    for seed in seeds:
        total_pnl, total_fills, total_orders, pnl_by_prod, positions = run_sim(
            price_data, trade_map, best_combo[0], seed, agg_fill_rate=best_combo[1])
        print(f"  seed={seed}: PnL={total_pnl:+.1f} (ratio={total_pnl/total_server_pnl:.2f}x), "
              f"fills={total_fills}, orders={total_orders}")
        for p in sorted(pnl_by_prod.keys()):
            srv = server_pnl.get(p, 0)
            print(f"    {p}: bt_pnl={pnl_by_prod[p]:+.1f} srv_pnl={srv:+.1f} pos={positions.get(p, 0)}")

    ticks = len(price_data)
    print(f"\n--- DIAGNOSTICS ---")
    print(f"  Ticks: {ticks}")
    print(f"  Server fills/tick: {server_fills/ticks:.3f}")
    print(f"  Calibrated passive rate: {best_combo[0]:.3f}")
    print(f"  Calibrated aggressive rate: {best_combo[1]:.2f}")

    return best_combo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate backtester fill rate")
    parser.add_argument("--server-log", type=str,
                        default="data/server_logs/18442.log",
                        help="Server log to calibrate against")
    args = parser.parse_args()

    calibrate(args.server_log)
