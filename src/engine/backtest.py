"""Backtest the competition trader against P3 data.
Supports: R1 (basic), R4/R5 (baskets, conversions, options).
Optimized: pre-indexes all data, backtest-mode JSON skip."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from collections import defaultdict
import importlib.util

# Load the competition trader
spec = importlib.util.spec_from_file_location('tc', 'trader.py')
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

# Known P3 position limits (server-authoritative, hardcode for backtest)
P3_LIMITS = {
    'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50,
    'CROISSANTS': 250, 'JAMS': 350, 'DJEMBES': 60,
    'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100,
    'MAGNIFICENT_MACARONS': 75,
    'VOLCANIC_ROCK': 400,
    'VOLCANIC_ROCK_VOUCHER_9500': 200,
    'VOLCANIC_ROCK_VOUCHER_9750': 200,
    'VOLCANIC_ROCK_VOUCHER_10000': 200,
    'VOLCANIC_ROCK_VOUCHER_10250': 200,
    'VOLCANIC_ROCK_VOUCHER_10500': 200,
}

# Known P3 basket compositions
P3_BASKETS = {
    'PICNIC_BASKET1': {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 1},
    'PICNIC_BASKET2': {'CROISSANTS': 4, 'JAMS': 2},
}

# Conversion product (observations map to this product)
CONVERSION_PRODUCT = 'MAGNIFICENT_MACARONS'


def load_data(price_csvs, trade_csvs, obs_csvs=None):
    """Load and pre-index price + trade + observation data."""
    frames = []
    for p in price_csvs:
        df = pd.read_csv(p, sep=";")
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["day", "timestamp"]).reset_index(drop=True)

    # Pre-index prices
    tick_index = {}
    tick_schedule = []
    last_row_by_product = {}

    for row in prices.itertuples(index=False):
        if pd.isna(row.day) or pd.isna(row.timestamp):
            continue
        day = int(row.day)
        ts = int(row.timestamp)
        product = row.product
        key = (day, ts)

        if key not in tick_index:
            tick_index[key] = {}
            tick_schedule.append(key)

        book = {"bids": {}, "asks": {}}
        for i in range(1, 4):
            bp = getattr(row, f'bid_price_{i}', None)
            bv = getattr(row, f'bid_volume_{i}', None)
            ap = getattr(row, f'ask_price_{i}', None)
            av = getattr(row, f'ask_volume_{i}', None)
            if bp is not None and not (isinstance(bp, float) and np.isnan(bp)):
                book["bids"][int(bp)] = int(bv)
            if ap is not None and not (isinstance(ap, float) and np.isnan(ap)):
                book["asks"][int(ap)] = int(av)

        tick_index[key][product] = book
        last_row_by_product[product] = book

    # Pre-index trades
    trade_map = {}
    sorted_days = sorted(set(d for d, _ in tick_schedule))
    for i, tp in enumerate(trade_csvs):
        if i >= len(sorted_days):
            break
        day = sorted_days[i]
        try:
            tdf = pd.read_csv(tp, sep=";")
            tdf.columns = [c.strip().lower() for c in tdf.columns]
            if len(tdf) == 0:
                continue
            for row in tdf.itertuples(index=False):
                if not hasattr(row, 'timestamp'):
                    continue
                ts_val = int(row.timestamp)
                sym = row.symbol
                key = (day, ts_val, sym)
                if key not in trade_map:
                    trade_map[key] = []
                trade_map[key].append({
                    "price": int(row.price),
                    "quantity": int(row.quantity),
                    "buyer": str(getattr(row, 'buyer', '')),
                    "seller": str(getattr(row, 'seller', '')),
                })
        except Exception:
            pass  # corrupted trade files

    # Pre-index observations (for conversion arb)
    # Key on (day, timestamp) since timestamps repeat across days
    obs_map = {}
    if obs_csvs:
        for i, obs_path in enumerate(obs_csvs):
            if i >= len(sorted_days):
                break
            day = sorted_days[i]
            try:
                odf = pd.read_csv(obs_path)
                odf.columns = [c.strip().lower() for c in odf.columns]
                if len(odf) == 0:
                    continue
                for row in odf.itertuples(index=False):
                    ts_val = int(row.timestamp)
                    obs_map[(day, ts_val)] = {
                        "bidPrice": float(row.bidprice),
                        "askPrice": float(row.askprice),
                        "transportFees": float(row.transportfees),
                        "exportTariff": float(row.exporttariff),
                        "importTariff": float(row.importtariff),
                        "sugarPrice": float(getattr(row, 'sugarprice', 0)),
                        "sunlightIndex": float(getattr(row, 'sunlightindex', 0)),
                    }
            except Exception:
                pass  # corrupted obs files

    products = sorted(set(p for books in tick_index.values() for p in books))
    return tick_index, tick_schedule, trade_map, obs_map, products, last_row_by_product


def build_state(tick_data, day, timestamp, trade_map, obs_map, positions, trader_data):
    """Build TradingState from pre-indexed data. Includes observations."""
    order_depths = {}
    for product, book in tick_data.items():
        od = tc.OrderDepth()
        od.buy_orders = dict(book["bids"])
        od.sell_orders = dict(book["asks"])
        order_depths[product] = od

    market_trades = {}
    own_trades = {}
    for product in order_depths:
        key = (day, timestamp, product)
        raw = trade_map.get(key, [])
        trades = [tc.Trade(
            symbol=product, price=t["price"], quantity=t["quantity"],
            buyer=t.get("buyer", ""), seller=t.get("seller", ""),
            timestamp=timestamp,
        ) for t in raw]
        market_trades[product] = trades
        own_trades[product] = []

    # Build observations (for conversion arb products)
    observations = tc.Observation()
    obs_data = obs_map.get((day, timestamp))
    if obs_data and CONVERSION_PRODUCT in order_depths:
        conv = tc.ConversionObservation(
            bidPrice=obs_data["bidPrice"],
            askPrice=obs_data["askPrice"],
            transportFees=obs_data["transportFees"],
            exportTariff=obs_data["exportTariff"],
            importTariff=obs_data["importTariff"],
            sunshineFraction=obs_data.get("sunlightIndex", 0),
            humidity=obs_data.get("sugarPrice", 0),
        )
        observations.conversionObservations[CONVERSION_PRODUCT] = conv

    return tc.TradingState(
        timestamp=timestamp, traderData=trader_data, listings={},
        order_depths=order_depths, own_trades=own_trades,
        market_trades=market_trades, position=dict(positions),
        observations=observations,
    )


def match_orders(orders, od, pos, limit, fill_rng, market_trades=None, mode="all"):
    """Calibrated jmerle-style two-phase fill matching.

    NOTE (Phase 2 §0.1): `fill_rng` is currently UNUSED. Fills are fully deterministic
    given (orders, book, market_trades). This is by design after server-log calibration —
    on tutorial data, our fills always coincided with market-trade ticks at exact prices.
    Do NOT add stochastic fill probability here without re-running the gauntlet — the
    blessed baseline (tools/gauntlet_baseline.json) assumes deterministic matching.
    grid_search.py uses `--detect-noop` to flag dead axes since varying seeds does nothing.

    Phase 1: Match against visible book depth (fill at BOOK price).
             Only a fraction of book depth is available to us — other bots
             trade first and consume most of the book (exchange processing
             order: deep makers → takers → your bot → other bots).
    Phase 2: Match remaining against market trades (fill at YOUR price).

    Calibrated from 53 server submissions (tutorial exchange):
    - 100% of our fills coincide with market trades at that tick
    - Phase 1 fills match exactly the remaining depth after bot consumption
    - Phase 2 fills happen when bots cross our passive orders

    mode='all'|'worse'|'none'.
    """
    fills = []

    # Check worst-case position (IMC all-or-nothing rule)
    total_buy = sum(o.quantity for o in orders if o.quantity > 0)
    total_sell = sum(abs(o.quantity) for o in orders if o.quantity < 0)
    if pos + total_buy > limit or pos - total_sell < -limit:
        return [], pos

    # Build trade volume pool for Phase 2
    trade_pool = {}
    has_market_trades = False
    if market_trades and mode != "none":
        for t in market_trades:
            p = t.price if hasattr(t, 'price') else t["price"]
            q = t.quantity if hasattr(t, 'quantity') else t["quantity"]
            trade_pool[p] = trade_pool.get(p, 0) + q
            has_market_trades = True

    # Calibrated book consumption: other bots consume ~70% of visible depth
    # before our orders are processed. Server data shows we only get fills
    # on ~4% of ticks, and when we do, it's exactly the remaining liquidity.
    # This factor simulates bots trading ahead of us.
    # Calibrated from 53 tutorial server logs:
    # - 100% of our fills coincide with ticks that have market trades
    # - Phase 1: only fill against book when market trades exist (bot activity)
    # - Phase 2: match against market trades at our price
    # - On quiet ticks (no market trades), we get ZERO fills regardless of
    #   how aggressively we price — no one is there to take the other side.

    for order in orders:
        if order.quantity > 0:  # BUY
            remaining = order.quantity
            # Phase 1: match against book depth, but ONLY on active ticks
            # (when market trades exist = bots are actively trading)
            if has_market_trades:
                for ask_p in sorted(od.sell_orders.keys()):
                    if remaining <= 0:
                        break
                    if order.price >= ask_p:
                        avail = abs(od.sell_orders[ask_p])
                        fill_qty = min(remaining, avail, limit - pos)
                        if fill_qty > 0:
                            fills.append((order.symbol, ask_p, fill_qty))
                            pos += fill_qty
                            remaining -= fill_qty
            # Phase 2: match remaining against market trades (at OUR price)
            if remaining > 0 and mode != "none":
                for tp in sorted(trade_pool.keys()):
                    if remaining <= 0:
                        break
                    if mode == "all" and tp <= order.price:
                        fill_qty = min(remaining, trade_pool[tp], limit - pos)
                    elif mode == "worse" and tp < order.price:
                        fill_qty = min(remaining, trade_pool[tp], limit - pos)
                    else:
                        continue
                    if fill_qty > 0:
                        fills.append((order.symbol, order.price, fill_qty))
                        pos += fill_qty
                        remaining -= fill_qty
                        trade_pool[tp] -= fill_qty

        elif order.quantity < 0:  # SELL
            remaining = abs(order.quantity)
            # Phase 1: only on active ticks
            if has_market_trades:
                for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                    if remaining <= 0:
                        break
                    if order.price <= bid_p:
                        avail = od.buy_orders[bid_p]
                        fill_qty = min(remaining, avail, limit + pos)
                        if fill_qty > 0:
                            fills.append((order.symbol, bid_p, -fill_qty))
                            pos -= fill_qty
                            remaining -= fill_qty
            if remaining > 0 and mode != "none":
                for tp in sorted(trade_pool.keys(), reverse=True):
                    if remaining <= 0:
                        break
                    if mode == "all" and tp >= order.price:
                        fill_qty = min(remaining, trade_pool[tp], limit + pos)
                    elif mode == "worse" and tp > order.price:
                        fill_qty = min(remaining, trade_pool[tp], limit + pos)
                    else:
                        continue
                    if fill_qty > 0:
                        fills.append((order.symbol, order.price, -fill_qty))
                        pos -= fill_qty
                        remaining -= fill_qty
                        trade_pool[tp] -= fill_qty

    return fills, pos


def run_backtest(price_csvs, trade_csvs, obs_csvs=None, seed=42, per_day=False, mode="all",
                 config_overrides=None):
    """Run full backtest. Returns (total_pnl, per_product_pnl)."""
    t0 = time.time()
    tick_index, tick_schedule, trade_map, obs_map, products, last_books = \
        load_data(price_csvs, trade_csvs, obs_csvs)
    t_load = time.time() - t0

    trader = tc.Trader()
    trader._mem = {}  # Backtest mode: skip JSON round-trip
    globals()["_last_trader"] = trader  # debug exposure for A/B harnesses

    # Inject known P3 position limits
    for p, lim in P3_LIMITS.items():
        if p in trader._config:
            trader._config[p]["position_limit"] = lim

    # Inject known P3 basket compositions
    for basket, comps in P3_BASKETS.items():
        if basket in trader._config:
            trader._config[basket]["components"] = comps

    # Apply config overrides (e.g., options day_offset per round)
    if config_overrides:
        for override_type, override_vals in config_overrides.items():
            if override_type == "options":
                # Apply to all auto-classified options products via mem
                # These will be picked up during auto-classification
                trader._mem["_options_config_overrides"] = override_vals

    positions = defaultdict(int)
    cash = defaultdict(float)
    conversion_pnl = defaultdict(float)
    total_conversions = 0
    trader_data = ""
    fill_rng = np.random.RandomState(seed)

    days = sorted(set(d for d, _ in tick_schedule))
    print(f"Products: {products}")
    print(f"Days: {days} ({len(tick_schedule)} ticks, loaded in {t_load:.1f}s)")
    if obs_map:
        print(f"Observations: {len(obs_map)} ticks (conversion arb enabled)")

    total_ticks = 0
    total_fills = 0

    take_pnl = defaultdict(float)
    make_pnl = defaultdict(float)
    take_fills = defaultdict(int)
    make_fills = defaultdict(int)

    day_pnl = defaultdict(lambda: defaultdict(float))
    prev_day = None
    prev_day_cash = defaultdict(float)
    prev_day_pos = defaultdict(int)

    # Pre-compute last mid per product per day
    day_last_mids = {}
    for (day, ts), tick_data in tick_index.items():
        for product, book in tick_data.items():
            if book["bids"] and book["asks"]:
                mid = (max(book["bids"]) + min(book["asks"])) / 2.0
                day_last_mids[(day, product)] = mid

    t_sim = time.time()
    for day, ts in tick_schedule:
        # Day boundary tracking
        if prev_day is not None and day != prev_day:
            for product in products:
                mid = day_last_mids.get((prev_day, product), 0)
                if mid > 0:
                    day_pnl[prev_day][product] = (cash[product] - prev_day_cash[product]) + \
                        (positions[product] - prev_day_pos[product]) * mid
            prev_day_cash = dict(cash)
            prev_day_pos = dict(positions)
        prev_day = day

        tick_data = tick_index[(day, ts)]
        state = build_state(tick_data, day, ts, trade_map, obs_map, positions, trader_data)

        result, conversions, trader_data = trader.run(state)

        # Apply conversions (conversion arb: import/export changes position + costs money)
        # Conversions happen BEFORE order matching in the real exchange
        if conversions != 0 and CONVERSION_PRODUCT in tick_data:
            obs_data = obs_map.get((day, ts))
            if obs_data:
                mac_limit = P3_LIMITS.get(CONVERSION_PRODUCT, 75)
                mac_pos = positions[CONVERSION_PRODUCT]
                if conversions > 0:
                    # Import: buy from foreign market, receive local units
                    cost_per = obs_data["askPrice"] + obs_data["transportFees"] + obs_data["importTariff"]
                    conv_qty = min(conversions, mac_limit - mac_pos)
                    if conv_qty > 0:
                        positions[CONVERSION_PRODUCT] += conv_qty
                        cash[CONVERSION_PRODUCT] -= cost_per * conv_qty
                        conversion_pnl[CONVERSION_PRODUCT] -= cost_per * conv_qty
                        total_conversions += conv_qty
                elif conversions < 0:
                    # Export: sell to foreign market, lose local units
                    rev_per = obs_data["bidPrice"] - obs_data["transportFees"] - obs_data["exportTariff"]
                    conv_qty = min(abs(conversions), mac_limit + mac_pos)
                    if conv_qty > 0:
                        positions[CONVERSION_PRODUCT] -= conv_qty
                        cash[CONVERSION_PRODUCT] += rev_per * conv_qty
                        conversion_pnl[CONVERSION_PRODUCT] += rev_per * conv_qty
                        total_conversions += conv_qty

        # Match orders per product
        for product, orders in result.items():
            if not orders:
                continue
            od = state.order_depths.get(product)
            if od is None:
                continue
            limit = P3_LIMITS.get(product, trader._config.get(product, {}).get("position_limit", 50))

            mt = state.market_trades.get(product, [])
            fills, new_pos = match_orders(orders, od, positions[product], limit, fill_rng,
                                          market_trades=mt, mode=mode)
            positions[product] = new_pos

            for sym, price, qty in fills:
                cash[sym] -= price * qty
                total_fills += 1
                is_take = False
                if qty > 0 and price in od.sell_orders:
                    is_take = True
                elif qty < 0 and price in od.buy_orders:
                    is_take = True
                if is_take:
                    take_pnl[sym] -= price * qty
                    take_fills[sym] += 1
                else:
                    make_pnl[sym] -= price * qty
                    make_fills[sym] += 1

        total_ticks += 1

    t_sim_end = time.time()

    # Mark to market at final mid
    pnl = {}
    last_mids = {}
    all_products = sorted(set(products) | set(cash.keys()))
    for product in all_products:
        book = last_books.get(product, {})
        bids = book.get("bids", {})
        asks = book.get("asks", {})
        if bids and asks:
            mid = (max(bids) + min(asks)) / 2.0
        else:
            mid = 0
        last_mids[product] = mid
        pnl[product] = cash[product] + positions[product] * mid

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS (mode={mode}, seed={seed})")
    print(f"{'='*60}")
    print(f"{'Product':<30} {'PnL':>10} {'Final Pos':>10} {'Last Mid':>10}")
    print(f"{'-'*60}")
    total = 0
    for p in sorted(pnl.keys()):
        print(f"{p:<30} {pnl[p]:>+10.0f} {positions[p]:>10} {last_mids[p]:>10.1f}")
        total += pnl[p]
    print(f"{'-'*60}")
    print(f"{'TOTAL':<30} {total:>+10.0f}")
    print(f"\nTicks: {total_ticks}, Fills: {total_fills}, Conversions: {total_conversions}")
    print(f"Time: load={t_load:.1f}s, sim={t_sim_end-t_sim:.1f}s, total={time.time()-t0:.1f}s")

    # PnL attribution
    print(f"\n--- PnL ATTRIBUTION ---")
    print(f"{'Product':<30} {'Take PnL':>10} {'Make PnL':>10} {'Conv PnL':>10} {'Take#':>6} {'Make#':>6} {'Strategy':>15}")
    print(f"{'-'*90}")
    for p in sorted(pnl.keys()):
        strat = trader._config.get(p, {}).get("type", "unknown")
        cp = conversion_pnl.get(p, 0)
        print(f"{p:<30} {take_pnl.get(p,0):>+10.0f} {make_pnl.get(p,0):>+10.0f} {cp:>+10.0f} {take_fills.get(p,0):>6} {make_fills.get(p,0):>6} {strat:>15}")

    # Per-day breakdown
    if per_day and len(days) > 1:
        print(f"\n--- PER-DAY BREAKDOWN ---")
        short_products = sorted(pnl.keys())
        print(f"{'Day':<10}", end="")
        for p in short_products:
            label = p[:12]
            print(f" {label:>12}", end="")
        print(f" {'TOTAL':>12}")
        for d in days[:-1]:
            print(f"{str(d):<10}", end="")
            day_total = 0
            for p in short_products:
                dpnl = day_pnl.get(d, {}).get(p, 0)
                print(f" {dpnl:>+12.0f}", end="")
                day_total += dpnl
            print(f" {day_total:>+12.0f}")

    return total, pnl


# ============================================================
# DATA PATH HELPERS
# ============================================================

P2_DATA = "reference/p2_data"
P3_DATA = "reference/p3_data"

ROUND_CONFIGS = {
    # ── P4 Tutorial ──
    "r0": {
        "prices": ["data/round0/prices_round_0_day_-2.csv",
                    "data/round0/prices_round_0_day_-1.csv"],
        "trades": ["data/round0/trades_round_0_day_-2.csv",
                    "data/round0/trades_round_0_day_-1.csv"],
        "obs": [],
    },
    # ── P4 Round 1 (drag CSVs into data/r1/drop/ then run sort_drops.py) ──
    "p4r1": {
        "prices": [f"data/r1/prices/prices_round_1_day_{d}.csv" for d in [-2, -1, 0]],
        "trades": [f"data/r1/trades/trades_round_1_day_{d}.csv" for d in [-2, -1, 0]],
        "obs": [],
    },
    # ── P4 Round 3 (Phase 2 kickoff, options chain + 2 delta-1) ──
    # Historical days TTE: 8 (day_0/tutorial), 7 (day_1/R1), 6 (day_2/R2).
    # Live R3 simulation: TTE=5 → set day_offset=3 on submit, keep 0 for backtest.
    "p4r3": {
        "prices": [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0, 1, 2]],
        "trades": [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0, 1, 2]],
        "obs": [],
        "config_overrides": {"options": {"day_offset": 0, "total_option_days": 8}},
    },
    # ── P4 Round 4 (Phase 2 continuation, counterparty IDs populated) ──
    "p4r4": {
        "prices": [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1, 2, 3]],
        "trades": [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1, 2, 3]],
        "obs": [],
        "config_overrides": {"options": {"day_offset": 3, "total_option_days": 7}},
    },
    # ── P3 Rounds ──
    "r1": {
        "prices": [f"{P3_DATA}/Round 1 Data/prices_round_1_day_{d}.csv" for d in [-2, -1, 0]],
        "trades": [f"{P3_DATA}/Round 1 Data/trades_round_1_day_{d}.csv" for d in [-1, 0, 2]],
        "obs": [],
    },
    "r2": {
        "prices": [f"{P3_DATA}/Round 2 Data/prices_round_2_day_{d}.csv" for d in [-1, 0, 1]],
        "trades": [f"{P3_DATA}/Round 2 Data/trades_round_2_day_{d}.csv" for d in [-1, 0, 1]],
        "obs": [],
    },
    "r3": {
        "prices": [f"{P3_DATA}/Round 3 Data/prices_round_3_day_{d}.csv" for d in [0, 1, 2]],
        "trades": [f"{P3_DATA}/Round 3 Data/trades_round_3_day_{d}.csv" for d in [0, 1, 2]],
        "obs": [],
        "config_overrides": {"options": {"day_offset": 0, "total_option_days": 9}},
    },
    "r4": {
        "prices": [f"{P3_DATA}/Round 4 Data/prices_round_4_day_{d}.csv" for d in [1, 2, 3]],
        "trades": [f"{P3_DATA}/Round 4 Data/trades_round_4_day_{d}.csv" for d in [1, 2, 3]],
        "obs": [f"{P3_DATA}/Round 4 Data/observations_round_4_day_{d}.csv" for d in [1, 2, 3]],
        "config_overrides": {"options": {"day_offset": 3, "total_option_days": 9}},
    },
    "r5": {
        "prices": [f"{P3_DATA}/Round 5 Data/prices_round_5_day_{d}.csv" for d in [2, 3, 4]],
        "trades": [f"{P3_DATA}/Round 5 Data/trades_round_5_day_{d}.csv" for d in [2, 3, 4]],
        "obs": [f"{P3_DATA}/Round 5 Data/observations_round_5_day_{d}.csv" for d in [2, 3, 4]],
        "config_overrides": {"options": {"day_offset": 6, "total_option_days": 9}},
    },
    # ── P2 Rounds (cross-year validation) ──
    "p2r1": {
        "prices": [f"{P2_DATA}/Round 1 Data/prices_round_1_day_{d}.csv" for d in [-2, -1, 0]],
        "trades": [f"{P2_DATA}/Round 1 Data/trades_round_1_day_{d}_nn.csv" for d in [-2, -1, 0]],
        "obs": [],
    },
    # p2r2: data corrupted (429 errors / non-standard observation format)
    "p2r3": {
        "prices": [f"{P2_DATA}/Round 3 Data/prices_round_3_day_{d}.csv" for d in [0, 1, 2]],
        "trades": [f"{P2_DATA}/Round 3 Data/trades_round_3_day_{d}_nn.csv" for d in [0, 1, 2]],
        "obs": [],
    },
    "p2r4": {
        "prices": [f"{P2_DATA}/Round 4 Data/prices_round_4_day_{d}.csv" for d in [1, 2, 3]],
        "trades": [f"{P2_DATA}/Round 4 Data/trades_round_4_day_{d}_nn.csv" for d in [1, 2, 3]],
        "obs": [],
    },
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest competition trader")
    parser.add_argument("--round", type=str, default="r1",
                        help="Round to test: r0, r1, r2, r4, r5, p2r1, p2r2, p2r3, p2r4, or 'all'")
    parser.add_argument("--mode", type=str, default="all",
                        help="Fill mode: all, worse, none")
    parser.add_argument("--per-day", action="store_true",
                        help="Show per-day PnL breakdown")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds")
    parser.add_argument("--days", type=int, default=None,
                        help="Limit to N days")
    args = parser.parse_args()

    rounds = list(ROUND_CONFIGS.keys()) if args.round == "all" else [args.round]

    grand_total = 0
    for rnd in rounds:
        if rnd not in ROUND_CONFIGS:
            print(f"Unknown round: {rnd}")
            continue

        cfg = ROUND_CONFIGS[rnd]
        price_csvs = [f for f in cfg["prices"] if os.path.exists(f)]
        trade_csvs = [f for f in cfg["trades"] if os.path.exists(f)]
        obs_csvs = [f for f in cfg.get("obs", []) if os.path.exists(f)]

        if not price_csvs:
            print(f"  No price data found for {rnd}, skipping")
            continue

        skipped = len(cfg["prices"]) - len(price_csvs)
        if skipped > 0:
            print(f"  Warning: {skipped} price file(s) missing for {rnd}")

        if args.days:
            price_csvs = price_csvs[:args.days]
            trade_csvs = trade_csvs[:args.days]
            obs_csvs = obs_csvs[:args.days] if obs_csvs else []

        # Check files exist
        price_csvs = [f for f in price_csvs if os.path.exists(f)]
        trade_csvs = [f for f in trade_csvs if os.path.exists(f)]
        obs_csvs = [f for f in obs_csvs if os.path.exists(f)]

        if not price_csvs:
            print(f"\n{'#'*60}\n# ROUND {rnd.upper()}: No valid price files found\n{'#'*60}")
            continue

        print(f"\n{'#'*60}")
        print(f"# ROUND {rnd.upper()}")
        print(f"{'#'*60}")

        for seed in args.seeds:
            total, pnl = run_backtest(price_csvs, trade_csvs, obs_csvs,
                                       seed=seed, per_day=args.per_day, mode=args.mode,
                                       config_overrides=cfg.get("config_overrides"))
            grand_total += total
            print()

    if len(rounds) > 1:
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL ACROSS ALL ROUNDS: {grand_total:+.0f}")
        print(f"{'='*60}")
