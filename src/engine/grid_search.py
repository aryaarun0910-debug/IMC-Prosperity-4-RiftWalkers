"""
Grid Search: Systematic parameter optimization for trader.py.

WARNING (Phase 2 §0.1, diagnosed 2026-04-19):
    The fill model in backtest.py:188-296 is FULLY DETERMINISTIC — `fill_rng` is
    threaded through but never consumed. Seeds [42, 123, 456] produce identical
    fills, so the multi-seed averaging here is decorative variance.

    REAL variance comes from PER-DAY PnL splits, not from seeds. We now group
    PnL by day and compute mean/std over days. Use --detect-noop on every fresh
    grid to identify dead axes (params that produce identical orders across all
    combos for the first N ticks).

    Historical "winner" rankings produced before this warning are statistically
    unsupported — re-validate with the gauntlet (tools/gauntlet.py) before
    shipping anything that came out of an old grid_search run.

Usage:
    py -3.12 grid_search.py                    # Run default grid
    py -3.12 grid_search.py --product KELP     # Optimize single product
    py -3.12 grid_search.py --fast             # Quick grid (fewer combos)
    py -3.12 grid_search.py --detect-noop      # Identify dead axes only, don't search
"""

import sys
import os
import json
import itertools
import copy
import importlib.util
import time
import math

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

# Import backtest runner
from backtest import run_backtest

# ============================================================
# PARAMETER GRIDS
# ============================================================

# Each grid: {product: {param_name: [values]}}
# Only vary parameters that matter most (Frankfurt: find flat regions)

GRIDS = {
    # P4 R1 products — grid tests parameters that actually affect PnL
    # (make_offset/take_width confirmed dead: penny-jump overrides them)
    # inventory_threshold now live — prevents position extremes and MTM losses
    "ASH_COATED_OSMIUM": {
        "inventory_threshold_1": [20, 30, 40, 50],
        "inventory_threshold_2": [45, 55, 65, 75],
    },
    "INTARIAN_PEPPER_ROOT": {
        # drift_bid_offset/drift_ask_margin confirmed dead: aggressive cross fills pos
        # immediately, passive bid rarely fires. Only fv_drift affects early-tick FV.
        "fv_drift": [0.08, 0.09, 0.10, 0.111, 0.12],
    },
    "RAINFOREST_RESIN": {
        "make_offset": [1, 2, 3, 4],
        "make_size": [30, 40, 45, 50],
        "take_width": [1, 2],
        "inventory_threshold_1": [25, 35, 45],
    },
    "KELP": {
        "make_size": [15, 20, 25, 30, 40],
        "ar_order": [3, 4, 5, 6],
        "ar_refit_every": [15, 25, 50],
        "olivia_window": [3, 5, 8],
    },
    "SQUID_INK": {
        "make_size": [5, 10, 15, 20],
        "vol_window": [30, 50, 80],
        "vol_mr_threshold": [1.5, 2.0, 2.5, 3.0],
    },
    "EMERALDS": {
        "make_offset": [5, 7, 9],
        "make_size": [60, 80],
        "inventory_threshold_1": [30, 40, 50],
    },
    "TOMATOES": {
        "take_ask_threshold": [4, 6, 8],
        "make_size": [60, 80],
        "ema_alpha": [0.003, 0.005, 0.008],
    },
}

FAST_GRIDS = {
    "ASH_COATED_OSMIUM": {
        "inventory_threshold_1": [20, 30, 40, 50],
        "inventory_threshold_2": [45, 55, 65, 75],
    },
    "INTARIAN_PEPPER_ROOT": {
        "fv_drift": [0.08, 0.09, 0.10, 0.111, 0.12],
    },
    "RAINFOREST_RESIN": {
        "make_offset": [1, 2, 3],
        "make_size": [35, 45],
    },
    "KELP": {
        "make_size": [20, 30],
        "ar_order": [4, 5],
    },
    "SQUID_INK": {
        "make_size": [10, 15],
        "vol_mr_threshold": [2.0, 2.5],
    },
}

# ============================================================
# GRID SEARCH ENGINE
# ============================================================

PRICE_CSVS_P3R1 = sorted([
    "data/p3_round1/prices_round_1_day_-2.csv",
    "data/p3_round1/prices_round_1_day_-1.csv",
    "data/p3_round1/prices_round_1_day_0.csv",
])
TRADE_CSVS_P3R1 = sorted([
    "data/p3_round1/trades_round_1_day_-2.csv",
    "data/p3_round1/trades_round_1_day_-1.csv",
    "data/p3_round1/trades_round_1_day_0.csv",
])
PRICE_CSVS_R0 = sorted([
    "data/round0/prices_round_0_day_-2.csv",
    "data/round0/prices_round_0_day_-1.csv",
])
TRADE_CSVS_R0 = sorted([
    "data/round0/trades_round_0_day_-2.csv",
    "data/round0/trades_round_0_day_-1.csv",
])

# P4 R1 data paths (drag CSVs into data/r1/drop/ then run sort_drops.py)
PRICE_CSVS_P4R1 = sorted([
    f"data/r1/prices/prices_round_1_day_{d}.csv" for d in [-2, -1, 0]
])
TRADE_CSVS_P4R1 = sorted([
    f"data/r1/trades/trades_round_1_day_{d}.csv" for d in [-2, -1, 0]
])

# Tutorial products use round0 data; everything else uses P3 R1
# When P4 R1 data exists, all products auto-switch to it
TUTORIAL_PRODUCTS = {"EMERALDS", "TOMATOES"}

def get_data_paths(product=None):
    """Return (price_csvs, trade_csvs) for the given product.
    Auto-detects P4 R1 data if it exists."""
    import os
    # If P4 R1 data exists, prefer it for everything
    if any(os.path.exists(f) for f in PRICE_CSVS_P4R1):
        return PRICE_CSVS_P4R1, TRADE_CSVS_P4R1
    if product and product in TUTORIAL_PRODUCTS:
        return PRICE_CSVS_R0, TRADE_CSVS_R0
    return PRICE_CSVS_P3R1, TRADE_CSVS_P3R1

# Default (overridden per-product in run_with_config_override)
PRICE_CSVS = PRICE_CSVS_P3R1
TRADE_CSVS = TRADE_CSVS_P3R1

SEEDS = [42, 123, 456]  # NOTE: dead axis — backtest.py fill model is deterministic. Variance comes from days.

_cached_data = None
_cached_data_key = None


def _hash_orders(result):
    """Stable hash of an order dict produced by trader.run() for the no-op detector.

    Two combos with identical hashes produced identical orders for that tick.
    If every combo on a parameter axis hashes the same way for the first N ticks,
    the axis is dead — varying it cannot change PnL.
    """
    key = []
    for sym in sorted(result.keys()):
        for o in result[sym]:
            key.append((sym, int(o.price), int(o.quantity)))
    return hash(tuple(key))


def param_combos(grid):
    """Generate all parameter combinations from a grid."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_with_config_override(product, overrides, seeds=None, per_day=False):
    """Run backtest with specific parameter overrides for one product.

    Returns (mean_pnl, min_pnl, max_pnl, pnls).

    Modes:
      per_day=False (default): pnls is per-seed (legacy). Variance is fake — fill
        model is deterministic, all seeds produce identical PnL.
      per_day=True: pnls is per-day (real variance). Runs ONE simulation pass and
        captures end-of-day MTM diffs as the per-day PnL series. This is what
        Bayesian/GP-UCB needs for a meaningful noise estimate.

    NOTE (Phase 2 §0.1): the multi-seed legacy mode is preserved for backward
    compat but should NOT be used to compare configs. Always validate via gauntlet.
    """
    if seeds is None:
        seeds = SEEDS

    # Import fresh trader module each time (backtest.py uses its own import)
    # We need to patch the CONFIG BEFORE backtest imports trader
    # Strategy: reload backtest's trader reference with our overrides
    import numpy as np
    from collections import defaultdict

    # Load trader fresh
    spec = importlib.util.spec_from_file_location("tc_gs", "trader.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)

    # Apply overrides
    original = {}
    if product in tc.CONFIG:
        for k, v in overrides.items():
            original[k] = tc.CONFIG[product].get(k)
            tc.CONFIG[product][k] = v

    # Load data (cache across calls, keyed by product type)
    global _cached_data, _cached_data_key
    price_csvs, trade_csvs = get_data_paths(product)
    cache_key = "tutorial" if product in TUTORIAL_PRODUCTS else "p3r1"
    if '_cached_data' not in globals() or _cached_data is None or _cached_data_key != cache_key:
        _cached_data_key = cache_key
        import pandas as pd
        frames = []
        for p in price_csvs:
            df = pd.read_csv(p, sep=";")
            df.columns = [c.strip().lower() for c in df.columns]
            frames.append(df)
        prices = pd.concat(frames, ignore_index=True)
        prices = prices.sort_values(["day", "timestamp"]).reset_index(drop=True)

        trade_map = {}
        sorted_days = sorted(prices["day"].unique())
        for i, tp in enumerate(trade_csvs):
            if i >= len(sorted_days):
                break
            day = sorted_days[i]
            tdf = pd.read_csv(tp, sep=";")
            tdf.columns = [c.strip().lower() for c in tdf.columns]
            for _, row in tdf.iterrows():
                key = (day, int(row["timestamp"]), row["symbol"])
                if key not in trade_map:
                    trade_map[key] = []
                trade_map[key].append({
                    "price": int(row["price"]),
                    "quantity": int(row["quantity"]),
                    "buyer": str(row.get("buyer", "")),
                    "seller": str(row.get("seller", "")),
                })
        _cached_data = (prices, trade_map)

    prices, trade_map = _cached_data
    import pandas as pd

    pnls = []
    per_day_runs = []  # list of dicts day -> mtm_at_eod for this product (per seed)
    seed_loop = [seeds[0]] if per_day else seeds
    for seed in seed_loop:
        fill_rng = np.random.RandomState(seed)
        trader = tc.Trader()
        positions = defaultdict(int)
        cash = defaultdict(float)
        trader_data = ""
        products_list = sorted(prices["product"].unique())

        # Per-day MTM tracking for the target product
        last_seen_day = None
        last_mid_per_day = {}      # day -> last (bid+ask)/2 seen for product
        per_day_eod_mtm = {}       # day -> cumulative MTM at end of that day
        for (day, ts), group in prices.groupby(["day", "timestamp"]):
            # Per-day MTM snapshot at day boundary (§0.1: real variance for GP-UCB)
            if per_day and last_seen_day is not None and day != last_seen_day:
                _lm = last_mid_per_day.get(last_seen_day)
                if _lm is not None:
                    per_day_eod_mtm[last_seen_day] = cash[product] + positions[product] * _lm
            last_seen_day = day
            # Build state
            order_depths = {}
            for _, row in group.iterrows():
                p = row["product"]
                od = tc.OrderDepth()
                for i in range(1, 4):
                    bp, bv = f"bid_price_{i}", f"bid_volume_{i}"
                    ap, av = f"ask_price_{i}", f"ask_volume_{i}"
                    if bp in row and pd.notna(row.get(bp)):
                        od.buy_orders[int(row[bp])] = int(row[bv])
                    if ap in row and pd.notna(row.get(ap)):
                        od.sell_orders[int(row[ap])] = int(row[av])
                order_depths[p] = od

            market_trades = {}
            own_trades = {}
            for p in order_depths:
                key = (day, ts, p)
                raw = trade_map.get(key, [])
                trades = [tc.Trade(symbol=p, price=t["price"], quantity=t["quantity"],
                                   buyer=t.get("buyer",""), seller=t.get("seller",""),
                                   timestamp=ts) for t in raw]
                market_trades[p] = trades
                own_trades[p] = []

            state = tc.TradingState(
                timestamp=ts, traderData=trader_data, listings={},
                order_depths=order_depths, own_trades=own_trades,
                market_trades=market_trades, position=dict(positions),
                observations=tc.Observation(),
            )

            result, conversions, trader_data = trader.run(state)

            # Match orders (calibrated jmerle-style: book depth + market trades)
            # Server data: other bots consume ~70% of book before us
            BOOK_CONSUMPTION = 0.95
            for p, orders in result.items():
                if not orders:
                    continue
                od = state.order_depths.get(p)
                if od is None:
                    continue
                limit = trader._config.get(p, {}).get("position_limit", 50)

                # IMC all-or-nothing
                total_buy = sum(o.quantity for o in orders if o.quantity > 0)
                total_sell = sum(abs(o.quantity) for o in orders if o.quantity < 0)
                if positions[p] + total_buy > limit or positions[p] - total_sell < -limit:
                    continue

                # Build trade pool for Phase 2
                trade_pool = {}
                key = (day, ts, p)
                for t in trade_map.get(key, []):
                    tp = int(t["price"])
                    trade_pool[tp] = trade_pool.get(tp, 0) + int(t["quantity"])

                for order in orders:
                    if order.quantity > 0:
                        remaining = order.quantity
                        # Phase 1: match against book (with consumption)
                        for ask_p in sorted(od.sell_orders.keys()):
                            if remaining <= 0:
                                break
                            if order.price >= ask_p:
                                full_avail = abs(od.sell_orders.get(ask_p, 0))
                                avail = max(1, int(full_avail * (1 - BOOK_CONSUMPTION)))
                                fill_qty = min(remaining, avail, limit - positions[p])
                                if fill_qty > 0:
                                    positions[p] += fill_qty
                                    cash[p] -= fill_qty * ask_p
                                    remaining -= fill_qty
                        # Phase 2: match against market trades (at OUR price)
                        if remaining > 0:
                            for tp in sorted(trade_pool.keys()):
                                if remaining <= 0:
                                    break
                                if tp <= order.price and trade_pool[tp] > 0:
                                    fill_qty = min(remaining, trade_pool[tp], limit - positions[p])
                                    if fill_qty > 0:
                                        positions[p] += fill_qty
                                        cash[p] -= fill_qty * order.price
                                        remaining -= fill_qty
                                        trade_pool[tp] -= fill_qty
                    elif order.quantity < 0:
                        remaining = abs(order.quantity)
                        # Phase 1: match against book (with consumption)
                        for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                            if remaining <= 0:
                                break
                            if order.price <= bid_p:
                                full_avail = od.buy_orders.get(bid_p, 0)
                                avail = max(1, int(full_avail * (1 - BOOK_CONSUMPTION)))
                                fill_qty = min(remaining, avail, limit + positions[p])
                                if fill_qty > 0:
                                    positions[p] -= fill_qty
                                    cash[p] += fill_qty * bid_p
                                    remaining -= fill_qty
                        # Phase 2: match against market trades (at OUR price)
                        if remaining > 0:
                            for tp in sorted(trade_pool.keys(), reverse=True):
                                if remaining <= 0:
                                    break
                                if tp >= order.price and trade_pool[tp] > 0:
                                    fill_qty = min(remaining, trade_pool[tp], limit + positions[p])
                                    if fill_qty > 0:
                                        positions[p] -= fill_qty
                                        cash[p] += fill_qty * order.price
                                        remaining -= fill_qty
                                        trade_pool[tp] -= fill_qty

            # Track latest mid for the target product (per-day MTM input)
            if per_day:
                _td_od = order_depths.get(product)
                if _td_od is not None and _td_od.buy_orders and _td_od.sell_orders:
                    _bb = max(_td_od.buy_orders.keys())
                    _ba = min(_td_od.sell_orders.keys())
                    last_mid_per_day[day] = (_bb + _ba) / 2.0

        # Final per-day snapshot after loop end
        if per_day and last_seen_day is not None:
            _lm = last_mid_per_day.get(last_seen_day)
            if _lm is not None:
                per_day_eod_mtm[last_seen_day] = cash[product] + positions[product] * _lm
            per_day_runs.append(dict(per_day_eod_mtm))

        # Mark to market
        product_pnl = {}
        for p in products_list:
            prod_data = prices[prices["product"] == p]
            last_row = prod_data.iloc[-1]
            mid = (last_row["bid_price_1"] + last_row["ask_price_1"]) / 2.0
            product_pnl[p] = cash[p] + positions[p] * mid

        pnls.append(product_pnl.get(product, 0.0))

    # Per-day mode: convert cumulative EOD MTM into per-day deltas (real variance)
    if per_day and per_day_runs:
        eod = per_day_runs[0]
        days_sorted = sorted(eod.keys())
        diffs, prev = [], 0.0
        for d in days_sorted:
            diffs.append(eod[d] - prev)
            prev = eod[d]
        if diffs:
            pnls = diffs

    mean_pnl = sum(pnls) / len(pnls) if pnls else 0
    return mean_pnl, min(pnls), max(pnls), pnls


def detect_no_op_axis(product, grid, n_ticks=1000, verbose=True):
    """Run trader.run() for the first n_ticks under each combo and hash the orders.

    If every combo on a parameter axis produces the same hash sequence, that axis
    is DEAD — varying it cannot affect PnL. This is the §0.1 fix: catch dead axes
    before wasting hours grid-searching them.

    Returns dict {axis_name: True if dead}.
    """
    from collections import defaultdict
    import pandas as pd
    import importlib.util

    # Load data once
    global _cached_data, _cached_data_key
    price_csvs, trade_csvs = get_data_paths(product)
    cache_key = "tutorial" if product in TUTORIAL_PRODUCTS else "p3r1"
    if _cached_data is None or _cached_data_key != cache_key:
        _cached_data_key = cache_key
        frames = []
        for p in price_csvs:
            df = pd.read_csv(p, sep=";")
            df.columns = [c.strip().lower() for c in df.columns]
            frames.append(df)
        prices = pd.concat(frames, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)
        trade_map = {}
        sorted_days = sorted(prices["day"].unique())
        for i, tp in enumerate(trade_csvs):
            if i >= len(sorted_days):
                break
            day = sorted_days[i]
            tdf = pd.read_csv(tp, sep=";")
            tdf.columns = [c.strip().lower() for c in tdf.columns]
            for _, row in tdf.iterrows():
                key = (day, int(row["timestamp"]), row["symbol"])
                trade_map.setdefault(key, []).append({
                    "price": int(row["price"]), "quantity": int(row["quantity"]),
                    "buyer": str(row.get("buyer", "")), "seller": str(row.get("seller", "")),
                })
        _cached_data = (prices, trade_map)
    prices, trade_map = _cached_data

    def run_combo(combo):
        spec = importlib.util.spec_from_file_location("tc_no", "trader.py")
        tc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tc)
        if product in tc.CONFIG:
            for k, v in combo.items():
                tc.CONFIG[product][k] = v
        trader = tc.Trader()
        positions = defaultdict(int)
        trader_data = ""
        hashes = []
        ticks = 0
        for (day, ts), group in prices.groupby(["day", "timestamp"]):
            if ticks >= n_ticks:
                break
            order_depths = {}
            for _, row in group.iterrows():
                p = row["product"]
                od = tc.OrderDepth()
                for i in range(1, 4):
                    bp, bv = f"bid_price_{i}", f"bid_volume_{i}"
                    ap, av = f"ask_price_{i}", f"ask_volume_{i}"
                    if bp in row and pd.notna(row.get(bp)):
                        od.buy_orders[int(row[bp])] = int(row[bv])
                    if ap in row and pd.notna(row.get(ap)):
                        od.sell_orders[int(row[ap])] = int(row[av])
                order_depths[p] = od
            market_trades, own_trades = {}, {}
            for p in order_depths:
                raw = trade_map.get((day, ts, p), [])
                market_trades[p] = [tc.Trade(symbol=p, price=t["price"], quantity=t["quantity"],
                                             buyer=t.get("buyer",""), seller=t.get("seller",""),
                                             timestamp=ts) for t in raw]
                own_trades[p] = []
            state = tc.TradingState(
                timestamp=ts, traderData=trader_data, listings={},
                order_depths=order_depths, own_trades=own_trades,
                market_trades=market_trades, position=dict(positions),
                observations=tc.Observation(),
            )
            result, _, trader_data = trader.run(state)
            hashes.append(_hash_orders({k: v for k, v in result.items() if k == product}))
            ticks += 1
        return tuple(hashes)

    if verbose:
        print(f"\n[NO-OP DETECTOR] {product}: hashing first {n_ticks} ticks across {sum(1 for _ in param_combos(grid))} combos")

    # Per-axis: hold all other axes constant at the first value, sweep this axis
    keys = list(grid.keys())
    dead = {}
    for axis in keys:
        # Build base combo from first values of other axes
        base = {k: grid[k][0] for k in keys}
        sigs = set()
        for v in grid[axis]:
            base[axis] = v
            sigs.add(run_combo(dict(base)))
        is_dead = (len(sigs) == 1)
        dead[axis] = is_dead
        if verbose:
            tag = "** DEAD AXIS **" if is_dead else "live"
            print(f"  axis={axis} values={grid[axis]} -> {tag}")

    return dead


def search_product(product, grid, seeds=None, verbose=True, per_day=False):
    """Grid search for a single product. Returns sorted results.

    per_day=True switches the cost function to per-day variance (§0.1 fix).
    """
    combos = list(param_combos(grid))
    total = len(combos)

    if verbose:
        print(f"\n{'='*70}")
        print(f"GRID SEARCH: {product}")
        print(f"Parameters: {list(grid.keys())}")
        print(f"Combinations: {total}")
        print(f"Mode: {'per-day variance' if per_day else 'multi-seed (DECORATIVE)'}")
        print(f"{'='*70}")

    results = []
    start = time.time()
    worst_day_cutoff = -5000  # Reject configs where any seed/day PnL < this

    for i, combo in enumerate(combos):
        mean_pnl, min_pnl, max_pnl, pnls = run_with_config_override(
            product, combo, seeds, per_day=per_day
        )
        spread = max_pnl - min_pnl
        # Sharpe ratio: mean / std (annualized doesn't matter for ranking)
        if len(pnls) > 1:
            pnl_std = max(1.0, (sum((p - mean_pnl)**2 for p in pnls) / (len(pnls) - 1)) ** 0.5)
            sharpe = mean_pnl / pnl_std
        else:
            sharpe = mean_pnl / 1000.0 if mean_pnl != 0 else 0.0
        # Worst-day filter: flag configs where any seed produces catastrophic loss
        worst_day_ok = bool(min_pnl >= worst_day_cutoff)
        results.append({
            "params": combo,
            "mean_pnl": mean_pnl,
            "min_pnl": min_pnl,
            "max_pnl": max_pnl,
            "spread": spread,
            "sharpe": sharpe,
            "worst_day_ok": worst_day_ok,
            "pnls": pnls,
        })

        if verbose:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            params_str = ", ".join(f"{k}={v}" for k, v in combo.items())
            flag = "" if worst_day_ok else " [WORST-DAY FAIL]"
            print(f"  [{i+1}/{total}] {params_str}: "
                  f"mean={mean_pnl:+.0f} sharpe={sharpe:.2f} spread={spread:.0f}{flag} "
                  f"[ETA: {eta:.0f}s]")

    # Sort: worst-day-ok first, then by mean PnL, then Sharpe as tiebreaker
    results.sort(key=lambda r: (r["worst_day_ok"], r["mean_pnl"], r["sharpe"]), reverse=True)

    if verbose:
        print(f"\n--- TOP 5 CONFIGS for {product} ---")
        rejected = sum(1 for r in results if not r["worst_day_ok"])
        if rejected:
            print(f"  ({rejected}/{total} configs rejected: worst-day PnL < {worst_day_cutoff})")
        for i, r in enumerate(results[:5]):
            params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            flag = " [REJECTED]" if not r["worst_day_ok"] else ""
            print(f"  #{i+1}: {params_str}")
            print(f"       mean={r['mean_pnl']:+.0f}  sharpe={r['sharpe']:.2f}  "
                  f"min={r['min_pnl']:+.0f}  max={r['max_pnl']:+.0f}  "
                  f"spread={r['spread']:.0f}{flag}")

        # Check for "flat region" — configs near the best that are also good
        best_mean = results[0]["mean_pnl"]
        flat_count = sum(1 for r in results if r["mean_pnl"] >= best_mean * 0.95)
        print(f"\n  Flat region: {flat_count}/{total} configs within 5% of best")
        if flat_count > total * 0.3:
            print(f"  GOOD: Wide flat region = robust parameters")
        else:
            print(f"  WARNING: Narrow peak = parameters may be overfit")

    return results


def bayesian_search(product, grid, max_evals=25, seeds=None, verbose=True, per_day=False):
    """Bayesian optimization: find optimal params in ~20 evals instead of 100+.

    Uses a simplified RBF kernel GP surrogate with Expected Improvement (EI).
    On gameday: ~20 backtests (5 min) instead of 100+ (25 min) for same quality.

    Algorithm:
    1. Evaluate 5 random starting points (Latin hypercube-ish)
    2. Fit GP surrogate to (params -> PnL) mapping
    3. Pick next point by maximizing EI acquisition function
    4. Repeat until max_evals

    per_day=True wires real per-day variance into the cost function (§0.1 fix —
    prerequisite for GP-UCB/EI to be meaningful, since the multi-seed variance is
    fake under the deterministic fill model).
    """
    import random
    if seeds is None:
        seeds = SEEDS

    all_combos = list(param_combos(grid))
    total_available = len(all_combos)
    max_evals = min(max_evals, total_available)
    keys = list(grid.keys())
    param_ranges = {k: grid[k] for k in keys}

    if verbose:
        print(f"\n{'='*70}")
        print(f"BAYESIAN OPTIMIZATION: {product}")
        print(f"Parameters: {keys}")
        print(f"Search space: {total_available} combos, evaluating ~{max_evals}")
        print(f"{'='*70}")

    # Normalize all combos to [0,1] for kernel computation
    def normalize(combo):
        vec = []
        for k in keys:
            vals = param_ranges[k]
            idx = vals.index(combo[k]) if combo[k] in vals else 0
            vec.append(idx / max(len(vals) - 1, 1))
        return vec

    def rbf_kernel(x1, x2, length_scale=0.3):
        """RBF kernel: k(x1,x2) = exp(-||x1-x2||^2 / (2*l^2))"""
        sq_dist = sum((a - b)**2 for a, b in zip(x1, x2))
        return math.exp(-sq_dist / (2 * length_scale**2))

    # Track evaluated points
    evaluated = []  # list of (combo_dict, normalized_vec, mean_pnl)
    evaluated_set = set()  # for dedup

    # Phase 1: Initial random sample (spread across space)
    n_init = min(5, max_evals)
    init_indices = list(range(total_available))
    random.seed(42)
    random.shuffle(init_indices)
    init_indices = init_indices[:n_init]

    start = time.time()
    for i, idx in enumerate(init_indices):
        combo = all_combos[idx]
        mean_pnl, min_pnl, max_pnl, pnls = run_with_config_override(product, combo, seeds, per_day=per_day)
        nvec = normalize(combo)
        evaluated.append((combo, nvec, mean_pnl))
        evaluated_set.add(idx)
        if verbose:
            params_str = ", ".join(f"{k}={v}" for k, v in combo.items())
            print(f"  [init {i+1}/{n_init}] {params_str}: mean={mean_pnl:+.0f}")

    # Phase 2: Bayesian optimization loop
    for iteration in range(n_init, max_evals):
        # Fit GP: compute kernel matrix + predictive distribution
        n = len(evaluated)
        y_vals = [e[2] for e in evaluated]
        y_mean = sum(y_vals) / n
        y_std = max(1.0, (sum((y - y_mean)**2 for y in y_vals) / n) ** 0.5)
        y_norm = [(y - y_mean) / y_std for y in y_vals]

        # Kernel matrix K + noise
        noise = 0.01
        K = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                K[i][j] = rbf_kernel(evaluated[i][1], evaluated[j][1])
                if i == j:
                    K[i][j] += noise

        # Invert K via Cholesky (reuse from trader.py logic)
        K_inv_y = _solve_linear(K, y_norm)
        if K_inv_y is None:
            # Fallback: pick random unevaluated point
            remaining = [i for i in range(total_available) if i not in evaluated_set]
            if not remaining:
                break
            best_idx = random.choice(remaining)
        else:
            # Compute EI for all unevaluated candidates
            best_y = max(y_norm)
            best_ei = -1
            best_idx = -1

            for idx in range(total_available):
                if idx in evaluated_set:
                    continue
                x_star = normalize(all_combos[idx])

                # Predictive mean: mu = k* @ K^-1 @ y
                k_star = [rbf_kernel(x_star, evaluated[i][1]) for i in range(n)]
                mu = sum(k_star[i] * K_inv_y[i] for i in range(n))

                # Predictive variance: sigma^2 = k(x*,x*) - k* @ K^-1 @ k*
                k_star_star = 1.0 + noise
                K_inv_k = _solve_linear(K, k_star)
                if K_inv_k is not None:
                    var = k_star_star - sum(k_star[i] * K_inv_k[i] for i in range(n))
                else:
                    var = k_star_star
                sigma = max(1e-6, var ** 0.5)

                # Expected Improvement
                z = (mu - best_y) / sigma
                # Approximate Phi(z) and phi(z) using logistic function
                phi = math.exp(-0.5 * z * z) / (2.5066282746)  # 1/sqrt(2pi)
                Phi = 1.0 / (1.0 + math.exp(-1.7 * z))  # logistic approx to normal CDF
                ei = (mu - best_y) * Phi + sigma * phi

                if ei > best_ei:
                    best_ei = ei
                    best_idx = idx

        if best_idx < 0 or best_idx in evaluated_set:
            remaining = [i for i in range(total_available) if i not in evaluated_set]
            if not remaining:
                break
            best_idx = remaining[0]

        combo = all_combos[best_idx]
        mean_pnl, min_pnl, max_pnl, pnls = run_with_config_override(product, combo, seeds, per_day=per_day)
        nvec = normalize(combo)
        evaluated.append((combo, nvec, mean_pnl))
        evaluated_set.add(best_idx)

        if verbose:
            elapsed = time.time() - start
            params_str = ", ".join(f"{k}={v}" for k, v in combo.items())
            print(f"  [bayes {iteration+1}/{max_evals}] {params_str}: "
                  f"mean={mean_pnl:+.0f} (EI={best_ei:.3f}) [{elapsed:.0f}s]")

    # Sort by PnL
    results = []
    for combo, nvec, pnl in evaluated:
        results.append({"params": combo, "mean_pnl": pnl, "min_pnl": pnl,
                        "max_pnl": pnl, "spread": 0, "pnls": [pnl]})
    results.sort(key=lambda r: r["mean_pnl"], reverse=True)

    if verbose:
        print(f"\n--- TOP 5 (from {len(evaluated)} evaluations) ---")
        for i, r in enumerate(results[:5]):
            params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            print(f"  #{i+1}: {params_str}  mean={r['mean_pnl']:+.0f}")

    return results


def _solve_linear(A, b):
    """Solve Ax=b via Cholesky for SPD matrix. Returns None if singular."""
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v < 1e-10:
                    return None
                L[i][j] = v ** 0.5
            else:
                L[i][j] = (A[i][j] - s) / L[j][j] if L[j][j] > 1e-10 else 0.0
    z = [0.0]*n
    for i in range(n):
        if abs(L[i][i]) < 1e-10:
            return None
        z[i] = (b[i] - sum(L[i][k]*z[k] for k in range(i))) / L[i][i]
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        if abs(L[i][i]) < 1e-10:
            return None
        x[i] = (z[i] - sum(L[j][i]*x[j] for j in range(i+1, n))) / L[i][i]
    return x


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grid search parameter optimization")
    parser.add_argument("--product", type=str, default=None,
                        help="Optimize single product (default: all)")
    parser.add_argument("--fast", action="store_true",
                        help="Use smaller grid for quick exploration")
    parser.add_argument("--bayes", action="store_true",
                        help="Use Bayesian optimization (faster, ~20 evals)")
    parser.add_argument("--max-evals", type=int, default=25,
                        help="Max evaluations for Bayesian mode (default: 25)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--detect-noop", action="store_true",
                        help="Detect dead axes for each product (does NOT search)")
    parser.add_argument("--noop-ticks", type=int, default=1000,
                        help="Ticks to hash per combo for dead-axis detection (default 1000)")
    parser.add_argument("--per-day", action="store_true",
                        help="Use per-day variance instead of per-seed (§0.1 fix; required for Bayesian)")
    args = parser.parse_args()

    print("\n" + "!" * 70)
    print("WARNING: backtest fill model is deterministic (Phase 2 §0.1).")
    print("         Multi-seed averaging is decorative; only per-day variance is real.")
    print("         Run with --detect-noop FIRST on any new grid to confirm axes are live.")
    print("         Validate every winner via tools/gauntlet.py before shipping.")
    print("!" * 70)

    grids = FAST_GRIDS if args.fast else GRIDS

    if args.product:
        if args.product not in grids:
            print(f"No grid defined for {args.product}")
            print(f"Available: {list(grids.keys())}")
            return
        grids = {args.product: grids[args.product]}

    if args.detect_noop:
        all_dead = {}
        for product, grid in grids.items():
            all_dead[product] = detect_no_op_axis(product, grid, n_ticks=args.noop_ticks)
        print(f"\n{'='*70}\nDEAD-AXIS REPORT\n{'='*70}")
        any_dead = False
        for product, dead in all_dead.items():
            for axis, is_dead in dead.items():
                if is_dead:
                    print(f"  ** {product}.{axis} is DEAD — varying it does nothing for first {args.noop_ticks} ticks")
                    any_dead = True
        if not any_dead:
            print("  All axes appear live within the first {0} ticks.".format(args.noop_ticks))
        else:
            print("\n  Recommendation: drop dead axes from the grid; pivot to:")
            print("    position_limit, kill_threshold, inventory_threshold_*, kalman_Q/R, bocpd_threshold")
        return

    all_results = {}
    best_config = {}

    for product, grid in grids.items():
        if args.bayes:
            results = bayesian_search(product, grid, max_evals=args.max_evals,
                                      seeds=args.seeds, per_day=args.per_day)
        else:
            results = search_product(product, grid, seeds=args.seeds, per_day=args.per_day)
        all_results[product] = results

        if results:
            best = results[0]
            best_config[product] = best["params"]
            print(f"\n  BEST for {product}: {best['params']}")
            print(f"  Apply with: CONFIG['{product}'].update({best['params']})")

    # Summary
    print(f"\n{'='*70}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*70}")

    print("\nBest configs to apply:")
    for product, params in best_config.items():
        print(f"  {product}: {params}")

    # Save results
    os.makedirs("grid_results", exist_ok=True)
    timestamp = int(time.time())
    result_path = f"grid_results/search_{timestamp}.json"
    save_data = {
        "timestamp": timestamp,
        "seeds": args.seeds,
        "best_config": best_config,
        "results": {
            product: [
                {"params": r["params"], "mean_pnl": float(r["mean_pnl"]),
                 "sharpe": float(r.get("sharpe", 0)), "spread": float(r["spread"]),
                 "worst_day_ok": bool(r.get("worst_day_ok", True))}
                for r in results[:10]
            ]
            for product, results in all_results.items()
        },
    }
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
