"""
Bot Fingerprinter — Reverse-engineer the simulator's data-generating process.

This is the practical implementation of the POMDP → MDP reduction:
1. Takes given price/trade CSVs (the "observations")
2. Identifies hidden bot policies from observable patterns
3. Estimates the true fair value process
4. Computes the THEORETICAL MAXIMUM PnL (information-theoretic bound)
5. Identifies which signals carry exploitable information vs noise

The output tells you: "Here is the DGP. Here are the exploitable patterns.
Here is the maximum PnL you can extract. Here is which strategy is optimal."

Usage:
    python analysis/bot_fingerprinter.py --prices data/r1/*.csv --trades data/r1/*.csv
"""

import csv
import os
import sys
import math
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_price_data(price_files):
    """Load price CSVs into structured data."""
    data = []
    for pf in price_files:
        with open(pf, newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                cleaned = {k.strip().lower(): v.strip() for k, v in row.items()}
                data.append(cleaned)
    data.sort(key=lambda r: (int(r.get("day", 0)), int(r.get("timestamp", 0))))
    return data


def load_trade_data(trade_files):
    """Load trade CSVs into structured data."""
    data = []
    for tf in trade_files:
        with open(tf, newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                cleaned = {k.strip().lower(): v.strip() for k, v in row.items()}
                data.append(cleaned)
    return data


def extract_product_series(price_data, product):
    """Extract time series for a single product."""
    series = []
    for row in price_data:
        if row.get("product") != product:
            continue
        try:
            bp1 = float(row.get("bid_price_1", 0))
            ap1 = float(row.get("ask_price_1", 0))
            bv1 = float(row.get("bid_volume_1", 0))
            av1 = float(row.get("ask_volume_1", 0))
            if bp1 > 0 and ap1 > 0:
                mid = (bp1 + ap1) / 2.0
                spread = ap1 - bp1
                microprice = (ap1 * bv1 + bp1 * abs(av1)) / (bv1 + abs(av1)) if (bv1 + abs(av1)) > 0 else mid
                series.append({
                    "day": int(row.get("day", 0)),
                    "ts": int(row.get("timestamp", 0)),
                    "mid": mid,
                    "spread": spread,
                    "bid": bp1,
                    "ask": ap1,
                    "bid_vol": bv1,
                    "ask_vol": abs(av1),
                    "microprice": microprice,
                })
        except (ValueError, TypeError):
            continue
    return series


def extract_product_trades(trade_data, product):
    """Extract trades for a single product."""
    trades = []
    for row in trade_data:
        if row.get("symbol") != product:
            continue
        try:
            trades.append({
                "ts": int(row.get("timestamp", 0)),
                "price": float(row.get("price", 0)),
                "qty": int(row.get("quantity", 0)),
                "buyer": row.get("buyer", ""),
                "seller": row.get("seller", ""),
            })
        except (ValueError, TypeError):
            continue
    return trades


# ============================================================
# DGP ANALYSIS TOOLS
# ============================================================

def analyze_price_process(series):
    """Characterize the price process: drift, volatility, autocorrelation, mean-reversion."""
    if len(series) < 50:
        return {"error": "insufficient data"}

    mids = [s["mid"] for s in series]
    n = len(mids)

    # Returns
    returns = [mids[i] - mids[i-1] for i in range(1, n)]

    # Basic stats
    mean_ret = sum(returns) / len(returns)
    var_ret = sum((r - mean_ret)**2 for r in returns) / (len(returns) - 1)
    vol = var_ret ** 0.5

    # Autocorrelation at lag 1-5
    autocorr = {}
    for lag in [1, 2, 3, 5, 10, 20]:
        if lag >= len(returns):
            break
        n_ac = len(returns) - lag
        mean_r = sum(returns[:n_ac]) / n_ac
        var_r = sum((returns[i] - mean_r)**2 for i in range(n_ac)) / n_ac
        if var_r < 1e-10:
            autocorr[lag] = 0.0
            continue
        cov = sum((returns[i] - mean_r) * (returns[i+lag] - mean_r)
                  for i in range(n_ac)) / n_ac
        autocorr[lag] = cov / var_r

    # Half-life of mean reversion (from AR(1) coefficient)
    ar1 = autocorr.get(1, 0)
    if ar1 < 0:
        half_life = -math.log(2) / math.log(abs(ar1)) if abs(ar1) > 0.01 else float('inf')
    else:
        half_life = float('inf')  # trending, no mean reversion

    # Spread stats
    spreads = [s["spread"] for s in series]
    mean_spread = sum(spreads) / len(spreads)
    min_spread = min(spreads)
    max_spread = max(spreads)
    spread_at_min_pct = sum(1 for s in spreads if s <= min_spread + 1) / len(spreads)

    # Price range
    price_range = max(mids) - min(mids)

    # Hurst exponent estimate (R/S analysis, simplified)
    # H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk
    def rs_hurst(data, min_window=10):
        n = len(data)
        windows = [w for w in [10, 20, 50, 100, 200] if w < n // 2]
        if len(windows) < 2:
            return 0.5
        log_rs = []
        log_n = []
        for w in windows:
            rs_values = []
            for start in range(0, n - w, w):
                chunk = data[start:start+w]
                mean_c = sum(chunk) / len(chunk)
                deviations = [x - mean_c for x in chunk]
                cumulative = []
                s = 0
                for d in deviations:
                    s += d
                    cumulative.append(s)
                R = max(cumulative) - min(cumulative)
                S = (sum(d**2 for d in deviations) / len(deviations)) ** 0.5
                if S > 0:
                    rs_values.append(R / S)
            if rs_values:
                log_rs.append(math.log(sum(rs_values)/len(rs_values)))
                log_n.append(math.log(w))

        if len(log_rs) < 2:
            return 0.5
        # Linear regression slope = Hurst exponent
        n_pts = len(log_rs)
        mean_x = sum(log_n) / n_pts
        mean_y = sum(log_rs) / n_pts
        num = sum((log_n[i] - mean_x) * (log_rs[i] - mean_y) for i in range(n_pts))
        den = sum((log_n[i] - mean_x)**2 for i in range(n_pts))
        return num / den if den > 0 else 0.5

    hurst = rs_hurst(returns)

    # Classify
    if abs(autocorr.get(1, 0)) < 0.05 and 0.45 < hurst < 0.55:
        process_type = "random_walk"
    elif autocorr.get(1, 0) < -0.1 or hurst < 0.45:
        process_type = "mean_reverting"
    elif autocorr.get(1, 0) > 0.1 or hurst > 0.55:
        process_type = "trending"
    else:
        process_type = "mixed"

    return {
        "process_type": process_type,
        "volatility": round(vol, 4),
        "mean_return": round(mean_ret, 6),
        "autocorrelation": {k: round(v, 4) for k, v in autocorr.items()},
        "hurst_exponent": round(hurst, 3),
        "half_life": round(half_life, 1) if half_life != float('inf') else "inf",
        "mean_spread": round(mean_spread, 2),
        "min_spread": min_spread,
        "spread_at_min_pct": round(spread_at_min_pct * 100, 1),
        "price_range": round(price_range, 2),
        "n_observations": n,
    }


def analyze_trade_informativeness(series, trades):
    """Measure how much trades predict future mid changes.
    This reveals whether there's an informed trader (Olivia-type bot)."""
    if len(series) < 50 or len(trades) < 20:
        return {"error": "insufficient data"}

    # Build timestamp -> mid map
    ts_to_mid = {}
    ts_list = []
    for s in series:
        key = (s["day"], s["ts"])
        ts_to_mid[key] = s["mid"]
        ts_list.append(key)

    # For each trade, compute: trade_direction * future_mid_change
    horizons = [1, 5, 10, 20, 50]
    correlations = {}

    for horizon in horizons:
        correct = 0
        total = 0
        weighted_correct = 0
        total_weight = 0

        for trade in trades:
            # Find this trade's timestamp index in the series
            trade_key = None
            for s in series:
                if s["ts"] == trade["ts"]:
                    trade_key = (s["day"], s["ts"])
                    break
            if trade_key is None or trade_key not in ts_to_mid:
                continue

            idx = None
            for i, k in enumerate(ts_list):
                if k == trade_key:
                    idx = i
                    break
            if idx is None or idx + horizon >= len(ts_list):
                continue

            current_mid = ts_to_mid[trade_key]
            future_mid = ts_to_mid[ts_list[idx + horizon]]
            mid_change = future_mid - current_mid

            # Trade direction: buy if price >= mid, sell if price < mid
            direction = 1 if trade["price"] >= current_mid else -1
            qty = trade["qty"]

            if direction * mid_change > 0:
                correct += 1
                weighted_correct += qty
            total += 1
            total_weight += qty

        hit_rate = correct / total if total > 0 else 0.5
        weighted_hr = weighted_correct / total_weight if total_weight > 0 else 0.5

        correlations[horizon] = {
            "hit_rate": round(hit_rate, 4),
            "weighted_hit_rate": round(weighted_hr, 4),
            "n_trades": total,
            "is_informed": hit_rate > 0.55,
        }

    # Detect specific bot patterns
    buyer_stats = defaultdict(lambda: {"buys": 0, "sells": 0, "total_qty": 0})
    for t in trades:
        if t["buyer"]:
            buyer_stats[t["buyer"]]["buys"] += 1
            buyer_stats[t["buyer"]]["total_qty"] += t["qty"]
        if t["seller"]:
            buyer_stats[t["seller"]]["sells"] += 1
            buyer_stats[t["seller"]]["total_qty"] += t["qty"]

    # Lot size analysis (bots trade in fixed lot sizes)
    lot_sizes = defaultdict(int)
    for t in trades:
        lot_sizes[t["qty"]] += 1
    top_lots = sorted(lot_sizes.items(), key=lambda x: -x[1])[:5]

    return {
        "trade_prediction": correlations,
        "n_total_trades": len(trades),
        "top_lot_sizes": top_lots,
        "trader_activity": {name: stats for name, stats in
                           sorted(buyer_stats.items(), key=lambda x: -x[1]["total_qty"])[:10]},
    }


def compute_theoretical_max_pnl(series, position_limit=50):
    """Compute the maximum PnL achievable with perfect information.
    This is the UPPER BOUND — no strategy can beat this.

    Method: at each tick, if we knew the next mid-price change,
    we'd hold +limit if price goes up, -limit if price goes down.
    Profit = limit * |mid_change| at each tick."""
    if len(series) < 2:
        return {"error": "insufficient data"}

    mids = [s["mid"] for s in series]

    # Perfect foresight PnL (knows future)
    perfect_pnl = 0.0
    for i in range(1, len(mids)):
        change = mids[i] - mids[i-1]
        perfect_pnl += position_limit * abs(change)

    # Realistic max: account for spread cost (must cross spread to trade)
    spreads = [s["spread"] for s in series]
    mean_spread = sum(spreads) / len(spreads)

    # Each position flip costs spread/2
    flip_cost = mean_spread / 2.0
    n_flips = sum(1 for i in range(2, len(mids))
                  if (mids[i] - mids[i-1]) * (mids[i-1] - mids[i-2]) < 0)

    realistic_max = perfect_pnl - n_flips * flip_cost * position_limit

    # What fraction of this can a good strategy capture?
    # Top teams capture 5-15% of theoretical max
    return {
        "perfect_foresight_pnl": round(perfect_pnl, 0),
        "realistic_max_pnl": round(realistic_max, 0),
        "mean_spread_cost": round(mean_spread, 2),
        "n_position_flips": n_flips,
        "top_team_estimate_5pct": round(realistic_max * 0.05, 0),
        "top_team_estimate_15pct": round(realistic_max * 0.15, 0),
        "position_limit": position_limit,
    }


def recommend_strategy(price_analysis, trade_analysis, max_pnl):
    """Given the DGP analysis, recommend the optimal strategy type."""
    recs = []

    proc = price_analysis.get("process_type", "unknown")
    spread = price_analysis.get("mean_spread", 5)
    hurst = price_analysis.get("hurst_exponent", 0.5)
    ac1 = price_analysis.get("autocorrelation", {}).get(1, 0)

    # Check for informed trading signal
    has_informed = False
    if isinstance(trade_analysis, dict) and "trade_prediction" in trade_analysis:
        for horizon, stats in trade_analysis["trade_prediction"].items():
            if isinstance(stats, dict) and stats.get("is_informed", False):
                has_informed = True
                break

    # Strategy recommendation
    if proc == "mean_reverting":
        recs.append({
            "strategy": "olivia_follow" if has_informed else "generic_mm (MR mode)",
            "reason": f"Mean-reverting process (AC1={ac1:.3f}, Hurst={hurst:.3f})",
            "priority": "HIGH",
            "key_params": {
                "mode": "mr",
                "vol_mr_threshold": 2.0,
                "make_size": 10 if spread < 3 else 20,
            }
        })
    elif proc == "trending":
        recs.append({
            "strategy": "ar_olivia" if has_informed else "generic_mm (OFI mode)",
            "reason": f"Trending process (AC1={ac1:.3f}, Hurst={hurst:.3f})",
            "priority": "HIGH",
            "key_params": {
                "mode": "ofi",
                "ar_order": 5,
                "make_size": 30,
            }
        })
    elif proc == "random_walk":
        recs.append({
            "strategy": "ar_olivia",
            "reason": f"Random walk — pure market making, no prediction",
            "priority": "MEDIUM",
            "key_params": {
                "make_size": 20 if spread < 3 else 30,
            }
        })

    if spread > 8:
        recs.append({
            "strategy": "wide_spread",
            "reason": f"Wide spread ({spread:.1f}) — penny-jump bot quotes",
            "priority": "HIGH" if spread > 12 else "MEDIUM",
        })
    elif spread < 3:
        recs.append({
            "strategy": "tight spread MM",
            "reason": f"Tight spread ({spread:.1f}) — aggressive undercutting",
            "priority": "MEDIUM",
        })

    if has_informed:
        recs.append({
            "strategy": "informed trader following",
            "reason": "Detected informed bot (trade prediction > 55%)",
            "priority": "CRITICAL — this is your biggest edge",
        })

    # PnL target
    if isinstance(max_pnl, dict):
        est_5 = max_pnl.get("top_team_estimate_5pct", 0)
        est_15 = max_pnl.get("top_team_estimate_15pct", 0)
        recs.append({
            "pnl_target": f"{est_5:.0f} - {est_15:.0f} seashells",
            "reason": "5-15% of theoretical max (top team capture rate)",
        })

    return recs


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Bot Fingerprinter — DGP Analysis")
    parser.add_argument("--prices", nargs="+", required=True)
    parser.add_argument("--trades", nargs="+", default=[])
    parser.add_argument("--limit", type=int, default=50, help="Position limit")
    args = parser.parse_args()

    # Expand globs
    price_files = []
    for p in args.prices:
        expanded = sorted(glob.glob(p))
        price_files.extend(expanded if expanded else [p])

    trade_files = []
    for t in args.trades:
        expanded = sorted(glob.glob(t))
        trade_files.extend(expanded if expanded else [t])

    print("=" * 70)
    print("BOT FINGERPRINTER — Data Generating Process Analysis")
    print("=" * 70)

    price_data = load_price_data(price_files)
    trade_data = load_trade_data(trade_files) if trade_files else []

    # Discover products
    products = sorted(set(r.get("product", "") for r in price_data if r.get("product")))
    print(f"Products: {products}\n")

    all_results = {}

    for product in products:
        print(f"\n{'='*60}")
        print(f"  PRODUCT: {product}")
        print(f"{'='*60}")

        series = extract_product_series(price_data, product)
        trades = extract_product_trades(trade_data, product)

        # 1. Price process analysis
        pa = analyze_price_process(series)
        print(f"\n  --- Price Process ---")
        print(f"  Type: {pa.get('process_type', '?')}")
        print(f"  Volatility: {pa.get('volatility', '?')}")
        print(f"  Hurst exponent: {pa.get('hurst_exponent', '?')} "
              f"({'trending' if pa.get('hurst_exponent', 0.5) > 0.55 else 'mean-reverting' if pa.get('hurst_exponent', 0.5) < 0.45 else 'random walk'})")
        print(f"  Autocorrelation: {pa.get('autocorrelation', {})}")
        print(f"  Half-life: {pa.get('half_life', '?')} ticks")
        print(f"  Spread: mean={pa.get('mean_spread', '?')}, min={pa.get('min_spread', '?')}, "
              f"at-min={pa.get('spread_at_min_pct', '?')}%")

        # 2. Trade informativeness
        if trades:
            ta = analyze_trade_informativeness(series, trades)
            print(f"\n  --- Trade Informativeness ---")
            if "trade_prediction" in ta:
                for h, stats in sorted(ta["trade_prediction"].items()):
                    marker = " *** INFORMED ***" if stats.get("is_informed") else ""
                    print(f"  {h}-tick: hit_rate={stats['hit_rate']:.1%} "
                          f"(n={stats['n_trades']}){marker}")
            if "top_lot_sizes" in ta:
                print(f"  Top lot sizes: {ta['top_lot_sizes'][:5]}")
        else:
            ta = {}

        # 3. Theoretical max PnL
        mp = compute_theoretical_max_pnl(series, args.limit)
        print(f"\n  --- Theoretical Maximum PnL ---")
        print(f"  Perfect foresight: {mp.get('perfect_foresight_pnl', '?'):,.0f}")
        print(f"  After spread costs: {mp.get('realistic_max_pnl', '?'):,.0f}")
        print(f"  Top team estimate: {mp.get('top_team_estimate_5pct', '?'):,.0f} - "
              f"{mp.get('top_team_estimate_15pct', '?'):,.0f}")

        # 4. Strategy recommendation
        recs = recommend_strategy(pa, ta, mp)
        print(f"\n  --- Strategy Recommendation ---")
        for r in recs:
            if "strategy" in r:
                print(f"  [{r.get('priority', '?')}] {r['strategy']}: {r['reason']}")
            elif "pnl_target" in r:
                print(f"  TARGET: {r['pnl_target']} ({r['reason']})")

        all_results[product] = {
            "price_process": pa,
            "trade_analysis": ta,
            "max_pnl": mp,
            "recommendations": recs,
        }

    # Save results
    os.makedirs("analysis/fingerprints", exist_ok=True)
    out_path = "analysis/fingerprints/dgp_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
