"""
first_log_intel.py — R2 First-Hour Intelligence Orchestrator

Single command that, given R2 server log + CSVs, outputs everything you need
to dominate R2 in the first hour:

  1. Bots ranked by hit-rate (auto-promotes informed traders for trader.py)
  2. Per-product classification with confidence (via rapid_deploy)
  3. Theoretical max PnL vs actual — headroom per product (via pnl_ceiling)
  4. Top specific actions ranked by expected delta

This is the R2 war-room summary. Run it the moment R2 data drops.

Usage:
    # Full analysis (best):
    py -3.12 tools/first_log_intel.py \\
        --log data/r2/server_logs/<latest>.log \\
        --prices data/r2/prices/*.csv \\
        --trades data/r2/trades/*.csv

    # Or just --dir data/r2 to auto-find everything:
    py -3.12 tools/first_log_intel.py --dir data/r2

    # Write the combined intel JSON for trader.py to consume:
    py -3.12 tools/first_log_intel.py --dir data/r2 --output data/r2_intel.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _TOOLS_DIR)

# Reuse existing tools
from pnl_ceiling import load_log, parse_activities, oracle_max_pnl, realistic_max_pnl, passive_max_pnl, LIMITS
from bot_watchlist import load_trades, load_prices, score_traders, MIN_TRADES, MIN_HIT_RATE, MIN_AVG_QTY


def find_files(root_dir):
    """Auto-discover log + price + trade files under a round dir."""
    files = {'logs': [], 'prices': [], 'trades': []}
    # Server logs
    for pat in ['server_logs/*.log', '*.log']:
        files['logs'] += glob.glob(os.path.join(root_dir, pat))
    # Prices
    for pat in ['prices/*.csv', 'prices_*.csv']:
        files['prices'] += glob.glob(os.path.join(root_dir, pat))
    # Trades
    for pat in ['trades/*.csv', 'trades_*.csv']:
        files['trades'] += glob.glob(os.path.join(root_dir, pat))
    # Dedup + sort
    for k in files:
        files[k] = sorted(set(files[k]))
    return files


def analyze_pnl_ceiling(log_path):
    """Returns [{product, actual, oracle, realistic, passive, pct_realistic, headroom}]."""
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        log = load_log(log_path)
        products = parse_activities(log.get('activitiesLog', ''))
    except Exception as e:
        print(f"  WARN: could not parse log {log_path}: {e}")
        return []

    summary = []
    for prod in sorted(products.keys()):
        rows = products[prod]
        clean = [r for r in rows
                 if r.get('mid_price') and r.get('bid_price_1') and r.get('ask_price_1')
                 and r['mid_price'] > 0]
        mids = [r['mid_price'] for r in clean]
        spreads = [r['ask_price_1'] - r['bid_price_1'] for r in clean]
        pnls = [r['profit_and_loss'] for r in rows if r.get('profit_and_loss') is not None]
        if len(mids) < 10:
            continue
        actual = pnls[-1] if pnls else 0
        limit = LIMITS.get(prod, 50)
        oracle = oracle_max_pnl(mids, spreads, limit, use_spread_filter=False)
        realistic = realistic_max_pnl(mids, spreads, limit)
        passive = passive_max_pnl(mids, spreads, limit)
        pct = (actual / realistic * 100) if realistic > 0 else (100 if actual > 0 else 0)
        summary.append({
            'product': prod,
            'actual': actual,
            'oracle': oracle,
            'realistic': realistic,
            'passive': passive,
            'pct_realistic': pct,
            'headroom': realistic - actual,
            'limit': limit,
            'n_ticks': len(mids),
        })
    return summary


def analyze_bots(trade_files, price_files):
    """Returns {promoted: [names], details: {name: stats}, all_scored: {name: stats}}."""
    if not trade_files:
        return {'promoted': [], 'details': {}, 'all_scored': {}}
    trades = load_trades(trade_files)
    mids = load_prices(price_files) if price_files else {}
    stats = score_traders(trades, mids)
    promoted = {}
    for name, s in stats.items():
        if (s['resolved'] >= MIN_TRADES and
            s['hit_rate'] >= MIN_HIT_RATE and
            s['avg_qty'] >= MIN_AVG_QTY):
            promoted[name] = {
                'hit_rate': round(s['hit_rate'], 3),
                'avg_qty': round(s['avg_qty'], 1),
                'resolved': s['resolved'],
                'products': s['products'],
            }
    return {
        'promoted': list(promoted.keys()),
        'details': promoted,
        'all_scored': stats,
    }


def classify_products(price_files):
    """Lightweight per-product archetype heuristic from price CSVs."""
    if not price_files:
        return {}
    try:
        from rapid_deploy import load_price_data
    except ImportError:
        return {}
    try:
        rows = load_price_data(price_files)
    except Exception as e:
        print(f"  WARN: classification failed: {e}")
        return {}

    by_prod = defaultdict(list)
    for r in rows:
        p = r.get('product')
        if p:
            by_prod[p].append(r)

    diagnoses = {}
    for prod, prod_rows in by_prod.items():
        mids = [r.get('mid_price') for r in prod_rows if r.get('mid_price')]
        bids = [r.get('bid_price_1') for r in prod_rows if r.get('bid_price_1')]
        asks = [r.get('ask_price_1') for r in prod_rows if r.get('ask_price_1')]
        if not mids or len(mids) < 10:
            diagnoses[prod] = {'archetype': 'insufficient_data', 'confidence': 0.0,
                               'strategy': 'generic_mm', 'spread': 0, 'volatility': 0,
                               'mean_mid': 0}
            continue
        mean_mid = sum(mids) / len(mids)
        std_mid = (sum((m - mean_mid) ** 2 for m in mids) / len(mids)) ** 0.5
        spreads = [a - b for a, b in zip(asks, bids) if a and b and a > b]
        median_spread = sorted(spreads)[len(spreads) // 2] if spreads else 0

        # Heuristic archetype
        nearest_round = round(mean_mid / 100) * 100
        round_dist = abs(mean_mid - nearest_round) / max(mean_mid, 1)
        vol_ratio = std_mid / max(mean_mid, 1)

        if round_dist < 0.001 and vol_ratio < 0.005:
            archetype, strategy, conf = 'pegged_round', 'pegged', 0.85
        elif 'VOUCHER' in prod.upper() or 'CALL' in prod.upper() or 'PUT' in prod.upper():
            archetype, strategy, conf = 'option', 'options', 0.9
        elif 'BASKET' in prod.upper() or 'PICNIC' in prod.upper():
            archetype, strategy, conf = 'basket', 'basket_arb', 0.9
        elif 'MACARON' in prod.upper() or 'CONVERSION' in prod.upper():
            archetype, strategy, conf = 'conversion', 'conversion_arb', 0.85
        elif median_spread >= 8:
            archetype, strategy, conf = 'wide_spread', 'wide_spread', 0.7
        elif vol_ratio > 0.02:
            archetype, strategy, conf = 'volatile', 'ar_olivia', 0.6
        else:
            archetype, strategy, conf = 'stationary_mm', 'ar_olivia', 0.55

        diagnoses[prod] = {
            'archetype': archetype,
            'confidence': conf,
            'strategy': strategy,
            'spread': round(median_spread, 2),
            'volatility': round(std_mid, 2),
            'mean_mid': round(mean_mid, 2),
        }
    return diagnoses


def rank_actions(pnl_ceiling, classifications, bot_intel):
    """Generate top actionable items, ranked by expected PnL delta."""
    actions = []

    # 1. Headroom opportunities — big gap between actual and realistic
    for s in pnl_ceiling:
        gap = s['headroom']
        if gap > 500 and s['pct_realistic'] < 60:
            actions.append({
                'priority': gap,
                'product': s['product'],
                'type': 'headroom',
                'action': f"Investigate {s['product']}: {s['pct_realistic']:.0f}% of ceiling, {gap:.0f} PnL on table",
                'expected_delta': gap * 0.3,  # conservative: capture 30%
            })
        elif s['actual'] < 0:
            actions.append({
                'priority': abs(s['actual']) + 10000,  # losses top priority
                'product': s['product'],
                'type': 'loss',
                'action': f"DISABLE {s['product']}: losing {s['actual']:.0f} — strategy broken",
                'expected_delta': abs(s['actual']),
            })

    # 2. Unknown products needing classification
    known_in_ceiling = {s['product'] for s in pnl_ceiling}
    for prod, cls in classifications.items():
        if cls['archetype'] in ('unknown', 'error'):
            actions.append({
                'priority': 500,
                'product': prod,
                'type': 'classify',
                'action': f"UNCLASSIFIED {prod}: falling back to generic_mm — manual review needed",
                'expected_delta': 0,
            })

    # 3. New informed traders — add to watchlist
    for name in bot_intel.get('promoted', []):
        d = bot_intel['details'][name]
        if name.lower() != 'olivia':  # Olivia already hardcoded
            actions.append({
                'priority': 2000,  # high — new alpha source
                'product': ','.join(d['products'][:3]),
                'type': 'informed_bot',
                'action': f"PROMOTE bot '{name}': {d['hit_rate']*100:.0f}% hit rate on {d['products']}",
                'expected_delta': 1000,  # rough estimate
            })

    return sorted(actions, key=lambda a: -a['priority'])


def print_report(log_path, pnl_ceiling, bot_intel, classifications, actions):
    print("\n" + "=" * 100)
    print(f"  R2 FIRST-HOUR INTEL REPORT")
    if log_path:
        print(f"  Log: {os.path.basename(log_path)}")
    print("=" * 100)

    # Section 1: PnL Ceiling
    if pnl_ceiling:
        print("\n[1] PnL CEILING ANALYSIS")
        print("-" * 100)
        print(f"{'Product':<28} {'Actual':>9} {'Realistic':>10} {'Passive':>10} {'% Real':>8} {'Headroom':>10}")
        print("-" * 100)
        total_actual = 0
        total_realistic = 0
        for s in sorted(pnl_ceiling, key=lambda x: -x['headroom']):
            print(f"{s['product']:<28} {s['actual']:>9.0f} {s['realistic']:>10.0f} "
                  f"{s['passive']:>10.0f} {s['pct_realistic']:>7.1f}% {s['headroom']:>10.0f}")
            total_actual += s['actual']
            total_realistic += s['realistic']
        print("-" * 100)
        pct_total = (total_actual / total_realistic * 100) if total_realistic > 0 else 0
        print(f"{'TOTAL':<28} {total_actual:>9.0f} {total_realistic:>10.0f} "
              f"{'':>10} {pct_total:>7.1f}% {total_realistic - total_actual:>10.0f}")

    # Section 2: Bot Intel
    print("\n[2] BOT HIT-RATE RANKING")
    print("-" * 100)
    if bot_intel.get('all_scored'):
        print(f"{'Bot':<16} {'Resolved':>9} {'HitRate':>8} {'AvgQty':>7} {'Products'}")
        print("-" * 100)
        ranked = sorted(bot_intel['all_scored'].items(), key=lambda x: -x[1]['hit_rate'])
        for name, s in ranked[:10]:
            prods = ",".join(s['products'][:3])
            if len(s['products']) > 3:
                prods += f"+{len(s['products'])-3}"
            flag = "[INFORMED]" if name in bot_intel.get('promoted', []) else ""
            print(f"{name:<16} {s['resolved']:>9} {s['hit_rate']:>7.1%} "
                  f"{s['avg_qty']:>7.1f} {prods} {flag}")
        if bot_intel.get('promoted'):
            print(f"\nPROMOTED INFORMED TRADERS: {', '.join(bot_intel['promoted'])}")
        else:
            print("\nNo bots met promotion criteria (may be first few ticks — retry after full day).")
    else:
        print("(no trade data — R1-R4 trades are anonymized, only R5 exposes names)")

    # Section 3: Classifications
    if classifications:
        print("\n[3] PRODUCT CLASSIFICATIONS")
        print("-" * 100)
        print(f"{'Product':<28} {'Archetype':<22} {'Strategy':<18} {'Conf':>6}")
        print("-" * 100)
        for prod, cls in sorted(classifications.items()):
            print(f"{prod:<28} {cls['archetype']:<22} {cls['strategy']:<18} "
                  f"{cls['confidence']:>5.1%}")

    # Section 4: Top Actions
    print("\n[4] TOP ACTIONS (ranked by expected PnL delta)")
    print("-" * 100)
    if actions:
        for i, a in enumerate(actions[:10], 1):
            print(f"  {i}. [+{a['expected_delta']:.0f}] {a['action']}")
    else:
        print("  (no high-priority actions — pipeline running at ceiling)")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="R2 first-hour intelligence orchestrator")
    parser.add_argument("--log", type=str, help="Server log file path")
    parser.add_argument("--prices", nargs="*", help="Price CSV files")
    parser.add_argument("--trades", nargs="*", help="Trade CSV files")
    parser.add_argument("--dir", type=str, help="Round dir (auto-finds log/prices/trades)")
    parser.add_argument("--output", type=str, help="Write combined intel JSON to this path")
    args = parser.parse_args()

    log_path = args.log
    price_files = args.prices or []
    trade_files = args.trades or []

    if args.dir:
        found = find_files(args.dir)
        if found['logs'] and not log_path:
            log_path = found['logs'][-1]  # latest
            print(f"Auto-found log: {log_path}")
        if found['prices'] and not price_files:
            price_files = found['prices']
            print(f"Auto-found {len(price_files)} price CSVs")
        if found['trades'] and not trade_files:
            trade_files = found['trades']
            print(f"Auto-found {len(trade_files)} trade CSVs")

    if not log_path and not price_files and not trade_files:
        parser.print_help()
        print("\nERROR: need --log, --prices/--trades, or --dir")
        sys.exit(1)

    print("\nAnalyzing PnL ceiling...")
    pnl_ceiling = analyze_pnl_ceiling(log_path)

    print("Analyzing bots...")
    bot_intel = analyze_bots(trade_files, price_files)

    print("Classifying products...")
    classifications = classify_products(price_files)

    actions = rank_actions(pnl_ceiling, classifications, bot_intel)

    print_report(log_path, pnl_ceiling, bot_intel, classifications, actions)

    # Write combined intel JSON
    out_path = args.output
    if not out_path:
        out_dir = args.dir if args.dir else os.path.join(_ROOT_DIR, "data")
        out_path = os.path.join(out_dir, "intel.json")
    intel = {
        'log': os.path.basename(log_path) if log_path else None,
        'pnl_ceiling': pnl_ceiling,
        'bots': {
            'promoted': bot_intel.get('promoted', []),
            'details': bot_intel.get('details', {}),
        },
        'classifications': classifications,
        'actions': actions,
        'informed_names': bot_intel.get('promoted', []),  # for trader.py consumption
    }
    # Make JSON-serializable
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if not k.startswith('_')}
        if isinstance(obj, (list, tuple)):
            return [_clean(x) for x in obj]
        if isinstance(obj, set):
            return sorted(obj)
        return obj
    # Strip non-serializable from all_scored
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_clean(intel), f, indent=2, default=str)
    print(f"\nIntel JSON written: {out_path}")
    print(f"  trader.py can load via: json.load(...)['informed_names']")


if __name__ == "__main__":
    main()
