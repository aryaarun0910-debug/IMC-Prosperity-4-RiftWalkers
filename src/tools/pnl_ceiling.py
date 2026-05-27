"""
pnl_ceiling.py — Information-theoretic max PnL per product.

For each product, computes:
  1. Theoretical max (oracle with perfect foresight + position limit)
  2. Actual PnL from server log
  3. Headroom % (how much juice left)

Tells you: "ASH at 98% of ceiling → stop optimizing" vs "KELP at 12% → 3x juice left"

Usage:
    py -3.12 tools/pnl_ceiling.py data/r1/server_logs/220150.log
    py -3.12 tools/pnl_ceiling.py --dir data/r1/server_logs --latest

Strategy:
  Oracle max PnL = sum over ticks of (next_mid - cur_mid) * limit
    with position constraint [-limit, +limit]
  i.e. "what would a trader with perfect foresight + limit L earn?"
"""

import json
import os
import sys
import argparse
from collections import defaultdict

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)

LIMITS = {
    'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50,
    'CROISSANTS': 250, 'JAMS': 350, 'DJEMBES': 60,
    'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100,
    'MAGNIFICENT_MACARONS': 75,
    'VOLCANIC_ROCK': 400,
    'VOLCANIC_ROCK_VOUCHER_9500': 200, 'VOLCANIC_ROCK_VOUCHER_9750': 200,
    'VOLCANIC_ROCK_VOUCHER_10000': 200, 'VOLCANIC_ROCK_VOUCHER_10250': 200,
    'VOLCANIC_ROCK_VOUCHER_10500': 200,
    'ASH_COATED_OSMIUM': 80, 'INTARIAN_PEPPER_ROOT': 80,
    'EMERALDS': 50, 'TOMATOES': 50,
}


def load_log(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def parse_activities(csv_str):
    lines = csv_str.split('\n')
    if not lines:
        return {}
    header = [h.strip() for h in lines[0].split(';')]
    products = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(';')
        row = {}
        for i, h in enumerate(header):
            if i < len(parts):
                val = parts[i].strip()
                if val == '':
                    row[h] = None
                else:
                    try:
                        row[h] = float(val) if '.' in val else int(val)
                    except ValueError:
                        row[h] = val
        prod = row.get('product')
        if not prod:
            continue
        products.setdefault(prod, []).append(row)
    return products


def oracle_max_pnl(mids, spreads, limit, use_spread_filter=True):
    """Max take-based PnL with tick-ahead foresight, paying half-spread per flip.
    DP state: position at each tick. Optimal action = full limit toward future direction
    IF the edge exceeds half-spread (otherwise hold).

    This is the THEORETICAL CEILING for an aggressive-taking strategy:
    - No passive MM edge (that's the separate passive_max_pnl)
    - Perfect tick-ahead knowledge
    - Real fill costs (half-spread per unit moved)"""
    if len(mids) < 2:
        return 0.0
    pnl = 0.0
    pos = 0
    for i in range(len(mids) - 1):
        fwd = mids[i+1] - mids[i]
        half_spread = (spreads[i] if i < len(spreads) else 4) / 2.0

        # Decision rule: target position based on edge vs cost
        if use_spread_filter:
            threshold = half_spread  # only flip if edge > cost to flip
        else:
            threshold = 0.0

        if fwd > threshold:
            target = limit
        elif fwd < -threshold:
            target = -limit
        else:
            target = pos  # hold

        delta = target - pos
        # Pay half-spread to move position
        pnl -= abs(delta) * half_spread
        pos = target
        # Mark-to-market: position accrues on next tick's move
        pnl += pos * fwd

    # End-of-round: flatten at spread cost
    if pos != 0:
        final_half_spread = (spreads[-1] if spreads else 4) / 2.0
        pnl -= abs(pos) * final_half_spread

    return pnl


def realistic_max_pnl(mids, spreads, limit):
    """Same as oracle, with spread filter (only flip when worth it).
    This IS the realistic ceiling — the oracle never beats this by much."""
    return oracle_max_pnl(mids, spreads, limit, use_spread_filter=True)


def passive_max_pnl(mids, spreads, limit, fill_rate=0.17):
    """Max from passive MM with perfect side-selection.
    Fill rate 0.17 matches empirical observation (17% at BBO).
    Strategy: quote the side market is moving AWAY from (i.e. bid when going up).
    We collect half-spread per fill."""
    if len(mids) < 2:
        return 0.0
    pnl = 0.0
    pos = 0
    for i in range(len(mids) - 1):
        fwd_move = mids[i+1] - mids[i]
        half_spread = (spreads[i] if i < len(spreads) else 4) / 2.0
        if fwd_move > 0 and pos < limit:
            fill = min(limit - pos, max(1, int(limit * fill_rate)))
            pnl += fill * half_spread  # earned half-spread
            pos += fill
        elif fwd_move < 0 and pos > -limit:
            fill = min(limit + pos, max(1, int(limit * fill_rate)))
            pnl += fill * half_spread
            pos -= fill
        pnl += pos * fwd_move  # mark-to-market appreciation
    return pnl


def analyze_log(log_path):
    log = load_log(log_path)
    products = parse_activities(log.get('activitiesLog', ''))

    print(f"\nLog: {os.path.basename(log_path)}")
    print("=" * 90)
    header = f"{'Product':<28} {'Actual':>8} {'Oracle':>10} {'Realistic':>10} {'Passive':>10} {'% Real':>7}"
    print(header)
    print("-" * 90)

    total_actual = 0
    total_realistic = 0
    summary = []

    for prod in sorted(products.keys()):
        rows = products[prod]
        # Only use rows where bid/ask/mid are all present AND non-zero
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

        pct_real = (actual / realistic * 100) if realistic > 0 else (100 if actual > 0 else 0)

        print(f"{prod:<28} {actual:>8.0f} {oracle:>10.0f} {realistic:>10.0f} {passive:>10.0f} {pct_real:>6.1f}%")

        total_actual += actual
        total_realistic += realistic

        summary.append({
            'product': prod,
            'actual': actual,
            'oracle': oracle,
            'realistic': realistic,
            'passive': passive,
            'pct_realistic': pct_real,
            'headroom': realistic - actual,
        })

    print("-" * 90)
    pct_total = (total_actual / total_realistic * 100) if total_realistic > 0 else 0
    print(f"{'TOTAL':<28} {total_actual:>8.0f} {'':>10} {total_realistic:>10.0f} {'':>10} {pct_total:>6.1f}%")

    # Verdicts
    print()
    print("VERDICTS:")
    for s in sorted(summary, key=lambda x: -x['headroom']):
        if s['pct_realistic'] >= 90:
            print(f"  [MAXED] {s['product']:<28} — {s['pct_realistic']:.0f}% of ceiling, stop optimizing")
        elif s['pct_realistic'] >= 60:
            print(f"  [OK]    {s['product']:<28} — {s['pct_realistic']:.0f}% of ceiling, minor gains possible (+{s['headroom']:.0f})")
        elif s['pct_realistic'] >= 20:
            print(f"  [JUICE] {s['product']:<28} — {s['pct_realistic']:.0f}% of ceiling, {s['headroom']:.0f} PnL left on table")
        elif s['pct_realistic'] >= 0:
            print(f"  [FIX!]  {s['product']:<28} — {s['pct_realistic']:.0f}% of ceiling, {s['headroom']:.0f} PnL missing — investigate")
        else:
            print(f"  [LOSS]  {s['product']:<28} — LOSING money, strategy broken, disable it")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Per-product PnL ceiling analysis")
    parser.add_argument("log", nargs="?", help="Server log file path")
    parser.add_argument("--dir", type=str, help="Directory of logs; use with --latest")
    parser.add_argument("--latest", action="store_true", help="Use latest log in --dir")
    args = parser.parse_args()

    if args.dir and args.latest:
        logs = [f for f in os.listdir(args.dir) if f.endswith('.log')]
        if not logs:
            print(f"No logs in {args.dir}")
            sys.exit(1)
        logs.sort()
        path = os.path.join(args.dir, logs[-1])
    elif args.log:
        path = args.log
    else:
        parser.print_help()
        sys.exit(1)

    analyze_log(path)


if __name__ == "__main__":
    main()
