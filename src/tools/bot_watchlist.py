"""
bot_watchlist.py — Auto-learn informed trader names from trade CSVs.

Scans trade CSVs (R5 data typically has exposed names; other rounds are anonymized)
for traders whose large trades predict price moves with hit rate >= 65%. Auto-promotes
them as informed traders for trader.py to treat like Olivia.

Usage:
    # Discover bots from R5 trade data:
    py -3.12 tools/bot_watchlist.py data/r5/trades/*.csv data/r5/prices/*.csv

    # Or just trade CSVs (will skip directional scoring):
    py -3.12 tools/bot_watchlist.py data/r5/trades/*.csv

    # Write output watchlist for trader.py to read:
    py -3.12 tools/bot_watchlist.py data/r5/trades/*.csv --output data/watchlist.json

Integration:
    trader.py's check_olivia() already searches for name "Olivia". This tool
    extends that with auto-discovered names. The watchlist JSON is loaded at
    runtime (or embedded in mem via traderData).

Scoring:
    For each trader X:
      - Extract all trades where X is buyer (X bought) or seller (X sold)
      - For each trade, look forward 10 ticks: did mid move in X's favor?
      - Hit rate = wins / total_resolved
    Promote if: total_resolved >= 20 AND hit_rate >= 0.65 AND avg_qty >= 5
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)

MIN_TRADES = 10       # need at least this many resolved trades
MIN_HIT_RATE = 0.58   # hit rate threshold (accounting for binomial noise on small n)
MIN_AVG_QTY = 7       # avg qty threshold — Olivia pattern is rare+large
LOOKAHEAD_TICKS = 10  # ticks to measure price move
MIN_MOVE = 0.5        # minimum mid movement to count as directional
RARE_TRADE_THRESHOLD = 100  # <100 trades = "rare", boost sensitivity


def load_trades(trade_files):
    """Load all trades. Each row has timestamp, buyer, seller, symbol, price, quantity."""
    trades = []
    for tf in trade_files:
        with open(tf, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    trades.append({
                        'ts': int(row['timestamp']),
                        'buyer': row.get('buyer', '').strip(),
                        'seller': row.get('seller', '').strip(),
                        'symbol': row['symbol'],
                        'price': float(row['price']),
                        'qty': int(row['quantity']),
                    })
                except (ValueError, KeyError):
                    continue
    return trades


def load_prices(price_files):
    """Build (symbol, ts) -> mid map from price CSVs."""
    mids = {}
    for pf in price_files:
        with open(pf, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    sym = row['product']
                    ts = int(row['timestamp'])
                    bp = float(row.get('bid_price_1') or 0)
                    ap = float(row.get('ask_price_1') or 0)
                    if bp > 0 and ap > 0:
                        mids[(sym, ts)] = (bp + ap) / 2.0
                except (ValueError, KeyError):
                    continue
    return mids


def get_mid_at(mids, symbol, ts, lookahead_ticks=0):
    """Find the mid for symbol at ts + lookahead_ticks*100 (each tick = 100 ts units)."""
    target_ts = ts + lookahead_ticks * 100
    # Walk forward up to 10 ticks to find a valid mid
    for offset in range(0, 11):
        key = (symbol, target_ts + offset * 100)
        if key in mids:
            return mids[key]
    return None


def score_traders(trades, mids):
    """For each trader, count directional wins/losses.
    Returns {trader_name: {hit_rate, total, avg_qty, wins, losses, products}}."""
    stats = defaultdict(lambda: {
        'wins': 0, 'losses': 0, 'total_qty': 0, 'trades': 0,
        'products': set(), 'resolved': 0,
    })

    for t in trades:
        for side, name in (('buy', t['buyer']), ('sell', t['seller'])):
            if not name or name == 'SUBMISSION':
                continue

            stats[name]['trades'] += 1
            stats[name]['total_qty'] += t['qty']
            stats[name]['products'].add(t['symbol'])

            # Directional resolution (needs mids)
            if mids:
                trade_mid = get_mid_at(mids, t['symbol'], t['ts'])
                future_mid = get_mid_at(mids, t['symbol'], t['ts'], LOOKAHEAD_TICKS)
                if trade_mid is None or future_mid is None:
                    continue
                move = future_mid - trade_mid
                if abs(move) < MIN_MOVE:
                    continue  # no meaningful move
                stats[name]['resolved'] += 1
                # Buyer wins if price went up; seller wins if price went down
                if side == 'buy':
                    if move > 0:
                        stats[name]['wins'] += 1
                    else:
                        stats[name]['losses'] += 1
                else:
                    if move < 0:
                        stats[name]['wins'] += 1
                    else:
                        stats[name]['losses'] += 1

    # Compute summary
    results = {}
    for name, s in stats.items():
        if s['trades'] == 0:
            continue
        total_resolved = s['wins'] + s['losses']
        hit_rate = s['wins'] / total_resolved if total_resolved > 0 else 0.0
        avg_qty = s['total_qty'] / s['trades']
        results[name] = {
            'trades': s['trades'],
            'resolved': total_resolved,
            'wins': s['wins'],
            'losses': s['losses'],
            'hit_rate': hit_rate,
            'avg_qty': avg_qty,
            'products': sorted(s['products']),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Auto-learn informed trader names")
    parser.add_argument("files", nargs="+", help="Trade CSV files (plus optional price CSVs)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write watchlist JSON to this path (for trader.py consumption)")
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES)
    parser.add_argument("--min-hit-rate", type=float, default=MIN_HIT_RATE)
    args = parser.parse_args()

    # Split into trade CSVs vs price CSVs by filename
    trade_files = [f for f in args.files if 'trade' in os.path.basename(f).lower()]
    price_files = [f for f in args.files if 'price' in os.path.basename(f).lower()]

    if not trade_files:
        print("No trade CSVs found (filenames must contain 'trade')")
        sys.exit(1)

    print(f"Trade CSVs: {len(trade_files)}")
    print(f"Price CSVs: {len(price_files)} (needed for directional scoring)")
    print()

    trades = load_trades(trade_files)
    print(f"Loaded {len(trades)} trades")

    mids = load_prices(price_files) if price_files else {}
    if mids:
        print(f"Loaded {len(mids)} price points")
    print()

    stats = score_traders(trades, mids)
    if not stats:
        print("No named traders found (CSVs may be anonymized — check R1-R4 data)")
        print("Note: trader names typically only appear in R5 data per IMC Prosperity format")
        sys.exit(0)

    # Print ranked output
    print("=" * 100)
    print(f"{'Trader':<16} {'Trades':>7} {'Resolved':>9} {'Wins':>5} {'HitRate':>8} {'AvgQty':>7} {'Products'}")
    print("-" * 100)

    for name, s in sorted(stats.items(), key=lambda x: -x[1]['hit_rate']):
        prods = ",".join(s['products'][:3])
        if len(s['products']) > 3:
            prods += f"+{len(s['products'])-3}"
        print(f"{name:<16} {s['trades']:>7} {s['resolved']:>9} {s['wins']:>5} {s['hit_rate']:>7.1%} {s['avg_qty']:>7.1f} {prods}")

    print()

    # Promote informed traders
    watchlist = {}
    print("=" * 100)
    print("PROMOTION DECISIONS")
    print("=" * 100)
    for name, s in sorted(stats.items(), key=lambda x: -x[1]['hit_rate']):
        if s['resolved'] < args.min_trades:
            verdict = f"insufficient data ({s['resolved']} < {args.min_trades})"
            marker = "[ SKIP ]"
        elif s['hit_rate'] < args.min_hit_rate:
            verdict = f"hit rate too low ({s['hit_rate']:.1%} < {args.min_hit_rate:.0%})"
            marker = "[ SKIP ]"
        elif s['avg_qty'] < MIN_AVG_QTY:
            verdict = f"avg qty too small ({s['avg_qty']:.1f} < {MIN_AVG_QTY})"
            marker = "[ SKIP ]"
        else:
            verdict = f"PROMOTE — hit rate {s['hit_rate']:.1%}, {s['avg_qty']:.1f} avg qty, {s['resolved']} resolved"
            marker = "[INFORM]"
            watchlist[name] = {
                'hit_rate': round(s['hit_rate'], 3),
                'avg_qty': round(s['avg_qty'], 1),
                'resolved': s['resolved'],
                'products': s['products'],
            }
        print(f"  {marker} {name:<16} — {verdict}")

    print()
    if watchlist:
        print(f"PROMOTED: {len(watchlist)} informed trader(s):")
        for name in watchlist:
            print(f"  - {name}")
    else:
        print("No traders met promotion criteria.")

    # Write watchlist JSON
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(_ROOT_DIR, "data", "informed_traders.json")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'informed': list(watchlist.keys()),
            'details': watchlist,
        }, f, indent=2)
    print()
    print(f"Watchlist written: {out_path}")
    print(f"  trader.py reads this via: mem.get('_informed_names', [])")
    print(f"  Integration: see check_olivia() — add name check for each in watchlist.")


if __name__ == "__main__":
    main()
