"""
Server Log Analyzer -- The iteration weapon for IMC Prosperity.

Answers 3 questions per product:
  1. How much did we make?
  2. Where did we leave money on the table?
  3. What should we change?

Usage:
    python server_log_analyzer.py <log>                    # Analyze single log
    python server_log_analyzer.py <log> --compare <best>   # Diff two runs
    python server_log_analyzer.py --dir <logdir>           # Rank all logs

The log can be a .log file or a directory containing one (IMC download format).
"""

import json
import os
import sys
from collections import defaultdict


# ============================================================
# LOG LOADING
# ============================================================

def load_log(path):
    """Load a server log. Handles .log files and directories."""
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith('.log'):
                path = os.path.join(path, f)
                break
        else:
            raise FileNotFoundError(f"No .log files in {path}")
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def parse_activities(csv_str):
    """Parse activitiesLog CSV into per-product time series."""
    lines = csv_str.split('\n')
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
            else:
                row[h] = None

        prod = row.get('product')
        if not prod:
            continue
        if prod not in products:
            products[prod] = []
        products[prod].append(row)

    return products


def parse_trades(trade_list):
    """Parse tradeHistory into our fills per product."""
    by_product = defaultdict(list)
    for t in trade_list:
        is_buy = t.get('buyer') == 'SUBMISSION'
        is_sell = t.get('seller') == 'SUBMISSION'
        if not is_buy and not is_sell:
            continue
        by_product[t['symbol']].append({
            'ts': int(t['timestamp']),
            'price': float(t['price']),
            'qty': int(t['quantity']),
            'side': 'BUY' if is_buy else 'SELL',
        })
    return dict(by_product)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_product(product, activity_rows, fills):
    """Full analysis of a single product. Returns stats dict."""
    # --- PnL ---
    pnl_series = [r['profit_and_loss'] for r in activity_rows if r.get('profit_and_loss') is not None]
    final_pnl = pnl_series[-1] if pnl_series else 0.0
    peak_pnl = max(pnl_series) if pnl_series else 0.0
    trough_pnl = min(pnl_series) if pnl_series else 0.0
    drawdown = peak_pnl - trough_pnl

    # --- Price series ---
    mids = [r['mid_price'] for r in activity_rows if r.get('mid_price') is not None]
    bids = [r['bid_price_1'] for r in activity_rows if r.get('bid_price_1') is not None]
    asks = [r['ask_price_1'] for r in activity_rows if r.get('ask_price_1') is not None]
    timestamps = [r['timestamp'] for r in activity_rows if r.get('timestamp') is not None]

    mid_start = mids[0] if mids else 0
    mid_end = mids[-1] if mids else 0
    mid_drift = mid_end - mid_start

    # Volatility (std of mid-to-mid changes)
    mid_changes = [mids[i] - mids[i-1] for i in range(1, len(mids))]
    vol = (sum(d*d for d in mid_changes) / len(mid_changes)) ** 0.5 if mid_changes else 0

    # Spread stats
    spreads = [a - b for a, b in zip(asks, bids) if a is not None and b is not None]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0
    min_spread = min(spreads) if spreads else 0
    tight_pct = sum(1 for s in spreads if s <= min_spread + 1) / len(spreads) * 100 if spreads else 0

    total_ticks = len(timestamps)

    # --- Fill analysis ---
    buys = [f for f in fills if f['side'] == 'BUY']
    sells = [f for f in fills if f['side'] == 'SELL']
    buy_qty = sum(f['qty'] for f in buys)
    sell_qty = sum(f['qty'] for f in sells)
    total_fills = len(fills)

    vwap_buy = sum(f['price'] * f['qty'] for f in buys) / buy_qty if buy_qty else 0
    vwap_sell = sum(f['price'] * f['qty'] for f in sells) / sell_qty if sell_qty else 0

    # Fill timing
    fill_ts = sorted(f['ts'] for f in fills)
    first_fill = fill_ts[0] if fill_ts else None
    last_fill = fill_ts[-1] if fill_ts else None
    fill_density = total_fills / total_ticks if total_ticks > 0 else 0

    # Fill gaps
    gaps = []
    for i in range(1, len(fill_ts)):
        gap = fill_ts[i] - fill_ts[i-1]
        if gap > 5000:
            gaps.append(gap)

    # --- Position tracking ---
    pos = 0
    max_pos = 0
    min_pos = 0
    pos_history = []
    time_at_limit = 0
    # Known P3/P4 position limits
    LIMITS = {
        'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50,
        'CROISSANTS': 250, 'JAMS': 350, 'DJEMBES': 60,
        'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100,
        'MAGNIFICENT_MACARONS': 75,
        'VOLCANIC_ROCK': 400,
        'VOLCANIC_ROCK_VOUCHER_9500': 200, 'VOLCANIC_ROCK_VOUCHER_9750': 200,
        'VOLCANIC_ROCK_VOUCHER_10000': 200, 'VOLCANIC_ROCK_VOUCHER_10250': 200,
        'VOLCANIC_ROCK_VOUCHER_10500': 200,
    }
    position_limit = LIMITS.get(product, 50)

    for f in sorted(fills, key=lambda x: x['ts']):
        if f['side'] == 'BUY':
            pos += f['qty']
        else:
            pos -= f['qty']
        max_pos = max(max_pos, pos)
        min_pos = min(min_pos, pos)
        pos_history.append((f['ts'], pos))

    # Estimate position utilization (how much of the limit we used)
    pos_utilization = max(abs(max_pos), abs(min_pos)) / position_limit * 100 if position_limit > 0 else 0

    # --- Toxicity ---
    ts_to_mid = {}
    for row in activity_rows:
        ts = row.get('timestamp')
        mp = row.get('mid_price')
        if ts is not None and mp is not None:
            ts_to_mid[ts] = mp

    toxic_count = 0
    total_scored = 0
    for f in fills:
        mid_at = ts_to_mid.get(f['ts'])
        # Look 5 ticks ahead (500ms)
        mid_after = None
        for look in range(f['ts'] + 100, f['ts'] + 600, 100):
            if look in ts_to_mid:
                mid_after = ts_to_mid[look]
                break
        if mid_at is not None and mid_after is not None:
            total_scored += 1
            if f['side'] == 'BUY' and mid_after < mid_at:
                toxic_count += 1
            elif f['side'] == 'SELL' and mid_after > mid_at:
                toxic_count += 1

    toxicity = toxic_count / total_scored if total_scored > 0 else 0.0

    # --- Money left on table ---
    # Spread capture: theoretical max if we captured full spread every tick
    theoretical_spread_income = sum(s for s in spreads) * 0.5  # one side per tick
    actual_income = final_pnl
    capture_pct = actual_income / theoretical_spread_income * 100 if theoretical_spread_income > 0 else 0

    # Unfilled ticks: ticks where we had no fill
    fill_tick_set = set(f['ts'] for f in fills)
    unfilled_ticks = total_ticks - len(fill_tick_set)
    unfilled_pct = unfilled_ticks / total_ticks * 100 if total_ticks > 0 else 0

    return {
        'final_pnl': final_pnl,
        'peak_pnl': peak_pnl,
        'trough_pnl': trough_pnl,
        'drawdown': drawdown,
        'mid_start': mid_start,
        'mid_end': mid_end,
        'mid_drift': mid_drift,
        'vol': vol,
        'avg_spread': avg_spread,
        'min_spread': min_spread,
        'tight_spread_pct': tight_pct,
        'total_ticks': total_ticks,
        'total_fills': total_fills,
        'buy_fills': len(buys),
        'sell_fills': len(sells),
        'buy_qty': buy_qty,
        'sell_qty': sell_qty,
        'vwap_buy': vwap_buy,
        'vwap_sell': vwap_sell,
        'fill_density': fill_density,
        'first_fill': first_fill,
        'last_fill': last_fill,
        'gaps_gt_5s': len(gaps),
        'max_gap': max(gaps) if gaps else 0,
        'max_pos': max_pos,
        'min_pos': min_pos,
        'final_pos': pos,
        'pos_utilization': pos_utilization,
        'toxicity': toxicity,
        'unfilled_pct': unfilled_pct,
        'spread_capture_pct': capture_pct,
        'position_limit': position_limit,
    }


# ============================================================
# SUGGESTIONS ENGINE
# ============================================================

def suggest_changes(product, stats):
    """Generate actionable parameter suggestions."""
    s = stats
    suggestions = []

    # No fills at all
    if s['total_fills'] == 0:
        suggestions.append(('CRITICAL', 'Zero fills -- strategy is not trading. Check if strategy type is correct.'))
        return suggestions

    # Very few fills
    if s['fill_density'] < 0.01:
        suggestions.append(('HIGH', f'Very low fill rate ({s["fill_density"]:.3f}/tick). Try: reduce make_offset by 1-2, or increase make_size.'))

    # High toxicity
    if s['toxicity'] > 0.55:
        suggestions.append(('HIGH', f'High toxicity ({s["toxicity"]:.0%}). Fills are adversely selected. Try: increase make_offset by 1, or add/widen spread.'))

    # Low toxicity = opportunity to tighten
    if s['toxicity'] < 0.30 and s['total_fills'] > 10:
        suggestions.append(('MEDIUM', f'Low toxicity ({s["toxicity"]:.0%}). Room to tighten quotes. Try: decrease make_offset by 1.'))

    # Large fill gaps
    if s['gaps_gt_5s'] > 3:
        suggestions.append(('MEDIUM', f'{s["gaps_gt_5s"]} gaps > 5s between fills (max {s["max_gap"]/1000:.0f}s). Strategy may be stalling. Check inventory/warmup logic.'))

    # Low position utilization
    if s['pos_utilization'] < 40:
        suggestions.append(('MEDIUM', f'Position utilization only {s["pos_utilization"]:.0f}%. Not using full limit. Try: increase make_size or reduce inventory thresholds.'))

    # High position utilization with negative PnL
    if s['pos_utilization'] > 90 and s['final_pnl'] < 0:
        suggestions.append(('HIGH', f'Max position ({s["max_pos"]}/{s["min_pos"]}) with negative PnL. Directional bet went wrong. Try: reduce target_position or add inventory mean-reversion.'))

    # Negative PnL per fill
    pnl_per_fill = s['final_pnl'] / s['total_fills'] if s['total_fills'] > 0 else 0
    if pnl_per_fill < -1:
        suggestions.append(('HIGH', f'Negative PnL per fill ({pnl_per_fill:+.1f}). Taking bad fills. Try: widen take_width or increase make_offset.'))

    # Late first fill
    if s['first_fill'] and s['first_fill'] > 10000:
        suggestions.append(('MEDIUM', f'First fill at t={s["first_fill"]} ({s["first_fill"]/1000:.0f}s). Slow warmup. Try: reduce ar_min_history or use Kalman.'))

    # Large final position (unrealized risk)
    if abs(s['final_pos']) > 0.6 * s.get('position_limit', 50):  # >60% of limit
        suggestions.append(('LOW', f'Large final position ({s["final_pos"]:+d}). Unrealized PnL at risk. Consider flattening in last 20s.'))

    # Buy/sell imbalance
    if s['buy_qty'] > 0 and s['sell_qty'] > 0:
        ratio = s['buy_qty'] / s['sell_qty']
        if ratio > 3 or ratio < 0.33:
            suggestions.append(('LOW', f'Buy/sell imbalance ({s["buy_qty"]}B vs {s["sell_qty"]}S). Strategy may be one-directional.'))

    # Drift vs our position
    if abs(s['mid_drift']) > 5 and s['final_pos'] != 0:
        drift_dir = 'up' if s['mid_drift'] > 0 else 'down'
        pos_dir = 'long' if s['final_pos'] > 0 else 'short'
        if (s['mid_drift'] > 0 and s['final_pos'] < 0) or (s['mid_drift'] < 0 and s['final_pos'] > 0):
            suggestions.append(('MEDIUM', f'Mid drifted {drift_dir} by {abs(s["mid_drift"]):.1f} but we are {pos_dir}. Momentum detection may help.'))

    if not suggestions:
        suggestions.append(('OK', 'No issues detected. Performance looks solid.'))

    return suggestions


# ============================================================
# OUTPUT
# ============================================================

def print_analysis(log_path, all_stats):
    """Print the full analysis."""
    log_name = os.path.basename(log_path).replace('.log', '')
    if os.path.isdir(log_path):
        log_name = os.path.basename(log_path)

    total_pnl = sum(s['final_pnl'] for s in all_stats.values())
    total_fills = sum(s['total_fills'] for s in all_stats.values())

    print(f"\n{'='*75}")
    print(f"  LOG: {log_name}  |  TOTAL PnL: {total_pnl:+.0f}  |  FILLS: {total_fills}")
    print(f"{'='*75}")

    for product in sorted(all_stats.keys()):
        s = all_stats[product]
        print(f"\n  --- {product} ---")
        print(f"  PnL: {s['final_pnl']:+.0f}  (peak {s['peak_pnl']:+.0f}, trough {s['trough_pnl']:+.0f}, drawdown {s['drawdown']:.0f})")
        print(f"  Fills: {s['total_fills']} ({s['buy_fills']}B/{s['sell_fills']}S)  "
              f"Qty: {s['buy_qty']}B/{s['sell_qty']}S  "
              f"Density: {s['fill_density']:.3f}/tick")

        vb = f"{s['vwap_buy']:.1f}" if s['buy_qty'] > 0 else "N/A"
        vs = f"{s['vwap_sell']:.1f}" if s['sell_qty'] > 0 else "N/A"
        print(f"  VWAP: buy={vb}  sell={vs}  "
              f"spread={s['avg_spread']:.1f} (min={s['min_spread']:.0f}, tight={s['tight_spread_pct']:.0f}%)")

        print(f"  Mid: {s['mid_start']:.0f} -> {s['mid_end']:.0f} (drift {s['mid_drift']:+.1f}, vol {s['vol']:.2f})")
        print(f"  Position: [{s['min_pos']:+d}, {s['max_pos']:+d}] final={s['final_pos']:+d}  "
              f"utilization={s['pos_utilization']:.0f}%")
        print(f"  Toxicity: {s['toxicity']:.0%}  "
              f"Unfilled ticks: {s['unfilled_pct']:.0f}%  "
              f"Gaps>5s: {s['gaps_gt_5s']}")

        # Suggestions
        suggestions = suggest_changes(product, s)
        if suggestions:
            for priority, msg in suggestions:
                icon = {'CRITICAL': '!!!', 'HIGH': '!! ', 'MEDIUM': '!  ', 'LOW': '   ', 'OK': '   '}
                print(f"  {icon.get(priority, '   ')} [{priority}] {msg}")


def print_comparison(stats_a, stats_b, name_a, name_b):
    """Diff two runs."""
    print(f"\n{'='*75}")
    print(f"  COMPARISON: {name_a} vs {name_b}")
    print(f"{'='*75}")

    all_products = sorted(set(list(stats_a.keys()) + list(stats_b.keys())))

    print(f"\n  {'Product':<15} {'dPnL':>8} {'dFills':>7} {'dToxic':>7} {'dDensity':>9} {'dPosUtil':>9}")
    print(f"  {'-'*15} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")

    total_delta = 0
    for product in all_products:
        a = stats_a.get(product, {})
        b = stats_b.get(product, {})

        dpnl = a.get('final_pnl', 0) - b.get('final_pnl', 0)
        dfills = a.get('total_fills', 0) - b.get('total_fills', 0)
        dtoxic = a.get('toxicity', 0) - b.get('toxicity', 0)
        ddensity = a.get('fill_density', 0) - b.get('fill_density', 0)
        dutil = a.get('pos_utilization', 0) - b.get('pos_utilization', 0)
        total_delta += dpnl

        print(f"  {product:<15} {dpnl:>+8.0f} {dfills:>+7} {dtoxic:>+7.2f} {ddensity:>+9.4f} {dutil:>+9.1f}%")

    print(f"  {'-'*15} {'-'*8}")
    print(f"  {'TOTAL':<15} {total_delta:>+8.0f}")

    if total_delta > 0:
        print(f"\n  >> {name_a} is BETTER by {total_delta:+.0f}")
    elif total_delta < 0:
        print(f"\n  >> {name_b} is BETTER by {-total_delta:+.0f}")
    else:
        print(f"\n  >> EQUAL")


def rank_logs(logdir):
    """Rank all logs in a directory."""
    entries = []
    for name in sorted(os.listdir(logdir)):
        path = os.path.join(logdir, name)
        try:
            data = load_log(path)
        except:
            continue

        products = parse_activities(data['activitiesLog'])
        trades = parse_trades(data['tradeHistory'])
        all_stats = {}
        for prod, rows in products.items():
            fills = trades.get(prod, [])
            all_stats[prod] = analyze_product(prod, rows, fills)

        total = sum(s['final_pnl'] for s in all_stats.values())
        total_fills = sum(s['total_fills'] for s in all_stats.values())
        entries.append((name, total, total_fills, all_stats))

    entries.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*75}")
    print(f"  ALL LOGS ({len(entries)} found)")
    print(f"{'='*75}")

    print(f"\n  {'#':<4} {'Log':<40} {'PnL':>8} {'Fills':>6}")
    print(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*6}")

    for i, (name, pnl, fills, _) in enumerate(entries):
        clean = name.replace('.log', '').replace('variantslogs', '').replace('probeslogs', '')
        marker = ' <-- BEST' if i == 0 else ''
        print(f"  {i+1:<4} {clean:<40} {pnl:>+8.0f} {fills:>6}{marker}")

    if len(entries) >= 2:
        best_pnl = entries[0][1]
        worst_pnl = entries[-1][1]
        avg_pnl = sum(e[1] for e in entries) / len(entries)
        print(f"\n  Best: {best_pnl:+.0f}  Worst: {worst_pnl:+.0f}  Avg: {avg_pnl:+.0f}  Range: {best_pnl - worst_pnl:.0f}")

    return entries


# ============================================================
# MAIN
# ============================================================

def main():
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python server_log_analyzer.py <log_path>")
        print("  python server_log_analyzer.py <log_path> --compare <other_log>")
        print("  python server_log_analyzer.py --dir <directory>")
        print("")
        print("  <log_path> can be a .log file or a directory containing one")
        return

    # Parse args
    log_path = None
    compare_path = None
    dir_path = None

    i = 0
    while i < len(args):
        if args[i] == '--compare' and i + 1 < len(args):
            compare_path = args[i + 1]
            i += 2
        elif args[i] == '--dir' and i + 1 < len(args):
            dir_path = args[i + 1]
            i += 2
        elif log_path is None:
            log_path = args[i]
            i += 1
        else:
            i += 1

    if dir_path:
        rank_logs(dir_path)
        return

    if not log_path:
        print("Provide a log path or --dir")
        return

    # Load and analyze
    data = load_log(log_path)
    products = parse_activities(data['activitiesLog'])
    trades = parse_trades(data['tradeHistory'])

    all_stats = {}
    for prod, rows in products.items():
        fills = trades.get(prod, [])
        all_stats[prod] = analyze_product(prod, rows, fills)

    print_analysis(log_path, all_stats)

    # Parse and display sandbox log errors (ERR| messages from our exception handler)
    sandbox_log = data.get('sandboxLog', '')
    if sandbox_log:
        err_lines = [line.strip() for line in sandbox_log.split('\n') if 'ERR|' in line]
        if err_lines:
            print(f"\n  {'='*75}")
            print(f"  ERROR LOG ({len(err_lines)} errors)")
            print(f"  {'='*75}")
            # Deduplicate and count
            err_counts = defaultdict(int)
            for line in err_lines:
                err_counts[line] += 1
            for line, count in sorted(err_counts.items(), key=lambda x: -x[1]):
                print(f"  [{count}x] {line[:100]}")
        else:
            print(f"\n  No ERR| messages in sandbox log (clean run)")

    # Parse traderData size warnings
    if sandbox_log:
        td_warnings = [l.strip() for l in sandbox_log.split('\n') if 'traderData' in l.lower() and ('size' in l.lower() or 'limit' in l.lower() or 'too large' in l.lower())]
        if td_warnings:
            print(f"\n  WARNING: traderData size issues detected:")
            for w in td_warnings[:5]:
                print(f"    {w[:120]}")

    # Compare mode
    if compare_path:
        data_b = load_log(compare_path)
        products_b = parse_activities(data_b['activitiesLog'])
        trades_b = parse_trades(data_b['tradeHistory'])
        stats_b = {}
        for prod, rows in products_b.items():
            fills = trades_b.get(prod, [])
            stats_b[prod] = analyze_product(prod, rows, fills)

        name_a = os.path.basename(log_path).replace('.log', '')
        name_b = os.path.basename(compare_path).replace('.log', '')
        print_comparison(all_stats, stats_b, name_a, name_b)


if __name__ == '__main__':
    main()
