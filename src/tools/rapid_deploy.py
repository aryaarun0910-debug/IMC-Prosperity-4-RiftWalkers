"""
R1 Rapid Deployment — Sub-10-Minute First Submission

This replaces deploy_r1.py with a faster, more thorough analysis pipeline.
Pre-built templates eliminate tuning guesswork. Cross-product detection
finds baskets/pairs automatically.

Usage:
    python rapid_deploy.py --prices data/r1/prices_*.csv
    python rapid_deploy.py --prices data/r1/prices_*.csv --trades data/r1/trades_*.csv
    python rapid_deploy.py --prices data/r1/prices_*.csv --output trader_r1_v1.py
    python rapid_deploy.py --prices data/r1/prices_*.csv --dry-run
"""

import sys
import os
import json
import re
import glob
import argparse
import math

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _TOOLS_DIR)  # for config_templates, pattern_library
sys.path.insert(0, _ROOT_DIR)   # for trader, datamodel
from config_templates import TEMPLATES, get_template, scale_to_limit
from pattern_library import diagnose_all, diagnose_product, P4_PREDICTIONS


def load_price_data(price_files):
    """Load and merge price CSVs. Returns list of rows."""
    rows = []
    for f in sorted(price_files):
        with open(f, encoding='utf-8') as fh:
            lines = fh.readlines()
        header = [h.strip().lower() for h in lines[0].split(';')]
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
            rows.append(row)
    return rows


def load_trade_data(trade_files):
    """Load trade CSVs. Returns list of rows."""
    rows = []
    for f in sorted(trade_files):
        with open(f, encoding='utf-8') as fh:
            lines = fh.readlines()
        header = [h.strip().lower() for h in lines[0].split(';')]
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
            rows.append(row)
    return rows


def analyze_product(product, product_rows):
    """Analyze a single product from price data."""
    mids = []
    spreads = []
    timestamps = []

    for row in product_rows:
        b1 = row.get('bid_price_1')
        a1 = row.get('ask_price_1')
        if b1 is not None and a1 is not None:
            mid = (b1 + a1) / 2.0
            mids.append(mid)
            spreads.append(a1 - b1)
            ts = row.get('timestamp', 0)
            timestamps.append(ts)

    if not mids:
        return None

    avg_mid = sum(mids) / len(mids)
    avg_spread = sum(spreads) / len(spreads)
    min_spread = min(spreads)
    max_spread = max(spreads)

    # Volatility (tick-to-tick)
    changes = [mids[i] - mids[i-1] for i in range(1, len(mids))]
    vol = (sum(d*d for d in changes) / len(changes)) ** 0.5 if changes else 0

    # Autocorrelation at lag 1
    ac1 = 0
    if len(changes) > 1:
        mean_c = sum(changes) / len(changes)
        var_c = sum((c - mean_c)**2 for c in changes) / len(changes)
        if var_c > 0:
            cov = sum((changes[i] - mean_c) * (changes[i-1] - mean_c)
                      for i in range(1, len(changes))) / (len(changes) - 1)
            ac1 = cov / var_c

    # Hurst exponent estimate (R/S method, simplified)
    hurst = estimate_hurst(mids) if len(mids) > 100 else 0.5

    # Round number detection
    nearest_round = None
    for base in [1000, 5000, 10000]:
        nearest = round(avg_mid / base) * base
        if abs(avg_mid - nearest) < 5:
            nearest_round = int(nearest)
            break

    # Price range and drift
    price_range = max(mids) - min(mids)
    drift = mids[-1] - mids[0]

    return {
        'mids': mids,
        'avg_mid': avg_mid,
        'avg_spread': avg_spread,
        'min_spread': min_spread,
        'max_spread': max_spread,
        'vol': vol,
        'ac1': ac1,
        'hurst': hurst,
        'nearest_round': nearest_round,
        'price_range': price_range,
        'drift': drift,
        'ticks': len(mids),
    }


def estimate_hurst(series):
    """Simplified Hurst exponent via R/S analysis."""
    n = len(series)
    if n < 20:
        return 0.5
    max_k = min(n // 2, 500)
    rs_list = []
    ns_list = []
    for size in [20, 50, 100, 200, 500]:
        if size > max_k:
            break
        chunks = n // size
        if chunks < 1:
            continue
        rs_vals = []
        for c in range(chunks):
            chunk = series[c*size:(c+1)*size]
            mean_c = sum(chunk) / len(chunk)
            devs = [x - mean_c for x in chunk]
            cumdevs = []
            s = 0
            for d in devs:
                s += d
                cumdevs.append(s)
            R = max(cumdevs) - min(cumdevs)
            S = (sum(d*d for d in devs) / len(devs)) ** 0.5
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            rs_list.append(math.log(sum(rs_vals) / len(rs_vals)))
            ns_list.append(math.log(size))

    if len(rs_list) < 2:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    n_pts = len(rs_list)
    sum_x = sum(ns_list)
    sum_y = sum(rs_list)
    sum_xy = sum(x * y for x, y in zip(ns_list, rs_list))
    sum_xx = sum(x * x for x in ns_list)
    denom = n_pts * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.5
    H = (n_pts * sum_xy - sum_x * sum_y) / denom
    return max(0.0, min(1.0, H))


def detect_cross_product_relationships(analyses):
    """Find correlated products that might be basket components or pairs."""
    products = list(analyses.keys())
    if len(products) < 2:
        return {}

    relationships = {}

    # Compute return correlations between all pairs
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            p1, p2 = products[i], products[j]
            m1 = analyses[p1]['mids']
            m2 = analyses[p2]['mids']

            # Align by length (take shorter)
            n = min(len(m1), len(m2))
            if n < 50:
                continue

            r1 = [m1[k] - m1[k-1] for k in range(1, n)]
            r2 = [m2[k] - m2[k-1] for k in range(1, n)]

            # Pearson correlation
            mean1 = sum(r1) / len(r1)
            mean2 = sum(r2) / len(r2)
            cov = sum((a - mean1) * (b - mean2) for a, b in zip(r1, r2)) / len(r1)
            std1 = (sum((a - mean1)**2 for a in r1) / len(r1)) ** 0.5
            std2 = (sum((b - mean2)**2 for b in r2) / len(r2)) ** 0.5

            if std1 > 0 and std2 > 0:
                corr = cov / (std1 * std2)
            else:
                corr = 0

            if abs(corr) > 0.3:
                relationships[f"{p1} <-> {p2}"] = {
                    'correlation': corr,
                    'price_ratio': analyses[p1]['avg_mid'] / analyses[p2]['avg_mid'] if analyses[p2]['avg_mid'] != 0 else 0,
                }

    # Check for basket structure: one expensive product ≈ sum of cheaper ones
    if len(products) >= 3:
        by_price = sorted(products, key=lambda p: analyses[p]['avg_mid'], reverse=True)
        expensive = by_price[0]
        exp_mid = analyses[expensive]['avg_mid']

        # Try all combinations of 2-4 cheaper products
        cheaper = by_price[1:]
        cheap_sum = sum(analyses[p]['avg_mid'] for p in cheaper)

        if cheap_sum > 0:
            ratio = exp_mid / cheap_sum
            if 0.5 < ratio < 2.0:
                # Possible basket! Try to find integer weights
                weights = {}
                for p in cheaper:
                    w = exp_mid / analyses[p]['avg_mid']
                    nearest_int = round(w)
                    if nearest_int > 0 and abs(w - nearest_int) < 0.5:
                        weights[p] = nearest_int

                if weights:
                    weighted_sum = sum(analyses[p]['avg_mid'] * w for p, w in weights.items())
                    premium = exp_mid - weighted_sum
                    relationships[f"BASKET: {expensive}"] = {
                        'components': weights,
                        'premium': premium,
                        'premium_pct': (premium / exp_mid) * 100 if exp_mid != 0 else 0,
                    }

    return relationships


def analyze_trades(product, trade_rows):
    """Analyze trade data for bot fingerprinting."""
    if not trade_rows:
        return None

    # Group trades by buyer/seller
    traders = {}
    for row in trade_rows:
        buyer = row.get('buyer', '')
        seller = row.get('seller', '')
        qty = row.get('quantity', 0)
        price = row.get('price', 0)

        if buyer and buyer != 'SUBMISSION':
            if buyer not in traders:
                traders[buyer] = {'buys': 0, 'sells': 0, 'buy_qty': 0, 'sell_qty': 0,
                                  'prices': [], 'sizes': []}
            traders[buyer]['buys'] += 1
            traders[buyer]['buy_qty'] += qty
            traders[buyer]['prices'].append(price)
            traders[buyer]['sizes'].append(qty)

        if seller and seller != 'SUBMISSION':
            if seller not in traders:
                traders[seller] = {'buys': 0, 'sells': 0, 'buy_qty': 0, 'sell_qty': 0,
                                   'prices': [], 'sizes': []}
            traders[seller]['sells'] += 1
            traders[seller]['sell_qty'] += qty
            traders[seller]['prices'].append(price)
            traders[seller]['sizes'].append(qty)

    # Classify each trader
    results = {}
    for name, stats in traders.items():
        total = stats['buys'] + stats['sells']
        if total < 5:
            continue

        buy_pct = stats['buys'] / total if total > 0 else 0.5
        avg_size = sum(stats['sizes']) / len(stats['sizes']) if stats['sizes'] else 0

        # Size consistency (CV)
        if len(stats['sizes']) > 1:
            mean_s = avg_size
            var_s = sum((s - mean_s)**2 for s in stats['sizes']) / len(stats['sizes'])
            cv = (var_s ** 0.5) / mean_s if mean_s > 0 else 0
        else:
            cv = 0

        if buy_pct > 0.7 or buy_pct < 0.3:
            role = "DIRECTIONAL"
        elif cv < 0.2:
            role = "MM (fixed-size)"
        else:
            role = "MIXED"

        results[name] = {
            'trades': total,
            'buy_pct': buy_pct,
            'avg_size': avg_size,
            'size_cv': cv,
            'role': role,
        }

    return results


def recommend_strategy(product, analysis, relationships=None):
    """Recommend strategy type and CONFIG from templates."""
    a = analysis

    # Check if part of a basket (from relationship detection)
    if relationships:
        for key, rel in relationships.items():
            if key.startswith("BASKET:") and product in key:
                cfg = get_template("basket_arb")
                if 'components' in rel:
                    cfg['components'] = rel['components']
                    cfg['premium_mean'] = rel.get('premium', 0)
                reason = f"Basket detected: {rel.get('components', {})}, premium={rel.get('premium', 0):.1f}"
                return cfg, reason

    # Name-based detection
    basket_keywords = ["BASKET", "ETF", "INDEX", "BUNDLE"]
    if any(kw in product.upper() for kw in basket_keywords):
        cfg = get_template("basket_arb")
        reason = "Name-based basket detection"
        return cfg, reason

    option_patterns = ["_VOUCHER_", "_COUPON_", "_CALL_", "_PUT_", "_OPTION_"]
    for pat in option_patterns:
        if pat in product:
            base = product.split(pat)[0]
            strike_str = product.split(pat)[-1]
            strike = None
            try:
                strike = float(strike_str)
            except (ValueError, TypeError):
                pass
            cfg = get_template("options", underlying=base, strike=strike or 10000)
            reason = f"Option on {base}, strike={strike or '?'}"
            return cfg, reason

    # Order book shape classification
    # 1. Pegged: near round number
    if a['nearest_round'] and a['avg_spread'] <= 20:
        if a['avg_spread'] <= 6:
            cfg = get_template("pegged_tight", fair_value=a['nearest_round'])
        elif a['avg_spread'] <= 14:
            cfg = get_template("pegged", fair_value=a['nearest_round'],
                               make_offset=max(2, int(a['avg_spread'] // 4)))
        else:
            cfg = get_template("pegged_wide", fair_value=a['nearest_round'],
                               make_offset=max(3, int(a['avg_spread'] // 3)))
        reason = f"FV={a['nearest_round']}, spread={a['avg_spread']:.1f}"
        return cfg, reason

    # 2. Tight spread
    if a['avg_spread'] <= 4:
        # High volatility + tight spread = olivia_follow (e.g. SQUID_INK)
        # price_range / avg_mid > 5% separates random-walk/volatile from moderate AR
        range_pct = a['price_range'] / a['avg_mid'] if a['avg_mid'] > 0 else 0
        if range_pct > 0.05:
            cfg = get_template("olivia_follow")
            reason = f"Tight spread ({a['avg_spread']:.1f}), high range ({range_pct*100:.1f}%) -> olivia_follow"
            return cfg, reason
        if a['ac1'] < -0.1:
            cfg = get_template("ar_olivia")
            reason = f"Tight spread ({a['avg_spread']:.1f}), mean-reverting (AC1={a['ac1']:.3f}, H={a['hurst']:.2f})"
        elif a['hurst'] > 0.6:
            cfg = get_template("ar_olivia_trending")
            reason = f"Tight spread ({a['avg_spread']:.1f}), trending (H={a['hurst']:.2f}, AC1={a['ac1']:.3f})"
        else:
            cfg = get_template("ar_olivia")
            reason = f"Tight spread ({a['avg_spread']:.1f}), neutral (AC1={a['ac1']:.3f}, H={a['hurst']:.2f})"
        return cfg, reason

    # 3. Medium spread (4-8)
    if a['avg_spread'] <= 8:
        cfg = get_template("generic_mm")
        reason = f"Medium spread ({a['avg_spread']:.1f}), generic adaptive MM"
        return cfg, reason

    # 4. Wide spread
    if a['avg_spread'] > 50:
        cfg = get_template("wide_spread_extreme",
                           take_ask_threshold=max(10, int(a['avg_spread'] // 3)),
                           bid_comp_threshold=max(8, int(a['avg_spread'] // 4)))
        reason = f"Extreme spread ({a['avg_spread']:.1f})"
    else:
        cfg = get_template("wide_spread",
                           take_ask_threshold=max(4, int(a['avg_spread'] // 2)),
                           bid_comp_threshold=max(3, int(a['avg_spread'] // 3)))
        reason = f"Wide spread ({a['avg_spread']:.1f})"
    return cfg, reason


def generate_config_block(configs):
    """Generate Python CONFIG dict as a string."""
    lines = ["CONFIG = {"]
    for product, cfg in sorted(configs.items()):
        lines.append(f'    "{product}": {{')
        for k, v in sorted(cfg.items()):
            if isinstance(v, str):
                lines.append(f'        "{k}": "{v}",')
            elif isinstance(v, bool):
                lines.append(f'        "{k}": {v},')
            elif isinstance(v, dict):
                lines.append(f'        "{k}": {json.dumps(v)},')
            elif isinstance(v, float):
                lines.append(f'        "{k}": {v},')
            else:
                lines.append(f'        "{k}": {v},')
        lines.append('    },')
    lines.append('}')
    return '\n'.join(lines)


def patch_trader(configs, output_path=None):
    """Patch trader.py with new CONFIG."""
    trader_path = os.path.join(os.path.dirname(__file__), 'trader.py')
    with open(trader_path, 'r', encoding='utf-8') as f:
        source = f.read()

    config_str = generate_config_block(configs)

    # Find CONFIG block using brace counting
    start = source.find('CONFIG = {')
    if start == -1:
        print("ERROR: Could not find CONFIG block in trader.py")
        return None

    depth = 0
    i = start + len('CONFIG = ')
    while i < len(source):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    else:
        print("ERROR: Could not find end of CONFIG block")
        return None

    patched = source[:start] + config_str + source[end:]

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'trader_r1_v1.py')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(patched)

    return output_path


def main():
    parser = argparse.ArgumentParser(description='R1 Rapid Deployment')
    parser.add_argument('--prices', nargs='+', required=True,
                        help='Price CSV files')
    parser.add_argument('--trades', nargs='+', default=None,
                        help='Trade CSV files (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for patched trader.py')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print recommendations without patching')
    args = parser.parse_args()

    # Expand globs
    price_files = []
    for pattern in args.prices:
        price_files.extend(glob.glob(pattern))
    if not price_files:
        print("ERROR: No price files found")
        return 1

    print(f"Loading {len(price_files)} price files...")
    rows = load_price_data(price_files)
    print(f"Loaded {len(rows)} rows")

    # Group by product
    by_product = {}
    for row in rows:
        prod = row.get('product')
        if not prod:
            continue
        if prod not in by_product:
            by_product[prod] = []
        by_product[prod].append(row)

    products = sorted(by_product.keys())
    print(f"Products found: {products}")

    # ── ANALYZE EACH PRODUCT ──
    print(f"\n{'='*70}")
    print(f"PRODUCT ANALYSIS")
    print(f"{'='*70}")

    analyses = {}
    for product in products:
        analysis = analyze_product(product, by_product[product])
        if analysis:
            analyses[product] = analysis

    for product, a in sorted(analyses.items()):
        print(f"\n  {product}:")
        print(f"    Price: avg={a['avg_mid']:.1f}  range={a['price_range']:.1f}  drift={a['drift']:+.1f}")
        print(f"    Spread: avg={a['avg_spread']:.1f}  min={a['min_spread']:.0f}  max={a['max_spread']:.0f}")
        print(f"    Vol: {a['vol']:.3f}  AC1: {a['ac1']:.3f}  Hurst: {a['hurst']:.2f}")
        if a['nearest_round']:
            print(f"    Round number: {a['nearest_round']}")

    # ── PATTERN DIAGNOSIS (cross-year intuition) ──
    print(f"\n{'='*70}")
    print(f"PATTERN DIAGNOSIS (cross-year intuition engine)")
    print(f"{'='*70}")

    pattern_report, _ = diagnose_all(analyses, products)
    print(pattern_report)

    # ── CROSS-PRODUCT RELATIONSHIPS ──
    relationships = detect_cross_product_relationships(analyses)

    if relationships:
        print(f"\n{'='*70}")
        print(f"CROSS-PRODUCT RELATIONSHIPS")
        print(f"{'='*70}")
        for key, rel in sorted(relationships.items()):
            if key.startswith("BASKET:"):
                print(f"\n  {key}")
                print(f"    Components: {rel.get('components', {})}")
                print(f"    Premium: {rel.get('premium', 0):.1f} ({rel.get('premium_pct', 0):.2f}%)")
            else:
                print(f"\n  {key}")
                print(f"    Correlation: {rel['correlation']:.3f}")
                print(f"    Price ratio: {rel['price_ratio']:.3f}")

    # ── BOT ANALYSIS ──
    trade_analyses = {}
    if args.trades:
        trade_files = []
        for pattern in args.trades:
            trade_files.extend(glob.glob(pattern))
        if trade_files:
            print(f"\n{'='*70}")
            print(f"BOT FINGERPRINTING")
            print(f"{'='*70}")

            trade_rows = load_trade_data(trade_files)
            trade_by_product = {}
            for row in trade_rows:
                prod = row.get('symbol') or row.get('product')
                if not prod:
                    continue
                if prod not in trade_by_product:
                    trade_by_product[prod] = []
                trade_by_product[prod].append(row)

            for product in products:
                ta = analyze_trades(product, trade_by_product.get(product, []))
                if ta:
                    trade_analyses[product] = ta
                    print(f"\n  {product}:")
                    for name, stats in sorted(ta.items(), key=lambda x: -x[1]['trades']):
                        print(f"    {name:<20} trades={stats['trades']:>4}  "
                              f"buy%={stats['buy_pct']:.0%}  avg_size={stats['avg_size']:.0f}  "
                              f"role={stats['role']}")

    # ── RECOMMEND STRATEGIES ──
    print(f"\n{'='*70}")
    print(f"STRATEGY RECOMMENDATIONS")
    print(f"{'='*70}")

    configs = {}
    for product in products:
        if product not in analyses:
            print(f"\n  {product}: No valid data, skipping")
            continue

        cfg, reason = recommend_strategy(product, analyses[product], relationships)
        configs[product] = cfg

        print(f"\n  {product}:")
        print(f"    --> {cfg['type']}  ({reason})")
        print(f"    Key params: ", end="")
        important = {k: v for k, v in cfg.items()
                     if k not in ['type', 'position_limit'] and not isinstance(v, dict)}
        print(", ".join(f"{k}={v}" for k, v in sorted(important.items())[:5]))

    # ── WARNINGS ──
    print(f"\n{'='*70}")
    print(f"WARNINGS / ACTION ITEMS")
    print(f"{'='*70}")

    warnings = []
    for product, cfg in configs.items():
        if cfg['type'] == 'basket_arb' and not cfg.get('components'):
            warnings.append(f"  CRITICAL: {product} is basket_arb but components are EMPTY. "
                            f"Must identify and hardcode components!")
        if cfg['type'] == 'basket_arb' and cfg.get('premium_mean', 0) == 0:
            warnings.append(f"  HIGH: {product} premium_mean=0. Analyze basket premium from data "
                            f"and set in CONFIG.")
        if cfg['type'] == 'options' and not cfg.get('underlying'):
            warnings.append(f"  CRITICAL: {product} is options but underlying is empty!")
        if cfg['type'] == 'generic_mm':
            warnings.append(f"  MEDIUM: {product} fell through to generic_mm. Review if a "
                            f"better strategy fits after first log.")

    if warnings:
        for w in warnings:
            print(w)
    else:
        print("  None. All products classified with confidence.")

    # ── GENERATE CONFIG ──
    print(f"\n{'='*70}")
    print(f"GENERATED CONFIG")
    print(f"{'='*70}")
    print()
    print(generate_config_block(configs))

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return 0

    # Patch trader.py
    output_path = patch_trader(configs, args.output)
    if output_path:
        print(f"\n{'='*70}")
        print(f"OUTPUT: {output_path}")
        print(f"{'='*70}")
        print(f"\n  1. Review the config above")
        print(f"  2. Upload {output_path} to IMC")
        print(f"  3. Download log -> python server_log_analyzer.py <log>")
        print(f"  4. Iterate! Target: 4 submissions/hour")

    return 0


if __name__ == '__main__':
    sys.exit(main())
