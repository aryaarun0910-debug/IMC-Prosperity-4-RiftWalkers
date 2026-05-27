"""
Cross-Year Price Pattern Mining: Compare same-archetype products across P2 and P3
to discover invariants that will hold in P4.

Key archetypes:
  Pegged ~10K:     P2 AMETHYSTS  vs P3 RAINFOREST_RESIN
  Volatile/AR:     P2 STARFRUIT  vs P3 KELP
  Mean-revert:     P2 (none R1)  vs P3 SQUID_INK

Usage:
    py -3.12 cross_year_analysis.py
"""

import os
import sys
import math

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_ANALYSIS_DIR)
sys.path.insert(0, _ROOT_DIR)
os.chdir(_ROOT_DIR)

P2_DATA = os.path.join(_ROOT_DIR, "reference/p2_data")
P3_DATA = os.path.join(_ROOT_DIR, "reference/p3_data")


def load_csv_rows(path):
    """Load CSV without pandas. Returns list of dicts."""
    rows = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    if not lines:
        return rows
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


def extract_product_stats(rows, product):
    """Extract statistical fingerprint from price rows for one product."""
    prod_rows = [r for r in rows if r.get('product') == product]
    if not prod_rows:
        return None

    mids = []
    spreads = []
    bid_depths_l1 = []
    ask_depths_l1 = []

    for r in prod_rows:
        b1 = r.get('bid_price_1')
        a1 = r.get('ask_price_1')
        if b1 is not None and a1 is not None:
            mids.append((b1 + a1) / 2.0)
            spreads.append(a1 - b1)
        bv1 = r.get('bid_volume_1')
        av1 = r.get('ask_volume_1')
        if bv1 is not None:
            bid_depths_l1.append(bv1)
        if av1 is not None:
            ask_depths_l1.append(abs(av1))

    if len(mids) < 50:
        return None

    # Returns (tick-to-tick changes)
    returns = [mids[i] - mids[i-1] for i in range(1, len(mids))]

    # Autocorrelation at lag 1
    ac1 = 0
    if len(returns) > 2:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r)**2 for r in returns) / len(returns)
        if var_r > 0:
            cov = sum((returns[i] - mean_r) * (returns[i-1] - mean_r)
                      for i in range(1, len(returns))) / (len(returns) - 1)
            ac1 = cov / var_r

    # Autocorrelation at lag 2-5
    acs = {}
    for lag in [2, 3, 5, 10]:
        if len(returns) > lag + 1:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r)**2 for r in returns) / len(returns)
            if var_r > 0:
                cov = sum((returns[i] - mean_r) * (returns[i-lag] - mean_r)
                          for i in range(lag, len(returns))) / (len(returns) - lag)
                acs[lag] = cov / var_r
            else:
                acs[lag] = 0
        else:
            acs[lag] = 0

    # Volatility
    vol = (sum(r*r for r in returns) / len(returns)) ** 0.5 if returns else 0

    # Hurst exponent (simplified R/S)
    hurst = estimate_hurst(mids) if len(mids) > 200 else 0.5

    # Spread stats
    avg_spread = sum(spreads) / len(spreads)
    min_spread = min(spreads)
    max_spread = max(spreads)
    spread_std = (sum((s - avg_spread)**2 for s in spreads) / len(spreads)) ** 0.5

    # Round number detection
    avg_mid = sum(mids) / len(mids)
    nearest_round = None
    for base in [1000, 5000, 10000]:
        nearest = round(avg_mid / base) * base
        if abs(avg_mid - nearest) < 5:
            nearest_round = int(nearest)
            break

    # Depth stats
    avg_bid_depth = sum(bid_depths_l1) / len(bid_depths_l1) if bid_depths_l1 else 0
    avg_ask_depth = sum(ask_depths_l1) / len(ask_depths_l1) if ask_depths_l1 else 0

    # Price range
    price_range = max(mids) - min(mids)
    drift = mids[-1] - mids[0]

    # Return distribution (skew, kurtosis approximation)
    if len(returns) > 10 and vol > 0:
        std_returns = [(r - sum(returns)/len(returns)) / vol for r in returns]
        skew = sum(r**3 for r in std_returns) / len(std_returns)
        kurtosis = sum(r**4 for r in std_returns) / len(std_returns) - 3  # excess
    else:
        skew = 0
        kurtosis = 0

    return {
        'product': product,
        'ticks': len(mids),
        'avg_mid': avg_mid,
        'avg_spread': avg_spread,
        'min_spread': min_spread,
        'max_spread': max_spread,
        'spread_std': spread_std,
        'vol': vol,
        'ac1': ac1,
        'acs': acs,
        'hurst': hurst,
        'nearest_round': nearest_round,
        'price_range': price_range,
        'drift': drift,
        'avg_bid_depth': avg_bid_depth,
        'avg_ask_depth': avg_ask_depth,
        'skew': skew,
        'kurtosis': kurtosis,
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


def compare_archetypes(p2_stats, p3_stats, archetype_name):
    """Compare two same-archetype products and extract invariants."""
    if p2_stats is None or p3_stats is None:
        return None

    print(f"\n{'='*70}")
    print(f"ARCHETYPE: {archetype_name}")
    print(f"  P2: {p2_stats['product']}  |  P3: {p3_stats['product']}")
    print(f"{'='*70}")

    metrics = [
        ('Avg Mid',        'avg_mid',        '.1f'),
        ('Avg Spread',     'avg_spread',     '.2f'),
        ('Min Spread',     'min_spread',     '.0f'),
        ('Spread Std',     'spread_std',     '.2f'),
        ('Volatility',     'vol',            '.3f'),
        ('AC(1)',          'ac1',            '.4f'),
        ('Hurst',          'hurst',          '.3f'),
        ('Price Range',    'price_range',    '.1f'),
        ('Drift',          'drift',          '.1f'),
        ('Avg Bid Depth',  'avg_bid_depth',  '.1f'),
        ('Avg Ask Depth',  'avg_ask_depth',  '.1f'),
        ('Skew',           'skew',           '.3f'),
        ('Kurtosis',       'kurtosis',       '.2f'),
        ('Round Number',   'nearest_round',  ''),
    ]

    print(f"\n  {'Metric':<20} {'P2':>12} {'P3':>12} {'Delta':>10} {'Stable?':>8}")
    print(f"  {'-'*62}")

    stable_count = 0
    total_comparable = 0
    invariants = []

    for name, key, fmt in metrics:
        v2 = p2_stats.get(key)
        v3 = p3_stats.get(key)

        if v2 is None and v3 is None:
            continue

        if key == 'nearest_round':
            s2 = str(v2) if v2 else 'None'
            s3 = str(v3) if v3 else 'None'
            stable = v2 == v3
            delta = 'same' if stable else 'DIFF'
        elif isinstance(v2, (int, float)) and isinstance(v3, (int, float)):
            s2 = f'{v2:{fmt}}'
            s3 = f'{v3:{fmt}}'
            if abs(v2) + abs(v3) > 0:
                pct_diff = abs(v2 - v3) / max(abs(v2), abs(v3), 1e-10) * 100
                delta = f'{pct_diff:.0f}%'
                # Stable if within 30% relative difference
                stable = pct_diff < 30
                total_comparable += 1
                if stable:
                    stable_count += 1
                    invariants.append((name, v2, v3))
            else:
                delta = '0%'
                stable = True
                total_comparable += 1
                stable_count += 1
        else:
            s2 = str(v2)
            s3 = str(v3)
            delta = '?'
            stable = False

        marker = 'YES' if stable else 'no'
        print(f"  {name:<20} {s2:>12} {s3:>12} {delta:>10} {marker:>8}")

    # Higher-order AC
    print(f"\n  Autocorrelation structure:")
    for lag in [2, 3, 5, 10]:
        ac2 = p2_stats['acs'].get(lag, 0)
        ac3 = p3_stats['acs'].get(lag, 0)
        print(f"    AC({lag}): P2={ac2:+.4f}  P3={ac3:+.4f}")

    print(f"\n  STABILITY SCORE: {stable_count}/{total_comparable} metrics within 30%")

    if invariants:
        print(f"\n  INVARIANTS (stable across years, safe to assume for P4):")
        for name, v2, v3 in invariants:
            avg = (v2 + v3) / 2
            print(f"    {name}: ~{avg:.3f}")

    return {
        'archetype': archetype_name,
        'p2': p2_stats['product'],
        'p3': p3_stats['product'],
        'stability': stable_count / total_comparable if total_comparable > 0 else 0,
        'invariants': invariants,
    }


def load_round_data(base_dir, round_num, day_nums):
    """Load price rows for a specific round and day range."""
    rows = []
    for d in day_nums:
        path = f"{base_dir}/Round {round_num} Data/prices_round_{round_num}_day_{d}.csv"
        if os.path.exists(path):
            rows.extend(load_csv_rows(path))
    return rows


def extract_basket_stats(rows, basket_product, component_products):
    """Extract basket-specific statistics: premium, NAV tracking error."""
    # Get mid time series for basket and components
    ticks = sorted(set((r.get('timestamp', 0), r.get('day', 0)) for r in rows))
    by_tick = {}
    for r in rows:
        key = (r.get('timestamp', 0), r.get('day', 0))
        p = r.get('product')
        if p and r.get('bid_price_1') and r.get('ask_price_1'):
            if key not in by_tick:
                by_tick[key] = {}
            by_tick[key][p] = (r['bid_price_1'] + r['ask_price_1']) / 2.0

    premiums = []
    for tick, mids in sorted(by_tick.items()):
        if basket_product not in mids:
            continue
        # Compute NAV from components
        nav = 0
        have_all = True
        for comp, weight in component_products.items():
            if comp in mids:
                nav += weight * mids[comp]
            else:
                have_all = False
                break
        if have_all and nav > 0:
            premium = mids[basket_product] - nav
            premiums.append(premium)

    if len(premiums) < 50:
        return None

    avg_prem = sum(premiums) / len(premiums)
    std_prem = (sum((p - avg_prem)**2 for p in premiums) / len(premiums)) ** 0.5
    min_prem = min(premiums)
    max_prem = max(premiums)

    # Premium autocorrelation
    prem_returns = [premiums[i] - premiums[i-1] for i in range(1, len(premiums))]
    prem_ac1 = 0
    if len(prem_returns) > 10:
        mean_pr = sum(prem_returns) / len(prem_returns)
        var_pr = sum((r - mean_pr)**2 for r in prem_returns) / len(prem_returns)
        if var_pr > 0:
            cov = sum((prem_returns[i] - mean_pr) * (prem_returns[i-1] - mean_pr)
                      for i in range(1, len(prem_returns))) / (len(prem_returns) - 1)
            prem_ac1 = cov / var_pr

    return {
        'basket': basket_product,
        'components': component_products,
        'ticks': len(premiums),
        'avg_premium': avg_prem,
        'std_premium': std_prem,
        'min_premium': min_prem,
        'max_premium': max_prem,
        'premium_ac1': prem_ac1,
    }


def compare_baskets(p2_bstats, p3_bstats, label):
    """Compare basket behavior across years."""
    if not p2_bstats or not p3_bstats:
        return None

    print(f"\n{'='*70}")
    print(f"BASKET ARCHETYPE: {label}")
    print(f"  P2: {p2_bstats['basket']} = {p2_bstats['components']}")
    print(f"  P3: {p3_bstats['basket']} = {p3_bstats['components']}")
    print(f"{'='*70}")

    metrics = [
        ('Avg Premium',    'avg_premium',   '.1f'),
        ('Std Premium',    'std_premium',   '.1f'),
        ('Min Premium',    'min_premium',   '.1f'),
        ('Max Premium',    'max_premium',   '.1f'),
        ('Premium AC(1)',  'premium_ac1',   '.4f'),
    ]

    print(f"\n  {'Metric':<20} {'P2':>12} {'P3':>12}")
    print(f"  {'-'*44}")
    for name, key, fmt in metrics:
        v2 = p2_bstats.get(key, 0)
        v3 = p3_bstats.get(key, 0)
        s2 = f"{v2:{fmt}}"
        s3 = f"{v3:{fmt}}"
        print(f"  {name:<20} {s2:>12} {s3:>12}")

    invariants = []
    # Premium mean-reversion is an invariant if both have AC1 < -0.1
    if p2_bstats['premium_ac1'] < -0.1 and p3_bstats['premium_ac1'] < -0.1:
        invariants.append("Premium is mean-reverting (AC1 < -0.1)")
    if abs(p2_bstats['avg_premium']) < 500 and abs(p3_bstats['avg_premium']) < 500:
        invariants.append("Premium stays within +/- 500 of mean")

    if invariants:
        print(f"\n  INVARIANTS:")
        for inv in invariants:
            print(f"    - {inv}")

    return {'label': label, 'invariants': invariants}


def main():
    print("=" * 70)
    print("CROSS-YEAR PRICE PATTERN MINING")
    print("P2 vs P3 — Finding invariants for P4 prediction")
    print("=" * 70)

    # ── SECTION 1: R1 Products (pegged, AR, mean-revert) ──

    # Load P2 R1 data
    p2_r1 = load_round_data(P2_DATA, 1, [-2, -1, 0])
    # Load P3 R1 data (local copy)
    p3_files = sorted([
        "data/p3_round1/prices_round_1_day_-2.csv",
        "data/p3_round1/prices_round_1_day_-1.csv",
        "data/p3_round1/prices_round_1_day_0.csv",
    ])
    p3_r1 = []
    for f in [f for f in p3_files if os.path.exists(f)]:
        p3_r1.extend(load_csv_rows(f))

    p2_r1_products = sorted(set(r.get('product') for r in p2_r1 if r.get('product')))
    p3_r1_products = sorted(set(r.get('product') for r in p3_r1 if r.get('product')))
    print(f"\nP2 R1 products: {p2_r1_products}")
    print(f"P3 R1 products: {p3_r1_products}")

    # Extract stats for R1 products
    all_stats = {}
    for p in p2_r1_products:
        stats = extract_product_stats(p2_r1, p)
        if stats:
            all_stats[f"P2_{p}"] = stats
            print(f"\n  P2 {p}: mid={stats['avg_mid']:.1f}, spread={stats['avg_spread']:.2f}, "
                  f"vol={stats['vol']:.3f}, AC1={stats['ac1']:.4f}, H={stats['hurst']:.3f}")
    for p in p3_r1_products:
        stats = extract_product_stats(p3_r1, p)
        if stats:
            all_stats[f"P3_{p}"] = stats
            print(f"  P3 {p}: mid={stats['avg_mid']:.1f}, spread={stats['avg_spread']:.2f}, "
                  f"vol={stats['vol']:.3f}, AC1={stats['ac1']:.4f}, H={stats['hurst']:.3f}")

    # Define R1 archetype pairs
    archetype_pairs = [
        ("Pegged ~10K", "AMETHYSTS", "RAINFOREST_RESIN"),
        ("Volatile/AR", "STARFRUIT", "KELP"),
    ]

    results = []
    for archetype, p2_name, p3_name in archetype_pairs:
        p2_s = all_stats.get(f"P2_{p2_name}")
        p3_s = all_stats.get(f"P3_{p3_name}")
        result = compare_archetypes(p2_s, p3_s, archetype)
        if result:
            results.append(result)

    # SQUID_INK standalone
    squid = all_stats.get("P3_SQUID_INK")
    if squid:
        print(f"\n{'='*70}")
        print(f"SQUID_INK — No direct P2 R1 equivalent")
        print(f"{'='*70}")
        print(f"  Stats: mid={squid['avg_mid']:.1f}, spread={squid['avg_spread']:.2f}, "
              f"vol={squid['vol']:.3f}, AC1={squid['ac1']:.4f}, H={squid['hurst']:.3f}")
        print(f"  Archetype: Mean-reverting at 1-tick horizon (AC1 < -0.1)")

    # ── SECTION 2: Basket Products (R2-R3) ──
    print(f"\n\n{'#'*70}")
    print(f"SECTION 2: BASKET / ETF PRODUCTS")
    print(f"{'#'*70}")

    # P2 R3: GIFT_BASKET = 4*CHOCOLATE + 6*STRAWBERRIES + 1*ROSES
    p2_r3 = load_round_data(P2_DATA, 3, [0, 1, 2])
    # P3 R2: PICNIC_BASKET1 = 6*CROISSANTS + 3*JAMS + 1*DJEMBES
    #         PICNIC_BASKET2 = 4*CROISSANTS + 2*JAMS
    p3_r2 = load_round_data(P3_DATA, 2, [-1, 0, 1])

    if p2_r3:
        p2_r3_products = sorted(set(r.get('product') for r in p2_r3 if r.get('product')))
        print(f"\n  P2 R3 products: {p2_r3_products}")
    if p3_r2:
        p3_r2_products = sorted(set(r.get('product') for r in p3_r2 if r.get('product')))
        print(f"  P3 R2 products: {p3_r2_products}")

    # Basket stats
    p2_basket = extract_basket_stats(p2_r3, "GIFT_BASKET",
        {"CHOCOLATE": 4, "STRAWBERRIES": 6, "ROSES": 1})
    p3_basket1 = extract_basket_stats(p3_r2, "PICNIC_BASKET1",
        {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1})
    p3_basket2 = extract_basket_stats(p3_r2, "PICNIC_BASKET2",
        {"CROISSANTS": 4, "JAMS": 2})

    basket_result = compare_baskets(p2_basket, p3_basket1, "Large Basket (GIFT vs PICNIC1)")
    if basket_result:
        results.append(basket_result)

    # Standalone basket stats
    for bstats, label in [(p2_basket, "P2 GIFT_BASKET"), (p3_basket1, "P3 PICNIC_BASKET1"),
                           (p3_basket2, "P3 PICNIC_BASKET2")]:
        if bstats:
            print(f"\n  {label}: avg_prem={bstats['avg_premium']:.1f}, "
                  f"std_prem={bstats['std_premium']:.1f}, "
                  f"prem_AC1={bstats['premium_ac1']:.4f}")

    # Extract microstructure stats for basket components
    basket_components = {}
    for p in ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]:
        s = extract_product_stats(p2_r3, p)
        if s:
            basket_components[f"P2_{p}"] = s
    for p in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]:
        s = extract_product_stats(p3_r2, p)
        if s:
            basket_components[f"P3_{p}"] = s

    if basket_components:
        print(f"\n  Component microstructure:")
        print(f"  {'Product':<25} {'Mid':>10} {'Spread':>8} {'Vol':>8} {'AC1':>8}")
        print(f"  {'-'*59}")
        for key in sorted(basket_components):
            s = basket_components[key]
            print(f"  {key:<25} {s['avg_mid']:>10.1f} {s['avg_spread']:>8.2f} "
                  f"{s['vol']:>8.3f} {s['ac1']:>8.4f}")

    # ── SECTION 3: Options Products (R4) ──
    print(f"\n\n{'#'*70}")
    print(f"SECTION 3: OPTIONS / DERIVATIVES")
    print(f"{'#'*70}")

    # P2 R4: COCONUT + COCONUT_COUPON
    p2_r4 = load_round_data(P2_DATA, 4, [1, 2, 3])
    # P3 R4: VOLCANIC_ROCK + VOLCANIC_ROCK_VOUCHER_*
    p3_r4 = load_round_data(P3_DATA, 4, [1, 2, 3])

    option_stats = {}
    for p in ["COCONUT", "COCONUT_COUPON"]:
        s = extract_product_stats(p2_r4, p)
        if s:
            option_stats[f"P2_{p}"] = s
    for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_10000",
              "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500",
              "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750"]:
        s = extract_product_stats(p3_r4, p)
        if s:
            option_stats[f"P3_{p}"] = s

    if option_stats:
        print(f"\n  Options microstructure:")
        print(f"  {'Product':<40} {'Mid':>10} {'Spread':>8} {'Vol':>8} {'AC1':>8}")
        print(f"  {'-'*74}")
        for key in sorted(option_stats):
            s = option_stats[key]
            print(f"  {key:<40} {s['avg_mid']:>10.1f} {s['avg_spread']:>8.2f} "
                  f"{s['vol']:>8.3f} {s['ac1']:>8.4f}")

    # Compare underlying: COCONUT vs VOLCANIC_ROCK
    p2_ul = option_stats.get("P2_COCONUT")
    p3_ul = option_stats.get("P3_VOLCANIC_ROCK")
    if p2_ul and p3_ul:
        ul_result = compare_archetypes(p2_ul, p3_ul, "Options Underlying")
        if ul_result:
            results.append(ul_result)

    # Compare option: COCONUT_COUPON vs closest ATM voucher
    p2_opt = option_stats.get("P2_COCONUT_COUPON")
    p3_opt = option_stats.get("P3_VOLCANIC_ROCK_VOUCHER_10000")
    if p2_opt and p3_opt:
        opt_result = compare_archetypes(p2_opt, p3_opt, "ATM Option/Voucher")
        if opt_result:
            results.append(opt_result)

    # ── SECTION 4: Conversion Products (R4+) ──
    print(f"\n\n{'#'*70}")
    print(f"SECTION 4: CONVERSION PRODUCTS")
    print(f"{'#'*70}")

    # P3 R4: MAGNIFICENT_MACARONS (no direct P2 equivalent in same round format)
    macaron = None
    for p in ["MAGNIFICENT_MACARONS"]:
        s = extract_product_stats(p3_r4, p)
        if s:
            macaron = s
            print(f"\n  P3 {p}: mid={s['avg_mid']:.1f}, spread={s['avg_spread']:.2f}, "
                  f"vol={s['vol']:.3f}, AC1={s['ac1']:.4f}, H={s['hurst']:.3f}")
    if macaron:
        print(f"  Note: Conversion products have external factors (sunlight, tariffs)")
        print(f"  P4 will likely have a new conversion product with similar mechanics")
        print(f"  Key: conversion_arb strategy + observation data parsing")

    # ── SUMMARY ──
    print(f"\n\n{'='*70}")
    print(f"P4 PREDICTION SUMMARY — ALL ARCHETYPES")
    print(f"{'='*70}")

    for r in results:
        label = r.get('archetype', r.get('label', '?'))
        p2 = r.get('p2', '')
        p3 = r.get('p3', '')
        stability = r.get('stability')
        print(f"\n  {label}:")
        if p2 and p3:
            print(f"    {p2} -> {p3} -> P4 new name")
        if stability is not None:
            print(f"    Stability: {stability:.0%}")
        inv = r.get('invariants', [])
        if inv:
            print(f"    Invariants for P4:")
            for item in inv:
                if isinstance(item, tuple):
                    name, v2, v3 = item
                    avg = (v2 + v3) / 2
                    print(f"      {name}: ~{avg:.3f}")
                else:
                    print(f"      {item}")

    print(f"\n  STRATEGY IMPLICATIONS:")
    print(f"    R1: Pegged (~10K FV) + AR (tight spread) + Mean-revert (Olivia)")
    print(f"    R2: Basket/ETF will appear. Premium is mean-reverting. Trade NAV spread.")
    print(f"    R3-R4: Options will appear. Underlying is volatile. IV smile matters.")
    print(f"    R4+: Conversion product with external observations (weather/tariffs).")
    print(f"    ALL: Position limits may change. Names WILL change. Detect by book shape.")


if __name__ == "__main__":
    main()
