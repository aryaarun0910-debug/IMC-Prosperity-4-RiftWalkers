"""
Data Profiler — Deep product analysis using cross-disciplinary techniques.

Layer 1 of the intelligence pipeline. Runs OFFLINE on R1 CSVs before submission.
Surfaces insights that no competition team looks for, then outputs CONFIG hints
for the self-activating frontier engine in trader.py.

Techniques:
  - Shannon entropy of trade direction (Information Theory)
  - Mutual information between products (Information Theory / RenTech)
  - Autocorrelation structure & regime detection (Time Series / Comp Bio)
  - Order flow toxicity VPIN estimate (Market Microstructure)
  - Bot fingerprinting from trade patterns (Pattern Recognition)
  - Lead-lag detection at multiple lags (Citadel approach)
  - Spread distribution & penny-jump frequency (Exchange Microstructure)
  - Hurst exponent for mean-reversion vs trending (Fractal Analysis)

Usage:
    py -3.12 tools/data_profiler.py data/r1/prices/*.csv
    py -3.12 tools/data_profiler.py data/r1/prices/*.csv --trades data/r1/trades/*.csv
    py -3.12 tools/data_profiler.py data/r1/prices/*.csv --output profile.json
"""

import sys
import os
import csv
import json
import math
import argparse
from collections import defaultdict, Counter

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_prices(csv_paths):
    """Load price CSVs into {product: [{timestamp, bid1, ask1, mid, spread, ...}]}."""
    products = defaultdict(list)
    for path in csv_paths:
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                product = row.get('product', '')
                if not product:
                    continue
                try:
                    ts = int(row.get('timestamp', 0))
                    # Parse bids/asks from columns
                    bids = {}
                    asks = {}
                    for i in range(1, 4):
                        bp = row.get(f'bid_price_{i}', '')
                        bv = row.get(f'bid_volume_{i}', '')
                        ap = row.get(f'ask_price_{i}', '')
                        av = row.get(f'ask_volume_{i}', '')
                        if bp and bv:
                            bids[float(bp)] = int(bv)
                        if ap and av:
                            asks[float(ap)] = abs(int(av))

                    if bids and asks:
                        best_bid = max(bids)
                        best_ask = min(asks)
                        mid = (best_bid + best_ask) / 2
                        spread = best_ask - best_bid
                        total_bid_vol = sum(bids.values())
                        total_ask_vol = sum(asks.values())

                        products[product].append({
                            'ts': ts, 'mid': mid, 'spread': spread,
                            'bid1': best_bid, 'ask1': best_ask,
                            'bid_vol': total_bid_vol, 'ask_vol': total_ask_vol,
                            'microprice': (best_bid * total_ask_vol + best_ask * total_bid_vol)
                                          / (total_bid_vol + total_ask_vol)
                                          if (total_bid_vol + total_ask_vol) > 0 else mid,
                        })
                except (ValueError, KeyError):
                    continue

    # Sort by timestamp
    for p in products:
        products[p].sort(key=lambda x: x['ts'])
    return dict(products)


def load_trades(csv_paths):
    """Load trade CSVs into {product: [{timestamp, price, quantity, buyer, seller}]}."""
    products = defaultdict(list)
    for path in csv_paths:
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                product = row.get('symbol', row.get('product', ''))
                if not product:
                    continue
                try:
                    products[product].append({
                        'ts': int(row.get('timestamp', 0)),
                        'price': float(row.get('price', 0)),
                        'quantity': int(row.get('quantity', 0)),
                        'buyer': row.get('buyer', ''),
                        'seller': row.get('seller', ''),
                    })
                except (ValueError, KeyError):
                    continue
    for p in products:
        products[p].sort(key=lambda x: x['ts'])
    return dict(products)


# ═══════════════════════════════════════════════════════════════
# ANALYSIS TECHNIQUES
# ═══════════════════════════════════════════════════════════════

def compute_returns(mids):
    """Compute tick-to-tick returns."""
    if len(mids) < 2:
        return []
    return [mids[i] - mids[i-1] for i in range(1, len(mids))]


def shannon_entropy(sequence, n_bins=2):
    """Shannon entropy of a discrete sequence. For trade directions: H=1.0 is random, H<0.7 is predictable."""
    if len(sequence) < 10:
        return 1.0
    counts = Counter(sequence)
    total = len(sequence)
    h = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h


def conditional_entropy(seq, lag=1):
    """H(X_t | X_{t-lag}). Low = previous value predicts current = exploitable."""
    if len(seq) < lag + 10:
        return 1.0
    # Build conditional counts
    joint = Counter()
    marginal = Counter()
    for i in range(lag, len(seq)):
        prev = seq[i - lag]
        curr = seq[i]
        joint[(prev, curr)] += 1
        marginal[prev] += 1

    h_cond = 0
    total = sum(joint.values())
    for (prev, curr), count in joint.items():
        p_joint = count / total
        p_cond = count / marginal[prev]
        if p_cond > 0:
            h_cond -= p_joint * math.log2(p_cond)
    return h_cond


def mutual_information(series_a, series_b, n_bins=10):
    """Mutual information between two continuous series. Discretized into bins.
    MI > 0 means there's exploitable dependency (even nonlinear)."""
    n = min(len(series_a), len(series_b))
    if n < 50:
        return 0.0

    a = series_a[-n:]
    b = series_b[-n:]

    # Discretize into bins
    def discretize(vals, nbins):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0] * len(vals)
        width = (mx - mn) / nbins
        return [min(nbins - 1, int((v - mn) / width)) for v in vals]

    da = discretize(a, n_bins)
    db = discretize(b, n_bins)

    # Joint and marginal distributions
    joint = Counter(zip(da, db))
    marg_a = Counter(da)
    marg_b = Counter(db)

    mi = 0.0
    for (ai, bi), count in joint.items():
        p_joint = count / n
        p_a = marg_a[ai] / n
        p_b = marg_b[bi] / n
        if p_joint > 0 and p_a > 0 and p_b > 0:
            mi += p_joint * math.log2(p_joint / (p_a * p_b))

    return mi


def autocorrelation(series, max_lag=20):
    """Autocorrelation function for lags 1..max_lag."""
    n = len(series)
    if n < max_lag + 10:
        return {}
    mean = sum(series) / n
    var = sum((x - mean)**2 for x in series) / n
    if var < 1e-10:
        return {}

    acf = {}
    for lag in range(1, max_lag + 1):
        cov = sum((series[i] - mean) * (series[i - lag] - mean)
                  for i in range(lag, n)) / n
        acf[lag] = cov / var
    return acf


def hurst_exponent(series, max_lag=100):
    """Hurst exponent via R/S analysis.
    H < 0.5: mean-reverting (anti-persistent)
    H = 0.5: random walk
    H > 0.5: trending (persistent)
    """
    n = len(series)
    if n < 100:
        return 0.5  # default to random walk

    lags = []
    rs_values = []

    for lag in [10, 20, 50, 100]:
        if lag > n // 4:
            continue
        n_segments = n // lag
        rs_list = []
        for seg in range(n_segments):
            chunk = series[seg * lag:(seg + 1) * lag]
            mean_c = sum(chunk) / len(chunk)
            # Cumulative deviation
            cumdev = []
            running = 0
            for x in chunk:
                running += (x - mean_c)
                cumdev.append(running)
            r = max(cumdev) - min(cumdev)
            s = (sum((x - mean_c)**2 for x in chunk) / len(chunk)) ** 0.5
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            lags.append(math.log(lag))
            rs_values.append(math.log(sum(rs_list) / len(rs_list)))

    if len(lags) < 2:
        return 0.5

    # Linear regression: log(R/S) = H * log(n)
    n_pts = len(lags)
    mean_x = sum(lags) / n_pts
    mean_y = sum(rs_values) / n_pts
    num = sum((lags[i] - mean_x) * (rs_values[i] - mean_y) for i in range(n_pts))
    den = sum((lags[i] - mean_x)**2 for i in range(n_pts))
    if abs(den) < 1e-10:
        return 0.5

    H = num / den
    return max(0.0, min(1.0, H))


def detect_lead_lag(returns_a, returns_b, max_lag=10):
    """Cross-correlation at multiple lags to detect lead-lag relationships.
    Returns (best_lag, correlation) where positive lag means A leads B."""
    n = min(len(returns_a), len(returns_b))
    if n < 50:
        return 0, 0.0

    a = returns_a[-n:]
    b = returns_b[-n:]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    std_a = (sum((x - mean_a)**2 for x in a) / n) ** 0.5
    std_b = (sum((x - mean_b)**2 for x in b) / n) ** 0.5
    if std_a < 1e-10 or std_b < 1e-10:
        return 0, 0.0

    best_lag = 0
    best_corr = 0.0

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        corr_sum = 0
        count = 0
        for i in range(max(0, lag), min(n, n + lag)):
            j = i - lag
            if 0 <= j < n:
                corr_sum += (a[j] - mean_a) * (b[i] - mean_b)
                count += 1
        if count > 20:
            corr = corr_sum / (count * std_a * std_b)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

    return best_lag, best_corr


def estimate_vpin(trades, n_buckets=50, bucket_size=None):
    """Volume-synchronized Probability of Informed Trading (VPIN).
    High VPIN (>0.5) = toxic flow, likely informed trading.
    From Easley, López de Prado, O'Hara (2012)."""
    if len(trades) < 100:
        return 0.0

    # Classify trades as buy/sell by price vs running mid
    prices = [t['price'] for t in trades]
    mid = sum(prices) / len(prices)

    if bucket_size is None:
        total_vol = sum(t['quantity'] for t in trades)
        bucket_size = max(1, total_vol // n_buckets)

    # Fill buckets
    buckets = []
    buy_vol = 0
    sell_vol = 0
    bucket_vol = 0
    running_mid = mid

    for t in trades:
        vol = t['quantity']
        # Lee-Ready: classify by price vs mid
        if t['price'] > running_mid:
            buy_vol += vol
        elif t['price'] < running_mid:
            sell_vol += vol
        else:
            # At mid: split 50/50
            buy_vol += vol / 2
            sell_vol += vol / 2

        bucket_vol += vol
        running_mid = 0.99 * running_mid + 0.01 * t['price']

        if bucket_vol >= bucket_size:
            buckets.append((buy_vol, sell_vol))
            buy_vol = 0
            sell_vol = 0
            bucket_vol = 0

    if len(buckets) < 5:
        return 0.0

    # VPIN = mean(|buy_vol - sell_vol| / (buy_vol + sell_vol)) across buckets
    vpins = []
    for bv, sv in buckets:
        total = bv + sv
        if total > 0:
            vpins.append(abs(bv - sv) / total)
    return sum(vpins) / len(vpins)


def fingerprint_bots(trades):
    """Fingerprint trading bots from trade patterns.
    Returns dict of {bot_name: {avg_qty, trade_count, directional_bias, timing_pattern}}."""
    bots = defaultdict(lambda: {'buys': 0, 'sells': 0, 'quantities': [],
                                 'timestamps': [], 'prices': []})

    for t in trades:
        for name, role in [(t.get('buyer', ''), 'buy'), (t.get('seller', ''), 'sell')]:
            if not name or name == 'SUBMISSION':
                continue
            bots[name][f'{role}s'] += 1
            bots[name]['quantities'].append(t['quantity'])
            bots[name]['timestamps'].append(t['ts'])
            bots[name]['prices'].append(t['price'])

    profiles = {}
    for name, data in bots.items():
        total_trades = data['buys'] + data['sells']
        if total_trades < 5:
            continue

        qtys = data['quantities']
        avg_qty = sum(qtys) / len(qtys)
        qty_std = (sum((q - avg_qty)**2 for q in qtys) / len(qtys)) ** 0.5

        # Directional bias: -1 (pure seller) to +1 (pure buyer)
        bias = (data['buys'] - data['sells']) / total_trades if total_trades > 0 else 0

        # Timing: inter-trade interval statistics
        ts = sorted(data['timestamps'])
        intervals = [ts[i] - ts[i-1] for i in range(1, len(ts))]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0

        # Classification
        if avg_qty > 8 and total_trades < 50:
            bot_type = "INFORMED"  # rare, large trades = Olivia pattern
        elif qty_std < 1.0 and total_trades > 100:
            bot_type = "MARKET_MAKER"  # fixed size, high frequency
        elif abs(bias) > 0.7:
            bot_type = "DIRECTIONAL"  # strongly one-sided
        else:
            bot_type = "NOISE"

        profiles[name] = {
            'type': bot_type,
            'total_trades': total_trades,
            'avg_qty': round(avg_qty, 1),
            'qty_std': round(qty_std, 1),
            'directional_bias': round(bias, 2),
            'avg_interval': round(avg_interval, 0),
            'buys': data['buys'],
            'sells': data['sells'],
        }

    return profiles


def spread_analysis(price_data):
    """Analyze spread distribution: mean, percentiles, compression frequency."""
    spreads = [p['spread'] for p in price_data if 'spread' in p]
    if not spreads:
        return {}

    spreads_sorted = sorted(spreads)
    n = len(spreads_sorted)

    # Compression detection: how often does spread go below mean-1std?
    mean_s = sum(spreads) / n
    std_s = (sum((s - mean_s)**2 for s in spreads) / n) ** 0.5
    compression_threshold = max(1, mean_s - std_s)
    compressions = sum(1 for s in spreads if s <= compression_threshold)

    return {
        'mean': round(mean_s, 2),
        'median': round(spreads_sorted[n // 2], 2),
        'p10': round(spreads_sorted[n // 10], 2),
        'p90': round(spreads_sorted[int(n * 0.9)], 2),
        'std': round(std_s, 2),
        'compression_rate': round(compressions / n, 3),
        'min': spreads_sorted[0],
        'max': spreads_sorted[-1],
    }


# ═══════════════════════════════════════════════════════════════
# PRODUCT PROFILER
# ═══════════════════════════════════════════════════════════════

def profile_product(product, price_data, trade_data=None):
    """Deep profile of a single product. Returns dict of insights."""
    profile = {'product': product, 'n_ticks': len(price_data)}

    mids = [p['mid'] for p in price_data]
    returns = compute_returns(mids)

    if not returns:
        return profile

    # ── Basic statistics ──
    vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
    mean_r = sum(returns) / len(returns)
    profile['volatility'] = round(vol, 4)
    profile['mean_return'] = round(mean_r, 6)
    profile['price_range'] = round(max(mids) - min(mids), 2)
    profile['mid_price'] = round(sum(mids) / len(mids), 2)

    # ── Spread analysis ──
    profile['spread'] = spread_analysis(price_data)

    # ── Hurst exponent (fractal) ──
    H = hurst_exponent(mids)
    profile['hurst'] = round(H, 3)
    if H < 0.4:
        profile['hurst_interpretation'] = "STRONGLY_MEAN_REVERTING"
    elif H < 0.48:
        profile['hurst_interpretation'] = "MEAN_REVERTING"
    elif H < 0.52:
        profile['hurst_interpretation'] = "RANDOM_WALK"
    elif H < 0.6:
        profile['hurst_interpretation'] = "TRENDING"
    else:
        profile['hurst_interpretation'] = "STRONGLY_TRENDING"

    # ── Autocorrelation structure ──
    acf = autocorrelation(returns, max_lag=10)
    profile['acf'] = {str(k): round(v, 3) for k, v in acf.items()}
    significant_lags = [lag for lag, corr in acf.items()
                        if abs(corr) > 2.0 / len(returns)**0.5]  # 95% CI
    profile['significant_acf_lags'] = significant_lags

    # ── Information theory: entropy ──
    # Discretize returns into {up, down, flat}
    directions = []
    for r in returns:
        if r > vol * 0.1:
            directions.append(1)  # up
        elif r < -vol * 0.1:
            directions.append(-1)  # down
        else:
            directions.append(0)  # flat

    profile['entropy'] = round(shannon_entropy(directions, n_bins=3), 3)
    profile['cond_entropy_lag1'] = round(conditional_entropy(directions, lag=1), 3)
    profile['cond_entropy_lag2'] = round(conditional_entropy(directions, lag=2), 3)

    # Information gain: H(X) - H(X|X_{t-1})
    info_gain = profile['entropy'] - profile['cond_entropy_lag1']
    profile['info_gain_lag1'] = round(info_gain, 3)

    if info_gain > 0.15:
        profile['entropy_interpretation'] = "HIGHLY_PREDICTABLE"
    elif info_gain > 0.05:
        profile['entropy_interpretation'] = "MODERATELY_PREDICTABLE"
    else:
        profile['entropy_interpretation'] = "LOW_PREDICTABILITY"

    # ── Trade analysis (if available) ──
    if trade_data:
        profile['n_trades'] = len(trade_data)
        profile['vpin'] = round(estimate_vpin(trade_data), 3)
        profile['bot_profiles'] = fingerprint_bots(trade_data)

        if profile['vpin'] > 0.5:
            profile['vpin_interpretation'] = "HIGH_TOXICITY"
        elif profile['vpin'] > 0.3:
            profile['vpin_interpretation'] = "MODERATE_TOXICITY"
        else:
            profile['vpin_interpretation'] = "LOW_TOXICITY"

    # ── Strategy recommendation ──
    profile['recommended_type'] = recommend_strategy(profile)

    return profile


def recommend_strategy(profile):
    """Recommend strategy type based on profile."""
    H = profile.get('hurst', 0.5)
    entropy = profile.get('entropy', 1.0)
    info_gain = profile.get('info_gain_lag1', 0)
    spread = profile.get('spread', {})
    mean_spread = spread.get('mean', 4)
    vol = profile.get('volatility', 1)
    vpin = profile.get('vpin', 0)
    mid = profile.get('mid_price', 1000)

    # Pegged: very low vol, integer-like mid, narrow spread
    if vol < 0.5 and mid % 100 < 5:
        return "pegged"

    # Wide spread: high mean spread relative to price
    if mean_spread / mid > 0.001 and spread.get('compression_rate', 0) > 0.05:
        return "wide_spread"

    # High predictability: AR model will work
    if info_gain > 0.1 and H < 0.5:
        return "ar_olivia"

    # High toxicity: informed trader present
    if vpin > 0.4:
        return "olivia_follow"

    # Trending: follow momentum
    if H > 0.6:
        return "ar_olivia"

    # Default
    return "ar_olivia"


# ═══════════════════════════════════════════════════════════════
# CROSS-PRODUCT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def cross_product_analysis(products_data):
    """Analyze relationships between products."""
    results = {}
    product_names = list(products_data.keys())

    if len(product_names) < 2:
        return results

    # Compute returns for all products
    all_returns = {}
    for p, data in products_data.items():
        mids = [d['mid'] for d in data]
        rets = compute_returns(mids)
        if len(rets) > 50:
            all_returns[p] = rets

    # Pairwise analysis
    pairs = []
    for i, p1 in enumerate(product_names):
        if p1 not in all_returns:
            continue
        for p2 in product_names[i+1:]:
            if p2 not in all_returns:
                continue

            # Correlation
            r1 = all_returns[p1]
            r2 = all_returns[p2]
            n = min(len(r1), len(r2))
            if n < 50:
                continue

            r1_n = r1[-n:]
            r2_n = r2[-n:]
            m1 = sum(r1_n) / n
            m2 = sum(r2_n) / n
            cov = sum((r1_n[i] - m1) * (r2_n[i] - m2) for i in range(n)) / n
            v1 = sum((x - m1)**2 for x in r1_n) / n
            v2 = sum((x - m2)**2 for x in r2_n) / n
            corr = cov / (v1**0.5 * v2**0.5) if v1 > 0 and v2 > 0 else 0

            # Mutual information (captures nonlinear dependencies)
            mi = mutual_information(r1_n, r2_n)

            # Lead-lag
            lag, lag_corr = detect_lead_lag(r1_n, r2_n)

            pair_result = {
                'products': [p1, p2],
                'correlation': round(corr, 3),
                'mutual_info': round(mi, 4),
                'lead_lag': lag,
                'lead_lag_corr': round(lag_corr, 3),
            }

            if abs(lag) > 0 and abs(lag_corr) > 0.15:
                leader = p1 if lag > 0 else p2
                follower = p2 if lag > 0 else p1
                pair_result['interpretation'] = f"{leader} LEADS {follower} by {abs(lag)} ticks"

            if mi > 0.1:
                pair_result['mi_interpretation'] = "STRONG_DEPENDENCY"
            elif mi > 0.03:
                pair_result['mi_interpretation'] = "MODERATE_DEPENDENCY"

            pairs.append(pair_result)

    results['pairs'] = sorted(pairs, key=lambda x: -abs(x.get('mutual_info', 0)))
    return results


# ═══════════════════════════════════════════════════════════════
# CONFIG HINT GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_config_hints(profiles, cross_analysis):
    """Generate CONFIG hints for the frontier engine based on profiles."""
    hints = {}

    for p_profile in profiles:
        product = p_profile['product']
        h = {}

        # Entropy-based bot detection hint
        if p_profile.get('info_gain_lag1', 0) > 0.1:
            h['hint_entropy_threshold'] = round(
                max(0.5, p_profile['entropy'] - 0.2), 2)

        # Hurst-based strategy hint
        H = p_profile.get('hurst', 0.5)
        if H < 0.45:
            h['hint_mean_reversion_strength'] = round(0.5 - H, 2)
        elif H > 0.55:
            h['hint_momentum_strength'] = round(H - 0.5, 2)

        # VPIN-based toxicity hint
        if p_profile.get('vpin', 0) > 0.3:
            h['hint_high_toxicity'] = True

        # Spread compression hint
        comp_rate = p_profile.get('spread', {}).get('compression_rate', 0)
        if comp_rate > 0.1:
            h['hint_compression_freq'] = round(comp_rate, 3)

        # Recommended type
        h['recommended_type'] = p_profile.get('recommended_type', 'ar_olivia')

        hints[product] = h

    # Cross-product hints
    pairs = cross_analysis.get('pairs', [])
    for pair in pairs:
        lag = pair.get('lead_lag', 0)
        lag_corr = pair.get('lead_lag_corr', 0)
        mi = pair.get('mutual_info', 0)

        if abs(lag) > 0 and abs(lag_corr) > 0.15:
            leader = pair['products'][0] if lag > 0 else pair['products'][1]
            follower = pair['products'][1] if lag > 0 else pair['products'][0]
            if follower in hints:
                hints[follower]['hint_lead_product'] = leader
                hints[follower]['hint_lead_lag'] = abs(lag)
                hints[follower]['hint_lead_corr'] = abs(lag_corr)

        if mi > 0.05:
            for p in pair['products']:
                if p in hints:
                    partner = [x for x in pair['products'] if x != p][0]
                    hints[p]['hint_mi_partner'] = partner
                    hints[p]['hint_mi_value'] = mi

    return hints


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

def print_report(profiles, cross_analysis, hints):
    """Print human-readable analysis report."""
    print("\n" + "=" * 80)
    print("  DATA PROFILER — Deep Product Analysis")
    print("=" * 80)

    for p in profiles:
        product = p['product']
        print(f"\n{'-' * 60}")
        print(f"  {product} ({p['n_ticks']} ticks)")
        print(f"{'-' * 60}")

        print(f"  Price:      mid={p.get('mid_price', '?')}, range={p.get('price_range', '?')}")
        print(f"  Volatility: {p.get('volatility', '?')}")

        sp = p.get('spread', {})
        if sp:
            print(f"  Spread:     mean={sp.get('mean')}, median={sp.get('median')}, "
                  f"compression={sp.get('compression_rate', 0)*100:.1f}%")

        print(f"\n  Hurst:      H={p.get('hurst', '?')} -> {p.get('hurst_interpretation', '?')}")

        print(f"  Entropy:    H(X)={p.get('entropy', '?')}, "
              f"H(X|X_{{t-1}})={p.get('cond_entropy_lag1', '?')}, "
              f"info_gain={p.get('info_gain_lag1', '?')}")
        print(f"              -> {p.get('entropy_interpretation', '?')}")

        acf = p.get('acf', {})
        sig_lags = p.get('significant_acf_lags', [])
        if sig_lags:
            print(f"  ACF:        significant at lags {sig_lags}")
            for lag in sig_lags[:3]:
                print(f"              lag-{lag}: {acf.get(str(lag), '?')}")

        if 'vpin' in p:
            print(f"  VPIN:       {p['vpin']} -> {p.get('vpin_interpretation', '?')}")

        if 'bot_profiles' in p:
            bots = p['bot_profiles']
            informed = [b for b, d in bots.items() if d['type'] == 'INFORMED']
            mms = [b for b, d in bots.items() if d['type'] == 'MARKET_MAKER']
            print(f"  Bots:       {len(bots)} detected, "
                  f"{len(informed)} informed, {len(mms)} market-makers")
            for name in informed[:3]:
                d = bots[name]
                print(f"              {name}: {d['total_trades']} trades, "
                      f"avg_qty={d['avg_qty']}, bias={d['directional_bias']}")

        h = hints.get(product, {})
        rec = h.get('recommended_type', '?')
        print(f"\n  -> RECOMMENDED: {rec}")
        if 'hint_lead_product' in h:
            print(f"  -> LEAD-LAG: {h['hint_lead_product']} leads by "
                  f"{h.get('hint_lead_lag', '?')} ticks "
                  f"(corr={h.get('hint_lead_corr', '?')})")
        if h.get('hint_high_toxicity'):
            print(f"  -> HIGH TOXICITY: informed trader likely present")
        if 'hint_mean_reversion_strength' in h:
            print(f"  -> MEAN REVERSION: strength={h['hint_mean_reversion_strength']}")
        if 'hint_momentum_strength' in h:
            print(f"  -> MOMENTUM: strength={h['hint_momentum_strength']}")

    # Cross-product
    pairs = cross_analysis.get('pairs', [])
    if pairs:
        print(f"\n{'-' * 60}")
        print(f"  CROSS-PRODUCT RELATIONSHIPS")
        print(f"{'-' * 60}")
        for pair in pairs[:10]:
            p1, p2 = pair['products']
            print(f"\n  {p1} x {p2}:")
            print(f"    Correlation: {pair['correlation']}")
            print(f"    Mutual Info: {pair['mutual_info']} "
                  f"({pair.get('mi_interpretation', 'weak')})")
            if pair.get('interpretation'):
                print(f"    Lead-Lag:    {pair['interpretation']} "
                      f"(corr={pair['lead_lag_corr']})")


def main():
    parser = argparse.ArgumentParser(description="Deep Product Data Profiler")
    parser.add_argument("prices", nargs="+", help="Price CSV files")
    parser.add_argument("--trades", nargs="*", default=None, help="Trade CSV files")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    # Expand globs on Windows
    import glob as globmod
    price_files = []
    for p in args.prices:
        expanded = sorted(globmod.glob(p))
        price_files.extend(expanded if expanded else [p])

    trade_files = []
    if args.trades:
        for t in args.trades:
            expanded = sorted(globmod.glob(t))
            trade_files.extend(expanded if expanded else [t])

    print(f"Loading {len(price_files)} price files...")
    products_data = load_prices(price_files)

    trades_data = {}
    if trade_files:
        print(f"Loading {len(trade_files)} trade files...")
        trades_data = load_trades(trade_files)

    print(f"Found {len(products_data)} products: {', '.join(sorted(products_data.keys()))}")

    # Profile each product
    profiles = []
    for product in sorted(products_data.keys()):
        profile = profile_product(product, products_data[product],
                                   trades_data.get(product))
        profiles.append(profile)

    # Cross-product analysis
    cross = cross_product_analysis(products_data)

    # Generate CONFIG hints
    hints = generate_config_hints(profiles, cross)

    # Print report
    print_report(profiles, cross, hints)

    # Print CONFIG hints summary
    print(f"\n{'=' * 80}")
    print(f"  CONFIG HINTS (paste into CONFIG or use as frontier hints)")
    print(f"{'=' * 80}")
    print(json.dumps(hints, indent=2))

    # Save JSON if requested
    if args.output:
        output = {
            'profiles': profiles,
            'cross_product': cross,
            'config_hints': hints,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
