"""v9.0 sanity check: does cross-strike IV residual mispricing exist in P4R3 data?

Reads 30K-tick VEV prices, BS-inverts IV per strike per tick, fits linear smile in
moneyness, computes per-strike z-score of residuals over rolling window. Reports
distribution of |z| to decide whether v9 IV-arb play is real."""
import csv, math, statistics
from collections import defaultdict
from glob import glob

# BS helpers (lifted from trader.py:3057+)
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(S, K, T, sigma, r=0.0):
    if sigma <= 1e-9 or T <= 1e-9:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_vega(S, K, T, sigma):
    if sigma <= 1e-9 or T <= 1e-9:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * _norm_pdf(d1)

def implied_vol(price, S, K, T, max_iter=20, tol=1e-5):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-3:
        return None  # zero-vol or intrinsic-only
    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        c = bs_call(S, K, T, sigma)
        v = bs_vega(S, K, T, sigma)
        if v < 1e-9:
            return None
        diff = price - c
        if abs(diff) < tol:
            return sigma
        sigma = sigma + diff / v
        if sigma <= 1e-4 or sigma > 5.0:
            return None
    return sigma

# Load data
files = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
print(f"Loading {len(files)} files...")

# rows[ts][product] = mid
rows = defaultdict(dict)
for f in files:
    with open(f) as fh:
        reader = csv.DictReader(fh, delimiter=';')
        day = None
        for r in reader:
            day = int(r['day'])
            ts = int(r['timestamp']) + day * 1_000_000
            try:
                rows[ts][r['product']] = float(r['mid_price'])
            except (ValueError, KeyError):
                continue

ts_sorted = sorted(rows.keys())
print(f"Ticks: {len(ts_sorted)}, range: [{min(ts_sorted)}, {max(ts_sorted)}]")

# Strikes to check
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TTE = 5.0 / 365.0  # P4R3 5 days, year fraction
UNDERLYING = 'VELVETFRUIT_EXTRACT'

# IV per strike per tick
iv_by_strike = {K: [] for K in STRIKES}
ts_seen = []

for ts in ts_sorted:
    if UNDERLYING not in rows[ts]:
        continue
    S = rows[ts][UNDERLYING]
    ts_seen.append(ts)
    for K in STRIKES:
        prod = f'VEV_{K}'
        if prod not in rows[ts]:
            iv_by_strike[K].append(None)
            continue
        price = rows[ts][prod]
        iv = implied_vol(price, S, K, TTE)
        iv_by_strike[K].append(iv)

print(f"Computed IV for {len(ts_seen)} ticks across {len(STRIKES)} strikes")

# Per-strike statistics
print("\n=== IV BY STRIKE ===")
print(f"{'strike':>8} {'%valid':>8} {'mean_IV':>10} {'std_IV':>10} {'min_IV':>10} {'max_IV':>10}")
for K in STRIKES:
    vals = [v for v in iv_by_strike[K] if v is not None]
    if not vals:
        print(f"{K:>8} {'0%':>8} {'-':>10}")
        continue
    pct = 100 * len(vals) / len(ts_seen)
    print(f"{K:>8} {pct:>7.1f}% {statistics.mean(vals):>10.4f} {statistics.pstdev(vals):>10.4f} {min(vals):>10.4f} {max(vals):>10.4f}")

# Per-tick smile fit + residual
# Linear in moneyness: IV = a + b * (K/S)
def fit_linear(xs, ys):
    n = len(xs)
    if n < 2:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    if sxx < 1e-9:
        return my, 0.0
    b = sxy / sxx
    a = my - b * mx
    return a, b

residuals = {K: [] for K in STRIKES}
for i, ts in enumerate(ts_seen):
    S = rows[ts][UNDERLYING]
    pts = []  # (K/S, IV, K)
    for K in STRIKES:
        iv = iv_by_strike[K][i]
        if iv is not None and 0.05 < iv < 2.0:  # filter outliers
            pts.append((K / S, iv, K))
    if len(pts) < 4:
        for K in STRIKES:
            residuals[K].append(None)
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    a, b = fit_linear(xs, ys)
    if a is None:
        for K in STRIKES:
            residuals[K].append(None)
        continue
    fit_K = {p[2]: a + b * p[0] for p in pts}
    for K in STRIKES:
        if K in fit_K:
            iv = iv_by_strike[K][i]
            residuals[K].append(iv - fit_K[K])
        else:
            residuals[K].append(None)

# Residual stats per strike
print("\n=== RESIDUAL Z-SCORE (residual / std_residual) ===")
print(f"{'strike':>8} {'%valid':>8} {'mean_res':>10} {'std_res':>10} {'%|z|>1':>9} {'%|z|>2':>9} {'%|z|>3':>9}")
for K in STRIKES:
    vals = [r for r in residuals[K] if r is not None]
    if len(vals) < 100:
        print(f"{K:>8} insufficient")
        continue
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals)
    if sd < 1e-9:
        print(f"{K:>8} zero variance")
        continue
    z_scores = [(r - mu) / sd for r in vals]
    p_z1 = 100 * sum(1 for z in z_scores if abs(z) > 1) / len(z_scores)
    p_z2 = 100 * sum(1 for z in z_scores if abs(z) > 2) / len(z_scores)
    p_z3 = 100 * sum(1 for z in z_scores if abs(z) > 3) / len(z_scores)
    pct = 100 * len(vals) / len(ts_seen)
    print(f"{K:>8} {pct:>7.1f}% {mu:>10.4f} {sd:>10.4f} {p_z1:>8.1f}% {p_z2:>8.1f}% {p_z3:>8.1f}%")

# Persistence: how long does |z|>2 stay |z|>2? (mean reversion timescale)
print("\n=== Z-SCORE MEAN-REVERSION TIMESCALE ===")
print(f"{'strike':>8} {'avg_run_len_>2':>15} {'max_run_>2':>12} {'#crossings':>12}")
for K in STRIKES:
    vals = [r for r in residuals[K] if r is not None]
    if len(vals) < 100:
        continue
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals)
    if sd < 1e-9:
        continue
    z_scores = [(r - mu) / sd for r in vals]
    runs = []
    cur = 0
    crossings = 0
    prev_in = False
    for z in z_scores:
        in_state = abs(z) > 2
        if in_state:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
        if in_state != prev_in:
            crossings += 1
        prev_in = in_state
    if cur > 0:
        runs.append(cur)
    if runs:
        avg_run = sum(runs) / len(runs)
        print(f"{K:>8} {avg_run:>14.1f} {max(runs):>11d} {crossings:>11d}")
    else:
        print(f"{K:>8} no |z|>2 events")
