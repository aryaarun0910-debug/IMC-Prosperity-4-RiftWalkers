"""Offline smile calibrator for p4r3 VEV voucher chain.

Reads prices_round_3_day_[0,1,2].csv, inverts voucher wall_mids to IV via BS
Newton iteration, fits quadratic IV = a*m^2 + b*m + c, validates OOS on day 2.

Output: calibrated smile_coeffs to paste into trader.py VEV config.

Usage:
  py -3.12 analysis/fit_p4r3_smile.py
"""
import math
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "r3", "prices")

# BS helpers (replicate trader.py exactly)
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def bs_vega(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return S * math.sqrt(T) * pdf

def implied_vol(price, S, K, T, max_iter=50, tol=1e-5):
    """Newton-Raphson IV inversion. Returns None if no convergence or intrinsic."""
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6:
        return None
    if T <= 0 or S <= 0:
        return None
    sigma = 0.2  # initial guess
    for _ in range(max_iter):
        bs = bs_call(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        if vega < 1e-8:
            return None
        diff = bs - price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        if sigma <= 1e-4 or sigma > 5.0:
            return None
    return None


VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
    "VEV_6500": 6500,
}


def wall_mid(row):
    """Mid-price of the largest-volume bid/ask (robust against 1-lot quotes)."""
    bids = []
    asks = []
    for i in range(1, 4):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv > 0:
            bids.append((float(bp), float(bv)))
        if pd.notna(ap) and pd.notna(av) and av > 0:
            asks.append((float(ap), float(av)))
    if not bids or not asks:
        return None
    # Max volume on each side
    bmax = max(bids, key=lambda x: x[1])[0]
    amax = max(asks, key=lambda x: x[1])[0]
    return 0.5 * (bmax + amax)


def tte_from(day, ts, total_days=7, day_offset=0):
    """Replicate trader.py TTE formula."""
    d = day + day_offset
    return max(1.0 - (365.0 - total_days + d + ts / 100.0 / 10000.0) / 365.0, 1e-6)


def collect_iv_observations(df, day_offset=0, total_days=7):
    """For each (day, ts), compute underlying mid + per-strike IV.

    Returns list of (moneyness, iv, strike, day) tuples.
    """
    # Pivot by (day, timestamp) -> {product: row}
    grouped = df.groupby(["day", "timestamp"])
    obs = []
    skipped_no_ul = 0
    skipped_no_iv = 0
    for (day, ts), group in grouped:
        rows = {r["product"]: r for _, r in group.iterrows()}
        ul_row = rows.get("VELVETFRUIT_EXTRACT")
        if ul_row is None:
            skipped_no_ul += 1
            continue
        S = wall_mid(ul_row)
        if S is None:
            skipped_no_ul += 1
            continue
        T = tte_from(day, ts, total_days=total_days, day_offset=day_offset)
        for strike_name, K in VEV_STRIKES.items():
            v_row = rows.get(strike_name)
            if v_row is None:
                continue
            price = wall_mid(v_row)
            if price is None:
                continue
            iv = implied_vol(price, S, K, T)
            if iv is None:
                skipped_no_iv += 1
                continue
            # Moneyness: log(K/S) normalized by sqrt(T) — matches p3 parameterization
            m = math.log(K / S) / math.sqrt(T)
            obs.append((m, iv, K, day, ts, S))
    print(f"  collected {len(obs)} obs (skipped {skipped_no_ul} no-UL, {skipped_no_iv} no-IV)")
    return obs


def fit_quadratic(obs):
    """IV = a*m^2 + b*m + c via least squares. Returns (a, b, c, residuals)."""
    m = np.array([o[0] for o in obs])
    iv = np.array([o[1] for o in obs])
    X = np.column_stack([m * m, m, np.ones_like(m)])
    coef, *_ = np.linalg.lstsq(X, iv, rcond=None)
    pred = X @ coef
    resid = iv - pred
    return coef[0], coef[1], coef[2], resid


def main():
    day_files = [os.path.join(DATA, f"prices_round_3_day_{d}.csv") for d in [0, 1, 2]]
    frames = [pd.read_csv(f, sep=";") for f in day_files]
    for fr in frames:
        fr.columns = [c.strip().lower() for c in fr.columns]
    day_dfs = frames
    # v10 (2026-04-25): wiki says TTE=5 days for P4R3 vouchers (not 7 from p3).
    # Allow override via CLI: `py -3.12 analysis/fit_p4r3_smile.py 5`
    total_days = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"=== Using total_days={total_days} ===")

    print("\n=== Day 0 fit ===")
    obs0 = collect_iv_observations(day_dfs[0].assign(day=0), day_offset=0, total_days=total_days)
    a0, b0, c0, r0 = fit_quadratic(obs0) if obs0 else (0, 0, 0, np.array([]))
    print(f"  coefs: [a={a0:.6f}, b={b0:.6f}, c={c0:.6f}]")
    if len(r0):
        print(f"  residuals: mean_abs={np.mean(np.abs(r0)):.4f}, p95={np.percentile(np.abs(r0), 95):.4f}")

    print("\n=== Day 1 fit ===")
    obs1 = collect_iv_observations(day_dfs[1].assign(day=1), day_offset=0, total_days=total_days)
    a1, b1, c1, r1 = fit_quadratic(obs1) if obs1 else (0, 0, 0, np.array([]))
    print(f"  coefs: [a={a1:.6f}, b={b1:.6f}, c={c1:.6f}]")
    if len(r1):
        print(f"  residuals: mean_abs={np.mean(np.abs(r1)):.4f}, p95={np.percentile(np.abs(r1), 95):.4f}")

    print("\n=== Day 0+1 combined fit (training) ===")
    obs_train = obs0 + obs1
    if obs_train:
        aT, bT, cT, rT = fit_quadratic(obs_train)
        print(f"  coefs: [a={aT:.6f}, b={bT:.6f}, c={cT:.6f}]")
        print(f"  residuals: mean_abs={np.mean(np.abs(rT)):.4f}, p95={np.percentile(np.abs(rT), 95):.4f}")
    else:
        aT = bT = cT = 0

    print("\n=== Day 2 OOS validation ===")
    obs2 = collect_iv_observations(day_dfs[2].assign(day=2), day_offset=0, total_days=total_days)
    if obs2:
        m2 = np.array([o[0] for o in obs2])
        iv2 = np.array([o[1] for o in obs2])
        pred2 = aT * m2 * m2 + bT * m2 + cT
        resid2 = iv2 - pred2
        print(f"  OOS residuals: mean_abs={np.mean(np.abs(resid2)):.4f}, p95={np.percentile(np.abs(resid2), 95):.4f}")
        print(f"  OOS signal: mean |resid| / mean IV = {np.mean(np.abs(resid2))/np.mean(iv2):.2%}")

        # Per-strike residual breakdown (bias detection)
        print(f"\n  Per-strike residuals (OOS day 2):")
        by_k = defaultdict(list)
        for o, r in zip(obs2, resid2):
            by_k[o[2]].append(r)
        for K in sorted(by_k):
            rs = np.array(by_k[K])
            print(f"    K={K}: n={len(rs):4}  mean={rs.mean():+.4f}  std={rs.std():.4f}")

    print("\n=== RECOMMENDED COEFS ===")
    print(f"  \"smile_coeffs\": [{aT:.8f}, {bT:.8f}, {cT:.8f}]")
    print(f"  (fit on days 0+1, validated on day 2)")

    # Compare to p3 static
    print("\n=== P3 static coefs for reference ===")
    p3 = [0.27362531, 0.01007566, 0.14876677]
    print(f"  {p3}")
    if obs_train:
        pred_p3 = p3[0]*m2*m2 + p3[1]*m2 + p3[2]
        resid_p3 = iv2 - pred_p3
        print(f"  P3 coefs on p4r3 day 2: mean_abs_resid={np.mean(np.abs(resid_p3)):.4f}, p95={np.percentile(np.abs(resid_p3), 95):.4f}")


if __name__ == "__main__":
    main()
