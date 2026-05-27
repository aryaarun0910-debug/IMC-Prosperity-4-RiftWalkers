"""Pre-fit GARCH(1,1) and polynomial smile constants on P3 R3 data.

Outputs distilled constants written to analysis/findings/prefit_constants.json
that ship as in-trader.py defaults (no runtime numpy needed).

Usage: py -3.12 analysis/prefit_constants.py
"""
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("reference/p3_data/Round 3 Data")
OUT = Path("analysis/findings/prefit_constants.json")


def load_mids(product_filter=None):
    """Per-product list of (timestamp, mid) tuples sorted in time."""
    out = defaultdict(list)
    for d in (0, 1, 2):
        fp = DATA_DIR / f"prices_round_3_day_{d}.csv"
        if not fp.exists():
            continue
        with open(fp) as f:
            rdr = csv.DictReader(f, delimiter=";")
            for row in rdr:
                p = row["product"]
                if product_filter and p not in product_filter:
                    continue
                try:
                    bid = float(row["bid_price_1"]) if row.get("bid_price_1") else None
                    ask = float(row["ask_price_1"]) if row.get("ask_price_1") else None
                    if bid is None or ask is None:
                        continue
                    mid = (bid + ask) / 2.0
                    ts = int(row["timestamp"]) + d * 1_000_000
                    out[p].append((ts, mid))
                except Exception:
                    continue
    for p in out:
        out[p].sort()
    return out


def fit_garch_11(returns):
    """Tiny GARCH(1,1) MLE via Nelder-Mead-ish brute force.
    Returns (omega, alpha, beta) or None if fit fails.
    """
    if len(returns) < 200:
        return None
    var_unc = sum(r * r for r in returns) / len(returns)
    if var_unc <= 0:
        return None

    def neg_ll(omega, alpha, beta):
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e18
        h = var_unc
        ll = 0.0
        for r in returns:
            if h <= 1e-12:
                return 1e18
            ll += 0.5 * (math.log(h) + r * r / h)
            h = omega + alpha * r * r + beta * h
        return ll

    best = None
    for alpha in (0.05, 0.08, 0.10, 0.15, 0.20):
        for beta in (0.70, 0.80, 0.85, 0.90, 0.93):
            if alpha + beta >= 0.99:
                continue
            omega = var_unc * (1 - alpha - beta)
            if omega <= 0:
                continue
            ll = neg_ll(omega, alpha, beta)
            if best is None or ll < best[0]:
                best = (ll, omega, alpha, beta)
    if best is None:
        return None
    return {"omega": best[1], "alpha": best[2], "beta": best[3], "var_unc": var_unc}


def fit_polynomial_smile(strike_to_iv, deg=2):
    """Fit y = c0 + c1*x + c2*x^2 via OLS normal equations.
    x = log(strike/spot), y = iv. Returns coeffs or None.
    """
    if len(strike_to_iv) < deg + 1:
        return None
    xs = list(strike_to_iv.keys())
    ys = list(strike_to_iv.values())
    n = len(xs)
    # Build X^T X (deg+1 x deg+1) and X^T y
    p = deg + 1
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for x, y in zip(xs, ys):
        row = [x ** k for k in range(p)]
        for i in range(p):
            Xty[i] += row[i] * y
            for j in range(p):
                XtX[i][j] += row[i] * row[j]
    # Tiny Gauss-Jordan inverse
    aug = [r + ([1.0 if i == j else 0.0 for j in range(p)]) for i, r in enumerate(XtX)]
    for i in range(p):
        piv = aug[i][i]
        if abs(piv) < 1e-12:
            return None
        for j in range(2 * p):
            aug[i][j] /= piv
        for k in range(p):
            if k == i:
                continue
            f = aug[k][i]
            for j in range(2 * p):
                aug[k][j] -= f * aug[i][j]
    inv = [r[p:] for r in aug]
    coeffs = [sum(inv[i][j] * Xty[j] for j in range(p)) for i in range(p)]
    return coeffs


def main():
    print(f"Loading P3 R3 mids from {DATA_DIR} ...")
    mids = load_mids()
    print(f"Products loaded: {len(mids)}")

    out = {"_meta": {"source": "P3 R3 days 0-2", "samples_per_product": {}}}

    for product, series in mids.items():
        prices = [m for _, m in series]
        if len(prices) < 200:
            continue
        out["_meta"]["samples_per_product"][product] = len(prices)
        # Returns (log returns, capped to remove outliers)
        rets = []
        for i in range(1, len(prices)):
            if prices[i - 1] <= 0:
                continue
            r = math.log(prices[i] / prices[i - 1])
            if abs(r) > 0.05:
                continue
            rets.append(r)
        if len(rets) < 200:
            continue
        gar = fit_garch_11(rets)
        if gar:
            out[f"garch_{product}"] = {
                "omega": round(gar["omega"], 12),
                "alpha": round(gar["alpha"], 4),
                "beta": round(gar["beta"], 4),
                "var_unc": round(gar["var_unc"], 12),
                "vol_unc": round(math.sqrt(gar["var_unc"]), 6),
                "n_returns": len(rets),
            }

    # SVI / polynomial smile fit on VOUCHERS
    voucher_strikes = {}
    for product in mids:
        if "VOUCHER" in product:
            try:
                strike = int(product.split("_")[-1])
                voucher_strikes[product] = strike
            except Exception:
                pass

    if "VOLCANIC_ROCK" in mids and voucher_strikes:
        rock_series = dict(mids["VOLCANIC_ROCK"])
        # Sample 5 timepoints across days for smile snapshots
        rock_ts = sorted(rock_series.keys())
        snapshots = [rock_ts[len(rock_ts) * i // 5] for i in range(1, 5)]
        smile_fits = []
        for ts in snapshots:
            spot = rock_series.get(ts)
            if spot is None or spot <= 0:
                continue
            strike_iv = {}
            for vp, strike in voucher_strikes.items():
                series_dict = dict(mids[vp])
                vmid = series_dict.get(ts)
                if vmid is None or vmid <= 0:
                    continue
                # Approximate IV via simple ATM-equivalence (no time decay)
                # We just fit the OBSERVED voucher mid as a function of moneyness here,
                # which gives us the "premium curve" — sufficient for warmup constants.
                moneyness = math.log(strike / spot)
                strike_iv[moneyness] = vmid
            if len(strike_iv) >= 3:
                cf = fit_polynomial_smile(strike_iv, deg=2)
                if cf:
                    smile_fits.append(cf)
        if smile_fits:
            avg = [sum(c[i] for c in smile_fits) / len(smile_fits) for i in range(3)]
            out["smile_voucher_premium_poly2"] = {
                "c0": round(avg[0], 6),
                "c1": round(avg[1], 6),
                "c2": round(avg[2], 6),
                "n_snapshots": len(smile_fits),
                "model": "premium = c0 + c1*log(K/S) + c2*log(K/S)^2",
            }

    # BasketSpreadManager-style hardcoded spread mean for PICNIC_BASKET1
    # NAV1 = 6*CROISSANTS + 3*JAMS + 1*DJEMBES
    if all(p in mids for p in ("CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1")):
        c_d = dict(mids["CROISSANTS"])
        j_d = dict(mids["JAMS"])
        d_d = dict(mids["DJEMBES"])
        b_d = dict(mids["PICNIC_BASKET1"])
        common_ts = sorted(set(c_d) & set(j_d) & set(d_d) & set(b_d))
        spreads = []
        for ts in common_ts:
            nav = 6 * c_d[ts] + 3 * j_d[ts] + 1 * d_d[ts]
            spreads.append(b_d[ts] - nav)
        if spreads:
            mu = sum(spreads) / len(spreads)
            var = sum((s - mu) ** 2 for s in spreads) / len(spreads)
            out["basket1_spread_prior"] = {
                "mean": round(mu, 4),
                "std": round(math.sqrt(var), 4),
                "n": len(spreads),
                "nav_formula": "6*CROISSANTS + 3*JAMS + 1*DJEMBES",
            }

    if all(p in mids for p in ("CROISSANTS", "JAMS", "PICNIC_BASKET2")):
        c_d = dict(mids["CROISSANTS"])
        j_d = dict(mids["JAMS"])
        b_d = dict(mids["PICNIC_BASKET2"])
        common_ts = sorted(set(c_d) & set(j_d) & set(b_d))
        spreads = []
        for ts in common_ts:
            nav = 4 * c_d[ts] + 2 * j_d[ts]
            spreads.append(b_d[ts] - nav)
        if spreads:
            mu = sum(spreads) / len(spreads)
            var = sum((s - mu) ** 2 for s in spreads) / len(spreads)
            out["basket2_spread_prior"] = {
                "mean": round(mu, 4),
                "std": round(math.sqrt(var), 4),
                "n": len(spreads),
                "nav_formula": "4*CROISSANTS + 2*JAMS",
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT} with {len(out) - 1} fitted entries")
    for k, v in out.items():
        if k == "_meta":
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
