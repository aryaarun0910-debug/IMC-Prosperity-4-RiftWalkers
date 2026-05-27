"""
K4 — OLS trade attribution.

Two ingestion modes:

1. CSV mode (legacy): explicit columns including `pnl` target.
       py -3.12 analysis/trade_attribution.py --log path.csv

2. JSONL mode (Gap 5): consumes the K4 fill log emitted by trader.py when
   IMC_TRADE_LOG_PATH is set. Pairs each fill (k=="fill") with the future
   mid recorded `--markout` ticks later to compute per-fill realized PnL,
   then regresses PnL against [obi, vpin, garch_sigma, position, olv_active].
       IMC_TRADE_LOG_PATH=/tmp/k4_fills.jsonl py -3.12 backtest.py --round p4r1
       py -3.12 analysis/trade_attribution.py --jsonl /tmp/k4_fills.jsonl --markout 30

Output: per-feature coefficient + R^2. Identifies which signals produce
edge versus noise. Use the sign + magnitude to decide what to amplify or
gate in the next ship.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


def ols_multi(X, y):
    """OLS: X is list of rows, y is list. Returns (coefs, residuals)."""
    n = len(X)
    if n == 0:
        return None, None
    p = len(X[0])
    xtx = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    # Gauss-Jordan solve
    A = [xtx[i] + [xty[i]] for i in range(p)]
    for i in range(p):
        piv = A[i][i]
        if abs(piv) < 1e-12:
            return None, None
        for j in range(p + 1):
            A[i][j] /= piv
        for k in range(p):
            if k != i:
                f = A[k][i]
                for j in range(p + 1):
                    A[k][j] -= f * A[i][j]
    coefs = [A[i][p] for i in range(p)]
    resid = [y[i] - sum(coefs[a] * X[i][a] for a in range(p)) for i in range(n)]
    return coefs, resid


def load_jsonl_fills(path, markout_ticks):
    """Read the K4 fill log; pair each fill with future mid for forward-markout PnL.

    Returns (X, y, feature_names) where each row is one fill.
    """
    feat_names = ["obi", "vpin", "garch_sigma", "position", "olv_active"]

    fills_by_product = defaultdict(list)         # p -> list of (ts, fill_dict)
    mids_by_product = defaultdict(list)          # p -> sorted [(ts, mid)]
    mids_seen_ts = defaultdict(set)

    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            kind = rec.get("k")
            ts = rec.get("ts")
            p = rec.get("p")
            if not p or ts is None:
                continue
            if kind == "fill":
                fills_by_product[p].append((int(ts), rec))
            # Capture mids from any record that carries one
            feat_block = rec.get("feat") if kind == "fill" else None
            mid_v = (feat_block or {}).get("mid") if feat_block else None
            if mid_v is not None and ts not in mids_seen_ts[p]:
                mids_by_product[p].append((int(ts), float(mid_v)))
                mids_seen_ts[p].add(ts)

    # Sort mid series; use binary-search-style scan for forward-markout lookup
    for p in mids_by_product:
        mids_by_product[p].sort()

    X, y = [], []
    skipped_no_future = 0
    for p, fills in fills_by_product.items():
        mids = mids_by_product.get(p, [])
        if not mids:
            continue
        cursor = 0
        for fill_ts, rec in sorted(fills, key=lambda r: r[0]):
            target_ts = fill_ts + markout_ticks
            while cursor < len(mids) and mids[cursor][0] < target_ts:
                cursor += 1
            if cursor >= len(mids):
                skipped_no_future += 1
                continue
            future_mid = mids[cursor][1]
            qty = int(rec.get("qty", 0))
            px = float(rec.get("px", 0.0))
            if qty == 0:
                continue
            # Sign: buy (qty>0) profits when future_mid > px; sell (qty<0) when future_mid < px
            pnl = (future_mid - px) * qty
            f = rec.get("feat") or {}
            obi = f.get("obi") or 0.0
            vpin = f.get("vpin") or 0.0
            garch = f.get("garch_sigma") or 0.0
            pos = f.get("pos") or 0
            olv = f.get("olv")
            olv_active = 1.0 if olv else 0.0
            X.append([1.0, float(obi), float(vpin), float(garch), float(pos), olv_active])
            y.append(float(pnl))

    if skipped_no_future:
        print(f"  (skipped {skipped_no_future} fills lacking +{markout_ticks}-tick future mid)")
    return X, y, feat_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="CSV log path (legacy mode)")
    ap.add_argument("--jsonl", help="K4 fill JSONL path (Gap 5 mode)")
    ap.add_argument("--markout", type=int, default=30,
                    help="Forward-markout horizon in ticks for JSONL mode (default 30)")
    ap.add_argument("--features", nargs="+",
                    default=["spread", "obi", "vpin", "position", "markout10",
                             "markout30", "time_since_fill", "olivia_active"])
    ap.add_argument("--target", default="pnl")
    args = ap.parse_args()

    if args.jsonl:
        if not os.path.exists(args.jsonl):
            print(f"ERROR: {args.jsonl} not found")
            sys.exit(1)
        X, y, feat_names = load_jsonl_fills(args.jsonl, args.markout)
        args.features = feat_names
        print(f"Loaded {len(X)} fills from JSONL with {args.markout}-tick markout")
        if len(X) < 30:
            print("Not enough data.")
            sys.exit(1)
    else:
        if not args.log or not os.path.exists(args.log):
            print(f"ERROR: provide --jsonl or --log <existing path>")
            sys.exit(1)

        X, y = [], []
        with open(args.log, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    feats = [float(row[f]) for f in args.features]
                    tgt = float(row[args.target])
                except (KeyError, ValueError):
                    continue
                X.append([1.0] + feats)
                y.append(tgt)

        print(f"Loaded {len(X)} observations, {len(args.features)} features")
        if len(X) < 30:
            print("Not enough data.")
            sys.exit(1)

    coefs, resid = ols_multi(X, y)
    if coefs is None:
        print("ERROR: regression failed (singular matrix)")
        sys.exit(1)

    n = len(X)
    p = len(coefs)
    ss_res = sum(r * r for r in resid)
    sigma2 = ss_res / max(1, n - p)
    my = sum(y) / n
    ss_tot = sum((v - my) ** 2 for v in y)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\nR^2: {r2:.3f}   sigma^2: {sigma2:.3f}   n: {n}")
    print(f"\n{'feature':<20} {'coef':>10}")
    print(f"{'intercept':<20} {coefs[0]:>10.4f}")
    for i, f in enumerate(args.features):
        print(f"{f:<20} {coefs[i+1]:>10.4f}")


if __name__ == "__main__":
    main()
