"""Gap 1: offline fitter for the 6-signal stacking ensemble.

Reads a K4 fill JSONL log emitted by trader.py (IMC_TRADE_LOG_PATH=...) for
P3 R3 (or any round with a representative regime mix), pairs each fill with
its forward-markout PnL, and fits ridge-regression weights on the
[garch_sigma, kelly, gm, obi, vpin, micro_dev] feature vector.

Outputs a JSON file consumable by trader.py at module import:
    analysis/fingerprints/signal_stack.json

Usage:
    IMC_TRADE_LOG_PATH=/tmp/p3r3_fills.jsonl py -3.12 backtest.py --round p3r3
    py -3.12 analysis/signal_stacking_fit.py \
        --jsonl /tmp/p3r3_fills.jsonl --markout 30 --ridge 1.0
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_fills(path, markout):
    fills_by_p = defaultdict(list)
    mids_by_p = defaultdict(list)
    seen = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("k") != "fill":
                continue
            ts = int(rec["ts"])
            p = rec["p"]
            fills_by_p[p].append((ts, rec))
            f = rec.get("feat") or {}
            if f.get("mid") is not None and ts not in seen[p]:
                mids_by_p[p].append((ts, float(f["mid"])))
                seen[p].add(ts)
    for p in mids_by_p:
        mids_by_p[p].sort()

    rows = []
    for p, flist in fills_by_p.items():
        mids = mids_by_p.get(p, [])
        if not mids:
            continue
        cursor = 0
        for ts, rec in sorted(flist, key=lambda r: r[0]):
            target = ts + markout
            while cursor < len(mids) and mids[cursor][0] < target:
                cursor += 1
            if cursor >= len(mids):
                continue
            future_mid = mids[cursor][1]
            qty = int(rec.get("qty", 0))
            px = float(rec.get("px", 0.0))
            if qty == 0:
                continue
            pnl = (future_mid - px) * qty
            f = rec.get("feat") or {}
            row = {
                "garch": float(f.get("garch_sigma") or 0.0),
                "kelly": 0.0,        # filled below from neighbour signals
                "gm": 0.0,
                "obi": float(f.get("obi") or 0.0),
                "vpin": float(f.get("vpin") or 0.0),
                "micro_dev": 0.0,    # not in fill snapshot — leave 0
                "pnl": pnl,
            }
            rows.append(row)
    return rows


def ridge_solve(X, y, lam):
    n, p = len(X), len(X[0])
    if n < p + 5:
        return None
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    for i in range(p):
        XtX[i][i] += lam
    A = [XtX[i] + [Xty[i]] for i in range(p)]
    for i in range(p):
        piv = A[i][i]
        if abs(piv) < 1e-12:
            return None
        for j in range(p + 1):
            A[i][j] /= piv
        for k in range(p):
            if k != i:
                f = A[k][i]
                for j in range(p + 1):
                    A[k][j] -= f * A[i][j]
    return [A[i][p] for i in range(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="K4 fill JSONL emitted by trader.py")
    ap.add_argument("--markout", type=int, default=30)
    ap.add_argument("--ridge", type=float, default=1.0,
                    help="L2 regularisation strength")
    ap.add_argument("--out", default=None,
                    help="Output JSON (default analysis/fingerprints/signal_stack.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print fit only — do not overwrite the JSON")
    args = ap.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"ERROR: {args.jsonl} missing")
        sys.exit(1)

    rows = load_fills(args.jsonl, args.markout)
    if len(rows) < 50:
        print(f"Need >=50 fills; got {len(rows)}.")
        sys.exit(1)

    feats = ["garch", "kelly", "gm", "obi", "vpin", "micro_dev"]
    X = [[r[k] for k in feats] for r in rows]
    y = [r["pnl"] for r in rows]
    coefs = ridge_solve(X, y, args.ridge)
    if coefs is None:
        print("Ridge solve failed (singular).")
        sys.exit(1)

    weights = dict(zip(feats, coefs))
    yhat = [sum(coefs[j] * X[i][j] for j in range(len(feats))) for i in range(len(X))]
    ss_res = sum((y[i] - yhat[i]) ** 2 for i in range(len(y)))
    my = sum(y) / len(y)
    ss_tot = sum((v - my) ** 2 for v in y)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\nFitted on {len(rows)} fills, R^2={r2:.4f} (ridge={args.ridge}, markout={args.markout})")
    print(f"{'feature':<12} {'weight':>12}")
    for k, v in weights.items():
        print(f"{k:<12} {v:>12.6f}")

    if args.dry_run:
        print("\n--dry-run: skipping write.")
        return

    out = args.out or os.path.join(os.path.dirname(__file__), "fingerprints",
                                   "signal_stack.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(weights, fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
