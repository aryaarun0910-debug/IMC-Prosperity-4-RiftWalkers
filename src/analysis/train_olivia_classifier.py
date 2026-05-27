"""Gap 2: train + distill the Olivia decision cascade.

Reads P3 R5 trade CSVs (where the Olivia bot is named in the buyer/seller
columns), labels each market trade as Olivia/non-Olivia, fits a depth-≤4
greedy decision tree on three cheap features, and emits the cascade as a
short list of if/else rules consumable by trader.py at module import.

Output: analysis/fingerprints/olivia_cascade.json
Format: {"rules": [{"feat": "size", "op": ">=", "thr": 8, "vote": "LONG"}, ...]}

Usage:
    py -3.12 analysis/train_olivia_classifier.py \
        --trade-glob "reference/p3_data/Round 5 Data/trades_round_5_day_*.csv"
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict


def load_trades(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for r in reader:
                try:
                    rows.append({
                        "ts": int(r["timestamp"]),
                        "sym": r["symbol"],
                        "px": float(r["price"]),
                        "qty": int(r["quantity"]),
                        "buyer": (r.get("buyer") or "").strip(),
                        "seller": (r.get("seller") or "").strip(),
                    })
                except (KeyError, ValueError):
                    continue
    return rows


def build_dataset(rows, lookahead=10):
    """For each trade, label as Olivia-buy/sell/none and snap features."""
    by_sym = defaultdict(list)
    for r in rows:
        by_sym[r["sym"]].append(r)

    dataset = []
    for sym, trades in by_sym.items():
        trades.sort(key=lambda t: t["ts"])
        for i, t in enumerate(trades):
            buyer = t["buyer"]
            seller = t["seller"]
            if buyer == "Olivia":
                label = "LONG"
            elif seller == "Olivia":
                label = "SHORT"
            else:
                label = "NONE"
            window = [u["px"] for u in trades[max(0, i - 50):i]]
            if window:
                srt = sorted(window)
                med = srt[len(srt) // 2]
            else:
                med = t["px"]
            size = abs(int(t["qty"]))
            side = 1 if t["qty"] > 0 else -1
            px_dev = float(t["px"]) - float(med)
            dataset.append({"size": size, "side": side, "px_dev": px_dev, "label": label})
    return dataset


def gini(labels):
    if not labels:
        return 0.0
    n = len(labels)
    counts = {"LONG": 0, "SHORT": 0, "NONE": 0}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def majority(labels):
    if not labels:
        return "NONE"
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts, key=counts.get)


def best_split(rows, feat):
    vals = sorted({r[feat] for r in rows})
    if len(vals) < 2:
        return None
    base = gini([r["label"] for r in rows])
    n = len(rows)
    best = None
    for v in vals:
        left = [r for r in rows if r[feat] < v]
        right = [r for r in rows if r[feat] >= v]
        if not left or not right:
            continue
        g = (len(left) / n) * gini([r["label"] for r in left]) \
            + (len(right) / n) * gini([r["label"] for r in right])
        gain = base - g
        if best is None or gain > best[0]:
            best = (gain, v, left, right)
    return best


def grow(rows, depth, max_depth, min_leaf):
    if depth >= max_depth or len(rows) < min_leaf:
        return {"leaf": majority([r["label"] for r in rows])}
    best = None
    for feat in ("size", "side", "px_dev"):
        cand = best_split(rows, feat)
        if cand and (best is None or cand[0] > best[0]):
            best = (cand[0], feat, cand[1], cand[2], cand[3])
    if best is None or best[0] <= 0:
        return {"leaf": majority([r["label"] for r in rows])}
    _, feat, thr, left, right = best
    return {
        "feat": feat, "thr": thr,
        "left": grow(left, depth + 1, max_depth, min_leaf),
        "right": grow(right, depth + 1, max_depth, min_leaf),
    }


def flatten_tree(node, prefix=None, rules=None):
    """Convert the tree into a fall-through rule list. First rule that matches wins."""
    if rules is None:
        rules = []
    if "leaf" in node:
        if node["leaf"] in ("LONG", "SHORT") and prefix:
            # AND together the prefix conditions into a single rule by picking the
            # tightest constraint per feature; since cascade harness only checks
            # one rule per match, emit one per leaf path.
            for cond in prefix:
                rules.append({"feat": cond["feat"], "op": cond["op"],
                              "thr": cond["thr"], "vote": node["leaf"]})
            return rules
        return rules
    feat = node["feat"]; thr = node["thr"]
    left_prefix = (prefix or []) + [{"feat": feat, "op": "<", "thr": thr}]
    right_prefix = (prefix or []) + [{"feat": feat, "op": ">=", "thr": thr}]
    flatten_tree(node["left"], left_prefix, rules)
    flatten_tree(node["right"], right_prefix, rules)
    return rules


def evaluate(rows, tree):
    correct = 0
    by_label = defaultdict(int)
    by_pred = defaultdict(int)
    matrix = defaultdict(int)
    for r in rows:
        node = tree
        while "leaf" not in node:
            if r[node["feat"]] < node["thr"]:
                node = node["left"]
            else:
                node = node["right"]
        pred = node["leaf"]
        by_label[r["label"]] += 1
        by_pred[pred] += 1
        matrix[(r["label"], pred)] += 1
        if pred == r["label"]:
            correct += 1
    return correct / max(1, len(rows)), by_label, by_pred, matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trade-glob", required=True)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-leaf", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_trades(args.trade_glob)
    if not rows:
        print(f"No trades matched {args.trade_glob}")
        sys.exit(1)
    ds = build_dataset(rows)
    n = len(ds)
    n_olv = sum(1 for r in ds if r["label"] != "NONE")
    print(f"Loaded {n} trades; {n_olv} labelled Olivia ({100*n_olv/max(1,n):.2f}%).")
    if n_olv < 30:
        print("Not enough Olivia trades to train.")
        sys.exit(1)

    tree = grow(ds, 0, args.max_depth, args.min_leaf)
    acc, by_label, by_pred, matrix = evaluate(ds, tree)
    print(f"\nTraining accuracy: {acc:.4f}")
    print(f"  Labels: {dict(by_label)}")
    print(f"  Preds : {dict(by_pred)}")
    olv_rows = [r for r in ds if r["label"] != "NONE"]
    if olv_rows:
        olv_acc, _, _, _ = evaluate(olv_rows, tree)
        print(f"  Olivia-only recall: {olv_acc:.4f}")

    rules = flatten_tree(tree)
    cascade = {"rules": rules, "trained_on": args.trade_glob,
               "max_depth": args.max_depth, "n_train": n, "accuracy": acc}
    print(f"\nCascade rules ({len(rules)}):")
    for r in rules:
        print(f"  if {r['feat']} {r['op']} {r['thr']:.4f}  -> {r['vote']}")

    if args.dry_run:
        print("\n--dry-run: skipping write.")
        return

    out = args.out or os.path.join(os.path.dirname(__file__), "fingerprints",
                                   "olivia_cascade.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(cascade, fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
