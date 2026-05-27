"""K4 aggregate — summarize the K4 trade emitter log.

Reads the JSONL produced by trader.py when IMC_TRADE_LOG_PATH is set.
Reports per-strategy / per-signal order count + qty histogram.

Usage:
    IMC_TRADE_LOG_PATH=/tmp/k4_test.jsonl py -3.12 backtest.py --round p4r1 --seeds 42
    py -3.12 analysis/k4_aggregate.py --log /tmp/k4_test.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    by_strat = Counter()
    by_strat_qty = defaultdict(int)
    by_olivia = Counter()
    by_strat_pos_bucket = defaultdict(lambda: defaultdict(int))
    n = 0

    with open(args.log) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n += 1
            strat = rec.get("strat", "?")
            sig = rec.get("sig", {}) or {}
            ords = rec.get("ords", []) or []
            by_strat[strat] += 1
            qty_total = sum(abs(q) for _, q in ords)
            by_strat_qty[strat] += qty_total
            olv = sig.get("olv")
            by_olivia[(strat, str(olv))] += 1
            pos = sig.get("pos", 0)
            bucket = "neg" if pos < -10 else "neutral" if abs(pos) <= 10 else "pos"
            by_strat_pos_bucket[strat][bucket] += 1

    print(f"K4 trade-log aggregate ({n} order packs)\n")
    print(f"{'strategy':<20} {'packs':>10} {'avg_qty':>10}")
    for s, c in by_strat.most_common():
        avg = by_strat_qty[s] / max(1, c)
        print(f"{s:<20} {c:>10d} {avg:>10.1f}")

    print("\nOlivia signal distribution per strategy:")
    for (s, olv), c in sorted(by_olivia.items()):
        print(f"  {s:<20}  olv={olv:<10}  {c:>10d}")

    print("\nPosition state at quote time per strategy:")
    for s, buckets in sorted(by_strat_pos_bucket.items()):
        print(f"  {s:<20}  neg={buckets['neg']:>6d}  neutral={buckets['neutral']:>6d}  pos={buckets['pos']:>6d}")


if __name__ == "__main__":
    main()
