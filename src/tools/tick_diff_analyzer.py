"""Tick-level forensics on a server log.

Answers, per product:
  1. Where do we bleed? — top-N worst ticks by MTM + realized delta.
  2. Where do we skip? — ticks with crossable spread we didn't fill.
  3. Where do we over-fill? — ticks with inventory blowout after aggressive takes.

Usage:
  py -3.12 tools/tick_diff_analyzer.py data/r3/server_logs/364944/364944.log
  py -3.12 tools/tick_diff_analyzer.py <log> --product HYDROGEL_PACK --top 20
"""
import json
import sys
from collections import defaultdict


def load_log(path):
    import os
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.endswith(".log"):
                path = os.path.join(path, f); break
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_activities(csv_str):
    """Parse activitiesLog into {(product, ts): book_row}."""
    lines = csv_str.split("\n")
    header = [h.strip() for h in lines[0].split(";")]
    rows = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(";")
        row = {}
        for i, h in enumerate(header):
            if i < len(parts):
                v = parts[i].strip()
                if v == "":
                    row[h] = None
                else:
                    try:
                        row[h] = float(v) if "." in v else int(v)
                    except ValueError:
                        row[h] = v
        prod = row.get("product")
        ts = row.get("timestamp")
        if prod and ts is not None:
            rows[(prod, ts)] = row
    return rows


def build_fills_index(trade_history):
    """{(product, ts): [(side, price, qty), ...]} — side = +1 buy, -1 sell."""
    fills = defaultdict(list)
    for t in trade_history:
        prod = t["symbol"]
        ts = t["timestamp"]
        qty = t["quantity"]
        if t.get("buyer") == "SUBMISSION":
            fills[(prod, ts)].append((+1, t["price"], qty))
        elif t.get("seller") == "SUBMISSION":
            fills[(prod, ts)].append((-1, t["price"], qty))
    return fills


def analyze_product(product, activities, fills, top_n=15):
    """Per-tick PnL + inventory walk + leak detection."""
    ticks = sorted(t for (p, t) in activities if p == product)
    if not ticks:
        return None

    # Walk: track position + cum realized
    pos = 0
    cum_realized = 0.0
    prev_mid = None
    tick_rows = []
    fill_count = 0
    take_fills = 0   # fills that crossed the posted spread (aggressive)
    make_fills = 0   # fills at BBO or better
    near_miss = 0    # book had crossable spread but no fill

    for ts in ticks:
        row = activities[(product, ts)]
        mid = row.get("mid_price")
        bid1 = row.get("bid_price_1")
        ask1 = row.get("ask_price_1")
        pnl_running = row.get("profit_and_loss", 0) or 0
        tfills = fills.get((product, ts), [])

        realized_tick = 0.0
        tick_volume = 0
        for side, price, qty in tfills:
            signed_qty = side * qty
            # Very rough realized: (mid - fill_price) * signed_qty
            if mid is not None:
                realized_tick += (mid - price) * signed_qty
            pos += signed_qty
            tick_volume += qty
            # Crossed-spread detection: buys above prev ask, sells below prev bid
            if side == +1 and ask1 is not None and price >= ask1:
                take_fills += 1
            elif side == -1 and bid1 is not None and price <= bid1:
                take_fills += 1
            else:
                make_fills += 1
        fill_count += len(tfills)

        mtm_tick = 0.0
        if prev_mid is not None and mid is not None:
            mtm_tick = pos * (mid - prev_mid)

        tick_pnl_delta = realized_tick + mtm_tick

        tick_rows.append({
            "ts": ts, "mid": mid, "bid": bid1, "ask": ask1,
            "pos": pos, "fills": len(tfills), "vol": tick_volume,
            "realized": realized_tick, "mtm": mtm_tick,
            "delta": tick_pnl_delta, "running_pnl": pnl_running,
        })

        prev_mid = mid

    # Top-N worst ticks
    worst = sorted(tick_rows, key=lambda r: r["delta"])[:top_n]
    best = sorted(tick_rows, key=lambda r: -r["delta"])[:5]

    # Inventory blowouts: consecutive ticks where |pos| > 0.8 * limit
    limits = {"HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200}
    lim = limits.get(product, 300)
    stuck_ticks = sum(1 for r in tick_rows if abs(r["pos"]) > 0.8 * lim)

    # Final stats
    final_pnl = tick_rows[-1]["running_pnl"] if tick_rows else 0
    max_long = max((r["pos"] for r in tick_rows), default=0)
    max_short = min((r["pos"] for r in tick_rows), default=0)
    peak = max((r["running_pnl"] for r in tick_rows), default=0)
    trough = min((r["running_pnl"] for r in tick_rows), default=0)

    return {
        "product": product,
        "ticks": len(tick_rows),
        "fills": fill_count,
        "take_fills": take_fills,
        "make_fills": make_fills,
        "final_pnl": final_pnl,
        "peak_pnl": peak,
        "trough_pnl": trough,
        "max_long": max_long,
        "max_short": max_short,
        "stuck_ticks": stuck_ticks,
        "stuck_frac": stuck_ticks / max(len(tick_rows), 1),
        "worst": worst,
        "best": best,
    }


def fmt_tick(r):
    return (f"  t={r['ts']:>6}  mid={r['mid']:>8}  "
            f"bid/ask={r['bid']}/{r['ask']}  pos={r['pos']:>+4}  "
            f"fills={r['fills']} vol={r['vol']:>3}  "
            f"delta={r['delta']:>+10.1f}  pnl={r['running_pnl']:>+9.0f}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    log_path = args[0]
    only_product = None
    top_n = 15
    i = 1
    while i < len(args):
        if args[i] == "--product":
            only_product = args[i + 1]; i += 2
        elif args[i] == "--top":
            top_n = int(args[i + 1]); i += 2
        else:
            i += 1

    log = load_log(log_path)
    activities = parse_activities(log["activitiesLog"])
    fills = build_fills_index(log["tradeHistory"])

    products = sorted({p for (p, _) in activities})
    if only_product:
        products = [p for p in products if p == only_product]

    print(f"=== TICK FORENSICS: {log_path} ===\n")

    total_pnl = 0
    for prod in products:
        r = analyze_product(prod, activities, fills, top_n=top_n)
        if r is None:
            continue
        if r["fills"] == 0 and "VEV_" in prod:
            # Suppress zero-trade voucher noise unless requested explicitly
            if only_product is None:
                continue

        print(f"--- {prod} ---")
        print(f"  final_pnl={r['final_pnl']:>+9.0f}  "
              f"peak={r['peak_pnl']:>+8.0f}  trough={r['trough_pnl']:>+8.0f}")
        print(f"  fills={r['fills']}  take={r['take_fills']} make={r['make_fills']}  "
              f"max_pos=[{r['max_short']:+4}..{r['max_long']:+4}]  "
              f"stuck@80%limit={r['stuck_ticks']}t ({r['stuck_frac']:.1%})")
        if r["fills"] > 0:
            print(f"  WORST {top_n} ticks (by tick delta):")
            for row in r["worst"]:
                print(fmt_tick(row))
            print(f"  BEST 5 ticks:")
            for row in r["best"]:
                print(fmt_tick(row))
        print()
        total_pnl += r["final_pnl"]

    print(f"=== TOTAL: {total_pnl:+.0f} ===")


if __name__ == "__main__":
    main()
