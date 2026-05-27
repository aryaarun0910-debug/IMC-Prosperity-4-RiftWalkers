"""R3 initial recon (2026-04-24).

Per-product stats across R3 historical days 0/1/2 (= tutorial/R1/R2 TTE 8/7/6):
  mid range, mean, std(returns), spread avg, day-to-day drift
For each VEV strike: moneyness vs VELVETFRUIT_EXTRACT mid, mean mid, intrinsic.

Writes JSON to data/r3/intel/r3_recon.json.
"""

import csv
import json
import math
import os
import statistics

PIPELINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_DIR = os.path.join(PIPELINE, "data", "r3", "prices")
INTEL_OUT = os.path.join(PIPELINE, "data", "r3", "intel", "r3_recon.json")

STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}


def load_day(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter=";")
        for r in rd:
            try:
                rows.append({
                    "ts": int(r["timestamp"]),
                    "product": r["product"],
                    "bid1": float(r["bid_price_1"]) if r["bid_price_1"] else None,
                    "ask1": float(r["ask_price_1"]) if r["ask_price_1"] else None,
                    "mid": float(r["mid_price"]) if r["mid_price"] else None,
                })
            except (ValueError, KeyError):
                continue
    return rows


def stats_for_product(rows):
    mids = [r["mid"] for r in rows if r["mid"] is not None]
    spreads = [r["ask1"] - r["bid1"] for r in rows
               if r["bid1"] is not None and r["ask1"] is not None]
    if not mids:
        return None
    rets = [math.log(mids[i] / mids[i-1]) for i in range(1, len(mids))
            if mids[i] > 0 and mids[i-1] > 0]
    return {
        "n_ticks": len(mids),
        "mid_mean": round(statistics.mean(mids), 2),
        "mid_std": round(statistics.stdev(mids) if len(mids) > 1 else 0, 2),
        "mid_min": round(min(mids), 2),
        "mid_max": round(max(mids), 2),
        "mid_first": round(mids[0], 2),
        "mid_last": round(mids[-1], 2),
        "ret_std": round(statistics.stdev(rets) if len(rets) > 1 else 0, 6),
        "spread_avg": round(statistics.mean(spreads), 3) if spreads else None,
    }


def main():
    days = sorted(os.listdir(PRICES_DIR))
    all_data = {}  # day -> product -> rows
    for dfn in days:
        path = os.path.join(PRICES_DIR, dfn)
        rows = load_day(path)
        by_prod = {}
        for r in rows:
            by_prod.setdefault(r["product"], []).append(r)
        all_data[dfn] = by_prod
        print(f"[{dfn}] loaded {sum(len(v) for v in by_prod.values())} rows, "
              f"{len(by_prod)} products")

    result = {"per_day": {}, "cross_day": {}}

    for dfn, by_prod in all_data.items():
        result["per_day"][dfn] = {}
        for p, rows in sorted(by_prod.items()):
            s = stats_for_product(rows)
            result["per_day"][dfn][p] = s

    # Cross-day for underlying (VELVETFRUIT_EXTRACT) + HYDROGEL_PACK
    for p in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
        mids = []
        for dfn in sorted(all_data.keys()):
            rows = all_data[dfn].get(p, [])
            mids += [r["mid"] for r in rows if r["mid"] is not None]
        if mids:
            rets = [math.log(mids[i] / mids[i-1]) for i in range(1, len(mids))
                    if mids[i] > 0 and mids[i-1] > 0]
            result["cross_day"][p] = {
                "total_ticks": len(mids),
                "mid_overall_mean": round(statistics.mean(mids), 2),
                "mid_overall_std": round(statistics.stdev(mids), 2),
                "ret_std_overall": round(statistics.stdev(rets), 6),
                "per_day_drift": [
                    (dfn, round(statistics.mean([r["mid"] for r in all_data[dfn].get(p, []) if r["mid"] is not None]), 2))
                    for dfn in sorted(all_data.keys())
                ],
            }

    # VEV moneyness snapshot (use VE mean as S_t)
    ve_mean = result["cross_day"]["VELVETFRUIT_EXTRACT"]["mid_overall_mean"]
    voucher_snap = {}
    for vev, strike in STRIKES.items():
        all_mids = []
        for dfn in sorted(all_data.keys()):
            rows = all_data[dfn].get(vev, [])
            all_mids += [r["mid"] for r in rows if r["mid"] is not None]
        if not all_mids:
            voucher_snap[vev] = {"strike": strike, "status": "NO_DATA"}
            continue
        avg_mid = statistics.mean(all_mids)
        intrinsic = max(ve_mean - strike, 0)
        voucher_snap[vev] = {
            "strike": strike,
            "ticks": len(all_mids),
            "voucher_mean": round(avg_mid, 2),
            "voucher_min": round(min(all_mids), 2),
            "voucher_max": round(max(all_mids), 2),
            "underlying_mean": ve_mean,
            "intrinsic_at_mean": round(intrinsic, 2),
            "time_value_at_mean": round(avg_mid - intrinsic, 2),
            "moneyness_ratio": round(ve_mean / strike, 4),
        }
    result["voucher_snapshot"] = voucher_snap

    os.makedirs(os.path.dirname(INTEL_OUT), exist_ok=True)
    with open(INTEL_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== R3 Recon Summary ===")
    print(f"VE mean: {ve_mean} | HYDROGEL mean: {result['cross_day']['HYDROGEL_PACK']['mid_overall_mean']}")
    print(f"VE ret_std: {result['cross_day']['VELVETFRUIT_EXTRACT']['ret_std_overall']}")
    print(f"HYDROGEL ret_std: {result['cross_day']['HYDROGEL_PACK']['ret_std_overall']}")
    print(f"\nVE day drift: {result['cross_day']['VELVETFRUIT_EXTRACT']['per_day_drift']}")
    print(f"HYDROGEL day drift: {result['cross_day']['HYDROGEL_PACK']['per_day_drift']}")
    print(f"\nVoucher snapshot (underlying mean ~{ve_mean}):")
    for vev, snap in voucher_snap.items():
        if snap.get("status") == "NO_DATA":
            print(f"  {vev:10s} strike {snap['strike']} : NO DATA")
        else:
            print(f"  {vev:10s} K={snap['strike']:4d} mid={snap['voucher_mean']:8.2f} "
                  f"intrinsic={snap['intrinsic_at_mean']:7.2f} TV={snap['time_value_at_mean']:7.2f} "
                  f"S/K={snap['moneyness_ratio']:.4f} ticks={snap['ticks']}")
    print(f"\nWrote {INTEL_OUT}")


if __name__ == "__main__":
    main()
