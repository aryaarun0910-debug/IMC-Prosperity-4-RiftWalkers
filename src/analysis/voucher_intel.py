"""Voucher liquidity + intrinsic-deviation analysis.

For each VEV strike, computes:
  - Average bid-ask spread
  - Trade frequency (fills per 1K ticks)
  - Avg deviation from intrinsic (max(UL-K, 0))
  - Implied vol if any time premium remains
  - Per-tick |return| (volatility of voucher itself)

Goal: identify which strikes have enough liquidity + edge to MM profitably.

Usage:
  py -3.12 analysis/voucher_intel.py
"""
import math
import os
import statistics
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "r3", "prices")
TRADES_DIR = os.path.join(ROOT, "data", "r3", "trades")

VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
    "VEV_6500": 6500,
}


def best_bid_ask(row):
    bp = row.get("bid_price_1")
    ap = row.get("ask_price_1")
    if pd.notna(bp) and pd.notna(ap):
        return float(bp), float(ap)
    return None, None


def main():
    days = []
    for d in [0, 1, 2]:
        path = os.path.join(DATA, f"prices_round_3_day_{d}.csv")
        df = pd.read_csv(path, sep=";")
        df.columns = [c.strip().lower() for c in df.columns]
        days.append(df)
    full = pd.concat(days, ignore_index=True)

    ul_rows = full[full["product"] == "VELVETFRUIT_EXTRACT"].copy()
    ul_rows = ul_rows.drop_duplicates(subset=["timestamp", "day"]).set_index(["day", "timestamp"])[["bid_price_1", "ask_price_1"]]
    print(f"UL rows: {len(ul_rows)}")

    print(f"\n{'Strike':>10} {'mid':>8} {'spread':>8} {'spr_p90':>8} {'tk/1Kt':>7}"
          f" {'|ret|':>7} {'intrinsic':>10} {'TV':>7} {'TV%':>6}")
    print("-" * 90)

    for prod, K in VEV_STRIKES.items():
        sub = full[full["product"] == prod].sort_values("timestamp").reset_index(drop=True)
        if len(sub) == 0:
            continue
        spreads = []
        mids = []
        TVs = []
        rets = []
        for _, row in sub.iterrows():
            b, a = best_bid_ask(row)
            if b is None or a is None or a <= b:
                continue
            spr = a - b
            mid = (a + b) / 2.0
            spreads.append(spr)
            mids.append(mid)
            ts = row["timestamp"]
            day = row.get("day", 0)
            try:
                ulrow = ul_rows.loc[(day, ts)]
                if pd.notna(ulrow["bid_price_1"]) and pd.notna(ulrow["ask_price_1"]):
                    ul_mid = 0.5 * (float(ulrow["bid_price_1"]) + float(ulrow["ask_price_1"]))
                    intrinsic = max(ul_mid - K, 0.0)
                    TVs.append(mid - intrinsic)
            except KeyError:
                pass

        if len(mids) < 2:
            continue
        for i in range(1, len(mids)):
            rets.append(abs(mids[i] - mids[i - 1]))

        spr_avg = statistics.mean(spreads)
        spr_p90 = statistics.quantiles(spreads, n=10)[8] if len(spreads) >= 10 else max(spreads)
        mid_avg = statistics.mean(mids)
        ret_avg = statistics.mean(rets) if rets else 0
        tv_avg = statistics.mean(TVs) if TVs else 0
        tv_pct = (tv_avg / mid_avg * 100) if mid_avg else 0
        intrinsic_avg = mid_avg - tv_avg

        # Volume / fill estimate from full data trades
        # Sample: count rows where mid changed = proxy for activity
        active_ratio = sum(1 for r in rets if r > 0) / len(rets) if rets else 0
        fills_per_1k = active_ratio * 1000 / 30  # rough: 30K ticks span

        print(f"{prod:>10} {mid_avg:>8.1f} {spr_avg:>8.2f} {spr_p90:>8.1f}"
              f" {fills_per_1k:>7.1f} {ret_avg:>7.2f} {intrinsic_avg:>10.1f}"
              f" {tv_avg:>7.2f} {tv_pct:>5.1f}%")


if __name__ == "__main__":
    main()
