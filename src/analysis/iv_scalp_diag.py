"""Diagnostic: compute theo_diff distribution for IV scalping on each strike.

Tells us:
  - typical |theo_diff| range (sets iv_scalp_thr)
  - avg_dev EMA stable level
  - signal frequency above various thresholds

Usage:
  py -3.12 analysis/iv_scalp_diag.py
"""
import math
import os
import statistics

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "r3", "prices")

SMILE_COEFFS = [0.02794260, 0.00040248, 0.25853717]
TOTAL_DAYS = 7


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x):
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def bs_call(S, K, TTE, sigma):
    if TTE <= 0 or sigma <= 0:
        return max(S - K, 0.0), 1.0 if S > K else 0.0, 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * TTE) / (sigma * math.sqrt(TTE))
    d2 = d1 - sigma * math.sqrt(TTE)
    call = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    vega = S * _norm_pdf(d1) * math.sqrt(TTE)
    return call, delta, vega


def smile_iv(S, K, TTE, coefs):
    m_t_k = math.log(K / S) / math.sqrt(TTE)
    return coefs[0] * m_t_k * m_t_k + coefs[1] * m_t_k + coefs[2]


def smile_theo(S, K, TTE, coefs):
    iv = smile_iv(S, K, TTE, coefs)
    return bs_call(S, K, TTE, iv)


def wall_mid(row):
    """Largest-volume bid/ask midpoint."""
    bids, asks = [], []
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
        return None, None, None
    bmax = max(bids, key=lambda x: x[1])[0]
    amax = max(asks, key=lambda x: x[1])[0]
    bbo_bid = max(b[0] for b in bids)
    bbo_ask = min(a[0] for a in asks)
    return 0.5 * (bmax + amax), bbo_bid, bbo_ask


def analyse(K, full, ul_full):
    sub = full[full["product"] == f"VEV_{K}"].sort_values(["day", "timestamp"]).reset_index(drop=True)
    if len(sub) == 0:
        return
    print(f"\n=== VEV_{K} ===  n={len(sub)}")

    theo_diffs = []
    sell_signals = []
    buy_signals = []
    deltas = []
    vegas = []
    UL_idx = ul_full.set_index(["day", "timestamp"])

    for _, row in sub.iterrows():
        day = row["day"]
        ts = row["timestamp"]
        if (day, ts) not in UL_idx.index:
            continue
        ulrow = UL_idx.loc[(day, ts)]
        if isinstance(ulrow, pd.DataFrame):
            ulrow = ulrow.iloc[0]
        if pd.isna(ulrow["bid_price_1"]) or pd.isna(ulrow["ask_price_1"]):
            continue
        S = 0.5 * (float(ulrow["bid_price_1"]) + float(ulrow["ask_price_1"]))
        wm, bbo_bid, bbo_ask = wall_mid(row)
        if wm is None:
            continue
        TTE = max(1.0 - (365.0 - TOTAL_DAYS + day + ts / 100.0 / 10000.0) / 365.0, 1e-6)
        theo, delta, vega = smile_theo(S, K, TTE, SMILE_COEFFS)
        td = wm - theo
        theo_diffs.append(td)
        # sell at bbo_bid: signal = (bbo_bid - theo) - 0 (no mean correction yet)
        sell_signals.append(bbo_bid - theo)
        buy_signals.append(bbo_ask - theo)
        deltas.append(delta)
        vegas.append(vega)

    if not theo_diffs:
        print("  NO DATA")
        return

    abs_td = [abs(t) for t in theo_diffs]
    print(f"  theo_diff: mean={statistics.mean(theo_diffs):+.3f}  "
          f"stdev={statistics.stdev(theo_diffs):.3f}  "
          f"|max|={max(abs_td):.3f}")
    print(f"  |theo_diff| quantiles: "
          f"p50={statistics.median(abs_td):.3f}  "
          f"p90={statistics.quantiles(abs_td, n=10)[8]:.3f}  "
          f"p99={statistics.quantiles(abs_td, n=100)[98]:.3f}")
    print(f"  delta avg={statistics.mean(deltas):.3f}  vega avg={statistics.mean(vegas):.2f}")

    abs_sell = [abs(s) for s in sell_signals]
    abs_buy = [abs(b) for b in buy_signals]
    print(f"  |sell_signal| (bbo_bid - theo) p50={statistics.median(abs_sell):.3f}  "
          f"p90={statistics.quantiles(abs_sell, n=10)[8]:.3f}")
    print(f"  |buy_signal| (bbo_ask - theo) p50={statistics.median(abs_buy):.3f}  "
          f"p90={statistics.quantiles(abs_buy, n=10)[8]:.3f}")

    # signal frequency above thresholds
    for thr in [0.05, 0.1, 0.3, 0.5, 1.0]:
        n_sell = sum(1 for s in sell_signals if s >= thr)
        n_buy = sum(1 for s in buy_signals if s <= -thr)
        print(f"  thr={thr:>4}: sell_fires={n_sell}/{len(sell_signals)} "
              f"({100*n_sell/len(sell_signals):.1f}%)  "
              f"buy_fires={n_buy} ({100*n_buy/len(sell_signals):.1f}%)")


def main():
    days = []
    for d in [0, 1, 2]:
        df = pd.read_csv(os.path.join(DATA, f"prices_round_3_day_{d}.csv"), sep=";")
        df.columns = [c.strip().lower() for c in df.columns]
        df["day"] = d
        days.append(df)
    full = pd.concat(days, ignore_index=True)
    ul_full = full[full["product"] == "VELVETFRUIT_EXTRACT"].drop_duplicates(["day", "timestamp"])
    print(f"UL rows: {len(ul_full)}")

    for K in [4500, 5000, 5100, 5200, 5300, 5400, 5500]:
        analyse(K, full, ul_full)


if __name__ == "__main__":
    main()
