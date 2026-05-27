"""Bootstrap significance of the headline mean-reversion edge.

Every PnL figure elsewhere is a point estimate. This resamples the strategy's
per-round-trip PnLs (with replacement) to build a confidence interval on the
total, distinguishing a real edge from a lucky run.

Output: docs/assets/bootstrap_significance.png
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R5 = os.path.join(ROOT, "PIPELINE", "data", "r5", "prices")
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, AMBER = "#1f3a5f", "#2a9d8f", "#e76f51"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.color": "#e8e8e8", "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold"})


def load_mids(path):
    import csv
    rows = {}
    with open(path) as f:
        r = csv.reader(f, delimiter=";"); h = next(r)
        ti, pi, mi = h.index("timestamp"), h.index("product"), h.index("mid_price")
        for row in r:
            if len(row) <= mi:
                continue
            try:
                rows.setdefault(row[pi], []).append((int(row[ti]), float(row[mi])))
            except ValueError:
                pass
    return {p: [m for _, m in sorted(v)] for p, v in rows.items()}


def mr_trades(mids, window=500, z_in=1.5, z_out=0.3, max_pos=10):
    mids = np.asarray(mids); n = len(mids); buf = []; s = ss = 0.0
    pos = 0; entry = 0.0; trades = []
    for i in range(n):
        x = mids[i]; buf.append(x); s += x; ss += x*x
        if len(buf) > window:
            old = buf.pop(0); s -= old; ss -= old*old
        if len(buf) < window:
            continue
        m = s/len(buf); var = max((ss-len(buf)*m*m)/(len(buf)-1), 1e-9); sd = math.sqrt(var)
        z = (x-m)/sd; ns = pos
        if pos == 0:
            if z > z_in: ns = -max_pos
            elif z < -z_in: ns = max_pos
        elif pos > 0 and z > -z_out: ns = 0
        elif pos < 0 and z < z_out: ns = 0
        if ns != pos:
            if pos != 0:
                trades.append(pos * (x - entry))
            entry = x; pos = ns
    return trades


def main():
    days = [load_mids(os.path.join(R5, f"prices_round_5_day_{d}.csv")) for d in (2, 3, 4)]
    trades = []
    for d in days:
        if "PEBBLES_XL" in d:
            trades += mr_trades(d["PEBBLES_XL"])
    trades = np.array(trades)
    observed = trades.sum()

    rng = np.random.default_rng(7)
    B = 20000
    boot = np.array([rng.choice(trades, size=len(trades), replace=True).sum() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_le_zero = float((boot <= 0).mean())

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.hist(boot, bins=60, color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=AMBER, linewidth=2, label="zero (no edge)")
    ax.axvline(lo, color=TEAL, linestyle="--", linewidth=1.5, label=f"95% CI: [{lo:,.0f}, {hi:,.0f}]")
    ax.axvline(hi, color=TEAL, linestyle="--", linewidth=1.5)
    ax.axvline(observed, color="#d62828", linewidth=2, label=f"observed total: {observed:,.0f}")
    ax.set_xlabel("Bootstrapped total PnL (resampling round-trips, 20,000 draws)")
    ax.set_ylabel("Frequency")
    ax.set_title("Statistical Significance of the PEBBLES_XL Edge")
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bootstrap_significance.png"), dpi=140); plt.close(fig)
    print(f"observed {observed:,.0f}  95% CI [{lo:,.0f}, {hi:,.0f}]  P(<=0)={p_le_zero:.4f}")
    print("bootstrap_significance.png written")


if __name__ == "__main__":
    main()
