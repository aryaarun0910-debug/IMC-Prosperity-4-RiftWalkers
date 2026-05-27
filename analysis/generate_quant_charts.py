"""Quantitative analysis figures, generated from the real competition data.

Produces (to docs/assets/):
  correlation_heatmap.png      Within-category return-correlation structure (R5)
  volatility_profile.png       Realized volatility across all 50 products (R5)
  mean_reversion_surface_3d.png  PnL over (window, z-entry) for PEBBLES_XL (R5)
  risk_analysis.png            Equity curve, drawdown, and return distribution (R5)
  vol_smile.png                Implied-volatility smile across option strikes (R4)
  vol_surface_3d.png           Implied-volatility surface, strike x time (R4)

Requires the competition CSVs under PIPELINE/data/ (not committed). Run from repo root:
    python analysis/generate_quant_charts.py
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
R5 = os.path.join(ROOT, "PIPELINE", "data", "r5", "prices")
R4 = os.path.join(ROOT, "PIPELINE", "data", "r4", "prices")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333", "axes.grid": True, "grid.color": "#e8e8e8",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
})
NAVY, TEAL, AMBER = "#1f3a5f", "#2a9d8f", "#e76f51"

CATEGORIES = {
    "GalaxySounds": ["GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS","GALAXY_SOUNDS_SOLAR_FLAMES"],
    "SleepPods": ["SLEEP_POD_SUEDE","SLEEP_POD_LAMB_WOOL","SLEEP_POD_POLYESTER","SLEEP_POD_NYLON","SLEEP_POD_COTTON"],
    "Microchips": ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_SQUARE","MICROCHIP_RECTANGLE","MICROCHIP_TRIANGLE"],
    "Pebbles": ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"],
    "Robots": ["ROBOT_VACUUMING","ROBOT_MOPPING","ROBOT_DISHES","ROBOT_LAUNDRY","ROBOT_IRONING"],
    "UVVisors": ["UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_MAGENTA"],
    "Translators": ["TRANSLATOR_SPACE_GRAY","TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_VOID_BLUE"],
    "Panels": ["PANEL_1X2","PANEL_2X2","PANEL_1X4","PANEL_2X4","PANEL_4X4"],
    "OxygenShakes": ["OXYGEN_SHAKE_MORNING_BREATH","OXYGEN_SHAKE_EVENING_BREATH","OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC"],
    "SnackPacks": ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"],
}


def load_mids(path):
    """product -> list of mids ordered by timestamp."""
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


def rets(s):
    s = np.asarray(s)
    return np.diff(s) / s[:-1]


# ============================================================
# 1. Correlation heatmaps (R5, day 4)
# ============================================================
def correlation_heatmap():
    mids = load_mids(os.path.join(R5, "prices_round_5_day_4.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, cat in zip(axes, ["SnackPacks", "Pebbles"]):
        prods = CATEGORIES[cat]
        R = np.array([rets(mids[p]) for p in prods])
        n = min(len(x) for x in R)
        R = np.array([x[:n] for x in R])
        C = np.corrcoef(R)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        labels = [p.split("_")[-1] for p in prods]
        ax.set_xticks(range(len(prods))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(prods))); ax.set_yticklabels(labels, fontsize=9)
        for i in range(len(prods)):
            for j in range(len(prods)):
                ax.text(j, i, f"{C[i,j]:+.2f}", ha="center", va="center",
                        color="white" if abs(C[i,j]) > 0.5 else "#333", fontsize=8)
        ax.set_title(f"{cat}: tick-return correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Within-Category Return Correlation Structure (R5, Day 4)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "correlation_heatmap.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  correlation_heatmap.png")


# ============================================================
# 2. Volatility profile (R5, all products, 3-day average)
# ============================================================
def volatility_profile():
    days = [load_mids(os.path.join(R5, f"prices_round_5_day_{d}.csv")) for d in (2, 3, 4)]
    cat_of = {p: c for c, ps in CATEGORIES.items() for p in ps}
    palette = {c: plt.cm.tab10(i) for i, c in enumerate(CATEGORIES)}
    vols = {}
    for p in cat_of:
        vv = []
        for d in days:
            if p in d:
                r = rets(d[p])
                vv.append(np.std(r) * math.sqrt(10000))  # annualized to a 10k-tick day
        if vv:
            vols[p] = np.mean(vv)
    order = sorted(vols, key=vols.get, reverse=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = [palette[cat_of[p]] for p in order]
    ax.barh(range(len(order)), [vols[p] for p in order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([p.replace("_", " ").title() for p in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Realized volatility (std of tick returns x sqrt(10,000), 3-day mean)")
    ax.set_title("Volatility Profile Across All 50 Products (R5)")
    handles = [plt.Rectangle((0,0),1,1,color=palette[c]) for c in CATEGORIES]
    ax.legend(handles, list(CATEGORIES.keys()), loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "volatility_profile.png"), dpi=140); plt.close(fig)
    print("  volatility_profile.png")


# ============================================================
# 3. Mean-reversion PnL surface over (window, z_in) - PEBBLES_XL
# ============================================================
def _mr_pnl(mids, window, z_in, z_out=0.3, max_pos=10):
    mids = np.asarray(mids); n = len(mids)
    pnl = pos = 0.0; entry = 0.0
    buf = []
    s = ss = 0.0
    for i in range(n):
        x = mids[i]; buf.append(x); s += x; ss += x*x
        if len(buf) > window:
            old = buf.pop(0); s -= old; ss -= old*old
        if len(buf) < window:
            continue
        m = s/len(buf); var = max((ss-len(buf)*m*m)/(len(buf)-1), 1e-9); sd = math.sqrt(var)
        z = (x-m)/sd
        ns = pos
        if pos == 0:
            if z > z_in: ns = -max_pos
            elif z < -z_in: ns = max_pos
        elif pos > 0 and z > -z_out: ns = 0
        elif pos < 0 and z < z_out: ns = 0
        if ns != pos:
            if pos != 0:
                pnl += pos * (x - entry)
            entry = x; pos = ns
    pnl += pos * (mids[-1] - entry)
    return pnl


def mean_reversion_surface():
    days = [load_mids(os.path.join(R5, f"prices_round_5_day_{d}.csv")) for d in (2, 3, 4)]
    windows = np.arange(100, 1001, 100)
    z_ins = np.arange(1.0, 3.01, 0.25)
    Z = np.zeros((len(z_ins), len(windows)))
    for a, zin in enumerate(z_ins):
        for b, w in enumerate(windows):
            tot = sum(_mr_pnl(d["PEBBLES_XL"], int(w), zin) for d in days if "PEBBLES_XL" in d)
            Z[a, b] = tot
    W, ZI = np.meshgrid(windows, z_ins)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(W, ZI, Z, cmap=cm.viridis, antialiased=True)
    # mark the peak
    ia, ib = np.unravel_index(np.argmax(Z), Z.shape)
    ax.scatter([windows[ib]], [z_ins[ia]], [Z[ia, ib]], color=AMBER, s=80,
               label=f"Peak: window={windows[ib]}, z={z_ins[ia]:.2f}  ->  {Z[ia,ib]:,.0f}")
    ax.set_xlabel("\nRolling window (ticks)")
    ax.set_ylabel("\nEntry z-score")
    ax.set_zlabel("\n3-day PnL")
    ax.set_title("Mean-Reversion Parameter Surface: PEBBLES_XL (R5)", y=0.99)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95))
    ax.view_init(elev=28, azim=-52)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=14, pad=0.1, label="3-day PnL")
    fig.savefig(os.path.join(OUT, "mean_reversion_surface_3d.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  mean_reversion_surface_3d.png")


# ============================================================
# 4. Risk analysis: equity curve, drawdown, return distribution
# ============================================================
def risk_analysis():
    days = [load_mids(os.path.join(R5, f"prices_round_5_day_{d}.csv")) for d in (2, 3, 4)]
    # Build a tick-level equity curve from PEBBLES_XL mean-reversion across the 3 days.
    equity = [0.0]
    window, z_in, z_out, max_pos = 500, 1.5, 0.3, 10
    for d in days:
        mids = np.asarray(d["PEBBLES_XL"]); buf = []; s = ss = 0.0; pos = 0.0; entry = 0.0
        cum = equity[-1]
        for i in range(len(mids)):
            x = mids[i]; buf.append(x); s += x; ss += x*x
            if len(buf) > window:
                old = buf.pop(0); s -= old; ss -= old*old
            mark = cum + (pos * (x - entry) if pos else 0)
            equity.append(mark)
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
                    cum += pos * (x - entry)
                entry = x; pos = ns
        cum += pos * (mids[-1] - entry); pos = 0
        equity[-1] = cum
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [2, 1, 1.3]})
    axes[0].plot(eq, color=NAVY, linewidth=1.4)
    for b in (len(eq)//3, 2*len(eq)//3):
        axes[0].axvline(b, color="#bbb", linestyle="--", linewidth=0.8)
    axes[0].set_title("Equity Curve: PEBBLES_XL Mean-Reversion (R5, 3 days concatenated)")
    axes[0].set_ylabel("Cumulative PnL")
    axes[0].annotate(f"Final: {eq[-1]:,.0f}", (len(eq)-1, eq[-1]), textcoords="offset points",
                     xytext=(-70, 6), fontsize=10, fontweight="bold")

    axes[1].fill_between(range(len(dd)), dd, 0, color=AMBER, alpha=0.6)
    axes[1].set_title(f"Drawdown (max: {dd.min():,.0f})")
    axes[1].set_ylabel("PnL below peak")

    # per-segment PnL increments as a return distribution
    inc = np.diff(eq)
    inc = inc[inc != 0]
    axes[2].hist(inc, bins=60, color=TEAL, alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[2].axvline(0, color="#333", linewidth=1)
    sharpe = inc.mean() / (inc.std() + 1e-9) * math.sqrt(len(inc))
    axes[2].set_title(f"Per-tick PnL change distribution  (annualized Sharpe ~ {sharpe:.1f})")
    axes[2].set_xlabel("PnL change per tick")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "risk_analysis.png"), dpi=140); plt.close(fig)
    print("  risk_analysis.png")


# ============================================================
# 5 & 6. Implied-volatility smile and surface (R4 vouchers)
# ============================================================
def _bs_call(S, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S-K, 0.0)
    d1 = (math.log(S/K) + 0.5*sigma*sigma*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    nd = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
    return S*nd(d1) - K*nd(d2)


def _implied_vol(price, S, K, T):
    intrinsic = max(S-K, 0.0)
    if price <= intrinsic + 1e-6 or T <= 0:
        return None
    lo, hi = 1e-4, 8.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if _bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)


def vol_smile_and_surface():
    strikes = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
    T = 21/252.0  # representative time-to-expiry (3 trading weeks)
    smiles = {}
    for day in (1, 2, 3):
        mids = load_mids(os.path.join(R4, f"prices_round_4_day_{day}.csv"))
        if "VELVETFRUIT_EXTRACT" not in mids:
            continue
        S = np.median(mids["VELVETFRUIT_EXTRACT"])
        pts = []
        for K in strikes:
            sym = f"VEV_{K}"
            if sym not in mids:
                continue
            price = np.median(mids[sym])
            iv = _implied_vol(price, S, K, T)
            if iv and 0.05 < iv < 6.0:
                pts.append((K/S, iv))   # moneyness, iv
        if len(pts) >= 4:
            smiles[day] = (S, pts)

    if smiles:
        # 2D smile
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        colors = [NAVY, TEAL, AMBER]
        for (day, (S, pts)), c in zip(sorted(smiles.items()), colors):
            xs = [m for m, _ in pts]; ys = [v*100 for _, v in pts]
            ax.plot(xs, ys, "o-", color=c, linewidth=2, markersize=7, label=f"Day {day} (spot {S:,.0f})")
        ax.axvline(1.0, color="#bbb", linestyle="--", linewidth=1)
        ax.text(1.005, ax.get_ylim()[1]*0.95, "at-the-money", fontsize=9, color="#888")
        ax.set_xlabel("Moneyness (strike / spot)")
        ax.set_ylabel("Implied volatility (%)")
        ax.set_title("Implied-Volatility Smile, VELVETFRUIT Vouchers (R4)")
        ax.legend(framealpha=0.9)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "vol_smile.png"), dpi=140); plt.close(fig)
        print("  vol_smile.png")

        # 3D surface across day (time proxy) x moneyness
        common_m = sorted(set(round(m, 3) for _, (S, pts) in smiles.items() for m, _ in pts))
        days_sorted = sorted(smiles)
        if len(days_sorted) >= 2 and len(common_m) >= 4:
            grid = np.full((len(days_sorted), len(common_m)), np.nan)
            for di, day in enumerate(days_sorted):
                S, pts = smiles[day]
                d = {round(m, 3): v for m, v in pts}
                for mi, m in enumerate(common_m):
                    # nearest available moneyness
                    near = min(d, key=lambda k: abs(k-m))
                    if abs(near-m) < 0.06:
                        grid[di, mi] = d[near]*100
            Mn, Dy = np.meshgrid(common_m, days_sorted)
            mask = ~np.isnan(grid)
            if mask.sum() >= 6:
                fig = plt.figure(figsize=(10, 7.5))
                ax = fig.add_subplot(111, projection="3d")
                G = np.where(np.isnan(grid), np.nanmean(grid), grid)
                surf = ax.plot_surface(Mn, Dy, G, cmap=cm.plasma, antialiased=True, alpha=0.9)
                ax.set_xlabel("\nMoneyness (K/S)")
                ax.set_ylabel("\nTrading day")
                ax.set_zlabel("\nImplied vol (%)")
                ax.set_yticks(days_sorted)
                ax.set_title("Implied-Volatility Surface, VELVETFRUIT Vouchers (R4)", y=0.99)
                ax.view_init(elev=26, azim=-60)
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=14, pad=0.1, label="IV (%)")
                fig.savefig(os.path.join(OUT, "vol_surface_3d.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
                print("  vol_surface_3d.png")
    else:
        print("  (vol smile skipped — IV inversion produced no clean points)")


# ============================================================
# 7. Walk-forward validation: train (day 2+3) vs held-out test (day 4)
# ============================================================
def walk_forward_validation():
    days = {d: load_mids(os.path.join(R5, f"prices_round_5_day_{d}.csv")) for d in (2, 3, 4)}
    products = [p for ps in CATEGORIES.values() for p in ps]
    grid = [(w, z) for w in (200, 500, 1000) for z in (1.5, 2.0, 2.5)]

    pts = []  # (train_pnl, test_pnl, passed)
    for p in products:
        if not all(p in days[d] for d in (2, 3, 4)):
            continue
        # pick the config with the best train (day2+day3) PnL, requiring both train days > 0
        best = None
        for w, z in grid:
            d2 = _mr_pnl(days[2][p], w, z)
            d3 = _mr_pnl(days[3][p], w, z)
            if d2 > 0 and d3 > 0 and (best is None or d2 + d3 > best[0]):
                best = (d2 + d3, w, z)
        if best is None:
            continue
        train = best[0]
        test = _mr_pnl(days[4][p], best[1], best[2])
        pts.append((train, test, test > 0))

    if not pts:
        print("  (walk-forward produced no points)")
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    surv = [(a, b) for a, b, ok in pts if ok]
    fail = [(a, b) for a, b, ok in pts if not ok]
    if surv:
        ax.scatter([a for a, _ in surv], [b for _, b in surv], s=60, color=TEAL,
                   edgecolor="white", linewidth=0.5, label="Survived out-of-sample (test > 0)", zorder=3)
    if fail:
        ax.scatter([a for a, _ in fail], [b for _, b in fail], s=60, color=AMBER,
                   edgecolor="white", linewidth=0.5, label="Failed out-of-sample (test < 0)", zorder=3)
    ax.axhline(0, color="#333", linewidth=1.2)
    lim = max(abs(v) for pr in pts for v in pr[:2]) * 1.1
    ax.plot([0, lim], [0, lim], color="#bbb", linestyle=":", linewidth=1, label="test = train (no decay)")
    ax.set_xlabel("Training PnL  (days 2 + 3, best in-sample config)")
    ax.set_ylabel("Held-out test PnL  (day 4, same config)")
    ax.set_title("Walk-Forward Validation: In-Sample Fit Does Not Guarantee Out-of-Sample PnL")
    # shade the failure region
    ax.axhspan(-lim, 0, color=AMBER, alpha=0.05)
    ax.annotate("Looked strong in training,\nlost money on the unseen day",
                xy=(lim*0.55, -lim*0.35), fontsize=9.5, color=AMBER, ha="center")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_xlim(0, lim); ax.set_ylim(-lim, lim)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "walkforward_validation.png"), dpi=140); plt.close(fig)
    print("  walkforward_validation.png")


if __name__ == "__main__":
    print("Generating quantitative analysis charts from real competition data:")
    correlation_heatmap()
    volatility_profile()
    mean_reversion_surface()
    risk_analysis()
    vol_smile_and_surface()
    walk_forward_validation()
    print("Done ->", OUT)
