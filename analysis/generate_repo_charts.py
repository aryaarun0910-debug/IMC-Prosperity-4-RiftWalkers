"""Generate professional charts for the repository documentation.

Outputs to docs/assets/. Charts:
  1. cumulative_progression.png  - cumulative XIRECs across rounds
  2. algo_vs_manual.png          - per-round algo vs manual PnL + rank
  3. overfitting_comparison.png  - 1-day aesthetic vs 3-day backtest (the key lesson)
  4. quadratic_fee_surface_3d.png- 3D PnL surface for the R5 manual fee model
  5. manual_optimal_sizing_2d.png- optimal allocation frontier + breakeven
  6. montecarlo_distribution.png - R5 manual PnL distribution (50k draws)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")
os.makedirs(OUT, exist_ok=True)

# Consistent professional styling
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
AMBER = "#e76f51"
GREY = "#6c757d"

# ---- Official results ----
rounds = ["R1", "R2", "R3", "R4", "R5"]
cumulative = [183387, 444349, 133190, 176419, 272456]  # note: phase 2 resets at R3
algo_pnl = [95490, 73268, 65477, 19664, 6849]
manual_pnl = [87897, 187694, 67713, 23566, 89187]
algo_rank = [1642, 3407, 504, 1265, 896]
manual_rank = [6, 128, 541, 699, 411]
TOTAL_TEAMS = 18803

# =====================================================================
# 1. Cumulative progression (split by phase)
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5))
phase1_x = [0, 1]; phase1_y = [183387, 444349]
phase2_x = [2, 3, 4]; phase2_y = [133190, 176419, 272456]
ax.plot(phase1_x, phase1_y, "o-", color=NAVY, linewidth=2.5, markersize=9, label="Phase 1 (Qualifier)")
ax.plot(phase2_x, phase2_y, "o-", color=TEAL, linewidth=2.5, markersize=9, label="Phase 2 (Final, leaderboard reset)")
ax.axvline(1.5, color=GREY, linestyle="--", linewidth=1, alpha=0.7)
ax.text(1.52, 420000, "Leaderboard\nreset", fontsize=9, color=GREY, va="top")
for x, y in zip(range(5), cumulative):
    ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(5)); ax.set_xticklabels(rounds)
ax.set_ylabel("Cumulative XIRECs")
ax.set_title("Cumulative Score Across Rounds")
ax.legend(loc="lower right", framealpha=0.9)
ax.set_ylim(0, 520000)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cumulative_progression.png"), dpi=140); plt.close(fig)

# =====================================================================
# 2. Algo vs Manual PnL + rank percentile
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(5); w = 0.38
ax1.bar(x - w/2, algo_pnl, w, color=NAVY, label="Algorithmic")
ax1.bar(x + w/2, manual_pnl, w, color=TEAL, label="Manual")
ax1.set_xticks(x); ax1.set_xticklabels(rounds)
ax1.set_ylabel("PnL (XIRECs)")
ax1.set_title("Per-Round PnL: Algorithmic vs Manual")
ax1.legend(framealpha=0.9)
for i in range(5):
    ax1.annotate(f"{algo_pnl[i]/1000:.0f}k", (i - w/2, algo_pnl[i]), textcoords="offset points", xytext=(0,4), ha="center", fontsize=8)
    ax1.annotate(f"{manual_pnl[i]/1000:.0f}k", (i + w/2, manual_pnl[i]), textcoords="offset points", xytext=(0,4), ha="center", fontsize=8)

# Rank percentile (lower = better) -> convert to top X%
algo_pct = [r / TOTAL_TEAMS * 100 for r in algo_rank]
manual_pct = [r / TOTAL_TEAMS * 100 for r in manual_rank]
ax2.plot(x, algo_pct, "o-", color=NAVY, linewidth=2, markersize=8, label="Algorithmic")
ax2.plot(x, manual_pct, "o-", color=TEAL, linewidth=2, markersize=8, label="Manual")
ax2.invert_yaxis()
ax2.set_xticks(x); ax2.set_xticklabels(rounds)
ax2.set_ylabel("Rank percentile (top %, lower is better)")
ax2.set_title("Per-Round Rank Percentile (of 18,803 teams)")
ax2.legend(framealpha=0.9)
ax2.annotate("R1 Manual: 6th globally\n(top 0.03%)", (0, manual_pct[0]),
             textcoords="offset points", xytext=(20, 20), fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=TEAL))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "algo_vs_manual.png"), dpi=140); plt.close(fig)

# =====================================================================
# 3. Overfitting comparison: 1-day aesthetic vs 3-day backtest
# =====================================================================
labels = ["Overfit\n(hardcoded direction)", "Signal-based\n(cross-day validated)"]
aesthetic = [174603, 34505]          # 1-day "looks good" test
backtest_total = [44486, 615749]      # 3-day integrated
worst_day = [-33050, 143139]          # worst single day

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(2); w = 0.25
b1 = ax.bar(x - w, aesthetic, w, color=AMBER, label="1-day aesthetic test")
b2 = ax.bar(x, backtest_total, w, color=NAVY, label="3-day integrated backtest (total)")
b3 = ax.bar(x + w, worst_day, w, color=TEAL, label="Worst single day")
ax.axhline(0, color="#333", linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("PnL (XIRECs)")
ax.set_title("Why the Best-Looking Algo Was the Worst: Single-Day vs Cross-Day Validation")
ax.legend(framealpha=0.9, loc="upper left")
for bars in (b1, b2, b3):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h/1000:+.0f}k", (b.get_x()+b.get_width()/2, h),
                    textcoords="offset points", xytext=(0, 5 if h >= 0 else -14),
                    ha="center", fontsize=8, fontweight="bold")
ax.annotate("Loses money on a\ndifferent day", (0 + w, worst_day[0]),
            textcoords="offset points", xytext=(30, -10), fontsize=9, color=AMBER,
            arrowprops=dict(arrowstyle="->", color=AMBER))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "overfitting_comparison.png"), dpi=140); plt.close(fig)

# =====================================================================
# 4. 3D PnL surface for the R5 manual quadratic-fee model
#    PnL(p, r) = p*10000*|r| - 100*p^2
# =====================================================================
p = np.linspace(0, 40, 120)        # allocation %
r = np.linspace(0, 0.6, 120)       # |expected return|
P, R = np.meshgrid(p, r)
PNL = P * 10000 * R - 100 * P**2

fig = plt.figure(figsize=(12, 8.5))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(P, R*100, PNL, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)
# optimal ridge: p* = 50r
r_line = np.linspace(0, 0.6, 50)
p_line = 50 * r_line
pnl_line = p_line * 10000 * r_line - 100 * p_line**2
ax.plot(p_line, r_line*100, pnl_line, color=AMBER, linewidth=3.5, label="Optimal ridge:  p* = 50|r|")
ax.set_xlabel("\nAllocation  p (%)", fontsize=11)
ax.set_ylabel("\nExpected return |r| (%)", fontsize=11)
ax.set_zlabel("\nPnL (XIRECs)", fontsize=11)
ax.set_title("Round 5 Manual: PnL Surface Under Quadratic Fees", y=0.98)
ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.92))
ax.view_init(elev=24, azim=-58)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=14, pad=0.10, label="PnL (XIRECs)")
fig.savefig(os.path.join(OUT, "quadratic_fee_surface_3d.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

# =====================================================================
# 5. 2D optimal sizing frontier + breakeven
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
rr = np.linspace(0, 0.6, 200)
p_star = 50 * rr
ax.plot(rr*100, p_star, color=NAVY, linewidth=2.5, label="Optimal allocation  p* = 50|r|")
ax.plot(rr*100, rr*100, color=AMBER, linewidth=2, linestyle="--", label="Breakeven  p = 100|r|  (above this, fees > gains)")
ax.fill_between(rr*100, p_star, rr*100, where=(p_star <= rr*100), color=TEAL, alpha=0.15)
ax.fill_between(rr*100, 0, p_star, color=NAVY, alpha=0.06)
# Our actual R5 manual allocations
prods = ["Lava Cake", "Thermalite", "Ashes", "Pyroflex", "Magma Ink", "Sulfur", "Obsidian"]
prod_r = [47, 40, 32, 27, 20, 12, 10]
prod_p = [24, 20, 16, 14, 10, 6, 5]
ax.scatter(prod_r, prod_p, color="#d62828", zorder=5, s=55, label="Our R5 manual positions")
for nm, xr, yp in zip(prods, prod_r, prod_p):
    ax.annotate(nm, (xr, yp), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Expected return |r| (%)")
ax.set_ylabel("Budget allocation p (%)")
ax.set_title("Round 5 Manual: Optimal Sizing Frontier")
ax.legend(framealpha=0.9, loc="upper left", fontsize=9)
ax.set_xlim(0, 55); ax.set_ylim(0, 30)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "manual_optimal_sizing_2d.png"), dpi=140); plt.close(fig)

# =====================================================================
# 6. Monte Carlo PnL distribution (R5 manual)
# =====================================================================
rng = np.random.default_rng(42)
# per-product (direction-correct) return draws ~ Normal(mu, sigma), allocations fixed
mus = np.array([0.47, 0.40, 0.32, 0.27, 0.20, 0.12, 0.10, 0.028, 0.019])
sigs = np.array([0.09, 0.08, 0.06, 0.06, 0.06, 0.045, 0.08, 0.05, 0.05])
ps = np.array([24, 20, 16, 14, 10, 6, 5, 1, 1])
N = 50000
draws = rng.normal(mus, sigs, size=(N, len(mus)))
pnl = (ps * 10000 * draws - 100 * ps**2).sum(axis=1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(pnl, bins=70, color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.4)
ax.axvline(np.median(pnl), color=AMBER, linewidth=2, label=f"Median  {np.median(pnl)/1000:.0f}k")
ax.axvline(np.percentile(pnl, 10), color=TEAL, linewidth=1.5, linestyle="--", label=f"10th pct  {np.percentile(pnl,10)/1000:.0f}k")
ax.axvline(np.percentile(pnl, 90), color=TEAL, linewidth=1.5, linestyle="--", label=f"90th pct  {np.percentile(pnl,90)/1000:.0f}k")
ax.axvline(89187, color="#d62828", linewidth=2, label="Actual delivered  89k")
ax.set_xlabel("Portfolio PnL (XIRECs)")
ax.set_ylabel("Frequency")
ax.set_title("Round 5 Manual: Monte-Carlo PnL Distribution (50,000 draws)")
ax.legend(framealpha=0.9, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "montecarlo_distribution.png"), dpi=140); plt.close(fig)

print("Charts written to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
