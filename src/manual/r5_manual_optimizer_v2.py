"""R5 Manual Round v2 — full Monte Carlo + uncertainty-aware optimizer.

Each product has a probability distribution over actual return r:
- Mean (point estimate)
- Std (estimation uncertainty)
- For ambiguous/satirical products, mixture of two scenarios

Outputs:
1. Optimal allocation (at point estimate)
2. Monte Carlo PnL distribution under uncertainty
3. Sensitivity analysis (what if our estimate is off by ±25%)
4. The integer-rounded submittable table
"""
import random
random.seed(42)

BUDGET = 1_000_000

# (direction, point_estimate_r, std, [optional: mixture])
PRODUCTS = {
    # HIGH-CONFIDENCE products — point estimate close to true mean, low uncertainty
    "Lava Cake":            {"dir": -1, "r": 0.47, "std": 0.08, "mix": None},   # sales halt + lawsuits
    "Thermalite Core":      {"dir": +1, "r": 0.40, "std": 0.07, "mix": None},   # 2.74x user growth
    "Ashes of the Phoenix": {"dir": -1, "r": 0.32, "std": 0.06, "mix": None},   # ESG + viral
    "Pyroflex Cells":       {"dir": -1, "r": 0.27, "std": 0.05, "mix": None},   # tax doubles
    "Magma Ink":            {"dir": +1, "r": 0.20, "std": 0.05, "mix": None},   # hot drop
    "Sulfur Reactor":       {"dir": +1, "r": 0.12, "std": 0.05, "mix": None},   # index inclusion
    "Obsidian Cutlery":     {"dir": +1, "r": 0.10, "std": 0.07, "mix": None},   # supply shock vs contamination
    # AMBIGUOUS products — use mixture of fade-vs-momentum scenarios
    "Volcanic Incense":     {"dir": -1, "r": 0.028, "std": 0.05,
                             "mix": [(0.60, -0.10, 0.06), (0.40, +0.08, 0.04)]},  # 60% fade, 40% momentum
    "Scoria Paste":         {"dir": -1, "r": 0.019, "std": 0.04,
                             "mix": [(0.70, -0.04, 0.04), (0.30, +0.03, 0.03)]},  # 70% fade, 30% momentum
}

def optimal_p_continuous(r):
    """Unconstrained optimum. Optimal p in % for given expected |r|."""
    return 50 * abs(r)

def pnl(p, r_signed):
    """PnL for allocation p (in %) and SIGNED actual return r_signed."""
    invest = p * 10_000
    fee = 100 * p * p
    gross = invest * r_signed       # signed return * invested
    return gross - fee               # if direction matches sign(r_signed), gross > 0

def integer_optimum(products):
    """Find integer allocation optimizing expected PnL with budget constraint.

    Greedy: start with 50|r| rounded, then add/remove 1% at margin until constraint met."""
    items = list(products.items())
    ps = []
    for name, info in items:
        p_star = optimal_p_continuous(info["r"])
        # round to nearest int, but pick the one that maximises pnl at point estimate
        lo, hi = int(p_star), int(p_star) + 1
        # NB: pnl at OPTIMAL point estimate (signed by direction)
        signed_r = info["dir"] * info["r"]
        # When we BUY (dir=+1), we want price to rise (r>0). signed_r > 0.
        # When we SELL (dir=-1), we want price to fall. signed_r < 0.
        # PnL = invest × signed_r when direction matches: but invest is unsigned, so:
        # PnL = p × 10000 × |r| - fee   (when direction is right)
        pnl_lo = lo * 10000 * info["r"] - 100 * lo * lo
        pnl_hi = hi * 10000 * info["r"] - 100 * hi * hi
        chosen = hi if pnl_hi >= pnl_lo and pnl_hi > 0 else (lo if pnl_lo > 0 else 0)
        ps.append(chosen)

    # Enforce budget
    while sum(ps) > 100:
        # downsize cheapest
        costs = [(ps[i] * 10000 * items[i][1]["r"] - 100 * ps[i]**2) -
                 ((ps[i]-1) * 10000 * items[i][1]["r"] - 100 * (ps[i]-1)**2)
                 for i in range(len(ps))]
        i_min = costs.index(min(costs))
        ps[i_min] -= 1

    return [(items[i][0], items[i][1], ps[i]) for i in range(len(items))]

def sample_r(info):
    """Draw one sample of actual return r (signed magnitude) from the distribution."""
    if info["mix"]:
        # mixture: pick component then draw
        u = random.random()
        cum = 0
        for prob, mu, sd in info["mix"]:
            cum += prob
            if u < cum:
                return random.gauss(mu, sd)
        return random.gauss(info["mix"][-1][1], info["mix"][-1][2])
    return random.gauss(info["dir"] * info["r"], info["std"])  # signed

def monte_carlo(allocation, n=50_000):
    """Returns (mean, std, p10, p50, p90, prob_pnl>0)."""
    pnls = []
    for _ in range(n):
        total = 0
        for name, info, p in allocation:
            r_actual = sample_r(info)
            # PnL accounting: our position is p% in info["dir"] direction.
            # If we BUY (dir=+1) and r > 0, we gain. If r < 0, we lose.
            # Generic: pnl = p * 10000 * (info["dir"] * r_actual) - fee
            pos_gain = p * 10000 * (info["dir"] * r_actual)
            fee = 100 * p * p
            total += pos_gain - fee
        pnls.append(total)
    pnls.sort()
    return {
        "mean": sum(pnls) / n,
        "std": (sum((x - sum(pnls)/n)**2 for x in pnls) / n) ** 0.5,
        "p10": pnls[int(0.10 * n)],
        "p50": pnls[int(0.50 * n)],
        "p90": pnls[int(0.90 * n)],
        "p_positive": sum(1 for p in pnls if p > 0) / n,
        "p_above_100k": sum(1 for p in pnls if p > 100_000) / n,
        "p_above_150k": sum(1 for p in pnls if p > 150_000) / n,
    }

def print_allocation(alloc, label):
    print(f"\n{'='*100}\n{label}\n{'='*100}")
    print(f"{'Product':<24} {'Dir':>5} {'|r|est':>8} {'p%':>5} {'Invest':>12} {'Fee':>10} {'E[gross]':>12} {'E[net]':>12}")
    print("-"*100)
    total_p, total_pnl, total_fee, total_inv = 0, 0, 0, 0
    for name, info, p in alloc:
        invest = p * 10_000
        fee = 100 * p * p
        # Expected gross = p * 10000 * E[dir × r_actual]. For non-mixture, E[dir × r_actual] = dir × dir × r = r (positive).
        # For mixture, compute E directly.
        if info["mix"]:
            e_signed_r = sum(prob * mu for prob, mu, sd in info["mix"])
            e_dir_r = info["dir"] * e_signed_r  # signed PnL contribution per dollar invested
        else:
            e_dir_r = info["r"]  # we bet direction; if right, PnL = invest * |r|
        e_gross = invest * e_dir_r
        e_net = e_gross - fee
        dir_str = "BUY" if info["dir"] > 0 else "SELL"
        print(f"{name:<24} {dir_str:>5} {info['r']:>8.2%} {p:>4d}% {invest:>12,.0f} {fee:>10,.0f} {e_gross:>+12,.0f} {e_net:>+12,.0f}")
        total_p += p; total_pnl += e_net; total_fee += fee; total_inv += invest
    print("-"*100)
    print(f"{'TOTALS':<24} {'':>5} {'':>8} {total_p:>4d}% {total_inv:>12,.0f} {total_fee:>10,.0f} {'':>12} {total_pnl:>+12,.0f}")
    print(f"  Unused budget (expires worthless): ${1_000_000 - total_inv:,.0f}")

if __name__ == "__main__":
    alloc = integer_optimum(PRODUCTS)
    print_allocation(alloc, "OPTIMAL INTEGER ALLOCATION (point estimates)")

    print(f"\n{'='*100}\nMONTE CARLO PnL DISTRIBUTION (50,000 draws)\n{'='*100}")
    stats = monte_carlo(alloc)
    print(f"  Mean PnL:    ${stats['mean']:>+10,.0f}")
    print(f"  Std PnL:     ${stats['std']:>10,.0f}")
    print(f"  10th %ile:   ${stats['p10']:>+10,.0f}  (1-in-10 worst)")
    print(f"  Median:      ${stats['p50']:>+10,.0f}")
    print(f"  90th %ile:   ${stats['p90']:>+10,.0f}  (1-in-10 best)")
    print(f"  P(PnL > 0):     {stats['p_positive']:>6.1%}")
    print(f"  P(PnL > $100K): {stats['p_above_100k']:>6.1%}")
    print(f"  P(PnL > $150K): {stats['p_above_150k']:>6.1%}")

    # Sensitivity: what if all magnitudes are 25% smaller?
    print(f"\n{'='*100}\nSENSITIVITY TEST: returns 25% smaller than estimated\n{'='*100}")
    pess = {k: dict(v, r=v["r"]*0.75) for k, v in PRODUCTS.items()}
    for k in pess:
        if PRODUCTS[k]["mix"]:
            pess[k]["mix"] = [(p, m*0.75, s) for p, m, s in PRODUCTS[k]["mix"]]
    pess_alloc = integer_optimum(pess)
    pess_stats = monte_carlo([(name, PRODUCTS[name], p) for name, info, p in alloc])  # use ORIGINAL allocation
    print(f"  If we keep our allocation but actual returns are 25% smaller:")
    print(f"    Mean PnL:  ${pess_stats['mean']:>+10,.0f}  (vs ${stats['mean']:>+,.0f} base)")

    # Sensitivity: 25% bigger
    print(f"\nSENSITIVITY TEST: returns 25% bigger than estimated")
    opt = {k: dict(v, r=v["r"]*1.25) for k, v in PRODUCTS.items()}
    for k in opt:
        if PRODUCTS[k]["mix"]:
            opt[k]["mix"] = [(p, m*1.25, s) for p, m, s in PRODUCTS[k]["mix"]]
    opt_stats_keep = monte_carlo([(name, opt[name], p) for name, info, p in alloc])
    print(f"  If we keep our allocation but actual returns are 25% bigger:")
    print(f"    Mean PnL:  ${opt_stats_keep['mean']:>+10,.0f}  (vs ${stats['mean']:>+,.0f} base)")

    # Compare with the conservative (BUY both) variant
    print(f"\n{'='*100}\nALTERNATIVE: Conservative variant (BUY Volcanic + BUY Scoria)\n{'='*100}")
    conservative = {**PRODUCTS}
    conservative["Volcanic Incense"] = {"dir": +1, "r": 0.06, "std": 0.06, "mix": None}
    conservative["Scoria Paste"] = {"dir": +1, "r": 0.03, "std": 0.04, "mix": None}
    cons_alloc = integer_optimum(conservative)
    cons_stats = monte_carlo(cons_alloc)
    print_allocation(cons_alloc, "CONSERVATIVE: BUY Volcanic & Scoria")
    print(f"\n  Mean PnL:    ${cons_stats['mean']:>+10,.0f}  (vs ${stats['mean']:>+,.0f} fade variant)")
    print(f"  P(PnL > 0):     {cons_stats['p_positive']:>6.1%}  (vs {stats['p_positive']:.1%} fade)")
