"""R5 Manual Round optimizer.

Solves the quadratic-fee optimal allocation problem given expected returns.
Edit RETURNS dict below to adjust scenarios; rerun for sensitivity analysis.

Math:
  fee_i = (p_i/100)^2 * BUDGET = 100 * p_i^2  (when p in %, BUDGET=1M)
  pnl_i = p_i * 10000 * |r_i| - 100 * p_i^2   (if direction correct)

  Unconstrained optimum: p_i* = 50 * |r_i|
  If sum > 100, apply Lagrangian: p_i* = max(0, 50*|r_i| - lambda/200)
                                  with lambda solved so sum = 100.
"""
BUDGET = 1_000_000

# Direction (+1 BUY, -1 SELL) and expected return magnitude
PRODUCTS = {
    "Lava Cake":            (-1, 0.45),  # sales halted, lawsuits
    "Thermalite Core":      (+1, 0.40),  # 2.7x user growth concrete numbers
    "Ashes of the Phoenix": (-1, 0.30),  # ESG outcry
    "Pyroflex Cells":       (-1, 0.25),  # tax cut ends, demand drop
    "Magma Ink":            (+1, 0.20),  # hot product post-merger
    "Sulfur Reactor":       (+1, 0.12),  # index inclusion (stock)
    "Obsidian Cutlery":     (+1, 0.12),  # supply shock
    "Volcanic Incense":     (+1, 0.08),  # pump signal small
    "Scoria Paste":         (+1, 0.03),  # sketchy endorsement small
}

def solve(products):
    """Returns list of (name, direction, |r|, p%, gross_revenue, fee, net_pnl)."""
    items = list(products.items())
    rs = [abs(r) for _, (_, r) in items]
    sum_r = sum(rs)
    sum_p_unc = 50 * sum_r

    if sum_p_unc <= 100:
        # unconstrained optimum
        ps = [50 * r for r in rs]
        lam = 0.0
    else:
        # constrained: lambda so sum p = 100
        # 50*sum_r - n*lambda/200 = 100  => lambda = (sum_p_unc - 100) * 200 / n
        n = len(items)
        lam = (sum_p_unc - 100) * 200 / n
        ps = [max(0.0, 50 * r - lam/200) for r in rs]
        # If any p hit zero, redistribute (rare; skip for now)

    out = []
    total_pnl = 0.0
    total_fee = 0.0
    total_invest = 0.0
    for (name, (d, r)), p in zip(items, ps):
        invest = p * 10_000           # = (p/100) * BUDGET
        fee = 100 * p * p              # = (p/100)^2 * BUDGET
        gross = invest * abs(r)        # absolute return × invested
        net = gross - fee
        total_pnl += net
        total_fee += fee
        total_invest += invest
        out.append((name, d, r, p, invest, fee, gross, net))
    return out, total_pnl, total_fee, total_invest, lam

def print_table(rows, total_pnl, total_fee, total_invest, lam):
    print(f"{'Product':<24} {'Dir':>4} {'|r|':>6} {'p%':>7} {'Invest':>10} {'Fee':>10} {'Gross':>10} {'Net PnL':>10}")
    print("-"*92)
    for name, d, r, p, inv, fee, gross, net in rows:
        dir_str = "BUY" if d > 0 else "SELL"
        print(f"{name:<24} {dir_str:>4} {r:>6.2%} {p:>6.2f}% {inv:>10,.0f} {fee:>10,.0f} {gross:>10,.0f} {net:>+10,.0f}")
    print("-"*92)
    total_p = sum(r[3] for r in rows)
    print(f"{'TOTALS':<24} {'':>4} {'':>6} {total_p:>6.2f}% {total_invest:>10,.0f} {total_fee:>10,.0f} {sum(r[6] for r in rows):>10,.0f} {total_pnl:>+10,.0f}")
    if lam > 0:
        print(f"\n  (Constraint binds: lambda = {lam:.2f}, allocations scaled down from unconstrained)")
    else:
        print(f"\n  (Unconstrained optimum, total allocation = {total_p:.2f}% of {100}% budget)")
    print(f"  Unused budget (expires worthless): {1_000_000 - total_invest:,.0f}")

def solve_integer(products):
    """Integer-constrained optimum (platform requires whole-percent allocations).
    For each product, the PnL is flat ±0.5% around p*=50|r|; tied integers give same PnL.
    Pick the rounding that maximises total budget use without exceeding 100%.
    """
    items = list(products.items())
    # For each product, compute candidate ints and their per-product PnL
    candidates = []
    for name, (d, r) in items:
        p_star = 50 * abs(r)
        lo = max(0, int(p_star))
        hi = lo + 1
        pnl_lo = lo * 10000 * abs(r) - 100 * lo * lo
        pnl_hi = hi * 10000 * abs(r) - 100 * hi * hi
        # Pick whichever is non-negative; if both, prefer the one that uses more budget
        # (since at exactly p* both give same PnL, but unused budget is worthless)
        if pnl_hi >= pnl_lo and pnl_hi > 0:
            chosen = hi
        elif pnl_lo > 0:
            chosen = lo
        else:
            chosen = 0
        candidates.append((name, d, abs(r), chosen, pnl_lo, pnl_hi))

    # Check budget constraint
    total_p = sum(c[3] for c in candidates)
    if total_p > 100:
        # downsize: drop from products where downsizing costs least PnL
        excess = total_p - 100
        # cost of downsizing product i by 1: pnl(p) - pnl(p-1)
        downsize_costs = []
        for i, (name, d, r, p, _, _) in enumerate(candidates):
            cost = (p * 10000 * r - 100 * p * p) - ((p-1) * 10000 * r - 100 * (p-1)**2)
            downsize_costs.append((cost, i))
        downsize_costs.sort()  # cheapest downsizes first
        for _ in range(excess):
            _, i = downsize_costs[0]
            name, d, r, p, plo, phi = candidates[i]
            candidates[i] = (name, d, r, p-1, plo, phi)
            # recompute marginal cost for that product
            new_cost = ((p-1) * 10000 * r - 100 * (p-1)**2) - ((p-2) * 10000 * r - 100 * (p-2)**2)
            downsize_costs[0] = (new_cost, i)
            downsize_costs.sort()

    # Build output rows
    out = []
    total_pnl = 0; total_fee = 0; total_invest = 0
    for name, d, r, p, _, _ in candidates:
        invest = p * 10000
        fee = 100 * p * p
        gross = invest * r
        net = gross - fee
        total_pnl += net; total_fee += fee; total_invest += invest
        out.append((name, d, r, p, invest, fee, gross, net))
    return out, total_pnl, total_fee, total_invest, 0.0

if __name__ == "__main__":
    print("=" * 92)
    print("CONTINUOUS OPTIMUM (theoretical, decimal %)")
    print("=" * 92)
    rows, pnl, fee, inv, lam = solve(PRODUCTS)
    print_table(rows, pnl, fee, inv, lam)

    print("\n\n" + "=" * 92)
    print("INTEGER-CONSTRAINED OPTIMUM (platform-submittable, whole %)")
    print("=" * 92)
    rows, pnl, fee, inv, lam = solve_integer(PRODUCTS)
    print_table(rows, pnl, fee, inv, lam)
    print("\n  ^ This is what to submit on the platform.")

    # Sensitivity: what if we OVERSIZE by 20% (more aggressive)?
    print("\n\n=== SENSITIVITY: AGGRESSIVE (+20% to all returns) ===")
    aggressive = {k: (d, r * 1.2) for k, (d, r) in PRODUCTS.items()}
    rows, pnl, fee, inv, lam = solve(aggressive)
    print_table(rows, pnl, fee, inv, lam)

    # Sensitivity: CONSERVATIVE (-30% to all returns)
    print("\n\n=== SENSITIVITY: CONSERVATIVE (-30% to all returns) ===")
    conservative = {k: (d, r * 0.7) for k, (d, r) in PRODUCTS.items()}
    rows, pnl, fee, inv, lam = solve(conservative)
    print_table(rows, pnl, fee, inv, lam)

    # What if we're WRONG on the most uncertain ones (Volcanic, Scoria)?
    print("\n\n=== SENSITIVITY: WRONG on Volcanic & Scoria (direction flipped) ===")
    wrong = dict(PRODUCTS)
    wrong["Volcanic Incense"] = (+1, 0.08)  # we said BUY, actual = -8% (we lose)
    wrong["Scoria Paste"]     = (+1, 0.03)  # we said BUY, actual = -3% (we lose)
    # Recompute PnL with WRONG direction (negative gross)
    rows, _, fee, inv, _ = solve(wrong)
    actual_pnl = 0.0
    print(f"\n{'Product':<24} {'p%':>7} {'gross':>10} {'fee':>10} {'net':>10}")
    for name, d, r, p, inv_i, fee_i, _, _ in rows:
        if name in ("Volcanic Incense", "Scoria Paste"):
            gross = -inv_i * r  # we bet wrong direction
        else:
            gross = inv_i * r
        net = gross - fee_i
        actual_pnl += net
        print(f"{name:<24} {p:>6.2f}% {gross:>+10,.0f} {fee_i:>10,.0f} {net:>+10,.0f}")
    print(f"  Total if wrong on Volcanic+Scoria: {actual_pnl:+,.0f}")
