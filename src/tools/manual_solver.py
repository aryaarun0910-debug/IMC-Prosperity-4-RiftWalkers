"""
Manual Trading Round Solver — Monte Carlo, Nash Equilibrium, Convex Optimization.

Top teams (CMU Physics, Frankfurt Hedgehogs) scored 30-200K from manual rounds.
This tool solves the common challenge types:

Usage:
    python manual_solver.py auction --params "..."
    python manual_solver.py nash --params "..."
    python manual_solver.py portfolio --params "..."
    python manual_solver.py conversion --params "..."

Or import and use programmatically:
    from manual_solver import AuctionSolver, NashSolver, PortfolioSolver
"""

import math
import random
import json
import sys
from itertools import product as cartesian_product


# ============================================================
# TYPE 1: AUCTION / PRICING SOLVER
# Expected value maximization with Monte Carlo simulation
# ============================================================

class AuctionSolver:
    """Solve auction/pricing problems.

    Common in R1 manual: "You can buy X at price P. What's the optimal bid?"

    Example: Fish gear auction with linear probability.
        - You bid B for gear worth V
        - P(win) = f(B) (e.g., linear: B/1000)
        - EV = P(win) * (V - B)
        - Optimal B maximizes EV
    """

    @staticmethod
    def optimal_bid_analytic(value, max_bid, prob_fn="linear"):
        """Find optimal bid analytically for common probability functions.
        prob_fn: 'linear' (P=B/max_bid), 'quadratic' (P=(B/max_bid)^2),
                 'sqrt' (P=sqrt(B/max_bid))"""
        best_bid = 0
        best_ev = 0

        # Brute force over integer bids
        for b in range(0, int(max_bid) + 1):
            ratio = b / max_bid
            if prob_fn == "linear":
                p_win = ratio
            elif prob_fn == "quadratic":
                p_win = ratio ** 2
            elif prob_fn == "sqrt":
                p_win = ratio ** 0.5
            else:
                p_win = ratio

            ev = p_win * (value - b)
            if ev > best_ev:
                best_ev = ev
                best_bid = b

        return best_bid, best_ev

    @staticmethod
    def monte_carlo_auction(value_distribution, cost_fn, n_sims=100000, seed=42):
        """Monte Carlo simulation for complex auction mechanics.

        value_distribution: callable() -> random value
        cost_fn: callable(bid) -> cost to you

        Returns: {bid: avg_profit} for each bid tried."""
        rng = random.Random(seed)
        random.seed(seed)

        results = {}
        bids_to_try = range(0, 1001, 10)  # try bids in steps of 10

        for bid in bids_to_try:
            total_profit = 0
            for _ in range(n_sims):
                value = value_distribution()
                cost = cost_fn(bid)
                # Simple model: you win if your bid >= some threshold
                # Override this for specific auction mechanics
                profit = max(0, value - cost) if bid > 0 else 0
                total_profit += profit
            results[bid] = total_profit / n_sims

        best_bid = max(results, key=results.get)
        return best_bid, results[best_bid], results

    @staticmethod
    def expected_value(outcomes, probabilities):
        """Calculate EV from discrete outcomes.
        outcomes: list of values
        probabilities: list of probabilities (must sum to 1)"""
        return sum(o * p for o, p in zip(outcomes, probabilities))


# ============================================================
# TYPE 2: CONVERSION / PATH OPTIMIZATION
# BFS/brute-force for optimal conversion paths
# ============================================================

class ConversionSolver:
    """Solve item conversion/path problems.

    Common in R2: "Convert items through a chain for maximum profit"
    Example: Shells -> Snowballs -> Pizzas -> Shells (arbitrage loop)
    """

    @staticmethod
    def find_best_path(rates, start, target=None, max_steps=10):
        """Find best conversion path via BFS.

        rates: dict of {(from, to): rate}
            e.g., {("SHELLS", "PIZZA"): 0.5, ("PIZZA", "SHELLS"): 2.5}
        start: starting item
        target: target item (None = same as start for arbitrage)
        max_steps: max number of conversions

        Returns: (best_value, best_path)"""
        if target is None:
            target = start

        # BFS with value tracking
        # State: (current_item, value_so_far, path)
        queue = [(start, 1.0, [start])]
        best_value = 1.0 if start == target else 0.0
        best_path = [start]

        for step in range(max_steps):
            next_queue = []
            for item, value, path in queue:
                # Try all conversions from current item
                for (src, dst), rate in rates.items():
                    if src == item:
                        new_value = value * rate
                        new_path = path + [dst]

                        if dst == target and new_value > best_value:
                            best_value = new_value
                            best_path = new_path

                        if len(new_path) <= max_steps:
                            next_queue.append((dst, new_value, new_path))

            queue = next_queue
            if not queue:
                break

        return best_value, best_path

    @staticmethod
    def brute_force_allocation(items, values, capacity):
        """0/1 knapsack for item selection problems.
        items: list of item names
        values: list of values
        capacity: max items to select (or weight limit)

        Returns: (best_value, selected_items)"""
        n = len(items)
        best_value = 0
        best_combo = []

        # Try all 2^n combinations (feasible for n < 20)
        for mask in range(1 << n):
            selected = []
            total_value = 0
            count = 0
            for i in range(n):
                if mask & (1 << i):
                    selected.append(items[i])
                    total_value += values[i]
                    count += 1
            if count <= capacity and total_value > best_value:
                best_value = total_value
                best_combo = selected

        return best_value, best_combo


# ============================================================
# TYPE 3: NASH EQUILIBRIUM / ANTI-HERDING
# Container/suitcase selection problems
# ============================================================

class NashSolver:
    """Solve Nash equilibrium problems for container/suitcase selection.

    Common in R2, R4: "Choose a container. Payout = base_value * multiplier /
    (residents + fraction_of_players_choosing_same)"

    Top teams (CMU Physics): Used Discord sentiment + Nash calculation.
    """

    @staticmethod
    def compute_ev(multipliers, n_players, player_distribution):
        """Compute EV for each choice given player distribution.

        multipliers: list of multipliers per choice
        n_players: total number of players
        player_distribution: list of fractions (sum to 1.0) for each choice

        Returns: list of EVs per choice"""
        base_value = 10000  # typical IMC base
        evs = []
        for i, mult in enumerate(multipliers):
            fraction = player_distribution[i]
            n_choosing = max(1, fraction * n_players)
            ev = base_value * mult / n_choosing
            evs.append(ev)
        return evs

    @staticmethod
    def find_nash_equilibrium(multipliers, n_players, residents=None,
                              base_value=10000, max_iter=10000):
        """Find Nash equilibrium distribution via iterated best response.

        multipliers: list of multipliers per choice
        residents: list of fixed residents per choice (default 0)
        n_players: number of players choosing
        base_value: base payout value

        At Nash equilibrium, all chosen options have equal EV.

        Returns: (equilibrium_fractions, equilibrium_ev, best_choice)"""
        n = len(multipliers)
        if residents is None:
            residents = [0] * n

        # At Nash equilibrium: EV_i = base * mult_i / (residents_i + frac_i * n_players)
        # Equal across all chosen options.
        # Solve: mult_i / (r_i + f_i * N) = mult_j / (r_j + f_j * N) for all i,j
        # Subject to: sum(f_i) = 1, f_i >= 0

        # Iterative approach: start uniform, then best-response dynamics
        fracs = [1.0 / n] * n

        for iteration in range(max_iter):
            # Compute EVs
            evs = []
            for i in range(n):
                denom = residents[i] + fracs[i] * n_players
                if denom <= 0:
                    denom = 0.01
                evs.append(base_value * multipliers[i] / denom)

            # Best response: shift mass toward highest EV options
            max_ev = max(evs)
            min_ev = min(evs)

            if max_ev - min_ev < 0.01:  # converged
                break

            # Gradient step
            lr = 0.01
            new_fracs = list(fracs)
            avg_ev = sum(evs) / n
            for i in range(n):
                adjustment = lr * (evs[i] - avg_ev) / max(1, avg_ev)
                new_fracs[i] = max(0.001, fracs[i] + adjustment)

            # Normalize
            total = sum(new_fracs)
            fracs = [f / total for f in new_fracs]

        # Final EVs
        evs = []
        for i in range(n):
            denom = residents[i] + fracs[i] * n_players
            evs.append(base_value * multipliers[i] / denom)

        eq_ev = sum(evs) / n

        # Anti-herding: best choice is the one LEAST popular but still chosen
        # In practice: pick the option with lowest expected players but decent multiplier
        best_idx = -1
        best_ev = 0
        for i in range(n):
            if evs[i] > best_ev:
                best_ev = evs[i]
                best_idx = i

        return fracs, eq_ev, best_idx

    @staticmethod
    def anti_herd_choice(multipliers, residents, n_players,
                         base_value=10000, crowd_estimates=None):
        """Anti-herding strategy: pick the option most players will AVOID.

        crowd_estimates: your estimate of what fraction picks each option.
            If None, assumes crowd follows multiplier size (bigger = more popular).

        Returns: (best_choice_idx, expected_ev, analysis_dict)"""
        n = len(multipliers)

        if crowd_estimates is None:
            # Naive crowd: picks proportional to multiplier
            total_mult = sum(multipliers)
            crowd_estimates = [m / total_mult for m in multipliers]

        # Calculate EV for each choice
        evs = []
        analysis = []
        for i in range(n):
            crowd_frac = crowd_estimates[i]
            n_choosing = residents[i] + crowd_frac * n_players + 1  # +1 = you
            ev = base_value * multipliers[i] / n_choosing
            evs.append(ev)
            analysis.append({
                "choice": i,
                "multiplier": multipliers[i],
                "crowd_fraction": crowd_frac,
                "expected_competitors": n_choosing,
                "ev": ev,
            })

        best_idx = max(range(n), key=lambda i: evs[i])
        analysis.sort(key=lambda x: x["ev"], reverse=True)

        return best_idx, evs[best_idx], analysis

    @staticmethod
    def predicted_density(multipliers, residents, n_players,
                          base_value=10000, open_cost=50000,
                          nice_numbers=None):
        """CMU Physics calibrated player model — predicts selection density.

        CMU post-analysis (P3 R2→R4):
        - 50-60% of players follow Nash equilibrium
        - 5-15% concentrate on most popular Nash choices
        - 5-10% pick least popular Nash choices (over-correctors)
        - 10-15% pick randomly
        - 10-15% favor "nice numbers" (primes like 7, 17, 37, 73)

        nice_numbers: list of indices that are "psychologically attractive"
            (multipliers containing 7, 3, 37, 73, 17, etc.)

        Returns: (predicted_fracs, evs, should_open_second, best_choices)"""
        n = len(multipliers)
        if residents is None:
            residents = [0] * n

        # Step 1: Compute Nash equilibrium
        nash_fracs, nash_ev, _ = NashSolver.find_nash_equilibrium(
            multipliers, n_players, residents, base_value)

        # Step 2: Build predicted density from calibrated priors
        pred = [0.0] * n

        # 55% follow Nash
        for i in range(n):
            pred[i] += 0.55 * nash_fracs[i]

        # 10% concentrate on most popular Nash choice
        max_nash_idx = max(range(n), key=lambda i: nash_fracs[i])
        pred[max_nash_idx] += 0.10

        # 7% pick least popular Nash choice (over-correctors)
        min_nash_idx = min(range(n), key=lambda i: nash_fracs[i])
        pred[min_nash_idx] += 0.07

        # 13% pick randomly
        for i in range(n):
            pred[i] += 0.13 / n

        # 15% favor nice numbers
        if nice_numbers and len(nice_numbers) > 0:
            nice_share = 0.15 / len(nice_numbers)
            for idx in nice_numbers:
                if 0 <= idx < n:
                    pred[idx] += nice_share
        else:
            # Default: distribute to all
            for i in range(n):
                pred[i] += 0.15 / n

        # Normalize
        total = sum(pred)
        pred = [p / total for p in pred]

        # Step 3: Compute EVs under predicted density
        evs = []
        for i in range(n):
            denom = residents[i] + pred[i] * n_players
            if denom <= 0:
                denom = 0.01
            evs.append(base_value * multipliers[i] / denom)

        # Step 4: Should we open a second container?
        sorted_evs = sorted(evs, reverse=True)
        should_open_second = len(sorted_evs) >= 2 and sorted_evs[0] + sorted_evs[1] - open_cost > sorted_evs[0]

        # Step 5: Best 1 or 2 choices
        ranked = sorted(range(n), key=lambda i: evs[i], reverse=True)
        best_choices = ranked[:2] if should_open_second else ranked[:1]

        # Analysis output
        analysis = []
        for i in range(n):
            analysis.append({
                "choice": i,
                "multiplier": multipliers[i],
                "nash_frac": round(nash_fracs[i], 4),
                "predicted_frac": round(pred[i], 4),
                "predicted_players": round(pred[i] * n_players, 1),
                "ev": round(evs[i], 2),
            })
        analysis.sort(key=lambda x: x["ev"], reverse=True)

        print(f"\n{'='*60}")
        print(f"CMU PREDICTED DENSITY MODEL")
        print(f"{'='*60}")
        print(f"Nash EV (if all rational): {nash_ev:.0f}")
        print(f"Should open 2nd container (cost {open_cost}): {'YES' if should_open_second else 'NO'}")
        print(f"\nBest choice(s): {best_choices}")
        print(f"\n{'Idx':<4} {'Mult':<8} {'Nash%':<8} {'Pred%':<8} {'~Players':<10} {'EV':<10}")
        print("-" * 48)
        for a in analysis:
            print(f"{a['choice']:<4} {a['multiplier']:<8} {a['nash_frac']*100:<8.1f} "
                  f"{a['predicted_frac']*100:<8.1f} {a['predicted_players']:<10.1f} {a['ev']:<10.0f}")

        return pred, evs, should_open_second, best_choices


# ============================================================
# TYPE 3.5: K-LEVEL NASH (Stahl & Wilson 1995, Nagel 1995, Camerer 2003)
# Bounded-rationality opponent modeling — what wins manual rounds.
# ============================================================

class KLevelSolver:
    """K-level reasoning solver: opponents are NOT fully rational.

    Behavioral game theory (Camerer's Cognitive Hierarchy / Nagel beauty contest):
    - Level-0 (~10-20% of players): uniform random / heuristic / "nice numbers"
    - Level-1 (~30-40%): best-respond to assumption everyone is level-0
    - Level-2 (~20-30%): best-respond to level-1
    - Level-3+ (~10-20%): higher orders, including Nash limit

    Empirical poisson(lambda) over k with lambda ~ 1.5 fits Prosperity-style
    populations well (mix of pros + amateurs).

    The winning play is the OPTIMAL RESPONSE to this MIXED opponent distribution,
    not Nash. Nash assumes everyone is k=infty, which over-estimates competition
    on "obvious" choices and under-estimates "least-anticipated" choices.
    """

    @staticmethod
    def poisson_pmf(k, lam, k_max=6):
        """Truncated Poisson PMF over [0..k_max]. Camerer's CH model uses lam=1.5."""
        import math
        raw = [(lam ** i) * math.exp(-lam) / math.factorial(i) for i in range(k_max + 1)]
        s = sum(raw)
        return [r / s for r in raw]

    @staticmethod
    def k_level_density_container(multipliers, residents, n_players,
                                   base_value=10000, k_max=4, lam=1.5,
                                   level_0="uniform", nice_numbers=None):
        """Iteratively compute level-0..k_max play distributions for container/anti-herd.

        Each level k best-responds to the expected play of levels 0..k-1, weighted
        by Poisson(lam) prior over opponent k.

        Returns: (per_level_dist, weighted_dist, ev_per_level)
        """
        n = len(multipliers)
        weights = KLevelSolver.poisson_pmf(k_max, lam, k_max)

        per_level = []

        # --- Level 0: heuristic / nice-numbers / uniform ---
        if level_0 == "uniform":
            d0 = [1.0 / n] * n
        elif level_0 == "nice_numbers" and nice_numbers:
            d0 = [0.0] * n
            for idx in nice_numbers:
                if 0 <= idx < n:
                    d0[idx] = 1.0 / len(nice_numbers)
            if sum(d0) == 0:
                d0 = [1.0 / n] * n
        elif level_0 == "biggest_mult":
            top = max(range(n), key=lambda i: multipliers[i])
            d0 = [0.0] * n; d0[top] = 1.0
        else:
            d0 = [1.0 / n] * n
        per_level.append(d0)

        # --- Levels 1..k_max: best-respond to weighted lower levels ---
        for k in range(1, k_max + 1):
            # Aggregate expected opponent play from levels 0..k-1 (renormalized weights)
            lower_weights = weights[:k]
            wsum = sum(lower_weights)
            if wsum <= 0:
                opp_dist = per_level[k - 1]
            else:
                opp_dist = [0.0] * n
                for j in range(k):
                    w = lower_weights[j] / wsum
                    for i in range(n):
                        opp_dist[i] += w * per_level[j][i]

            # Compute EV under opp_dist for each option (1 = us; opp_dist scales other players)
            evs = []
            for i in range(n):
                competitors = residents[i] + opp_dist[i] * (n_players - 1) + 1
                evs.append(base_value * multipliers[i] / max(0.01, competitors))

            # Level-k plays the argmax (pure best response)
            best_i = max(range(n), key=lambda i: evs[i])
            d = [0.0] * n; d[best_i] = 1.0
            per_level.append(d)

        # Weighted average: what we expect the field to actually do
        weighted = [0.0] * n
        for k in range(k_max + 1):
            for i in range(n):
                weighted[i] += weights[k] * per_level[k][i]

        # Our EV of each choice under the weighted field
        ev_each = []
        for i in range(n):
            competitors = residents[i] + weighted[i] * (n_players - 1) + 1
            ev_each.append(base_value * multipliers[i] / max(0.01, competitors))

        return per_level, weighted, ev_each

    @staticmethod
    def best_choice_k_level(multipliers, residents, n_players,
                            base_value=10000, k_max=4, lam=1.5,
                            nice_numbers=None, open_cost=50000, allow_two=True,
                            verbose=True):
        """Pick the optimal 1 or 2 containers under k-level opponent assumption.

        Returns: (best_choice_indices, expected_total_value)
        """
        if residents is None:
            residents = [0] * len(multipliers)
        per_level, weighted, evs = KLevelSolver.k_level_density_container(
            multipliers, residents, n_players, base_value, k_max, lam,
            level_0="nice_numbers" if nice_numbers else "uniform",
            nice_numbers=nice_numbers,
        )
        ranked = sorted(range(len(multipliers)), key=lambda i: evs[i], reverse=True)
        ev1 = evs[ranked[0]]
        ev2 = evs[ranked[1]] if len(ranked) >= 2 else 0.0
        # Open second only if marginal exceeds cost
        open_second = allow_two and (ev2 - open_cost > 0)
        choices = ranked[:2] if open_second else ranked[:1]

        if verbose:
            print(f"\n{'='*60}\nK-LEVEL NASH (lambda={lam}, k_max={k_max})\n{'='*60}")
            print(f"  Poisson weights: {[round(w, 3) for w in KLevelSolver.poisson_pmf(k_max, lam, k_max)]}")
            print(f"  Weighted field play distribution:")
            for i, w in enumerate(weighted):
                marker = " <- pick" if i in choices else ""
                print(f"    opt {i+1}: mult={multipliers[i]:>4} pred={w*100:>5.1f}%  ev={evs[i]:>9,.0f}{marker}")
            print(f"  Open 2nd container? {'YES' if open_second else 'NO'} (cost {open_cost}, ev2 {ev2:,.0f})")
            print(f"  Picks: {[c+1 for c in choices]}  total expected: {sum(evs[c] for c in choices) - (open_cost if open_second else 0):,.0f}")
        return choices, sum(evs[c] for c in choices) - (open_cost if open_second else 0)

    @staticmethod
    def first_price_auction_bid(value, max_bid, k_max=4, lam=1.5, n_opponents=None,
                                level_0_bid_frac=0.5, verbose=True):
        """K-level bidding for first-price sealed-bid auction (Nagel 1995 generalized).

        Each level bids a fraction of value:
          L0 bids level_0_bid_frac * value (default 0.5 — naive midpoint)
          Lk bids epsilon-shading above max(L0..L(k-1)) bid

        Returns recommended bid as a fraction of value, scaled to max_bid.
        """
        # Level 0 bids level_0_bid_frac of value (capped to max_bid)
        bids_at_k = [min(level_0_bid_frac * value, max_bid)]
        eps = max(1.0, 0.01 * value)
        for k in range(1, k_max + 1):
            bids_at_k.append(min(bids_at_k[-1] + eps, max_bid))
        weights = KLevelSolver.poisson_pmf(k_max, lam, k_max)
        # Our optimal: bid epsilon above the WEIGHTED MEAN of opponent bids
        opp_mean = sum(weights[k] * bids_at_k[k] for k in range(k_max + 1))
        our_bid = min(opp_mean + eps, max_bid)
        if verbose:
            print(f"\nK-LEVEL FIRST-PRICE AUCTION (value={value}, max={max_bid})")
            print(f"  Opp bids by level:    {[round(b, 1) for b in bids_at_k]}")
            print(f"  Poisson weights:      {[round(w, 3) for w in weights]}")
            print(f"  Weighted opp mean:    {opp_mean:.1f}")
            print(f"  Our optimal bid:      {our_bid:.1f}  (shade {our_bid/value*100:.1f}% of value)")
        return our_bid


# ============================================================
# TYPE 4: PORTFOLIO / SENTIMENT OPTIMIZATION
# Convex optimization: maximize returns minus quadratic fees
# ============================================================

class PortfolioSolver:
    """Solve portfolio allocation problems with quadratic transaction fees.

    Common in R5: "Allocate capital across products. Fee = k * x^2"
    CMU Physics: Used historical P2 data to map sentiment -> optimal allocation.

    Problem: maximize sum(r_i * x_i) - sum(k_i * x_i^2)
    Subject to: sum(x_i) = budget, x_i >= 0
    """

    @staticmethod
    def optimize_quadratic(returns, fee_coeffs, budget, n_products=None):
        """Solve: max sum(r_i * x_i - k_i * x_i^2) subject to sum(x_i) = budget.

        returns: list of expected returns per unit
        fee_coeffs: list of quadratic fee coefficients (e.g., 120 for fee=120*x^2)
        budget: total to allocate

        Solution via Lagrangian:
            r_i - 2*k_i*x_i - lambda = 0
            x_i = (r_i - lambda) / (2*k_i)

        Returns: (allocations, total_profit)"""
        n = len(returns)

        # Binary search on lambda (Lagrange multiplier)
        lo, hi = -10000.0, 10000.0

        for _ in range(200):  # binary search iterations
            lam = (lo + hi) / 2.0
            allocs = []
            for i in range(n):
                if fee_coeffs[i] <= 0:
                    allocs.append(budget)  # no fee = all in
                else:
                    x = (returns[i] - lam) / (2.0 * fee_coeffs[i])
                    allocs.append(max(0.0, x))

            total = sum(allocs)
            if total > budget + 0.01:
                lo = lam
            elif total < budget - 0.01:
                hi = lam
            else:
                break

        # Final allocation
        allocs = []
        for i in range(n):
            if fee_coeffs[i] <= 0:
                allocs.append(0)
            else:
                x = (returns[i] - lam) / (2.0 * fee_coeffs[i])
                allocs.append(max(0.0, x))

        # Normalize to budget
        total = sum(allocs)
        if total > 0:
            allocs = [a * budget / total for a in allocs]

        # Calculate profit
        profit = sum(returns[i] * allocs[i] - fee_coeffs[i] * allocs[i]**2
                     for i in range(n))

        return allocs, profit

    @staticmethod
    def grid_search_allocation(returns, fee_coeffs, budget, step=100):
        """Brute-force grid search for small allocation problems.
        Tries all integer allocations in steps.

        Returns: (best_allocs, best_profit)"""
        n = len(returns)
        best_profit = float('-inf')
        best_allocs = [0] * n

        if n == 2:
            for x0 in range(0, budget + 1, step):
                x1 = budget - x0
                profit = (returns[0] * x0 - fee_coeffs[0] * x0**2 +
                          returns[1] * x1 - fee_coeffs[1] * x1**2)
                if profit > best_profit:
                    best_profit = profit
                    best_allocs = [x0, x1]
        elif n == 3:
            for x0 in range(0, budget + 1, step):
                for x1 in range(0, budget - x0 + 1, step):
                    x2 = budget - x0 - x1
                    profit = sum(returns[i] * [x0,x1,x2][i] - fee_coeffs[i] * [x0,x1,x2][i]**2
                                 for i in range(3))
                    if profit > best_profit:
                        best_profit = profit
                        best_allocs = [x0, x1, x2]
        else:
            # For n >= 4, use the analytic solver
            return PortfolioSolver.optimize_quadratic(returns, fee_coeffs, budget)

        return best_allocs, best_profit

    @staticmethod
    def sentiment_to_returns(sentiment_scores, base_return=1.0):
        """Convert sentiment scores (e.g., from news/data) to expected returns.
        CMU Physics technique for R5 manual trading.

        sentiment_scores: dict of {product: score} where score in [-1, 1]
        Returns: dict of {product: expected_return}"""
        returns = {}
        for product, score in sentiment_scores.items():
            # Linear mapping: sentiment of +1 = 2x base return, -1 = 0
            returns[product] = base_return * (1.0 + score)
        return returns


# ============================================================
# TYPE 5: BAYESIAN INFORMATION SOLVER
# Update beliefs from signals, compute posterior EV
# ============================================================

class BayesSolver:
    """Solve information-revelation problems via Bayes' theorem.

    Common in P1-P3: "You observe signal S. Should you buy/sell?"
    Key trap: base rate neglect (ignoring priors).
    """

    @staticmethod
    def posterior(prior, likelihood_if_true, likelihood_if_false):
        """Bayes update: P(H|evidence) given P(H), P(E|H), P(E|~H).
        Returns: P(H|evidence)"""
        p_evidence = likelihood_if_true * prior + likelihood_if_false * (1 - prior)
        if p_evidence == 0:
            return prior
        return likelihood_if_true * prior / p_evidence

    @staticmethod
    def multi_signal_update(prior, signals):
        """Update belief through a sequence of signals.

        signals: list of (likelihood_if_true, likelihood_if_false) tuples
        Returns: final posterior"""
        p = prior
        for lt, lf in signals:
            p = BayesSolver.posterior(p, lt, lf)
        return p

    @staticmethod
    def ev_with_info(prior, value_if_true, value_if_false,
                     signal_accuracy, signal_observed_positive):
        """Compute posterior EV after observing a signal.

        prior: P(good state) before signal
        value_if_true/false: payoff in each state
        signal_accuracy: P(positive signal | good state) = P(negative | bad state)
        signal_observed_positive: True if we saw positive signal

        Returns: (posterior_prob, posterior_ev)"""
        if signal_observed_positive:
            lt, lf = signal_accuracy, 1 - signal_accuracy
        else:
            lt, lf = 1 - signal_accuracy, signal_accuracy
        post = BayesSolver.posterior(prior, lt, lf)
        ev = post * value_if_true + (1 - post) * value_if_false
        return post, ev

    @staticmethod
    def value_of_information(prior, value_true, value_false,
                             signal_accuracy, action_cost=0):
        """How much is a signal worth? Compare EV(act with info) vs EV(act without).

        Returns: max additional EV from having the signal"""
        # Without info: pick best action based on prior
        ev_no_info = max(prior * value_true + (1 - prior) * value_false, 0)

        # With info: expected EV across both signal outcomes
        # P(positive signal) = accuracy * prior + (1-accuracy) * (1-prior)
        p_pos = signal_accuracy * prior + (1 - signal_accuracy) * (1 - prior)
        p_neg = 1 - p_pos

        _, ev_if_pos = BayesSolver.ev_with_info(
            prior, value_true, value_false, signal_accuracy, True)
        _, ev_if_neg = BayesSolver.ev_with_info(
            prior, value_true, value_false, signal_accuracy, False)

        # With info, we take the best action conditional on each signal
        ev_with_info = p_pos * max(ev_if_pos, 0) + p_neg * max(ev_if_neg, 0)

        return ev_with_info - ev_no_info - action_cost


# ============================================================
# TYPE 6: SEQUENTIAL / OPTIMAL STOPPING
# Dynamic programming for multi-round buy/wait decisions
# ============================================================

class SequentialSolver:
    """Solve sequential decision / optimal stopping problems.

    Common pattern: "You have N rounds. Each round you see a price and can
    buy or wait." Solved by backward induction.

    Also handles: secretary problem, prophet inequality variants.
    """

    @staticmethod
    def optimal_stopping_uniform(n_rounds, low=0, high=1000):
        """Optimal stopping with uniform iid draws.

        Each round you see a value drawn from Uniform(low, high).
        You can accept (game over) or reject (see next draw, but can't go back).
        Last round you must accept.

        Returns: list of thresholds — accept if value >= threshold[round]"""
        # Work backwards: at last round, accept anything
        # At round t: EV(wait) = expected value under optimal play from t+1
        # Accept if current_value >= EV(wait)
        thresholds = [0.0] * n_rounds
        thresholds[-1] = low  # must accept last round

        ev_future = (low + high) / 2.0  # EV of last round
        for t in range(n_rounds - 2, -1, -1):
            # Threshold = EV of continuing
            thresholds[t] = ev_future
            # EV of this round under optimal play:
            # P(accept) * E[value | value >= threshold] + P(reject) * ev_future
            if thresholds[t] >= high:
                ev_this = ev_future
            elif thresholds[t] <= low:
                ev_this = (low + high) / 2.0
            else:
                p_accept = (high - thresholds[t]) / (high - low)
                e_if_accept = (thresholds[t] + high) / 2.0
                ev_this = p_accept * e_if_accept + (1 - p_accept) * ev_future
            ev_future = ev_this

        return thresholds

    @staticmethod
    def dp_decision_tree(states, transitions, rewards, n_rounds):
        """Generic backward-induction DP for decision trees.

        states: list of state names
        transitions: dict of {(state, action): [(next_state, probability), ...]}
        rewards: dict of {(state, action): immediate_reward}
        n_rounds: number of rounds

        Returns: (values_per_round, best_actions_per_round)
            values_per_round[t][state] = optimal EV from state at round t
            best_actions_per_round[t][state] = best action at state/round"""

        # Terminal values: 0 (or override)
        values = [{s: 0.0 for s in states} for _ in range(n_rounds + 1)]
        best_actions = [{} for _ in range(n_rounds)]

        for t in range(n_rounds - 1, -1, -1):
            for s in states:
                best_v = float('-inf')
                best_a = None
                # Find all available actions from this state
                available = [a for (st, a) in transitions if st == s]
                for a in available:
                    immediate = rewards.get((s, a), 0)
                    future = sum(
                        p * values[t + 1].get(ns, 0)
                        for ns, p in transitions.get((s, a), [])
                    )
                    total = immediate + future
                    if total > best_v:
                        best_v = total
                        best_a = a
                values[t][s] = best_v if best_v > float('-inf') else 0
                if best_a is not None:
                    best_actions[t][s] = best_a

        return values, best_actions

    @staticmethod
    def multi_stage_ev(stages):
        """Compute EV for a multi-stage problem where you choose a path.

        stages: list of lists of (probability, reward, next_stage_options)
        Simpler interface for hand-entered decision trees.

        Returns: optimal EV and path"""
        # Evaluate from last stage backward
        if not stages:
            return 0, []

        def _eval(stage_idx):
            if stage_idx >= len(stages):
                return 0, []
            best_ev = float('-inf')
            best_choice = -1
            for i, (prob, reward, _) in enumerate(stages[stage_idx]):
                future_ev, _ = _eval(stage_idx + 1)
                ev = prob * (reward + future_ev)
                if ev > best_ev:
                    best_ev = ev
                    best_choice = i
            return best_ev, [best_choice]

        return _eval(0)


# ============================================================
# TYPE 7: OPTION / DERIVATIVE PRICING
# Binomial tree + Monte Carlo for option valuation
# ============================================================

class OptionSolver:
    """Price options/derivatives for manual trading challenges.

    Prosperity challenges typically give discrete outcomes, so binomial
    pricing or direct EV calculation is more appropriate than Black-Scholes.
    """

    @staticmethod
    def discrete_option_ev(outcomes, probabilities, strike, option_type="call"):
        """Price option from discrete outcome distribution.

        outcomes: list of possible underlying prices at expiry
        probabilities: list of probabilities (sum to 1)
        strike: strike price
        option_type: 'call' or 'put'

        Returns: option fair value"""
        ev = 0.0
        for price, prob in zip(outcomes, probabilities):
            if option_type == "call":
                payoff = max(0, price - strike)
            else:
                payoff = max(0, strike - price)
            ev += prob * payoff
        return ev

    @staticmethod
    def binomial_tree(S, K, u, d, p, n, option_type="call", american=False):
        """Binomial tree option pricing.

        S: current price, K: strike, u: up factor, d: down factor
        p: risk-neutral probability of up, n: number of steps
        option_type: 'call' or 'put'
        american: True for American (early exercise)

        Returns: option price"""
        # Build price tree at expiry
        prices = [S * (u ** j) * (d ** (n - j)) for j in range(n + 1)]

        # Option values at expiry
        if option_type == "call":
            values = [max(0, p_val - K) for p_val in prices]
        else:
            values = [max(0, K - p_val) for p_val in prices]

        # Work backwards
        for step in range(n - 1, -1, -1):
            new_values = []
            for j in range(step + 1):
                hold = p * values[j + 1] + (1 - p) * values[j]
                if american:
                    price_here = S * (u ** j) * (d ** (step - j))
                    if option_type == "call":
                        exercise = max(0, price_here - K)
                    else:
                        exercise = max(0, K - price_here)
                    new_values.append(max(hold, exercise))
                else:
                    new_values.append(hold)
            values = new_values

        return values[0]

    @staticmethod
    def bs_quick(S, K, sigma, T, option_type="call"):
        """Quick Black-Scholes using logistic approximation for N(x).
        No scipy needed. Accuracy ~0.5% for typical inputs."""
        if T <= 0 or sigma <= 0:
            if option_type == "call":
                return max(0, S - K)
            return max(0, K - S)

        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        # Logistic approximation: N(x) ≈ 1 / (1 + exp(-1.7 * x))
        def N(x):
            return 1.0 / (1.0 + math.exp(-1.7 * x))

        if option_type == "call":
            return S * N(d1) - K * N(d2)
        else:
            return K * N(-d2) - S * N(-d1)


# ============================================================
# INTERACTIVE CLI
# ============================================================

def interactive_mode():
    """Interactive solver for manual trading rounds."""
    print("=" * 60)
    print("MANUAL TRADING SOLVER")
    print("=" * 60)
    print("\nSolver types:")
    print("  1. Auction/Pricing (EV maximization)")
    print("  2. Conversion/Path (find best trade chain)")
    print("  3. Nash/Container (anti-herding equilibrium)")
    print("  4. Portfolio (optimal allocation with fees)")
    print("  5. Bayes/Information (posterior EV from signals)")
    print("  6. Sequential/Stopping (multi-round buy/wait)")
    print("  7. Option/Derivative (discrete or binomial pricing)")
    print("  q. Quit")

    while True:
        choice = input("\nSelect solver [1-7, q]: ").strip()

        if choice == 'q':
            break

        elif choice == '1':
            print("\n--- AUCTION SOLVER ---")
            value = float(input("  Item value to you: "))
            max_bid = float(input("  Maximum possible bid: "))
            prob_type = input("  Probability type [linear/quadratic/sqrt]: ").strip() or "linear"

            bid, ev = AuctionSolver.optimal_bid_analytic(value, max_bid, prob_type)
            print(f"\n  OPTIMAL BID: {bid}")
            print(f"  Expected value: {ev:.2f}")
            print(f"  Win probability: {bid/max_bid:.2%}")

        elif choice == '2':
            print("\n--- CONVERSION SOLVER ---")
            print("  Enter conversion rates as 'FROM TO RATE' (one per line, empty to stop):")
            rates = {}
            while True:
                line = input("    > ").strip()
                if not line:
                    break
                parts = line.split()
                if len(parts) == 3:
                    rates[(parts[0], parts[1])] = float(parts[2])

            start = input("  Starting item: ").strip()
            target = input("  Target item (empty=same as start): ").strip() or None

            value, path = ConversionSolver.find_best_path(rates, start, target)
            print(f"\n  BEST PATH: {' -> '.join(path)}")
            print(f"  Value multiplier: {value:.4f}")
            if value > 1.0:
                print(f"  ARBITRAGE: {(value-1)*100:.2f}% profit!")

        elif choice == '3':
            print("\n--- NASH / CONTAINER SOLVER ---")
            n = int(input("  Number of options: "))
            multipliers = []
            residents = []
            for i in range(n):
                m = float(input(f"  Option {i+1} multiplier: "))
                r = float(input(f"  Option {i+1} residents: "))
                multipliers.append(m)
                residents.append(r)

            n_players = int(input("  Estimated total players choosing: "))

            fracs, eq_ev, best = NashSolver.find_nash_equilibrium(
                multipliers, n_players, residents
            )

            print(f"\n  Nash Equilibrium Distribution:")
            for i in range(n):
                print(f"    Option {i+1}: {fracs[i]:.1%} of players "
                      f"(~{fracs[i]*n_players:.0f} people)")
            print(f"  Equilibrium EV: {eq_ev:.0f}")

            # Anti-herd recommendation
            best_ah, ev_ah, analysis = NashSolver.anti_herd_choice(
                multipliers, residents, n_players
            )
            print(f"\n  ANTI-HERD RECOMMENDATION: Option {best_ah + 1}")
            print(f"  Expected EV: {ev_ah:.0f}")
            print(f"\n  Full analysis:")
            for a in analysis:
                print(f"    Option {a['choice']+1}: mult={a['multiplier']}, "
                      f"crowd={a['crowd_fraction']:.1%}, "
                      f"competitors={a['expected_competitors']:.0f}, "
                      f"EV={a['ev']:.0f}")

            # CMU Physics predicted density model
            nice_input = input("\n  Nice number indices (comma-sep, e.g. 2,5,7) or empty: ").strip()
            nice_nums = [int(x.strip()) - 1 for x in nice_input.split(",") if x.strip()] if nice_input else None
            open_cost = float(input("  Cost to open 2nd container (0 if N/A): ").strip() or "0")

            pred, evs, should_open, best_choices = NashSolver.predicted_density(
                multipliers, residents, n_players,
                open_cost=open_cost if open_cost > 0 else 50000,
                nice_numbers=nice_nums
            )
            print(f"\n  >>> CMU MODEL BEST: Option(s) {[c+1 for c in best_choices]}")

        elif choice == '4':
            print("\n--- PORTFOLIO SOLVER ---")
            n = int(input("  Number of products: "))
            returns = []
            fees = []
            names = []
            for i in range(n):
                name = input(f"  Product {i+1} name: ").strip()
                r = float(input(f"  Expected return per unit: "))
                k = float(input(f"  Fee coefficient (fee = k * x^2): "))
                names.append(name)
                returns.append(r)
                fees.append(k)

            budget = float(input("  Total budget to allocate: "))

            allocs, profit = PortfolioSolver.optimize_quadratic(returns, fees, budget)

            print(f"\n  OPTIMAL ALLOCATION:")
            for i in range(n):
                print(f"    {names[i]}: {allocs[i]:.1f} units "
                      f"(return={returns[i]*allocs[i]:.0f}, "
                      f"fee={fees[i]*allocs[i]**2:.0f})")
            print(f"  Total profit: {profit:.0f}")

        elif choice == '5':
            print("\n--- BAYES / INFORMATION SOLVER ---")
            prior = float(input("  Prior P(good state): "))
            v_true = float(input("  Value if good state: "))
            v_false = float(input("  Value if bad state: "))
            accuracy = float(input("  Signal accuracy (0-1): "))
            observed = input("  Signal observed [positive/negative]: ").strip().lower()
            is_pos = observed.startswith('p')

            post, ev = BayesSolver.ev_with_info(
                prior, v_true, v_false, accuracy, is_pos)
            voi = BayesSolver.value_of_information(
                prior, v_true, v_false, accuracy)

            print(f"\n  Prior P(good): {prior:.2%}")
            print(f"  Posterior P(good|signal): {post:.2%}")
            print(f"  Posterior EV: {ev:.2f}")
            print(f"  Value of information: {voi:.2f}")
            print(f"  {'BUY' if ev > 0 else 'PASS'}")

        elif choice == '6':
            print("\n--- SEQUENTIAL / OPTIMAL STOPPING ---")
            n = int(input("  Number of rounds: "))
            lo = float(input("  Min possible value: "))
            hi = float(input("  Max possible value: "))

            thresholds = SequentialSolver.optimal_stopping_uniform(n, lo, hi)
            print(f"\n  Optimal thresholds (accept if value >=):")
            for i, t in enumerate(thresholds):
                label = " (must accept)" if i == n - 1 else ""
                print(f"    Round {i+1}: >= {t:.1f}{label}")

        elif choice == '7':
            print("\n--- OPTION / DERIVATIVE SOLVER ---")
            sub = input("  Mode [discrete/binomial/bs]: ").strip().lower()
            if sub.startswith('d'):
                n = int(input("  Number of outcomes: "))
                outcomes = []
                probs = []
                for i in range(n):
                    o = float(input(f"    Outcome {i+1} price: "))
                    p = float(input(f"    Outcome {i+1} probability: "))
                    outcomes.append(o)
                    probs.append(p)
                K = float(input("  Strike price: "))
                otype = input("  Type [call/put]: ").strip().lower() or "call"
                val = OptionSolver.discrete_option_ev(outcomes, probs, K, otype)
                print(f"\n  Option fair value: {val:.2f}")
            elif sub.startswith('b'):
                S = float(input("  Current price: "))
                K = float(input("  Strike: "))
                u = float(input("  Up factor (e.g. 1.1): "))
                d = float(input("  Down factor (e.g. 0.9): "))
                p = float(input("  Risk-neutral prob of up: "))
                n = int(input("  Number of steps: "))
                otype = input("  Type [call/put]: ").strip().lower() or "call"
                val = OptionSolver.binomial_tree(S, K, u, d, p, n, otype)
                print(f"\n  Option price: {val:.2f}")
            else:
                S = float(input("  Spot price: "))
                K = float(input("  Strike: "))
                sigma = float(input("  Volatility (e.g. 0.3): "))
                T = float(input("  Time to expiry (years): "))
                otype = input("  Type [call/put]: ").strip().lower() or "call"
                val = OptionSolver.bs_quick(S, K, sigma, T, otype)
                print(f"\n  BS price: {val:.2f}")

        else:
            print("  Invalid choice. Try 1-7 or q.")


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "test":
            # Run self-tests
            print("Running self-tests...")

            # Test auction
            bid, ev = AuctionSolver.optimal_bid_analytic(1000, 1000, "linear")
            print(f"  Auction (value=1000, max=1000, linear): bid={bid}, ev={ev:.0f}")
            assert 400 <= bid <= 600, f"Bid should be ~500, got {bid}"

            # Test conversion
            rates = {
                ("A", "B"): 2.0,
                ("B", "C"): 1.5,
                ("C", "A"): 0.5,
                ("A", "C"): 0.8,
            }
            val, path = ConversionSolver.find_best_path(rates, "A", "A")
            print(f"  Conversion: path={path}, value={val:.2f}")
            assert val >= 1.0, "Should find profitable loop"

            # Test Nash
            fracs, ev, best = NashSolver.find_nash_equilibrium(
                [3, 5, 8], 1000, [10, 20, 30]
            )
            print(f"  Nash: fracs={[f'{f:.2f}' for f in fracs]}, ev={ev:.0f}, best={best}")

            # Test portfolio
            allocs, profit = PortfolioSolver.optimize_quadratic(
                [10, 8, 12], [0.01, 0.02, 0.015], 1000
            )
            print(f"  Portfolio: allocs={[f'{a:.0f}' for a in allocs]}, profit={profit:.0f}")

            # Test anti-herd
            best_ah, ev_ah, _ = NashSolver.anti_herd_choice(
                [3, 5, 8], [10, 20, 30], 1000
            )
            print(f"  Anti-herd: best={best_ah+1}, ev={ev_ah:.0f}")

            # Test Bayes
            # Classic: 1% disease, 90% accurate test, positive result
            post = BayesSolver.posterior(0.01, 0.9, 0.1)
            print(f"  Bayes (1% prior, 90% test, positive): P(disease)={post:.3f}")
            assert 0.08 < post < 0.10, f"Should be ~8.3%, got {post:.3f}"

            # Test Bayes EV
            post2, ev2 = BayesSolver.ev_with_info(0.5, 1000, -500, 0.8, True)
            print(f"  Bayes EV (prior=0.5, acc=0.8, pos signal): post={post2:.2f}, ev={ev2:.0f}")
            assert post2 > 0.5, "Positive signal should increase posterior"

            # Test value of information
            voi = BayesSolver.value_of_information(0.5, 1000, -500, 0.8)
            print(f"  Value of information: {voi:.0f}")
            assert voi > 0, "Information should have positive value"

            # Test sequential stopping
            thresholds = SequentialSolver.optimal_stopping_uniform(5, 0, 1000)
            print(f"  Stopping thresholds (5 rounds, 0-1000): {[f'{t:.0f}' for t in thresholds]}")
            assert thresholds[0] > thresholds[-1], "Earlier rounds should have higher thresholds"
            assert thresholds[-1] == 0, "Last round threshold should be 0 (must accept)"

            # Test option pricing
            # Discrete: 50/50 chance of price being 110 or 90, call strike 100
            val = OptionSolver.discrete_option_ev([110, 90], [0.5, 0.5], 100, "call")
            print(f"  Discrete option (50/50 110/90, K=100 call): {val:.1f}")
            assert abs(val - 5.0) < 0.01, f"Should be 5.0, got {val}"

            # Binomial tree: 1-step, S=100, K=100, u=1.1, d=0.9, p=0.6
            bt_val = OptionSolver.binomial_tree(100, 100, 1.1, 0.9, 0.6, 1, "call")
            print(f"  Binomial (1-step): {bt_val:.2f}")
            assert bt_val > 0, "Option should have positive value"

            # BS quick
            bs_val = OptionSolver.bs_quick(100, 100, 0.3, 1.0, "call")
            print(f"  BS quick (ATM, vol=30%, T=1): {bs_val:.2f}")
            assert 10 < bs_val < 15, f"ATM call should be ~12, got {bs_val:.2f}"

            print("\nAll tests passed!")
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python manual_solver.py [test]")
            print("       python manual_solver.py  (interactive mode)")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
