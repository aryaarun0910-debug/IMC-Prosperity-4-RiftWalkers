"""
Pattern Library -Compressed Cross-Year Intuition for IMC Prosperity

This is the structural prior. Every product IMC has ever introduced across P1-P4,
mapped to: what signals identify it, what the trick is, what kills you, and what
CONFIG to deploy.

When a new product appears in R1, this library pattern-matches it against every
known archetype and returns ranked hypotheses with confidence scores.

This is not guessing. This is:
  - 4 years of competition data
  - 15+ top-team writeups
  - Every bot behavior ever documented
  - Every failure mode ever reported
  compressed into executable pattern matching.
"""

import math
import re


# ============================================================
# THE PATTERN LIBRARY
# Every archetype IMC has ever used, with detection signals
# ============================================================

ARCHETYPES = [
    # ── PEGGED AT 10K ──
    # Appears: P1 (PEARLS), P2 (AMETHYSTS), P3 (RAINFOREST_RESIN)
    # Frequency: 100% of years, always R1
    {
        "id": "PEGGED_10K",
        "name": "Pegged at 10,000",
        "examples": ["PEARLS", "AMETHYSTS", "RAINFOREST_RESIN"],
        "signals": {
            "mid_near_round": True,       # mid within 5 of 10000
            "round_number": 10000,
            "avg_spread": (2, 20),         # typical range
            "price_range_pct": (0, 0.5),   # very tight range
            "zero_return_pct": 0.7,        # >70% of ticks have zero mid change
        },
        "trick": "FV is EXACTLY 10000. Hardcode it. Never drift. Take anything "
                 "mispriced vs 10000. Make around it with full position limit.",
        "kills_you": "Trying to estimate FV dynamically. It's 10000. Period.",
        "pnl_range": "30-40K per round",
        "config": {
            "type": "pegged",
            "fair_value": 10000,
            "take_width": 1,
            "make_offset": 3,
            "make_size": 35,
        },
        "confidence_weight": 10,  # highest: this pattern is rock-solid
    },

    # ── PEGGED AT OTHER ROUND NUMBERS ──
    # Less common but possible (1K, 5K)
    {
        "id": "PEGGED_ROUND",
        "name": "Pegged at round number (not 10K)",
        "examples": [],
        "signals": {
            "mid_near_round": True,
            "round_number": None,  # any of 1000, 5000
            "avg_spread": (2, 20),
            "price_range_pct": (0, 1.0),
        },
        "trick": "Same as 10K peg but confirm FV from data. Watch for drift.",
        "kills_you": "Assuming FV is exactly the round number if it actually drifts.",
        "pnl_range": "20-40K per round",
        "config": {
            "type": "pegged",
            "take_width": 1,
            "make_offset": 3,
            "make_size": 35,
        },
        "confidence_weight": 8,
    },

    # ── SLOW RANDOM WALK (KELP-LIKE) ──
    # Appears: P1 (BANANAS), P2 (STARFRUIT), P3 (KELP)
    # Frequency: 100% of years, always R1
    {
        "id": "SLOW_WALK",
        "name": "Slow random walk, tight spread, mean-reverting",
        "examples": ["BANANAS", "STARFRUIT", "KELP"],
        "signals": {
            "avg_spread": (1, 4),
            "ac1": (-0.6, -0.1),       # mean-reverting
            "vol_pct": (0, 0.1),       # low relative vol
            "hurst": (0.3, 0.5),       # mean-reverting Hurst
        },
        "trick": "Wall_mid as FV (use the PRIMARY MM bot's mid, not market mid). "
                 "AR prediction is marginal. The real edge is Olivia detection: "
                 "find the informed trader and copy. Frankfurt detected Olivia "
                 "BEFORE R5 by looking for large lots that predict future direction.",
        "kills_you": "Overfitting AR model. Trying to predict direction. You can't "
                     "predict a random walk -just make the market and follow the "
                     "informed trader.",
        "pnl_range": "5-8K per round (MM only), 20-40K with bot signal",
        "config": {
            "type": "ar_olivia",
            "ar_order": 4,
            "ar_refit_every": 25,
            "ar_min_history": 50,
            "olivia_window": 5,
            "make_size": 25,
        },
        "confidence_weight": 9,
    },

    # ── VOLATILE SPIKE PRODUCT (SQUID_INK-LIKE) ──
    # Appears: P3 (SQUID_INK), possibly P1 (BANANAS had some of this)
    # Frequency: ~50% of years in R1
    {
        "id": "VOLATILE_SPIKE",
        "name": "Volatile, spiky, wide range, follows informed trader",
        "examples": ["SQUID_INK"],
        "signals": {
            "avg_spread": (1, 5),
            "price_range_pct": (5, 50),    # large range relative to price
            "vol_pct": (0.05, 0.5),        # high relative vol
            "ac1": (-0.3, 0.1),            # weakly MR or neutral
        },
        "trick": "DON'T predict. FOLLOW. Find the informed trader (Olivia-like). "
                 "CMU: rolling std spike detection, enter opposite = mean reversion. "
                 "Frankfurt: copy Olivia's direction. Only 10% allocation due to risk. "
                 "Vol mean-reversion works: when vol spikes above 2x normal, fade it.",
        "kills_you": "Full position sizing on a prediction. One wrong spike = -40K. "
                     "Also: trying to MM this product like KELP. The dynamics are "
                     "completely different.",
        "pnl_range": "8-20K per round (high variance)",
        "config": {
            "type": "olivia_follow",
            "target_position": 50,
            "vol_window": 50,
            "vol_mr_threshold": 2.0,
            "make_size": 10,
        },
        "confidence_weight": 7,
    },

    # ── BASKET/ETF ──
    # Appears: P2 (GIFT_BASKET), P3 (PICNIC_BASKET 1&2)
    # Frequency: 100% from R2 onward
    {
        "id": "BASKET",
        "name": "Basket/ETF -price = weighted sum of components + premium",
        "examples": ["GIFT_BASKET", "PICNIC_BASKET1", "PICNIC_BASKET2"],
        "signals": {
            "name_contains": ["BASKET", "ETF", "INDEX", "BUNDLE"],
            "price_high": True,            # baskets are usually expensive
            "correlated_products": True,   # multiple correlated products exist
        },
        "trick": "HARDCODE the mean premium from CSV analysis. Rolling std for entry. "
                 "P2: GIFT_BASKET = 4*CHOCOLATE + 6*STRAWBERRY + 1*ROSE + 370. "
                 "P3: Basket1 = 6*CROISSANTS + 3*JAMS + 1*DJEMBE. "
                 "CMU used entry_z=20 (very wide), 100% position limit. "
                 "Stanford insight: trade BASKET ONLY, not components -basket "
                 "prices LED component prices. Don't bother hedging both legs.",
        "kills_you": "Not hardcoding the premium mean. Trying to estimate it online "
                     "from the first 100 ticks = garbage estimate = bad trades. "
                     "Also: over-hedging. Frankfurt used 50% hedge ratio for a reason.",
        "pnl_range": "40-60K per round",
        "config": {
            "type": "basket_arb",
            "components": {},
            "premium_mean": 0,      # MUST SET FROM DATA
            "premium_std": 15.0,
            "entry_z": 2.0,
            "exit_z": 0.3,
            "trade_size": 50,
            "hedge_ratio": 0.5,
        },
        "confidence_weight": 9,
    },

    # ── OPTIONS ──
    # Appears: P2 (COCONUT_COUPON), P3 (VOLCANIC_ROCK_VOUCHER with 5 strikes)
    # Frequency: 100% from R3 onward
    {
        "id": "OPTIONS",
        "name": "Options on underlying -IV smile trading",
        "examples": ["COCONUT_COUPON", "VOLCANIC_ROCK_VOUCHER_9500",
                      "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10500"],
        "signals": {
            "name_contains": ["VOUCHER", "COUPON", "CALL", "PUT", "OPTION"],
            "underlying_exists": True,
            "price_much_lower_than_underlying": True,
        },
        "trick": "Fit IV parabola across strikes. Trade deviations from smile. "
                 "CMU CRITICAL insight: DON'T HEDGE. Hedging costs 40K/day in spreads, "
                 "unhedged delta risk is only 16K worst-case. Net: +24K/day from "
                 "NOT hedging. Rolling window mean IV >> quadratic fit (CMU went "
                 "80K -> 250K daily by switching).",
        "kills_you": "Delta hedging (costs more than it saves). Overfitting the IV smile. "
                     "AWS Lambda memory resets wiping your IV history mid-round.",
        "pnl_range": "75-250K per round",
        "config": {
            "type": "options",
            "underlying": "",       # MUST SET
            "strike": 10000,        # MUST SET
            "start_T_days": 250,
            "timestamps_per_day": 10000,
            "iv_window": 150,
            "entry_z": 1.0,
            "trade_size": 5,
            "delta_hedge": False,   # DON'T HEDGE
            "underlying_limit": 400,
        },
        "confidence_weight": 8,
    },

    # ── CONVERSION/TRANSPORT ──
    # Appears: P2 (ORCHIDS), P3 (MAGNIFICENT_MACARONS)
    # Frequency: 100% from R4 onward
    {
        "id": "CONVERSION",
        "name": "Cross-market conversion arbitrage",
        "examples": ["ORCHIDS", "MAGNIFICENT_MACARONS"],
        "signals": {
            "has_conversion_observation": True,
            "transport_fees": True,
            "tariffs": True,
        },
        "trick": "This is the HIGHEST PnL product type (100-450K). CMU: 447K on macarons. "
                 "Formula: profit = local_sell - foreign_ask - transport - import_tariff. "
                 "Frankfurt: sell at int(foreignBid + 0.5) for max fills. "
                 "CMU: order size 10->30 DOUBLED volume. Accept short position (up to "
                 "max_short=30) to maximize conversion opportunities. "
                 "CHECK IF IMPORT TARIFF IS NEGATIVE -P3 macarons had negative "
                 "tariffs = free money.",
        "kills_you": "Conservative order sizing (10 instead of 30). Not going short. "
                     "Missing negative tariffs. Not checking ALL fee components.",
        "pnl_range": "100-450K per round (HIGHEST)",
        "config": {
            "type": "conversion_arb",
            "max_conversions": 10,
            "order_size": 30,
            "max_short": 30,
        },
        "confidence_weight": 9,
    },

    # ── WIDE SPREAD BOT-DRIVEN ──
    # Less common as primary archetype, but appears
    {
        "id": "WIDE_SPREAD",
        "name": "Wide spread, bot-driven, compression trading",
        "examples": ["TOMATOES"],
        "signals": {
            "avg_spread": (8, 200),
            "bot_driven": True,
        },
        "trick": "EMA tracking + compression detection. Penny-jump when spread "
                 "compresses. Use bot activity as signal for FV direction.",
        "kills_you": "Market-making with fixed FV when the spread is 50+. "
                     "Penny-jumping into no-man's land on wide spreads.",
        "pnl_range": "5-15K per round",
        "config": {
            "type": "wide_spread",
            "take_ask_threshold": 6,
            "bid_comp_threshold": 5,
            "ema_alpha": 0.005,
            "make_size": 40,
        },
        "confidence_weight": 6,
    },
]


# ============================================================
# PATTERN MATCHER
# Given product analysis, return ranked archetype hypotheses
# ============================================================

def match_product(product_name, analysis, has_conversion_obs=False,
                  other_products=None):
    """Match a product against all known archetypes.

    Returns list of (archetype, confidence, reasons) sorted by confidence desc.
    """
    matches = []

    for arch in ARCHETYPES:
        score = 0
        reasons = []
        signals = arch["signals"]

        # Name-based matching
        if "name_contains" in signals:
            for kw in signals["name_contains"]:
                if kw in product_name.upper():
                    score += 50
                    reasons.append(f"Name contains '{kw}'")
                    break

        # Conversion observation
        if signals.get("has_conversion_observation") and has_conversion_obs:
            score += 50
            reasons.append("ConversionObservation present")

        if analysis is None:
            if score > 0:
                matches.append((arch, score, reasons))
            continue

        a = analysis

        # Round number detection
        if signals.get("mid_near_round"):
            target = signals.get("round_number")
            if target and a.get("nearest_round") == target:
                score += 30
                reasons.append(f"Mid near {target}")
            elif target is None and a.get("nearest_round"):
                score += 20
                reasons.append(f"Mid near round number {a['nearest_round']}")

        # Spread range
        if "avg_spread" in signals:
            lo, hi = signals["avg_spread"]
            spread = a.get("avg_spread", 999)
            if lo <= spread <= hi:
                score += 15
                reasons.append(f"Spread {spread:.1f} in [{lo}, {hi}]")
            elif spread < lo:
                score -= 5
            elif spread > hi:
                score -= 10

        # Autocorrelation
        if "ac1" in signals:
            lo, hi = signals["ac1"]
            ac1 = a.get("ac1", 0)
            if lo <= ac1 <= hi:
                score += 10
                reasons.append(f"AC1 {ac1:.3f} in [{lo}, {hi}]")

        # Volatility (relative to price)
        if "vol_pct" in signals:
            lo, hi = signals["vol_pct"]
            vol_pct = a.get("vol", 0) / a.get("avg_mid", 1) * 100
            if lo <= vol_pct <= hi:
                score += 8
                reasons.append(f"Vol% {vol_pct:.2f} in [{lo}, {hi}]")

        # Price range (relative)
        if "price_range_pct" in signals:
            lo, hi = signals["price_range_pct"]
            range_pct = a.get("price_range", 0) / a.get("avg_mid", 1) * 100
            if lo <= range_pct <= hi:
                score += 8
                reasons.append(f"Range% {range_pct:.1f} in [{lo}, {hi}]")

        # Hurst
        if "hurst" in signals:
            lo, hi = signals["hurst"]
            hurst = a.get("hurst", 0.5)
            if lo <= hurst <= hi:
                score += 8
                reasons.append(f"Hurst {hurst:.2f} in [{lo}, {hi}]")

        # Cross-product correlation (for baskets)
        if signals.get("correlated_products") and other_products:
            if len(other_products) >= 2:
                score += 10
                reasons.append(f"{len(other_products)} other products exist")

        # Underlying exists (for options)
        if signals.get("underlying_exists") and other_products:
            name_parts = re.split(r'_(?:VOUCHER|COUPON|CALL|PUT|OPTION)_', product_name)
            if len(name_parts) > 1 and name_parts[0] in (other_products or []):
                score += 25
                reasons.append(f"Underlying '{name_parts[0]}' found")

        # Weight by archetype confidence
        score = score * arch["confidence_weight"] / 10.0

        if score > 0:
            matches.append((arch, score, reasons))

    matches.sort(key=lambda x: -x[1])
    return matches


def diagnose_product(product_name, analysis, has_conversion_obs=False,
                     other_products=None):
    """Full diagnosis: match + trick + warnings.

    Returns a formatted string ready for human consumption.
    """
    matches = match_product(product_name, analysis, has_conversion_obs,
                            other_products)

    lines = [f"\n{'='*60}"]
    lines.append(f"PATTERN DIAGNOSIS: {product_name}")
    lines.append(f"{'='*60}")

    if not matches:
        lines.append("  No known archetype matched. Using generic_mm.")
        lines.append("  This is a truly novel product -iterate fast on server logs.")
        return "\n".join(lines), "generic_mm", {}

    # Top match
    best = matches[0]
    arch, score, reasons = best

    lines.append(f"\n  BEST MATCH: {arch['name']} (confidence: {score:.0f})")
    lines.append(f"  Historical examples: {', '.join(arch['examples']) or 'None known'}")
    lines.append(f"  Signals matched:")
    for r in reasons:
        lines.append(f"    + {r}")

    lines.append(f"\n  THE TRICK:")
    for trick_line in arch["trick"].split(". "):
        if trick_line.strip():
            lines.append(f"    > {trick_line.strip()}.")

    lines.append(f"\n  WHAT KILLS YOU:")
    for kill_line in arch["kills_you"].split(". "):
        if kill_line.strip():
            lines.append(f"    ! {kill_line.strip()}.")

    lines.append(f"\n  EXPECTED PnL: {arch['pnl_range']}")

    # Alternative hypotheses
    if len(matches) > 1:
        lines.append(f"\n  ALTERNATIVE HYPOTHESES:")
        for alt_arch, alt_score, alt_reasons in matches[1:3]:
            lines.append(f"    - {alt_arch['name']} (confidence: {alt_score:.0f}): "
                         f"{', '.join(alt_reasons[:2])}")

    # Action items
    lines.append(f"\n  ACTION ITEMS:")
    cfg = dict(arch["config"])
    if arch["id"] == "BASKET":
        lines.append("    1. IMMEDIATELY compute component weights from price data")
        lines.append("    2. HARDCODE premium_mean from CSV analysis")
        lines.append("    3. Set position_limit from server listings")
    elif arch["id"] == "OPTIONS":
        lines.append("    1. Identify underlying product name")
        lines.append("    2. Extract strike from product name")
        lines.append("    3. DO NOT enable delta hedging (costs > benefit)")
    elif arch["id"] == "CONVERSION":
        lines.append("    1. CHECK IF IMPORT TARIFF IS NEGATIVE (= free money)")
        lines.append("    2. Set order_size to 30 (not 10)")
        lines.append("    3. Accept short position up to max_short=30")
    elif arch["id"] in ("PEGGED_10K", "PEGGED_ROUND"):
        if analysis and analysis.get("nearest_round"):
            cfg["fair_value"] = analysis["nearest_round"]
            lines.append(f"    1. FV set to {analysis['nearest_round']}")
        lines.append("    2. Set make_size to ~70% of position_limit")
        lines.append("    3. Increase make_offset if toxicity > 0.55 in first log")
    elif arch["id"] in ("SLOW_WALK", "VOLATILE_SPIKE"):
        lines.append("    1. Watch for informed trader (large lots predicting future mid)")
        lines.append("    2. If detected, switch to bot-following mode")
        lines.append("    3. Check first log for toxicity -if high, widen spread")

    return "\n".join(lines), arch["config"]["type"], cfg


def diagnose_all(analyses, product_names, conversion_products=None):
    """Diagnose all products and print full report."""
    conversion_products = conversion_products or set()

    all_configs = {}
    full_report = []

    for product in sorted(product_names):
        analysis = analyses.get(product)
        has_conv = product in conversion_products
        other = [p for p in product_names if p != product]

        report, strategy_type, cfg = diagnose_product(
            product, analysis, has_conv, other)

        full_report.append(report)
        all_configs[product] = cfg

    return "\n".join(full_report), all_configs


# ── P4 ROUND PREDICTIONS ──
# Based on cross-year patterns

P4_PREDICTIONS = {
    "R1": {
        "expected_products": [
            "1 pegged product near 10K (100% certain, every year)",
            "1-2 dynamic products: slow walk + possibly volatile (90% certain)",
        ],
        "strategy": "Deploy pegged + ar_olivia + olivia_follow templates. "
                    "Submit within 10 minutes. Iterate on spread/size params.",
        "expected_pnl": "40-60K total if executed well",
    },
    "R2": {
        "expected_products": [
            "R1 products continue",
            "1 basket/ETF product (90% certain)",
            "2-4 component products (90% certain)",
        ],
        "strategy": "Identify basket composition from price data immediately. "
                    "HARDCODE premium mean. Trade basket only (not components) "
                    "per Stanford Cardinal insight.",
        "expected_pnl": "80-120K total",
    },
    "R3": {
        "expected_products": [
            "R1+R2 products continue",
            "1 options product with 1-5 strikes (85% certain)",
            "1 underlying product",
        ],
        "strategy": "IV smile fitting. DO NOT HEDGE. Rolling window mean IV.",
        "expected_pnl": "150-350K total",
    },
    "R4": {
        "expected_products": [
            "All prior products",
            "1 conversion/transport product (85% certain)",
        ],
        "strategy": "Maximize conversion volume. Order size 30. Go short. "
                    "Check for negative tariffs.",
        "expected_pnl": "250-800K total",
    },
    "R5": {
        "expected_products": [
            "All prior products",
            "Bot identities revealed in trade data",
        ],
        "strategy": "Find Olivia (or equivalent). YOLO max position on her signal. "
                    "CMU: 120K range * max position = massive PnL.",
        "expected_pnl": "500K-1.4M total",
    },
}


if __name__ == "__main__":
    # Quick test with synthetic data
    test_analysis = {
        "avg_mid": 10000.5,
        "avg_spread": 6.0,
        "nearest_round": 10000,
        "price_range": 7.0,
        "vol": 2.0,
        "ac1": -0.5,
        "hurst": 0.5,
        "drift": 0,
    }

    report, stype, cfg = diagnose_product("JUNGLE_RESIN", test_analysis)
    print(report)
    print(f"\nStrategy: {stype}")
    print(f"Config: {cfg}")

    print("\n\n" + "="*60)
    print("P4 ROUND PREDICTIONS")
    print("="*60)
    for rnd, pred in P4_PREDICTIONS.items():
        print(f"\n{rnd}:")
        for p in pred["expected_products"]:
            print(f"  - {p}")
        print(f"  Strategy: {pred['strategy']}")
        print(f"  Expected PnL: {pred['expected_pnl']}")
