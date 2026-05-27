"""
Pre-built CONFIG templates for every product archetype.
Used by rapid_deploy.py to generate trader.py in < 60 seconds.

Each template is calibrated from P1-P4 winning strategies.
Position limits are defaults — overridden by server listings at runtime.
"""

TEMPLATES = {
    # ── PEGGED: FV anchored at known round number ──
    # Use when: mid ≈ 1K/5K/10K, spread ≤ 20
    # Examples: RAINFOREST_RESIN (10K), EMERALDS (10K)
    "pegged": {
        "type": "pegged",
        "position_limit": 50,
        "take_width": 1,
        "make_offset": 3,       # tune: 2-7 depending on spread
        "make_size": 35,        # tune: set to ~70% of position_limit
        "inventory_threshold_1": 35,
        "inventory_threshold_2": 45,
    },

    # ── PEGGED AGGRESSIVE: tight spread pegged, maximize fills ──
    # Use when: pegged + spread ≤ 6
    "pegged_tight": {
        "type": "pegged",
        "position_limit": 50,
        "take_width": 2,
        "make_offset": 2,
        "make_size": 40,
        "inventory_threshold_1": 35,
        "inventory_threshold_2": 45,
    },

    # ── PEGGED WIDE: pegged but spread is 10-20 ──
    # Use when: pegged + spread 10-20
    "pegged_wide": {
        "type": "pegged",
        "position_limit": 50,
        "take_width": 1,
        "make_offset": 5,
        "make_size": 30,
        "inventory_threshold_1": 30,
        "inventory_threshold_2": 45,
    },

    # ── AR_OLIVIA: tight spread, mean-reverting, predictable ──
    # Use when: spread ≤ 4, AC1 < -0.1
    # Examples: KELP
    "ar_olivia": {
        "type": "ar_olivia",
        "position_limit": 50,
        "ar_order": 4,
        "ar_refit_every": 25,
        "ar_min_history": 50,
        "olivia_window": 5,
        "make_size": 25,
    },

    # ── AR_OLIVIA TRENDING: tight spread but trending ──
    # Use when: spread ≤ 4, AC1 > 0
    "ar_olivia_trending": {
        "type": "ar_olivia",
        "position_limit": 50,
        "ar_order": 5,
        "ar_refit_every": 25,
        "ar_min_history": 50,
        "olivia_window": 5,
        "make_size": 15,        # smaller: trending = riskier passive quotes
    },

    # ── OLIVIA_FOLLOW: random walk, follow the informed trader ──
    # Use when: spread ≤ 4, AC1 ≈ 0, informed trader detected
    # Examples: SQUID_INK
    "olivia_follow": {
        "type": "olivia_follow",
        "position_limit": 50,
        "target_position": 50,
        "vol_window": 50,
        "vol_mr_threshold": 2.0,
        "make_size": 10,
    },

    # ── WIDE_SPREAD: bot-driven, compression trading ──
    # Use when: spread > 8
    # Examples: TOMATOES
    "wide_spread": {
        "type": "wide_spread",
        "position_limit": 50,
        "take_ask_threshold": 6,    # tune: spread/2
        "bid_comp_threshold": 5,    # tune: spread/3
        "ema_alpha": 0.005,
        "ema_bid_skip_thr": 5.0,
        "ema_ask_skip_thr": 2.5,
        "make_size": 40,
        "inventory_threshold_1": 35,
        "inventory_threshold_2": 45,
    },

    # ── WIDE_SPREAD EXTREME: spread > 50 ──
    "wide_spread_extreme": {
        "type": "wide_spread",
        "position_limit": 50,
        "take_ask_threshold": 20,
        "bid_comp_threshold": 15,
        "ema_alpha": 0.003,
        "ema_bid_skip_thr": 10.0,
        "ema_ask_skip_thr": 5.0,
        "make_size": 30,
        "inventory_threshold_1": 30,
        "inventory_threshold_2": 45,
    },

    # ── BASKET_ARB: ETF/basket vs components ──
    # Use when: name contains BASKET/ETF/INDEX/BUNDLE
    # CRITICAL: premium_mean MUST be set from data analysis
    "basket_arb": {
        "type": "basket_arb",
        "position_limit": 50,
        "components": {},           # MUST FILL: {"COMP_A": weight, "COMP_B": weight}
        "premium_mean": 0.0,       # MUST FILL from data
        "premium_std": 15.0,       # auto-estimated if left at 15
        "entry_z": 2.0,
        "exit_z": 0.3,
        "trade_size": 50,
        "hedge_ratio": 0.5,
    },

    # ── CONVERSION_ARB: local vs foreign market ──
    # Use when: ConversionObservation present for this product
    # Examples: ROSES
    "conversion_arb": {
        "type": "conversion_arb",
        "position_limit": 50,
        "max_conversions": 10,
        "order_size": 30,
        "max_short": 30,
    },

    # ── OPTIONS: vol surface trading ──
    # Use when: name contains VOUCHER/COUPON/CALL/PUT/OPTION
    "options": {
        "type": "options",
        "position_limit": 200,
        "underlying": "",           # MUST FILL: base product name
        "strike": 10000,            # MUST FILL from product name
        "start_T_days": 250,
        "timestamps_per_day": 10000,
        "iv_window": 150,
        "entry_z": 1.0,
        "trade_size": 5,
        "delta_hedge": False,
        "underlying_limit": 400,
    },

    # ── GENERIC_MM: unknown product, adaptive ──
    # Fallback when nothing else matches
    "generic_mm": {
        "type": "generic_mm",
        "position_limit": 50,
        "make_size": 15,
    },
}


def get_template(archetype, **overrides):
    """Get a config template with optional overrides.

    Usage:
        cfg = get_template("pegged", fair_value=10000, make_offset=5, position_limit=80)
    """
    if archetype not in TEMPLATES:
        archetype = "generic_mm"
    cfg = dict(TEMPLATES[archetype])
    cfg.update(overrides)
    return cfg


def scale_to_limit(cfg, position_limit):
    """Scale size parameters proportionally to position limit."""
    old_limit = cfg.get("position_limit", 50)
    if old_limit == position_limit or old_limit == 0:
        cfg["position_limit"] = position_limit
        return cfg

    ratio = position_limit / old_limit
    cfg["position_limit"] = position_limit

    # Scale sizes
    for key in ["make_size", "order_size", "trade_size", "target_position"]:
        if key in cfg:
            cfg[key] = max(1, int(cfg[key] * ratio))

    # Scale inventory thresholds
    for key in ["inventory_threshold_1", "inventory_threshold_2", "max_short"]:
        if key in cfg:
            cfg[key] = max(1, int(cfg[key] * ratio))

    return cfg
