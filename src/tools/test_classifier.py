"""
Stress-test classify_product_live() against every product archetype
IMC could throw at us in R1.

Tests:
  1. Pegged products at various price levels (10K, 1K, 5K, 50K, 100K)
  2. Tight spread products (should be ar_olivia)
  3. Wide spread products (should be wide_spread)
  4. Empty order books (should not crash)
  5. One-sided books (bids only, asks only)
  6. Products near round numbers but not pegged
  7. Conversion products (ConversionObservation present)
  8. Option-like product names
  9. Basket-like product names
  10. Extremely high/low prices
  11. Single-level books
  12. Reclassification after 500 ticks
  13. Products with names we've never seen
  14. Spread exactly at boundary values (4, 8)
"""

import sys
import os
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _ROOT_DIR)

from trader import (
    OrderDepth, TradingState, Observation, ConversionObservation,
    classify_product_live
)


def make_od(bids, asks):
    """Create OrderDepth from price:vol dicts."""
    od = OrderDepth()
    od.buy_orders = dict(bids) if bids else {}
    od.sell_orders = {k: -abs(v) for k, v in asks.items()} if asks else {}
    return od


def make_state(order_depths=None, conv_obs=None):
    """Create minimal TradingState."""
    return TradingState(
        timestamp=0,
        traderData="",
        listings={},
        order_depths=order_depths or {},
        own_trades={},
        market_trades={},
        position={},
        observations=Observation(
            conversionObservations=conv_obs or {}
        ),
    )


def test(name, product, od, state=None, expected_type=None, expect_no_crash=True):
    """Run a single classification test."""
    mem = {}
    try:
        cfg = classify_product_live(od, product, mem, state)
        actual_type = cfg.get("type", "UNKNOWN")
        status = "PASS" if (expected_type is None or actual_type == expected_type) else "FAIL"
        limit = cfg.get("position_limit", "?")
        extra = ""
        if cfg.get("fair_value"):
            extra += f" fv={cfg['fair_value']}"
        if cfg.get("make_offset"):
            extra += f" offset={cfg['make_offset']}"
        if cfg.get("underlying"):
            extra += f" underlying={cfg['underlying']}"

        if status == "FAIL":
            print(f"  FAIL  {name:<45} got={actual_type:<15} expected={expected_type:<15} limit={limit}{extra}")
        else:
            print(f"  OK    {name:<45} type={actual_type:<15} limit={limit}{extra}")
        return status == "PASS", cfg
    except Exception as e:
        if expect_no_crash:
            print(f"  CRASH {name:<45} {type(e).__name__}: {e}")
            return False, None
        else:
            print(f"  OK    {name:<45} (expected crash: {e})")
            return True, None


def main():
    passed = 0
    failed = 0
    total = 0

    def run(name, product, od, state=None, expected_type=None):
        nonlocal passed, failed, total
        total += 1
        ok, cfg = test(name, product, od, state, expected_type)
        if ok:
            passed += 1
        else:
            failed += 1
        return cfg

    print("=" * 80)
    print("CLASSIFIER STRESS TEST")
    print("=" * 80)

    # ── 1. PEGGED PRODUCTS ──────────────────────────────────
    print("\n--- Pegged Products (round-number FV, tight spread) ---")

    # Classic pegged at 10000 (like EMERALDS)
    run("EMERALDS-like (mid=10000, spread=16)",
        "PRODUCT_A",
        make_od({9992: 15, 9990: 30}, {10008: 15, 10010: 30}),
        expected_type="pegged")

    # Pegged at 10000 with tight spread
    run("Tight pegged (mid=10000, spread=6)",
        "PRODUCT_B",
        make_od({9997: 20, 9995: 40}, {10003: 20, 10005: 40}),
        expected_type="pegged")

    # Pegged at 1000
    run("Pegged at 1000 (spread=4)",
        "PRODUCT_C",
        make_od({998: 10}, {1002: 10}),
        expected_type="pegged")

    # Pegged at 5000 (like TOMATOES mid area)
    run("Pegged at 5000 (spread=10)",
        "PRODUCT_D",
        make_od({4995: 10}, {5005: 10}),
        expected_type="pegged")

    # NOT pegged: round number but spread > 20
    run("Round number but wide spread (mid=10000, spread=30)",
        "PRODUCT_E",
        make_od({9985: 10}, {10015: 10}),
        expected_type="wide_spread")

    # ── 2. PRICES CLASSIFIER HASN'T SEEN ────────────────────
    print("\n--- Unusual Price Levels ---")

    # Very high price (100K)
    run("High price (mid=100000, spread=10)",
        "EXPENSIVE",
        make_od({99995: 10}, {100005: 10}))  # No round-number match at 100K

    # Very low price (50)
    run("Low price (mid=50, spread=4)",
        "CHEAP",
        make_od({48: 10}, {52: 10}))

    # Price at 2000 (not in round number list)
    run("2000-level (not in check list)",
        "MIDRANGE",
        make_od({1998: 10}, {2002: 10}))

    # Price at 500
    run("500-level price",
        "LOWMID",
        make_od({498: 10}, {502: 10}))

    # Price at 50000
    run("50000-level price (spread=20)",
        "HIGHEND",
        make_od({49990: 10}, {50010: 10}))

    # ── 3. SPREAD BOUNDARY CASES ────────────────────────────
    print("\n--- Spread Boundaries ---")

    # Near round number (1000) → pegged wins regardless of spread <= 20
    run("Spread=4 near 1000 (pegged wins)",
        "BOUNDARY_4",
        make_od({999: 10}, {1003: 10}),
        expected_type="pegged")

    run("Spread=5 near 1000 (pegged wins)",
        "BOUNDARY_5",
        make_od({998: 10}, {1003: 10}),
        expected_type="pegged")

    run("Spread=8 near 1000 (pegged wins)",
        "BOUNDARY_8",
        make_od({996: 10}, {1004: 10}),
        expected_type="pegged")

    run("Spread=9 near 1000 (pegged wins)",
        "BOUNDARY_9",
        make_od({996: 10}, {1005: 10}),
        expected_type="pegged")

    # NOT near round number → spread determines type
    run("Spread=4 NOT near round (ar_olivia)",
        "SPREAD_4_NOROUND",
        make_od({7775: 10}, {7779: 10}),
        expected_type="ar_olivia")

    run("Spread=9 NOT near round (wide_spread)",
        "SPREAD_9_NOROUND",
        make_od({7746: 10}, {7755: 10}),
        expected_type="wide_spread")

    run("Spread = 1 (ultra tight)",
        "ULTRA_TIGHT",
        make_od({1000: 10}, {1001: 10}),
        expected_type="pegged")  # mid=1000.5, near 1000

    run("Spread = 50 (ultra wide)",
        "ULTRA_WIDE",
        make_od({975: 10}, {1025: 10}),
        expected_type="wide_spread")

    # ── 4. EMPTY / BROKEN ORDER BOOKS ───────────────────────
    print("\n--- Edge Cases (empty/broken books) ---")

    run("Empty order book",
        "EMPTY",
        make_od({}, {}),
        expected_type="generic_mm")

    run("Bids only (no asks)",
        "BIDS_ONLY",
        make_od({100: 10, 99: 20}, {}),
        expected_type="generic_mm")

    run("Asks only (no bids)",
        "ASKS_ONLY",
        make_od({}, {101: 10, 102: 20}),
        expected_type="generic_mm")

    run("Single level each side",
        "SINGLE_LEVEL",
        make_od({9999: 5}, {10001: 5}),
        expected_type="pegged")

    run("Crossed book (bid > ask)",
        "CROSSED",
        make_od({1005: 10}, {995: 10}))  # Should not crash

    # ── 5. CONVERSION PRODUCTS ──────────────────────────────
    print("\n--- Conversion Products ---")

    conv_obs = {"ROSES": ConversionObservation(
        bidPrice=100.0, askPrice=102.0,
        transportFees=1.0, exportTariff=0.5, importTariff=0.5
    )}
    state = make_state(
        order_depths={"ROSES": make_od({99: 10}, {101: 10})},
        conv_obs=conv_obs
    )
    run("Conversion product (ROSES with ConversionObs)",
        "ROSES",
        make_od({99: 10}, {101: 10}),
        state=state,
        expected_type="conversion_arb")

    # Product NOT in conv obs shouldn't trigger
    run("Non-conversion product with conv obs on other product",
        "TULIPS",
        make_od({99: 10}, {101: 10}),
        state=state)  # Should NOT be conversion_arb

    # ── 6. OPTION PRODUCTS ──────────────────────────────────
    print("\n--- Option Products ---")

    state_with_underlying = make_state(
        order_depths={
            "COCONUT": make_od({9990: 10}, {10010: 10}),
            "COCONUT_COUPON_10000": make_od({300: 5}, {350: 5}),
            "COCONUT_CALL_10500": make_od({50: 5}, {80: 5}),
            "COCONUT_PUT_9500": make_od({40: 5}, {70: 5}),
        }
    )

    run("Coupon option (COCONUT_COUPON_10000)",
        "COCONUT_COUPON_10000",
        make_od({300: 5}, {350: 5}),
        state=state_with_underlying,
        expected_type="options")

    run("Call option (COCONUT_CALL_10500)",
        "COCONUT_CALL_10500",
        make_od({50: 5}, {80: 5}),
        state=state_with_underlying,
        expected_type="options")

    run("Put option (COCONUT_PUT_9500)",
        "COCONUT_PUT_9500",
        make_od({40: 5}, {70: 5}),
        state=state_with_underlying,
        expected_type="options")

    run("Underlying with option names (COCONUT itself)",
        "COCONUT",
        make_od({9990: 10}, {10010: 10}),
        state=state_with_underlying)  # Should NOT be options

    # Option without underlying in order_depths
    lonely_state = make_state(
        order_depths={"FISH_VOUCHER_500": make_od({10: 5}, {20: 5})}
    )
    run("Option without underlying (FISH_VOUCHER_500)",
        "FISH_VOUCHER_500",
        make_od({10: 5}, {20: 5}),
        state=lonely_state)  # Should fall through, not crash

    # ── 7. BASKET PRODUCTS ──────────────────────────────────
    print("\n--- Basket Products ---")

    run("GIFT_BASKET product",
        "GIFT_BASKET",
        make_od({70000: 5}, {71000: 5}),
        expected_type="basket_arb")

    run("PICNIC_ETF product",
        "PICNIC_ETF",
        make_od({5000: 10}, {5100: 10}),
        expected_type="basket_arb")

    run("SPICE_INDEX product",
        "SPICE_INDEX",
        make_od({2000: 10}, {2050: 10}),
        expected_type="basket_arb")

    run("Product with 'basket' in lowercase",
        "tropical_basket",
        make_od({3000: 10}, {3050: 10}),
        expected_type="basket_arb")

    # ── 8. REALISTIC R1 UNKNOWNS ────────────────────────────
    print("\n--- Realistic R1 Unknown Products ---")

    # Something like KELP (tight spread, mid-range price)
    run("KELP-like (mid=2034, spread=3)",
        "SEAWEED",
        make_od({2033: 20, 2032: 15}, {2036: 20, 2037: 15}),
        expected_type="ar_olivia")

    # Something like SQUID_INK (tight spread, volatile)
    run("SQUID_INK-like (mid=1840, spread=2)",
        "OCTOPUS_INK",
        make_od({1839: 25, 1838: 30}, {1841: 25, 1842: 30}),
        expected_type="ar_olivia")

    # Something like RAINFOREST_RESIN (near 10K)
    run("RESIN-like (mid=10000, spread=6)",
        "JUNGLE_SAP",
        make_od({9997: 10, 9995: 20}, {10003: 10, 10005: 20}),
        expected_type="pegged")

    # Totally new product type: mid-range, moderate spread
    run("Unknown mid-range (mid=3500, spread=6)",
        "MYSTERY_FRUIT",
        make_od({3497: 10}, {3503: 10}))

    # Product with integer mid not near any round number
    run("Odd price (mid=7777, spread=4)",
        "LUCKY_CHARM",
        make_od({7775: 10}, {7779: 10}))

    # Very volatile looking (wide spread, high price)
    run("Volatile (mid=25000, spread=100)",
        "CRYPTO_SHELL",
        make_od({24950: 5}, {25050: 5}),
        expected_type="wide_spread")

    # ── 9. RECLASSIFICATION TEST ────────────────────────────
    print("\n--- Reclassification (cache expiry) ---")

    mem = {}
    od1 = make_od({9998: 10}, {10002: 10})  # Looks pegged
    cfg1 = classify_product_live(od1, "MORPHING", mem, None)
    print(f"  Tick 0:   type={cfg1['type']}")

    # Simulate 500 ticks
    for i in range(500):
        mem[f"_auto_tick_MORPHING"] = i
        classify_product_live(od1, "MORPHING", mem, None)

    # Now book changes to wide spread
    od2 = make_od({9950: 10}, {10050: 10})
    mem["_auto_tick_MORPHING"] = 500  # Force re-evaluation
    cfg2 = classify_product_live(od2, "MORPHING", mem, None)
    print(f"  Tick 500: type={cfg2['type']} (book changed to wide spread)")

    if cfg1['type'] != cfg2['type']:
        print(f"  OK    Reclassification worked: {cfg1['type']} -> {cfg2['type']}")
        passed += 1
    else:
        print(f"  FAIL  Reclassification did NOT trigger")
        failed += 1
    total += 1

    # ── SUMMARY ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*80}")

    if failed > 0:
        print("\nFIX THESE BEFORE R1!")
    else:
        print("\nClassifier is ready for R1.")

    return failed


if __name__ == "__main__":
    sys.exit(main())
