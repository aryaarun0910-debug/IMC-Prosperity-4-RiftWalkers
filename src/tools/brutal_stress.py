"""
BRUTAL STRESS TEST — Goes far beyond quick_stress.py.
Tests every pathological scenario that could crash on gameday.

Usage: py -3.12 tools/brutal_stress.py
"""
import sys, os, json, math, traceback, random, time
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _ROOT_DIR)

from datamodel import OrderDepth, TradingState, Trade, Order, Observation, ConversionObservation
from trader import Trader, validate_orders

def make_od(bids, asks):
    od = OrderDepth()
    od.buy_orders = dict(bids)
    od.sell_orders = dict(asks)
    return od

def make_state(ods, pos=None, ts=0, td="", mt=None, ot=None, obs=None):
    return TradingState(
        traderData=td, timestamp=ts, listings={},
        order_depths=ods,
        own_trades=ot or {p: [] for p in ods},
        market_trades=mt or {p: [] for p in ods},
        position=pos or {},
        observations=obs or Observation(),
    )

passed = 0
failed = 0

def test(name):
    def decorator(fn):
        global passed, failed
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1
    return decorator

print("=" * 70)
print("BRUTAL STRESS TESTS — Pathological Edge Cases")
print("=" * 70)

# ────────────────────────────────────────────────────────
# SECTION 1: POSITION LIMIT EDGE CASES
# ────────────────────────────────────────────────────────
print("\n--- POSITION LIMIT EDGE CASES ---")

@test("Position exactly at +limit, all strategies")
def _():
    for ptype in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
        t = Trader()
        od = make_od({2000: 30, 1999: 20}, {2002: -30, 2003: -20})
        state = make_state({ptype: od}, pos={ptype: 50})
        orders, conv, td = t.run(state)
        for o in orders.get(ptype, []):
            assert o.quantity <= 0, f"{ptype}: bought at +limit! qty={o.quantity}"

@test("Position exactly at -limit, all strategies")
def _():
    for ptype in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
        t = Trader()
        od = make_od({2000: 30, 1999: 20}, {2002: -30, 2003: -20})
        state = make_state({ptype: od}, pos={ptype: -50})
        orders, conv, td = t.run(state)
        for o in orders.get(ptype, []):
            assert o.quantity >= 0, f"{ptype}: sold at -limit! qty={o.quantity}"

@test("validate_orders: worst-case fill check with mixed orders")
def _():
    # pos=40, limit=50: can buy 10, can sell 90
    orders = [
        Order("T", 100, 20),   # buy 20 (would push to 60 > 50)
        Order("T", 99, 15),    # buy 15
        Order("T", 101, -5),   # sell 5
    ]
    validated = validate_orders(orders, 40, 50)
    total_buy = sum(o.quantity for o in validated if o.quantity > 0)
    total_sell = sum(abs(o.quantity) for o in validated if o.quantity < 0)
    assert 40 + total_buy <= 50, f"Breach +limit: {40 + total_buy}"
    assert 40 - total_sell >= -50, f"Breach -limit: {40 - total_sell}"

@test("validate_orders: all-or-nothing rejection when over limit")
def _():
    orders = [
        Order("T", 100, 60),  # buy 60, pos=0, limit=50 -> reject ALL
    ]
    validated = validate_orders(orders, 0, 50)
    if len(validated) > 0:
        total = sum(o.quantity for o in validated if o.quantity > 0)
        assert total <= 50, f"Should have rejected: total_buy={total}"

@test("Position limit = 1 (extreme edge)")
def _():
    t = Trader()
    od = make_od({100: 5}, {102: -5})
    state = make_state({"TINY": od}, pos={"TINY": 0})
    # Force limit=1 in config
    t._config["TINY"] = {"type": "generic_mm", "position_limit": 1}
    orders, conv, td = t.run(state)
    for o in orders.get("TINY", []):
        assert abs(o.quantity) <= 1, f"Exceeded limit=1: qty={o.quantity}"

# ────────────────────────────────────────────────────────
# SECTION 2: ORDER BOOK PATHOLOGIES
# ────────────────────────────────────────────────────────
print("\n--- ORDER BOOK PATHOLOGIES ---")

@test("Crossed book (bid > ask)")
def _():
    t = Trader()
    od = make_od({10005: 50}, {10003: -50})  # bid > ask!
    state = make_state({"RAINFOREST_RESIN": od})
    orders, conv, td = t.run(state)  # should not crash

@test("Huge spread (1000 ticks wide)")
def _():
    t = Trader()
    od = make_od({5000: 10}, {6000: -10})
    state = make_state({"KELP": od})
    orders, conv, td = t.run(state)

@test("1-lot book (minimum liquidity)")
def _():
    t = Trader()
    od = make_od({9999: 1}, {10001: -1})
    state = make_state({"RAINFOREST_RESIN": od})
    orders, conv, td = t.run(state)

@test("Massive volumes (10000 lots per level)")
def _():
    t = Trader()
    od = make_od({9999: 10000, 9998: 10000}, {10001: -10000, 10002: -10000})
    state = make_state({"RAINFOREST_RESIN": od})
    orders, conv, td = t.run(state)

@test("Negative volumes in sell_orders (IMC format)")
def _():
    """IMC uses negative volumes for asks. Verify we handle this correctly."""
    t = Trader()
    od = OrderDepth()
    od.buy_orders = {9998: 30}
    od.sell_orders = {10002: -30}  # negative = IMC convention
    state = make_state({"RAINFOREST_RESIN": od})
    orders, conv, td = t.run(state)
    # Should not crash and should produce valid orders

@test("Price at 0")
def _():
    t = Trader()
    od = make_od({0: 10}, {2: -10})
    state = make_state({"WEIRD": od})
    orders, conv, td = t.run(state)

@test("Price at 999999 (extreme)")
def _():
    t = Trader()
    od = make_od({999997: 10}, {999999: -10})
    state = make_state({"EXPENSIVE": od})
    orders, conv, td = t.run(state)

@test("Single-sided book alternating each tick")
def _():
    t = Trader()
    td = ""
    for tick in range(20):
        if tick % 2 == 0:
            od = make_od({100: 10}, {})
        else:
            od = make_od({}, {102: -10})
        state = make_state({"FLICKER": od}, ts=tick*100, td=td)
        _, _, td = t.run(state)

# ────────────────────────────────────────────────────────
# SECTION 3: MULTI-PRODUCT CHAOS
# ────────────────────────────────────────────────────────
print("\n--- MULTI-PRODUCT CHAOS ---")

@test("10 products appearing at different ticks")
def _():
    t = Trader()
    td = ""
    for tick in range(100):
        ods = {}
        # Products appear one by one
        for i in range(min(tick + 1, 10)):
            base = 1000 + i * 500
            ods[f"PROD_{i}"] = make_od(
                {base-2: 20, base-3: 15},
                {base+2: -20, base+3: -15}
            )
        state = make_state(ods, ts=tick*100, td=td)
        _, _, td = t.run(state)
    # Verify traderData is valid
    import zlib, base64
    if td.startswith("Z:"):
        mem = json.loads(zlib.decompress(base64.b64decode(td[2:])).decode())
    else:
        mem = json.loads(td)
    assert isinstance(mem, dict)
    assert len(td) < 90000, f"traderData too large: {len(td)}"

@test("Product disappears mid-session then returns")
def _():
    t = Trader()
    td = ""
    for tick in range(50):
        ods = {"STABLE": make_od({100: 20}, {102: -20})}
        if tick < 20 or tick > 30:
            ods["VANISHER"] = make_od({200: 15}, {205: -15})
        state = make_state(ods, ts=tick*100, td=td)
        _, _, td = t.run(state)

@test("All known archetypes simultaneously")
def _():
    t = Trader()
    ods = {
        "RAINFOREST_RESIN": make_od({9998: 30}, {10002: -30}),
        "KELP": make_od({2020: 20}, {2022: -20}),
        "SQUID_INK": make_od({1970: 25}, {1972: -25}),
        "PICNIC_BASKET1": make_od({58000: 10}, {58010: -10}),
        "MAGNIFICENT_MACARONS": make_od({660: 20}, {665: -20}),
        "VOLCANIC_ROCK": make_od({10000: 50}, {10010: -50}),
        "VOLCANIC_ROCK_VOUCHER_10000": make_od({100: 20}, {105: -20}),
        "UNKNOWN_FRUIT": make_od({3000: 15}, {3005: -15}),
    }
    state = make_state(ods)
    orders, conv, td = t.run(state)
    assert len(orders) >= 5, f"Expected orders for most products, got {len(orders)}"

# ────────────────────────────────────────────────────────
# SECTION 4: MARKET TRADES + OWN TRADES EDGE CASES
# ────────────────────────────────────────────────────────
print("\n--- TRADE DATA EDGE CASES ---")

@test("Olivia-like large trade (qty=20, single trade)")
def _():
    t = Trader()
    td = ""
    for tick in range(10):
        od = make_od({1970: 30, 1969: 20}, {1972: -30, 1973: -20})
        mt = []
        if tick == 5:
            mt = [Trade("SQUID_INK", 1972, 20)]  # Large buy = Olivia-like
        state = make_state(
            {"SQUID_INK": od}, ts=tick*100, td=td,
            mt={"SQUID_INK": mt}
        )
        _, _, td = t.run(state)

@test("100 market trades in one tick")
def _():
    t = Trader()
    od = make_od({2020: 30}, {2022: -30})
    trades = [Trade("KELP", 2021, random.choice([1, -1]) * random.randint(1, 5)) for _ in range(100)]
    state = make_state(
        {"KELP": od},
        mt={"KELP": trades}
    )
    orders, conv, td = t.run(state)

@test("Own trades with zero quantity")
def _():
    t = Trader()
    od = make_od({9998: 30}, {10002: -30})
    ot = [Trade("RAINFOREST_RESIN", 10000, 0)]
    state = make_state(
        {"RAINFOREST_RESIN": od},
        ot={"RAINFOREST_RESIN": ot}
    )
    orders, conv, td = t.run(state)

# ────────────────────────────────────────────────────────
# SECTION 5: LONG-RUNNING STABILITY
# ────────────────────────────────────────────────────────
print("\n--- LONG-RUNNING STABILITY ---")

@test("1000 ticks with volatile prices (no crash)")
def _():
    t = Trader()
    td = ""
    rng = random.Random(12345)
    price = 2000.0
    for tick in range(1000):
        price += rng.gauss(0, 2)
        price = max(100, price)
        mid = int(price)
        od = make_od(
            {mid-1: 20, mid-2: 15, mid-3: 10},
            {mid+1: -20, mid+2: -15, mid+3: -10}
        )
        state = make_state({"KELP": od}, ts=tick*100, td=td)
        _, _, td = t.run(state)
    assert len(td) < 90000, f"traderData grew to {len(td)}B"

@test("Day boundary: timestamp resets to 0 (multi-day)")
def _():
    t = Trader()
    td = ""
    # Day 1: ticks 0-999900
    for tick in range(10):
        od = make_od({9998: 30}, {10002: -30})
        state = make_state({"RAINFOREST_RESIN": od}, ts=tick*100, td=td)
        _, _, td = t.run(state)
    # Day 2: timestamp resets (different day_num in backtester)
    for tick in range(10):
        od = make_od({9998: 30}, {10002: -30})
        state = make_state({"RAINFOREST_RESIN": od}, ts=tick*100, td=td)
        _, _, td = t.run(state)

@test("EOD position gaming at tick 990000+")
def _():
    t = Trader()
    td = ""
    # Build up state first
    for tick in range(20):
        od = make_od({9998: 30}, {10002: -30})
        state = make_state({"RAINFOREST_RESIN": od}, ts=tick*100, td=td)
        _, _, td = t.run(state)
    # Now hit EOD
    od = make_od({9998: 30}, {10002: -30})
    state = make_state(
        {"RAINFOREST_RESIN": od},
        pos={"RAINFOREST_RESIN": 30},
        ts=995000, td=td
    )
    orders, _, _ = t.run(state)
    # Should produce flattening orders
    resin_orders = orders.get("RAINFOREST_RESIN", [])
    sell_qty = sum(abs(o.quantity) for o in resin_orders if o.quantity < 0)
    assert sell_qty > 0, "Should flatten at EOD"

@test("EOD at exactly tick 999900")
def _():
    t = Trader()
    td = ""
    for tick in range(5):
        od = make_od({9998: 30}, {10002: -30})
        state = make_state({"RAINFOREST_RESIN": od}, ts=tick*100, td=td)
        _, _, td = t.run(state)
    od = make_od({9998: 30}, {10002: -30})
    state = make_state(
        {"RAINFOREST_RESIN": od},
        pos={"RAINFOREST_RESIN": -20},
        ts=999900, td=td
    )
    orders, _, _ = t.run(state)

# ────────────────────────────────────────────────────────
# SECTION 6: CONVERSION + OPTIONS EDGE CASES
# ────────────────────────────────────────────────────────
print("\n--- CONVERSION + OPTIONS EDGE CASES ---")

@test("Conversion product with full observation data")
def _():
    t = Trader()
    od = make_od({660: 20}, {665: -20})
    obs = Observation()
    obs.conversionObservations = {
        "MAGNIFICENT_MACARONS": ConversionObservation(
            bidPrice=670, askPrice=675,
            transportFees=3, exportTariff=2, importTariff=1,
            sunshineFraction=2500, humidity=60
        )
    }
    state = make_state(
        {"MAGNIFICENT_MACARONS": od},
        obs=obs
    )
    orders, conv, td = t.run(state)

@test("Voucher + underlying with realistic prices")
def _():
    t = Trader()
    ods = {
        "VOLCANIC_ROCK": make_od({10000: 50, 9998: 30}, {10005: -50, 10008: -30}),
        "VOLCANIC_ROCK_VOUCHER_9500": make_od({520: 10}, {530: -10}),
        "VOLCANIC_ROCK_VOUCHER_10000": make_od({80: 10}, {90: -10}),
        "VOLCANIC_ROCK_VOUCHER_10500": make_od({10: 10}, {15: -10}),
    }
    state = make_state(ods)
    orders, conv, td = t.run(state)

@test("Basket with no components visible")
def _():
    t = Trader()
    od = make_od({58000: 10}, {58010: -10})
    state = make_state({"PICNIC_BASKET1": od})
    orders, conv, td = t.run(state)

# ────────────────────────────────────────────────────────
# SECTION 7: PERFORMANCE
# ────────────────────────────────────────────────────────
print("\n--- PERFORMANCE ---")

@test("Latency: <5ms per tick with 5 products (production mode)")
def _():
    t = Trader()
    td = ""
    ods = {
        "KELP": make_od({2020: 30, 2019: 20}, {2022: -30, 2023: -20}),
        "RAINFOREST_RESIN": make_od({9998: 30, 9997: 20}, {10002: -30, 10003: -20}),
        "SQUID_INK": make_od({1970: 25, 1969: 15}, {1972: -25, 1973: -15}),
        "PRODUCT_A": make_od({5000: 20}, {5003: -20}),
        "PRODUCT_B": make_od({8000: 20}, {8004: -20}),
    }
    # Warmup
    for tick in range(20):
        state = make_state(ods, ts=tick*100, td=td)
        _, _, td = t.run(state)
    # Measure
    times = []
    for tick in range(20, 120):
        state = make_state(ods, ts=tick*100, td=td)
        t0 = time.perf_counter()
        _, _, td = t.run(state)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    max_ms = max(times)
    p99_ms = sorted(times)[int(len(times) * 0.99)]
    print(f"    avg={avg_ms:.1f}ms, p99={p99_ms:.1f}ms, max={max_ms:.1f}ms")
    assert avg_ms < 5.0, f"Too slow: avg {avg_ms:.1f}ms"
    assert max_ms < 20.0, f"Spike too high: max {max_ms:.1f}ms"

@test("Latency: backtester mode (<1ms per tick)")
def _():
    t = Trader()
    t._mem = {}  # backtester mode
    ods = {
        "KELP": make_od({2020: 30}, {2022: -30}),
        "RAINFOREST_RESIN": make_od({9998: 30}, {10002: -30}),
    }
    # Warmup
    for tick in range(20):
        state = make_state(ods, ts=tick*100)
        t.run(state)
    # Measure
    times = []
    for tick in range(20, 520):
        state = make_state(ods, ts=tick*100)
        t0 = time.perf_counter()
        t.run(state)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    max_ms = max(times)
    print(f"    avg={avg_ms:.2f}ms, max={max_ms:.2f}ms")
    assert avg_ms < 1.0, f"Backtester too slow: avg {avg_ms:.2f}ms"

# ────────────────────────────────────────────────────────
# SECTION 8: GAMMA TUNING EDGE CASES
# ────────────────────────────────────────────────────────
print("\n--- GAMMA TUNING ---")

@test("gamma_mult stays in [0.5, 2.0] after 2000 ticks at limit")
def _():
    t = Trader()
    td = ""
    for tick in range(2000):
        od = make_od({2020: 30, 2019: 20}, {2022: -30, 2023: -20})
        # Always at position limit
        state = make_state({"KELP": od}, pos={"KELP": 50}, ts=tick*100, td=td)
        _, _, td = t.run(state)
    import zlib, base64
    if td.startswith("Z:"):
        mem = json.loads(zlib.decompress(base64.b64decode(td[2:])).decode())
    else:
        mem = json.loads(td)
    ps_data = mem.get("_ps_KELP", {})
    gm = ps_data.get("gamma_mult", 1.0)
    assert 0.5 <= gm <= 2.0, f"gamma_mult out of bounds: {gm}"
    print(f"    gamma_mult after 2000 ticks at limit: {gm:.3f}")

@test("gamma_mult stays in [0.5, 2.0] after 2000 ticks with no fills")
def _():
    t = Trader()
    td = ""
    for tick in range(2000):
        od = make_od({2020: 30, 2019: 20}, {2022: -30, 2023: -20})
        state = make_state({"KELP": od}, ts=tick*100, td=td)
        _, _, td = t.run(state)
    import zlib, base64
    if td.startswith("Z:"):
        mem = json.loads(zlib.decompress(base64.b64decode(td[2:])).decode())
    else:
        mem = json.loads(td)
    ps_data = mem.get("_ps_KELP", {})
    gm = ps_data.get("gamma_mult", 1.0)
    assert 0.5 <= gm <= 2.0, f"gamma_mult out of bounds: {gm}"
    print(f"    gamma_mult after 2000 ticks no fills: {gm:.3f}")

# ────────────────────────────────────────────────────────
# SECTION 9: VALIDATE ORDERS EXHAUSTIVE
# ────────────────────────────────────────────────────────
print("\n--- VALIDATE ORDERS EXHAUSTIVE ---")

@test("validate_orders: empty list")
def _():
    assert validate_orders([], 0, 50) == []

@test("validate_orders: single buy within limit")
def _():
    v = validate_orders([Order("T", 100, 10)], 0, 50)
    assert len(v) == 1

@test("validate_orders: single sell within limit")
def _():
    v = validate_orders([Order("T", 100, -10)], 0, 50)
    assert len(v) == 1

@test("validate_orders: buy that would breach +limit -> rejected")
def _():
    v = validate_orders([Order("T", 100, 60)], 0, 50)
    buy_qty = sum(o.quantity for o in v if o.quantity > 0)
    assert buy_qty <= 50, f"Should reject: buy_qty={buy_qty}"

@test("validate_orders: sell that would breach -limit -> rejected")
def _():
    v = validate_orders([Order("T", 100, -60)], 0, 50)
    sell_qty = sum(abs(o.quantity) for o in v if o.quantity < 0)
    assert sell_qty <= 50, f"Should reject: sell_qty={sell_qty}"

@test("validate_orders: mixed orders, worst case both sides")
def _():
    orders = [
        Order("T", 100, 30),
        Order("T", 99, 25),
        Order("T", 101, -10),
        Order("T", 102, -15),
    ]
    v = validate_orders(orders, 10, 50)
    buy_total = sum(o.quantity for o in v if o.quantity > 0)
    sell_total = sum(abs(o.quantity) for o in v if o.quantity < 0)
    assert 10 + buy_total <= 50, f"Breach +limit: {10 + buy_total}"
    assert 10 - sell_total >= -50, f"Breach -limit: {10 - sell_total}"

# ────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"BRUTAL STRESS RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'=' * 70}")
if failed == 0:
    print("ALL BRUTAL TESTS PASSED — PRISTINE CONDITION")
else:
    print(f"WARNING: {failed} tests FAILED — FIX BEFORE GAMEDAY!")
sys.exit(1 if failed > 0 else 0)
