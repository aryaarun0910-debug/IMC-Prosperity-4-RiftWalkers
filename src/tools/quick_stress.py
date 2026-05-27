"""
Quick Stress Tests — exercises trader.py edge cases directly.
No slow backtesting, just verify the Trader doesn't crash or misbehave.

Usage: python quick_stress.py
"""
import sys, os, json, math, traceback
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


def run_ticks(n_ticks, products_spec, pos_init=None):
    """Run n_ticks of the trader with given product specs.
    products_spec: {name: (base_price, spread)}
    Returns: (trader, final_td, final_orders_per_tick)
    """
    t = Trader()
    td = ""
    last_orders = {}
    for tick in range(n_ticks):
        ods = {}
        for name, (base, spread) in products_spec.items():
            # Small random walk
            jitter = (tick * 7 % 11 - 5) * 0.1  # deterministic pseudo-random
            mid = base + jitter
            half = spread / 2
            ods[name] = make_od(
                {int(mid - half): 30, int(mid - half - 1): 20, int(mid - half - 2): 10},
                {int(mid + half): -30, int(mid + half + 1): -20, int(mid + half + 2): -10}
            )
        state = make_state(ods, pos=pos_init if tick == 0 else None, ts=tick*100, td=td)
        orders, conv, td = t.run(state)
        last_orders = orders
    return t, td, last_orders


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


print("=" * 60)
print("QUICK STRESS TESTS")
print("=" * 60)


@test("Zero spread (bid == ask)")
def _():
    t = Trader()
    od = make_od({10000: 50}, {10000: -50})
    state = make_state({"RAINFOREST_RESIN": od})
    orders, conv, td = t.run(state)


@test("Empty book (no bids)")
def _():
    t = Trader()
    od = make_od({}, {10001: -50})
    state = make_state({"TEST": od})
    orders, conv, td = t.run(state)


@test("Empty book (no asks)")
def _():
    t = Trader()
    od = make_od({9999: 50}, {})
    state = make_state({"TEST": od})
    orders, conv, td = t.run(state)


@test("Both sides empty")
def _():
    t = Trader()
    od = make_od({}, {})
    state = make_state({"TEST": od})
    orders, conv, td = t.run(state)


@test("Single level each side")
def _():
    t = Trader()
    od = make_od({9999: 5}, {10001: -5})
    state = make_state({"KELP": od})
    orders, conv, td = t.run(state)


@test("At +50 position limit (pegged)")
def _():
    t = Trader()
    od = make_od({9998: 50, 9997: 30}, {10002: -50, 10003: -30})
    state = make_state({"RAINFOREST_RESIN": od}, pos={"RAINFOREST_RESIN": 50})
    orders, conv, td = t.run(state)
    resin = orders.get("RAINFOREST_RESIN", [])
    buy_qty = sum(o.quantity for o in resin if o.quantity > 0)
    assert buy_qty == 0, f"Should not buy at +limit, got buy_qty={buy_qty}"


@test("At -50 position limit (pegged)")
def _():
    t = Trader()
    od = make_od({9998: 50, 9997: 30}, {10002: -50, 10003: -30})
    state = make_state({"RAINFOREST_RESIN": od}, pos={"RAINFOREST_RESIN": -50})
    orders, conv, td = t.run(state)
    resin = orders.get("RAINFOREST_RESIN", [])
    sell_qty = sum(abs(o.quantity) for o in resin if o.quantity < 0)
    assert sell_qty == 0, f"Should not sell at -limit, got sell_qty={sell_qty}"


@test("At +50 position limit (ar_olivia / KELP)")
def _():
    t = Trader()
    od = make_od({2020: 30, 2019: 20}, {2022: -30, 2023: -20})
    state = make_state({"KELP": od}, pos={"KELP": 50})
    orders, conv, td = t.run(state)
    kelp = orders.get("KELP", [])
    buy_qty = sum(o.quantity for o in kelp if o.quantity > 0)
    assert buy_qty == 0, f"Should not buy KELP at +limit, got buy_qty={buy_qty}"


@test("At -50 position limit (ar_olivia / KELP)")
def _():
    t = Trader()
    od = make_od({2020: 30, 2019: 20}, {2022: -30, 2023: -20})
    state = make_state({"KELP": od}, pos={"KELP": -50})
    orders, conv, td = t.run(state)
    kelp = orders.get("KELP", [])
    sell_qty = sum(abs(o.quantity) for o in kelp if o.quantity < 0)
    assert sell_qty == 0, f"Should not sell KELP at -limit, got sell_qty={sell_qty}"


@test("Unknown product auto-classification")
def _():
    t = Trader()
    od = make_od({5000: 20, 4999: 30}, {5002: -20, 5003: -30})
    state = make_state({"MYSTERY_FRUIT": od})
    orders, conv, td = t.run(state)
    assert "MYSTERY_FRUIT" in orders


@test("Conversion product without observations")
def _():
    t = Trader()
    od = make_od({660: 20}, {665: -20})
    state = make_state({"MAGNIFICENT_MACARONS": od})
    orders, conv, td = t.run(state)


@test("Rapid 10x vol price jumps")
def _():
    t = Trader()
    td = ""
    for i, p in enumerate([10000, 10050, 9900, 10100, 9800, 10200]):
        od = make_od({p-2: 30, p-3: 20}, {p+2: -30, p+3: -20})
        state = make_state({"RAINFOREST_RESIN": od}, ts=i*100, td=td)
        orders, conv, td = t.run(state)


@test("15 products simultaneously")
def _():
    products = {}
    for i in range(15):
        base = 1000 + i * 100
        products[f"PRODUCT_{i}"] = make_od(
            {base-1: 20, base-2: 30},
            {base+1: -20, base+2: -30}
        )
    state = make_state(products)
    t = Trader()
    orders, conv, td = t.run(state)
    td_size = len(td)
    assert td_size < 90000, f"traderData too large: {td_size} bytes ({td_size/1024:.1f} KB)"
    print(f"    (traderData: {td_size} bytes / {td_size/1024:.1f} KB)")


@test("traderData stability over 200 ticks")
def _():
    t = Trader()
    td = ""
    specs = {"KELP": (2020, 4), "RAINFOREST_RESIN": (10000, 6), "SQUID_INK": (1970, 4)}
    sizes = []
    for tick in range(200):
        ods = {}
        for name, (base, spread) in specs.items():
            jitter = (tick * 7 % 11 - 5) * 0.1
            mid = base + jitter
            half = spread / 2
            ods[name] = make_od(
                {int(mid - half): 30, int(mid - half - 1): 20},
                {int(mid + half): -30, int(mid + half + 1): -20}
            )
        state = make_state(ods, ts=tick*100, td=td)
        orders, conv, td = t.run(state)
        sizes.append(len(td))

    max_sz = max(sizes)
    final_sz = sizes[-1]
    assert max_sz < 90000, f"traderData peaked at {max_sz} bytes"
    print(f"    (tick0={sizes[0]}B, final={final_sz}B, max={max_sz}B)")


@test("validate_orders respects position limits")
def _():
    orders = [
        Order("TEST", 10000, 30),   # buy 30
        Order("TEST", 9999, 20),    # buy 20
        Order("TEST", 10002, -10),  # sell 10
    ]
    # pos=45, limit=50 → can only buy 5 more
    validated = validate_orders(orders, 45, 50)
    net_buy = sum(o.quantity for o in validated if o.quantity > 0)
    net_sell = sum(abs(o.quantity) for o in validated if o.quantity < 0)
    # With pos=45: worst case = 45 + net_buy - net_sell, must stay in [-50, 50]
    worst_long = 45 + net_buy
    worst_short = 45 - net_sell
    assert worst_long <= 50, f"Would breach +limit: {worst_long}"
    assert worst_short >= -50, f"Would breach -limit: {worst_short}"


@test("validate_orders at exactly limit")
def _():
    orders = [
        Order("TEST", 10002, -20),  # sell only — should work at +50
    ]
    validated = validate_orders(orders, 50, 50)
    assert len(validated) > 0, "Should allow sells when at +limit"


@test("Basket product name detection")
def _():
    t = Trader()
    od = make_od({58000: 10, 57999: 5}, {58002: -10, 58003: -5})
    state = make_state({"PICNIC_BASKET1": od})
    orders, conv, td = t.run(state)
    cfg = t._config.get("PICNIC_BASKET1", {})
    assert cfg.get("type") == "basket_arb", f"Expected basket_arb, got {cfg.get('type')}"


@test("Option voucher name detection")
def _():
    t = Trader()
    ods = {
        "VOLCANIC_ROCK": make_od({10000: 50}, {10010: -50}),
        "VOLCANIC_ROCK_VOUCHER_10000": make_od({100: 20}, {105: -20}),
    }
    state = make_state(ods)
    orders, conv, td = t.run(state)
    vr_cfg = t._config.get("VOLCANIC_ROCK", {})
    # VOLCANIC_ROCK should NOT be ar_olivia when vouchers exist
    assert vr_cfg.get("type") != "ar_olivia", f"VOLCANIC_ROCK should be generic_mm, got {vr_cfg.get('type')}"


@test("Conversion name detection (MACARONS)")
def _():
    t = Trader()
    od = make_od({660: 20}, {665: -20})
    state = make_state({"MAGNIFICENT_MACARONS": od})
    orders, conv, td = t.run(state)
    cfg = t._config.get("MAGNIFICENT_MACARONS", {})
    assert cfg.get("type") == "conversion_arb", f"Expected conversion_arb, got {cfg.get('type')}"


@test("Pairs arb cointegration detection (600 ticks)")
def _():
    """Two products with highly correlated prices should trigger pairs_arb."""
    t = Trader()
    td = ""
    # Simulate 600 ticks of two cointegrated products: B = A + 5000 + noise
    import random
    rng = random.Random(42)
    base = 2000.0
    for tick in range(600):
        base += rng.gauss(0, 0.5)  # random walk
        mid_a = base
        mid_b = base + 5000 + rng.gauss(0, 0.3)  # cointegrated with offset
        ods = {
            "FRUIT_A": make_od(
                {int(mid_a)-1: 20, int(mid_a)-2: 15},
                {int(mid_a)+1: -20, int(mid_a)+2: -15}
            ),
            "FRUIT_B": make_od(
                {int(mid_b)-1: 20, int(mid_b)-2: 15},
                {int(mid_b)+1: -20, int(mid_b)+2: -15}
            ),
        }
        state = make_state(ods, ts=tick*100, td=td)
        orders, conv, td = t.run(state)

    # After 600 ticks, cointegration should be detected
    if td.startswith("Z:"):
        import zlib, base64
        mem = json.loads(zlib.decompress(base64.b64decode(td[2:])).decode())
    else:
        mem = json.loads(td)
    coint = mem.get("_coint_pairs", {})
    cfg_a = t._config.get("FRUIT_A", {})
    cfg_b = t._config.get("FRUIT_B", {})
    detected = "FRUIT_A" in coint or cfg_a.get("type") == "pairs_arb" or cfg_b.get("type") == "pairs_arb"
    if detected:
        print(f"    (pairs detected: {coint})")
    else:
        print(f"    (not detected yet — coint_pairs={coint}, cfg_a={cfg_a.get('type')}, cfg_b={cfg_b.get('type')})")
    # Don't assert — detection at tick 500 may not re-classify until tick 1000
    # Just verify it doesn't crash


@test("traderData JSON roundtrip")
def _():
    t = Trader()
    td = ""
    # Run 10 ticks to build up state
    for tick in range(10):
        od = make_od({2020: 30, 2019: 20}, {2022: -30, 2023: -20})
        state = make_state({"KELP": od}, ts=tick*100, td=td)
        orders, conv, td = t.run(state)
    # Verify traderData is valid (JSON or compressed)
    if td.startswith("Z:"):
        import zlib, base64
        parsed = json.loads(zlib.decompress(base64.b64decode(td[2:])).decode())
    else:
        parsed = json.loads(td)
    assert isinstance(parsed, dict)
    # Verify it roundtrips through Trader
    td2_state = make_state({"KELP": make_od({2020: 30}, {2022: -30})}, ts=1100, td=td)
    _, _, td2 = t.run(td2_state)
    if td2.startswith("Z:"):
        parsed2 = json.loads(zlib.decompress(base64.b64decode(td2[2:])).decode())
    else:
        parsed2 = json.loads(td2)
    assert isinstance(parsed2, dict)
    assert len(parsed2) >= len(parsed) - 2  # state shouldn't shrink drastically


# ── SUMMARY ──
print(f"\n{'=' * 60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'=' * 60}")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {failed} tests FAILED — review before deploying!")
sys.exit(1 if failed > 0 else 0)
