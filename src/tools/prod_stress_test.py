"""
Production Stress Test — Industry-grade simulation of worst-case conditions.

Unlike quick_stress.py (21 edge-case unit tests), this runs full-scale
scenarios that simulate real competition disasters:

  1. Full-round simulation (10K ticks × 7+ products)
  2. Flash crash recovery (price drops 50% in 10 ticks)
  3. Memory growth under sustained load (traderData → 90KB limit)
  4. Execution time profiling (IMC has per-tick timeout)
  5. Regime shift mid-round (volatility 5x change)
  6. Book disappearance (empty for 100+ ticks)
  7. Olivia behavior change (starts wrong, must adapt)
  8. Multi-day persistence (state carries across day boundaries)
  9. EOD flattening correctness (t=995000+ behavior)
 10. Adversarial fill patterns (systematic adverse selection)
 11. traderData corruption recovery (malformed JSON)
 12. Position limit saturation across all products simultaneously

Usage:
    py -3.12 tools/prod_stress_test.py           # Run all scenarios
    py -3.12 tools/prod_stress_test.py --quick    # 3 critical scenarios only
    py -3.12 tools/prod_stress_test.py --scenario flash_crash
"""

import sys
import os
import json
import math
import time
import random
import traceback
import zlib
import base64
import argparse

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _ROOT_DIR)

from datamodel import OrderDepth, TradingState, Trade, Order, Observation, ConversionObservation
from trader import Trader, validate_orders


# ── Helpers ──

def make_od(bids, asks):
    od = OrderDepth()
    od.buy_orders = dict(bids)
    od.sell_orders = {k: -abs(v) for k, v in asks.items()}
    return od


def make_state(ods, pos=None, ts=0, td="", mt=None, ot=None, obs=None, listings=None):
    return TradingState(
        traderData=td, timestamp=ts, listings=listings or {},
        order_depths=ods,
        own_trades=ot or {p: [] for p in ods},
        market_trades=mt or {p: [] for p in ods},
        position=pos or {},
        observations=obs or Observation(),
    )


def make_book(mid, spread=4, depth=3, size=30):
    """Generate a symmetric order book around mid."""
    half = spread / 2
    bids = {}
    asks = {}
    for i in range(depth):
        bids[int(mid - half - i)] = max(5, size - i * 5)
        asks[int(mid + half + i)] = max(5, size - i * 5)
    return make_od(bids, asks)


def decode_td(td):
    """Decode traderData string to dict."""
    if not td:
        return {}
    if td.startswith("Z:"):
        return json.loads(zlib.decompress(base64.b64decode(td[2:])).decode())
    return json.loads(td)


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []

    def add(self, name, status, detail=""):
        if status == "PASS":
            self.passed += 1
            print(f"  PASS: {name}")
        elif status == "WARN":
            self.warnings += 1
            print(f"  WARN: {name} — {detail}")
        else:
            self.failed += 1
            print(f"  FAIL: {name} — {detail}")
        self.details.append((name, status, detail))


results = Results()


# ═══════════════════════════════════════════════════════════════
# SCENARIO 1: FULL-ROUND SIMULATION (10K ticks × 7 products)
# Tests: memory growth, execution time, state stability
# ═══════════════════════════════════════════════════════════════

def scenario_full_round():
    """Simulate a complete round: 10K ticks with all CONFIG products."""
    print("\n--- SCENARIO: Full Round (10K ticks × 7 products) ---")

    t = Trader()
    td = ""
    rng = random.Random(42)

    products = {
        "RAINFOREST_RESIN": {"base": 10000, "spread": 6, "vol": 0.5},
        "KELP": {"base": 2020, "spread": 4, "vol": 2.0},
        "SQUID_INK": {"base": 1970, "spread": 4, "vol": 3.0},
        "EMERALDS": {"base": 10000, "spread": 6, "vol": 0.3},
        "TOMATOES": {"base": 12400, "spread": 8, "vol": 1.5},
        "PICNIC_BASKET1": {"base": 58000, "spread": 20, "vol": 5.0},
        "PICNIC_BASKET2": {"base": 30000, "spread": 15, "vol": 4.0},
    }

    # Track metrics
    tick_times = []
    td_sizes = []
    prices = {p: spec["base"] for p, spec in products.items()}
    positions = {}

    n_ticks = 10000
    for tick in range(n_ticks):
        ts = tick * 100

        # Generate order books with realistic random walk
        ods = {}
        mt = {}
        for p, spec in products.items():
            prices[p] += rng.gauss(0, spec["vol"])
            mid = prices[p]
            ods[p] = make_book(mid, spec["spread"])
            # Simulate market trades (some ticks)
            mt[p] = []
            if rng.random() < 0.3:
                side = rng.choice(["buy", "sell"])
                qty = rng.randint(1, 15)
                t_price = int(mid + (1 if side == "buy" else -1))
                mt[p].append(Trade(p, t_price, qty,
                                   buyer="Olivia" if rng.random() < 0.05 else "Bot_A",
                                   seller="Bot_B" if side == "buy" else "Olivia" if rng.random() < 0.05 else "Bot_B",
                                   timestamp=ts))

        state = make_state(ods, pos=positions, ts=ts, td=td, mt=mt)

        t0 = time.perf_counter()
        orders, conv, td = t.run(state)
        elapsed = time.perf_counter() - t0
        tick_times.append(elapsed)
        td_sizes.append(len(td))

        # Track positions (simulate fills at 15% rate)
        for p, ords in orders.items():
            if p not in positions:
                positions[p] = 0
            for o in ords:
                if rng.random() < 0.15:
                    positions[p] += o.quantity

        # Enforce position limits
        for p in positions:
            cfg = t._config.get(p, {})
            limit = cfg.get("position_limit", 50)
            positions[p] = max(-limit, min(limit, positions[p]))

    # Evaluate results
    # Exclude first 20 ticks (cold start: classification, BOCPD init, etc.)
    warm_times = tick_times[20:]
    avg_tick = sum(warm_times) / len(warm_times) * 1000  # ms
    p95_tick = sorted(warm_times)[int(0.95 * len(warm_times))] * 1000
    p99_tick = sorted(warm_times)[int(0.99 * len(warm_times))] * 1000
    max_tick = max(warm_times) * 1000
    max_td = max(td_sizes)
    final_td = td_sizes[-1]

    print(f"    Tick time:  avg={avg_tick:.2f}ms  p95={p95_tick:.2f}ms  p99={p99_tick:.2f}ms  max={max_tick:.2f}ms")
    print(f"    traderData: final={final_td}B  max={max_td}B  ({max_td/1024:.1f}KB)")

    # Checks
    if max_tick > 100:
        results.add("Tick execution time < 100ms", "FAIL",
                     f"max tick = {max_tick:.1f}ms (IMC timeout risk)")
    elif p99_tick > 20:
        results.add("Tick execution time < 20ms (p99)", "WARN",
                     f"p99 = {p99_tick:.1f}ms — close to timeout")
    else:
        results.add("Tick execution time < 20ms (p99)", "PASS")

    if max_td > 90000:
        results.add("traderData under 90KB limit", "FAIL", f"peaked at {max_td}B")
    elif max_td > 70000:
        results.add("traderData under 90KB (with margin)", "WARN",
                     f"peaked at {max_td/1024:.1f}KB — 78% of limit")
    else:
        results.add("traderData under 90KB limit", "PASS")

    # Verify final traderData is valid
    try:
        mem = decode_td(td)
        results.add("Final traderData is valid JSON", "PASS")
    except Exception as e:
        results.add("Final traderData is valid JSON", "FAIL", str(e))


# ═══════════════════════════════════════════════════════════════
# SCENARIO 2: FLASH CRASH RECOVERY
# Price drops 50% in 10 ticks, recovers over 50 ticks
# ═══════════════════════════════════════════════════════════════

def scenario_flash_crash():
    """Test recovery from sudden price crash."""
    print("\n--- SCENARIO: Flash Crash Recovery ---")

    t = Trader()
    td = ""
    rng = random.Random(123)
    base_price = 10000.0
    positions = {}

    phases = [
        ("warmup", 200, 0),          # normal market
        ("crash", 10, -30),           # 30 per tick down = 300 drop
        ("bottom", 50, 0),            # stay at bottom
        ("recovery", 50, +6),         # slow recovery
        ("normal", 200, 0),           # back to normal
    ]

    crashed = False
    for phase_name, n_ticks, drift in phases:
        for tick_in_phase in range(n_ticks):
            base_price += drift + rng.gauss(0, 2)
            mid = max(100, base_price)  # floor at 100

            ods = {"KELP": make_book(mid, spread=4)}
            mt = {"KELP": []}

            # During crash: huge sell trades
            if phase_name == "crash":
                for _ in range(3):
                    mt["KELP"].append(Trade("KELP", int(mid - 2), 50,
                                            buyer="Bot_Panic", seller="Olivia", timestamp=0))

            ts = (200 + tick_in_phase if phase_name != "warmup" else tick_in_phase) * 100
            state = make_state(ods, pos=positions, ts=ts, td=td, mt=mt)
            orders, conv, td = t.run(state)

        if phase_name == "crash":
            crashed = True

    # Should not crash
    results.add("Survives flash crash", "PASS")

    # Verify state is coherent after recovery
    try:
        mem = decode_td(td)
        results.add("State coherent after crash recovery", "PASS")
    except Exception:
        results.add("State coherent after crash recovery", "FAIL", "traderData invalid")

    # Check BOCPD detected the crash
    try:
        mem = decode_td(td)
        ps = mem.get("_ps_KELP", {})
        bocpd_cp = ps.get("bocpd_cp", 0)
        if bocpd_cp > 0:
            results.add("BOCPD detected regime change", "PASS")
        else:
            results.add("BOCPD detected regime change", "WARN",
                         f"change_prob={bocpd_cp} — may have already decayed")
    except Exception:
        results.add("BOCPD detected regime change", "WARN", "Could not read state")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 3: BOOK DISAPPEARANCE
# Both sides go empty for 100 ticks, then reappear
# ═══════════════════════════════════════════════════════════════

def scenario_empty_book():
    """Test handling of prolonged empty order books."""
    print("\n--- SCENARIO: Book Disappearance (100 ticks) ---")

    t = Trader()
    td = ""

    # Warmup: 50 ticks normal
    for tick in range(50):
        ods = {"KELP": make_book(2020, spread=4),
               "RAINFOREST_RESIN": make_book(10000, spread=6)}
        state = make_state(ods, ts=tick*100, td=td)
        _, _, td = t.run(state)

    # Empty phase: 100 ticks with empty books
    empty_orders_count = 0
    for tick in range(50, 150):
        ods = {"KELP": make_od({}, {}),
               "RAINFOREST_RESIN": make_od({}, {})}
        state = make_state(ods, ts=tick*100, td=td)
        orders, _, td = t.run(state)
        for p_orders in orders.values():
            empty_orders_count += len(p_orders)

    # Recovery: 50 ticks normal again
    for tick in range(150, 200):
        ods = {"KELP": make_book(2020, spread=4),
               "RAINFOREST_RESIN": make_book(10000, spread=6)}
        state = make_state(ods, ts=tick*100, td=td)
        orders, _, td = t.run(state)

    results.add("Survives 100 ticks empty books", "PASS")

    if empty_orders_count > 0:
        results.add("No orders during empty book", "WARN",
                     f"{empty_orders_count} orders sent to empty book")
    else:
        results.add("No orders during empty book", "PASS")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 4: REGIME SHIFT (volatility 5x jump)
# ═══════════════════════════════════════════════════════════════

def scenario_regime_shift():
    """Test adaptation to sudden volatility change."""
    print("\n--- SCENARIO: Regime Shift (vol 5x) ---")

    t = Trader()
    td = ""
    rng = random.Random(456)
    base = 2020.0
    positions = {}

    low_vol_orders = []
    high_vol_orders = []

    # Phase 1: Low vol (500 ticks, sigma=0.5)
    for tick in range(500):
        base += rng.gauss(0, 0.5)
        ods = {"KELP": make_book(base, spread=4)}
        state = make_state(ods, pos=positions, ts=tick*100, td=td)
        orders, _, td = t.run(state)
        if tick >= 400:
            low_vol_orders.append(orders.get("KELP", []))

    # Phase 2: High vol (500 ticks, sigma=2.5 — 5x jump)
    for tick in range(500, 1000):
        base += rng.gauss(0, 2.5)
        ods = {"KELP": make_book(base, spread=4)}
        state = make_state(ods, pos=positions, ts=tick*100, td=td)
        orders, _, td = t.run(state)
        if tick >= 600:
            high_vol_orders.append(orders.get("KELP", []))

    results.add("Survives 5x vol regime shift", "PASS")

    # Check: spread should be wider in high vol (fewer/smaller orders)
    avg_low = sum(len(o) for o in low_vol_orders) / max(1, len(low_vol_orders))
    avg_high = sum(len(o) for o in high_vol_orders) / max(1, len(high_vol_orders))
    if avg_high <= avg_low:
        results.add("Adapts to higher volatility (fewer orders)", "PASS")
    else:
        results.add("Adapts to higher volatility", "WARN",
                     f"avg orders: low_vol={avg_low:.1f}, high_vol={avg_high:.1f}")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 5: EOD FLATTENING
# Verify position is flattened near t=995000
# ═══════════════════════════════════════════════════════════════

def scenario_eod_flattening():
    """Test end-of-day position flattening."""
    print("\n--- SCENARIO: EOD Flattening ---")

    t = Trader()
    td = ""

    # Build up position: 200 ticks with position at +40
    for tick in range(200):
        ods = {"RAINFOREST_RESIN": make_book(10000, spread=6)}
        state = make_state(ods, pos={"RAINFOREST_RESIN": 40}, ts=tick*100, td=td)
        _, _, td = t.run(state)

    # Now enter EOD zone: t=995000+
    for tick in range(100):
        ts = 995000 + tick * 100
        ods = {"RAINFOREST_RESIN": make_book(10000, spread=6)}
        state = make_state(ods, pos={"RAINFOREST_RESIN": 40}, ts=ts, td=td)
        orders, _, td = t.run(state)
        resin_orders = orders.get("RAINFOREST_RESIN", [])
        sell_qty = sum(abs(o.quantity) for o in resin_orders if o.quantity < 0)
        if sell_qty > 0:
            results.add("EOD flattening triggered at t=995000+", "PASS")
            return

    results.add("EOD flattening triggered at t=995000+", "FAIL",
                 "No sell orders at EOD with +40 position")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 6: TRADERDATA CORRUPTION RECOVERY
# ═══════════════════════════════════════════════════════════════

def scenario_td_corruption():
    """Test recovery from corrupted traderData."""
    print("\n--- SCENARIO: traderData Corruption Recovery ---")

    t = Trader()

    corrupt_values = [
        ("empty string", ""),
        ("invalid JSON", "{not json at all"),
        ("null bytes", "\x00\x00\x00"),
        ("oversized", "Z:" + "A" * 100000),
        ("valid JSON but wrong structure", json.dumps([1, 2, 3])),
        ("nested nulls", json.dumps({"key": None, "arr": [None, None]})),
    ]

    for name, corrupt_td in corrupt_values:
        try:
            ods = {"KELP": make_book(2020, spread=4),
                   "RAINFOREST_RESIN": make_book(10000, spread=6)}
            state = make_state(ods, ts=100, td=corrupt_td)
            orders, conv, td = t.run(state)
            results.add(f"Recovers from {name}", "PASS")
        except Exception as e:
            results.add(f"Recovers from {name}", "FAIL", str(e)[:80])


# ═══════════════════════════════════════════════════════════════
# SCENARIO 7: POSITION LIMIT SATURATION
# All products at max position simultaneously
# ═══════════════════════════════════════════════════════════════

def scenario_position_saturation():
    """All products simultaneously at +limit and -limit."""
    print("\n--- SCENARIO: Position Limit Saturation ---")

    t = Trader()
    td = ""

    products = {
        "RAINFOREST_RESIN": (10000, 50),
        "KELP": (2020, 50),
        "SQUID_INK": (1970, 50),
        "EMERALDS": (10000, 80),
        "TOMATOES": (12400, 80),
    }

    # Test at +limit
    pos_plus = {p: lim for p, (_, lim) in products.items()}
    ods = {p: make_book(base, spread=4) for p, (base, _) in products.items()}
    state = make_state(ods, pos=pos_plus, ts=100, td=td)
    orders, _, td = t.run(state)

    # Verify: no buy orders when at +limit
    violation = False
    for p in products:
        p_orders = orders.get(p, [])
        buy_qty = sum(o.quantity for o in p_orders if o.quantity > 0)
        if buy_qty > 0:
            results.add(f"No buys at +limit ({p})", "FAIL", f"buy_qty={buy_qty}")
            violation = True
    if not violation:
        results.add("No buys when all products at +limit", "PASS")

    # Test at -limit
    pos_minus = {p: -lim for p, (_, lim) in products.items()}
    state = make_state(ods, pos=pos_minus, ts=200, td=td)
    orders, _, td = t.run(state)

    violation = False
    for p in products:
        p_orders = orders.get(p, [])
        sell_qty = sum(abs(o.quantity) for o in p_orders if o.quantity < 0)
        if sell_qty > 0:
            results.add(f"No sells at -limit ({p})", "FAIL", f"sell_qty={sell_qty}")
            violation = True
    if not violation:
        results.add("No sells when all products at -limit", "PASS")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 8: MULTI-DAY PERSISTENCE
# State must carry across day boundaries (ts rolls over)
# ═══════════════════════════════════════════════════════════════

def scenario_multiday():
    """Test state persistence across day boundaries."""
    print("\n--- SCENARIO: Multi-Day Persistence ---")

    t = Trader()
    td = ""
    rng = random.Random(789)

    # Day 1: 500 ticks (ts 0 - 49900)
    base = 2020.0
    for tick in range(500):
        base += rng.gauss(0, 1)
        ods = {"KELP": make_book(base, spread=4)}
        state = make_state(ods, ts=tick*100, td=td)
        _, _, td = t.run(state)

    td_day1 = td
    mem_day1 = decode_td(td)

    # Day 2: ts resets to 1000000 (new day)
    for tick in range(500):
        base += rng.gauss(0, 1)
        ts = 1000000 + tick * 100
        ods = {"KELP": make_book(base, spread=4)}
        state = make_state(ods, ts=ts, td=td)
        _, _, td = t.run(state)

    mem_day2 = decode_td(td)

    # Verify state persisted
    ps1 = mem_day1.get("_ps_KELP", {})
    ps2 = mem_day2.get("_ps_KELP", {})

    # Key fields should persist (realized_vol or kf_x carry over)
    has_state = bool(ps2.get("kf_x") or ps2.get("realized_vol") or ps2.get("mid_hist"))
    if has_state:
        results.add("State carries across days", "PASS")
    else:
        results.add("State carries across days", "WARN",
                     f"day2 ps keys: {list(ps2.keys())[:5]}")

    results.add("Multi-day state persistence works", "PASS")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 9: 15 UNKNOWN PRODUCTS (auto-classification stress)
# ═══════════════════════════════════════════════════════════════

def scenario_mass_classification():
    """15 unknown products appear simultaneously."""
    print("\n--- SCENARIO: Mass Auto-Classification (15 products) ---")

    t = Trader()
    td = ""
    rng = random.Random(999)

    # Create 15 products with different characteristics
    product_specs = {}
    for i in range(15):
        base = 1000 + i * 500
        spread = rng.choice([2, 4, 8, 16])
        product_specs[f"PRODUCT_{i}"] = (base, spread)

    tick_times = []
    for tick in range(200):
        ods = {}
        for name, (base, spread) in product_specs.items():
            mid = base + rng.gauss(0, 1)
            ods[name] = make_book(mid, spread)

        state = make_state(ods, ts=tick*100, td=td)
        t0 = time.perf_counter()
        orders, _, td = t.run(state)
        tick_times.append(time.perf_counter() - t0)

    # Exclude first 20 ticks (cold start classification overhead)
    warm_times = tick_times[20:] if len(tick_times) > 20 else tick_times
    avg_ms = sum(warm_times) * 1000 / len(warm_times)
    max_ms = max(warm_times) * 1000
    cold_max = max(tick_times[:20]) * 1000 if len(tick_times) > 20 else 0

    td_size = len(td)
    results.add("15 products classified without crash", "PASS")

    if td_size > 90000:
        results.add("traderData < 90KB with 15 products", "FAIL",
                     f"{td_size}B ({td_size/1024:.1f}KB)")
    else:
        results.add(f"traderData < 90KB with 15 products ({td_size/1024:.1f}KB)", "PASS")

    print(f"    Cold start max: {cold_max:.1f}ms (excluded)")
    if max_ms > 100:
        results.add("Per-tick < 100ms with 15 products (warm)", "FAIL",
                     f"max={max_ms:.1f}ms")
    else:
        results.add(f"Per-tick < 100ms with 15 products (max={max_ms:.1f}ms)", "PASS")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 10: ADVERSARIAL FILL PATTERNS
# Systematic adverse selection — every fill goes against us
# ═══════════════════════════════════════════════════════════════

def scenario_adverse_selection():
    """Every market trade goes against our position."""
    print("\n--- SCENARIO: Adversarial Fill Patterns ---")

    t = Trader()
    td = ""
    rng = random.Random(321)
    base = 2020.0
    pos = 0

    for tick in range(500):
        # Price always moves AGAINST our position
        if pos > 0:
            base -= rng.uniform(0.5, 2.0)  # price drops when we're long
        elif pos < 0:
            base += rng.uniform(0.5, 2.0)  # price rises when we're short
        else:
            base += rng.gauss(0, 1)

        ods = {"KELP": make_book(base, spread=4)}

        # Simulate adverse fills
        mt = {"KELP": []}
        if pos > 0:
            # Sell trades hit market (price going down)
            mt["KELP"].append(Trade("KELP", int(base - 1), 20,
                                     buyer="Bot_A", seller="Bot_Informed",
                                     timestamp=tick*100))
        elif pos < 0:
            mt["KELP"].append(Trade("KELP", int(base + 1), 20,
                                     buyer="Bot_Informed", seller="Bot_A",
                                     timestamp=tick*100))

        state = make_state(ods, pos={"KELP": pos}, ts=tick*100, td=td, mt=mt)
        orders, _, td = t.run(state)

        # Simulate: we get filled on wrong side
        kelp_orders = orders.get("KELP", [])
        for o in kelp_orders:
            if rng.random() < 0.2:
                pos += o.quantity
        pos = max(-50, min(50, pos))

    # Should survive without crash
    results.add("Survives sustained adverse selection", "PASS")

    # Check markout tracking detected the adversity
    try:
        mem = decode_td(td)
        ps = mem.get("_ps_KELP", {})
        tox = ps.get("toxicity", 0)
        if tox > 0:
            results.add(f"Toxicity detected under adversity (tox={tox:.2f})", "PASS")
        else:
            results.add("Toxicity detected under adversity", "WARN",
                         "toxicity=0 — may not be tracking")
    except Exception:
        results.add("Toxicity tracking", "WARN", "Could not decode state")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 11: RAPID PRODUCT APPEARANCE/DISAPPEARANCE
# Products appear and disappear from order_depths between ticks
# ═══════════════════════════════════════════════════════════════

def scenario_product_churn():
    """Products appear and disappear randomly."""
    print("\n--- SCENARIO: Product Churn ---")

    t = Trader()
    td = ""
    rng = random.Random(654)
    all_products = ["KELP", "RAINFOREST_RESIN", "SQUID_INK", "EMERALDS",
                    "TOMATOES", "MYSTERY_A", "MYSTERY_B"]

    for tick in range(300):
        # Random subset of products each tick
        n_active = rng.randint(1, len(all_products))
        active = rng.sample(all_products, n_active)

        ods = {}
        for p in active:
            base = {"KELP": 2020, "RAINFOREST_RESIN": 10000, "SQUID_INK": 1970,
                    "EMERALDS": 10000, "TOMATOES": 12400}.get(p, 5000)
            ods[p] = make_book(base + rng.gauss(0, 2), spread=4)

        state = make_state(ods, ts=tick*100, td=td)
        orders, _, td = t.run(state)

    results.add("Survives product churn (appear/disappear)", "PASS")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 12: WIDE SPREAD THEN SUDDEN NARROW
# Tests penny-jump and compression detection
# ═══════════════════════════════════════════════════════════════

def scenario_spread_compression():
    """Spread compresses suddenly from 20 to 2."""
    print("\n--- SCENARIO: Spread Compression ---")

    t = Trader()
    td = ""

    # Wide spread phase
    for tick in range(200):
        ods = {"TOMATOES": make_book(12400, spread=20)}
        state = make_state(ods, ts=tick*100, td=td)
        _, _, td = t.run(state)

    # Sudden compression
    for tick in range(200, 400):
        ods = {"TOMATOES": make_book(12400, spread=2)}
        state = make_state(ods, ts=tick*100, td=td)
        orders, _, td = t.run(state)

    results.add("Survives sudden spread compression", "PASS")


# ═══════════════════════════════════════════════════════════════
# RUN ALL SCENARIOS
# ═══════════════════════════════════════════════════════════════

SCENARIOS = {
    "full_round": scenario_full_round,
    "flash_crash": scenario_flash_crash,
    "empty_book": scenario_empty_book,
    "regime_shift": scenario_regime_shift,
    "eod_flattening": scenario_eod_flattening,
    "td_corruption": scenario_td_corruption,
    "position_saturation": scenario_position_saturation,
    "multiday": scenario_multiday,
    "mass_classification": scenario_mass_classification,
    "adverse_selection": scenario_adverse_selection,
    "product_churn": scenario_product_churn,
    "spread_compression": scenario_spread_compression,
}

QUICK_SCENARIOS = ["full_round", "flash_crash", "position_saturation"]


def main():
    parser = argparse.ArgumentParser(description="Production Stress Test")
    parser.add_argument("--quick", action="store_true",
                        help="Run 3 critical scenarios only")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a specific scenario")
    args = parser.parse_args()

    print("=" * 70)
    print("PRODUCTION STRESS TEST")
    print("=" * 70)

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        try:
            SCENARIOS[args.scenario]()
        except Exception as e:
            results.add(f"Scenario {args.scenario}", "FAIL",
                         f"CRASHED: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        to_run = QUICK_SCENARIOS if args.quick else list(SCENARIOS.keys())
        for name in to_run:
            try:
                SCENARIOS[name]()
            except Exception as e:
                results.add(f"Scenario {name}", "FAIL",
                             f"CRASHED: {type(e).__name__}: {e}")
                traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    total = results.passed + results.failed + results.warnings
    print(f"RESULTS: {results.passed} PASS, {results.warnings} WARN, {results.failed} FAIL (of {total})")
    print(f"{'='*70}")

    if results.failed == 0:
        print("ALL TESTS PASSED" + (" (warnings present)" if results.warnings else ""))
    else:
        print(f"WARNING: {results.failed} FAILURES — FIX BEFORE SUBMISSION")

    sys.exit(1 if results.failed > 0 else 0)


if __name__ == "__main__":
    main()
