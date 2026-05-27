"""R5 trader v1 — pair-spread z-score + single-asset z-score MR.

Alphas (validated on R5 day 2/3/4):
- Tier 1 #1: PISTACHIO + STRAWBERRY - 2*RASPBERRY (window=500)
- Tier 1 #2: CHOCOLATE + VANILLA (window=200)
- Tier 1 #3: (PEBBLES_XS+S+M+L) + 2*PEBBLES_XL (window=500)
- Tier 2: TRANSLATOR_ASTRO_BLACK, TRANSLATOR_VOID_BLUE, MICROCHIP_CIRCLE (single-asset MR, window=500)

All other 41 R5 products: do_nothing.
"""
import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


# ============================================================================
# CONFIG
# ============================================================================

# Position limit for ALL R5 products = 10 (per wiki)
POS_LIMIT = 10

# Tier 1 — pair / basket spread alphas
SPREADS_CONFIG = {
    # v3: Snackpack alpha 1 disabled — backtest showed -$6K day 2 vs +$3-4K days 3/4 = ~break-even net,
    # adding day-2 tail risk for negligible EV. Re-enable if more days of data validate it.
    # CHOC+VAN: std=42 too tight vs slippage.
    # Pebbles basket: structural mismatch — XL alone is the alpha; basket legs eat into it.
}

# Tier 2 — single-asset z-score reversion
# v3: 9 products selected from systematic grid (z_in × mom_thresh). All-positive across day 2/3/4 in realistic backtest.
# Configs vary per product; higher z_in (2.0+) reduces slippage tax for weak per-trade edges.
SINGLE_MR_CONFIG = {
    # v3.2 alphas (8) — proven in INTEGRATED backtest with realistic fill model
    "PEBBLES_XL":            {"window": 500, "z_in": 1.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.5},  # primary alpha
    "PEBBLES_M":             {"window": 500, "z_in": 2.0, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 2.0},
    "PEBBLES_S":             {"window": 500, "z_in": 2.0, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.5},
    "MICROCHIP_RECTANGLE":   {"window": 500, "z_in": 2.0, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.5},
    "TRANSLATOR_ASTRO_BLACK":{"window": 500, "z_in": 2.0, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 2.0},
    "TRANSLATOR_VOID_BLUE":  {"window": 500, "z_in": 1.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.0},
    "SLEEP_POD_NYLON":       {"window": 500, "z_in": 2.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 2.0},
    "MICROCHIP_CIRCLE":      {"window": 500, "z_in": 1.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.0},
    # v3.5: re-add new alphas now that mem-bloat issue is fixed (momentum derived from main buffer)
    "MICROCHIP_SQUARE":          {"window": 500,  "z_in": 1.8, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.0},
    "TRANSLATOR_ECLIPSE_CHARCOAL":{"window": 1000, "z_in": 2.5, "z_out": 0.3, "max_pos": 10},
    "ROBOT_DISHES":              {"window": 1000, "z_in": 2.2, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 2.0},
    "MICROCHIP_TRIANGLE":        {"window": 1000, "z_in": 2.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 2.5},
    "PEBBLES_XS":                {"window": 1000, "z_in": 2.5, "z_out": 0.3, "max_pos": 10},
    "UV_VISOR_RED":              {"window": 1000, "z_in": 1.8, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 0.8},
    "UV_VISOR_MAGENTA":          {"window": 500,  "z_in": 3.0, "z_out": 0.3, "max_pos": 10},
    "ROBOT_IRONING":             {"window": 1000, "z_in": 2.2, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.0},
}

# Products in spreads (these are managed by spread strategy, not single MR)
_SPREAD_PRODUCTS = set()
for _sc in SPREADS_CONFIG.values():
    for _p, _, _ in _sc["legs"]:
        _SPREAD_PRODUCTS.add(_p)

# All 50 R5 products — anything not in spreads or single MR sits out
ALL_R5_PRODUCTS = [
    "GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS","GALAXY_SOUNDS_SOLAR_FLAMES",
    "SLEEP_POD_SUEDE","SLEEP_POD_LAMB_WOOL","SLEEP_POD_POLYESTER","SLEEP_POD_NYLON","SLEEP_POD_COTTON",
    "MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_SQUARE","MICROCHIP_RECTANGLE","MICROCHIP_TRIANGLE",
    "PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL",
    "ROBOT_VACUUMING","ROBOT_MOPPING","ROBOT_DISHES","ROBOT_LAUNDRY","ROBOT_IRONING",
    "UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_MAGENTA",
    "TRANSLATOR_SPACE_GRAY","TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_VOID_BLUE",
    "PANEL_1X2","PANEL_2X2","PANEL_1X4","PANEL_2X4","PANEL_4X4",
    "OXYGEN_SHAKE_MORNING_BREATH","OXYGEN_SHAKE_EVENING_BREATH","OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC",
    "SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY",
]


# ============================================================================
# Helpers
# ============================================================================

def best_bid_ask(od):
    """Returns (best_bid, best_ask, mid). None values if book is empty."""
    if not od or not od.buy_orders or not od.sell_orders:
        return None, None, None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, (bb + ba) / 2.0


def update_rolling_stats(mem, key, x, window):
    """Incremental rolling mean/std. Stored in mem under key + suffixes."""
    buf = mem.get(f"{key}_buf", [])
    buf.append(x)
    if len(buf) > window:
        buf = buf[-window:]
    mem[f"{key}_buf"] = buf
    if len(buf) < max(20, window // 4):  # need some warmup
        return None, None
    m = sum(buf) / len(buf)
    var = sum((v - m) ** 2 for v in buf) / max(1, len(buf) - 1)
    sd = math.sqrt(max(var, 1e-10))
    return m, sd


def validate_orders(orders, pos, limit):
    """Trim orders so total buy/sell volume can't push position past +/- limit.

    Keeps takes (most aggressive prices) before trimming makes.
    """
    if not orders:
        return []
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]

    max_buy = limit - pos  # remaining room to buy
    max_sell = limit + pos  # remaining room to sell

    # Sort buys descending (highest price = most aggressive); same for sells ascending (lowest = most aggressive)
    buys.sort(key=lambda o: -o.price)
    sells.sort(key=lambda o: o.price)

    out = []
    used_buy = 0
    for o in buys:
        room = max_buy - used_buy
        if room <= 0:
            break
        if o.quantity <= room:
            out.append(o)
            used_buy += o.quantity
        else:
            out.append(Order(o.symbol, o.price, room))
            used_buy += room
            break
    used_sell = 0
    for o in sells:
        room = max_sell - used_sell
        if room <= 0:
            break
        if -o.quantity <= room:
            out.append(o)
            used_sell += -o.quantity
        else:
            out.append(Order(o.symbol, o.price, -room))
            used_sell += room
            break
    return out


# ============================================================================
# Strategy: spread z-score reversion
# ============================================================================

def run_spread(name, cfg, state, mem, result):
    """Run a multi-leg spread z-score reversion. Writes orders to result dict for each leg."""
    legs = cfg["legs"]
    window = cfg.get("window", 500)
    z_in = cfg.get("z_in", 1.5)
    z_out = cfg.get("z_out", 0.3)

    # 1. Compute current spread from mids
    mids = {}
    for prod, _, _ in legs:
        od = state.order_depths.get(prod)
        bb, ba, mid = best_bid_ask(od)
        if mid is None:
            return  # missing book = skip this tick
        mids[prod] = (bb, ba, mid)

    spread_val = sum(mids[p][2] * w for p, w, _ in legs)

    # 2. Update rolling stats
    m, sd = update_rolling_stats(mem, f"sp_{name}", spread_val, window)
    if m is None:
        return  # warmup
    z = (spread_val - m) / sd if sd > 0 else 0

    # 3. Determine target position state (with hysteresis)
    cur_state = mem.get(f"sp_{name}_state", 0)  # +1 long spread, -1 short spread, 0 flat
    new_state = cur_state
    if cur_state == 0:
        if z > z_in:
            new_state = -1  # spread too high → short the spread
        elif z < -z_in:
            new_state = +1
    else:
        # Exit ONLY when spread crosses zero to the other side or |z| < z_out
        if cur_state == -1 and z < z_out:
            new_state = 0
        elif cur_state == +1 and z > -z_out:
            new_state = 0
    mem[f"sp_{name}_state"] = new_state

    # 4. EVERY TICK: rebalance toward target position (handles partial fills + drift)
    for prod, w, max_contracts in legs:
        target_pos = new_state * (1 if w > 0 else -1) * max_contracts
        cur_pos = state.position.get(prod, 0)
        delta = target_pos - cur_pos
        if delta == 0:
            continue
        bb, ba, _ = mids[prod]
        # When entering (target != 0), cross the spread to fill fast
        # When exiting (target == 0), also cross to flatten quickly
        if delta > 0:
            order = Order(prod, ba, delta)
        else:
            order = Order(prod, bb, delta)
        validated = validate_orders([order], cur_pos, POS_LIMIT)
        if validated:
            result.setdefault(prod, []).extend(validated)


# ============================================================================
# Strategy: single-asset z-score reversion
# ============================================================================

def run_single_mr(prod, cfg, state, mem, result):
    od = state.order_depths.get(prod)
    bb, ba, mid = best_bid_ask(od)
    if mid is None:
        return
    window = cfg.get("window", 500)
    z_in = cfg.get("z_in", 1.5)
    z_out = cfg.get("z_out", 0.3)
    max_pos = cfg.get("max_pos", 10)
    mom_window = cfg.get("mom_window", 0)
    mom_thresh = cfg.get("mom_thresh", None)

    m, sd = update_rolling_stats(mem, f"sm_{prod}", mid, window)
    if m is None:
        return
    z = (mid - m) / sd if sd > 0 else 0

    # Momentum filter: skip new entries when in a strong trend
    # Derive momentum from MAIN buffer (no separate buffer — saves ~50% mem)
    mom_blocked = False
    if mom_window > 0 and mom_thresh is not None:
        main_buf = mem.get(f"sm_{prod}_buf", [])
        if len(main_buf) >= mom_window:
            past_mid = main_buf[-mom_window]
            mom = (mid - past_mid) / max(sd, 1.0)
            if abs(mom) > mom_thresh:
                mom_blocked = True

    cur_state = mem.get(f"sm_{prod}_state", 0)
    new_state = cur_state
    if cur_state == 0:
        # Only enter if NOT blocked by momentum
        if not mom_blocked:
            if z > z_in:
                new_state = -1
            elif z < -z_in:
                new_state = +1
    else:
        # Always allow exit (close out positions even if trending)
        if cur_state == -1 and z < z_out:
            new_state = 0
        elif cur_state == +1 and z > -z_out:
            new_state = 0
    mem[f"sm_{prod}_state"] = new_state

    # Rebalance every tick (handles partial fills + position drift)
    target_pos = new_state * max_pos
    cur_pos = state.position.get(prod, 0)
    delta = target_pos - cur_pos
    if delta == 0:
        return
    # Cap order size to level-1 volume to AVOID walking through book depth (which costs slippage)
    # Patient fills across multiple ticks > eating level-2/3 prices
    if delta > 0:
        # buying: cap to level-1 ask volume
        l1_vol = abs(od.sell_orders.get(ba, 0)) if od.sell_orders else 0
        size = min(delta, max(1, l1_vol)) if l1_vol > 0 else delta
        order = Order(prod, ba, size)
    else:
        # selling: cap to level-1 bid volume
        l1_vol = od.buy_orders.get(bb, 0) if od.buy_orders else 0
        size = max(delta, -max(1, l1_vol)) if l1_vol > 0 else delta
        order = Order(prod, bb, size)
    validated = validate_orders([order], cur_pos, POS_LIMIT)
    if validated:
        result.setdefault(prod, []).extend(validated)


# ============================================================================
# Trader
# ============================================================================

class Trader:
    def run(self, state: TradingState):
        # Restore mem from traderData
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}

        result: Dict[str, List[Order]] = {}

        # 1. Run all spread strategies
        for name, cfg in SPREADS_CONFIG.items():
            try:
                run_spread(name, cfg, state, mem, result)
            except Exception as e:
                print(f"ERR_SPREAD|{name}|{type(e).__name__}|{e}")

        # 2. Run single-asset MR strategies
        for prod, cfg in SINGLE_MR_CONFIG.items():
            if prod in _SPREAD_PRODUCTS:
                continue  # spread takes priority
            try:
                run_single_mr(prod, cfg, state, mem, result)
            except Exception as e:
                print(f"ERR_SINGLE|{prod}|{type(e).__name__}|{e}")

        # 3. All other products: do_nothing (no orders)

        # Persist mem (keep under 90KB)
        traderData = json.dumps(mem, separators=(",", ":"))
        # Pruning safety: only triggers near 85KB. Each strategy needs >= window for z-score AND
        # >= mom_window for momentum, so don't prune below 1000 entries (sufficient for w=1000 + mom_window=200).
        if len(traderData) > 85000:
            for k in list(mem.keys()):
                if k.endswith("_buf") and isinstance(mem[k], list) and len(mem[k]) > 1000:
                    mem[k] = mem[k][-1000:]
            traderData = json.dumps(mem, separators=(",", ":"))

        return result, 0, traderData
