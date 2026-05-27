"""Systematic test: single-asset MR + momentum filter on all 50 products.

For each product, test multiple z_in/mom_thresh combinations and find the best stable config.
Filter: ALL 3 days positive AND min daily PnL > $500."""
import math
from collections import defaultdict, deque

ALL_PRODUCTS = [
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

def load_with_books(path):
    out = defaultdict(list)
    with open(path,"r") as f:
        h = f.readline().strip().split(";")
        idx = {c: i for i,c in enumerate(h)}
        for ln in f:
            p = ln.strip().split(";")
            if len(p)<len(h): continue
            try:
                prod = p[idx["product"]]
                ts = int(p[idx["timestamp"]])
                mid = float(p[idx["mid_price"]])
                bid1 = int(p[idx["bid_price_1"]]) if p[idx["bid_price_1"]] else None
                ask1 = int(p[idx["ask_price_1"]]) if p[idx["ask_price_1"]] else None
                out[prod].append((ts, mid, bid1, ask1))
            except: pass
    for k in out: out[k] = sorted(out[k])
    return out

def mr_backtest(track, window=500, z_in=1.5, z_out=0.3, mom_window=200, mom_thresh=None, max_pos=10):
    """Single-asset MR with momentum filter. Realistic bid/ask fills."""
    pnl = 0.0
    pos = 0
    entry_price = 0
    fills = 0
    buf = deque(); s_sum = 0.0; s_sumsq = 0.0
    for i in range(len(track)):
        ts, mid, bid, ask = track[i]
        if bid is None or ask is None: continue
        buf.append(mid); s_sum += mid; s_sumsq += mid*mid
        if len(buf) > window:
            old = buf.popleft(); s_sum -= old; s_sumsq -= old*old
        if len(buf) < window: continue
        m = s_sum/len(buf)
        var = (s_sumsq - len(buf)*m*m)/(len(buf)-1)
        sd = math.sqrt(max(var, 1e-10))
        z = (mid - m)/sd
        # momentum
        mom_blocked = False
        if mom_thresh is not None and i >= mom_window:
            past_mid = track[i - mom_window][1]
            mom = (mid - past_mid) / max(sd, 1)
            if abs(mom) > mom_thresh:
                mom_blocked = True
        cur_state = pos
        new_state = pos
        if pos == 0:
            if not mom_blocked:
                if z > z_in: new_state = -1
                elif z < -z_in: new_state = +1
        else:
            if pos == -1 and z < z_out: new_state = 0
            elif pos == +1 and z > -z_out: new_state = 0
        if new_state == cur_state: continue
        target = new_state * max_pos
        delta = target - pos
        if delta > 0:
            fill_price = ask
        else:
            fill_price = bid
        if cur_state == 0:
            entry_price = fill_price
            pos = new_state
        else:
            pnl += pos * (fill_price - entry_price) * max_pos
            pos = 0
            fills += 1
            if new_state != 0:
                # immediate re-entry
                if new_state > 0: entry_price = ask
                else: entry_price = bid
                pos = new_state
        fills += 1
    if pos != 0:
        last_mid = track[-1][1]
        pnl += pos * (last_mid - entry_price) * max_pos
    return pnl, fills

def main():
    days = {d: load_with_books(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    # Try a grid of (z_in, mom_thresh, window) for each product
    results = []
    grids = [
        (1.5, 1.0, 500),
        (1.5, 1.5, 500),
        (1.5, 2.0, 500),
        (2.0, 1.5, 500),
        (2.0, 2.0, 500),
        (2.5, 2.0, 500),
        (1.5, None, 500),  # no momentum filter
    ]
    print(f"Testing {len(grids)} configs × {len(ALL_PRODUCTS)} products...")
    for p in ALL_PRODUCTS:
        best_config = None
        best_total = -1e18
        best_per_day = None
        for z_in, mom_thresh, window in grids:
            per_day = []
            for d in [2,3,4]:
                if p not in days[d]:
                    per_day.append(0); continue
                pnl, _ = mr_backtest(days[d][p], window=window, z_in=z_in, mom_thresh=mom_thresh)
                per_day.append(pnl)
            total = sum(per_day)
            min_day = min(per_day)
            # require ALL POSITIVE + min > 500
            all_pos = all(x > 500 for x in per_day)
            if all_pos and total > best_total:
                best_total = total
                best_config = (z_in, mom_thresh, window)
                best_per_day = per_day
        if best_config:
            results.append((p, best_total, best_config, best_per_day))

    # Sort by total
    results.sort(key=lambda x: -x[1])
    print(f"\n{'='*120}")
    print(f"PRODUCTS WITH ALL-POSITIVE-DAYS CONFIGS (min daily PnL > $500)")
    print(f"{'='*120}")
    grand = 0
    for p, total, cfg, pd in results:
        z_in, mom_thresh, window = cfg
        mt_str = f"mom={mom_thresh}" if mom_thresh else "mom=OFF"
        print(f"  {p:<32} 3day={total:+8.0f}  per_day=[{', '.join(f'{x:+5.0f}' for x in pd)}]  z_in={z_in} {mt_str}")
        grand += total
    print(f"\n  TOTAL across all qualifying products: {grand:+.0f} (3-day)")
    print(f"  Per-day average: {grand/3:+.0f}")

if __name__ == "__main__":
    main()
