"""Quick OBI strategy backtest with realistic bid/ask fills.

Strategy: target_pos = -sign(OBI) * scale_factor * abs(OBI)^power
Hold position; only re-quote when target_pos changes meaningfully.
"""
import math
from collections import defaultdict, deque

# Top OBI candidates from cross-day stable signal
CANDIDATES = [
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
    "UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_MAGENTA",
    "GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS","GALAXY_SOUNDS_SOLAR_FLAMES",
    "PANEL_1X2","PANEL_2X4",
    "OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC",
    "TRANSLATOR_ASTRO_BLACK","TRANSLATOR_VOID_BLUE","TRANSLATOR_GRAPHITE_MIST",
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
                bv = sum(int(p[idx[f"bid_volume_{k}"]]) for k in [1,2,3] if p[idx[f"bid_volume_{k}"]])
                av = sum(int(p[idx[f"ask_volume_{k}"]]) for k in [1,2,3] if p[idx[f"ask_volume_{k}"]])
                out[prod].append((ts, mid, bid1, ask1, bv, av))
            except: pass
    for k in out: out[k] = sorted(out[k])
    return out

def obi_backtest(track, obi_thresh=0.3, max_pos=10):
    """Strategy: when OBI > thresh, target = -max_pos (short). When OBI < -thresh, target = +max_pos.
    Otherwise: flat.

    Cross spread on entry/exit (use bid/ask prices)."""
    pnl = 0.0
    pos = 0
    # entry tracking for PnL (FIFO style)
    entries = deque()  # (price, qty)
    fills = 0
    for i in range(len(track)):
        ts, mid, bid, ask, bv, av = track[i]
        if bid is None or ask is None: continue
        if bv + av == 0: continue
        obi = (bv - av) / (bv + av)
        if obi > obi_thresh:
            target = -max_pos
        elif obi < -obi_thresh:
            target = +max_pos
        else:
            target = 0
        delta = target - pos
        if delta == 0: continue
        # execute at adverse side
        if delta > 0:
            # buying delta units at ask
            fill_price = ask
            for _ in range(abs(delta)):
                # FIFO: if we have a short entry, close it
                if pos < 0 and entries and entries[0][1] < 0:
                    e_price, e_qty = entries.popleft()
                    pnl += (e_price - fill_price)  # short P&L = entry - exit
                    pos += 1
                    if e_qty < -1:  # split if more than 1
                        entries.appendleft((e_price, e_qty + 1))
                else:
                    entries.append((fill_price, +1))
                    pos += 1
                fills += 1
        else:
            fill_price = bid
            for _ in range(abs(delta)):
                if pos > 0 and entries and entries[0][1] > 0:
                    e_price, e_qty = entries.popleft()
                    pnl += (fill_price - e_price)
                    pos -= 1
                    if e_qty > 1:
                        entries.appendleft((e_price, e_qty - 1))
                else:
                    entries.append((fill_price, -1))
                    pos -= 1
                fills += 1
    # Close remaining position at last mid
    last_mid = track[-1][1]
    while entries:
        e_price, e_qty = entries.popleft()
        if e_qty > 0:
            pnl += (last_mid - e_price) * e_qty
        else:
            pnl += (e_price - last_mid) * (-e_qty)
    return pnl, fills

def main():
    days = {d: load_with_books(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    # Test multiple OBI thresholds
    print("="*100)
    print("OBI BACKTEST: target = -sign(OBI)*max_pos when |OBI| > threshold")
    print("="*100)
    for thresh in [0.2, 0.3, 0.5, 0.7]:
        print(f"\n--- obi_thresh = {thresh} ---")
        all_total = 0
        for d in [2,3,4]:
            day_total = 0
            day_fills = 0
            for p in CANDIDATES:
                if p not in days[d]: continue
                pnl, fills = obi_backtest(days[d][p], obi_thresh=thresh, max_pos=10)
                day_total += pnl
                day_fills += fills
            print(f"  Day {d}: total_pnl={day_total:+8.0f}  total_fills={day_fills}")
            all_total += day_total
        print(f"  3-day total: {all_total:+8.0f}")

    # Per-product PnL at best threshold
    print("\n" + "="*100)
    print("PER-PRODUCT OBI BACKTEST PnL @ obi_thresh=0.3 (3-day sum)")
    print("="*100)
    rows = []
    for p in CANDIDATES:
        per_day = []
        for d in [2,3,4]:
            if p not in days[d]: per_day.append(0); continue
            pnl, _ = obi_backtest(days[d][p], obi_thresh=0.3, max_pos=10)
            per_day.append(pnl)
        rows.append((p, sum(per_day), per_day))
    rows.sort(key=lambda x: -x[1])
    for p, total, per_day in rows:
        all_pos = "Y" if all(x > 0 for x in per_day) else " "
        print(f"  {p:<32} 3day_total={total:+8.0f}  per_day=[{', '.join(f'{x:+5.0f}' for x in per_day)}]  all_pos={all_pos}")

if __name__ == "__main__":
    main()
