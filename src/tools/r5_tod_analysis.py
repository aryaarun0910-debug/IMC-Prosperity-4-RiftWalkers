"""Time-of-day analysis: when does PnL concentrate? Are there 'dead zones' where strategies bleed?

Build a per-tick attribution of v3.2 backtest, then bin into 1000-tick windows.
"""
import sys, os, math
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_systematic_mr import load_with_books, mr_backtest

# Top alphas
ALPHAS = {
    "PEBBLES_XL":            {"window": 500, "z_in": 1.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.5},
    "PEBBLES_M":             {"window": 500, "z_in": 2.0, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 2.0},
    "PEBBLES_S":             {"window": 500, "z_in": 2.0, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.5},
    "TRANSLATOR_ASTRO_BLACK":{"window": 500, "z_in": 2.0, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 2.0},
    "TRANSLATOR_VOID_BLUE":  {"window": 500, "z_in": 1.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.0},
    "MICROCHIP_CIRCLE":      {"window": 500, "z_in": 1.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.0},
    "MICROCHIP_RECTANGLE":   {"window": 500, "z_in": 2.0, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 1.5},
    "SLEEP_POD_NYLON":       {"window": 500, "z_in": 2.5, "z_out": 0.3, "max_pos": 10, "mom_window": 200, "mom_thresh": 2.0},
}

def mr_with_pnl_curve(track, cfg):
    """Returns (cumulative_pnl_per_tick, n_ticks)."""
    window = cfg["window"]; z_in = cfg["z_in"]; z_out = cfg["z_out"]; max_pos = cfg["max_pos"]
    mom_window = cfg.get("mom_window", 0); mom_thresh = cfg.get("mom_thresh", None)
    pnl = 0.0; pos = 0; entry_price = 0
    pnl_curve = []
    buf = deque(); s_sum = 0.0; s_sumsq = 0.0
    for i in range(len(track)):
        ts, mid, bid, ask = track[i]
        if bid is None or ask is None:
            pnl_curve.append(pnl)
            continue
        buf.append(mid); s_sum += mid; s_sumsq += mid*mid
        if len(buf) > window:
            old=buf.popleft(); s_sum -= old; s_sumsq -= old*old
        if len(buf) < window:
            pnl_curve.append(pnl)
            continue
        m = s_sum/len(buf)
        var = (s_sumsq - len(buf)*m*m)/(len(buf)-1)
        sd = math.sqrt(max(var,1e-10))
        z = (mid-m)/sd
        mom_blocked = False
        if mom_thresh is not None and i >= mom_window:
            past_mid = track[i - mom_window][1]
            mom = (mid - past_mid)/max(sd,1)
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
        if new_state != cur_state:
            target = new_state * max_pos
            delta = target - pos
            fill_price = ask if delta > 0 else bid
            if cur_state == 0:
                entry_price = fill_price; pos = new_state
            else:
                pnl += pos * (fill_price - entry_price) * max_pos
                pos = 0
                if new_state != 0:
                    fp = ask if new_state > 0 else bid
                    entry_price = fp; pos = new_state
        # MTM-adjusted pnl per tick (approximation: realized + unrealized)
        unrealized = pos * (mid - entry_price) * max_pos if pos != 0 else 0
        pnl_curve.append(pnl + unrealized)
    return pnl_curve

def main():
    days = {d: load_with_books(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    BIN_SIZE = 1000  # ticks per bin
    print("="*100)
    print(f"PER-{BIN_SIZE}-TICK PNL ATTRIBUTION (REALIZED+UNREALIZED, by alpha)")
    print("="*100)

    for d in [2,3,4]:
        print(f"\n--- DAY {d} ---")
        per_alpha_curves = {}
        for prod, cfg in ALPHAS.items():
            track = days[d].get(prod, [])
            if not track: continue
            curve = mr_with_pnl_curve(track, cfg)
            per_alpha_curves[prod] = curve

        # Per-bin per-alpha PnL deltas
        n = max(len(c) for c in per_alpha_curves.values())
        n_bins = (n + BIN_SIZE - 1) // BIN_SIZE
        # Total PnL per bin (sum across alphas)
        bin_totals = []
        for b in range(n_bins):
            start = b * BIN_SIZE
            end = min(start + BIN_SIZE, n) - 1
            total_at_end = sum(c[end] if end < len(c) else c[-1] for c in per_alpha_curves.values())
            total_at_start = sum(c[start-1] if start > 0 and start-1 < len(c) else 0 for c in per_alpha_curves.values())
            delta = total_at_end - total_at_start
            bin_totals.append((start, end, delta))

        for start, end, delta in bin_totals:
            bar = "+" * int(min(40, abs(delta)/500)) if delta > 0 else "-" * int(min(40, abs(delta)/500))
            print(f"  ticks {start:>5}-{end:>5}: delta={delta:+8.0f} {bar}")
        print(f"  TOTAL: {sum(b[2] for b in bin_totals):+.0f}")

if __name__ == "__main__":
    main()
