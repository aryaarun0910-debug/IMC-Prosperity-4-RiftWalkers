"""Diagnose PEBBLES_XL regime: when does mean-reversion fail?

Hypothesis: trending periods (large rolling momentum) are when reversion strategy bleeds.
Test: does adding a 'momentum filter' (only trade when |momentum| < threshold) improve PnL?
"""
import sys, os, math
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_xl(d):
    out = []
    with open(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv","r") as f:
        h = f.readline().strip().split(";")
        ti=h.index("timestamp"); pi=h.index("product"); mi=h.index("mid_price")
        bp1=h.index("bid_price_1"); ap1=h.index("ask_price_1")
        for ln in f:
            p = ln.strip().split(";")
            if len(p)<len(h): continue
            if p[pi] != "PEBBLES_XL": continue
            try: out.append((int(p[ti]), float(p[mi]), int(p[bp1]) if p[bp1] else None, int(p[ap1]) if p[ap1] else None))
            except: pass
    return sorted(out)

def backtest_with_filter(track, window=500, z_in=1.5, z_out=0.3, mom_window=200, mom_thresh=None, leg_size=10):
    """If mom_thresh is set, skip new entries when |mom| > mom_thresh.
    mom = (mid_now - mid_mom_window_ago) / std_of_window"""
    pnl = 0.0; pos = 0; entry = 0; trips = 0
    buf = deque(); s_sum=0.0; s_sumsq=0.0
    skipped_by_mom = 0
    for i in range(len(track)):
        ts, mid, bid, ask = track[i]
        if bid is None or ask is None: continue
        buf.append(mid); s_sum += mid; s_sumsq += mid*mid
        if len(buf) > window:
            old=buf.popleft(); s_sum -= old; s_sumsq -= old*old
        if len(buf) < window: continue
        m = s_sum/len(buf)
        var = (s_sumsq - len(buf)*m*m)/(len(buf)-1)
        sd = math.sqrt(max(var,1e-10))
        z = (mid - m)/sd
        # momentum: change over mom_window normalized by spread of window
        mom = 0
        if mom_thresh is not None and i >= mom_window:
            past = track[i-mom_window]
            past_mid = past[1]
            mom = (mid - past_mid) / max(sd, 1)  # in std units
        cur_state = pos
        new_state = pos
        if pos == 0:
            if z > z_in:
                if mom_thresh is None or abs(mom) < mom_thresh:
                    new_state = -1
                else:
                    skipped_by_mom += 1
            elif z < -z_in:
                if mom_thresh is None or abs(mom) < mom_thresh:
                    new_state = +1
                else:
                    skipped_by_mom += 1
        else:
            if pos == -1 and z < z_out: new_state = 0
            elif pos == +1 and z > -z_out: new_state = 0
        # Simulate fill at bid/ask
        if new_state != cur_state:
            if cur_state == 0:
                # entering: fill at adverse side
                if new_state > 0: entry_price = ask  # buying at ask
                else: entry_price = bid  # selling at bid
                entry = entry_price
                pos = new_state
            else:
                # exiting: also adverse
                if cur_state > 0: exit_price = bid  # closing long → sell at bid
                else: exit_price = ask  # closing short → buy at ask
                pnl += pos * (exit_price - entry) * leg_size
                trips += 1
                pos = 0
                if new_state != 0:
                    # immediately reentry
                    if new_state > 0: entry = ask
                    else: entry = bid
                    pos = new_state
    if pos != 0:
        last_mid = track[-1][1]
        pnl += pos * (last_mid - entry) * leg_size
        trips += 1
    return pnl, trips, skipped_by_mom

if __name__ == "__main__":
    days = {d: load_xl(d) for d in [2,3,4]}

    print("PEBBLES_XL — backtest with momentum filter")
    print(f"{'mom_thresh':>12} | {'Day2':>10} | {'Day3':>10} | {'Day4':>10} | {'Total':>10} | trips_total | skipped_total")
    for mom_t in [None, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        results = []
        for d in [2,3,4]:
            pnl, trips, skipped = backtest_with_filter(days[d], mom_thresh=mom_t)
            results.append((pnl, trips, skipped))
        total = sum(r[0] for r in results)
        trips_total = sum(r[1] for r in results)
        skipped_total = sum(r[2] for r in results)
        label = "OFF" if mom_t is None else f"{mom_t:.1f}"
        print(f"{label:>12} | {results[0][0]:+10.0f} | {results[1][0]:+10.0f} | {results[2][0]:+10.0f} | {total:+10.0f} | {trips_total:>5} | {skipped_total:>5}")

    # Test on the LIVE 1K slice (first 1K ticks of day 4) since live result was +$149
    print("\n--- Same filter on FIRST 1K ticks of day 4 (matches live aesthetic test) ---")
    for mom_t in [None, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        d4_slice = days[4][:1000]
        pnl, trips, skipped = backtest_with_filter(d4_slice, mom_thresh=mom_t)
        label = "OFF" if mom_t is None else f"{mom_t:.1f}"
        print(f"  mom_thresh={label:>4}  PnL={pnl:+8.0f}  trips={trips}  skipped={skipped}")
