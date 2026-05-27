"""Full alpha validation: Pebbles backtest, Robots trend analysis, weak-category drift profiling."""
import math
from collections import defaultdict

CATEGORIES = {
    "GalaxySounds": ["GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS","GALAXY_SOUNDS_SOLAR_FLAMES"],
    "SleepPods":   ["SLEEP_POD_SUEDE","SLEEP_POD_LAMB_WOOL","SLEEP_POD_POLYESTER","SLEEP_POD_NYLON","SLEEP_POD_COTTON"],
    "Microchips":  ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_SQUARE","MICROCHIP_RECTANGLE","MICROCHIP_TRIANGLE"],
    "Pebbles":     ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"],
    "Robots":      ["ROBOT_VACUUMING","ROBOT_MOPPING","ROBOT_DISHES","ROBOT_LAUNDRY","ROBOT_IRONING"],
    "UVVisors":    ["UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_MAGENTA"],
    "Translators": ["TRANSLATOR_SPACE_GRAY","TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_VOID_BLUE"],
    "Panels":      ["PANEL_1X2","PANEL_2X2","PANEL_1X4","PANEL_2X4","PANEL_4X4"],
    "OxygenShakes":["OXYGEN_SHAKE_MORNING_BREATH","OXYGEN_SHAKE_EVENING_BREATH","OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC"],
    "SnackPacks":  ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"],
}

def mean(xs): return sum(xs)/len(xs) if xs else 0
def std(xs):
    if len(xs)<2: return 0
    m=mean(xs); v=sum((x-m)**2 for x in xs)/(len(xs)-1)
    return math.sqrt(v)

def load_day(path):
    out=defaultdict(list)
    with open(path,"r") as f:
        h=f.readline().strip().split(";")
        ti=h.index("timestamp"); pi=h.index("product"); mi=h.index("mid_price")
        for ln in f:
            p=ln.strip().split(";")
            if len(p)<len(h): continue
            try: out[p[pi]].append((int(p[ti]),float(p[mi])))
            except: pass
    for k in out: out[k]=[m for _,m in sorted(out[k])]
    return out

def hurst(ts):
    if len(ts)<200: return 0.5
    log_lags, log_rs = [], []
    for lag in range(2, min(100, len(ts)//4)):
        diffs=[ts[i+lag]-ts[i] for i in range(len(ts)-lag)]
        if len(diffs)<20: continue
        s=std(diffs)
        if s<=0: continue
        log_lags.append(math.log(lag)); log_rs.append(math.log(s))
    if len(log_lags)<5: return 0.5
    n=len(log_lags); mx=mean(log_lags); my=mean(log_rs)
    num=sum((log_lags[i]-mx)*(log_rs[i]-my) for i in range(n))
    den=sum((log_lags[i]-mx)**2 for i in range(n))
    return num/den if den else 0.5

def z_backtest(spread, leg_size=5, z_in=1.5, z_out=0.3, rolling_window=None):
    """Run z-score reversion backtest on a spread series.

    If rolling_window is set, use rolling mean/std (online). Else use full-sample (look-ahead).
    leg_size = contracts per LEG of the spread (PnL in spread-units * leg_size).
    """
    n = len(spread)
    pnl = 0.0; pos = 0; entry = 0; trips = 0
    if rolling_window is None:
        m = mean(spread); s = std(spread)
        for i in range(n):
            z = (spread[i]-m)/s if s>0 else 0
            if pos==0:
                if z > z_in: pos=-1; entry=spread[i]
                elif z < -z_in: pos=+1; entry=spread[i]
            else:
                if abs(z) < z_out:
                    pnl += pos * (spread[i] - entry) * leg_size
                    pos=0; trips += 1
        if pos!=0: pnl += pos*(spread[-1]-entry)*leg_size; trips+=1
    else:
        # online: use rolling stats over rolling_window prior ticks
        for i in range(rolling_window, n):
            window = spread[i-rolling_window:i]
            m = mean(window); s = std(window)
            z = (spread[i]-m)/s if s>0 else 0
            if pos==0:
                if z > z_in: pos=-1; entry=spread[i]
                elif z < -z_in: pos=+1; entry=spread[i]
            else:
                if abs(z) < z_out:
                    pnl += pos*(spread[i]-entry)*leg_size
                    pos=0; trips += 1
        if pos!=0: pnl += pos*(spread[-1]-entry)*leg_size; trips+=1
    return pnl, trips

def main():
    days = {d: load_day(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    # ============================================
    # 1. Pebbles backtest (XL vs basket)
    # ============================================
    print("="*100); print("PEBBLES BACKTEST: spread = (XS+S+M+L) + 2*XL"); print("="*100)
    for d in [2,3,4]:
        bp = days[d]
        prods = ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"]
        n = min(len(bp[p]) for p in prods)
        spread = [sum(bp[p][i] for p in prods[:4]) + 2*bp["PEBBLES_XL"][i] for i in range(n)]
        # full-sample
        pnl_fs, trips_fs = z_backtest(spread, leg_size=2, z_in=1.5, z_out=0.3)
        # online
        pnl_on, trips_on = z_backtest(spread, leg_size=2, z_in=1.5, z_out=0.3, rolling_window=500)
        print(f"  Day {d}: full-sample PnL={pnl_fs:+8.0f} trips={trips_fs}  |  online (rolling-500) PnL={pnl_on:+8.0f} trips={trips_on}")

    # ============================================
    # 2. Online (live-realistic) backtest of all 3 alphas
    # ============================================
    print()
    print("="*100); print("ONLINE BACKTESTS (rolling 500-tick stats — no look-ahead)"); print("="*100)
    print("\nAlpha 1: PIST+STRAW-2*RASP")
    for d in [2,3,4]:
        bp = days[d]
        n = min(len(bp[p]) for p in ["SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"])
        spread = [bp["SNACKPACK_PISTACHIO"][i] + bp["SNACKPACK_STRAWBERRY"][i] - 2*bp["SNACKPACK_RASPBERRY"][i] for i in range(n)]
        for win in [200, 500, 1000]:
            pnl, trips = z_backtest(spread, leg_size=5, z_in=1.5, z_out=0.3, rolling_window=win)
            print(f"  Day {d}: window={win:>4}  PnL={pnl:+8.0f}  trips={trips}")

    print("\nAlpha 2: CHOC+VAN")
    for d in [2,3,4]:
        bp = days[d]
        n = min(len(bp[p]) for p in ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA"])
        spread = [bp["SNACKPACK_CHOCOLATE"][i] + bp["SNACKPACK_VANILLA"][i] for i in range(n)]
        for win in [200, 500, 1000]:
            pnl, trips = z_backtest(spread, leg_size=5, z_in=1.5, z_out=0.3, rolling_window=win)
            print(f"  Day {d}: window={win:>4}  PnL={pnl:+8.0f}  trips={trips}")

    # ============================================
    # 3. Robots trend analysis
    # ============================================
    print()
    print("="*100); print("ROBOTS DAY-START vs DAY-END (drift signature)"); print("="*100)
    for p in CATEGORIES["Robots"]:
        for d in [2,3,4]:
            s = days[d].get(p, [])
            if not s: continue
            # check first 100, middle, end
            o = s[0]; e = s[-1]; pct = (e-o)/o*100
            # check the first 200 ticks: does direction predict the rest?
            early = s[200] if len(s)>200 else s[-1]
            early_dir = "+" if early > o else "-"
            full_dir = "+" if e > o else "-"
            match = "MATCH" if early_dir == full_dir else "MISMATCH"
            print(f"  Day {d} {p:<24}  open={o:.0f}  early(t200)={early:.0f}({early_dir})  close={e:.0f}({full_dir})  full_pct={pct:+5.1f}%  [{match}]")
        print()

    # ============================================
    # 4. Per-product price-level mean reversion (single-asset z-score) for weak categories
    # ============================================
    print("="*100); print("WEAK CATEGORIES: per-product mean-reversion (single-asset, z-score)"); print("="*100)
    for cat in ["GalaxySounds","SleepPods","Microchips","UVVisors","Translators","Panels","OxygenShakes"]:
        prods = CATEGORIES[cat]
        print(f"\n[{cat}]")
        for p in prods:
            results = []
            for d in [2,3,4]:
                s = days[d].get(p, [])
                if not s: continue
                h = hurst(s)
                # toy backtest: deviation from rolling mean
                pnl, trips = z_backtest(s, leg_size=10, z_in=1.5, z_out=0.3, rolling_window=500)
                results.append((d, h, pnl, trips, max(s)-min(s)))
            avg_pnl = mean([r[2] for r in results])
            avg_h = mean([r[1] for r in results])
            print(f"  {p:<32} avg_pnl={avg_pnl:+8.0f} (3 days)  avg_hurst={avg_h:.3f}  range~{int(mean([r[4] for r in results]))}")

if __name__ == "__main__":
    main()
