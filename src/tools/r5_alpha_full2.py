"""Faster online backtest using incremental mean/std (O(n) total)."""
import math
from collections import defaultdict, deque

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

def online_z_backtest(spread, leg_size=5, z_in=1.5, z_out=0.3, window=500):
    """Incremental rolling mean/std for O(n) speed."""
    n = len(spread)
    pnl = 0.0; pos = 0; entry = 0; trips = 0
    buf = deque()
    s_sum = 0.0; s_sumsq = 0.0
    for i in range(n):
        x = spread[i]
        buf.append(x); s_sum += x; s_sumsq += x*x
        if len(buf) > window:
            old = buf.popleft()
            s_sum -= old; s_sumsq -= old*old
        if len(buf) < window: continue
        m = s_sum / len(buf)
        var = (s_sumsq - len(buf)*m*m) / (len(buf)-1)
        sd = math.sqrt(max(var, 1e-10))
        z = (x - m)/sd
        if pos==0:
            if z > z_in: pos=-1; entry=x
            elif z < -z_in: pos=+1; entry=x
        else:
            if abs(z) < z_out:
                pnl += pos*(x-entry)*leg_size
                pos=0; trips+=1
    if pos!=0:
        pnl += pos*(spread[-1]-entry)*leg_size; trips+=1
    return pnl, trips

def main():
    days = {d: load_day(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    # ===== ONLINE BACKTEST: 3 alphas =====
    print("="*100); print("ONLINE BACKTEST (rolling window, no look-ahead)"); print("="*100)
    print("\n[Alpha 1: PIST+STRAW-2*RASP, leg=5]")
    for d in [2,3,4]:
        bp = days[d]
        n = min(len(bp[p]) for p in ["SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"])
        spread = [bp["SNACKPACK_PISTACHIO"][i]+bp["SNACKPACK_STRAWBERRY"][i]-2*bp["SNACKPACK_RASPBERRY"][i] for i in range(n)]
        for win in [200,500,1000]:
            pnl, trips = online_z_backtest(spread, leg_size=5, window=win)
            print(f"  Day {d}: window={win:>4}  PnL={pnl:+8.0f}  trips={trips}")

    print("\n[Alpha 2: CHOC+VAN, leg=5]")
    for d in [2,3,4]:
        bp = days[d]
        n = min(len(bp[p]) for p in ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA"])
        spread = [bp["SNACKPACK_CHOCOLATE"][i]+bp["SNACKPACK_VANILLA"][i] for i in range(n)]
        for win in [200,500,1000]:
            pnl, trips = online_z_backtest(spread, leg_size=5, window=win)
            print(f"  Day {d}: window={win:>4}  PnL={pnl:+8.0f}  trips={trips}")

    print("\n[Alpha 3: Pebbles (XS+S+M+L)+2*XL, leg=2]")
    for d in [2,3,4]:
        bp = days[d]
        prods = ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"]
        n = min(len(bp[p]) for p in prods)
        spread = [sum(bp[p][i] for p in prods[:4]) + 2*bp["PEBBLES_XL"][i] for i in range(n)]
        for win in [500,1000,2000]:
            pnl, trips = online_z_backtest(spread, leg_size=2, window=win)
            print(f"  Day {d}: window={win:>4}  PnL={pnl:+8.0f}  trips={trips}")

    # ===== ROBOTS DAY-DRIFT =====
    print()
    print("="*100); print("ROBOTS — drift signature"); print("="*100)
    for p in CATEGORIES["Robots"]:
        line = f"  {p:<24}"
        for d in [2,3,4]:
            s = days[d].get(p, [])
            if not s: continue
            o=s[0]; e=s[-1]; pct=(e-o)/o*100
            t200 = s[200] if len(s)>200 else s[-1]
            early_dir = "+" if t200>o else "-"
            full_dir = "+" if e>o else "-"
            match = "MATCH" if early_dir==full_dir else "MISS"
            line += f"  d{d}:open={o:.0f}->close={e:.0f}({pct:+.1f}%)[{match}]"
        print(line)

    # ===== WEAK CATEGORIES: per-product mean reversion =====
    print()
    print("="*100); print("WEAK CATEGORIES: single-asset mean-reversion (z-score on price level)"); print("="*100)
    for cat in ["GalaxySounds","SleepPods","Microchips","UVVisors","Translators","Panels","OxygenShakes"]:
        prods = CATEGORIES[cat]
        print(f"\n[{cat}]")
        for p in prods:
            pnls = []
            for d in [2,3,4]:
                s = days[d].get(p, [])
                if not s: continue
                pnl, trips = online_z_backtest(s, leg_size=10, window=500, z_in=1.5, z_out=0.3)
                pnls.append((d,pnl,trips))
            avg = sum(x[1] for x in pnls)/len(pnls) if pnls else 0
            print(f"  {p:<32} 3-day avg PnL={avg:+8.0f}  per-day=[{', '.join(f'{x[1]:+5.0f}' for x in pnls)}]")

if __name__ == "__main__":
    main()
