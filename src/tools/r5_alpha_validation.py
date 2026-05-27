"""Validate the alphas: simulate toy pair-trade PnL, check spread stationarity, check lead-lag at offset>0."""
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
def corr(xs,ys):
    n=min(len(xs),len(ys))
    if n<2: return 0
    xs,ys=xs[:n],ys[:n]
    mx,my=mean(xs),mean(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=math.sqrt(sum((x-mx)**2 for x in xs)); dy=math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy else 0
def rets(s): return [(s[i+1]-s[i])/s[i] for i in range(len(s)-1) if s[i]]

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

def half_life(spread):
    """OU process half-life via lag-1 AR. <100 ticks = fast revert."""
    if len(spread)<200: return float('inf')
    s = spread[:-1]; ds = [spread[i+1]-spread[i] for i in range(len(spread)-1)]
    ms=mean(s); s_dev=[x-ms for x in s]
    num=sum(s_dev[i]*ds[i] for i in range(len(ds)))
    den=sum(x*x for x in s_dev)
    if den<=0: return float('inf')
    theta = -num/den
    if theta<=0: return float('inf')
    return math.log(2)/theta

def main():
    days = {d: load_day(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    # =============================================
    # 1. SnackPacks spread #1: PIST + STRAW - 2*RASP
    # =============================================
    print("="*100); print("ALPHA 1: SnackPacks spread = PISTACHIO + STRAWBERRY - 2*RASPBERRY"); print("="*100)
    for d in [2,3,4]:
        bp = days[d]
        n = min(len(bp[p]) for p in ["SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"])
        spread = [bp["SNACKPACK_PISTACHIO"][i] + bp["SNACKPACK_STRAWBERRY"][i] - 2*bp["SNACKPACK_RASPBERRY"][i] for i in range(n)]
        sp_mean = mean(spread); sp_std = std(spread); sp_h = hurst(spread); sp_hl = half_life(spread)
        print(f"  Day {d}: mean={sp_mean:+8.1f}  std={sp_std:.1f}  range=[{min(spread):.1f},{max(spread):.1f}]  hurst={sp_h:.3f}  half_life={sp_hl:.1f}t")

    # =============================================
    # 2. SnackPacks spread #2: CHOC + VAN
    # =============================================
    print()
    print("="*100); print("ALPHA 2: SnackPacks spread = CHOCOLATE + VANILLA  (should be near-constant)"); print("="*100)
    for d in [2,3,4]:
        bp = days[d]
        n = min(len(bp[p]) for p in ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA"])
        spread = [bp["SNACKPACK_CHOCOLATE"][i] + bp["SNACKPACK_VANILLA"][i] for i in range(n)]
        sp_mean=mean(spread); sp_std=std(spread); sp_h=hurst(spread); sp_hl=half_life(spread)
        print(f"  Day {d}: mean={sp_mean:.1f}  std={sp_std:.1f}  range=[{min(spread):.1f},{max(spread):.1f}]  hurst={sp_h:.3f}  half_life={sp_hl:.1f}t")

    # =============================================
    # 3. Pebbles spread: 4*XL + (XS+S+M+L)
    # =============================================
    print()
    print("="*100); print("ALPHA 3: Pebbles spread = (XS+S+M+L) + 2*XL  (since corr ~ -0.5)"); print("="*100)
    for d in [2,3,4]:
        bp = days[d]
        prods = ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"]
        n = min(len(bp[p]) for p in prods)
        spread = [sum(bp[p][i] for p in prods[:4]) + 2*bp["PEBBLES_XL"][i] for i in range(n)]
        sp_mean=mean(spread); sp_std=std(spread); sp_h=hurst(spread); sp_hl=half_life(spread)
        print(f"  Day {d}: mean={sp_mean:.1f}  std={sp_std:.1f}  range=[{min(spread):.1f},{max(spread):.1f}]  hurst={sp_h:.3f}  half_life={sp_hl:.1f}t")

    # =============================================
    # 4. Toy pair-trade PnL on alpha 1 (SnackPacks)
    # =============================================
    print()
    print("="*100); print("BACKTEST: SnackPack alpha 1 (PIST + STRAW - 2*RASP), z-score reversion"); print("="*100)
    print(f"  Position limits: PIST=10, STRAW=10, RASP=10 (each)")
    print(f"  Strategy: when |z| > 1.5, take fully (long/short pos at limit). Exit when |z| < 0.3.")
    for d in [2,3,4]:
        bp = days[d]
        n = min(len(bp[p]) for p in ["SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"])
        spread = [bp["SNACKPACK_PISTACHIO"][i] + bp["SNACKPACK_STRAWBERRY"][i] - 2*bp["SNACKPACK_RASPBERRY"][i] for i in range(n)]
        m = mean(spread); s = std(spread)

        pos = 0  # +1 = long spread (long PIST+STRAW, short 2x RASP), -1 = short, 0 = flat
        entry_spread = 0
        pnl = 0.0
        n_round_trips = 0
        for i in range(n):
            z = (spread[i]-m)/s if s>0 else 0
            if pos==0:
                if z > 1.5:
                    pos = -1; entry_spread = spread[i]  # short the spread
                elif z < -1.5:
                    pos = +1; entry_spread = spread[i]
            else:
                if abs(z) < 0.3:
                    pnl += pos * (spread[i] - entry_spread) * 5  # 5 contracts each side
                    pos = 0; n_round_trips += 1
        # close at end
        if pos != 0:
            pnl += pos * (spread[-1] - entry_spread) * 5
            n_round_trips += 1
        print(f"  Day {d}: PnL={pnl:+8.0f}  round_trips={n_round_trips}  (10K ticks)")

    # =============================================
    # 5. CHOC+VAN spread trade  (Alpha 2)
    # =============================================
    print()
    print("="*100); print("BACKTEST: SnackPack alpha 2 (CHOCOLATE + VANILLA), z-score reversion"); print("="*100)
    for d in [2,3,4]:
        bp = days[d]
        n = min(len(bp[p]) for p in ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA"])
        spread = [bp["SNACKPACK_CHOCOLATE"][i] + bp["SNACKPACK_VANILLA"][i] for i in range(n)]
        m=mean(spread); s=std(spread)
        pos=0; entry=0; pnl=0.0; trips=0
        for i in range(n):
            z = (spread[i]-m)/s if s>0 else 0
            if pos==0:
                if z > 1.5: pos=-1; entry=spread[i]
                elif z < -1.5: pos=+1; entry=spread[i]
            else:
                if abs(z) < 0.3:
                    pnl += pos*(spread[i]-entry)*5
                    pos=0; trips += 1
        if pos!=0:
            pnl += pos*(spread[-1]-entry)*5
            trips += 1
        print(f"  Day {d}: PnL={pnl:+8.0f}  round_trips={trips}")

    # =============================================
    # 6. Lead-lag at lags 1..10 on the 7 weak-corr categories
    # =============================================
    print()
    print("="*100); print("LEAD-LAG (lag>0) on the 7 weakly-correlated categories"); print("="*100)
    for cat, prods in CATEGORIES.items():
        if cat in ("SnackPacks","Pebbles","Robots"): continue
        print(f"\n[{cat}]")
        any_significant = False
        for d in [4]:  # day 4 only
            bp = days[d]
            for i in range(len(prods)):
                for j in range(len(prods)):
                    if i==j: continue
                    a, b = bp[prods[i]], bp[prods[j]]
                    n = min(len(a),len(b))
                    ra = rets(a[:n]); rb = rets(b[:n])
                    # check if a@t correlates with b@t-k (b leads a by k)
                    for k in [1,2,3,5,10]:
                        if len(ra)>k and len(rb)>k:
                            c = corr(ra[k:], rb[:-k])
                            if abs(c) > 0.05:
                                any_significant = True
                                print(f"  day{d}: {prods[j]:<32} leads {prods[i]:<32} by {k} ticks  corr={c:+.3f}")
        if not any_significant:
            print("  (no |c|>0.05 lead-lag found)")

if __name__ == "__main__":
    main()
