"""Order Book Imbalance (OBI) study: does (bid_vol - ask_vol)/(bid_vol + ask_vol) predict next-tick mid moves?

Tests at lags 1, 5, 10 ticks ahead. Cross-day stability is the key filter.
"""
import math
from collections import defaultdict

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

def mean(xs): return sum(xs)/len(xs) if xs else 0
def std(xs):
    if len(xs)<2: return 0
    m=mean(xs); v=sum((x-m)**2 for x in xs)/(len(xs)-1)
    return math.sqrt(v)
def corr(xs, ys):
    n = min(len(xs),len(ys))
    if n<2: return 0
    xs,ys = xs[:n], ys[:n]
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs)); dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy else 0

def load_with_books(path):
    """Returns {product: [(ts, mid, total_bid_vol, total_ask_vol), ...]}"""
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
                bv = sum(int(p[idx[f"bid_volume_{k}"]]) for k in [1,2,3] if p[idx[f"bid_volume_{k}"]])
                av = sum(int(p[idx[f"ask_volume_{k}"]]) for k in [1,2,3] if p[idx[f"ask_volume_{k}"]])
                out[prod].append((ts, mid, bv, av))
            except: pass
    for k in out: out[k] = sorted(out[k])
    return out

def main():
    days = {d: load_with_books(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    # For each product, compute OBI series + future return series, then correlation
    # OBI[t] = (bv[t] - av[t]) / (bv[t] + av[t])
    # FutureRet[t,lag] = (mid[t+lag] - mid[t]) / mid[t]
    LAGS = [1, 3, 5, 10]
    # results[product][lag] = list of corr across days
    results = defaultdict(lambda: {l: [] for l in LAGS})
    for d in [2,3,4]:
        bp = days[d]
        for p in ALL_PRODUCTS:
            if p not in bp: continue
            track = bp[p]
            obi = []
            mids = []
            for t in track:
                _, mid, bv, av = t
                if bv + av == 0:
                    obi.append(0)
                else:
                    obi.append((bv - av) / (bv + av))
                mids.append(mid)
            for lag in LAGS:
                if len(mids) <= lag: continue
                future_rets = [(mids[i+lag]-mids[i])/mids[i] if mids[i] else 0 for i in range(len(mids)-lag)]
                obi_now = obi[:len(future_rets)]
                c = corr(obi_now, future_rets)
                results[p][lag].append(c)

    # Filter: |avg| > 0.05 AND std < 0.03
    print("="*110)
    print("OBI -> FUTURE-RETURN CORRELATION (|avg|>0.05, cross-day std<0.03)")
    print("="*110)
    found_any = False
    for p in ALL_PRODUCTS:
        for lag in LAGS:
            cs = results[p][lag]
            if len(cs) < 3: continue
            avg = sum(cs)/3
            sdv = std(cs)
            if abs(avg) > 0.05 and sdv < 0.03:
                tag = "POS" if avg > 0 else "NEG"
                print(f"  {p:<32} lag={lag:<3}  corr={avg:+.3f}  std={sdv:.3f}  [{tag}]")
                found_any = True
    if not found_any:
        print("  (no high-confidence OBI signals at this threshold)")

    # Lower threshold
    print()
    print("="*110)
    print("LOWER-THRESHOLD OBI (|avg|>0.03, std<0.02)")
    print("="*110)
    found_any = False
    for p in ALL_PRODUCTS:
        for lag in LAGS:
            cs = results[p][lag]
            if len(cs) < 3: continue
            avg = sum(cs)/3
            sdv = std(cs)
            if abs(avg) > 0.03 and sdv < 0.02:
                tag = "POS" if avg > 0 else "NEG"
                print(f"  {p:<32} lag={lag:<3}  corr={avg:+.3f}  std={sdv:.3f}  [{tag}]")
                found_any = True

    # Show ALL avg corr, sorted by abs at lag=1
    print()
    print("="*110)
    print("ALL OBI -> next-tick correlation (lag=1) — sorted by |avg|")
    print("="*110)
    rows = []
    for p in ALL_PRODUCTS:
        cs = results[p][1]
        if len(cs) < 3: continue
        rows.append((p, sum(cs)/3, std(cs)))
    rows.sort(key=lambda x: -abs(x[1]))
    for p, avg, sdv in rows[:25]:
        print(f"  {p:<32} corr={avg:+.4f}  std={sdv:.4f}")

if __name__ == "__main__":
    main()
