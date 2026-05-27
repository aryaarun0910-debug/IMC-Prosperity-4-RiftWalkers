"""Long-lag autocorrelation: identify products with momentum or mean-reversion at scales 5-500 ticks."""
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
    xs,ys=xs[:n],ys[:n]
    mx,my=mean(xs),mean(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=math.sqrt(sum((x-mx)**2 for x in xs))
    dy=math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy else 0

def load_day(path):
    out = defaultdict(list)
    with open(path,"r") as f:
        h = f.readline().strip().split(";")
        ti=h.index("timestamp"); pi=h.index("product"); mi=h.index("mid_price")
        for ln in f:
            p = ln.strip().split(";")
            if len(p)<len(h): continue
            try: out[p[pi]].append((int(p[ti]), float(p[mi])))
            except: pass
    for k in out: out[k] = [m for _,m in sorted(out[k])]
    return out

def main():
    days = {d: load_day(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    # Compute return-difference autocorrelation at multiple lags
    # Specifically: corr(ret[t-lag], ret[t]). Negative = mean-revert at that lag, positive = momentum.
    LAGS = [1, 5, 10, 20, 50, 100, 200, 500]

    # Per-product per-day per-lag autocorr
    table = defaultdict(lambda: {l: [] for l in LAGS})
    for d in [2,3,4]:
        bp = days[d]
        for p in ALL_PRODUCTS:
            if p not in bp: continue
            s = bp[p]
            r = [(s[i+1]-s[i])/s[i] for i in range(len(s)-1) if s[i]]
            for lag in LAGS:
                if len(r) > lag:
                    c = corr(r[lag:], r[:-lag])
                    table[p][lag].append(c)

    # Aggregate cross-day average + std
    results = []
    for p in ALL_PRODUCTS:
        row = [p]
        for lag in LAGS:
            cs = table[p][lag]
            if len(cs) < 3:
                row.append((0.0, 0.0))
                continue
            row.append((sum(cs)/3, std(cs)))
        results.append(row)

    # Print
    print(f"\n{'Product':<32} | " + " | ".join(f"lag={l:<4}" for l in LAGS))
    print("-" * 110)
    for row in results:
        line = f"{row[0]:<32} | "
        for v, sd in row[1:]:
            tag = "**" if abs(v) > 0.05 and sd < 0.03 else ("." if abs(v)>0.03 else " ")
            line += f"{v:+.3f}{tag:>2} | "
        print(line)

    # Highlights: print only high-confidence signals
    print("\n" + "="*110)
    print("HIGH-CONFIDENCE LONG-LAG AUTOCORR (|avg|>0.05, std<0.03 across days)")
    print("="*110)
    found = False
    for row in results:
        p = row[0]
        for i, lag in enumerate(LAGS):
            v, sd = row[i+1]
            if abs(v) > 0.05 and sd < 0.03:
                tag = "MEAN-REVERT" if v < 0 else "MOMENTUM"
                print(f"  {p:<32} lag={lag:<4} corr={v:+.3f} std={sd:.3f} [{tag}]")
                found = True
    if not found:
        print("  (no high-confidence signals)")

if __name__ == "__main__":
    main()
