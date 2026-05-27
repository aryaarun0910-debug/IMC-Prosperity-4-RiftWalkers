"""Cross-category research: 50x50 return-correlation matrix to find hidden factor relationships
that span categories (not just within categories).
"""
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
ALL = [p for ps in CATEGORIES.values() for p in ps]
P_TO_CAT = {p: c for c, ps in CATEGORIES.items() for p in ps}

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
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy else 0
def rets(s): return [(s[i+1]-s[i])/s[i] for i in range(len(s)-1) if s[i]]

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

    # Build cross-day stable correlation map
    pair_corrs = defaultdict(list)  # (a,b) -> [c_d2, c_d3, c_d4]
    for d in [2,3,4]:
        bp = days[d]
        # precompute returns
        rets_map = {p: rets(bp[p]) for p in ALL if p in bp}
        for i, a in enumerate(ALL):
            for j, b in enumerate(ALL):
                if i >= j: continue
                if a not in rets_map or b not in rets_map: continue
                c = corr(rets_map[a], rets_map[b])
                pair_corrs[(a,b)].append(c)

    # Aggregate: average corr + cross-day std
    summary = []
    for (a,b), cs in pair_corrs.items():
        if len(cs) < 3: continue
        avg = sum(cs)/3
        sdv = std(cs)
        summary.append((a, b, avg, sdv, P_TO_CAT[a], P_TO_CAT[b]))

    # Filter: |avg| > 0.10 AND cross-day std < 0.05 (stable signal)
    print("="*120)
    print("CROSS-CATEGORY STABLE PAIRS (|avg corr| > 0.10, cross-day std < 0.05)")
    print("="*120)
    cross_cat = [s for s in summary if s[4] != s[5] and abs(s[2]) > 0.10 and s[3] < 0.05]
    cross_cat.sort(key=lambda x: -abs(x[2]))
    for a, b, c, sd, ca, cb in cross_cat[:30]:
        print(f"  {a:<32} <> {b:<32}  avg={c:+.3f}  std={sd:.3f}  [{ca} x {cb}]")
    if not cross_cat:
        print("  (no cross-category stable pairs at this threshold)")

    # Same threshold but lower (0.05)
    print()
    print("="*120)
    print("CROSS-CATEGORY weak-but-stable pairs (|avg corr| > 0.05, cross-day std < 0.03)")
    print("="*120)
    weak_stable = [s for s in summary if s[4] != s[5] and abs(s[2]) > 0.05 and s[3] < 0.03]
    weak_stable.sort(key=lambda x: -abs(x[2]))
    for a, b, c, sd, ca, cb in weak_stable[:40]:
        print(f"  {a:<32} <> {b:<32}  avg={c:+.3f}  std={sd:.3f}  [{ca} x {cb}]")
    if not weak_stable:
        print("  (no weak-stable cross-category pairs found)")

    # Within-category — confirm what we already know
    print()
    print("="*120)
    print("WITHIN-CATEGORY STABLE PAIRS (re-confirm — avg > 0.10, std < 0.05)")
    print("="*120)
    within = [s for s in summary if s[4] == s[5] and abs(s[2]) > 0.10 and s[3] < 0.05]
    within.sort(key=lambda x: -abs(x[2]))
    for a, b, c, sd, ca, cb in within[:30]:
        print(f"  {a:<32} <> {b:<32}  avg={c:+.3f}  std={sd:.3f}  [{ca}]")

if __name__ == "__main__":
    main()
