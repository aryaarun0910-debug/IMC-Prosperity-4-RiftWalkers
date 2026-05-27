"""Deep dive on SnackPacks + return-correlation across all categories.

Goal: find the structural relationships that survive across all 3 days.
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

def mean(xs): return sum(xs)/len(xs) if xs else 0
def std(xs):
    if len(xs)<2: return 0
    m = mean(xs); v = sum((x-m)**2 for x in xs)/(len(xs)-1)
    return math.sqrt(v)
def corr(xs, ys):
    n = min(len(xs),len(ys))
    if n<2: return 0
    xs,ys=xs[:n],ys[:n]
    mx,my=mean(xs),mean(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=math.sqrt(sum((x-mx)**2 for x in xs)); dy=math.sqrt(sum((y-my)**2 for y in ys))
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

# Returns
def rets(s): return [(s[i+1]-s[i])/s[i] for i in range(len(s)-1) if s[i]]

def ret_corr_matrix(by_prod, prods):
    M = {}
    for i,a in enumerate(prods):
        for j,b in enumerate(prods):
            if i<j:
                M[(a,b)] = corr(rets(by_prod[a]), rets(by_prod[b]))
    return M

def main():
    days = {d: load_day(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    # 1. Per-category return-correlation matrix, averaged across 3 days
    print("="*100)
    print("RETURN-CORRELATION MATRIX (averaged day 2/3/4)")
    print("="*100)
    cat_avgcorrs = []
    for cat, prods in CATEGORIES.items():
        # average abs corr across days
        all_pairs = defaultdict(list)
        for d, by_prod in days.items():
            M = ret_corr_matrix(by_prod, prods)
            for k,v in M.items(): all_pairs[k].append(v)
        avg = {k: mean(v) for k,v in all_pairs.items()}
        std_pair = {k: std(v) for k,v in all_pairs.items()}
        absavg = mean([abs(c) for c in avg.values()])
        # max abs across days
        maxabs = max(abs(c) for c in avg.values()) if avg else 0
        cat_avgcorrs.append((cat, absavg, maxabs, avg, std_pair, prods))
        print(f"\n[{cat}]  mean|corr|={absavg:.3f}  max|corr|={maxabs:.3f}")
        # show all pairs sorted by abs
        for (a,b),c in sorted(avg.items(), key=lambda kv: -abs(kv[1])):
            stab = std_pair[(a,b)]
            stab_flag = "STABLE" if stab < 0.05 else ("MODERATE" if stab < 0.15 else "UNSTABLE")
            print(f"  {a:<32} <> {b:<32}  corr={c:+.3f}  cross-day-std={stab:.3f} [{stab_flag}]")

    # 2. Rank by avg abs return correlation
    print("\n" + "="*100)
    print("CATEGORIES RANKED BY MEAN ABSOLUTE RETURN-CORRELATION (cross-day-stable signals)")
    print("="*100)
    for cat, absavg, maxabs, avg, stds, prods in sorted(cat_avgcorrs, key=lambda x:-x[1]):
        # count how many pairs have |corr|>0.5 AND std<0.1
        strong_stable = sum(1 for k in avg if abs(avg[k])>0.5 and stds[k]<0.1)
        print(f"  {cat:<14} mean|c|={absavg:.3f}  max={maxabs:.3f}  strong-stable-pairs={strong_stable}/10")

    # 3. SnackPacks deep dive: identify the factor structure
    print("\n" + "="*100)
    print("SNACKPACKS FACTOR STRUCTURE")
    print("="*100)
    sp = CATEGORIES["SnackPacks"]
    # use day 4
    print("\nDay 4 levels — open/close/range:")
    for p in sp:
        s = days[4][p]
        print(f"  {p:<32} open={s[0]:.1f} close={s[-1]:.1f} min={min(s):.1f} max={max(s):.1f} range={max(s)-min(s):.0f}")

    # signed-direction grouping: at each tick, who moves up vs down?
    # for each pair, check sign of return correlation
    M4 = ret_corr_matrix(days[4], sp)
    print("\nDay 4 return-corr signs (>0 = same direction, <0 = mirror):")
    for k,v in sorted(M4.items()):
        sign = "+" if v>0 else "-"
        print(f"  {k[0]:<32} {sign}  {k[1]:<32}  c={v:+.3f}")

    # try to find a hidden orthogonal factor: pick the pair with highest |corr|, then see who else groups with each
    # PISTACHIO vs STRAWBERRY = +0.914 → same group
    # CHOCOLATE vs VANILLA = -0.912 → opposite groups
    # PISTACHIO vs RASPBERRY = -0.829 → opposite
    # STRAWBERRY vs RASPBERRY = -0.917 → opposite
    # So group A = {PISTACHIO, STRAWBERRY}, group B = {RASPBERRY}; need to place CHOCOLATE & VANILLA
    # Need: corr(CHOC, PIST), corr(CHOC, STRAW), corr(VAN, PIST), etc.
    print("\nGroup assignment via signed correlations:")
    for p in sp:
        for q in sp:
            if p<q:
                c = M4.get((p,q), M4.get((q,p), 0))
                # tag with arrow
                print(f"  {p:<28} -> {q:<28} {'+' if c>0 else '-'}  c={c:+.3f}")

    # 4. Robots (mean-reverting) deep dive
    print("\n" + "="*100)
    print("ROBOTS — Hurst < 0.45 across days")
    print("="*100)
    rb = CATEGORIES["Robots"]
    for p in rb:
        for d in [2,3,4]:
            s = days[d].get(p, [])
            if s: print(f"  Day {d} {p:<24} levels: open={s[0]:.1f} close={s[-1]:.1f} range={max(s)-min(s):.0f}")
        print()

if __name__ == "__main__":
    main()
