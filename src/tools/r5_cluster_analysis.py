"""R5 cluster-level analysis: per-category correlation, cointegration proxy, lead-lag, vol/trend.

Output: ranked categories by exploitability + within-category pair signals.
"""
import os, sys, math
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

def load_mids(paths):
    """Returns {product: [(day, ts, mid), ...]} sorted by (day, ts)."""
    mids = defaultdict(list)
    for path in paths:
        with open(path,"r") as f:
            header = f.readline().strip().split(";")
            di = header.index("day"); ti = header.index("timestamp")
            pi = header.index("product"); mi = header.index("mid_price")
            for ln in f:
                p = ln.strip().split(";")
                if len(p) < len(header): continue
                try:
                    mids[p[pi]].append((int(p[di]), int(p[ti]), float(p[mi])))
                except: pass
    for k in mids: mids[k].sort()
    return mids

def mean(xs): return sum(xs)/len(xs) if xs else 0
def std(xs):
    if len(xs)<2: return 0
    m = mean(xs); v = sum((x-m)**2 for x in xs)/(len(xs)-1)
    return math.sqrt(v)
def corr(xs, ys):
    n = min(len(xs),len(ys))
    if n<2: return 0
    xs, ys = xs[:n], ys[:n]
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    if dx*dy == 0: return 0
    return num/(dx*dy)

def hurst(ts, lags=range(2, 100)):
    """Rough Hurst exponent. <0.5=mean-revert, ~0.5=random walk, >0.5=trending."""
    if len(ts) < 200: return 0.5
    log_lags, log_rs = [], []
    for lag in lags:
        diffs = [ts[i+lag]-ts[i] for i in range(len(ts)-lag)]
        if len(diffs) < 20: continue
        s = std(diffs)
        if s <= 0: continue
        log_lags.append(math.log(lag))
        log_rs.append(math.log(s))
    if len(log_lags) < 5: return 0.5
    n = len(log_lags)
    mx = mean(log_lags); my = mean(log_rs)
    num = sum((log_lags[i]-mx)*(log_rs[i]-my) for i in range(n))
    den = sum((log_lags[i]-mx)**2 for i in range(n))
    return num/den if den else 0.5

def lead_lag_xcorr(xs, ys, max_lag=20):
    """Returns (best_lag, best_corr). Positive lag = ys leads xs."""
    best = (0, corr(xs, ys))
    for lag in range(1, max_lag+1):
        if len(xs)>lag and len(ys)>lag:
            c1 = corr(xs[lag:], ys[:-lag])  # xs lags ys by lag
            if abs(c1) > abs(best[1]): best = (-lag, c1)  # ys leads
            c2 = corr(xs[:-lag], ys[lag:])  # ys lags xs by lag
            if abs(c2) > abs(best[1]): best = (lag, c2)   # xs leads
    return best

def main():
    paths = [f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv" for d in [2,3,4]]
    print("Loading...", flush=True)
    mids_full = load_mids(paths)
    print(f"Loaded {len(mids_full)} products. Per-product samples:", min(len(v) for v in mids_full.values()), "to", max(len(v) for v in mids_full.values()), flush=True)

    # Use day 4 only for analysis (most recent), then sanity check on day 2+3 combined
    by_day = {}
    for p, rows in mids_full.items():
        for d, ts, m in rows:
            by_day.setdefault(d,{}).setdefault(p,[]).append((ts,m))
    for d in by_day:
        for p in by_day[d]:
            by_day[d][p] = [m for _,m in sorted(by_day[d][p])]

    # Per-category structural metrics on day 4
    DAY = 4
    print(f"\n{'='*100}\nPer-category analysis (day {DAY}, 10K ticks each)\n{'='*100}")
    cat_scores = []
    for cat, prods in CATEGORIES.items():
        series = {p: by_day[DAY].get(p, []) for p in prods}
        n = min(len(s) for s in series.values()) if series else 0
        if n < 100:
            print(f"\n[{cat}]  insufficient data ({n})"); continue
        # truncate to common length
        for p in series: series[p] = series[p][:n]

        # 1. mean pairwise correlation (high = clustered)
        cs = []
        for i in range(len(prods)):
            for j in range(i+1,len(prods)):
                cs.append(corr(series[prods[i]], series[prods[j]]))
        mean_corr = mean(cs)

        # 2. spread mean-reversion: take a representative pair (highest corr) and check Hurst
        max_pair = max(((i,j) for i in range(len(prods)) for j in range(i+1,len(prods))), key=lambda ij: abs(corr(series[prods[ij[0]]], series[prods[ij[1]]])))
        a, b = series[prods[max_pair[0]]], series[prods[max_pair[1]]]
        spread = [a[k]-b[k] for k in range(n)]
        h = hurst(spread)

        # 3. price ranges (helps gauge tradability — too tight = no edge after spread)
        ranges = [max(series[p])-min(series[p]) for p in prods]
        mean_range = mean(ranges)

        # 4. volatility of returns (rough)
        rets = [[(s[i+1]-s[i])/s[i] for i in range(n-1) if s[i]] for s in series.values()]
        vols = [std(r)*math.sqrt(10000) for r in rets]
        mean_vol = mean(vols)

        # score: high corr + low Hurst (mean-revert) + decent range = good pair-trade target
        exploitability = abs(mean_corr) * (1 - abs(h-0.5)*2) * min(mean_range/10, 1.0)
        cat_scores.append((cat, mean_corr, h, mean_range, mean_vol, exploitability, prods, max_pair))

        best_pair_str = f"{prods[max_pair[0]]} vs {prods[max_pair[1]]}"
        print(f"\n[{cat}] mean_pairwise_corr={mean_corr:+.3f}  spread_hurst={h:.3f}  mean_range={mean_range:.1f}  mean_vol={mean_vol:.3f}")
        print(f"  Best-corr pair: {best_pair_str}")
        print(f"  Exploitability score: {exploitability:.3f}")

    # Rank
    print(f"\n{'='*100}\nCATEGORIES RANKED BY EXPLOITABILITY (day {DAY})\n{'='*100}")
    for cat, mc, h, mr, mv, score, prods, mp in sorted(cat_scores, key=lambda x:-x[5]):
        h_label = "MEAN-REVERT" if h<0.45 else ("TREND" if h>0.55 else "RANDOM-WALK")
        print(f"  {cat:<14} score={score:.3f}  corr={mc:+.3f}  hurst={h:.3f} ({h_label})  range={mr:5.1f}  vol={mv:.3f}")

    # Lead-lag matrix on top 3 categories
    print(f"\n{'='*100}\nLEAD-LAG within top-3 categories (day {DAY})\n{'='*100}")
    for cat, mc, h, mr, mv, score, prods, mp in sorted(cat_scores, key=lambda x:-x[5])[:3]:
        print(f"\n[{cat}]")
        series = {p: by_day[DAY].get(p, [])[:min(len(by_day[DAY].get(p,[])) for p in prods)] for p in prods}
        for i in range(len(prods)):
            for j in range(i+1,len(prods)):
                a, b = series[prods[i]], series[prods[j]]
                # use returns for lead-lag (more sensitive than levels)
                ra = [(a[k+1]-a[k])/a[k] if a[k] else 0 for k in range(len(a)-1)]
                rb = [(b[k+1]-b[k])/b[k] if b[k] else 0 for k in range(len(b)-1)]
                lag, c = lead_lag_xcorr(ra, rb, max_lag=10)
                if abs(c) > 0.1:  # only print meaningful
                    direction = ">>" if lag > 0 else ("<<" if lag < 0 else "==")
                    print(f"  {prods[i]:<32} {direction} {prods[j]:<32}  lag={lag:+3d}  corr={c:+.3f}")

if __name__ == "__main__":
    main()
