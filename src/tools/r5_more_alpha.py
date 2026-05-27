"""Find more alpha: relax walk-forward criteria and expand grid search.

For each product, try MORE configs (8 z_in × 6 mom_thresh × 2 windows = 96 configs each).
Pick best on train (d2+d3), require: BOTH train days > 0 + test (d4) > 0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_systematic_mr import load_with_books, mr_backtest, ALL_PRODUCTS

# Larger grid
GRIDS = []
for window in [500, 1000]:
    for z_in in [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
        for mt in [None, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
            GRIDS.append((z_in, mt, window))

EXISTING = set([
    "PEBBLES_XL","PEBBLES_M","PEBBLES_S",
    "TRANSLATOR_ASTRO_BLACK","TRANSLATOR_VOID_BLUE",
    "MICROCHIP_CIRCLE","MICROCHIP_RECTANGLE",
    "SLEEP_POD_NYLON",
])

def main():
    days = {d: load_with_books(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    print(f"Testing {len(GRIDS)} configs × {len(ALL_PRODUCTS) - len(EXISTING)} new products...")
    print(f"Walk-forward: train d2+d3 (both>0), test d4 (>0)")

    # Restrict to products NOT in existing v3.2
    new_candidates = [p for p in ALL_PRODUCTS if p not in EXISTING]

    findings = []
    for p in new_candidates:
        best_cfg = None
        best_train = -1e18
        best_test = None
        for z_in, mt, w in GRIDS:
            d2_pnl, _ = mr_backtest(days[2][p], window=w, z_in=z_in, mom_thresh=mt)
            d3_pnl, _ = mr_backtest(days[3][p], window=w, z_in=z_in, mom_thresh=mt)
            if d2_pnl > 0 and d3_pnl > 0:
                train_total = d2_pnl + d3_pnl
                if train_total > best_train:
                    best_train = train_total
                    best_cfg = (z_in, mt, w, d2_pnl, d3_pnl)
        if best_cfg is None: continue
        z_in, mt, w, d2, d3 = best_cfg
        d4_pnl, _ = mr_backtest(days[4][p], window=w, z_in=z_in, mom_thresh=mt)
        if d4_pnl > 0:
            findings.append((p, z_in, mt, w, d2, d3, d4_pnl))

    findings.sort(key=lambda x: -x[6])  # by d4 PnL
    print()
    print("="*120)
    print("NEW ALPHA CANDIDATES (passed train + test on day 4)")
    print("="*120)
    print(f"{'Product':<32} {'cfg':>20} {'d2':>8} {'d3':>8} {'d4':>8} {'train_avg':>10} {'test/avg':>10}")
    for p, z_in, mt, w, d2, d3, d4 in findings:
        cfg = f"z={z_in},mom={mt},w={w}"
        ratio = d4 / ((d2+d3)/2 + 0.01)
        print(f"  {p:<32} {cfg:>20} {d2:+8.0f} {d3:+8.0f} {d4:+8.0f} {(d2+d3)/2:+10.0f} {ratio:>9.0%}")
    if not findings:
        print("  (no new candidates)")

if __name__ == "__main__":
    main()
