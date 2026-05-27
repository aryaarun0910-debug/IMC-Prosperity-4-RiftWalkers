"""Walk-forward validation: pick config on day 2+3, evaluate on day 4 (held out)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_systematic_mr import load_with_books, mr_backtest, ALL_PRODUCTS

GRIDS = [
    (1.5, 1.0, 500), (1.5, 1.5, 500), (1.5, 2.0, 500),
    (2.0, 1.5, 500), (2.0, 2.0, 500), (2.5, 2.0, 500),
    (1.5, None, 500),
]

def main():
    days = {d: load_with_books(f"C:/Users/aryaa/Documents/IMC LOCK IN/PIPELINE/data/r5/prices/prices_round_5_day_{d}.csv") for d in [2,3,4]}

    print("Walk-forward: train on day 2+3, test on day 4")
    print("="*120)
    print(f"{'Product':<32} {'cfg':>20} {'train_d2+d3':>14} {'test_d4':>10} {'train_avg':>10} {'verdict':>10}")
    print("-"*120)
    survivors = []
    for p in ALL_PRODUCTS:
        if p not in days[2]: continue
        # find best cfg by train (d2+d3 sum, requires both positive >$500)
        best_cfg = None
        best_train = -1e18
        for z_in, mt, w in GRIDS:
            d2_pnl, _ = mr_backtest(days[2][p], window=w, z_in=z_in, mom_thresh=mt)
            d3_pnl, _ = mr_backtest(days[3][p], window=w, z_in=z_in, mom_thresh=mt)
            train_total = d2_pnl + d3_pnl
            # require BOTH train days positive AND > $500
            if d2_pnl > 500 and d3_pnl > 500 and train_total > best_train:
                best_train = train_total
                best_cfg = (z_in, mt, w, d2_pnl, d3_pnl)
        if best_cfg is None: continue
        z_in, mt, w, d2_pnl, d3_pnl = best_cfg
        # evaluate on test
        d4_pnl, _ = mr_backtest(days[4][p], window=w, z_in=z_in, mom_thresh=mt)
        verdict = "PASS" if d4_pnl > 0 else "FAIL"
        cfg_str = f"z={z_in},mom={mt}"
        print(f"  {p:<32} {cfg_str:>20} {best_train:>+14.0f} {d4_pnl:>+10.0f} {best_train/2:>+10.0f} {verdict:>10}")
        if d4_pnl > 0:
            survivors.append((p, z_in, mt, w, best_train, d4_pnl))

    # Survivors summary
    print()
    print("="*120)
    print(f"SURVIVORS (positive on held-out day 4)")
    print("="*120)
    survivors.sort(key=lambda x: -x[5])
    grand_train = 0; grand_test = 0
    for p, z_in, mt, w, train, test in survivors:
        print(f"  {p:<32} cfg=(z={z_in},mom={mt})  train(d2+d3)={train:+8.0f}  test(d4)={test:+8.0f}")
        grand_train += train
        grand_test += test
    print(f"\n  TOTALS: train={grand_train:+.0f}  test={grand_test:+.0f}  test/train_avg = {grand_test/(grand_train/2 + 0.01):.1%}")

if __name__ == "__main__":
    main()
