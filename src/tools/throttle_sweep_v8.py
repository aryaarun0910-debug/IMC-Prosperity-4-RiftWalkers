"""Throttle sweep for HYD+VEF — find if looser inventory caps capture leader-style PnL.

Field intel: leaders show +37K to +85K on Day 2 first 10% live. We're at +7.5K.
Their drawdowns (-50K observed) prove they run un-throttled. Our 100/180 throttle
caps risk but caps reward. Test if a looser setting recovers some upside without
breaking backtest.

Tests EACH variant on:
  (a) Full 30K p4r3 — must stay >= +220K to ship (no worse than v7 baseline)
  (b) Day 2 first 10% slice (1K ticks) — must show > +12K to be worth shipping
"""
import sys, os, importlib, csv, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

# === Build Day 2 first 10% slice CSVs ===
src_prices = 'data/r3/prices/prices_round_3_day_2.csv'
src_trades = 'data/r3/trades/trades_round_3_day_2.csv'
tmpdir = tempfile.mkdtemp(prefix='throttle_d2_')
slice_prices = os.path.join(tmpdir, 'prices_round_3_day_2.csv')
slice_trades = os.path.join(tmpdir, 'trades_round_3_day_2.csv')
MAX_TS = 99900

with open(src_prices) as f, open(slice_prices, 'w', newline='') as o:
    o.write(f.readline())
    for line in f:
        parts = line.split(';')
        if len(parts) >= 2 and int(parts[1]) <= MAX_TS:
            o.write(line)
with open(src_trades) as f, open(slice_trades, 'w', newline='') as o:
    o.write(f.readline())
    for line in f:
        parts = line.split(';')
        if len(parts) >= 2:
            try:
                if int(parts[0]) <= MAX_TS:
                    o.write(line)
            except (ValueError, IndexError):
                continue

full_prices = sorted(glob('data/r3/prices/prices_round_3_day_*.csv'))
full_trades = sorted(glob('data/r3/trades/trades_round_3_day_*.csv'))

# === Variants ===
# (label, hyd_thr1, hyd_thr2, vef_thr1, vef_thr2)  — None means throttle OFF
variants = [
    ("v7 baseline (100/180)",     100, 180, 100, 180),
    ("loose (130/195)",           130, 195, 130, 195),
    ("looser (160/200)",          160, 200, 160, 200),
    ("OFF (HYD)",                 None, None, 100, 180),
    ("OFF (VEF)",                 100, 180, None, None),
    ("OFF (BOTH)",                None, None, None, None),
    ("HYD_loose+VEF_keep",        130, 195, 100, 180),
    ("HYD_keep+VEF_loose",        100, 180, 130, 195),
]

def apply_variant(htr1, htr2, vtr1, vtr2):
    if htr1 is None:
        bt.tc.CONFIG["HYDROGEL_PACK"]["throttle_takes_by_inventory"] = False
    else:
        bt.tc.CONFIG["HYDROGEL_PACK"]["throttle_takes_by_inventory"] = True
        bt.tc.CONFIG["HYDROGEL_PACK"]["inventory_threshold_1"] = htr1
        bt.tc.CONFIG["HYDROGEL_PACK"]["inventory_threshold_2"] = htr2
    if vtr1 is None:
        bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["throttle_takes_by_inventory"] = False
    else:
        bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["throttle_takes_by_inventory"] = True
        bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["inventory_threshold_1"] = vtr1
        bt.tc.CONFIG["VELVETFRUIT_EXTRACT"]["inventory_threshold_2"] = vtr2

print()
print("=" * 110)
print(f"{'variant':<28} {'30K_TOTAL':>12} {'30K_HYD':>10} {'30K_VEF':>10} {'1K_TOTAL':>10} {'1K_HYD':>10} {'1K_VEF':>10}")
print("=" * 110)

for label, htr1, htr2, vtr1, vtr2 in variants:
    # 30K full backtest
    importlib.reload(bt)
    apply_variant(htr1, htr2, vtr1, vtr2)
    total30, per30 = bt.run_backtest(full_prices, full_trades)
    hyd30 = per30.get('HYDROGEL_PACK', 0)
    vef30 = per30.get('VELVETFRUIT_EXTRACT', 0)

    # 1K Day 2 first 10% slice
    importlib.reload(bt)
    apply_variant(htr1, htr2, vtr1, vtr2)
    total1k, per1k = bt.run_backtest([slice_prices], [slice_trades])
    hyd1k = per1k.get('HYDROGEL_PACK', 0)
    vef1k = per1k.get('VELVETFRUIT_EXTRACT', 0)

    print(f"{label:<28} {total30:>+12.0f} {hyd30:>+10.0f} {vef30:>+10.0f} {total1k:>+10.0f} {hyd1k:>+10.0f} {vef1k:>+10.0f}", flush=True)

print()
print("Ship gates:  30K_TOTAL >= +220,000 AND 1K_TOTAL >= +12,000")
print("Baseline reference: v7 30K=+218,823 / 1K=+11,841")
