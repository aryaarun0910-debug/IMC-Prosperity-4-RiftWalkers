"""Parameter sweep to find quick wins on existing CONFIG fields.

Sweeps make_size, take_edge_pegged for HYD and VEF independently.
Reports both 30K and 1K-slice. Ship gates: 30K >= +233K AND 1K >= +12.9K.
"""
import sys, os, importlib, csv, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt
from glob import glob

src_prices = 'data/r3/prices/prices_round_3_day_2.csv'
src_trades = 'data/r3/trades/trades_round_3_day_2.csv'
tmpdir = tempfile.mkdtemp(prefix='ps_v8_')
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

variants = [
    ("baseline (ms60/ms60/te15/te5)", {}),
    # HYD make_size sweep
    ("HYD ms=80",  {"HYDROGEL_PACK": {"make_size": 80}}),
    ("HYD ms=100", {"HYDROGEL_PACK": {"make_size": 100}}),
    ("HYD ms=120", {"HYDROGEL_PACK": {"make_size": 120}}),
    ("HYD ms=40",  {"HYDROGEL_PACK": {"make_size": 40}}),
    # VEF make_size sweep
    ("VEF ms=80",  {"VELVETFRUIT_EXTRACT": {"make_size": 80}}),
    ("VEF ms=100", {"VELVETFRUIT_EXTRACT": {"make_size": 100}}),
    ("VEF ms=120", {"VELVETFRUIT_EXTRACT": {"make_size": 120}}),
    ("VEF ms=40",  {"VELVETFRUIT_EXTRACT": {"make_size": 40}}),
    # take_edge sweep
    ("HYD te=10",  {"HYDROGEL_PACK": {"take_edge_pegged": 10}}),
    ("HYD te=20",  {"HYDROGEL_PACK": {"take_edge_pegged": 20}}),
    ("VEF te=3",   {"VELVETFRUIT_EXTRACT": {"take_edge_pegged": 3}}),
    ("VEF te=8",   {"VELVETFRUIT_EXTRACT": {"take_edge_pegged": 8}}),
    # combined good candidates
    ("BOTH ms=100", {"HYDROGEL_PACK": {"make_size": 100}, "VELVETFRUIT_EXTRACT": {"make_size": 100}}),
    ("BOTH ms=80",  {"HYDROGEL_PACK": {"make_size": 80}, "VELVETFRUIT_EXTRACT": {"make_size": 80}}),
]

results = []
for label, overrides in variants:
    importlib.reload(bt)
    for prod, patches in overrides.items():
        existing = bt.tc.CONFIG.get(prod, {})
        bt.tc.CONFIG[prod] = {**existing, **patches}
    total30, per30 = bt.run_backtest(full_prices, full_trades)

    importlib.reload(bt)
    for prod, patches in overrides.items():
        existing = bt.tc.CONFIG.get(prod, {})
        bt.tc.CONFIG[prod] = {**existing, **patches}
    total1k, per1k = bt.run_backtest([slice_prices], [slice_trades])

    results.append((label, total30, per30.get('HYDROGEL_PACK', 0), per30.get('VELVETFRUIT_EXTRACT', 0),
                    total1k, per1k.get('HYDROGEL_PACK', 0), per1k.get('VELVETFRUIT_EXTRACT', 0)))

print()
print("=" * 110)
print(f"{'variant':<30} {'30K':>10} {'30K_HYD':>10} {'30K_VEF':>10} {'1K':>10} {'1K_HYD':>10} {'1K_VEF':>10}")
print("=" * 110)
for r in results:
    print(f"{r[0]:<30} {r[1]:>+10.0f} {r[2]:>+10.0f} {r[3]:>+10.0f} {r[4]:>+10.0f} {r[5]:>+10.0f} {r[6]:>+10.0f}")
print()
print("Ship gates:  30K >= +233,000 AND 1K >= +12,891")
