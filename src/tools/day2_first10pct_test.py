"""Backtest v7 on JUST first 10% of Day 2 — same slice as leaderboard 1K-tick test.

If our 30K backtest +222K is real alpha, this slice should produce ~22K.
If we get ~5K, the leaderboard slice is structurally bad for us (real gap).
If we get ~25K, the leaderboard 5.9K is RW noise — Day 3 should restore us."""
import sys, os, importlib, csv, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Build filtered CSVs: Day 2 only, timestamp 0-99900 (first 10%)
src_prices = 'data/r3/prices/prices_round_3_day_2.csv'
src_trades = 'data/r3/trades/trades_round_3_day_2.csv'

tmpdir = tempfile.mkdtemp(prefix='day2_10pct_')
out_prices = os.path.join(tmpdir, 'prices_round_3_day_2.csv')
out_trades = os.path.join(tmpdir, 'trades_round_3_day_2.csv')

MAX_TS = 99900  # first 10% of Day 2

# Filter prices
n_in = n_out = 0
with open(src_prices) as f, open(out_prices, 'w', newline='') as o:
    header = f.readline()
    o.write(header)
    for line in f:
        n_in += 1
        parts = line.split(';')
        if len(parts) < 2:
            continue
        ts = int(parts[1])
        if ts <= MAX_TS:
            o.write(line)
            n_out += 1
print(f"prices: {n_in} -> {n_out} rows (ts<={MAX_TS})")

# Filter trades
nt_in = nt_out = 0
with open(src_trades) as f, open(out_trades, 'w', newline='') as o:
    header = f.readline()
    o.write(header)
    for line in f:
        nt_in += 1
        parts = line.split(';')
        if len(parts) < 2:
            continue
        try:
            ts = int(parts[0])  # trades CSV: timestamp is column 0
        except (ValueError, IndexError):
            continue
        if ts <= MAX_TS:
            o.write(line)
            nt_out += 1
print(f"trades: {nt_in} -> {nt_out} rows")

import backtest as bt
total, per_prod = bt.run_backtest([out_prices], [out_trades])

print()
print("=" * 60)
print(f"v7 ON DAY 2 FIRST 10% (same slice as leaderboard 1K test)")
print("=" * 60)
print(f"TOTAL: {total:+.0f}")
print()
print(f"{'Product':<25} {'PnL':>10}")
print("-" * 40)
for p in sorted(per_prod.keys()):
    print(f"{p:<25} {per_prod[p]:>+10.0f}")
print()
print(f"For reference: live shipped v7 = +5,944")
print(f"               full 30K backtest = +218,823 (~+7,300/1K-ticks)")
print(f"               competitor leaders = +25K to +50K on this slice")
