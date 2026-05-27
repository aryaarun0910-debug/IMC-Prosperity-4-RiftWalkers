"""Analyze v7 live log 412166: WHERE does HYD/VEF bleed live PnL vs backtest?

Goal: identify the systematic difference between backtest and live behavior so
we can fix the 95% PnL evaporation (backtest +218K -> live +5.9K)."""
import json, sys, os
from collections import defaultdict

LOG = "data/r3/server_logs/412166/412166.log"

with open(LOG) as f:
    data = json.load(f)

print(f"Submission: {data.get('submissionId', 'unk')[:8]}...")
print(f"Keys: {list(data.keys())}")

# activitiesLog is a CSV string with prices
# tradeHistory and ours_trades may be separate

# Parse activitiesLog (PnL by product per tick)
actlog = data.get('activitiesLog', '')
lines = actlog.strip().split('\n')
header = lines[0].split(';')
# day;timestamp;product;...;mid_price;profit_and_loss

# Per-product PnL trajectory
pnl_traj = defaultdict(list)  # product -> [(ts, pnl, mid)]
for line in lines[1:]:
    fields = line.split(';')
    if len(fields) < len(header):
        continue
    row = dict(zip(header, fields))
    try:
        day = int(row['day'])
        ts = int(row['timestamp']) + day * 1_000_000
        prod = row['product']
        pnl = float(row['profit_and_loss']) if row['profit_and_loss'] else 0.0
        mid = float(row['mid_price']) if row['mid_price'] else 0.0
        pnl_traj[prod].append((ts, pnl, mid))
    except (ValueError, KeyError):
        continue

print(f"\nProducts: {sorted(pnl_traj.keys())}")
print(f"\n=== PER-PRODUCT FINAL PnL ===")
total = 0
for prod in sorted(pnl_traj.keys()):
    if pnl_traj[prod]:
        final_pnl = pnl_traj[prod][-1][1]
        first_mid = pnl_traj[prod][0][2]
        last_mid = pnl_traj[prod][-1][2]
        n = len(pnl_traj[prod])
        print(f"  {prod:<25} final_pnl={final_pnl:>+10.0f}  ticks={n:>4}  mid={first_mid:>8.1f}->{last_mid:>8.1f}  drift={last_mid-first_mid:>+7.1f}")
        total += final_pnl
print(f"\n  TOTAL = {total:+.0f}")

# PnL trajectory for HYD/VEF — find max drawdown and final
print(f"\n=== HYD PnL TRAJECTORY ===")
hyd = pnl_traj.get('HYDROGEL_PACK', [])
if hyd:
    pnls = [p[1] for p in hyd]
    mids = [p[2] for p in hyd]
    print(f"  ticks: {len(hyd)}")
    print(f"  PnL min/max/final: {min(pnls):+.0f} / {max(pnls):+.0f} / {pnls[-1]:+.0f}")
    print(f"  mid min/max/final: {min(mids):.1f} / {max(mids):.1f} / {mids[-1]:.1f}")
    # 100-tick checkpoints
    for i in range(0, len(hyd), max(1, len(hyd)//10)):
        ts, pnl, mid = hyd[i]
        print(f"    ts={ts:>8}  pnl={pnl:>+8.0f}  mid={mid:.1f}")
    print(f"    ts={hyd[-1][0]:>8}  pnl={hyd[-1][1]:>+8.0f}  mid={hyd[-1][2]:.1f}  (final)")

print(f"\n=== VEF PnL TRAJECTORY ===")
vef = pnl_traj.get('VELVETFRUIT_EXTRACT', [])
if vef:
    pnls = [p[1] for p in vef]
    mids = [p[2] for p in vef]
    print(f"  ticks: {len(vef)}")
    print(f"  PnL min/max/final: {min(pnls):+.0f} / {max(pnls):+.0f} / {pnls[-1]:+.0f}")
    print(f"  mid min/max/final: {min(mids):.1f} / {max(mids):.1f} / {mids[-1]:.1f}")
    for i in range(0, len(vef), max(1, len(vef)//10)):
        ts, pnl, mid = vef[i]
        print(f"    ts={ts:>8}  pnl={pnl:>+8.0f}  mid={mid:.1f}")
    print(f"    ts={vef[-1][0]:>8}  pnl={vef[-1][1]:>+8.0f}  mid={vef[-1][2]:.1f}  (final)")
