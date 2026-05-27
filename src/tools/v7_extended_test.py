"""V7 extended test: OTM_short on more strikes + VEV_4500 enable + Mark 14/38 arb tighter."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]
r3_p = [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0,1,2]]
r3_t = [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0,1,2]]

OTM = lambda short, stop: {
    "type": "otm_short", "position_limit": 300,
    "target_short": short, "stop_loss_mid": stop, "max_sell_per_tick": 30,
}
INTRINSIC = lambda strike: {
    "type": "voucher_intrinsic_mm", "position_limit": 300,
    "underlying": "VELVETFRUIT_EXTRACT", "strike": strike,
    "make_size": 15, "make_edge": 10, "take_edge": 8,
    "max_pos_frac": 0.6, "inv_skew_ticks": 4,
}

# v6 baseline already has VEV_5300/5400/5500 short=200
v6_default = {
    "VEV_5300": OTM(200, 100),
    "VEV_5400": OTM(200, 50),
    "VEV_5500": OTM(200, 30),
}

variants = [
    ("v6 baseline (5300/5400/5500 short=200)", v6_default),
    ("v7-A: + VEV_5200 short=100 stop=130", {**v6_default, "VEV_5200": OTM(100, 130)}),
    ("v7-B: + VEV_5200 short=200 stop=130", {**v6_default, "VEV_5200": OTM(200, 130)}),
    ("v7-C: + VEV_5100 short=50 stop=200", {**v6_default, "VEV_5100": OTM(50, 200)}),
    ("v7-D: + 5200 short=150 + 5100 short=50", {**v6_default, "VEV_5200": OTM(150, 130), "VEV_5100": OTM(50, 200)}),
    ("v7-E: + VEV_4500 intrinsic_mm", {**v6_default, "VEV_4500": INTRINSIC(4500)}),
    ("v7-F: + 5200 short=200 + 4500 intrinsic", {**v6_default, "VEV_5200": OTM(200, 130), "VEV_4500": INTRINSIC(4500)}),
]

print()
print("=" * 110)
print(f"{'variant':<48} {'R4_30K':>10} {'R3_30K':>10} {'V52':>7} {'V51':>7} {'V45':>7}")
print("=" * 110)
for label, cfgs in variants:
    importlib.reload(bt)
    for prod, c in cfgs.items():
        bt.tc.CONFIG[prod] = c
    r4_30, r4_per = bt.run_backtest(r4_p, r4_t)
    importlib.reload(bt)
    for prod, c in cfgs.items():
        bt.tc.CONFIG[prod] = c
    r3_30, _ = bt.run_backtest(r3_p, r3_t)
    print(f"{label:<48} {r4_30:>+10.0f} {r3_30:>+10.0f} "
          f"{r4_per.get('VEV_5200',0):>+7.0f} {r4_per.get('VEV_5100',0):>+7.0f} {r4_per.get('VEV_4500',0):>+7.0f}", flush=True)
print()
print("Ship: variant must beat v6 baseline on R4_30K AND show positive new-strike PnL.")
