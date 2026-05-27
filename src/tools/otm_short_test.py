"""OTM_short strategy test on R4 historical 30K. Validates theta capture."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]
r3_p = [f"data/r3/prices/prices_round_3_day_{d}.csv" for d in [0,1,2]]
r3_t = [f"data/r3/trades/trades_round_3_day_{d}.csv" for d in [0,1,2]]

OTM_CFG = lambda strike, target=100, stop=999999: {
    "type": "otm_short",
    "position_limit": 300,
    "target_short": target,
    "stop_loss_mid": stop,
    "max_sell_per_tick": 30,
}

variants = [
    ("v5 baseline (all OTM do_nothing)", {}),
    ("v6-A: 5300/5400/5500 short=50", {
        "VEV_5300": OTM_CFG(5300, 50, 100),
        "VEV_5400": OTM_CFG(5400, 50, 50),
        "VEV_5500": OTM_CFG(5500, 50, 30),
    }),
    ("v6-B: 5300/5400/5500 short=100", {
        "VEV_5300": OTM_CFG(5300, 100, 100),
        "VEV_5400": OTM_CFG(5400, 100, 50),
        "VEV_5500": OTM_CFG(5500, 100, 30),
    }),
    ("v6-C: 5300/5400/5500 short=200", {
        "VEV_5300": OTM_CFG(5300, 200, 100),
        "VEV_5400": OTM_CFG(5400, 200, 50),
        "VEV_5500": OTM_CFG(5500, 200, 30),
    }),
    ("v6-D: 5300 only short=200", {
        "VEV_5300": OTM_CFG(5300, 200, 100),
    }),
    ("v6-E: 5500 only short=200", {
        "VEV_5500": OTM_CFG(5500, 200, 30),
    }),
]

print()
print("=" * 110)
print(f"{'variant':<40} {'R4_30K':>10} {'R3_30K':>10} {'V53':>8} {'V54':>8} {'V55':>8}")
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
    print(f"{label:<40} {r4_30:>+10.0f} {r3_30:>+10.0f} "
          f"{r4_per.get('VEV_5300',0):>+8.0f} {r4_per.get('VEV_5400',0):>+8.0f} {r4_per.get('VEV_5500',0):>+8.0f}", flush=True)
print()
print("Ship: variant must beat baseline on R4_30K AND show positive V53/V54/V55 PnL.")
