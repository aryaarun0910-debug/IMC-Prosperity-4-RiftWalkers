"""V8 sweep: HYD FV at higher levels (10015-10040) PLUS v7 OTM extensions."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest as bt

r4_p = [f"data/r4/prices/prices_round_4_day_{d}.csv" for d in [1,2,3]]
r4_t = [f"data/r4/trades/trades_round_4_day_{d}.csv" for d in [1,2,3]]

OTM = lambda short, stop: {
    "type": "otm_short", "position_limit": 300,
    "target_short": short, "stop_loss_mid": stop, "max_sell_per_tick": 30,
}

# v6 baseline OTM config
v6 = {
    "VEV_5300": OTM(200, 100),
    "VEV_5400": OTM(200, 50),
    "VEV_5500": OTM(200, 30),
}

print()
print("=" * 90)
print(f"{'variant':<40} {'R4_30K':>10} {'R4_HYD':>10} {'R4_VEV_OTM':>12}")
print("=" * 90)

# 1. HYD FV sweep with current OTM
for fv in [10010, 10015, 10020, 10025, 10030, 10031, 10035, 10040]:
    importlib.reload(bt)
    bt.tc.CONFIG["HYDROGEL_PACK"]["fair_value"] = fv
    for prod, c in v6.items():
        bt.tc.CONFIG[prod] = c
    r4_30, per = bt.run_backtest(r4_p, r4_t)
    hyd = per.get("HYDROGEL_PACK", 0)
    vev_otm = per.get("VEV_5300", 0) + per.get("VEV_5400", 0) + per.get("VEV_5500", 0)
    print(f"FV={fv:<6} (v6 OTM 200)                    {r4_30:>+10.0f} {hyd:>+10.0f} {vev_otm:>+12.0f}", flush=True)

print()
print("Friend's live config: FV=10031 → +25,351 live")
print("Backtest of FV=10031 should help us decide ship vs match-friend.")
