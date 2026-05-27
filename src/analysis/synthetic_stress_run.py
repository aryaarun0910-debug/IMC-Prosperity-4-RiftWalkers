"""K7 runner: generate synthetic regimes via synthetic_stress_gen.py, then
execute trader.py against each via backtest.run_backtest().

Outputs PnL-per-regime to analysis/findings/k7_synthetic_results.json.

Usage: py -3.12 analysis/synthetic_stress_run.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    regimes = ["trending", "mean_revert", "crash", "high_vol", "low_vol", "regime_switch"]
    product = "ASH_COATED_OSMIUM"  # use a real product so trader.py routes it sensibly
    ticks = 10000

    # Step 1: generate
    for r in regimes:
        out_dir = ROOT / f"data/synthetic/{r}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["py", "-3.12", "analysis/synthetic_stress_gen.py",
               "--regime", r, "--product", product, "--ticks", str(ticks),
               "--days", "1", "--out", "data/synthetic"]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
        # Synthesize empty trades CSV
        trade_path = out_dir / "trades_round_0_day_0.csv"
        with open(trade_path, "w") as f:
            f.write("timestamp;buyer;seller;symbol;currency;price;quantity\n")

    # Step 2: run backtest per regime
    from backtest import run_backtest
    results = {}
    for r in regimes:
        price_csvs = [str(ROOT / f"data/synthetic/{r}/prices_round_0_day_0.csv")]
        trade_csvs = [str(ROOT / f"data/synthetic/{r}/trades_round_0_day_0.csv")]
        try:
            total, pnls = run_backtest(price_csvs, trade_csvs, seed=42)
            results[r] = {"per_product": dict(pnls), "total": int(total)}
            print(f"{r:>15s}  total={int(total):+,d}")
        except Exception as e:
            results[r] = {"error": str(e)}
            print(f"{r:>15s}  ERROR: {e}")

    out = ROOT / "analysis/findings/k7_synthetic_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
