"""Replay a live server log via backtest harness.

Converts activitiesLog (CSV string in log JSON) and tradeHistory (JSON list)
into the price/trade CSV format that backtest.py consumes, then runs
backtest.run_backtest against them. Result = "what PnL would our CURRENT
trader.py achieve if rerun against the exact live order book and bot trades".

Usage:
    py -3.12 tools/live_replay.py <log.json>

If --config-overrides is provided as a JSON dict, applies to trader.CONFIG.
"""
import sys, os, json, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest as bt

def convert(log_path, work_dir):
    with open(log_path, 'r') as f:
        d = json.load(f)
    activities = d['activitiesLog']
    trades = d.get('tradeHistory', [])

    os.makedirs(work_dir, exist_ok=True)
    price_csv = os.path.join(work_dir, 'prices.csv')
    trade_csv = os.path.join(work_dir, 'trades.csv')

    # activitiesLog: header already present, day column already there (=2 in log)
    # backtest expects day column too. Just rewrite.
    with open(price_csv, 'w', newline='') as f:
        f.write(activities if activities.endswith('\n') else activities + '\n')

    # tradeHistory: list of dicts. Filter to bot-to-bot only (exclude SUBMISSION)
    # Backtest's "trades" CSV expects: timestamp;buyer;seller;symbol;currency;price;quantity
    with open(trade_csv, 'w', newline='') as f:
        f.write('timestamp;buyer;seller;symbol;currency;price;quantity\n')
        for t in trades:
            if t.get('buyer') == 'SUBMISSION' or t.get('seller') == 'SUBMISSION':
                continue
            f.write(f"{t['timestamp']};{t.get('buyer','')};{t.get('seller','')};{t['symbol']};{t.get('currency','XIRECS')};{t['price']};{t['quantity']}\n")

    return price_csv, trade_csv

def apply_cfg_overrides(overrides):
    """overrides = {product: {field: value, ...}, ...}"""
    if not overrides:
        return
    for prod, fields in overrides.items():
        if prod in bt.tc.CONFIG:
            for k, v in fields.items():
                bt.tc.CONFIG[prod][k] = v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', help='path to <id>.log file (has tradeHistory; .json may not)')
    ap.add_argument('--cfg', default=None, help='JSON dict of {product: {field: value}} overrides')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    work_dir = '/tmp/live_replay'
    price_csv, trade_csv = convert(args.log, work_dir)
    if not args.quiet:
        print(f"Converted: {price_csv}, {trade_csv}")

    if args.cfg:
        overrides = json.loads(args.cfg)
        apply_cfg_overrides(overrides)
        if not args.quiet:
            print(f"Applied overrides: {overrides}")

    total, per_prod = bt.run_backtest([price_csv], [trade_csv])
    return total, per_prod

if __name__ == '__main__':
    main()
