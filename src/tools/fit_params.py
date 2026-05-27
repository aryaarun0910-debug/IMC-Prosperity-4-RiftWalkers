"""
Gameday Parameter Fitting — Run when new round data drops.

Given price + trade CSVs from a new P4 round, this script:
1. Classifies every product by archetype (pegged, AR, volatile, basket, options, conversion)
2. Fits optimal parameters for each strategy type
3. Outputs a CONFIG block ready to paste into trader.py
4. Fits vol smile coefficients for options products

Usage:
    py -3.12 tools/fit_params.py --prices data/r1/prices_*.csv
    py -3.12 tools/fit_params.py --prices data/r1/prices_*.csv --trades data/r1/trades_*.csv
"""

import sys
import os
import csv
import math
import json
import argparse
from collections import defaultdict

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _ROOT_DIR)


def load_csvs(file_patterns):
    """Load and merge CSVs. Returns list of dicts."""
    rows = []
    for f in file_patterns:
        if not os.path.exists(f):
            continue
        with open(f, 'r') as fh:
            reader = csv.DictReader(fh, delimiter=';')
            for row in reader:
                rows.append(row)
    return rows


def extract_product_data(price_rows):
    """Extract per-product time series from price CSV rows."""
    products = defaultdict(lambda: {"mids": [], "spreads": [], "bids": [], "asks": [],
                                      "bid_vols": [], "ask_vols": [], "timestamps": []})
    for row in price_rows:
        try:
            product = row.get("product", "")
            bid = float(row.get("bid_price_1", 0))
            ask = float(row.get("ask_price_1", 0))
            bid_vol = int(float(row.get("bid_volume_1", 0)))
            ask_vol = int(float(row.get("ask_volume_1", 0)))
            ts = int(float(row.get("timestamp", 0)))
            day = int(float(row.get("day", 0)))
        except (ValueError, TypeError):
            continue

        if bid <= 0 or ask <= 0 or ask <= bid:
            continue

        mid = (bid + ask) / 2.0
        spread = ask - bid

        p = products[product]
        p["mids"].append(mid)
        p["spreads"].append(spread)
        p["bids"].append(bid)
        p["asks"].append(ask)
        p["bid_vols"].append(bid_vol)
        p["ask_vols"].append(ask_vol)
        p["timestamps"].append((day, ts))

    return dict(products)


def classify_product(name, data, all_products):
    """Classify a product by its statistical properties."""
    mids = data["mids"]
    spreads = data["spreads"]

    if len(mids) < 50:
        return "generic_mm", {}

    mean_mid = sum(mids) / len(mids)
    std_mid = (sum((x - mean_mid)**2 for x in mids) / len(mids)) ** 0.5
    cv = std_mid / mean_mid if mean_mid > 0 else 999  # coefficient of variation
    mean_spread = sum(spreads) / len(spreads)
    spread_pct = mean_spread / mean_mid if mean_mid > 0 else 999

    # Check for conversion product keywords
    conv_keywords = ["MACARON", "ORCHID"]
    if any(kw in name.upper() for kw in conv_keywords):
        return "conversion_arb", {"fair_value": round(mean_mid)}

    # Check for options
    opt_patterns = ["_VOUCHER_", "_COUPON_", "_CALL_", "_PUT_", "_OPTION_"]
    for pat in opt_patterns:
        if pat in name:
            base = name.split(pat)[0]
            try:
                strike = float(name.split(pat)[-1])
            except (ValueError, TypeError):
                strike = mean_mid
            return "options", {"underlying": base, "strike": strike}

    # Check for basket keywords
    basket_keywords = ["BASKET", "ETF", "INDEX", "BUNDLE", "HAMPER", "BOX", "CRATE"]
    if any(kw in name.upper() for kw in basket_keywords):
        return "basket_arb", {}

    # Pegged: very low CV (< 0.1%)
    if cv < 0.001:
        return "pegged", {"fair_value": round(mean_mid)}

    # Wide spread: spread > 0.15% of mid or absolute spread > 8
    if spread_pct > 0.0015 or mean_spread > 8:
        return "wide_spread", {}

    # High volatility + tight spread = olivia_follow (e.g. SQUID_INK CV=2.87%)
    # vs AR/trending = moderate volatility + tight spread (e.g. KELP CV=0.47%)
    if cv > 0.015:
        return "olivia_follow", {}

    # AR/trending: moderate volatility, tight spread
    return "ar_olivia", {}


def fit_pegged(name, data):
    """Fit parameters for pegged product."""
    mids = data["mids"]
    fv = round(sum(mids) / len(mids))
    return {
        "type": "pegged",
        "fair_value": fv,
        "position_limit": 50,
        "take_width": 1,
        "make_offset": 4,
        "make_size": 50,
        "inventory_threshold_1": 45,
    }


def fit_ar_olivia(name, data):
    """Fit parameters for AR + Olivia product."""
    mids = data["mids"]
    spreads = data["spreads"]
    mean_spread = sum(spreads) / len(spreads)
    return {
        "type": "ar_olivia",
        "position_limit": 50,
        "ar_order": 5,
        "ar_refit_every": 25,
        "ar_min_history": 50,
        "olivia_window": 5,
        "make_size": 30,
    }


def fit_wide_spread(name, data):
    """Fit parameters for wide-spread product."""
    mids = data["mids"]
    spreads = data["spreads"]
    mean_spread = sum(spreads) / len(spreads)
    return {
        "type": "wide_spread",
        "position_limit": 50,
        "take_ask_threshold": max(2, round(mean_spread * 0.8)),
        "bid_comp_threshold": max(2, round(mean_spread * 0.6)),
        "ema_alpha": 0.005,
        "ema_bid_skip_thr": round(mean_spread * 0.7, 1),
        "ema_ask_skip_thr": round(mean_spread * 0.35, 1),
        "make_size": 50,
        "inventory_threshold_1": 40,
        "inventory_threshold_2": 48,
    }


def fit_olivia_follow(name, data):
    """Fit parameters for Olivia-follow product."""
    return {
        "type": "olivia_follow",
        "position_limit": 50,
        "target_position": 50,
        "vol_window": 50,
        "vol_mr_threshold": 2.0,
        "make_size": 15,
    }


def fit_basket(name, data, all_products):
    """Fit parameters for basket product."""
    # Try to discover components from correlated products
    return {
        "type": "basket_arb",
        "position_limit": 60,
        "components": {},  # will be discovered at runtime
        "premium_mean": 0.0,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "trade_size": 60,
        "hedge_ratio": 0.0,
        "auto_discover_weights": True,
    }


def fit_conversion(name, data):
    """Fit parameters for conversion arb product."""
    return {
        "type": "conversion_arb",
        "position_limit": 75,
        "max_conversions": 10,
        "order_size": 50,
        "max_short_stockpile": 30,
    }


def fit_options(name, data, all_products, all_data):
    """Fit options parameters including vol smile."""
    base_cfg = {
        "type": "options",
        "position_limit": 200,
        "underlying": data.get("underlying", ""),
        "strike": data.get("strike", 10000),
        "total_option_days": 8,
        "day_offset": 0,
        "theo_norm_window": 20,
        "iv_scalp_window": 100,
        "iv_scalp_thr": 0.7,
        "thr_open": 0.5,
        "thr_close": 0.0,
        "low_vega_thr_adj": 0.5,
        "ul_mr_window": 10,
        "ul_mr_thr": 15,
        "opt_mr_window": 30,
        "opt_mr_thr": 5,
        "underlying_limit": 400,
    }
    # Smile coefficients will be auto-fitted at runtime by _auto_fit_smile
    # We don't hardcode them here — they'll be discovered from live data
    return base_cfg


def fit_vol_smile(all_data, underlying_name):
    """Fit quadratic vol smile from all option products sharing an underlying.
    Returns [a, b, c] coefficients or None."""
    # This would require BS model + IV computation — heavy.
    # For gameday, the auto-fitter in trader.py handles this live.
    # This function is for offline analysis only.
    return None


def generate_config(product_configs):
    """Generate Python CONFIG dict string."""
    lines = ["CONFIG = {"]
    for name, cfg in sorted(product_configs.items()):
        lines.append(f'    "{name}": {{')
        for k, v in cfg.items():
            if isinstance(v, str):
                lines.append(f'        "{k}": "{v}",')
            elif isinstance(v, bool):
                lines.append(f'        "{k}": {str(v)},')
            elif isinstance(v, dict):
                lines.append(f'        "{k}": {json.dumps(v)},')
            elif isinstance(v, list):
                lines.append(f'        "{k}": {json.dumps(v)},')
            else:
                lines.append(f'        "{k}": {v},')
        lines.append('    },')
    lines.append('}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Fit strategy parameters from price/trade data")
    parser.add_argument("--prices", nargs="+", required=True, help="Price CSV files")
    parser.add_argument("--trades", nargs="+", default=[], help="Trade CSV files")
    parser.add_argument("--output", type=str, default=None, help="Output file for CONFIG")
    args = parser.parse_args()

    print("Loading price data...")
    price_rows = load_csvs(args.prices)
    print(f"  {len(price_rows)} price rows loaded")

    print("\nExtracting product data...")
    all_data = extract_product_data(price_rows)
    print(f"  {len(all_data)} products found: {list(all_data.keys())}")

    print("\n" + "="*60)
    print("PRODUCT ANALYSIS")
    print("="*60)

    configs = {}
    for name, data in sorted(all_data.items()):
        mids = data["mids"]
        spreads = data["spreads"]
        mean_mid = sum(mids) / len(mids) if mids else 0
        std_mid = (sum((x - mean_mid)**2 for x in mids) / len(mids)) ** 0.5 if mids else 0
        mean_spread = sum(spreads) / len(spreads) if spreads else 0

        ptype, extra = classify_product(name, data, all_data)

        print(f"\n  {name}:")
        print(f"    Mid: {mean_mid:.1f} (std: {std_mid:.2f}, CV: {std_mid/mean_mid*100:.3f}%)")
        print(f"    Spread: {mean_spread:.1f} ({mean_spread/mean_mid*100:.3f}% of mid)")
        print(f"    Ticks: {len(mids)}")
        print(f"    => Type: {ptype}")

        # Fit parameters
        if ptype == "pegged":
            cfg = fit_pegged(name, data)
        elif ptype == "ar_olivia":
            cfg = fit_ar_olivia(name, data)
        elif ptype == "wide_spread":
            cfg = fit_wide_spread(name, data)
        elif ptype == "olivia_follow":
            cfg = fit_olivia_follow(name, data)
        elif ptype == "basket_arb":
            cfg = fit_basket(name, data, all_data)
        elif ptype == "conversion_arb":
            cfg = fit_conversion(name, data)
        elif ptype == "options":
            data.update(extra)  # add underlying/strike from classification
            cfg = fit_options(name, data, all_data, all_data)
        else:
            cfg = {"type": "generic_mm", "position_limit": 50, "make_size": 15}

        configs[name] = cfg

    # Generate CONFIG
    config_str = generate_config(configs)
    print("\n" + "="*60)
    print("GENERATED CONFIG")
    print("="*60)
    print(config_str)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(config_str)
        print(f"\nCONFIG written to {args.output}")

    print("\n" + "="*60)
    print("GAMEDAY ACTIONS")
    print("="*60)
    print("1. Review the CONFIG above")
    print("2. Copy it into trader.py (replace existing CONFIG)")
    print("3. Run: py -3.12 backtest.py --round p4r1")
    print("4. Submit trader.py to IMC")
    print("5. Download server log -> py -3.12 tools/server_log_analyzer.py <log>")
    print("6. Iterate: adjust parameters, re-submit")


if __name__ == "__main__":
    main()
