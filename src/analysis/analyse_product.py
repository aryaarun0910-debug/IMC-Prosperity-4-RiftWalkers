"""
Product Analysis Script -- Run this on any IMC price CSV when a new round opens.

Implements the OBSERVE -> ANALYZE protocol from CLAUDE_CODE_PROMPT.md Step 2:
  - Wall Mid (best fair value estimator)
  - Return autocorrelation at lags 1, 2, 5, 10
  - Hurst exponent
  - Trade size distribution (look for Olivia's signature)
  - Daily extreme analysis (Olivia-type bot detection)
  - Pairwise correlations (for basket/ETF products)

Usage:
    python analysis/analyse_product.py --csv data/prices_round_1_day_0.csv
    python analysis/analyse_product.py --csv data/prices_round_1_day_0.csv --product KELP
    python analysis/analyse_product.py --csv data/prices_round_1_day_0.csv --all
    python analysis/analyse_product.py --csv data/prices_round_1_day_0.csv --trades data/trades_round_1_day_0.csv --all

Outputs:
  - Plots saved to analysis/plots/<product>_*.png
  - Config JSON block printed to stdout (ready to paste into config/roundN.json)
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import adfuller, coint, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels not installed. ADF test and ACF plots unavailable.")
    print("       pip install statsmodels")

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
#  Data Loading                                                        #
# ------------------------------------------------------------------ #

def load_trades(trades_csv_path: str, product: str) -> pd.DataFrame:
    """Load IMC trades CSV for a specific product."""
    df = pd.read_csv(trades_csv_path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    data = df[df["symbol"] == product].copy().reset_index(drop=True) if "symbol" in df.columns else pd.DataFrame()
    if data.empty and "product" in df.columns:
        data = df[df["product"] == product].copy().reset_index(drop=True)
    return data


def load_data(csv_path: str, product: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    data = df[df["product"] == product].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError(f"Product '{product}' not found in {csv_path}")
    if "bid_price_1" in data.columns and "ask_price_1" in data.columns:
        data["mid_price"] = (data["bid_price_1"] + data["ask_price_1"]) / 2
    if "spread" not in data.columns and "bid_price_1" in data.columns:
        data["spread"] = data["ask_price_1"] - data["bid_price_1"]
    return data


# ------------------------------------------------------------------ #
#  Statistical Tests                                                   #
# ------------------------------------------------------------------ #

def hurst_exponent(prices: np.ndarray) -> float:
    prices = np.array(prices)
    if len(prices) < 20:
        return 0.5
    log_prices = np.log(prices + 1e-9)
    returns    = np.diff(log_prices)
    n = len(returns)
    mu = np.mean(returns)
    devs   = returns - mu
    cumsum = np.cumsum(devs)
    R = cumsum.max() - cumsum.min()
    S = np.std(returns, ddof=1)
    if S == 0 or R == 0:
        return 0.5
    return float(np.log(R / S) / np.log(n))


def variance_ratio_test(prices: np.ndarray, q: int = 5) -> dict:
    """
    Lo-MacKinlay Variance Ratio test.

    VR(q) = Var(r_t(q)) / (q * Var(r_t))
      VR < 0.8 -> mean-reverting  (use MeanReversion / MarketMaking)
      VR > 1.2 -> trending / momentum (use ARPredictor / Momentum)
      VR ~= 1   -> random walk (use MarketMaking)

    More reliable than Hurst for short series; has known distribution under H0.
    """
    if len(prices) < q * 4:
        return {}
    r1 = np.diff(prices.astype(float))
    rq = prices[q:].astype(float) - prices[:-q].astype(float)

    var_1 = float(np.var(r1, ddof=1))
    var_q = float(np.var(rq, ddof=1))
    if var_1 < 1e-12:
        return {}

    vr = var_q / (q * var_1)

    if vr < 0.8:
        signal = "mean-reverting"
    elif vr > 1.2:
        signal = "trending"
    else:
        signal = "random walk"

    return {"vr": round(vr, 4), "q": q, "signal": signal}


def wall_mid_series(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Wall Mid for each timestamp.
    Wall Mid = average of (bid with highest qty) and (ask with lowest abs qty).
    From: domination plan Section 2.1 -- best fair value estimator.
    Falls back to simple mid if wall data unavailable.
    """
    wm = []
    for _, row in df.iterrows():
        # Find bid wall (level with highest bid volume)
        bid_wall_p = None
        bid_wall_v = -1
        for lvl in range(1, 4):
            p_col = f"bid_price_{lvl}"
            v_col = f"bid_volume_{lvl}"
            if p_col in row and pd.notna(row[p_col]) and pd.notna(row.get(v_col, np.nan)):
                if row[v_col] > bid_wall_v:
                    bid_wall_v = row[v_col]
                    bid_wall_p = row[p_col]

        # Find ask wall (level with lowest ask volume)
        ask_wall_p = None
        ask_wall_v = float("inf")
        for lvl in range(1, 4):
            p_col = f"ask_price_{lvl}"
            v_col = f"ask_volume_{lvl}"
            if p_col in row and pd.notna(row[p_col]) and pd.notna(row.get(v_col, np.nan)):
                if abs(row[v_col]) < ask_wall_v:
                    ask_wall_v = abs(row[v_col])
                    ask_wall_p = row[p_col]

        if bid_wall_p is not None and ask_wall_p is not None:
            wm.append((bid_wall_p + ask_wall_p) / 2.0)
        elif "mid_price" in row:
            wm.append(row["mid_price"])
        else:
            wm.append(np.nan)
    return np.array(wm)


def autocorrelation_at_lags(returns: np.ndarray, lags=(1, 2, 5, 10)) -> dict:
    """
    Compute return autocorrelation at multiple lags.
    From: domination plan Section 2.3.
    Negative = mean reverting. Positive = trending.
    """
    n = len(returns)
    mu = np.mean(returns)
    var = np.var(returns)
    result = {}
    for lag in lags:
        if n <= lag:
            result[lag] = 0.0
            continue
        cov = np.mean((returns[:n - lag] - mu) * (returns[lag:] - mu))
        result[lag] = float(cov / var) if var > 1e-12 else 0.0
    return result


def trade_size_distribution(trades_df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Compute trade size distribution -- look for Olivia's signature size.
    From: domination plan Section 2.3.

    In P3, Olivia always traded 15 lots. Look for a spike at a specific size.
    """
    if trades_df is None or trades_df.empty:
        return {}

    qty_col = None
    for c in ["quantity", "qty", "volume"]:
        if c in trades_df.columns:
            qty_col = c
            break
    if qty_col is None:
        return {}

    sizes = trades_df[qty_col].abs()
    counts = sizes.value_counts().head(top_n).to_dict()
    return {int(k): int(v) for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)}


def daily_extreme_detector(trades_df: pd.DataFrame, mid_prices: np.ndarray,
                            extreme_tol: float = 1.5) -> dict:
    """
    Detect Olivia-type bots that trade at daily price extremes.
    From: domination plan Section 2.3.

    Returns summary of suspected informed trades.
    """
    if trades_df is None or trades_df.empty:
        return {}

    qty_col   = next((c for c in ["quantity", "qty"] if c in trades_df.columns), None)
    price_col = next((c for c in ["price"] if c in trades_df.columns), None)

    if qty_col is None or price_col is None:
        return {}

    daily_min = float(np.nanmin(mid_prices))
    daily_max = float(np.nanmax(mid_prices))

    suspects_buy  = []
    suspects_sell = []

    for _, trade in trades_df.iterrows():
        price = trade[price_col]
        qty   = trade[qty_col]
        if pd.isna(price) or pd.isna(qty):
            continue
        if float(price) <= daily_min + extreme_tol and float(qty) > 0:
            suspects_buy.append({"price": float(price), "qty": float(qty)})
        elif float(price) >= daily_max - extreme_tol and float(qty) < 0:
            suspects_sell.append({"price": float(price), "qty": float(qty)})

    return {
        "daily_min": round(daily_min, 2),
        "daily_max": round(daily_max, 2),
        "daily_range": round(daily_max - daily_min, 2),
        "buys_at_low": len(suspects_buy),
        "sells_at_high": len(suspects_sell),
        "suspect_buy_sizes":  sorted(set(round(t["qty"]) for t in suspects_buy)),
        "suspect_sell_sizes": sorted(set(abs(round(t["qty"])) for t in suspects_sell)),
        "olivia_likely": len(suspects_buy) > 2 or len(suspects_sell) > 2,
    }


def roll_spread_estimate(prices: np.ndarray) -> dict:
    """
    Roll (1984) effective spread estimator -- requires only price data.

    If returns have negative serial covariance, a spread exists.
    S_effective = 2 * sqrt(-Cov(r_t, r_{t+1}))

    Negative serial covariance = spread effect (mean reversion around mid).
    High negative cov -> large effective spread -> more room for passive MM.
    """
    if len(prices) < 4:
        return {}
    r = np.diff(prices)
    cov = float(np.cov(r[:-1], r[1:])[0, 1])
    eff_half_spread = np.sqrt(max(0.0, -cov))
    return {
        "serial_cov":     round(cov, 6),
        "eff_half_spread": round(eff_half_spread, 4),
        "eff_spread":     round(2 * eff_half_spread, 4),
    }


def glosten_milgrom_decomp(prices_df: pd.DataFrame, trades_df: pd.DataFrame,
                            T_ahead: int = 5) -> dict:
    """
    Glosten-Milgrom adverse selection decomposition (Huang-Stoll 1997).

    For each trade at time t with price p and direction d (+1=buy, -1=sell):
        realized_half_spread  = |p - mid(t)|
        price_impact          = d * (mid(t+T) - mid(t))
        adverse_selection     = price_impact (portion that goes against MM)

    If adverse_selection / realized_half_spread > 0.5 -> >50% of spread is
    due to informed trading -> taking is expensive, MM must widen spreads.

    Returns:
        adverse_selection_frac : 0..1 (fraction of spread from informed traders)
        inventory_frac         : remainder (inventory / processing cost)
        interpretation         : human-readable string
    """
    if trades_df is None or trades_df.empty:
        return {}
    price_col = next((c for c in ["price"] if c in trades_df.columns), None)
    qty_col   = next((c for c in ["quantity", "qty"] if c in trades_df.columns), None)
    ts_col    = next((c for c in ["timestamp"] if c in trades_df.columns), None)
    if not all([price_col, qty_col, ts_col]):
        return {}

    # Build mid-price time-series indexed by timestamp
    mid_by_ts = {}
    for _, row in prices_df.iterrows():
        if pd.notna(row.get("bid_price_1")) and pd.notna(row.get("ask_price_1")):
            ts = int(row["timestamp"])
            mid_by_ts[ts] = (float(row["bid_price_1"]) + float(row["ask_price_1"])) / 2.0

    sorted_ts = sorted(mid_by_ts.keys())
    ts_index  = {ts: i for i, ts in enumerate(sorted_ts)}

    realized_spreads = []
    price_impacts    = []

    for _, trade in trades_df.iterrows():
        ts    = int(trade[ts_col])
        price = float(trade[price_col])
        qty   = float(trade[qty_col])
        if ts not in mid_by_ts:
            continue
        mid_t = mid_by_ts[ts]
        direction = 1.0 if qty > 0 else -1.0

        realized_half = abs(price - mid_t)
        if realized_half < 0.01:
            continue

        # Mid T_ahead ticks later
        idx_t = ts_index.get(ts, -1)
        idx_future = idx_t + T_ahead
        if idx_future < len(sorted_ts):
            ts_future = sorted_ts[idx_future]
            mid_future = mid_by_ts.get(ts_future)
            if mid_future is not None:
                impact = direction * (mid_future - mid_t)
                realized_spreads.append(realized_half)
                price_impacts.append(impact)

    if not realized_spreads:
        return {}

    avg_rs = np.mean(realized_spreads)
    avg_pi = np.mean(price_impacts)
    adv_frac = float(np.clip(avg_pi / avg_rs, 0.0, 1.0)) if avg_rs > 0 else 0.0
    inv_frac = 1.0 - adv_frac

    if adv_frac > 0.6:
        interp = "HIGH adverse selection -- informed bots present; widen spread or use BotExploiter"
    elif adv_frac > 0.3:
        interp = "MODERATE adverse selection -- some informed flow; use caution with tight quoting"
    else:
        interp = "LOW adverse selection -- mostly noise traders; passive MM viable"

    return {
        "n_trades":           len(realized_spreads),
        "avg_realized_half_spread": round(avg_rs, 4),
        "avg_price_impact":   round(avg_pi, 4),
        "adverse_selection_frac": round(adv_frac, 3),
        "inventory_frac":     round(inv_frac, 3),
        "interpretation":     interp,
    }


def print_deployment_decision(product: str, result: dict):
    """
    Print a clean 10-line deployment decision block.
    This is the output you read in the first 10 minutes of a new round.
    """
    regime     = result.get("regime", "unknown")
    H          = result.get("hurst", 0.5)
    adf_pval   = result.get("adf_pval")
    kpss_pval  = result.get("kpss_pval")
    vr_result  = result.get("vr_result", {})
    ou         = result.get("ou_params", {})
    extremes   = result.get("daily_extremes", {})
    gm         = result.get("gm_decomp", {})
    roll       = result.get("roll_spread", {})
    autocorrs  = result.get("autocorrs", {})
    config     = result.get("suggested_config", {})

    print(f"\n{'#'*60}")
    print(f"  DEPLOYMENT DECISION -- {product}")
    print(f"{'#'*60}")
    print(f"  Hurst:           {H:.4f}  ({'trending' if H>0.55 else 'mean-rev' if H<0.45 else 'rand walk'})")
    if adf_pval is not None:
        print(f"  ADF p-value:     {adf_pval:.4f}  ({'STATIONARY [OK]' if adf_pval<0.05 else 'non-stationary'})")
    if kpss_pval is not None:
        kpss_verdict = "STATIONARY [OK]" if kpss_pval > 0.05 else "non-stationary"
        print(f"  KPSS p-value:    {kpss_pval:.4f}  ({kpss_verdict})")
    if vr_result:
        print(f"  Variance Ratio:  VR({vr_result.get('q',5)})={vr_result.get('vr','n/a')}"
              f"  -> {vr_result.get('signal','')}")
    if autocorrs:
        ac1 = autocorrs.get(1, 0.0)
        ac5 = autocorrs.get(5, 0.0)
        print(f"  Autocorr lag-1:  {ac1:+.4f}  lag-5: {ac5:+.4f}")
    if roll:
        print(f"  Effective spread (Roll): {roll.get('eff_spread', 'n/a')}")
    if gm:
        print(f"  Adverse selection (GM):  {gm.get('adverse_selection_frac','n/a'):.0%}  -- {gm.get('interpretation','')}")
    if extremes:
        olivia = "OLIVIA-TYPE BOT DETECTED [W]" if extremes.get("olivia_likely") else "No strong bot signature"
        sizes  = extremes.get("suspect_buy_sizes") or extremes.get("suspect_sell_sizes")
        print(f"  Bot detection:   {olivia}")
        if sizes:
            print(f"  Signature sizes: {sizes}")
    if ou and not ou.get("non_reverting"):
        print(f"  OU half-life:    {ou.get('half_life','n/a')} timestamps  mu={ou.get('mu','n/a')}")
    print(f"  Detected regime: {regime.upper()}")
    print(f"  -> STRATEGY:      {config.get('strategy','?')}")
    key_params = {k: v for k, v in config.items()
                  if k not in ("strategy", "trade_size", "position_limit")}
    if key_params:
        print(f"  -> KEY PARAMS:    {key_params}")
    print(f"{'#'*60}\n")


def estimate_ou(prices: np.ndarray) -> dict:
    prices = np.array(prices)
    P_lag  = prices[:-1]
    P_now  = prices[1:]
    try:
        b, a = np.polyfit(P_lag, P_now, 1)
    except Exception:
        return {}
    theta = 1.0 - b
    if theta <= 0.005:
        return {"theta": theta, "non_reverting": True}
    mu       = a / theta
    resid    = P_now - (a + b * P_lag)
    sigma_ps = float(np.std(resid, ddof=1))
    sigma_st = sigma_ps / np.sqrt(2.0 * theta)
    hl       = np.log(2.0) / theta
    return {
        "mu":         round(float(mu), 4),
        "theta":      round(float(theta), 6),
        "sigma_per_step": round(float(sigma_ps), 4),
        "sigma_stat": round(float(sigma_st), 4),
        "half_life":  round(float(hl), 2),
        "non_reverting": False,
    }


# ------------------------------------------------------------------ #
#  Full Product Analysis                                               #
# ------------------------------------------------------------------ #

def analyse_product(csv_path: str, product: str, save_plots: bool = True,
                    trades_csv: str = None) -> dict:
    print(f"\n{'='*60}")
    print(f"  ANALYSING: {product}")
    print(f"{'='*60}")

    data   = load_data(csv_path, product)
    prices = data["mid_price"].dropna().values
    n      = len(prices)

    print(f"  Timestamps:    {n}")
    print(f"  Price range:   [{prices.min():.2f}, {prices.max():.2f}]")
    print(f"  Mean:          {prices.mean():.4f}")
    print(f"  Std:           {prices.std():.4f}")

    # -- Wall Mid ----------------------------------------------------------
    wm_series = wall_mid_series(data)
    wm_valid  = wm_series[~np.isnan(wm_series)]
    if len(wm_valid) > 0:
        wm_vs_mid = float(np.mean(np.abs(wm_valid - prices[:len(wm_valid)])))
        print(f"  Wall Mid:      mean={wm_valid.mean():.4f}  "
              f"avg_deviation_from_raw_mid={wm_vs_mid:.4f}")

    returns = np.diff(prices)

    # -- Multi-lag autocorrelation -----------------------------------------
    if len(returns) > 10:
        autocorrs = autocorrelation_at_lags(returns)
        print(f"\n  Return autocorrelation:")
        for lag, ac in autocorrs.items():
            direction = "v mean-reverting" if ac < -0.05 else ("^ trending" if ac > 0.05 else "~= random walk")
            print(f"    lag-{lag:2d}:  {ac:+.4f}  {direction}")

    # -- Trade-level analysis (Olivia detection) ---------------------------
    trades_df = None
    if trades_csv and os.path.exists(trades_csv):
        trades_df = load_trades(trades_csv, product)

    if trades_df is not None and not trades_df.empty:
        print(f"\n  Trade analysis ({len(trades_df)} trades):")
        size_dist = trade_size_distribution(trades_df)
        if size_dist:
            top5 = list(size_dist.items())[:5]
            print(f"    Size distribution (top 5): {top5}")
            top_size, top_count = list(size_dist.items())[0]
            if top_count > len(trades_df) * 0.1:
                print(f"    [W] Size {top_size} appears {top_count}x ({100*top_count/len(trades_df):.0f}%)"
                      f" -- possible signature bot")

        extreme_info = daily_extreme_detector(trades_df, prices)
        if extreme_info:
            olivia_flag = "[W] OLIVIA-TYPE BOT LIKELY" if extreme_info["olivia_likely"] else "no strong signal"
            print(f"    Daily range: [{extreme_info['daily_min']:.2f}, {extreme_info['daily_max']:.2f}]"
                  f"  range={extreme_info['daily_range']:.2f}")
            print(f"    Buys at low: {extreme_info['buys_at_low']}  "
                  f"Sells at high: {extreme_info['sells_at_high']}  -> {olivia_flag}")
            if extreme_info["suspect_buy_sizes"]:
                print(f"    Suspect buy sizes at low:  {extreme_info['suspect_buy_sizes']}")
            if extreme_info["suspect_sell_sizes"]:
                print(f"    Suspect sell sizes at high: {extreme_info['suspect_sell_sizes']}")
    else:
        if trades_csv:
            print(f"  Trade CSV not found or no trades for {product}")
        else:
            print(f"  No trades CSV provided -- run with --trades to detect Olivia bots")

    # -- Roll Spread Estimate ----------------------------------------------
    roll_spread = roll_spread_estimate(prices)
    if roll_spread:
        print(f"\n  Roll effective spread:  {roll_spread['eff_spread']:.4f}  "
              f"(half: {roll_spread['eff_half_spread']:.4f}  "
              f"serial_cov: {roll_spread['serial_cov']:.6f})")

    # -- Glosten-Milgrom Decomposition ------------------------------------
    gm_decomp = {}
    if trades_df is not None and not trades_df.empty:
        gm_decomp = glosten_milgrom_decomp(data, trades_df)
        if gm_decomp:
            print(f"\n  Glosten-Milgrom spread decomposition ({gm_decomp['n_trades']} trades):")
            print(f"    Avg realized half-spread: {gm_decomp['avg_realized_half_spread']:.4f}")
            print(f"    Avg price impact (T+5):   {gm_decomp['avg_price_impact']:.4f}")
            print(f"    Adverse selection:        {gm_decomp['adverse_selection_frac']:.1%}")
            print(f"    Inventory/processing:     {gm_decomp['inventory_frac']:.1%}")
            print(f"    -> {gm_decomp['interpretation']}")

    # ---- Regime Detection ----
    adf_pval  = None
    kpss_pval = None
    if HAS_STATSMODELS and n > 20:
        try:
            adf_result = adfuller(prices, maxlag=min(10, n // 10))
            adf_pval   = float(adf_result[1])
        except Exception:
            adf_pval = None
        try:
            # KPSS null hypothesis = stationary; p < 0.05 -> REJECT stationarity
            _, kpss_pval, _, _ = kpss(prices, regression="c", nlags="auto")
            kpss_pval = float(kpss_pval)
        except Exception:
            kpss_pval = None

    # Variance Ratio test (q=5)
    vr_result = variance_ratio_test(prices, q=5)

    H = hurst_exponent(prices)

    # Dual-confirmation stationarity:
    #   ADF rejects unit root (p<0.05) AND KPSS fails to reject stationarity (p>0.05)
    adf_stat  = adf_pval is not None and adf_pval < 0.05
    kpss_stat = kpss_pval is not None and kpss_pval > 0.05  # KPSS null = stationary
    stationary = adf_stat and kpss_stat

    # Both formal tests available AND agree on non-stationarity -- hard veto on mean_reversion.
    # TOMATOES case: ADF p=0.26 (non-stat) + KPSS p=0.01 (non-stat) -> never mean_reversion.
    both_tests_available = adf_pval is not None and kpss_pval is not None
    both_non_stationary  = both_tests_available and (not adf_stat) and (not kpss_stat)

    # VR signal supplements Hurst for regime detection
    vr_stat = vr_result.get("vr", 1.0)

    regime_confidence = "HIGH"    # downgraded when formal tests disagree or unavailable

    if stationary:
        # ADF + KPSS dual-confirmed stationary -- highest confidence
        regime = "mean_reversion"
        regime_confidence = "HIGH"
    elif both_non_stationary:
        # Both tests agree: non-stationary. Hurst/VR distinguishes momentum vs random walk.
        if H > 0.55 and vr_stat > 1.1:
            regime = "momentum"
            regime_confidence = "MEDIUM"   # Hurst unreliable on n<500
        else:
            regime = "random_walk"
            regime_confidence = "HIGH"
    elif H < 0.45 and vr_stat < 0.9:
        # Formal tests inconclusive -- fallback to Hurst + VR
        regime = "mean_reversion"
        regime_confidence = "LOW"
    elif H > 0.55 and vr_stat > 1.1:
        regime = "momentum"
        regime_confidence = "LOW"
    else:
        regime = "random_walk"
        regime_confidence = "MEDIUM"

    print(f"\n  Hurst exponent:  {H:.4f}")
    print(f"  ADF p-value:     {adf_pval if adf_pval is not None else 'n/a'}"
          f"  ({'stationary [OK]' if adf_stat else 'non-stationary'})")
    if kpss_pval is not None:
        kpss_verdict = "stationary [OK]" if kpss_pval > 0.05 else "NON-STATIONARY [NO]"
        print(f"  KPSS p-value:    {kpss_pval:.4f}  ({kpss_verdict})")
        if adf_stat and kpss_stat:
            print(f"  Dual-confirm:    CONFIRMED STATIONARY (ADF + KPSS agree)")
        elif both_non_stationary:
            print(f"  Dual-confirm:    CONFIRMED NON-STATIONARY (ADF + KPSS agree) -- mean_reversion VETOED")
        elif adf_stat and not kpss_stat:
            print(f"  Dual-confirm:    CONFLICTED (ADF says stationary, KPSS disagrees)")
        else:
            print(f"  Dual-confirm:    INCONCLUSIVE -- using Hurst/VR fallback")
    if vr_result:
        print(f"  Variance Ratio:  VR({vr_result['q']})={vr_result['vr']:.4f}"
              f"  -> {vr_result['signal']}")
    conf_tag = {"HIGH": "[HIGH confidence]", "MEDIUM": "[MEDIUM confidence -- verify with more data]",
                "LOW":  "[W] LOW confidence -- fallback to Hurst/VR, treat with skepticism"}
    print(f"  Detected regime: {regime.upper()}  {conf_tag[regime_confidence]}")

    # ---- OU Params ----
    ou_params = {}
    if regime == "mean_reversion" and n > 30:
        ou_params = estimate_ou(prices)
        if ou_params and not ou_params.get("non_reverting"):
            print(f"\n  OU Parameters:")
            print(f"    mu (long-run mean):  {ou_params['mu']:.4f}")
            print(f"    theta (reversion speed): {ou_params['theta']:.6f}")
            print(f"    sigma_stat:             {ou_params['sigma_stat']:.4f}")
            print(f"    Half-life:          {ou_params['half_life']:.1f} timestamps")

    # ---- Spread Analysis ----
    spread_info = {}
    if "spread" in data.columns:
        spreads = data["spread"].dropna().values
        spread_info = {
            "mean":   round(float(spreads.mean()), 4),
            "median": round(float(np.median(spreads)), 4),
            "std":    round(float(spreads.std()), 4),
        }
        print(f"\n  Bid-Ask Spread:")
        print(f"    Mean:    {spread_info['mean']:.3f}")
        print(f"    Median:  {spread_info['median']:.3f}")

    # ---- Variance of returns (for Kalman tuning) ----
    ret_var = float(np.var(returns)) if len(returns) > 1 else 1.0

    # ---- AR Auto-Fitting (Dragon v5 discovery) ----
    # ALL profitable P1-P3 dynamic products turned out to be AR-predictable.
    # Auto-fit AR(1..8) and select best order by AIC.
    autocorrs = autocorrelation_at_lags(returns) if len(returns) > 10 else {}
    ar_info = {}
    if len(prices) > 60:
        ar_order, ar_coeffs, ar_r2, ar_aic = fit_ar_coefficients(
            prices.tolist() if hasattr(prices, 'tolist') else list(prices), max_order=8)
        if ar_order is not None:
            ar_info = {
                "order": ar_order,
                "coeffs": ar_coeffs,
                "r2": ar_r2,
                "aic": ar_aic,
            }
            print(f"\n  AR Auto-Fit (best by AIC):")
            print(f"    Best order:  AR({ar_order})")
            print(f"    R^2:         {ar_r2:.4f}")
            lag_coeffs = ar_coeffs[1:]  # skip intercept
            print(f"    Lag coeffs:  {[round(c, 4) for c in lag_coeffs]}")
            print(f"    Coeff sum:   {sum(lag_coeffs):.4f}  (near 1.0 = unit root)")
            if autocorrs.get(1, 0) < -0.1:
                print(f"    ACF(1):      {autocorrs[1]:.4f}  -> NEGATIVE (tick-level MR, AR-predictable)")
            print(f"    Verdict:     {'AR-PREDICTABLE -> use ARPredictor' if ar_r2 > 0.85 and autocorrs.get(1, 0) < -0.05 else 'Not strongly AR-predictable'}")

    # ---- Build Config ----
    config = _build_config(regime, ou_params, spread_info, ret_var, hurst=H,
                           prices=prices, returns=returns, autocorrs=autocorrs,
                           ar_info=ar_info)
    print(f"\n  Suggested config:")
    print(json.dumps({product: config}, indent=4))
    if "_rationale" in config:
        print(f"  Rationale: {config['_rationale']}")
    if "_template" in config:
        print(f"  Template:  {config['_template']}")

    # ---- Plots ----
    if save_plots:
        _plot_overview(product, prices, returns, PLOTS_DIR)
        if regime == "mean_reversion" and ou_params and not ou_params.get("non_reverting"):
            _plot_zscore(product, prices, ou_params, PLOTS_DIR)
        if HAS_STATSMODELS and len(returns) > 10:
            _plot_acf(product, returns, PLOTS_DIR)

    # Trade-level results for return dict
    trade_size_dist = trade_size_distribution(trades_df) if trades_df is not None else {}
    extreme_info    = daily_extreme_detector(trades_df, prices) if trades_df is not None else {}

    result = {
        "product":          product,
        "regime":           regime,
        "regime_confidence": regime_confidence,
        "n":                n,
        "hurst":            round(H, 4),
        "adf_pval":         adf_pval,
        "kpss_pval":        kpss_pval,
        "vr_result":        vr_result,
        "ou_params":        ou_params,
        "spread":           spread_info,
        "roll_spread":      roll_spread,
        "gm_decomp":        gm_decomp,
        "autocorrs":        autocorrelation_at_lags(returns) if len(returns) > 10 else {},
        "trade_size_dist":  trade_size_dist,
        "daily_extremes":   extreme_info,
        "suggested_config": config,
    }
    print_deployment_decision(product, result)
    return result


# ------------------------------------------------------------------ #
#  Config Generation                                                   #
# ------------------------------------------------------------------ #

def fit_ar_coefficients(prices, max_order=6):
    """
    Auto-fit AR(p) via OLS and select optimal order by BIC (penalizes complexity more).
    Returns (best_order, coefficients, r2, bic) or (None, None, None, None).

    Key insight from P3/P4 testing:
      KELP AR(4) = [0.31, 0.27, 0.20, 0.21]  (Starfruit clone)
      SQUID_INK AR(2) = [0.79, 0.21]          (near random walk but still tradeable)
    Products with sum(coeffs) close to 1.0 and negative ACF(1) are AR-predictable.

    IMPORTANT: We use BIC (not AIC) because higher-order AR models overfit in-sample
    but underperform out-of-sample for TRADING. BIC's log(n) penalty correctly
    favors parsimonious models. Verified: AR(2) beats AR(8) on SQUID_INK by +19K.
    """
    n = len(prices)
    if n < max_order + 20:
        return None, None, None, None

    best_bic = float("inf")
    best_order = None
    best_coeffs = None
    best_r2 = None
    all_fits = []

    for p in range(1, max_order + 1):
        rows = n - p
        if rows < p + 5:
            continue

        p1 = p + 1
        XtX = np.zeros((p1, p1))
        Xty = np.zeros(p1)

        for t in range(rows):
            row = np.array([1.0] + [prices[t + p - lag] for lag in range(1, p1)])
            y_t = prices[t + p]
            XtX += np.outer(row, row)
            Xty += row * y_t

        try:
            coeffs = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue

        ss_res = 0.0
        ss_tot = 0.0
        mean_y = np.mean(prices[p:])
        for t in range(rows):
            pred = coeffs[0] + sum(coeffs[lag] * prices[t + p - lag] for lag in range(1, p1))
            ss_res += (prices[t + p] - pred) ** 2
            ss_tot += (prices[t + p] - mean_y) ** 2

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # BIC = n * log(RSS/n) + k * log(n) — heavier complexity penalty than AIC
        bic = rows * np.log(ss_res / rows + 1e-20) + p1 * np.log(rows)
        all_fits.append({"order": p, "r2": r2, "bic": bic})

        if bic < best_bic:
            best_bic = bic
            best_order = p
            best_coeffs = coeffs.tolist()
            best_r2 = r2

    return best_order, best_coeffs, best_r2, best_bic


def _is_pegged_fv(prices, returns):
    """Detect pegged/stable fair value products (Pearls, Amethysts, Emeralds, Rainforest Resin)."""
    zero_frac = np.sum(np.abs(returns) < 0.01) / max(len(returns), 1)
    price_range = np.ptp(prices)
    mean_price = np.mean(prices)
    # Pegged if >70% zero returns OR range < 0.2% of mean
    return zero_frac > 0.70 or (price_range / max(mean_price, 1) < 0.002)


def _build_config(regime, ou_params, spread_info, ret_var, hurst=None,
                  prices=None, returns=None, autocorrs=None, ar_info=None):
    base = {"trade_size": 20, "position_limit": 80}

    # ---- 1. PEGGED FV DETECTION (Emerald strategy) ----
    # The guaranteed money maker: pegged products (Pearls, Amethysts, Resin)
    # These have near-zero returns and tight range around a fixed value.
    if prices is not None and returns is not None and _is_pegged_fv(prices, returns):
        fv = round(float(np.median(prices)))
        avg_spread = float(spread_info.get("mean", 6.0))
        # make_spread is half-spread (bid = fv - make_spread, ask = fv + make_spread)
        # Must be less than half the book spread to capture fills
        # Optimal: just under half spread. For wide books (>12), step back 1 tick.
        half = int(avg_spread / 2)
        ms = max(2, half - 1 if avg_spread > 12 else half)
        return {
            **base,
            "strategy":              "Emerald",
            "fair_value":            fv,
            "make_spread":           ms,
            "trade_size":            60,
            "inventory_threshold_1": 30,
            "inventory_threshold_2": 60,
            "_template":             "_PRODUCT_STABLE",
            "_rationale":            f"Pegged FV detected at {fv}. Spread={avg_spread:.1f}, make_spread={ms}.",
        }

    # ---- 1b. WIDE-SPREAD BOT PRODUCT (Tomatoes strategy) ----
    # Products with wide spreads (>8) and bot trades are best handled by
    # compression detection + passive quoting. The Tomatoes strategy
    # outperforms ARPredictor on these (+1688 vs ~+500 on tutorial).
    avg_spread = float(spread_info.get("mean", 0))
    if avg_spread > 8 and prices is not None:
        price_range = float(np.ptp(prices))
        mean_price = float(np.mean(prices))
        # Wide spread + non-pegged = Tomatoes-type product
        if price_range / max(mean_price, 1) > 0.005:
            return {
                **base,
                "strategy":              "Tomatoes",
                "trade_size":            80,
                "take_ask_threshold":    6,
                "bid_comp_threshold":    5,
                "inventory_threshold_1": 70,
                "inventory_threshold_2": 78,
                "ema_alpha":             0.005,
                "ema_bid_skip_thr":      5.0,
                "ema_ask_skip_thr":      2.5,
                "postbc_window":         0,
                "vel_window":            0,
                "vel_adjust":            0,
                "_template":             "_PRODUCT_PASSIVE_MM",
                "_rationale":            f"Wide spread ({avg_spread:.1f}) + non-pegged. "
                                         f"Compression-based passive MM.",
            }

    # ---- 2. AR-PREDICTABLE DETECTION (ARPredictor strategy) ----
    # Key discovery from Dragon v5: products with negative ACF(1) and decent AR fit
    # are HIGHLY profitable with ARPredictor. This catches:
    #   - KELP (P3): AR(4)=[0.31,0.27,0.20,0.21], +1K/day
    #   - SQUID_INK (P3): AR(2)=[0.79,0.21], +2K/day
    #   - STARFRUIT (P2): AR(4)=[0.19,0.21,0.26,0.34]
    # ALL dynamic products in P1-P3 R1 were AR-predictable.
    if ar_info and ar_info.get("order") is not None:
        ar_order = ar_info["order"]
        ar_r2 = ar_info.get("r2", 0)
        ar_coeffs = ar_info.get("coeffs", [])
        acf1 = autocorrs.get(1, 0) if autocorrs else 0

        # AR is profitable when: R^2 > 0.85 (good fit) and negative ACF(1) (mean-reversion at tick level)
        ar_usable = ar_r2 > 0.85 and acf1 < -0.05
        # Also usable if very high R^2 even without negative ACF
        ar_usable = ar_usable or ar_r2 > 0.95

        if ar_usable:
            # Determine spread from volatility: wider for more volatile products
            price_std = float(np.std(prices)) if prices is not None else 1.0
            mean_price = float(np.mean(prices)) if prices is not None else 1.0
            rel_vol = price_std / mean_price  # relative volatility

            if rel_vol < 0.005:
                # Low vol (like KELP): spread=2, larger size
                spread = 2
                size = 20
                gamma = 0.01
            elif rel_vol < 0.03:
                # Medium vol (like SQUID_INK): spread=2, moderate size
                # SQUID_INK (rel_vol ~0.022) is profitable with spread=2, loses with spread=3
                spread = 2
                size = 15
                gamma = 0.005
            else:
                # Very high vol: wider spread, smaller size
                spread = 3
                size = 10
                gamma = 0.003

            return {
                **base,
                "strategy":       "ARPredictor",
                "ar_order":       ar_order,
                "update_every":   25,
                "min_history":    50,
                "spread":         spread,
                "gamma":          gamma,
                "trade_size":     size,
                "ar_clip_ticks":  max(5, int(price_std * 0.3)),
                "take_threshold": 0,
                "_template":      "_PRODUCT_TRENDING",
                "_rationale":   f"AR({ar_order}) R2={ar_r2:.4f}, ACF(1)={acf1:.3f}. "
                                f"Coeffs={[round(c,3) for c in ar_coeffs[1:] if isinstance(ar_coeffs, list)]}",
            }

    # ---- 3. MEAN-REVERTING (OU-based) ----
    if regime == "mean_reversion" and ou_params and not ou_params.get("non_reverting"):
        entry_z = 2.0
        if spread_info.get("median") and ou_params.get("sigma_stat"):
            entry_z = max(2.0, spread_info["median"] / ou_params["sigma_stat"] + 1.5)
        return {
            **base,
            "strategy":    "MeanReversion",
            "entry_z":     round(entry_z, 2),
            "exit_z":      0.3,
            "min_history": max(30, int(ou_params.get("half_life", 10) * 3)),
            "update_every": 50,
            "kalman_Q":    round(ret_var, 4),
            "kalman_R":    round(4 * ret_var, 4),
            "use_kalman_fv": True,
            "_template":    "_PRODUCT_MEANREVERT",
            "_rationale":   f"OU mean-reverting. Half-life={ou_params.get('half_life', '?'):.1f}.",
        }

    # ---- 4. MOMENTUM / HIGH HURST ----
    if hurst is not None and hurst > 0.55 and regime != "mean_reversion":
        return {
            **base,
            "strategy":   "MarketMaking",
            "gamma":      0.002,
            "kappa":      3.0,
            "lambda_obi": 0.5,
            "kalman_Q":   round(ret_var, 4),
            "kalman_R":   round(4 * ret_var, 4),
            "min_delta":  round(float(spread_info.get("median", 1.0)) * 0.5, 1),
            "_template":  "_PRODUCT_MEANREVERT",
            "_rationale": f"Momentum/trending. Hurst={hurst:.3f}.",
        }

    # ---- 5. DEFAULT: Conservative MarketMaking ----
    # High gamma = less trading = less adverse selection loss.
    # This is the "do no harm" default for unknown products.
    return {
        **base,
        "strategy":   "MarketMaking",
        "gamma":      0.05,
        "kappa":      2.0,
        "lambda_obi": 0.3,
        "kalman_Q":   round(ret_var, 4),
        "kalman_R":   round(4 * ret_var, 4),
        "min_delta":  round(float(spread_info.get("median", 2.0)), 1),
        "max_spread": 15.0,
        "session_length": 999999,
        "_template":  "DEFAULT_CONSERVATIVE",
        "_rationale": f"Unknown regime. High gamma to minimize adverse selection.",
    }


# ------------------------------------------------------------------ #
#  Plots                                                               #
# ------------------------------------------------------------------ #

def _plot_overview(product, prices, returns, plots_dir):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(prices, color='steelblue', linewidth=0.8)
    ax1.axhline(prices.mean(), color='orange', linestyle='--', label=f'Mean={prices.mean():.1f}')
    ax1.set_title(f'{product} -- Price History')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(returns, color='green', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_title('Returns (DeltaP)')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Return')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(returns, bins=50, color='green', edgecolor='white', density=True, alpha=0.7)
    from scipy.stats import norm
    x = np.linspace(returns.min(), returns.max(), 200)
    ax3.plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r-', linewidth=2, label='Normal')
    ax3.set_title('Return Distribution')
    ax3.set_xlabel('Return')
    ax3.legend()

    plt.suptitle(f'{product} -- Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{product}_overview.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_zscore(product, prices, ou_params, plots_dir):
    mu        = ou_params["mu"]
    sigma_st  = ou_params["sigma_stat"]
    z         = (prices - mu) / sigma_st

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(prices, color='steelblue', linewidth=0.8)
    ax1.axhline(mu, color='orange', linestyle='--', label=f'mu={mu:.1f}')
    ax1.axhline(mu + 2*sigma_st, color='red',  linestyle=':', label='+2sigma_stat')
    ax1.axhline(mu - 2*sigma_st, color='blue', linestyle=':', label='-2sigma_stat')
    ax1.legend(fontsize=8)
    ax1.set_title(f'{product} -- Price with OU Bounds')
    ax1.grid(True, alpha=0.3)

    ax2.plot(z, color='purple', linewidth=0.8)
    ax2.axhline(+2, color='red',  linestyle='--', label='Sell (z>+2)')
    ax2.axhline(-2, color='blue', linestyle='--', label='Buy (z<-2)')
    ax2.axhline(0,  color='black', linewidth=0.5)
    ax2.fill_between(range(len(z)), np.where(z>2, z, 2), 2, alpha=0.2, color='red')
    ax2.fill_between(range(len(z)), np.where(z<-2, z, -2), -2, alpha=0.2, color='blue')
    ax2.legend(fontsize=8)
    ax2.set_title(f'Z-Score (sigma_stat={sigma_st:.3f}, half-life={ou_params["half_life"]:.1f} ts)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Timestamp')

    n_signals = int(np.sum(np.abs(z) > 2))
    plt.suptitle(f'{product} -- Z-Score Analysis  |  {n_signals} entry signals', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{product}_zscore.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_acf(product, returns, plots_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(returns, lags=min(30, len(returns)//3), ax=ax1, alpha=0.05)
    ax1.set_title(f'{product} -- ACF of Returns')
    plot_pacf(returns, lags=min(30, len(returns)//4), ax=ax2, alpha=0.05)
    ax2.set_title(f'{product} -- PACF of Returns')
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{product}_acf.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ------------------------------------------------------------------ #
#  Correlation Analysis (multi-product)                                #
# ------------------------------------------------------------------ #

def cointegration_analysis(x: np.ndarray, y: np.ndarray,
                           name_x: str, name_y: str) -> dict:
    """
    Full Engle-Granger cointegration analysis between two price series.

    Steps:
      1. OLS: Y = alpha + beta * X + e  (estimate hedge ratio)
      2. ADF on residuals e (test if spread is stationary)
      3. OU fit on spread: theta (reversion speed) -> half-life
      4. Compute z-score of current spread vs rolling mean/std

    Returns a dict with results or None if not cointegrated (p >= 0.05).
    Prints a formatted report.
    """
    if not HAS_STATSMODELS:
        return None
    if len(x) < 30 or len(y) < 30:
        return None
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # -- Step 1: OLS hedge ratio --------------------------------------
    try:
        beta, alpha = np.polyfit(x, y, 1)
    except Exception:
        return None

    spread = y - (alpha + beta * x)

    # -- Step 2: Engle-Granger test (ADF on spread) ------------------
    try:
        _, eg_pval, _ = coint(y, x)
    except Exception:
        return None

    # ADF directly on spread for confirmation
    try:
        adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(spread, maxlag=min(10, len(spread)//10))
    except Exception:
        adf_pval = None
        adf_stat = None
        adf_crit = {}

    cointegrated = eg_pval < 0.05

    # -- Step 3: OU fit on spread -------------------------------------
    ou = estimate_ou(spread + abs(spread.min()) + 1.0)  # shift > 0 for log safety

    # Better OU: direct AR(1) on the spread (no log transform)
    sp_lag = spread[:-1]
    sp_now = spread[1:]
    try:
        b_ar, a_ar = np.polyfit(sp_lag, sp_now, 1)
        theta_ar = 1.0 - b_ar
        mu_ar    = a_ar / theta_ar if theta_ar > 1e-9 else np.mean(spread)
        hl_ar    = np.log(2.0) / theta_ar if theta_ar > 1e-9 else float("inf")
        resid    = sp_now - (a_ar + b_ar * sp_lag)
        sigma_ar = float(np.std(resid, ddof=1))
    except Exception:
        theta_ar = None
        hl_ar    = None
        mu_ar    = float(np.mean(spread))
        sigma_ar = float(np.std(spread, ddof=1))

    # -- Step 4: Current z-score --------------------------------------
    spread_mean = float(np.mean(spread))
    spread_std  = float(np.std(spread, ddof=1))
    current_z   = (spread[-1] - spread_mean) / spread_std if spread_std > 1e-9 else 0.0

    result = {
        "x":           name_x,
        "y":           name_y,
        "beta":        round(float(beta), 6),
        "alpha":       round(float(alpha), 4),
        "eg_pval":     round(eg_pval, 4),
        "adf_pval":    round(adf_pval, 4) if adf_pval is not None else None,
        "cointegrated": cointegrated,
        "spread_mean": round(spread_mean, 4),
        "spread_std":  round(spread_std, 4),
        "current_z":   round(current_z, 3),
        "theta":       round(theta_ar, 6) if theta_ar else None,
        "half_life":   round(hl_ar, 1)   if hl_ar and hl_ar < 1e6 else None,
        "sigma_step":  round(sigma_ar, 4),
        "spread":      spread,
    }

    # -- Print report -------------------------------------------------
    marker = "COINTEGRATED" if cointegrated else "not cointegrated"
    print(f"\n  {name_y} = {beta:+.4f} * {name_x} + {alpha:+.2f} + spread")
    adf_pval_str = f"{adf_pval:.4f}" if adf_pval is not None else "n/a"
    print(f"  Engle-Granger p = {eg_pval:.4f}  ADF(spread) p = "
          f"{adf_pval_str}  --> {marker}")

    if cointegrated:
        print(f"  Spread:  mean={spread_mean:.4f}  std={spread_std:.4f}  "
              f"current_z={current_z:+.2f}")
        if hl_ar and hl_ar < 1e4:
            print(f"  OU:      theta={theta_ar:.4f}  half-life={hl_ar:.1f} ts  "
                  f"sigma_step={sigma_ar:.4f}")
        else:
            print(f"  OU:      very slow reversion (half-life > 10,000 ts) -- "
                  f"spread may not be tradeable")

        # Trading recommendation
        if hl_ar and hl_ar < 500:
            entry_z = 2.0
            print(f"  RECOMMENDATION:")
            print(f"    Strategy: BasketArb or PairsTrader")
            print(f"    Hedge:    SELL 1 {name_y}, BUY {beta:.4f} {name_x}  (or round)")
            print(f"    Entry z:  +/-{entry_z:.1f}  Exit z: 0.3")
            print(f"    Entry at: spread > {spread_mean + entry_z*spread_std:.2f} "
                  f"(short) or < {spread_mean - entry_z*spread_std:.2f} (long)")
        elif hl_ar and hl_ar < 2000:
            print(f"  RECOMMENDATION: slow reversion -- monitor but use wide entry_z >= 2.5")
        else:
            print(f"  RECOMMENDATION: half-life too long for active trading")

        return result

    return None  # Not cointegrated -- exclude from coint_pairs


def _plot_cointegration(coint_pairs: list, plots_dir: str) -> None:
    """Plot spread time-series and z-score for each cointegrated pair."""
    for res in coint_pairs:
        spread     = res["spread"]
        name_x     = res["x"]
        name_y     = res["y"]
        spread_mean = res["spread_mean"]
        spread_std  = res["spread_std"]
        z          = (spread - spread_mean) / spread_std if spread_std > 0 else spread * 0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(spread, color='steelblue', linewidth=0.8, label='Spread')
        ax1.axhline(spread_mean, color='black', linestyle='--', linewidth=0.8, label=f'Mean={spread_mean:.2f}')
        ax1.axhline(spread_mean + 2*spread_std, color='red',  linestyle=':', label='+2sigma')
        ax1.axhline(spread_mean - 2*spread_std, color='blue', linestyle=':', label='-2sigma')
        ax1.set_title(f'Cointegration Spread: {name_y} - {res["beta"]:.4f}*{name_x}')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(z, color='purple', linewidth=0.8, label='Z-Score')
        ax2.axhline(+2,  color='red',   linestyle='--', label='Sell threshold')
        ax2.axhline(-2,  color='blue',  linestyle='--', label='Buy threshold')
        ax2.axhline(0.3, color='gray',  linestyle=':',  label='Exit')
        ax2.axhline(-0.3,color='gray',  linestyle=':')
        ax2.axhline(0,   color='black', linewidth=0.5)
        ax2.fill_between(range(len(z)), np.where(z > 2, z, 2),  2,  alpha=0.2, color='red')
        ax2.fill_between(range(len(z)), np.where(z < -2, z, -2), -2, alpha=0.2, color='blue')
        hl = res.get('half_life')
        ax2.set_title(f'Z-Score  |  EG p={res["eg_pval"]:.4f}  '
                      f'half-life={hl:.1f} ts' if hl else f'Z-Score  |  EG p={res["eg_pval"]:.4f}')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Timestamp')

        plt.suptitle(f'Cointegration: {name_y} vs {name_x}', fontweight='bold')
        plt.tight_layout()
        path = os.path.join(plots_dir, f"coint_{name_y}_vs_{name_x}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def analyse_correlations(csv_path: str, products: list, save_plots: bool = True):
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]

    price_series = {}
    for prod in products:
        rows = df[df["product"] == prod].copy()
        if rows.empty:
            continue
        rows["mid"] = (rows["bid_price_1"] + rows["ask_price_1"]) / 2
        price_series[prod] = rows["mid"].values

    if len(price_series) < 2:
        return

    # Align lengths
    min_len = min(len(v) for v in price_series.values())
    price_df = pd.DataFrame({k: v[:min_len] for k, v in price_series.items()})

    corr = price_df.corr()
    print("\n=== CORRELATION MATRIX ===")
    print(corr.round(3).to_string())

    # Cointegration analysis
    prods = list(price_df.columns)
    print("\n=== COINTEGRATION TESTS (Engle-Granger) ===")
    coint_pairs = []
    for i in range(len(prods)):
        for j in range(i+1, len(prods)):
            a, b = prods[i], prods[j]
            if abs(corr.loc[a, b]) > 0.3 and HAS_STATSMODELS:
                res = cointegration_analysis(
                    price_df[a].values, price_df[b].values, a, b
                )
                if res:
                    coint_pairs.append(res)

    if not coint_pairs:
        print("  No cointegrated pairs found (correlation threshold 0.3, p < 0.05)")

    if coint_pairs and save_plots:
        _plot_cointegration(coint_pairs, PLOTS_DIR)

    if save_plots:
        try:
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='RdBu', center=0, fmt='.2f',
                        vmin=-1, vmax=1, square=True)
            plt.title('Product Correlation Matrix')
            plt.tight_layout()
            path = os.path.join(PLOTS_DIR, "correlation_matrix.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"\n  Saved: {path}")
        except ImportError:
            print("[WARN] seaborn not installed -- skipping correlation heatmap")


# ------------------------------------------------------------------ #
# IMP 5: CROSS-PRODUCT RELATIONSHIP DETECTION                         #
# Detect baskets (one product = weighted sum of others) and pairs     #
# ------------------------------------------------------------------ #

def detect_cross_product_relationships(csv_path: str, products: list) -> list:
    """Detect basket/ETF relationships: one product as linear combo of others.
    Returns list of {'basket': str, 'components': {prod: weight}, 'r2': float, 'premium_mean': float}."""
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]

    price_series = {}
    for prod in products:
        rows = df[df["product"] == prod].copy()
        if rows.empty:
            continue
        rows["mid"] = (rows["bid_price_1"] + rows["ask_price_1"]) / 2
        price_series[prod] = rows["mid"].values

    if len(price_series) < 3:
        return []

    min_len = min(len(v) for v in price_series.values())
    price_df = pd.DataFrame({k: v[:min_len] for k, v in price_series.items()})

    relationships = []
    prods = list(price_df.columns)

    # For each product, try to explain it as a linear combo of others
    for target in prods:
        others = [p for p in prods if p != target]
        if len(others) < 2:
            continue

        y = price_df[target].values
        X = price_df[others].values

        # Simple OLS: beta = (X'X)^-1 X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue

        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if r2 > 0.90:  # strong linear relationship
            # Check if weights look like integer multiples (basket)
            components = {}
            for prod_name, w in zip(others, beta):
                if abs(w) > 0.1:  # meaningful weight
                    components[prod_name] = round(float(w), 2)

            if len(components) >= 2:
                residual = y - y_hat
                relationships.append({
                    "basket": target,
                    "components": components,
                    "r2": round(float(r2), 4),
                    "premium_mean": round(float(np.mean(residual)), 2),
                    "premium_std": round(float(np.std(residual)), 2),
                })

    # Sort by R² descending
    relationships.sort(key=lambda x: x["r2"], reverse=True)

    if relationships:
        print("\n=== CROSS-PRODUCT RELATIONSHIPS (IMP 5) ===")
        for rel in relationships:
            comp_str = " + ".join(f"{w}*{p}" for p, w in rel["components"].items())
            print(f"  {rel['basket']} ≈ {comp_str}")
            print(f"    R² = {rel['r2']}, premium = {rel['premium_mean']} ± {rel['premium_std']}")
    else:
        print("\n=== CROSS-PRODUCT RELATIONSHIPS: None detected (R² < 0.90) ===")

    return relationships


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="IMC Product Analyser")
    parser.add_argument("--csv",       default=None, help="Path to single price CSV")
    parser.add_argument("--multi-csv", nargs="+",    help="Multiple price CSVs to concatenate (multi-day analysis)")
    parser.add_argument("--trades",    default=None, help="Path to trades CSV (for Olivia detection)")
    parser.add_argument("--product",   default=None, help="Single product to analyse")
    parser.add_argument("--all",       action="store_true", help="Analyse all products in CSV")
    parser.add_argument("--no-plots",  action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    if args.multi_csv:
        # Concatenate multiple daily CSVs for multi-day analysis
        import tempfile
        frames = []
        for path in args.multi_csv:
            if not os.path.exists(path):
                print(f"[WARN] File not found: {path} -- skipping")
                continue
            f = pd.read_csv(path, sep=";")
            f.columns = [c.strip().lower() for c in f.columns]
            frames.append(f)
        if not frames:
            print("ERROR: no valid CSVs found in --multi-csv list")
            sys.exit(1)
        combined = pd.concat(frames, ignore_index=True)
        print(f"Multi-CSV: merged {len(frames)} files, {len(combined)} rows total")
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
        combined.to_csv(tmp, sep=";", index=False)
        tmp.close()
        args.csv = tmp.name
    elif not args.csv:
        print("ERROR: provide --csv or --multi-csv")
        sys.exit(1)

    df = pd.read_csv(args.csv, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    available = df["product"].unique().tolist()

    if args.all:
        products = available
    elif args.product:
        products = [args.product]
    else:
        print(f"Available products: {available}")
        prod = input("Enter product name to analyse: ").strip()
        products = [prod]

    all_configs = {}
    all_results = []

    for prod in products:
        if prod not in available:
            print(f"[WARN] '{prod}' not found in CSV. Skipping.")
            continue
        result = analyse_product(args.csv, prod, save_plots=not args.no_plots,
                                 trades_csv=args.trades)
        all_results.append(result)
        all_configs[prod] = result["suggested_config"]

    # Correlation analysis (if multiple products)
    if len(products) > 1:
        analyse_correlations(args.csv, products, save_plots=not args.no_plots)
        # IMP 5: Cross-product basket/pair detection
        rels = detect_cross_product_relationships(args.csv, products)
        if rels:
            for rel in rels:
                basket = rel["basket"]
                if basket in all_configs:
                    all_configs[basket]["type"] = "basket_arb"
                    all_configs[basket]["components"] = rel["components"]
                    all_configs[basket]["premium_mean"] = rel["premium_mean"]
                    all_configs[basket]["premium_std"] = rel["premium_std"]
                    print(f"  -> Updated {basket} config to basket_arb")

    # Print final combined config
    print(f"\n{'='*60}")
    print("FINAL SUGGESTED CONFIG (copy into config/roundN.json):")
    print('='*60)
    print(json.dumps(all_configs, indent=2))

    # Save config to file
    out_path = os.path.join(os.path.dirname(args.csv), "suggested_config.json")
    with open(out_path, "w") as f:
        json.dump(all_configs, f, indent=2)
    print(f"\nConfig saved to: {out_path}")


if __name__ == "__main__":
    main()
