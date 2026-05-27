"""
train_macaron_classifier.py — G1: offline LR training for MACARONS direction gate.

OFFLINE ONLY. This script imports numpy/pandas/sklearn. Never ship it; ship the
distilled constants (MACARON_LR_COEFS, MACARON_LR_INTERCEPT) into trader.py.

Pipeline:
  1. Load P3 Round 4 observation CSVs (sunlight, humidity) + MACARON price CSVs
  2. Compute features: [sunlight, sunlight_diff_50, sunlight_diff_500, humidity,
                        hour_of_day, sunlight*humidity]
  3. Label: sign(next-100-tick return) at each timestamp
  4. Train logistic regression with held-out day for validation
  5. Print distilled constants + holdout accuracy

Usage:
    py -3.12 analysis/train_macaron_classifier.py

Ship criteria: holdout accuracy > 55% AND gauntlet (with gate enabled on macarons)
shows net PnL improvement vs baseline.

Expected leverage: +20-40K on R4 if accuracy >= 55%.
"""
import json
import os
import sys

# Dependency check — this script uses numpy/pandas/sklearn; they MUST NOT be imported
# anywhere trader.py depends on. Keep this isolated to analysis/.
try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
except ImportError as e:
    print(f"ERROR: offline training deps missing: {e}")
    print("pip install numpy pandas scikit-learn")
    sys.exit(1)

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_ANALYSIS_DIR)
DATA_DIR = os.path.join(_ROOT_DIR, "reference", "p3_data", "Round 4 Data")
OUT_PATH = os.path.join(_ROOT_DIR, "analysis", "fingerprints", "macaron_lr.json")

FORECAST_TICKS = 100  # label: sign of return 100 ticks ahead


def load_r4_data():
    """Load all P3 R4 observation + price CSVs and join on timestamp (day from filename).

    Observation schema (comma-separated):
        timestamp, bidPrice, askPrice, transportFees, exportTariff, importTariff,
        sugarPrice, sunlightIndex
    Prices schema (semicolon-separated) filtered to MAGNIFICENT_MACARONS:
        day;timestamp;product;bid_price_1;...;mid_price;profit_and_loss
    """
    import re
    obs_files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("observations_round_4"))
    price_files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("prices_round_4"))
    if not obs_files or not price_files:
        raise RuntimeError(f"No P3 R4 CSVs in {DATA_DIR}")

    def day_of(fn):
        m = re.search(r"day_(\d+)", fn)
        return int(m.group(1)) if m else 0

    obs_frames = []
    for f in obs_files:
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        df["day"] = day_of(f)
        obs_frames.append(df)
    obs = pd.concat(obs_frames, ignore_index=True)

    price_frames = []
    for f in price_files:
        df = pd.read_csv(os.path.join(DATA_DIR, f), sep=";")
        price_frames.append(df[df["product"] == "MAGNIFICENT_MACARONS"].copy())
    prices = pd.concat(price_frames, ignore_index=True)
    prices["mid"] = prices["mid_price"]

    joined = prices[["day", "timestamp", "mid"]].merge(obs, on=["day", "timestamp"], how="inner")
    return joined.sort_values(["day", "timestamp"]).reset_index(drop=True)


def build_features(df):
    # P3 R4 schema uses `sunlightIndex` + `sugarPrice` (not humidity).
    df["sunlight"] = df["sunlightIndex"]
    df["sugar"] = df["sugarPrice"]
    df["sunlight_diff_50"] = df.groupby("day")["sunlight"].diff(50).fillna(0)
    df["sunlight_diff_500"] = df.groupby("day")["sunlight"].diff(500).fillna(0)
    df["hour_of_day"] = (df["timestamp"] // 100) % 24  # rough proxy
    df["sun_sugar"] = df["sunlight"] * df["sugar"]
    df["fwd_ret"] = df.groupby("day")["mid"].shift(-FORECAST_TICKS) - df["mid"]
    df["label"] = (df["fwd_ret"] > 0).astype(int)
    return df.dropna(subset=["label", "sunlight_diff_50", "sunlight_diff_500"])


def train(df):
    feature_cols = ["sunlight", "sunlight_diff_50", "sunlight_diff_500",
                    "sugar", "hour_of_day", "sun_sugar"]
    days = sorted(df["day"].unique())
    if len(days) < 2:
        raise RuntimeError("Need at least 2 days; train on [0..-1], holdout on last")
    train_df = df[df["day"].isin(days[:-1])]
    holdout_df = df[df["day"] == days[-1]]
    X_tr, y_tr = train_df[feature_cols].values, train_df["label"].values
    X_ho, y_ho = holdout_df[feature_cols].values, holdout_df["label"].values
    # Standardize (distilled into coeff scale)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    Z_tr = (X_tr - mu) / sd
    Z_ho = (X_ho - mu) / sd
    model = LogisticRegression(max_iter=1000, C=0.5)
    model.fit(Z_tr, y_tr)
    pred = model.predict(Z_ho)
    acc = accuracy_score(y_ho, pred)
    # Rebake coefficients back into original feature space: w' = w/sd, b' = b - sum(w*mu/sd)
    w = model.coef_[0] / sd
    b = float(model.intercept_[0] - float(np.sum(model.coef_[0] * mu / sd)))
    return {
        "features": feature_cols,
        "coefs": [float(c) for c in w],
        "intercept": b,
        "holdout_accuracy": float(acc),
        "training_days": [int(d) for d in days[:-1]],
        "holdout_day": int(days[-1]),
        "forecast_ticks": FORECAST_TICKS,
    }


def main():
    print(f"Loading P3 R4 data from {DATA_DIR}...")
    df = load_r4_data()
    print(f"  rows: {len(df):,d}, days: {sorted(df['day'].unique())}")
    df = build_features(df)
    print(f"  labeled rows: {len(df):,d}")
    result = train(df)
    print(f"\n=== DISTILLED LR (holdout accuracy {result['holdout_accuracy']:.3f}) ===")
    print(f"Features:  {result['features']}")
    print(f"Coefs:     {['%+.6f' % c for c in result['coefs']]}")
    print(f"Intercept: {result['intercept']:+.6f}")
    print(f"\nShip criteria: accuracy >= 0.55 -> paste into trader.py as MACARON_LR_COEFS")
    print(f"Current: {'SHIP' if result['holdout_accuracy'] >= 0.55 else 'DO NOT SHIP -- under 55%'}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
