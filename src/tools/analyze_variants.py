"""
analyze_variants.py — Rank R2 variant submissions by PnL.

Reads server logs in data/r2/variant_logs/ (named to include variant key)
and produces:
  - Ranked table: variant, total PnL, per-product PnL, delta vs baseline
  - Hypothesis groupings (IPR drift regime, ASH aggression, etc.)
  - WINNER recommendation with ship-command

Naming convention for variant logs (drop IMC downloads directly):
  Include variant key in filename, e.g.  r2_baseline_199999.log
  If key not detected, file is skipped (run with --show-skipped to list).

Usage:
    py -3.12 tools/analyze_variants.py
    py -3.12 tools/analyze_variants.py --dir <logdir>
    py -3.12 tools/analyze_variants.py --json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _TOOLS_DIR)

DEFAULT_DIR = os.path.join(_ROOT_DIR, "data", "r2", "variant_logs")
MIN_TICKS_FOR_CONFIDENCE = 10000  # A4: <10K ticks gets low-confidence penalty

# Order matters for detect_variant() — longest-first so r2_ipr_drift_extreme
# is matched before r2_ipr_drift, etc. sorted() at use-site handles that.
KNOWN_VARIANTS = {
    "r2_baseline":          "control — current trader.py, bid()=3000",
    "r2_ipr_drift_high":    "IPR fv_drift=0.13",
    "r2_ipr_drift_low":     "IPR fv_drift=0.09",
    "r2_ipr_drift_extreme": "IPR fv_drift=0.15",
    "r2_ipr_drift_zero":    "IPR fv_drift=0.0 (no drift prior)",
    "r2_ipr_dbo_tight":     "IPR drift_bid_offset=2",
    "r2_ipr_dbo_wide":      "IPR drift_bid_offset=8",
    "r2_ash_aggro":         "ASH take_width=2, inv_thr1=40",
    "r2_ash_pure_make":     "ASH take_width=0 (pure passive)",
    "r2_ash_tight_make":    "ASH make_offset=2",
    "r2_ash_wide_make":     "ASH make_offset=6",
    "r2_ash_take3":         "ASH take_width=3",
    "r2_ash_size_40":       "ASH make_size=40",
    "r2_combo_aggro":       "ASH tw=2 + IPR drift=0.13",
}


def load_log(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def parse_activities(csv_str):
    lines = csv_str.split('\n')
    if not lines:
        return {}
    header = [h.strip() for h in lines[0].split(';')]
    products = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(';')
        row = {}
        for i, h in enumerate(header):
            if i < len(parts):
                val = parts[i].strip()
                try:
                    row[h] = float(val) if '.' in val else int(val)
                except (ValueError, AttributeError):
                    row[h] = val
            else:
                row[h] = None
        prod = row.get('product')
        if not prod:
            continue
        products.setdefault(prod, []).append(row)
    return products


def detect_variant(filename):
    base = os.path.basename(filename).lower()
    for v in sorted(KNOWN_VARIANTS, key=len, reverse=True):
        if v in base:
            return v
    return None


def pnl_per_product(log):
    try:
        activities = parse_activities(log.get('activitiesLog', ''))
    except Exception as e:
        return None, None, f"parse failed: {e}"
    pnls = {}
    max_ts = 0
    for prod, rows in activities.items():
        series = [r['profit_and_loss'] for r in rows if r.get('profit_and_loss') is not None]
        pnls[prod] = series[-1] if series else 0.0
        for r in rows:
            ts = r.get('timestamp')
            if isinstance(ts, (int, float)) and ts > max_ts:
                max_ts = int(ts)
    # IMC emits rows every 100 ts units, so tick count = max_ts/100 + 1
    tick_count = (max_ts // 100) + 1 if max_ts > 0 else 0
    return pnls, tick_count, None


def fill_counts(log):
    trades = log.get('tradeHistory', [])
    out = defaultdict(int)
    for t in trades:
        if t.get('buyer') == 'SUBMISSION' or t.get('seller') == 'SUBMISSION':
            out[t['symbol']] += int(t.get('quantity', 0))
    return dict(out)


def find_logs(log_dir):
    """Return list of (path, basename_for_detection) — recurses one level for IMC folder packaging."""
    out = []
    if not os.path.isdir(log_dir):
        return out
    for entry in sorted(os.listdir(log_dir)):
        full = os.path.join(log_dir, entry)
        if os.path.isfile(full) and entry.endswith('.log'):
            out.append((full, entry))
        elif os.path.isdir(full):
            # IMC download folder: contains <id>.log — use folder name for variant detection
            for f in sorted(os.listdir(full)):
                if f.endswith('.log'):
                    out.append((os.path.join(full, f), entry))
                    break
    return out


def analyze_dir(log_dir, show_skipped=False):
    results, skipped = [], []
    logs = find_logs(log_dir)
    if not logs:
        print(f"ERROR: no logs found in {log_dir}")
        return results, skipped

    for path, detect_name in logs:
        fname = os.path.basename(path)
        variant = detect_variant(detect_name)
        if variant is None:
            skipped.append(fname)
            continue

        try:
            log = load_log(path)
        except Exception as e:
            results.append({"variant": variant, "file": fname, "error": f"load: {e}"})
            continue

        pnls, tick_count, err = pnl_per_product(log)
        if err:
            results.append({"variant": variant, "file": fname, "error": err})
            continue

        results.append({
            "variant": variant,
            "file": fname,
            "total": sum(pnls.values()),
            "pnl": pnls,
            "fills": fill_counts(log),
            "ticks": tick_count,
        })

    if show_skipped and skipped:
        print("\n--- SKIPPED (no variant key in filename) ---")
        for f in skipped:
            print(f"  {f}  (expected one of: {', '.join(sorted(KNOWN_VARIANTS))})")

    return results, skipped


def print_ranked(results):
    good = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]

    if bad:
        print("\n--- FAILED ---")
        for r in bad:
            print(f"  {r['variant']:<28} {r['file']}: {r['error']}")

    if not good:
        print("\nNo successful runs to rank.")
        return

    baseline = next((r for r in good if r["variant"] == "r2_baseline"), None)
    base_total = baseline["total"] if baseline else 0

    ranked = sorted(good, key=lambda r: -r["total"])
    print("\n" + "=" * 110)
    print(f"R2 VARIANT RANKING — {len(good)} variants"
          + (f"  (baseline total: {base_total:+,.0f})" if baseline else "  (no baseline log)"))
    print("=" * 110)

    all_products = set()
    for r in good:
        all_products.update(r["pnl"].keys())
    prods = sorted(all_products)

    header = f"{'Variant':<26} {'Total':>12} {'D base':>12} {'Ticks':>7}"
    for p in prods:
        header += f" {p[:14]:>15}"
    header += f" {'fills':>8}"
    print(header)
    print("-" * len(header))

    low_conf_variants = []
    for r in ranked:
        delta = r["total"] - base_total
        ticks = r.get("ticks", 0)
        low_conf = ticks < MIN_TICKS_FOR_CONFIDENCE
        if low_conf and r["variant"] != "r2_baseline":
            low_conf_variants.append(r["variant"])

        marker = ""
        if r["variant"] == "r2_baseline":
            marker = "  <<baseline>>"
        elif delta > 10000:
            marker = "  ** BIG WIN"
        elif delta > 2000:
            marker = "  +"
        elif delta < -10000:
            marker = "  ** BIG LOSS"
        elif delta < -2000:
            marker = "  -"
        if low_conf:
            marker += "  [LOW-CONF <10K ticks]"

        line = f"{r['variant']:<26} {r['total']:>12,.0f} {delta:>+12,.0f} {ticks:>7,d}"
        for p in prods:
            v = r["pnl"].get(p, 0)
            line += f" {v:>15,.0f}"
        line += f" {sum(r['fills'].values()):>8}"
        line += marker
        print(line)

    if low_conf_variants:
        print(f"\n  WARNING: {len(low_conf_variants)} variant(s) ran <{MIN_TICKS_FOR_CONFIDENCE} ticks — rankings unreliable.")
        print(f"  Low-conf: {', '.join(low_conf_variants)}")
        print(f"  Re-run these with full duration before trusting the ranking.")

    print("\n" + "=" * 110)
    print("HYPOTHESIS READOUT")
    print("=" * 110)

    def find(name):
        for r in good:
            if r["variant"] == name:
                return r["total"]
        return None

    groups = [
        ("IPR drift regime (which value of fv_drift wins?)",
         ["r2_ipr_drift_zero", "r2_ipr_drift_low", "r2_baseline",
          "r2_ipr_drift_high", "r2_ipr_drift_extreme"]),
        ("ASH take aggression (passive to aggressive)",
         ["r2_ash_pure_make", "r2_baseline", "r2_ash_aggro", "r2_ash_take3"]),
        ("ASH make_offset (spread width)",
         ["r2_ash_tight_make", "r2_baseline", "r2_ash_wide_make"]),
        ("IPR passive bid depth",
         ["r2_ipr_dbo_tight", "r2_baseline", "r2_ipr_dbo_wide"]),
        ("ASH make_size sensitivity",
         ["r2_ash_size_40", "r2_baseline"]),
    ]

    for title, variants in groups:
        vals = [(v, find(v)) for v in variants if find(v) is not None]
        if len(vals) < 2:
            continue
        print(f"\n  {title}:")
        lo = min(t for _, t in vals)
        for v, t in vals:
            bar_len = int(max(0, (t - lo) / 1000))
            print(f"    {v:<26} {t:>12,.0f}  {'#' * min(bar_len, 50)}")

    # Winner — require full tick count for trust
    trusted = [r for r in ranked if r.get("ticks", 0) >= MIN_TICKS_FOR_CONFIDENCE]
    best = trusted[0] if trusted else ranked[0]
    print("\n" + "=" * 110)
    if best["variant"] == "r2_baseline":
        print(f"WINNER: baseline (no variant beats it; ship current trader_submit.py)")
    else:
        delta = best["total"] - base_total
        conf = "HIGH" if best.get("ticks", 0) >= MIN_TICKS_FOR_CONFIDENCE else "LOW — DO NOT SHIP WITHOUT RE-RUN"
        print(f"WINNER ({conf}): {best['variant']} @ {best['total']:+,.0f}  "
              f"(delta {delta:+,.0f} vs baseline, {best.get('ticks',0):,d} ticks)")
        print(f"  {KNOWN_VARIANTS[best['variant']]}")
    print("=" * 110)

    print(f"\nTo ship winner:")
    print(f"  cp data/r2/variants/{best['variant']}.py trader_submit.py")
    print(f"  # or patch trader.py with the winning params and re-run tools/submit.py")
    print(f"\nRemember: verify bid() value in final trader_submit.py before upload.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"Log dir (default: {DEFAULT_DIR})")
    ap.add_argument("--json", action="store_true", help="Dump full results as JSON")
    ap.add_argument("--show-skipped", action="store_true", help="List files with no variant match")
    args = ap.parse_args()

    print(f"Scanning: {args.dir}")
    results, skipped = analyze_dir(args.dir, show_skipped=args.show_skipped)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    print_ranked(results)
    if skipped and not args.show_skipped:
        print(f"\n({len(skipped)} files skipped — run with --show-skipped to list)")


if __name__ == "__main__":
    main()
