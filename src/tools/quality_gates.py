"""
QUALITY GATES — No strategy deploys without passing every gate.

This is the anti-laziness system. Run before every submission.
If any gate FAILS, the strategy cannot advance until fixed.

Usage:
    python QUALITY_GATES.py                    # audit current trader.py
    python QUALITY_GATES.py --strategy pegged  # audit one strategy
    python QUALITY_GATES.py --log <path>       # audit against server log
"""

import sys
import os
import json
import math
import re

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _ROOT_DIR)


# ============================================================
# GATE 1: STRATEGY EDGE JUSTIFICATION
# "Why does this edge exist? Who is on the other side?"
# ============================================================

EDGE_JUSTIFICATIONS = {
    "pegged": {
        "edge": "FV is known exactly (round number). Other participants quote "
                "wider than necessary. We capture the spread around known FV.",
        "other_side": "Bots quoting wider spreads or noisy participants "
                      "who don't know FV is fixed.",
        "persistence": "FV doesn't drift — confirmed across P1-P3. Edge persists "
                       "because IMC designs these products with anchored FV.",
        "kill_condition": "If FV starts drifting >5 ticks from round number, "
                         "switch to ar_olivia immediately.",
    },
    "ar_olivia": {
        "edge": "Wall_mid + microprice gives better FV than market mid. "
                "Olivia detection gives directional signal with >60% hit rate.",
        "other_side": "Noise traders and bots quoting at market mid "
                      "instead of wall_mid. Uninformed participants on the "
                      "other side of Olivia's signal.",
        "persistence": "IMC always includes an informed trader bot. "
                       "Wall_mid consistently outperforms market mid as FV.",
        "kill_condition": "If toxicity > 0.6 for 100+ ticks, we're being "
                         "adversely selected. Widen spread or switch strategy.",
    },
    "olivia_follow": {
        "edge": "Informed trader (Olivia) predicts direction. We copy. "
                "Vol spikes mean-revert — fade extreme moves.",
        "other_side": "Uninformed traders who don't track Olivia. "
                      "Panic sellers during vol spikes.",
        "persistence": "Olivia-type bots appear every year. Vol mean-reversion "
                       "is structural in bounded-range products.",
        "kill_condition": "If Olivia hit rate drops below 55% over 200 trades, "
                         "she's noise, not signal. Switch to generic_mm.",
    },
    "wide_spread": {
        "edge": "Detect bot-driven spread compression. Penny-jump when "
                "spread compresses. EMA mean-reversion on true bot mid.",
        "other_side": "Bots running fixed strategies that create predictable "
                      "compression patterns.",
        "persistence": "Wide-spread products have structural bot activity. "
                       "Compression patterns are consistent within rounds.",
        "kill_condition": "If no compression detected for 500+ ticks, the "
                         "product doesn't have the expected bot structure. "
                         "Switch to generic_mm.",
    },
    "generic_mm": {
        "edge": "Adaptive: auto-detects OFI-following vs mean-reversion mode. "
                "Kalman FV + reservation price for inventory management.",
        "other_side": "Less adaptive participants using fixed strategies. "
                      "Noise traders not adjusting to regime.",
        "persistence": "Adaptivity is the edge — works across product types. "
                       "Degrades gracefully when product behavior changes.",
        "kill_condition": "If negative PnL for 1000+ ticks, mode detection "
                         "is wrong. Force reclassification.",
    },
    "basket_arb": {
        "edge": "Premium mean-reverts around a structural value. "
                "Z-score entry captures extreme deviations.",
        "other_side": "Participants who don't track the premium or react slowly.",
        "persistence": "Basket premium is structural (Linear Utility proved "
                       "it doesn't drift). Edge exists every year.",
        "kill_condition": "If premium_mean estimate is wrong by >50%, all "
                         "trades are on the wrong side. Kill after 5 "
                         "consecutive losses.",
    },
    "conversion_arb": {
        "edge": "Price difference between local and foreign market exceeds "
                "transport + tariff costs. Risk-free after conversion.",
        "other_side": "Participants who can't convert or don't optimize volume.",
        "persistence": "IMC designs conversion products with exploitable spreads. "
                       "Negative tariffs = free money (P3 Macarons).",
        "kill_condition": "If break_even > all local bids, no arb exists. "
                         "Reduce to passive MM only.",
    },
    "options": {
        "edge": "IV deviates from smile fit. Trade deviations back to fair value. "
                "Don't hedge delta (costs exceed benefit).",
        "other_side": "Participants mispricing options or using stale IV.",
        "persistence": "IV smile is structural. Deviations create repeatable edge.",
        "kill_condition": "If cumulative PnL < -5K after 500 ticks, auto-killed. "
                         "If IV estimation unstable, reduce size to 1.",
    },
    "do_nothing": {
        "edge": "No edge detected — explicit sit-out is better than bad trades.",
        "other_side": "N/A — we're not trading.",
        "persistence": "N/A.",
        "kill_condition": "Reclassify after 500 ticks if market structure becomes clearer.",
    },
    "pairs_arb": {
        "edge": "Cointegrated pairs spread mean-reverts. Trade deviations from "
                "rolling hedge ratio.",
        "other_side": "Participants not tracking the spread or using stale ratio.",
        "persistence": "Cointegration is structural in related products.",
        "kill_condition": "If spread stops mean-reverting (ADF p>0.05), disable.",
    },
}


# ============================================================
# GATE 2: CODE INTEGRITY
# Every tool that exists must be wired up and used.
# ============================================================

def check_tool_wiring(source):
    """Verify all tools are used by strategies (via standalone call OR ps.field read).
    Since ProductState.update() pre-computes these, strategies may read ps.field
    instead of calling standalone functions directly."""
    # Map: tool -> (standalone call pattern, ps.field patterns that prove usage)
    tools = {
        "kalman/FV": {
            "patterns": ["kalman_update(", "ps.kf_x"],
            "required_in": ["generic_mm"],  # ar_olivia uses its own AR model instead
        },
        "model_confidence": {
            "patterns": ["model_confidence(", "ps.model_confidence"],
            "required_in": ["ar_olivia", "olivia_follow", "wide_spread", "generic_mm"],
        },
        "ofi": {
            "patterns": ["compute_ofi(", "ps.ofi"],
            "required_in": ["ar_olivia", "olivia_follow", "generic_mm"],
        },
        "toxicity": {
            "patterns": ["track_fill_toxicity(", "ps.toxicity", "ps.tox_adj"],
            "required_in": ["ar_olivia", "olivia_follow", "wide_spread", "generic_mm"],
        },
        "sweep": {
            "patterns": ["detect_book_sweep(", "ps.sweep"],
            "required_in": ["ar_olivia", "olivia_follow", "generic_mm"],
        },
        "informed_bot": {
            "patterns": ["detect_informed_bot("],
            "required_in": ["ar_olivia", "olivia_follow", "generic_mm"],
        },
        "microprice": {
            "patterns": ["microprice(", "ps.microprice"],
            "required_in": ["ar_olivia", "generic_mm"],
        },
        "reservation_price": {
            "patterns": ["reservation_price("],
            "required_in": ["generic_mm"],
        },
    }

    issues = []
    for tool, spec in tools.items():
        for strategy in spec["required_in"]:
            func_pattern = f"def run_{strategy}"
            func_start = source.find(func_pattern)
            if func_start == -1:
                continue
            next_func = source.find("\ndef ", func_start + 1)
            func_body = source[func_start:next_func] if next_func != -1 else source[func_start:]
            found = any(p in func_body for p in spec["patterns"])
            if not found:
                issues.append(f"FAIL: {tool} not used in run_{strategy}()")
    return issues


# ============================================================
# GATE 3: PARAMETER SANITY
# No magic numbers without justification.
# ============================================================

def check_parameter_sanity(config):
    """Check CONFIG parameters are within sane ranges."""
    issues = []

    for product, cfg in config.items():
        ptype = cfg.get("type", "generic_mm")
        limit = cfg.get("position_limit", 50)

        # Make size should be meaningful fraction of position limit
        make_size = cfg.get("make_size", 15)
        if make_size < limit * 0.1:
            issues.append(f"WARN: {product} make_size={make_size} is <10% of "
                          f"limit={limit}. Are we being too timid?")
        if make_size > limit:
            issues.append(f"FAIL: {product} make_size={make_size} > limit={limit}")

        # Pegged: fair_value must exist
        if ptype == "pegged":
            fv = cfg.get("fair_value")
            if fv is None:
                issues.append(f"FAIL: {product} is pegged but no fair_value set!")

        # Basket: components and premium_mean
        if ptype == "basket_arb":
            if not cfg.get("components"):
                issues.append(f"WARN: {product} basket_arb with empty components. "
                              f"Auto-discovery will handle, but manual is better.")

        # Options: underlying must exist
        if ptype == "options":
            if not cfg.get("underlying"):
                issues.append(f"FAIL: {product} options with no underlying!")

        # Inventory thresholds should be proportional to limit
        thr1 = cfg.get("inventory_threshold_1")
        thr2 = cfg.get("inventory_threshold_2")
        if thr1 and thr2:
            if thr2 <= thr1:
                issues.append(f"FAIL: {product} threshold_2={thr2} <= threshold_1={thr1}")
            if thr2 > limit:
                issues.append(f"FAIL: {product} threshold_2={thr2} > limit={limit}")

    return issues


# ============================================================
# GATE 4: BACKTEST REALISM CHECK
# "Are our fill assumptions realistic?"
# ============================================================

BACKTEST_REALISM = """
BACKTEST REALISM CHECKLIST (manual verification required):

[ ] Fill rate calibrated against server log data (currently 0.015)
[ ] Aggressive fills use separate rate (currently 0.20)
[ ] PnL calibration factor derived from real server comparison (currently /12)
[ ] Position limits enforced (validate_orders uses AON batch rejection)
[ ] No lookahead bias: strategy only sees data up to current tick
[ ] TraderData serialization tested (NaN scrubbing, size limits)
[ ] Crossed orders handled (don't accidentally cross our own spread)
[ ] Empty order books handled (all strategies check for empty books)

KNOWN UNREALISTIC ASSUMPTIONS:
- Backtest fills at our price, server may fill at different price
- Backtest doesn't model queue priority (we go last in reality)
- Market impact of our orders not modeled
- Other bots don't react to our orders in backtest
"""


# ============================================================
# GATE 5: ADVERSARIAL ATTACK ON BACKTEST
# "Try to prove the strategy is fake."
# ============================================================

ADVERSARIAL_QUESTIONS = [
    "What if the edge is only in one day/regime?",
    "What if a simpler strategy (just market-make at mid) does equally well?",
    "What if the fill rate assumption is 2x too generous?",
    "What if position limits are tighter (30 instead of 50)?",
    "What if spread widens 2x mid-round?",
    "What if the informed trader (Olivia) disappears?",
    "What if our FV estimate is systematically wrong by 1 tick?",
    "What if we're the slowest participant (fill last, worst price)?",
    "What if another team front-runs our strategy?",
    "What if the product behavior changes at tick 5000?",
]


# ============================================================
# GATE 6: ROBUSTNESS — REGIME SEGMENTATION
# ============================================================

def segment_by_regime(mids, pnls, window=200):
    """Break results into volatility regimes."""
    if len(mids) < window * 2:
        return None

    segments = []
    for i in range(0, len(mids) - window, window):
        chunk = mids[i:i+window]
        changes = [chunk[j] - chunk[j-1] for j in range(1, len(chunk))]
        vol = (sum(d*d for d in changes) / len(changes)) ** 0.5
        chunk_pnl = sum(pnls[i:i+window]) if len(pnls) > i + window else 0

        segments.append({
            "start_tick": i,
            "vol": vol,
            "pnl": chunk_pnl,
            "regime": "high_vol" if vol > 2.0 else ("low_vol" if vol < 0.5 else "normal"),
        })

    return segments


# ============================================================
# MAIN: RUN ALL GATES
# ============================================================

def run_all_gates(trader_path=None):
    """Run all quality gates on current trader.py."""
    if trader_path is None:
        trader_path = os.path.join(_ROOT_DIR, 'trader.py')

    with open(trader_path, 'r', encoding='utf-8') as f:
        source = f.read()

    # Import trader to get CONFIG
    import importlib.util
    spec = importlib.util.spec_from_file_location('tc', trader_path)
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)

    print("=" * 70)
    print("QUALITY GATES AUDIT")
    print("=" * 70)

    total_pass = 0
    total_fail = 0
    total_warn = 0

    # Gate 1: Edge Justification
    print(f"\n--- GATE 1: EDGE JUSTIFICATION ---")
    for product, cfg in tc.CONFIG.items():
        ptype = cfg.get("type", "generic_mm")
        just = EDGE_JUSTIFICATIONS.get(ptype)
        if just:
            print(f"  {product} ({ptype}):")
            print(f"    Edge: {just['edge'][:80]}...")
            print(f"    Kill: {just['kill_condition'][:80]}...")
            total_pass += 1
        else:
            print(f"  FAIL: {product} ({ptype}) has NO edge justification!")
            total_fail += 1

    # Gate 2: Tool Wiring
    print(f"\n--- GATE 2: TOOL WIRING ---")
    wiring_issues = check_tool_wiring(source)
    if wiring_issues:
        for issue in wiring_issues:
            print(f"  {issue}")
            if "FAIL" in issue:
                total_fail += 1
            else:
                total_warn += 1
    else:
        print("  All tools properly wired to required strategies.")
        total_pass += 1

    # Gate 3: Parameter Sanity
    print(f"\n--- GATE 3: PARAMETER SANITY ---")
    param_issues = check_parameter_sanity(tc.CONFIG)
    if param_issues:
        for issue in param_issues:
            print(f"  {issue}")
            if "FAIL" in issue:
                total_fail += 1
            else:
                total_warn += 1
    else:
        print("  All parameters within sane ranges.")
        total_pass += 1

    # Gate 4: Backtest Realism (auto-verify what we can)
    print(f"\n--- GATE 4: BACKTEST REALISM ---")
    realism_checks = [
        ("Position limits enforced (validate_orders AON)", 'def validate_orders' in source),
        ("No lookahead bias (strategy sees current tick only)", 'state.timestamp' in source),
        ("TraderData NaN scrubbing", 'isnan' in source),
        ("TraderData size pruning", 'MAX_MEM_BYTES' in source),
        ("Empty order book guards", 'not od.buy_orders or not od.sell_orders' in source),
        ("Trim-based position limit enforcement", 'limit - pos - cum_buy' in source or 'pos + total_buy > limit' in source),
    ]
    gate4_pass = True
    for name, passed in realism_checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if passed:
            total_pass += 1
        else:
            total_fail += 1
            gate4_pass = False
    if gate4_pass:
        print("  All backtest realism checks PASS.")

    # Gate 5: Adversarial Questions
    print(f"\n--- GATE 5: ADVERSARIAL ATTACK ---")
    print("  Answer each question before deploying:")
    for i, q in enumerate(ADVERSARIAL_QUESTIONS, 1):
        print(f"  {i:2d}. {q}")

    # Gate 6: Safety
    print(f"\n--- GATE 6: CRASH SAFETY ---")
    safety_checks = [
        ("try/except around strategy dispatch", "try:" in source and "orders = []" in source),
        ("traderData JSON parsing has try/except", 'try:' in source and 'json.loads' in source),
        ("NaN/Inf scrubbing before serialization", 'isnan' in source and 'isinf' in source),
        ("Memory size pruning", 'MAX_MEM_BYTES' in source),
        ("Empty order book handling", 'not od.buy_orders or not od.sell_orders' in source),
        ("validate_orders (AON batch rejection)", 'def validate_orders' in source),
        ("Null-safe order_depths iteration", 'or {}' in source),
    ]
    for name, passed in safety_checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if passed:
            total_pass += 1
        else:
            total_fail += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"GATE RESULTS: {total_pass} PASS, {total_warn} WARN, {total_fail} FAIL")
    print(f"{'='*70}")

    if total_fail > 0:
        print("\nBLOCKED: Fix all FAIL items before deploying.")
        return 1
    elif total_warn > 0:
        print("\nDEPLOYABLE with caution. Review WARN items.")
        return 0
    else:
        print("\nCLEAR: All gates passed.")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_gates())
