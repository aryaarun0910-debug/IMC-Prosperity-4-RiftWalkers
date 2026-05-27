"""
Competition Trader v1 -- Lean, proven techniques from P1-P3 top finishers.
Target: Top 10 out of 20,000+ in IMC Prosperity 4.

Architecture:
  - Pegged FV products (RESIN, EMERALDS): Take-then-make, wall_mid FV
  - AR-predictable products (KELP): Wall_mid FV, Olivia tracking, passive MM
  - Random-walk products (SQUID_INK): Olivia following + vol mean-reversion
  - Wide-spread bot products (TOMATOES): Compression detection, penny-jump

Techniques from top teams:
  - Frankfurt Hedgehogs (2nd, 1.4M): wall_mid, Olivia tracking, competitive undercutting
  - chrispyroberts (7th, 107K R1): wall_mid, two-phase take-then-make
  - Alpha Animals (9th): MM-bot mid tracking

No external dependencies. Single file. Under 25KB.
"""

import json
import math
import random
import zlib
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# K4 trade-log emitter (backtest-only; no-op on IMC server where env is unset)
_TRADE_LOG_PATH = None
_TRADE_LOG_FH = None

def _emit_trade_log(ts, product, strategy, signal_reasons, orders):
    """Append per-tick orders + signals to a JSONL file when env var is set.
    Used by analysis/trade_attribution.py (K4) for per-trade PnL attribution.
    No-op on IMC server (env var unset)."""
    global _TRADE_LOG_FH
    if not _TRADE_LOG_PATH:
        return
    try:
        if _TRADE_LOG_FH is None:
            _TRADE_LOG_FH = open(_TRADE_LOG_PATH, "a")
        rec = {"k": "ord", "ts": ts, "p": product, "strat": strategy,
               "sig": signal_reasons,
               "ords": [(o.price, o.quantity) for o in orders]}
        _TRADE_LOG_FH.write(json.dumps(rec) + "\n")
        _TRADE_LOG_FH.flush()
    except Exception:
        pass


def _emit_fill_log(ts, product, strategy, fills, features):
    """Gap 5: append per-fill records with feature snapshot to the same JSONL.
    Used by analysis/trade_attribution.py for OLS fill-time attribution.
    No-op on IMC server (env var unset)."""
    global _TRADE_LOG_FH
    if not _TRADE_LOG_PATH or not fills:
        return
    try:
        if _TRADE_LOG_FH is None:
            _TRADE_LOG_FH = open(_TRADE_LOG_PATH, "a")
        for f in fills:
            rec = {"k": "fill", "ts": ts, "p": product, "strat": strategy,
                   "px": int(getattr(f, "price", 0)),
                   "qty": int(getattr(f, "quantity", 0)),
                   "buyer": getattr(f, "buyer", "") or "",
                   "seller": getattr(f, "seller", "") or "",
                   "feat": features}
            _TRADE_LOG_FH.write(json.dumps(rec) + "\n")
        _TRADE_LOG_FH.flush()
    except Exception:
        pass

# ============================================================
# DATAMODEL (mirrors IMC's datamodel.py)
# ============================================================

Symbol = str

@dataclass
class Order:
    symbol: Symbol
    price: int
    quantity: int
    def __repr__(self):
        return f"Order({'BUY' if self.quantity > 0 else 'SELL'} {abs(self.quantity)}x {self.symbol} @ {self.price})"

@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)

@dataclass
class Trade:
    symbol: Symbol
    price: int
    quantity: int
    buyer: str = ""
    seller: str = ""
    timestamp: int = 0

@dataclass
class ConversionObservation:
    bidPrice: float = 0.0
    askPrice: float = 0.0
    transportFees: float = 0.0
    exportTariff: float = 0.0
    importTariff: float = 0.0
    sunshineFraction: float = 0.0
    humidity: float = 0.0

@dataclass
class Observation:
    plainValueObservations: Dict[Symbol, float] = field(default_factory=dict)
    conversionObservations: Dict[Symbol, ConversionObservation] = field(default_factory=dict)

@dataclass
class TradingState:
    timestamp: int
    traderData: str
    listings: Dict[Symbol, Any]
    order_depths: Dict[Symbol, OrderDepth]
    own_trades: Dict[Symbol, List[Trade]]
    market_trades: Dict[Symbol, List[Trade]]
    position: Dict[Symbol, int]
    observations: Observation

# ============================================================
# Day boundary — centralized so R3+ can override if tick granularity changes.
# Phase A.A3: runtime-detect boundary via timestamp resets as fallback.
# ============================================================
DAY_PERIOD = 1000000  # default IMC convention: 10K ticks × 100 ts/tick

def tick_in_day(ts, mem=None):
    """Return tick index within current day.
    Primary: ts % DAY_PERIOD. Fallback: observed boundary in mem['_day_anchor']
    (written by main loop when ts jumps backwards or discontinuously)."""
    if mem is not None:
        anchor = mem.get("_day_anchor")
        if anchor is not None and ts >= anchor:
            tid = ts - anchor
            if tid < DAY_PERIOD * 2:  # sanity: anchor still valid
                return tid
    return ts % DAY_PERIOD

# ============================================================
# BOCPD -- Bayesian Online Changepoint Detection (lightweight)
# Detects regime changes in return series. When change_prob > 0.7,
# strategies should widen spreads, reduce sizing, reset AR coeffs.
# ~30 lines, stdlib only, O(1) amortized per tick (truncated).
# ============================================================

class BOCPD:
    """Lightweight Bayesian Online Changepoint Detection.
    Gaussian conjugate model with hazard rate pruning."""
    __slots__ = ('hazard', 'mu0', 'kappa0', 'alpha0', 'beta0',
                 'run_probs', 'muN', 'kappaN', 'alphaN', 'betaN', 'change_prob')

    def __init__(self, hazard=1/200):
        self.hazard = hazard
        self.mu0, self.kappa0 = 0.0, 1.0
        self.alpha0, self.beta0 = 1.0, 1.0
        self.run_probs = [1.0]
        self.muN = [0.0]; self.kappaN = [1.0]
        self.alphaN = [1.0]; self.betaN = [1.0]
        self.change_prob = 0.0

    def update(self, x):
        n = len(self.run_probs)
        # Student-t predictive probabilities for each run length
        pred = []
        for i in range(n):
            mu, kappa, alpha, beta = self.muN[i], self.kappaN[i], self.alphaN[i], self.betaN[i]
            scale = beta * (kappa + 1) / (alpha * kappa) if alpha > 0 and kappa > 0 else 1.0
            nu = 2 * alpha
            t_val = (x - mu) / max(scale ** 0.5, 1e-10)
            # Approximate Student-t pdf (good enough for detection)
            p = (1 + t_val**2 / max(nu, 1)) ** (-(nu + 1) / 2)
            pred.append(max(p, 1e-300))

        # Growth probabilities
        growth = [(1 - self.hazard) * self.run_probs[i] * pred[i] for i in range(n)]
        # Changepoint probability
        cp = self.hazard * sum(self.run_probs[i] * pred[i] for i in range(n))

        # Normalize
        total = cp + sum(growth)
        if total > 0:
            cp /= total
            growth = [g / total for g in growth]
        self.change_prob = cp

        # Update sufficient statistics
        new_mu = [self.mu0]; new_kappa = [self.kappa0]
        new_alpha = [self.alpha0]; new_beta = [self.beta0]
        for i in range(n):
            k = self.kappaN[i]
            new_mu.append((k * self.muN[i] + x) / (k + 1))
            new_kappa.append(k + 1)
            new_alpha.append(self.alphaN[i] + 0.5)
            new_beta.append(self.betaN[i] + k * (x - self.muN[i])**2 / (2 * (k + 1)))

        self.run_probs = [cp] + growth
        self.muN = new_mu; self.kappaN = new_kappa
        self.alphaN = new_alpha; self.betaN = new_beta

        # Truncate for memory (keep top 50 run lengths)
        if len(self.run_probs) > 50:
            self.run_probs = self.run_probs[:50]
            self.muN = self.muN[:50]; self.kappaN = self.kappaN[:50]
            self.alphaN = self.alphaN[:50]; self.betaN = self.betaN[:50]
            total = sum(self.run_probs)
            if total > 0:
                self.run_probs = [p / total for p in self.run_probs]

    def to_list(self):
        """Minimal serialization for traderData."""
        return [self.change_prob]  # only persist the signal, not internals

    @staticmethod
    def from_list(data):
        b = BOCPD()
        if data and len(data) >= 1:
            b.change_prob = data[0]
        return b

# ============================================================
# PRE-FIT CONSTANTS -- Distilled from P3 R3 (analysis/prefit_constants.py)
# Used as startup priors. Online updates via garch_update / SVI fit override.
# ============================================================

PREFIT_GARCH = {
    "VOLCANIC_ROCK": {"omega": 1.253e-09, "alpha": 0.05, "beta": 0.93, "vol_unc": 0.00025},
    "VOLCANIC_ROCK_VOUCHER_9500": {"omega": 2.14224e-07, "alpha": 0.05, "beta": 0.93, "vol_unc": 0.003273},
    "VOLCANIC_ROCK_VOUCHER_9750": {"omega": 4.78358e-07, "alpha": 0.08, "beta": 0.9, "vol_unc": 0.004891},
    "VOLCANIC_ROCK_VOUCHER_10000": {"omega": 2.730596e-06, "alpha": 0.08, "beta": 0.9, "vol_unc": 0.011685},
    "VOLCANIC_ROCK_VOUCHER_10250": {"omega": 4.030253e-06, "alpha": 0.08, "beta": 0.9, "vol_unc": 0.014196},
    "VOLCANIC_ROCK_VOUCHER_10500": {"omega": 3.197081e-06, "alpha": 0.08, "beta": 0.9, "vol_unc": 0.012643},
    "SQUID_INK": {"omega": 2.0281e-08, "alpha": 0.08, "beta": 0.9, "vol_unc": 0.001007},
    "KELP": {"omega": 2.1752e-08, "alpha": 0.15, "beta": 0.7, "vol_unc": 0.000381},
    "RAINFOREST_RESIN": {"omega": 1.429e-08, "alpha": 0.15, "beta": 0.7, "vol_unc": 0.000309},
    "PICNIC_BASKET1": {"omega": 1.293e-09, "alpha": 0.05, "beta": 0.7, "vol_unc": 7.2e-05},
    "PICNIC_BASKET2": {"omega": 1.784e-09, "alpha": 0.05, "beta": 0.7, "vol_unc": 8.4e-05},
    "CROISSANTS": {"omega": 8.47e-10, "alpha": 0.05, "beta": 0.85, "vol_unc": 9.2e-05},
    "JAMS": {"omega": 3.21e-10, "alpha": 0.05, "beta": 0.9, "vol_unc": 8e-05},
    "DJEMBES": {"omega": 4.26e-10, "alpha": 0.05, "beta": 0.85, "vol_unc": 6.5e-05},
}

PREFIT_BASKET_SPREAD = {
    "PICNIC_BASKET1": {"mean": 57.6438, "std": 88.3783, "nav": "6*CROISSANTS+3*JAMS+1*DJEMBES"},
    "PICNIC_BASKET2": {"mean": 22.5695, "std": 57.9171, "nav": "4*CROISSANTS+2*JAMS"},
}

# Polynomial smile fit on P3 R3: premium = c0 + c1*log(K/S) + c2*log(K/S)^2
PREFIT_VOUCHER_SMILE = {"c0": 95.875082, "c1": -6129.349356, "c2": 35597.608018}

# ============================================================
# PRODUCT STATE -- Structured per-product state object
# Replaces scattered mem[f"xxx_{product}"] with ps.field access.
# JSON-serializable, ~3-5KB per product. Computed once per tick.
# ============================================================

class ProductState:
    """Per-product state. JSON-serializable, ~3-5KB.
    Research-grounded: fields from Frankfurt (#2), Linear Utility (#2), jmerle (#25).
    Our unique edge: OFI (R²=0.65) -- no other top team uses it."""

    # Fields to persist across ticks (serialized to traderData)
    _PERSIST = {
        "tick_count", "mid_hist", "kf_x", "kf_P", "kf_innov_hist",
        "ofi_prev_book", "trade_flow_hist", "tox_hist",
        "ar_coeffs", "ar_fit_ts",
        "olivia_dir", "olivia_ts", "bot_det",
        "adapt_mode", "adapt_strength", "adapt_trade_signs",
        "adapt_mid_changes", "adapt_counter",
        "ema", "prem_hist", "prem_mean", "prem_n", "iv_hist",
        "spread_ema", "prev_sweep",
        "fills_attempted", "fills_received", "fill_rate_ema",
        "markout_pending", "markout_5", "markout_10", "markout_20", "markout_count",
        "gamma_mult", "gamma_inv_ema", "gamma_fill_ema",
        # Frontier engine (self-activating)
        "fr_sig_xtx", "fr_sig_xty", "fr_sig_n",
        "fr_trade_dirs", "fr_run_len", "fr_run_dir",
        "fr_kyle_lambda",
    }

    def __init__(self, product: str):
        self.product = product
        self.tick_count = 0

        # --- Raw Book (current tick, NOT persisted) ---
        self.bids = []
        self.asks = []

        # --- Price State ---
        self.mid = 0.0
        self.mid_hist = []      # last 80
        self.best_bid = 0
        self.best_ask = 0
        self.spread = 0
        self.microprice = 0.0
        self.wall_mid = 0.0
        self.pop_mid = 0.0
        self.mm_mid = 0.0       # Linear Utility: mid of vol>=15 levels

        # --- Kalman Filter ---
        self.kf_x = 0.0
        self.kf_P = 10.0
        self.kf_innov_hist = []  # last 30

        # --- Depth ---
        self.bid_depth = 0
        self.ask_depth = 0
        self.depth_imbalance = 0.0
        self.level_depths = []

        # --- Flow ---
        self.ofi = 0.0
        self.ofi_prev_book = {}
        self.trade_flow = 0.0
        self.trade_flow_hist = []  # last 30
        self.sweep = 0
        self.prev_sweep = {}

        # --- Deltas ---
        self.bid_delta = 0
        self.ask_delta = 0
        self.price_move = 0.0
        self.spread_change = 0

        # --- Toxicity ---
        self.toxicity = 0.5
        self.tox_hist = []  # last 30

        # --- AR ---
        self.ar_coeffs = None
        self.ar_fit_ts = -999
        self.ar_fv = 0.0

        # --- Bot Detection ---
        self.olivia_dir = None
        self.olivia_ts = 0
        self.bot_det = {}

        # --- Volatility ---
        self.realized_vol = 0.0
        self.spread_ema = 0.0

        # --- Adaptive Classification ---
        self.adapt_mode = None
        self.adapt_strength = 0.0
        self.adapt_trade_signs = []
        self.adapt_mid_changes = []
        self.adapt_counter = 0

        # --- Strategy-specific ---
        self.ema = 0.0
        self.prem_hist = []
        self.prem_mean = 0.0
        self.prem_n = 0
        self.iv_hist = []

        # --- Fill tracking ---
        self.fills_attempted = 0
        self.fills_received = 0
        self.fill_rate_ema = 0.1

        # --- Markout tracking (multi-horizon adverse selection) ---
        # Pending fills: [(tick, fill_price, side)] waiting for future mids
        self.markout_pending = []  # max 20 pending
        # EMA of markout at each horizon (in price units, negative = adverse)
        self.markout_5 = 0.0   # 5-tick markout
        self.markout_10 = 0.0  # 10-tick markout
        self.markout_20 = 0.0  # 20-tick markout
        self.markout_count = 0

        # --- Online gamma tuning ---
        self.gamma_mult = 1.0   # multiplier on base gamma (0.5 - 2.0)
        self.gamma_inv_ema = 0.0  # EMA of |position|/limit
        self.gamma_fill_ema = 0.1  # EMA of fill rate per tick

        # --- Regime detection (BOCPD) ---
        self.regime_change_prob = 0.0  # P(changepoint at current tick)
        self._bocpd = None  # BOCPD instance (not persisted, rebuilt)

        # --- Frontier engine (self-activating) ---
        # Online signal weight learning (EWLS): learns FV fusion weights from data
        self.fr_sig_xtx = None   # 3x3 running X'X matrix (persisted as flat list)
        self.fr_sig_xty = None   # 3x1 running X'y vector (persisted as list)
        self.fr_sig_n = 0        # sample count for warmup gating
        # Trade sequence momentum (Markov run detection)
        self.fr_trade_dirs = []  # last 20 trade directions (+1/-1)
        self.fr_run_len = 0      # current consecutive same-direction run
        self.fr_run_dir = 0      # direction of current run
        # Kyle's lambda (price impact per unit flow)
        self.fr_kyle_lambda = 0.0

        # --- Inventory (not persisted, set each tick) ---
        self.position = 0
        self.position_limit = 50
        self.inv_ratio = 0.0

        # --- Derived (not persisted, computed each tick) ---
        self.model_confidence = 1.0
        self.tox_adj = 1.0

    def update(self, od, market_trades, own_trades, ts, pos, limit):
        """Incremental state update. Call once per tick per product.
        Computes ALL derived fields. Strategies just read ps.field."""
        self.tick_count += 1
        self.position = pos
        self.position_limit = limit
        self.inv_ratio = abs(pos) / limit if limit > 0 else 0

        if not od.buy_orders or not od.sell_orders:
            return

        # 1. Raw book levels
        sorted_bids = sorted(od.buy_orders.items(), reverse=True)[:3]
        sorted_asks = sorted(od.sell_orders.items())[:3]
        self.bids = sorted_bids
        self.asks = sorted_asks
        prev_mid = self.mid
        prev_spread = self.spread
        self.best_bid = sorted_bids[0][0]
        self.best_ask = sorted_asks[0][0]
        self.spread = self.best_ask - self.best_bid
        self.mid = (self.best_bid + self.best_ask) / 2.0

        # 2. Price variants
        # Microprice (all top teams)
        v_bid = sorted_bids[0][1]
        v_ask = abs(sorted_asks[0][1])
        if v_bid + v_ask > 0:
            self.microprice = (self.best_ask * v_bid + self.best_bid * v_ask) / (v_bid + v_ask)
        else:
            self.microprice = self.mid

        # Wall mid (Frankfurt #2: worst_bid + worst_ask / 2)
        worst_bid = min(p for p, _ in sorted_bids)
        worst_ask = max(p for p, _ in sorted_asks)
        self.wall_mid = (worst_bid + worst_ask) / 2.0

        # Pop mid (jmerle #25: max-volume levels)
        pop_bid_p = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
        pop_ask_p = min(od.sell_orders.keys(), key=lambda p: od.sell_orders[p])
        self.pop_mid = (pop_bid_p + pop_ask_p) / 2.0

        # MM-filtered mid (Linear Utility #2: ignore orders < 15 lots)
        mm_bids = [p for p, v in od.buy_orders.items() if v >= 15]
        mm_asks = [p for p, v in od.sell_orders.items() if abs(v) >= 15]
        if mm_bids and mm_asks:
            self.mm_mid = (max(mm_bids) + min(mm_asks)) / 2.0
        else:
            self.mm_mid = self.mid

        # 3. Depth
        self.bid_depth = sum(od.buy_orders.values())
        self.ask_depth = sum(abs(v) for v in od.sell_orders.values())
        total = self.bid_depth + self.ask_depth
        self.depth_imbalance = (self.bid_depth - self.ask_depth) / total if total > 0 else 0.0
        self.level_depths = (
            [v for _, v in sorted_bids] +
            [0] * (3 - len(sorted_bids)) +
            [abs(v) for _, v in sorted_asks] +
            [0] * (3 - len(sorted_asks))
        )

        # 4. Deltas
        if prev_mid > 0:
            self.price_move = self.mid - prev_mid
            self.spread_change = self.spread - prev_spread

        # 5. Rolling history (max 80)
        self.mid_hist.append(self.mid)
        if len(self.mid_hist) > 80:
            self.mid_hist = self.mid_hist[-80:]

        # 6. Kalman filter (Q/R configurable per product type)
        Q = getattr(self, '_kalman_Q', 1.0)
        R = getattr(self, '_kalman_R', 4.0)
        P_pred = self.kf_P + Q
        K = P_pred / (P_pred + R)
        self.kf_x = self.kf_x + K * (self.mid - self.kf_x) if self.kf_x != 0 else self.mid
        self.kf_P = (1 - K) * P_pred

        # 7. Model confidence (Kalman innovation)
        innovation = abs(self.mid - self.kf_x)
        self.kf_innov_hist.append(innovation)
        if len(self.kf_innov_hist) > 30:
            self.kf_innov_hist = self.kf_innov_hist[-30:]
        if len(self.kf_innov_hist) >= 5:
            mean_innov = sum(self.kf_innov_hist) / len(self.kf_innov_hist)
            if mean_innov > 1e-9:
                ratio = innovation / mean_innov
                self.model_confidence = min(1.0, 1.0 / max(ratio, 1.0))
            else:
                self.model_confidence = 1.0
        else:
            self.model_confidence = 1.0

        # 8. OFI (inline computation)
        prev_bids = self.ofi_prev_book.get("bids", {})
        prev_asks = self.ofi_prev_book.get("asks", {})
        curr_bids = {str(p): v for p, v in sorted_bids}
        curr_asks = {str(p): abs(v) for p, v in sorted_asks}
        ofi_raw = 0.0
        for p in set(prev_bids) | set(curr_bids):
            ofi_raw += curr_bids.get(p, 0) - prev_bids.get(p, 0)
        for p in set(prev_asks) | set(curr_asks):
            ofi_raw -= curr_asks.get(p, 0) - prev_asks.get(p, 0)
        self.ofi_prev_book = {"bids": curr_bids, "asks": curr_asks}
        total_vol = sum(curr_bids.values()) + sum(curr_asks.values())
        self.ofi = ofi_raw / max(total_vol, 1.0)

        # 9. Book deltas (net volume changes)
        old_bid_vol = sum(int(v) for v in prev_bids.values())
        new_bid_vol = sum(curr_bids.values())
        old_ask_vol = sum(int(v) for v in prev_asks.values())
        new_ask_vol = sum(curr_asks.values())
        self.bid_delta = new_bid_vol - old_bid_vol
        self.ask_delta = new_ask_vol - old_ask_vol

        # 10. Sweep detection (inline)
        curr_sweep = {"bd": self.bid_depth, "ad": self.ask_depth,
                      "bb": self.best_bid, "ba": self.best_ask}
        prev_sw = self.prev_sweep
        self.sweep = 0
        if prev_sw:
            if prev_sw.get("ad", 0) > 0:
                consumed = 1.0 - (self.ask_depth / max(prev_sw["ad"], 1))
                if consumed > 0.4 and self.best_ask > prev_sw.get("ba", 0):
                    self.sweep = +1
            if prev_sw.get("bd", 0) > 0:
                consumed = 1.0 - (self.bid_depth / max(prev_sw["bd"], 1))
                if consumed > 0.4 and self.best_bid < prev_sw.get("bb", 999999):
                    self.sweep = -1
        self.prev_sweep = curr_sweep

        # 11. Trade flow
        flow = 0.0
        for t in market_trades:
            if t.price >= self.mid:
                flow += t.quantity
            else:
                flow -= t.quantity
        self.trade_flow = flow
        self.trade_flow_hist.append(flow)
        if len(self.trade_flow_hist) > 30:
            self.trade_flow_hist = self.trade_flow_hist[-30:]

        # 12. Toxicity (inline)
        for trade in own_trades:
            if trade.quantity > 0:
                self.tox_hist.append(1.0 if self.mid < trade.price else 0.0)
            else:
                self.tox_hist.append(1.0 if self.mid > trade.price else 0.0)
        if len(self.tox_hist) > 30:
            self.tox_hist = self.tox_hist[-30:]
        if len(self.tox_hist) >= 5:
            self.toxicity = sum(self.tox_hist) / len(self.tox_hist)
        else:
            self.toxicity = 0.5

        # 13. Tox adjustment (incorporates markout when available)
        base_tox = 0.7 if self.toxicity > 0.55 else (1.2 if self.toxicity < 0.40 else 1.0)
        if self.markout_count >= 10 and self.markout_10 < -1.0:
            # Markout is significantly adverse — reduce sizing further
            base_tox = min(base_tox, 0.5)
        self.tox_adj = base_tox

        # 14. Realized vol (MAD-based: robust to outliers like Olivia trades)
        # MAD = median(|x_i - median(x)|) * 1.4826 ≈ stddev for Gaussian
        # Falls back to stddev with <10 points (MAD needs decent sample)
        if len(self.mid_hist) >= 10:
            returns = [self.mid_hist[i] - self.mid_hist[i-1]
                       for i in range(max(1, len(self.mid_hist)-20), len(self.mid_hist))]
            if len(returns) >= 10:
                sorted_r = sorted(returns)
                n = len(sorted_r)
                median_r = sorted_r[n // 2] if n % 2 else (sorted_r[n//2 - 1] + sorted_r[n//2]) / 2
                abs_devs = sorted(abs(r - median_r) for r in returns)
                mad = abs_devs[len(abs_devs) // 2] if len(abs_devs) % 2 else \
                      (abs_devs[len(abs_devs)//2 - 1] + abs_devs[len(abs_devs)//2]) / 2
                self.realized_vol = max(0.1, mad * 1.4826)
            elif returns:
                mu = sum(returns) / len(returns)
                var = sum((r - mu)**2 for r in returns) / max(len(returns) - 1, 1)
                self.realized_vol = max(0.1, var ** 0.5)
        else:
            self.realized_vol = 1.0

        # 14b. BOCPD regime detection (every 5 ticks for performance)
        if self.tick_count % 5 == 0 and len(self.mid_hist) >= 2:
            ret = self.mid_hist[-1] - self.mid_hist[-2]
            if self._bocpd is None:
                self._bocpd = BOCPD()
            self._bocpd.update(ret)
            self.regime_change_prob = self._bocpd.change_prob

        # 15. Spread EMA (alpha configurable per product type)
        alpha = 0.08
        if self.spread_ema == 0:
            self.spread_ema = float(self.spread)
        else:
            self.spread_ema = alpha * self.spread + (1 - alpha) * self.spread_ema

        # 16. Fill tracking
        self.fills_received += len(own_trades)

        # 17. Markout tracking (multi-horizon adverse selection)
        # Add new fills as pending markout observations
        for trade in own_trades:
            side = 1 if trade.quantity > 0 else -1
            self.markout_pending.append((self.tick_count, trade.price, side))
        # Cap pending list (oldest first)
        if len(self.markout_pending) > 20:
            self.markout_pending = self.markout_pending[-20:]
        # Resolve pending markouts that have enough history
        if len(self.mid_hist) >= 2:
            still_pending = []
            for (fill_tick, fill_price, side) in self.markout_pending:
                ticks_elapsed = self.tick_count - fill_tick
                if ticks_elapsed >= 20:
                    # Resolve all horizons
                    alpha = 0.1
                    # markout = side * (future_mid - fill_price)
                    # Positive = profitable, negative = adverse
                    idx_base = len(self.mid_hist) - 1 - ticks_elapsed
                    for horizon, attr in [(5, "markout_5"), (10, "markout_10"), (20, "markout_20")]:
                        idx = idx_base + horizon
                        if 0 <= idx < len(self.mid_hist):
                            mo = side * (self.mid_hist[idx] - fill_price)
                            old = getattr(self, attr)
                            setattr(self, attr, alpha * mo + (1 - alpha) * old)
                    self.markout_count += 1
                else:
                    still_pending.append((fill_tick, fill_price, side))
            self.markout_pending = still_pending

        # 18. Online gamma tuning
        # Track inventory severity and fill rate, adjust gamma_mult every 200 ticks
        a = 0.02
        self.gamma_inv_ema = a * self.inv_ratio + (1 - a) * self.gamma_inv_ema
        got_fills = 1.0 if len(own_trades) > 0 else 0.0
        self.gamma_fill_ema = a * got_fills + (1 - a) * self.gamma_fill_ema
        if self.tick_count % 50 == 0 and self.tick_count >= 100:
            if self.gamma_inv_ema > 0.6:
                self.gamma_mult = min(2.0, self.gamma_mult * 1.1)
            elif self.gamma_fill_ema < 0.05 and self.gamma_mult > 0.6:
                self.gamma_mult = max(0.5, self.gamma_mult * 0.9)
            elif self.gamma_inv_ema < 0.3 and self.gamma_fill_ema > 0.15:
                self.gamma_mult = max(0.5, self.gamma_mult * 0.97)

        # 19. FRONTIER: Trade sequence momentum (Markov run detection)
        # Track direction of last 20 market trades. Consecutive same-direction
        # runs signal momentum beyond what OFI captures (HRT approach).
        for t in market_trades:
            d = 1 if t.price >= self.mid else -1
            self.fr_trade_dirs.append(d)
        if len(self.fr_trade_dirs) > 20:
            self.fr_trade_dirs = self.fr_trade_dirs[-20:]
        # Compute run length
        if self.fr_trade_dirs:
            self.fr_run_dir = self.fr_trade_dirs[-1]
            self.fr_run_len = 0
            for d in reversed(self.fr_trade_dirs):
                if d == self.fr_run_dir:
                    self.fr_run_len += 1
                else:
                    break

        # 20. FRONTIER: Kyle's lambda (price impact per unit flow)
        # lambda = Cov(dp, flow) / Var(flow). Updated every 100 ticks.
        # Tells us minimum edge needed to overcome adverse selection.
        if self.tick_count % 100 == 0 and self.tick_count >= 200 and len(self.mid_hist) >= 100:
            recent_mids = self.mid_hist[-100:]
            dp = [recent_mids[i] - recent_mids[i-1] for i in range(1, len(recent_mids))]
            # Use trade_flow_hist as signed flow proxy
            flow = self.trade_flow_hist[-99:] if len(self.trade_flow_hist) >= 99 else []
            n = min(len(dp), len(flow))
            if n >= 50:
                m_dp = sum(dp[-n:]) / n
                m_fl = sum(flow[-n:]) / n
                cov = sum((dp[-n+i] - m_dp) * (flow[-n+i] - m_fl) for i in range(n)) / n
                var_fl = sum((flow[-n+i] - m_fl)**2 for i in range(n)) / n
                if var_fl > 1e-6:
                    self.fr_kyle_lambda = max(0, cov / var_fl)

        # 21. FRONTIER: Online signal weight learning (EWLS)
        # After warmup, learns which FV signals predict next-tick mid best.
        # Signals: [microprice, wall_mid, ar_fv_proxy]. Target: next-tick mid.
        # Decay factor lambda=0.995 (half-life ~139 ticks).
        if self.tick_count >= 3 and len(self.mid_hist) >= 2:
            # We have last tick's signals -> this tick's mid is the target
            prev_mid = self.mid_hist[-2] if len(self.mid_hist) >= 2 else self.mid
            x = [self.microprice, self.wall_mid if self.wall_mid else self.mid, prev_mid]
            y = self.mid  # actual mid this tick
            decay = 0.995
            if self.fr_sig_xtx is None:
                self.fr_sig_xtx = [0.0] * 9  # 3x3 flat
                self.fr_sig_xty = [0.0] * 3
                self.fr_sig_n = 0
            # Update running statistics with exponential decay
            for i in range(3):
                self.fr_sig_xty[i] = decay * self.fr_sig_xty[i] + x[i] * y
                for j in range(3):
                    self.fr_sig_xtx[i * 3 + j] = decay * self.fr_sig_xtx[i * 3 + j] + x[i] * x[j]
            self.fr_sig_n += 1

    def is_drift_regime(self):
        """Runtime detector for drift products (monotonic mid movement).
        Used by EOD flatten to skip flattening on auto-classified drift products
        that never got a CONFIG fv_drift entry. See plan A2."""
        h = self.mid_hist
        if not h or len(h) < 100:
            return False
        w = h[-200:]
        net = w[-1] - w[0]
        total = sum(abs(w[i] - w[i-1]) for i in range(1, len(w)))
        if total < 1:
            return False
        return abs(net) / total > 0.6 and abs(net) > 5

    def to_dict(self):
        """Serialize to JSON-safe dict. Only non-default fields."""
        d = {}
        for key in self._PERSIST:
            val = getattr(self, key, None)
            if val is None:
                continue
            if isinstance(val, (list, dict)) and not val:
                continue
            if isinstance(val, float) and val == 0.0 and key not in ("kf_x", "kf_P", "toxicity"):
                continue
            if isinstance(val, int) and val == 0 and key != "tick_count":
                continue
            # Scrub NaN/Inf
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            if isinstance(val, list):
                val = [x if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else 0.0 for x in val]
            d[key] = val
        return d

    @classmethod
    def from_dict(cls, product, d):
        """Restore from JSON dict."""
        ps = cls(product)
        if not d:
            return ps
        for k, v in d.items():
            if hasattr(ps, k):
                setattr(ps, k, v)
        return ps

    def sync_to_mem(self, mem):
        """Write ProductState fields back to mem dict for backward compatibility.
        This allows strategies to still read from mem during migration."""
        p = self.product
        mem[f"kf_x_{p}"] = self.kf_x
        mem[f"kf_P_{p}"] = self.kf_P
        mem[f"kf_innov_hist_{p}"] = self.kf_innov_hist
        mem[f"ofi_prev_{p}"] = self.ofi_prev_book
        mem[f"tox_{p}"] = self.tox_hist
        mem[f"sweep_{p}"] = self.prev_sweep
        mem[f"hist_{p}"] = self.mid_hist
        if self.olivia_dir is not None:
            mem[f"olv_{p}"] = self.olivia_dir
            mem[f"olv_ts_{p}"] = self.olivia_ts
        if self.bot_det:
            mem[f"_bot_det_{p}"] = self.bot_det
        if self.ar_coeffs is not None:
            mem[f"ar_c_{p}"] = self.ar_coeffs
            mem[f"ar_ft_{p}"] = self.ar_fit_ts
        if self.ema != 0:
            mem[f"ema_{p}"] = self.ema

    def sync_from_mem(self, mem):
        """Read mem dict values into ProductState for backward compatibility.
        Used during migration: strategies write to mem, we read back."""
        p = self.product
        if f"kf_x_{p}" in mem: self.kf_x = mem[f"kf_x_{p}"]
        if f"kf_P_{p}" in mem: self.kf_P = mem[f"kf_P_{p}"]
        if f"kf_innov_hist_{p}" in mem: self.kf_innov_hist = mem[f"kf_innov_hist_{p}"]
        if f"ofi_prev_{p}" in mem: self.ofi_prev_book = mem[f"ofi_prev_{p}"]
        if f"tox_{p}" in mem: self.tox_hist = mem[f"tox_{p}"]
        if f"sweep_{p}" in mem: self.prev_sweep = mem[f"sweep_{p}"]
        if f"hist_{p}" in mem: self.mid_hist = mem[f"hist_{p}"]
        if f"olv_{p}" in mem: self.olivia_dir = mem[f"olv_{p}"]
        if f"olv_ts_{p}" in mem: self.olivia_ts = mem.get(f"olv_ts_{p}", 0)
        if f"_bot_det_{p}" in mem: self.bot_det = mem[f"_bot_det_{p}"]
        if f"ar_c_{p}" in mem: self.ar_coeffs = mem[f"ar_c_{p}"]
        if f"ar_ft_{p}" in mem: self.ar_fit_ts = mem.get(f"ar_ft_{p}", -999)
        if f"ema_{p}" in mem: self.ema = mem[f"ema_{p}"]


# ============================================================
# CONFIG -- Edit this per round
# ============================================================

CONFIG = {
    # === P4 ROUND 3 PRODUCTS (Phase 2 leaderboard reset, 2026-04-24) ===
    # Recon: HYDROGEL stable ~9990 (pegged, ASH-like).
    # VE ~5250 with tiny upward drift (5246→5255 over 3 days) — treat as pegged.
    # Voucher S≈5250: deep ITM 4000/4500 trade intrinsic (TV=0.01),
    # tradable ITM/ATM 5000-5500, dead OTM 6000/6500 (mid=0.50 decays to 0).
    "HYDROGEL_PACK": {
        "type": "pegged",
        # R4 v7 (2026-04-27): FV=10031 — matches friend's WINNING live config (+25K).
        # Live FV evidence (monotonic):
        #   FV=9990 → +$7K, FV=10000 → +$9.6K, FV=10010 → +$11.7K → +$17.4K (our v6),
        #   FV=10031 → +$25.4K (friend's). Bigger live wins as FV approaches actual mid.
        # Backtest regresses by $32K but historical walks ≠ live R4 day 2/3 walks.
        # Mechanic: higher FV → wall_mid higher → OFFLOAD COVER (ask<=wall_mid)
        # fires regularly → 235 buys + 406 sells (vs our 22+222 at FV=10010).
        "fair_value": 10031,
        "position_limit": 200,
        "take_width": 1,
        "make_offset": 2,
        "make_size": 60,
        "take_edge_pegged": 10,  # v8 sweep: te=10 +$1.1K bt, +$1.6K 1K-slice vs te=15
        # v7 LIVE-TOXICITY FIX: log 402706 showed HYD pinned at +200 while mid fell -51,
        # toxicity 53%, drawdown 16K. take side blew past thresholds because
        # throttle_takes_by_inventory was OFF. ENABLE NOW + sweep-tuned 100/180.
        # Sweep: 60/120=154K, 80/160=200K, 100/180=222K, 120/180=222K (ties).
        # v8 RE-SWEEP 2026-04-26: looser 160/200 = +233K bt (+15K vs 100/180) AND
        # +12.9K 1K-slice (+1K vs 100/180). Field intel showed leaders run un-throttled
        # at +37K-80K live; 160/200 captures more upside while keeping a stop near limit.
        "throttle_takes_by_inventory": True,
        "inventory_threshold_1": 160,
        "inventory_threshold_2": 200,
        "overload_offload": True,
        "overload_frac": 0.9,
        "overload_tol": 2,
        "overload_cut_frac": 0.25,
        # v8 SLOPE-DEFENSIVE: live log 412166 ended +185 long while mid drifted -51
        # (~9.5K MTM bleed). Block alpha buy-takes when 50-tick OLS slope < -0.2.
        # R4 v2 sweep (2026-04-26): live log 490589 showed mirror failure (-200 short,
        # mid up, $2.5K bleed). Sweep tested {thr, gate} pairs:
        #   v8 (0.20/160): R3=234,750 R4=224,687 (baseline)
        #   v2a (0.10/100): R3=233,602 (-1148) R4=224,471
        #   v2c (0.15/120): R3=235,740 (+990)  R4=224,684 (≈flat) ← CHOSEN
        #   v2e (0.20/100): R3=232,626 (-2124) R4=225,277 (+590)
        # v2c strictly dominates v8 on backtest (no regression) AND engages defense
        # earlier (gate 120 vs 160) for live HYD short-pin protection.
        "slope_defensive_thr": 0.15,
        "slope_defensive_window": 50,
        "slope_defensive_inv_gate": 120,
        # R4 v3 (2026-04-27): per-tick take rate cap. Without this, take loop can
        # consume 200 units in 1-2 ticks (live log 496734: HYD pinned -200 in <100t).
        # Cap=50 doesn't bind on normal takes but prevents catastrophic pinning.
        # Sweep: cap=50 strictly beats no-cap on R3_30K (+$86), R3_1K (+$45), R4_30K (+$129).
        "max_take_per_tick": 50,
    },
    "VELVETFRUIT_EXTRACT": {
        "type": "pegged",
        "fair_value": 5250,
        "position_limit": 200,
        "take_width": 1,
        "make_offset": 2,
        "make_size": 60,
        # v6 take_edge_pegged=5: VEF take_pnl was -733K (toxic crosses on Gaussian
        # random walk). Returns are confirmed Gaussian (skew -0.03, kurt 0.35,
        # AR(1)=0.997) — no directional edge, only spread+toxicity-filter alpha.
        # Wider take_edge filters out random-walk-induced bad crosses. +8.7K bt.
        "take_edge_pegged": 5,
        # v7 LIVE-TOXICITY FIX: log 402706 showed VEF pinned at -200 short,
        # toxicity 44%. throttle_takes_by_inventory was OFF — takes piled to limit.
        # ENABLE NOW + sweep-tuned 100/180 (mirrors HYD).
        # v8 RE-SWEEP 2026-04-26: mirrors HYD upgrade to 160/200.
        "throttle_takes_by_inventory": True,
        "inventory_threshold_1": 160,
        "inventory_threshold_2": 200,
        "is_options_underlying": True,  # 10 VEVs trade on this
        # Reserve 50 units of UL capacity for skew_mm delta hedging (v4).
        # Pegged uses the full 200 limit; hedging layers on top and validate_orders
        # trims if total would breach. hedge_min_adj suppresses tiny rebalances.
        "hedge_budget": 50,
        "hedge_min_adj": 3,
        # v8 SLOPE-DEFENSIVE: live log 412166 ended -196 short on neutral random walk.
        # Smaller thr (VEF vol 1.16 vs HYD 2.28) — block alpha takes on smaller slopes.
        "slope_defensive_thr": 0.1,
        "slope_defensive_window": 50,
        # R4 v1: informed-counterparty lean. Offline p4r4 analysis identified
        # Mark 67 (165 buys, +$1.97 fwd) as the only informed VEF trader (pure
        # Olivia-equivalent: rare+large+100% one-sided). Mark 49 was hypothesized
        # as informed seller but in-trader fwd-tracking shows fs_avg=+1.19 (price
        # rises AFTER Mark 49 sells) → Mark 49 is just the dumb counterparty getting
        # picked off by Mark 67. Detection rule auto-filters Mark 49 out.
        # Detection is runtime; no names hardcoded.
        # No-ops on R3-style data (buyer/seller fields None).
        "informed_lean": True,
        "informed_max_relax": 10,
        "informed_relax_mult": 5.0,
        "informed_fwd_thr": 0.5,
        # R4 v3: same per-tick take cap as HYD — prevents fast pinning on VEF too
        "max_take_per_tick": 50,
    },
    # VEV voucher chain — options strategy with Frankfurt IV scalping.
    # total_option_days=8 matches historical CSV day_0 (tutorial) starting at TTE=8.
    # For live R3 simulation, day_offset is set at runtime via overrides (R3 = offset 3).
    # Deep ITM: TV≈0-1, options MR loses money (VEV_4000 -13.5K baseline) — sit out.
    # v5 ALPHA HUNT 2026-04-24: deep-ITM vouchers have spread 20.8/15.9 (widest in chain).
    # 30 fills/1K ticks activity. Quote inside the wide BBO at intrinsic±make_edge.
    # Delta=1 (deep ITM) so each fill creates 1:1 UL exposure, hedged via skew_delta_agg.
    "VEV_4000": {
        "type": "voucher_intrinsic_mm", "position_limit": 300,
        "underlying": "VELVETFRUIT_EXTRACT", "strike": 4000,
        "make_size": 15, "make_edge": 10, "take_edge": 8,
        "max_pos_frac": 0.6, "inv_skew_ticks": 4,
    },
    # VEV_4500 has only 1 trade in entire p4r3 dataset — no counterparty. Skip.
    "VEV_4500": {"type": "do_nothing", "position_limit": 300},
    # ITM strikes: live test 364694 showed VEV_5100 -355 (-4.7/fill, taking bad fills),
    # VEV_5000 +22 (no edge worth the risk). Both pulled to do_nothing.
    # Options unlock 2026-04-24 v2: smile calibrated offline on p4r3 days 0+1
    # (analysis/fit_p4r3_smile.py). OOS day-2 mean |resid|=0.0065 (vs p3 coefs
    # 0.0738 — 11x better fit). Per-strike residual biases detected:
    #   K=5100: -0.0043 (overpriced), K=5200: +0.0031, K=5300: +0.0080 (underpriced)
    #   K=5400: -0.0107 (overpriced), K=5500: +0.0082 (underpriced)
    # Enabling strikes with |bias| > 0.004 — largest deviations = strongest signal.
    # v4 (2026-04-24): skew_mm on underpriced strikes. Pure 2-sided MM around
    # offline-calibrated theo (analysis/fit_p4r3_smile.py), no EMA self-reference.
    # Portfolio delta hedged via VEF at end of tick.
    # Per-strike OOS bias (day 2): K=5300 +0.80%, K=5500 +0.82%, K=5100 -0.43%, K=5400 -1.07%.
    # Starting with K=5300 (largest underpricing + healthiest mid=58).
    "VEV_5000": {"type": "do_nothing", "position_limit": 300},
    # v5 IV-scalp battery: Frankfurt-style with offline-calibrated smile.
    # iv_scalp_diag.py shows theo_diff stdevs: 5100=0.93, 5200=0.91, 5300=1.13, 5400=0.77, 5500=0.42
    # Bias correction (mean_theo_diff EMA) absorbs persistent smile mispricing,
    # we trade RESIDUAL deviations. iv_scalp_thr ~ 0.5*stdev to keep gate open.
    # v5 (2026-04-25): switched OTM strikes from options→skew_mm with bias-corrected
    # smile theo. Persistent per-strike smile bias up to ±2 was causing one-sided
    # take fires; bias EMA absorbs it, MM trades around (theo + bias).
    # OTM strikes parked: tested skew_mm with bias-corrected smile (TTE=7, refit
    # 2026-04-25). Take-only: ~0 fills (signals too small after bias absorbs).
    # 2-sided: 3-6 make fills/strike, all small losses (adverse selection).
    # Net p4r3: +740 over baseline. Not worth the complexity. Revisit if better
    # smile fit or microstructure model emerges.
    # Wiki: position_limit=300 for ALL voucher strikes. Fix from prior 50 typo.
    # R4 v8 (2026-04-28): bumped OTM_SHORT targets to MAX position (300) on
    # 5300/5400/5500 — friend's v8 at 250 hit +$26,872 live, sweep shows 300 adds
    # another +$1.8K backtest. V52=200 sweet spot (V52=250 sells extra units at
    # worse prices). Combined OTM alpha: V51 $177 + V52 $1,711 + V53 $5,709
    # + V54 $3,082 + V55 $1,557 = $12,236 backtest alpha.
    "VEV_5100": {"type": "otm_short", "position_limit": 300,
                 "target_short": 50, "stop_loss_mid": 200, "max_sell_per_tick": 15},
    "VEV_5200": {"type": "otm_short", "position_limit": 300,
                 "target_short": 200, "stop_loss_mid": 130, "max_sell_per_tick": 30},
    "VEV_5300": {"type": "otm_short", "position_limit": 300,
                 "target_short": 300, "stop_loss_mid": 100, "max_sell_per_tick": 30},
    "VEV_5400": {"type": "otm_short", "position_limit": 300,
                 "target_short": 300, "stop_loss_mid": 50, "max_sell_per_tick": 30},
    "VEV_5500": {"type": "otm_short", "position_limit": 300,
                 "target_short": 300, "stop_loss_mid": 30, "max_sell_per_tick": 30},
    # Dead vouchers: mid=0.50, converges to 0 at expiry. Not worth quoting.
    "VEV_6000": {"type": "do_nothing", "position_limit": 300},
    "VEV_6500": {"type": "do_nothing", "position_limit": 300},

    # === P4 ROUND 1 PRODUCTS ===
    "ASH_COATED_OSMIUM": {
        "type": "pegged",
        "fair_value": 10000,
        "position_limit": 80,
        "take_width": 3,               # R2 empirical: tw=3 gave +169 vs tw=1 on test submission
        "make_offset": 4,
        "make_size": 80,
        "inventory_threshold_1": 20,   # grid: thr1 flat (20-50 equal), 20 is fine
        "inventory_threshold_2": 75,   # grid: 75 > 65 by +218 PnL — less choking
    },
    "INTARIAN_PEPPER_ROOT": {
        "type": "pegged",
        "fair_value": 13000,     # day 1 predicted start (day0=12000, +1000/day)
        "fv_drift": 0.15,         # R2 empirical: 0.15 gave +177 vs 0.111 baseline (any >0.09 works ~same)
        "position_limit": 80,
        "take_width": 1,         # tight take at drift FV — drift provides the edge
        "make_offset": 1,        # max tight — 18% toxicity on server
        "make_size": 80,         # max position — known FV with drift
        "inventory_threshold_1": 50,
        "inventory_threshold_2": 70,
        "drift_bid_offset": 4,   # passive bid = wall_mid + X (grid-searched)
        "drift_ask_margin": 3,   # passive bid = best_ask - Y (grid-searched)
    },

    # === P3 products (kept for auto-classifier fallback) ===
    "KELP": {
        "type": "ar_olivia",
        "position_limit": 50,
        "ar_order": 5,
        "ar_refit_every": 25,
        "ar_min_history": 50,
        "olivia_window": 5,
        "make_size": 30,
    },
    "RAINFOREST_RESIN": {
        "type": "pegged",
        "fair_value": 10000,
        "position_limit": 50,
        "take_width": 1,
        "make_offset": 4,
        "make_size": 50,
        "inventory_threshold_1": 45,
    },
    "SQUID_INK": {
        "type": "olivia_follow",
        "position_limit": 50,
        "target_position": 50,
        "vol_window": 50,
        "vol_mr_threshold": 2.0,
        "make_size": 15,
    },
    "PICNIC_BASKET1": {
        "type": "basket_arb",
        "position_limit": 60,
        "components": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
        "entry_z": 4.0,
        "exit_z": 0.3,
        "informed_thr_adj": 10,
        "trade_size": 60,
        "hedge_ratio": 0.0,
    },
    "PICNIC_BASKET2": {
        "type": "basket_arb",
        "position_limit": 100,
        "components": {"CROISSANTS": 4, "JAMS": 2},
        "entry_z": 3.0,
        "exit_z": 0.3,
        "informed_thr_adj": 10,
        "trade_size": 100,
        "hedge_ratio": 0.0,
    },
    "EMERALDS": {
        "type": "pegged",
        "fair_value": 10000,
        "position_limit": 80,
        "take_width": 1,
        "make_offset": 4,
        "make_size": 80,
        "inventory_threshold_1": 40,
        "inventory_threshold_2": 65,
    },
    "TOMATOES": {
        "type": "wide_spread",
        "position_limit": 80,
        "take_ask_threshold": 6,
        "bid_comp_threshold": 5,
        "ema_alpha": 0.005,
        "ema_bid_skip_thr": 5.0,
        "ema_ask_skip_thr": 2.5,
        "make_size": 80,
        "inventory_threshold_1": 70,
        "inventory_threshold_2": 78,
    },
}

# ============================================================
# IMP 4: RUNTIME AUTO-CLASSIFICATION
# Classifies unknown products from first order book snapshot.
# If a product appears in state but not in CONFIG, this assigns
# a strategy type so we never fall through to dumb generic_mm.
# ============================================================

def classify_product_live(od: OrderDepth, product: str, mem: dict,
                          state=None) -> dict:
    """Classify an unknown product from its order book + state context.
    Returns a config dict with type and sensible defaults.
    Re-evaluates at tick 50 (with accumulated stats) then every 500 ticks."""
    cache_key = f"_auto_cfg_{product}"
    tick_key = f"_auto_tick_{product}"

    # Re-evaluate classification: at tick 50 (stat-based upgrade) then every 500
    cached = mem.get(cache_key)
    ticks_since = mem.get(tick_key, 0)
    reclassified_key = f"_auto_reclass_{product}"
    already_reclassified = mem.get(reclassified_key, False)
    if cached:
        # Reclassify at tick 50 if we haven't done a stat-based upgrade yet
        if ticks_since == 50 and not already_reclassified:
            pass  # fall through to reclassify with stats
        elif ticks_since < 500:
            mem[tick_key] = ticks_since + 1
            return cached
        # else: 500-tick periodic recheck
    mem[tick_key] = 0

    # --- Name-based detection (works with or without state) ---
    # CONVERSION: detect by known conversion product name patterns
    conv_keywords = ["MACARON", "ORCHID"]  # Only confirmed conversion products; new ones detected via conversionObservations
    if any(kw in product.upper() for kw in conv_keywords):
        cfg = {"type": "conversion_arb", "position_limit": 75,
               "max_conversions": 10, "order_size": 50}
        mem[cache_key] = cfg
        return cfg

    # OPTIONS: detect from product name patterns (check BEFORE baskets —
    # option patterns are more specific and basket keywords like "SET" can
    # false-match substrings in option names like "DREAM_ASSET_PUT_8000")
    option_patterns = ["_VOUCHER_", "_COUPON_", "_CALL_", "_PUT_", "_OPTION_"]
    for pat in option_patterns:
        if pat in product:
            base = product.split(pat)[0]
            strike_str = product.split(pat)[-1] if pat in product else ""
            strike = None
            try:
                strike = float(strike_str)
            except (ValueError, TypeError):
                pass
            # Only classify as options if underlying exists in order_depths
            if state and base in state.order_depths:
                cfg = {"type": "options", "position_limit": 200,
                       "underlying": base, "strike": strike or 10000,
                       "total_option_days": 8, "day_offset": 0,
                       "smile_coeffs": [0.27362531, 0.01007566, 0.14876677],
                       "theo_norm_window": 20, "iv_scalp_window": 100,
                       "iv_scalp_thr": 0.7, "thr_open": 0.5, "thr_close": 0.0,
                       "low_vega_thr_adj": 0.5,
                       "iv_scalp_min_strike": 9750, "mr_strike": 9500,
                       "ul_mr_window": 10, "ul_mr_thr": 15,
                       "opt_mr_window": 30, "opt_mr_thr": 5,
                       "underlying_limit": 400}
                # Apply overrides from backtest config (e.g., day_offset per round)
                overrides = mem.get("_options_config_overrides", {})
                cfg.update(overrides)
                mem[cache_key] = cfg
                return cfg

    # OPTIONS UNDERLYING: if this product is the underlying for option products,
    # use minimal passive MM (options delta hedge handles directional exposure)
    if state is not None:
        is_underlying = False
        for other_p in state.order_depths:
            for pat in ["_VOUCHER_", "_COUPON_", "_CALL_", "_PUT_", "_OPTION_"]:
                if pat in other_p and product in other_p and other_p != product:
                    is_underlying = True
                    break
            if is_underlying:
                break
        if is_underlying:
            # Options strategy handles underlying MR via hedge_orders.
            # Use minimal generic_mm as fallback — options run_options will
            # override with ul_mr orders via hedge_orders mechanism.
            cfg = {"type": "generic_mm", "position_limit": 400, "make_size": 5,
                   "is_options_underlying": True}
            mem[cache_key] = cfg
            return cfg

    # PAIRS: detect if this product already has a cointegrated partner discovered
    coint_pairs = mem.get("_coint_pairs", {})
    if product in coint_pairs:
        partner = coint_pairs[product]
        lim = 50  # default, will be overridden by server
        cfg = {"type": "pairs_arb", "position_limit": lim,
               "partner": partner,
               "partner_limit": lim,
               "entry_z": 2.0, "exit_z": 0.3,
               "trade_size": 10, "spread_mean": 0.0}
        mem[cache_key] = cfg
        return cfg

    # BASKET: detect by product name keywords
    # "SET" removed — too generic, matches substrings like "ASSET", "OFFSET", "SUNSET"
    basket_keywords = ["BASKET", "ETF", "INDEX", "BUNDLE", "HAMPER", "BOX", "CRATE"]
    if any(kw in product.upper() for kw in basket_keywords):
        known_baskets = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2},
            "GIFT_BASKET": {"CHOCOLATE": 4, "STRAWBERRIES": 6, "ROSES": 1},
        }
        # Frankfurt: BASKET_THRESHOLDS = [80, 50] for PB1, PB2
        # P2 GIFT_BASKET has different premium scale — use z-score fallback
        # Absolute thresholds tested and lost money in backtest — premium deviation
        # oscillates but adverse selection kills PnL. Use z-score for all baskets.
        known_entry_abs = {}  # all baskets use z-score fallback
        components = known_baskets.get(product, {})
        if not components and state is not None:
            candidate_components = []
            for other_p in state.order_depths:
                if other_p == product:
                    continue
                up = other_p.upper()
                if any(kw in up for kw in basket_keywords + ["VOUCHER", "COUPON", "CALL", "PUT", "OPTION"]):
                    continue
                if any(kw in up for kw in ["MACARON", "ORCHID"]):
                    continue
                candidate_components.append(other_p)
            if candidate_components:
                components = {c: 1 for c in candidate_components}
        limit = 60
        # Frankfurt: absolute thresholds + high informed adj
        # Stanford: NO component hedging (hedge_ratio=0.0) — adds risk, not edge
        basket_entry = known_entry_abs.get(product, None)
        cfg = {"type": "basket_arb", "position_limit": limit,
               "components": components, "premium_mean": 0.0,
               "entry_z": 3.0, "exit_z": 0.3,
               "informed_thr_adj": 10, "trade_size": limit,
               "hedge_ratio": 0.0,
               "auto_discover_weights": len(known_baskets.get(product, {})) == 0}
        if basket_entry is not None:
            cfg["entry_abs"] = basket_entry
        mem[cache_key] = cfg
        return cfg

    # --- State-dependent detection ---
    if state is not None:
        # CONVERSION: if product has ConversionObservation
        if hasattr(state, 'observations') and state.observations:
            conv = state.observations.conversionObservations
            if conv and product in conv:
                cfg = {"type": "conversion_arb", "position_limit": 75,
                       "max_conversions": 10, "order_size": 50}
                mem[cache_key] = cfg
                return cfg

    # --- Standard classification from order book shape + accumulated stats ---
    if not od.buy_orders or not od.sell_orders:
        cfg = {"type": "generic_mm", "position_limit": 50, "make_size": 15}
        mem[cache_key] = cfg
        return cfg

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2.0

    # Pull accumulated stats from ProductState if available (tick 50+ reclassification)
    ps_data = mem.get(f"_ps_{product}", {})
    mid_hist = ps_data.get("mid_hist", [])
    rvol = ps_data.get("realized_vol", 0)
    has_stats = len(mid_hist) >= 30

    # Compute accumulated stats for informed classification
    range_pct = 0.0
    ac1 = 0.0
    if has_stats:
        price_range = max(mid_hist) - min(mid_hist)
        avg_mid = sum(mid_hist) / len(mid_hist)
        range_pct = (price_range / avg_mid * 100) if avg_mid > 0 else 0
        # Compute AC1 from mid changes
        changes = [mid_hist[i] - mid_hist[i-1] for i in range(1, len(mid_hist))]
        if len(changes) > 5:
            mc = sum(changes) / len(changes)
            var_c = sum((c - mc)**2 for c in changes) / len(changes)
            if var_c > 0:
                cov = sum((changes[i] - mc) * (changes[i-1] - mc) for i in range(1, len(changes))) / (len(changes) - 1)
                ac1 = cov / var_c

    # COMPONENT: if this product is a component of any basket, use ar_olivia
    # Frankfurt pattern: basket components (CROISSANTS etc) are Olivia-informed
    config_ref = mem.get("_config_ref")
    if not isinstance(config_ref, dict):
        config_ref = {}
    for cfg_name, cfg_val in config_ref.items():
        if isinstance(cfg_val, dict) and cfg_val.get("type") == "basket_arb":
            comps = cfg_val.get("components", {})
            if product in comps:
                cfg = {"type": "ar_olivia", "position_limit": 50,
                       "ar_order": 4, "ar_refit_every": 25, "ar_min_history": 50,
                       "olivia_window": 5, "make_size": 30}
                mem[cache_key] = cfg
                return cfg
    # Also check known basket patterns from discovery
    for disc_k, disc_v in mem.items():
        if disc_k.startswith("disc_") and isinstance(disc_v, dict):
            disc_result = disc_v.get("result", {})
            if product in disc_result:
                cfg = {"type": "ar_olivia", "position_limit": 50,
                       "ar_order": 4, "ar_refit_every": 25, "ar_min_history": 50,
                       "olivia_window": 5, "make_size": 30}
                mem[cache_key] = cfg
                return cfg

    # Check if pegged: mid near a round number with reasonable spread
    for base in [100, 500, 1000, 2500, 5000, 10000, 20000, 50000]:
        nearest = round(mid / base) * base
        if nearest > 0 and abs(mid - nearest) < max(5, base * 0.001) and spread <= 20:
            cfg = {"type": "pegged", "fair_value": int(nearest),
                   "position_limit": 50, "take_width": 1,
                   "make_offset": max(2, int(spread // 4)), "make_size": 30}
            mem[cache_key] = cfg
            mem[reclassified_key] = True
            return cfg

    if spread > 8:
        cfg = {"type": "wide_spread", "position_limit": 50,
               "take_ask_threshold": max(4, spread // 2),
               "bid_comp_threshold": max(3, spread // 3),
               "ema_alpha": 0.005, "ema_bid_skip_thr": 5.0,
               "ema_ask_skip_thr": 2.5, "make_size": 40,
               "inventory_threshold_1": 35, "inventory_threshold_2": 45}
    elif spread <= 4:
        if has_stats and range_pct > 5.0:
            # High range + tight spread = volatile spike product (SQUID_INK-like)
            # ar_olivia would get destroyed here — need olivia_follow
            cfg = {"type": "olivia_follow", "position_limit": 50,
                   "target_position": 50, "vol_window": 50,
                   "vol_mr_threshold": 2.0, "make_size": 10}
        elif has_stats and ac1 < -0.15:
            # Strongly mean-reverting + tight spread = KELP-like
            cfg = {"type": "ar_olivia", "position_limit": 50,
                   "ar_order": 4, "ar_refit_every": 25, "ar_min_history": 50,
                   "olivia_window": 5, "make_size": 25}
        else:
            # Default tight-spread: ar_olivia (safe, works on both)
            cfg = {"type": "ar_olivia", "position_limit": 50,
                   "ar_order": 4, "ar_refit_every": 25, "ar_min_history": 50,
                   "olivia_window": 5, "make_size": 20}
    elif has_stats:
        # Medium spread (4-8): with stats, try to promote from generic_mm
        if range_pct > 5.0:
            # Volatile with medium spread — olivia_follow with caution
            cfg = {"type": "olivia_follow", "position_limit": 50,
                   "target_position": 50, "vol_window": 50,
                   "vol_mr_threshold": 2.0, "make_size": 8}
        elif ac1 < -0.15:
            # Mean-reverting medium spread — ar_olivia still works
            cfg = {"type": "ar_olivia", "position_limit": 50,
                   "ar_order": 4, "ar_refit_every": 25, "ar_min_history": 50,
                   "olivia_window": 5, "make_size": 15}
        else:
            cfg = {"type": "generic_mm", "position_limit": 50, "make_size": 15}
    else:
        cfg = {"type": "generic_mm", "position_limit": 50, "make_size": 15}

    if has_stats:
        mem[reclassified_key] = True
    mem[cache_key] = cfg
    return cfg

# ============================================================
# PROVEN TOOLS (ported from 18442.py — scored +2,734 on server)
# Kalman FV, OFI, toxicity tracking, book sweep detection
# ============================================================

# Standalone FV/OFI/toxicity/sweep functions REMOVED — all computed in ProductState.update()
# Remaining utilities: check_olivia, detect_informed_bot, AR, cholesky_solve

# Gap 2: distilled Olivia decision cascade — depth ≤ 4 if/else chain trained
# offline on P3 R5 by analysis/train_olivia_classifier.py and written to
# analysis/fingerprints/olivia_cascade.json. Format:
#   {"rules": [{"feat": "size", "op": ">=", "thr": 8, "vote": "LONG"}, ...]}
# Default empty = inert; check_olivia behaviour unchanged unless cfg gates it on
# AND a fitted cascade is present.
OLIVIA_CASCADE = {"rules": []}
try:
    _cascade_path = ""
    if False:
        with open(_cascade_path) as _fh:
            OLIVIA_CASCADE = json.load(_fh)
except Exception:
    pass


def _apply_olivia_cascade(market_trades, mid_hist):
    """Evaluate the distilled cascade against the most recent large market trade.

    Features per trade:
      size       — abs(quantity)
      side       — +1 buy / -1 sell
      px_dev     — trade_price - rolling_median(mid)
    Returns 'LONG', 'SHORT', or None.
    """
    rules = (OLIVIA_CASCADE or {}).get("rules") or []
    if not rules or not market_trades:
        return None
    # Pick the largest-size recent trade — Olivia bot signature is rare + large.
    best = max(market_trades, key=lambda t: abs(int(t.quantity)))
    size = abs(int(best.quantity))
    side = 1 if best.quantity > 0 else -1
    if mid_hist:
        srt = sorted(mid_hist[-50:])
        med = srt[len(srt) // 2]
    else:
        med = float(best.price)
    px_dev = float(best.price) - float(med)
    feats = {"size": size, "side": side, "px_dev": px_dev}

    def _cmp(val, op, thr):
        if op == ">=": return val >= thr
        if op == ">":  return val >  thr
        if op == "<=": return val <= thr
        if op == "<":  return val <  thr
        if op == "==": return val == thr
        return False

    for r in rules:
        # Support both single-condition (feat/op/thr) and multi-condition (conds: [...])
        conds = r.get("conds")
        if not conds:
            f = r.get("feat"); op = r.get("op"); thr = r.get("thr")
            if f is None or op is None or thr is None:
                continue
            conds = [{"feat": f, "op": op, "thr": thr}]
        if all(_cmp(feats.get(c.get("feat"), 0), c.get("op"), c.get("thr")) for c in conds):
            v = r.get("vote")
            if v in ("LONG", "SHORT"):
                return v
    return None


def check_olivia(market_trades: List[Trade], own_trades: List[Trade]) -> Optional[str]:
    """Scan trades for informed trader 'Olivia'. Returns 'LONG', 'SHORT', or None.
    Frankfurt Hedgehogs' secret weapon -- worth 5-10K PnL on SQUID alone."""
    latest_buy_ts = -1
    latest_sell_ts = -1
    all_trades = list(market_trades) + list(own_trades)
    for t in all_trades:
        if t.buyer == "Olivia":
            latest_buy_ts = max(latest_buy_ts, t.timestamp)
        if t.seller == "Olivia":
            latest_sell_ts = max(latest_sell_ts, t.timestamp)
    if latest_buy_ts > latest_sell_ts and latest_buy_ts >= 0:
        return "LONG"
    elif latest_sell_ts > latest_buy_ts and latest_sell_ts >= 0:
        return "SHORT"
    return None

def detect_informed_bot(market_trades: List[Trade], mid: float,
                        mem: dict, product: str) -> Optional[str]:
    """Bayesian informed-trader detection — replaces frequency-based hit_rate > 0.65.

    Maintains Beta(alpha, beta) posterior on P(large trade predicts price move).
    Prior: Beta(1, 1) = uniform. Each resolved trade updates the posterior.
    Uninformed baseline: ~40% of random large trades "predict" moves > 0.3.

    Advantages over frequency-based:
    - Graceful confidence ramp (8 correct out of 10 ≠ 80 out of 100)
    - No arbitrary thresholds — uses P(informed) > 0.85 credible interval
    - Exponential decay on old evidence (lambda=0.995 per tick ≈ 500-tick half-life)
    - Smaller state: ~200 bytes vs ~2KB (no mid_hist/trade lists)

    Returns 'LONG', 'SHORT', or None."""
    key = f"_bot_det_{product}"
    det = mem.get(key, {
        "a": 1.0, "b": 1.0,       # Beta posterior: alpha, beta
        "pend": [],                 # pending trades awaiting resolution: [{side, mid, ts}]
        "sig": None,               # current signal direction
        "sig_ts": 0,               # tick when signal was set
        "tick": 0,                 # tick counter
        "last_side": None,         # most recent large trade side
    })

    det["tick"] = det.get("tick", 0) + 1
    tick = det["tick"]

    # --- Decay old evidence: shrink alpha/beta toward prior ---
    # lambda=0.995/tick ≈ half-life of 139 ticks. Prevents stale P2/P3 evidence
    # from dominating. Floor at 1.0 (the prior) so we never go below uniform.
    decay = 0.995
    det["a"] = max(1.0, det["a"] * decay)
    det["b"] = max(1.0, det["b"] * decay)

    # --- Resolve pending trades that now have 10-tick lookahead ---
    lookahead = 10
    resolved = []
    still_pending = []
    for p in det.get("pend", []):
        if tick - p["ts"] >= lookahead:
            move = mid - p["mid"]  # approximate: current mid vs trade-time mid
            # Trade "correct" if buy preceded up-move or sell preceded down-move
            if (p["side"] == "buy" and move > 0.3) or (p["side"] == "sell" and move < -0.3):
                det["a"] = det["a"] + 1.0
            else:
                det["b"] = det["b"] + 1.0
            resolved.append(p)
        else:
            still_pending.append(p)
    det["pend"] = still_pending

    # H6: adaptive "large trade" threshold via rolling p95 of trade sizes.
    # Floors at 10 (existing hardcode) so behavior never weakens. Refresh every
    # 50 ticks to bound per-tick cost.
    sz_hist = mem.get(f"_sz_hist_{product}", [])
    for t in market_trades:
        sz_hist.append(int(t.quantity))
    if len(sz_hist) > 200:
        sz_hist = sz_hist[-200:]
    mem[f"_sz_hist_{product}"] = sz_hist
    thr_key = f"_sz_thr_{product}"
    thr_ts_key = f"_sz_thr_ts_{product}"
    if tick - mem.get(thr_ts_key, -999) >= 50 and len(sz_hist) >= 30:
        srt = sorted(sz_hist)
        p95 = srt[int(0.95 * (len(srt) - 1))]
        mem[thr_key] = max(10, p95)
        mem[thr_ts_key] = tick
    size_thr = mem.get(thr_key, 10)

    # --- Log new large trades ---
    for t in market_trades:
        if t.quantity < size_thr:
            continue
        side = "buy" if t.price >= mid else "sell"
        det["pend"].append({"side": side, "mid": mid, "ts": tick})
        det["last_side"] = side
    # Cap pending list (shouldn't grow large, but safety)
    if len(det["pend"]) > 30:
        det["pend"] = det["pend"][-30:]

    # --- Signal decay: expire after 50 ticks ---
    if det.get("sig") and tick - det.get("sig_ts", 0) > 50:
        det["sig"] = None

    # --- Bayesian decision: is this bot informed? ---
    # P(hit_rate > 0.5 | data) using Beta CDF complement
    # For Beta(a, b), P(p > 0.5) = 1 - I_0.5(a, b) where I is regularized incomplete beta
    # We use a fast approximation: posterior mean + concentration check
    a, b = det["a"], det["b"]
    n = a + b - 2  # effective sample size (subtract prior)
    posterior_mean = a / (a + b)

    # Require: (1) enough evidence, (2) posterior mean well above chance (0.4 baseline)
    # With Beta, P(p > 0.5 | Beta(a,b)) ≈ 1 when mean >> 0.5 and n is large
    # Threshold: mean > 0.58 AND effective samples > 6
    # This is equivalent to ~85% credible interval excluding 0.5
    if posterior_mean > 0.58 and n > 6:
        if det.get("last_side"):
            det["sig"] = "LONG" if det["last_side"] == "buy" else "SHORT"
            det["sig_ts"] = tick
    # Don't null signal on low posterior — let it decay via 50-tick window

    mem[key] = det
    return det.get("sig")


def update_informed_counterparty(market_trades, mid, mem, product, ts,
                                  min_obs=20, bias_thr=0.7, fwd_thr=0.5,
                                  lookahead=100):
    """R4+ counterparty alpha: detect informed buyers/sellers from buyer/seller fields.

    Pre-R4 the buyer/seller fields were None — this function silently no-ops then.
    R4+ they're populated ("Mark 01" through "Mark 67" in p4r4 historical data).

    Detection rule (offline-validated on p4r4 days 1-3):
      - counterparty has >= min_obs trades observed
      - one-sided bias > bias_thr (e.g. 90% buys → 0.8 bias)
      - avg forward return aligned with direction (buys precede up-moves)

    Mark 67 (165 buys, 0 sells, +$1.97 fwd) and Mark 49 (105 sells, +$2 fwd) both
    pass on VEF in offline data — pure Olivia-equivalents.

    Returns ('buy', edge, name) | ('sell', edge, name) | None.
    Edge is the absolute avg fwd return — caller scales lean aggressiveness by it.
    """
    if not market_trades:
        return None
    key = f"_cp_{product}"
    cp = mem.get(key, {"marks": {}})
    marks = cp["marks"]

    # Resolve pending forward returns from prior ticks
    for name, s in marks.items():
        new_pb = []
        for pts, pmid in s.get("pb", []):
            if ts - pts >= lookahead:
                s["fb_sum"] = s.get("fb_sum", 0.0) + (mid - pmid)
                s["fb_n"] = s.get("fb_n", 0) + 1
            else:
                new_pb.append((pts, pmid))
        s["pb"] = new_pb
        new_ps = []
        for pts, pmid in s.get("ps", []):
            if ts - pts >= lookahead:
                s["fs_sum"] = s.get("fs_sum", 0.0) + (mid - pmid)
                s["fs_n"] = s.get("fs_n", 0) + 1
            else:
                new_ps.append((pts, pmid))
        s["ps"] = new_ps

    fired = None
    for t in market_trades:
        buyer = getattr(t, "buyer", None)
        seller = getattr(t, "seller", None)
        for name, side in [(buyer, "buy"), (seller, "sell")]:
            if not name:
                continue
            s = marks.setdefault(name, {
                "nb": 0, "ns": 0, "fb_sum": 0.0, "fb_n": 0,
                "fs_sum": 0.0, "fs_n": 0, "pb": [], "ps": [],
            })
            if side == "buy":
                s["nb"] += 1
                s["pb"].append((ts, mid))
                if len(s["pb"]) > 30:
                    s["pb"] = s["pb"][-30:]
            else:
                s["ns"] += 1
                s["ps"].append((ts, mid))
                if len(s["ps"]) > 30:
                    s["ps"] = s["ps"][-30:]

            n_total = s["nb"] + s["ns"]
            if n_total < min_obs:
                continue
            bias = (s["nb"] - s["ns"]) / n_total
            if side == "buy" and bias > bias_thr and s["fb_n"] >= 5:
                avg_fwd = s["fb_sum"] / s["fb_n"]
                if avg_fwd > fwd_thr:
                    fired = ("buy", avg_fwd, name)
            elif side == "sell" and bias < -bias_thr and s["fs_n"] >= 5:
                avg_fwd = s["fs_sum"] / s["fs_n"]
                if avg_fwd < -fwd_thr:
                    fired = ("sell", abs(avg_fwd), name)

    mem[key] = cp
    return fired


def reservation_price(mid: float, pos: int, sigma: float,
                      gamma: float = 0.1, T_remaining: float = 0.5) -> float:
    """Avellaneda-Stoikov reservation price: skews FV based on inventory.
    r = mid - pos * gamma * sigma^2 * T_remaining
    When long -> reservation < mid (want to sell), when short -> reservation > mid.
    gamma: risk aversion (0.01-1.0), sigma: mid-price volatility."""
    return mid - pos * gamma * (sigma ** 2) * T_remaining

def cholesky_solve(A, b):
    """Solve Ax=b for SPD A via Cholesky. Returns None if singular."""
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v < 1e-10: return None
                L[i][j] = v**0.5
            else:
                L[i][j] = (A[i][j]-s)/L[j][j] if L[j][j]>1e-10 else 0.0
    z = [0.0]*n
    for i in range(n):
        if abs(L[i][i]) < 1e-10: return None
        z[i] = (b[i]-sum(L[i][k]*z[k] for k in range(i)))/L[i][i]
    x = [0.0]*n
    for i in range(n-1,-1,-1):
        if abs(L[i][i]) < 1e-10: return None
        x[i] = (z[i]-sum(L[j][i]*x[j] for j in range(i+1,n)))/L[i][i]
    return x

def fit_ar(prices, p):
    """Fit AR(p) via OLS normal equations. Returns [c0, c1, ..., cp] or None."""
    n = len(prices)
    if n < p + 5: return None
    rows = n - p
    p1 = p + 1
    XtX = [[0.0]*p1 for _ in range(p1)]
    Xty = [0.0]*p1
    for t in range(rows):
        row = [1.0] + [prices[t+p-lag] for lag in range(1,p1)]
        y_t = prices[t+p]
        for j in range(p1):
            Xty[j] += row[j]*y_t
            for k in range(p1):
                XtX[j][k] += row[j]*row[k]
    return cholesky_solve(XtX, Xty)

def predict_ar(coeffs, recent):
    """Predict next price from AR coefficients."""
    if not coeffs or not recent:
        return None
    p = len(coeffs)-1
    if len(recent) < p:
        return None
    pred = coeffs[0]
    for lag in range(1, p+1):
        pred += coeffs[lag] * recent[-lag]
    return float(pred)

# ============================================================
# PHASE-B ENGINES — OBI, layering, markout, fill-probability
# Additive helpers; strategies opt in per-product.
# ============================================================

def compute_obi(order_depth, levels=3):
    """Order-book imbalance over top N levels. Returns float in [-1, +1].
    +1 = pure buy pressure, -1 = pure sell pressure. Cartea/Wang alpha signal."""
    if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
        return 0.0
    bids = sorted(order_depth.buy_orders.keys(), reverse=True)[:levels]
    asks = sorted(order_depth.sell_orders.keys())[:levels]
    bv = sum(abs(order_depth.buy_orders[p]) for p in bids)
    av = sum(abs(order_depth.sell_orders[p]) for p in asks)
    tot = bv + av
    if tot == 0:
        return 0.0
    return (bv - av) / tot


# Gap 1: signal stacking ensemble. Default weights are zero so combine_signals is
# inert when shipped without a fitted JSON. Replace via analysis/signal_stacking_fit.py
# (writes analysis/fingerprints/signal_stack.json — loaded at module import).
_SIGNAL_STACK_WEIGHTS = {
    "garch": 0.0, "kelly": 0.0, "gm": 0.0,
    "obi": 0.0, "vpin": 0.0, "micro_dev": 0.0,
}
try:
    _stack_path = ""
    if False:
        with open(_stack_path) as _fh:
            _loaded = json.load(_fh)
        for _k in _SIGNAL_STACK_WEIGHTS:
            if _k in _loaded:
                _SIGNAL_STACK_WEIGHTS[_k] = float(_loaded[_k])
except Exception:
    pass


def combine_signals(garch=0.0, kelly=0.0, gm=0.0, obi=0.0, vpin=0.0, micro_dev=0.0):
    """Linear combiner for the 6-signal stack — returns FV adjustment (price units).

    All inputs are unit-normalised at the call site; weights are fitted offline on
    P3 data via analysis/signal_stacking_fit.py. Default weights are zero so this
    helper is inert until a fitted JSON is shipped.
    """
    w = _SIGNAL_STACK_WEIGHTS
    return (w["garch"] * garch + w["kelly"] * kelly + w["gm"] * gm
            + w["obi"] * obi + w["vpin"] * vpin + w["micro_dev"] * micro_dev)

def micro_price(order_depth):
    """Stoikov micro-price: (BV*ask + AV*bid)/(BV+AV). Weighted mid biased by
    imbalance. Provably beats raw mid as short-horizon FV anchor."""
    if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    bb = max(order_depth.buy_orders)
    ba = min(order_depth.sell_orders)
    bv = abs(order_depth.buy_orders[bb])
    av = abs(order_depth.sell_orders[ba])
    tot = bv + av
    if tot == 0:
        return (bb + ba) / 2.0
    return (bv * ba + av * bb) / tot

def build_layered_orders(product, side, anchor_price, edge_ticks, size_cap, levels=3):
    """B2: build N-level ladder. side='buy' or 'sell'. anchor is fair value.
    Returns list of Order(product, price, qty) with size split 50/30/20 across levels.
    Reserves 20% of size_cap for aggressive takes (caller handles).
    Qty is signed per IMC convention (positive=buy, negative=sell)."""
    if size_cap <= 0 or levels < 1:
        return []
    splits_by_levels = {1: [1.0], 2: [0.65, 0.35], 3: [0.5, 0.3, 0.2],
                        4: [0.4, 0.3, 0.2, 0.1], 5: [0.35, 0.25, 0.2, 0.12, 0.08]}
    splits = splits_by_levels.get(levels, [1.0/levels]*levels)
    passive_cap = int(size_cap * 0.8)
    out = []
    direction = 1 if side == "buy" else -1
    sign = -1 if side == "buy" else 1  # buy -> price below anchor, sell -> above
    for i, frac in enumerate(splits):
        qty = int(round(passive_cap * frac))
        if qty <= 0:
            continue
        price = int(round(anchor_price + sign * edge_ticks * (i + 1)))
        out.append(Order(product, price, direction * qty))
    return out

def update_markout_buffer(mem, product, own_trades, current_mid):
    """B3: append fills from this tick; evict entries older than 200 ticks.
    Buffer shape: [[ts, fill_price, side(+1 buy / -1 sell), size, mid_at_fill], ...]
    Caller passes current_mid so later markout scoring can compare."""
    key = f"markout_{product}"
    buf = mem.get(key, [])
    if own_trades:
        for t in own_trades:
            ts = getattr(t, "timestamp", 0)
            price = getattr(t, "price", 0)
            qty = getattr(t, "quantity", 0)
            # Convention: quantity sign encodes side in IMC Trade object — positive
            # when bot was buyer, negative when seller (match_orders semantics).
            if qty == 0:
                continue
            side = 1 if qty > 0 else -1
            buf.append([int(ts), float(price), int(side), int(abs(qty)), float(current_mid or price)])
    # Trim to last 200 fills to bound memory (whitelisted via A5 prefix match).
    if len(buf) > 200:
        buf = buf[-200:]
    mem[key] = buf

def markout_score(mem, product, current_mid, horizon_ticks=10000):
    """B3: avg PnL per unit if we paper-closed each fill at current_mid.
    Negative = adverse selection (we were picked off). Only considers fills
    older than a small grace window so we measure post-fill drift."""
    key = f"markout_{product}"
    buf = mem.get(key, [])
    if not buf or current_mid is None:
        return 0.0
    total_pnl = 0.0
    total_size = 0
    for entry in buf:
        if len(entry) < 5:
            continue
        _, price, side, size, _ = entry
        total_pnl += side * (current_mid - price) * size
        total_size += size
    if total_size == 0:
        return 0.0
    return total_pnl / total_size

def fillprob_update(mem, product, placed_orders, own_trades):
    """B5: per-offset fill-rate calibration. placed_orders = list of (price, qty, bbo_bid_or_ask)
    Actually we store just {offset_from_bbo: [attempts, fills]}. Caller must pass tuples
    of (offset_bucket, filled_bool) for each placed order this tick."""
    key = f"fillprob_{product}"
    prob = mem.get(key, {})
    # placed_orders here is a pre-computed list of (bucket_str, filled_bool)
    for bucket, filled in placed_orders:
        k = str(bucket)
        if k not in prob:
            prob[k] = [0, 0]
        prob[k][0] += 1
        if filled:
            prob[k][1] += 1
    mem[key] = prob

def fillprob_rate(mem, product, offset_bucket):
    """Query fill rate for a specific offset bucket. Returns None if <20 attempts."""
    key = f"fillprob_{product}"
    prob = mem.get(key, {})
    entry = prob.get(str(offset_bucket))
    if not entry or entry[0] < 20:
        return None
    return entry[1] / entry[0]

# ============================================================
# PHASE-C HELPERS — novel alpha (VPIN, SPRT, delay, ensemble FV)
# Additive; strategies opt in per-product via cfg flags.
# ============================================================

def precompute_taker_sentiment(state, mem):
    """C1: before strategies dispatch, aggregate market_trades sentiment per product.
    Every other team only reads own_trades; we also front-run via market_trades.
    Writes mem[f"tick_taker_sentiment_{product}"] = net_signed_volume over this tick."""
    if not state.market_trades:
        return
    for product, trades in state.market_trades.items():
        if not trades:
            continue
        net = 0
        for t in trades:
            qty = getattr(t, "quantity", 0)
            price = getattr(t, "price", 0)
            od = (state.order_depths or {}).get(product)
            if not od or not od.buy_orders or not od.sell_orders:
                continue
            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
            sign = 1 if price > mid else (-1 if price < mid else 0)
            net += sign * abs(qty)
        mem[f"tick_taker_sentiment_{product}"] = net

def ensemble_fv(product, state, ps):
    """C3: trimmed-mean FV across 5 estimators (drops hi+lo, averages middle 3).
    Returns None if insufficient data. Bad-tick immune."""
    od = (state.order_depths or {}).get(product)
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    estimators = [(best_bid + best_ask) / 2.0]
    if ps and ps.wall_mid:
        estimators.append(ps.wall_mid)
    mp = micro_price(od)
    if mp is not None:
        estimators.append(mp)
    if ps and ps.kf_x:
        estimators.append(ps.kf_x)
    if ps and hasattr(ps, "mid_hist") and len(ps.mid_hist) >= 10:
        # VWAP-ish: EMA on mid_hist last 10
        vw = sum(ps.mid_hist[-10:]) / 10.0
        estimators.append(vw)
    if len(estimators) < 3:
        return sum(estimators) / len(estimators) if estimators else None
    # Trim extremes, mean middle
    estimators.sort()
    middle = estimators[1:-1] if len(estimators) >= 5 else estimators
    return sum(middle) / len(middle)

def vpin_update(mem, product, own_trades, market_trades, bucket_size=100):
    """C9: bucket-VPIN approximation. VPIN = |buy_vol - sell_vol| / total_vol
    over rolling 50 volume-buckets. High VPIN = toxic flow; caller withdraws.
    We stream signed volume into the current bucket and close it at bucket_size."""
    key = f"_vpin_{product}"
    state = mem.get(key, {"cur_buy": 0, "cur_sell": 0, "buckets": []})
    # Aggregate this tick's signed volume (classify via price vs mid).
    for src in (own_trades or []), (market_trades or []):
        for t in src:
            qty = abs(getattr(t, "quantity", 0))
            if qty == 0:
                continue
            # Classify: from own_trades, IMC uses signed qty; from market_trades we
            # guess via price. Here we just alternate: use timestamp parity as a
            # tiebreaker fallback (rare edge case).
            price = getattr(t, "price", 0)
            buyer = getattr(t, "buyer", "") or ""
            seller = getattr(t, "seller", "") or ""
            if buyer == "SUBMISSION":
                state["cur_buy"] += qty
            elif seller == "SUBMISSION":
                state["cur_sell"] += qty
            else:
                # market_trade: use price vs nothing — assume half each
                state["cur_buy"] += qty / 2.0
                state["cur_sell"] += qty / 2.0
    # Close bucket when full
    while state["cur_buy"] + state["cur_sell"] >= bucket_size:
        over = state["cur_buy"] + state["cur_sell"] - bucket_size
        scale = bucket_size / (state["cur_buy"] + state["cur_sell"])
        b = state["cur_buy"] * scale
        s = state["cur_sell"] * scale
        state["buckets"].append([b, s])
        # Roll overflow
        scale_out = over / (state["cur_buy"] + state["cur_sell"])
        state["cur_buy"] *= scale_out
        state["cur_sell"] *= scale_out
        if over <= 1e-6:
            state["cur_buy"] = 0.0
            state["cur_sell"] = 0.0
    # Trim to last 50 buckets (rolling window)
    if len(state["buckets"]) > 50:
        state["buckets"] = state["buckets"][-50:]
    mem[key] = state

def vpin_score(mem, product):
    """C9: query current VPIN. Returns value in [0,1] or None if insufficient."""
    state = mem.get(f"_vpin_{product}")
    if not state or len(state["buckets"]) < 10:
        return None
    total_imb = 0.0
    total_vol = 0.0
    for b, s in state["buckets"]:
        total_imb += abs(b - s)
        total_vol += b + s
    if total_vol == 0:
        return None
    return total_imb / total_vol

def vpin_forecast(mem, product):
    """H3: AR(1) forecast of next-bucket VPIN. Returns predicted VPIN in [0,1] or None.
    Toxicity is persistent — predicting one bucket ahead lets us widen quotes BEFORE
    the bad fills land, not after. Strict additive helper; consumers gate on cfg flag."""
    state = mem.get(f"_vpin_{product}")
    if not state or len(state["buckets"]) < 12:
        return None
    series = []
    for b, s in state["buckets"]:
        tot = b + s
        series.append(abs(b - s) / tot if tot > 0 else 0.0)
    n = len(series) - 1
    if n < 10:
        return None
    mx = sum(series[:-1]) / n
    my = sum(series[1:]) / n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = series[i] - mx
        num += dx * (series[i + 1] - my)
        den += dx * dx
    if den < 1e-9:
        return series[-1]
    b1 = num / den
    a1 = my - b1 * mx
    pred = a1 + b1 * series[-1]
    return max(0.0, min(1.0, pred))

def adversary_flow_update(mem, product, market_trades, mid):
    """H11: stream signed flow per product for next-tick prediction.
    Sign = +qty if taker bought (price >= mid), -qty if sold. Used by
    adversary_predict_drift() to forecast next-tick mid shift from clustering."""
    key = f"_advflow_{product}"
    h = mem.get(key, [])
    for t in (market_trades or []):
        qty = int(getattr(t, "quantity", 0))
        px = float(getattr(t, "price", 0))
        if qty <= 0:
            continue
        sign = 1 if px >= mid else -1
        h.append(sign * qty)
    if len(h) > 80:
        h = h[-80:]
    mem[key] = h

def adversary_predict_drift(mem, product, lookback=30, decay=0.9):
    """H11: forecast next-tick mid drift from exponentially-weighted recent flow.
    Returns predicted price ticks (signed). Caller uses sign+magnitude to skew quotes
    one tick ahead of the predicted move — captures the cross-tick spread before
    the slow taker bots arrive. Returns 0.0 if insufficient data."""
    h = mem.get(f"_advflow_{product}", [])
    if len(h) < 8:
        return 0.0
    use = h[-lookback:]
    w = 1.0
    num = 0.0
    den = 0.0
    for v in reversed(use):
        num += w * v
        den += w * abs(v) if v else w
        w *= decay
    if den < 1e-6:
        return 0.0
    # Empirical scaling: mean signed flow / 50 ~= price-tick drift
    return (num / max(den, 1.0)) * 1.0

def hurst_dfa(series, min_box=4, max_box=None):
    """I1: Detrended Fluctuation Analysis estimate of Hurst exponent.
    H ~ 0.5 = random walk, H > 0.5 = trending/persistent, H < 0.5 = mean-reverting.
    Caller switches MM mode (passive vs taker) based on which half H lives in.
    Returns H in [0,1] or None if insufficient data."""
    n = len(series)
    if n < 32:
        return None
    if max_box is None:
        max_box = n // 4
    # cumulative integrated profile (subtract mean first)
    mean = sum(series) / n
    Y = []
    cum = 0.0
    for x in series:
        cum += (x - mean)
        Y.append(cum)
    log_box = []
    log_F = []
    s = min_box
    while s <= max_box:
        n_seg = n // s
        if n_seg < 2:
            break
        sse = 0.0
        for k in range(n_seg):
            seg = Y[k * s:(k + 1) * s]
            # OLS detrend
            xs = list(range(s))
            mx = sum(xs) / s
            my = sum(seg) / s
            num = sum((xs[i] - mx) * (seg[i] - my) for i in range(s))
            den = sum((xs[i] - mx) ** 2 for i in range(s))
            slope = num / den if abs(den) > 1e-12 else 0.0
            intercept = my - slope * mx
            for i in range(s):
                r = seg[i] - (intercept + slope * xs[i])
                sse += r * r
        F = (sse / (n_seg * s)) ** 0.5
        if F > 0:
            log_box.append(math.log(s))
            log_F.append(math.log(F))
        s = max(s + 1, int(s * 1.4))
    if len(log_box) < 4:
        return None
    # OLS slope of log_F vs log_box = Hurst exponent
    mx = sum(log_box) / len(log_box)
    my = sum(log_F) / len(log_F)
    num = sum((log_box[i] - mx) * (log_F[i] - my) for i in range(len(log_box)))
    den = sum((log_box[i] - mx) ** 2 for i in range(len(log_box)))
    if abs(den) < 1e-12:
        return None
    H = num / den
    return max(0.0, min(1.0, H))

def garch_update(mem, key, ret, omega=None, alpha=0.1, beta=0.85, init_var=1.0):
    """I2: GARCH(1,1) variance forecast: sigma^2_t = omega + alpha*ret^2 + beta*sigma^2_{t-1}.
    Constraint: alpha+beta < 1 for stationarity. omega defaults to (1-alpha-beta)*init_var.
    Returns next-period sigma forecast (sqrt of variance). Captures vol clustering and
    leverage that rolling stddev under-reacts to."""
    s = mem.get(key)
    if s is None:
        s = {"var": float(init_var)}
    if omega is None:
        omega = max(1e-9, (1.0 - alpha - beta) * init_var)
    s["var"] = omega + alpha * (ret * ret) + beta * s["var"]
    mem[key] = s
    return s["var"] ** 0.5

def roll_spread(prices):
    """I3: Roll's (1984) effective half-spread from trade-price autocovariance.
    s = 2*sqrt(-cov(dp_t, dp_{t-1})) when covariance is negative (bid-ask bounce).
    Returns half-spread in price units, or None if cov >= 0 (no bid-ask bounce regime)."""
    n = len(prices)
    if n < 8:
        return None
    dp = [prices[i] - prices[i - 1] for i in range(1, n)]
    m = len(dp) - 1
    if m < 4:
        return None
    mean_dp = sum(dp) / len(dp)
    cov = 0.0
    for i in range(m):
        cov += (dp[i] - mean_dp) * (dp[i + 1] - mean_dp)
    cov /= m
    if cov >= -1e-9:
        return None
    return (-cov) ** 0.5

def kyle_lambda(rets, signed_vols):
    """I4: Kyle's (1985) price-impact lambda from OLS of return on signed volume.
    return_t = lambda * signed_volume_t + noise. Returns lambda (price-ticks per unit).
    Caller scales position sizing inversely to lambda — high impact = trade smaller."""
    n = min(len(rets), len(signed_vols))
    if n < 10:
        return None
    x = signed_vols[-n:]
    y = rets[-n:]
    mx = sum(x) / n
    my = sum(y) / n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = x[i] - mx
        num += dx * (y[i] - my)
        den += dx * dx
    if den < 1e-9:
        return None
    return num / den

def mutual_info_binned(x, y, bins=8):
    """I5: estimator of MI(X;Y) via equal-frequency binning (no numpy).
    Catches nonlinear lead-lag that lagged_corr (H8) misses. Returns nats; >0.05 ~= signal.
    O(n log n) per call due to sorting; intended for offline runs or low-frequency online."""
    n = min(len(x), len(y))
    if n < 30:
        return None
    sx = sorted(x[-n:])
    sy = sorted(y[-n:])
    # equal-freq bin edges
    edges_x = [sx[int(i * n / bins)] for i in range(1, bins)]
    edges_y = [sy[int(i * n / bins)] for i in range(1, bins)]
    def binid(v, edges):
        lo, hi = 0, len(edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if v <= edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo
    joint = {}
    px = [0] * bins
    py = [0] * bins
    for i in range(n):
        bx = binid(x[-n + i], edges_x)
        by = binid(y[-n + i], edges_y)
        joint[(bx, by)] = joint.get((bx, by), 0) + 1
        px[bx] += 1
        py[by] += 1
    mi = 0.0
    for (bx, by), c in joint.items():
        if c == 0 or px[bx] == 0 or py[by] == 0:
            continue
        pxy = c / n
        mi += pxy * math.log(pxy * n / (px[bx] * py[by]) + 1e-12)
    return max(0.0, mi)

def cusum_update(mem, key, x, target=0.0, k=0.5, h=5.0):
    """I6: Page's CUSUM changepoint. Streams x; returns ('UP' | 'DOWN' | None).
    h_pos = max(0, h_pos + (x - target - k));  h_neg = min(0, h_neg + (x - target + k)).
    Trigger when |h_pos| >= h or |h_neg| >= h. Lighter than BOCPD — runs everywhere."""
    s = mem.get(key, {"hp": 0.0, "hn": 0.0})
    s["hp"] = max(0.0, s["hp"] + (x - target - k))
    s["hn"] = min(0.0, s["hn"] + (x - target + k))
    sig = None
    if s["hp"] >= h:
        sig = "UP"
        s["hp"] = 0.0
    elif -s["hn"] >= h:
        sig = "DOWN"
        s["hn"] = 0.0
    mem[key] = s
    return sig

def almgren_chriss_unwind(pos, ticks_left, sigma, eta=0.01, gamma=1e-4, lam=1e-5):
    """I7: closed-form Almgren-Chriss optimal liquidation slice for THIS tick.
    Decay rate kappa = sqrt(lam*sigma^2 / eta); slice_t = pos * sinh(kappa*(T - t)) /
    sinh(kappa*T) — but we just need next-tick slice ratio.
    Returns int qty to trade THIS tick (signed; positive = sell long, negative = cover short)."""
    if ticks_left <= 0 or pos == 0:
        return -pos  # final tick: dump remainder
    if ticks_left == 1:
        return -pos
    if sigma is None or sigma <= 0:
        # No vol info: linear schedule
        return -int(round(pos / ticks_left))
    kappa = (max(lam, 1e-12) * sigma * sigma / max(eta, 1e-12)) ** 0.5
    # slice_t = pos * (cosh(kappa*(T-t)) - cosh(kappa*(T-t-1))) / sinh(kappa*T)... approx:
    # next-tick reduction ratio:
    T = float(ticks_left)
    try:
        ch_full = math.cosh(kappa * T)
        ch_next = math.cosh(kappa * (T - 1))
        sh_full = math.sinh(kappa * T)
        if abs(sh_full) < 1e-9:
            return -int(round(pos / ticks_left))
        reduce_ratio = (ch_full - ch_next) / sh_full
    except OverflowError:
        return -pos
    qty = int(round(pos * reduce_ratio))
    # Always trade at least 1 if pos != 0 to make progress
    if qty == 0 and pos != 0:
        qty = 1 if pos > 0 else -1
    return -qty if pos > 0 else qty

def thompson_sample_update(mem, key, arm, reward):
    """I8: Beta-Bernoulli Thompson sampling — record reward (in [0,1]) for chosen arm."""
    s = mem.get(key, {})
    a = s.get(arm, {"a": 1.0, "b": 1.0})
    r = max(0.0, min(1.0, float(reward)))
    a["a"] += r
    a["b"] += (1.0 - r)
    s[arm] = a
    mem[key] = s

def thompson_sample_pick(mem, key, arms, rng):
    """I8: sample posterior mean for each arm, return argmax. rng is a deterministic
    pseudo-random source: pass a Random instance or accept (lambda: deterministic float).
    For competition use: build rng from hash(state.timestamp) so picks are reproducible."""
    s = mem.get(key, {})
    best = None
    best_v = -1.0
    for arm in arms:
        a = s.get(arm, {"a": 1.0, "b": 1.0})
        # Beta sample via inverse-CDF approx not stdlib — use Marsaglia trick:
        # X = Gamma(a,1), Y = Gamma(b,1), Beta = X/(X+Y).
        # Stdlib has random.betavariate via random module — but `rng` may be a callable.
        if hasattr(rng, "betavariate"):
            v = rng.betavariate(a["a"], a["b"])
        else:
            # Posterior mean fallback (no exploration)
            v = a["a"] / (a["a"] + a["b"])
        if v > best_v:
            best_v = v
            best = arm
    return best if best is not None else arms[0]

def kelly_fraction(edge, variance, cap=0.25):
    """I9: Kelly bet fraction f* = edge / variance (continuous version), capped to avoid
    blowup on noisy edge estimates. Returns f in [-cap, cap]."""
    if variance is None or variance <= 0:
        return 0.0
    f = edge / variance
    if f > cap:
        return cap
    if f < -cap:
        return -cap
    return f

def kl_divergence(p, q, eps=1e-9):
    """I10: discrete KL(p || q) over equal-length probability lists.
    Caller provides histograms; sums normalized inside. Returns nats."""
    n = min(len(p), len(q))
    if n == 0:
        return 0.0
    sp = sum(p[:n]) or 1.0
    sq = sum(q[:n]) or 1.0
    out = 0.0
    for i in range(n):
        pi = p[i] / sp + eps
        qi = q[i] / sq + eps
        out += pi * math.log(pi / qi)
    return out

def glosten_milgrom_halfspread(p_informed, sigma_value, mid):
    """J1: Glosten-Milgrom (1985) bid-ask half-spread under adverse selection.
    s/2 = p_informed * sigma_value (one-tick value uncertainty)
    Caller widens passive quotes by this amount when toxic-flow detected."""
    return max(0.0, p_informed * sigma_value)

def haar_decompose(series, levels=3):
    """J2: Haar wavelet decomposition. Returns (approx, [details_l1, details_l2, ...]).
    approx[k] = (s[2k] + s[2k+1])/sqrt(2);  detail[k] = (s[2k] - s[2k+1])/sqrt(2).
    Levels iteratively decompose the approx. Use detail energy at each scale to detect
    micro-noise vs meso-trend."""
    sqrt2 = 2.0 ** 0.5
    cur = list(series)
    details = []
    for _ in range(levels):
        n = len(cur) // 2
        if n < 2:
            break
        a = [(cur[2 * i] + cur[2 * i + 1]) / sqrt2 for i in range(n)]
        d = [(cur[2 * i] - cur[2 * i + 1]) / sqrt2 for i in range(n)]
        details.append(d)
        cur = a
    return cur, details

def pid_step(mem, key, error, kp=0.5, ki=0.05, kd=0.1, dt=1.0, i_clip=20.0):
    """J3: PID controller step. Returns control output for THIS tick.
    Use case: error = -inventory (target=0); output = price-tick offset to shift quotes.
    Anti-windup: integral term clipped to +-i_clip."""
    s = mem.get(key, {"i": 0.0, "prev": 0.0})
    i_term = s["i"] + error * dt
    if i_term > i_clip:
        i_term = i_clip
    elif i_term < -i_clip:
        i_term = -i_clip
    d_term = (error - s["prev"]) / dt if dt > 0 else 0.0
    out = kp * error + ki * i_term + kd * d_term
    s["i"] = i_term
    s["prev"] = error
    mem[key] = s
    return out

def ou_fit(series):
    """J4: Ornstein-Uhlenbeck process fit via OLS on dx = theta*(mu - x)*dt + sigma*dW.
    Discretize: x_{t+1} - x_t = theta*(mu - x_t) + eps. OLS on (x_t, x_{t+1}).
    Returns dict {theta, mu, sigma, half_life} or None."""
    n = len(series)
    if n < 30:
        return None
    x = series[:-1]
    y = [series[i + 1] - series[i] for i in range(n - 1)]
    m = len(x)
    sx = sum(x); sy = sum(y); sxx = sum(v * v for v in x); sxy = sum(x[i] * y[i] for i in range(m))
    det = m * sxx - sx * sx
    if abs(det) < 1e-9:
        return None
    # y = a + b*x;  b = -theta;  a = theta*mu
    b = (m * sxy - sx * sy) / det
    a = (sy - b * sx) / m
    theta = -b
    if theta <= 0 or abs(theta) > 5:
        return None
    mu = a / theta if theta != 0 else 0.0
    # Residual std for sigma estimate
    resid_sq = 0.0
    for i in range(m):
        e = y[i] - (a + b * x[i])
        resid_sq += e * e
    sigma = (resid_sq / max(m - 2, 1)) ** 0.5
    half_life = math.log(2.0) / theta if theta > 0 else float("inf")
    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life": half_life}

def permutation_entropy(series, m=3):
    """J5: Bandt-Pompe permutation entropy — model-free predictability measure in [0,1].
    Low = highly ordered/predictable; high = random. Use as a regime tag orthogonal to Hurst."""
    n = len(series)
    if n < m + 5:
        return None
    counts = {}
    for i in range(n - m + 1):
        window = series[i:i + m]
        # rank pattern (argsort)
        idx = sorted(range(m), key=lambda j: window[j])
        counts[tuple(idx)] = counts.get(tuple(idx), 0) + 1
    total = sum(counts.values())
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log(p)
    Hmax = math.log(math.factorial(m))
    return H / Hmax if Hmax > 0 else 0.0

def pa_classifier_update(mem, key, features, label, C=0.1):
    """J6: Crammer et al. Passive-Aggressive online linear classifier (PA-I).
    label in {-1, +1}; features a list. Updates weights; returns post-update margin.
    Rule: margin = label * w.x; loss = max(0, 1 - margin); tau = min(C, loss/||x||^2)."""
    n = len(features)
    s = mem.get(key, {"w": [0.0] * n, "b": 0.0})
    if len(s["w"]) != n:
        s["w"] = [0.0] * n
        s["b"] = 0.0
    margin = sum(s["w"][i] * features[i] for i in range(n)) + s["b"]
    margin *= label
    loss = max(0.0, 1.0 - margin)
    if loss > 0:
        norm_sq = sum(v * v for v in features) + 1.0
        tau = min(C, loss / norm_sq) if norm_sq > 0 else 0.0
        for i in range(n):
            s["w"][i] += tau * label * features[i]
        s["b"] += tau * label
    mem[key] = s
    return margin

def pa_classifier_predict(mem, key, features):
    """J6: PA classifier prediction. Returns score (sign = predicted label, magnitude = confidence)."""
    s = mem.get(key)
    if not s:
        return 0.0
    n = len(features)
    if len(s["w"]) != n:
        return 0.0
    return sum(s["w"][i] * features[i] for i in range(n)) + s["b"]

def book_shape_metrics(od, depth=5):
    """L1: extract structural features from order book.
    Returns dict {asymmetry, taper_bid, taper_ask, depth_total, top_share}.
    asymmetry = (bid_total - ask_total) / (bid_total + ask_total) over top `depth` levels.
    taper = ratio of L1 size to (L1+L2+L3) size — high taper = thin behind front line.
    top_share = top-of-book volume / total displayed volume."""
    if od is None or not od.buy_orders or not od.sell_orders:
        return None
    bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])[:depth]
    asks = sorted(od.sell_orders.items(), key=lambda x: x[0])[:depth]
    bv = sum(abs(s) for _, s in bids)
    av = sum(abs(s) for _, s in asks)
    tot = bv + av
    if tot == 0:
        return None
    asym = (bv - av) / tot
    bid_top = abs(bids[0][1]) if bids else 0
    ask_top = abs(asks[0][1]) if asks else 0
    taper_bid = bid_top / bv if bv > 0 else 1.0
    taper_ask = ask_top / av if av > 0 else 1.0
    top_share = (bid_top + ask_top) / tot
    return {"asymmetry": asym, "taper_bid": taper_bid,
            "taper_ask": taper_ask, "depth_total": tot, "top_share": top_share}

def price_improvement(own_trades, mem, key):
    """L4: tracks our trade-fill prices vs the BBO at submission tick.
    mem[key] = list of (improvement, fill_price). Caller seeds via storing
    quote BBO at submit; this reads back. Returns rolling avg improvement."""
    h = mem.get(key, [])
    for t in (own_trades or []):
        h.append(0.0)  # caller fills in actual improvement post-hoc
    if len(h) > 50:
        h = h[-50:]
    mem[key] = h
    return sum(h) / len(h) if h else 0.0

def huber_fit_ar(series, p=2, delta=1.345, iters=5):
    """L9: robust AR(p) fit using Huber loss (downweights outliers > delta*sigma).
    Iteratively-reweighted least squares (IRLS). Returns coefficient list of length p
    or None if data too short. Drop-in replacement for fit_ar() when outliers are an issue."""
    n = len(series)
    if n < 3 * p + 5:
        return None
    # Build design matrix
    rows = []
    targets = []
    for t in range(p, n):
        rows.append(series[t - p:t])
        targets.append(series[t])
    m = len(rows)
    # Initial OLS
    coefs = [0.0] * p
    weights = [1.0] * m
    for _ in range(iters):
        # Weighted normal equations: A^T W A coefs = A^T W y
        ATA = [[0.0] * p for _ in range(p)]
        ATy = [0.0] * p
        for i in range(m):
            w = weights[i]
            r = rows[i]
            ti = targets[i]
            for a in range(p):
                ATy[a] += w * r[a] * ti
                for b in range(p):
                    ATA[a][b] += w * r[a] * r[b]
        try:
            coefs = cholesky_solve(ATA, ATy)
        except Exception:
            return None
        # Recompute residuals + Huber weights
        resid = []
        for i in range(m):
            pred = sum(coefs[a] * rows[i][a] for a in range(p))
            resid.append(targets[i] - pred)
        # Robust scale: 1.4826 * MAD
        abs_r = sorted(abs(r) for r in resid)
        mad = abs_r[len(abs_r) // 2]
        scale = 1.4826 * mad if mad > 1e-9 else 1.0
        thr = delta * scale
        for i in range(m):
            ar = abs(resid[i])
            weights[i] = 1.0 if ar <= thr else thr / ar
    return coefs

def as_inventory_skew(pos, sigma, gamma=0.05, T_remaining=1.0, max_skew=3.0):
    """H9: Avellaneda-Stoikov reservation-price shift from inventory.
    Returns price offset to SUBTRACT from mid: r = mid - q*gamma*sigma^2*T.
    Long inventory -> negative output -> r < mid -> ask hit faster, bid slower.
    Capped at +-max_skew ticks so it never violates basic spread sanity."""
    if sigma is None or sigma <= 0:
        return 0.0
    raw = float(pos) * gamma * (sigma ** 2) * T_remaining
    if raw > max_skew:
        return max_skew
    if raw < -max_skew:
        return -max_skew
    return raw

def as_spread_half(sigma, gamma=0.05, T_remaining=1.0, k=1.5):
    """H9: A-S optimal half-spread component from quoting model.
    delta_half = 0.5*gamma*sigma^2*T + (1/gamma)*ln(1 + gamma/k).
    Caller picks max(this, min_spread) so passive book never crosses."""
    if sigma is None or sigma <= 0 or k <= 0:
        return 0.0
    return 0.5 * gamma * (sigma ** 2) * T_remaining + (1.0 / gamma) * math.log(1.0 + gamma / k)

def xprod_buffer_update(mem, product, mid, cap=300):
    """H8: rolling mid history for cross-product lead-lag analysis."""
    key = f"_xprod_{product}"
    h = mem.get(key, [])
    h.append(float(mid))
    if len(h) > cap:
        h = h[-cap:]
    mem[key] = h

def lagged_corr(mem, lead, follow, lag=1, n=200):
    """H8: corr(lead_return at t-lag, follow_return at t) over last n ticks.
    Positive value with lag>0 means lead leads follow — caller can use lead's
    latest return as a predictor of follow's next return. Returns None if data short."""
    a = mem.get(f"_xprod_{lead}", [])
    b = mem.get(f"_xprod_{follow}", [])
    if len(a) < n + lag + 2 or len(b) < n + lag + 2:
        return None
    a = a[-(n + lag + 1):]
    b = b[-(n + lag + 1):]
    ra = [a[i] - a[i - 1] for i in range(1, len(a))]
    rb = [b[i] - b[i - 1] for i in range(1, len(b))]
    if len(ra) <= lag or len(rb) <= lag:
        return None
    x = ra[: len(ra) - lag]
    y = rb[lag:]
    m = min(len(x), len(y))
    x = x[-m:]; y = y[-m:]
    mx = sum(x) / m; my = sum(y) / m
    num = sum((x[i] - mx) * (y[i] - my) for i in range(m))
    dx = sum((x[i] - mx) ** 2 for i in range(m))
    dy = sum((y[i] - my) ** 2 for i in range(m))
    den = (dx * dy) ** 0.5
    return (num / den) if den > 1e-9 else None

def phantom_observe(mem, product, market_trades):
    """H7: record which price levels have actually traded. Caller filters book
    to drop phantom (never-traded) levels before computing FV. Cheap streaming op.
    Distinct prices are bounded — no need for a cap unless we see >1K levels."""
    s = mem.get(f"_phantom_{product}")
    if s is None:
        s = {"traded": {}, "ticks": 0}
    for t in (market_trades or []):
        p = int(getattr(t, "price", 0))
        s["traded"][str(p)] = s["traded"].get(str(p), 0) + int(getattr(t, "quantity", 0))
    s["ticks"] = s.get("ticks", 0) + 1
    mem[f"_phantom_{product}"] = s

def phantom_filter_levels(mem, product, levels, min_volume=1, warmup=200):
    """H7: levels = dict {price: size}. Returns dict with phantom prices removed.
    During warmup returns input unchanged (insufficient evidence)."""
    s = mem.get(f"_phantom_{product}")
    if not s or s.get("ticks", 0) < warmup:
        return levels
    traded = s.get("traded", {})
    out = {}
    for px, sz in levels.items():
        if traded.get(str(int(px)), 0) >= min_volume:
            out[px] = sz
    return out if out else levels

def kalman_spread_update(mem, key, obs, Q=0.5, R=8.0):
    """H5: 1D Kalman on a noisy spread/premium series. Returns (filtered, z).
    z = (obs - filtered_predict) / sqrt(P + R) — innovation in sigma units, robust
    to regime drift (no rolling-window var that explodes on jumps)."""
    s = mem.get(key)
    if not s:
        s = {"x": float(obs), "P": float(R)}
    x_pred = s["x"]
    P_pred = s["P"] + Q
    innov = float(obs) - x_pred
    S = P_pred + R
    z = innov / (S ** 0.5) if S > 0 else 0.0
    K = P_pred / S if S > 0 else 0.0
    s["x"] = x_pred + K * innov
    s["P"] = (1.0 - K) * P_pred
    mem[key] = s
    return s["x"], z

def hawkes_update(mem, product, n_events, alpha=0.6, decay=0.92):
    """H4: Hawkes-lite self-exciting intensity for MO arrivals.
    intensity_t = decay*intensity_{t-1} + alpha*n_events_t.
    n_events = count of taker prints this tick. Caller widens quotes when
    intensity exceeds threshold — adverse-selection risk is rising before fills land."""
    key = f"_hawkes_{product}"
    cur = mem.get(key, 0.0) * decay + alpha * float(n_events)
    mem[key] = cur
    return cur

def hawkes_intensity(mem, product):
    """H4: query current MO intensity. Returns 0.0 if uninitialized."""
    return mem.get(f"_hawkes_{product}", 0.0)

def sprt_update(mem, key, obs, h0_prob, h1_prob, thr=3.0):
    """C11: Sequential Probability Ratio Test. obs=bool observation;
    h0_prob, h1_prob = P(obs | null), P(obs | alt). Returns 'H0', 'H1', or 'UNKNOWN'.
    Lower-weight alternative to full BOCPD when decision is binary (e.g., 'is Olivia present')."""
    mkey = f"_sprt_{key}"
    cur = mem.get(mkey, 0.0)
    if obs:
        ratio = h1_prob / max(h0_prob, 1e-6)
    else:
        ratio = (1 - h1_prob) / max(1 - h0_prob, 1e-6)
    cur += math.log(max(ratio, 1e-6))
    # Clamp to avoid runaway drift after decision
    cur = max(-thr - 1, min(thr + 1, cur))
    mem[mkey] = cur
    if cur >= thr:
        return "H1"
    if cur <= -thr:
        return "H0"
    return "UNKNOWN"

def should_delay_requote(order_depth, our_side_price, our_side, fill_prob_thr=0.7):
    """C10: Lehalle & Mounjid — high fill probability ≡ high adverse selection.
    If our quote is at or inside BBO with favorable imbalance, return True to signal
    caller to HOLD (not re-quote) for 1 tick. Simple heuristic without Markov math."""
    if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
        return False
    obi = compute_obi(order_depth)
    if our_side == "buy":
        # buy fills when asks thin + more buyers = OBI > 0 means we fill fast
        return obi > fill_prob_thr - 0.3
    elif our_side == "sell":
        return obi < -(fill_prob_thr - 0.3)
    return False

# ============================================================
# FRONTIER ENGINE — Self-activating institutional techniques
# Each function checks its own activation criteria and returns
# a neutral value (no effect) when insufficient data or no signal.
# ============================================================

def frontier_fv_weights(ps):
    """Online signal weight learning (RenTech EWLS approach).
    Solves weights = (X'X)^-1 * X'y from running statistics.
    Returns learned [w_mp, w_wmid, w_ar] or None if insufficient data.
    Self-activates after 50 ticks of warmup."""
    if ps is None or ps.fr_sig_n < 50 or ps.fr_sig_xtx is None:
        return None
    # Solve 3x3 system: A*w = b where A=X'X, b=X'y
    A = ps.fr_sig_xtx
    b = ps.fr_sig_xty
    # Cramer's rule for 3x3
    def det3(m):
        return (m[0]*(m[4]*m[8]-m[5]*m[7]) -
                m[1]*(m[3]*m[8]-m[5]*m[6]) +
                m[2]*(m[3]*m[7]-m[4]*m[6]))
    D = det3(A)
    if abs(D) < 1e-10:
        return None
    w = []
    for col in range(3):
        M = list(A)
        for row in range(3):
            M[row * 3 + col] = b[row]
        w.append(det3(M) / D)
    # Sanity: weights should be non-negative and sum to ~1
    total = sum(abs(x) for x in w)
    if total < 0.1 or total > 5.0:
        return None
    # Normalize to sum to 1
    s = sum(w)
    if s > 0.1:
        w = [x / s for x in w]
    else:
        return None
    # Clamp individual weights to [0.05, 0.8]
    w = [max(0.05, min(0.8, x)) for x in w]
    s = sum(w)
    w = [x / s for x in w]
    return w


def frontier_trade_momentum(ps):
    """Trade sequence momentum (HRT approach).
    Returns FV adjustment based on consecutive same-direction trades.
    Self-activates when run_length >= 4."""
    if ps is None or ps.fr_run_len < 4:
        return 0.0
    # Capped contribution: max ±1.5 ticks
    strength = min(ps.fr_run_len - 3, 3) * 0.3
    return ps.fr_run_dir * strength


def frontier_adaptive_offset(ps, base_offset):
    """Fill-rate adaptive spread (HRT approach).
    Uses gamma_fill_ema (already tracked) to auto-tune make_offset.
    Self-activates after 100 ticks."""
    if ps is None or ps.tick_count < 100:
        return base_offset
    fill_rate = ps.gamma_fill_ema
    # Target: 15-25% fill rate per tick
    if fill_rate > 0.30:
        return min(base_offset + 2, base_offset * 2)  # widen: adverse selection
    elif fill_rate > 0.20:
        return base_offset + 1
    elif fill_rate < 0.05 and base_offset > 1:
        return max(1, base_offset - 1)  # tighten: missing volume
    return base_offset


def frontier_kyle_gate(ps, edge):
    """Kyle's lambda adverse selection gate (institutional).
    Returns True if edge is sufficient to overcome price impact.
    Self-activates after 200 ticks when lambda is estimated."""
    if ps is None or ps.fr_kyle_lambda <= 0 or ps.tick_count < 200:
        return True  # no gate when insufficient data
    # Expected fill qty ~= make_size. Edge must exceed lambda * qty.
    min_edge = ps.fr_kyle_lambda * 5  # assume ~5 lot fill
    return abs(edge) > min_edge


# ============================================================
# STRATEGY: PEGGED FV (RESIN / EMERALDS)
# ============================================================

def run_pegged(product, cfg, od, pos, mem, market_trades, own_trades, ps=None, ts=0):
    """Frankfurt StaticTrader pattern: sweep ALL mispriced, then overbid/undercut.
    Uses wall_mid as FV anchor (more robust than fixed FV for discovery).
    Supports fv_drift: linear FV drift per tick (for trending pegged products)."""
    fv = cfg["fair_value"]
    limit = cfg["position_limit"]
    orders = []
    buy_cap = limit - pos
    sell_cap = limit + pos

    if not od.buy_orders or not od.sell_orders:
        return orders, mem

    # Dynamic FV drift: self-calibrating from observed price data.
    # After 100 ticks, estimates drift rate from linear regression on mid_hist.
    # Adapts to any drift rate on any day — no hardcoded parameter needed.
    fv_drift = cfg.get("fv_drift", 0)
    if fv_drift:
        base_key = f"_fv_base_{product}"
        drift_key = f"_fv_drift_est_{product}"
        tid = tick_in_day(ts, mem)
        tick_num = tid // 100
        # Reset base at start of each day
        if tid == 0 or base_key not in mem:
            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
            mem[base_key] = mid
            mem[drift_key] = fv_drift  # start with configured drift
        # Self-calibrate after tick 500 only (let fixed drift work for warmup)
        if ps and len(ps.mid_hist) >= 30 and tick_num >= 500 and tick_num % 500 == 0:
            estimated_drift = (ps.mid_hist[-1] - ps.mid_hist[-30]) / 30.0
            if 0.03 < estimated_drift < 0.25:
                mem[drift_key] = round(estimated_drift, 4)
                cur_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                mem[base_key] = cur_mid - estimated_drift * tick_num
        actual_drift = mem.get(drift_key, fv_drift)
        fv = mem[base_key] + actual_drift * tick_num

    # Use wall_mid as FV when available (Frankfurt: most stable anchor)
    # Fall back to configured fair_value for known pegged products
    wall_mid = ps.wall_mid if ps and ps.wall_mid else fv

    # For known pegged products: Kalman + wall_mid + position tilt
    if abs(cfg["fair_value"] - round(cfg["fair_value"])) < 1 and not fv_drift:
        wall_mid = fv
        if ps and ps.kf_x and abs(ps.kf_x - fv) < 15:
            wall_mid = 0.9 * fv + 0.1 * ps.kf_x
        # Wall_mid blend: 50/50 static/dynamic (data: wall_mid is 120% more accurate,
        # 50/50 is conservative — keeps static FV anchor while tracking actual mid)
        if ps and ps.wall_mid and abs(ps.wall_mid - fv) < 20:
            wall_mid = 0.50 * wall_mid + 0.50 * ps.wall_mid
        # Position tilt (proven +2 PnL)
        if pos > 20:
            wall_mid -= 0.5
        elif pos < -20:
            wall_mid += 0.5
    elif fv_drift:
        wall_mid = fv  # drift FV is authoritative

    bid_wall = min(od.buy_orders.keys())
    ask_wall = max(od.sell_orders.keys())

    # v8.6 SLOPE-BIAS ACTIVE — momentum entry. 50-tick OLS slope on mid_hist:
    # when |slope| > thr, cross spread to reach target_pos = sign*slope_bias_target.
    # Hypothesis confirmed by 50K-parallel chart: top traders ride the directional
    # drift on HYD/VEF rather than MM-grinding. Default thr=0 keeps legacy off.
    slope_bias_thr = cfg.get("slope_bias_thr", 0.0)
    slope_bias_target = cfg.get("slope_bias_target", 0)
    slope_bias_window = cfg.get("slope_bias_window", 50)
    slope_dir_active = 0
    if slope_bias_thr > 0 and ps and getattr(ps, "mid_hist", None) and len(ps.mid_hist) >= slope_bias_window:
        sub = ps.mid_hist[-slope_bias_window:]
        n = len(sub)
        mx = (n - 1) / 2.0
        sxx = sum((i - mx) ** 2 for i in range(n))
        if sxx > 0:
            slope_val = sum((i - mx) * sub[i] for i in range(n)) / sxx
            if slope_val > slope_bias_thr:
                slope_dir_active = +1
            elif slope_val < -slope_bias_thr:
                slope_dir_active = -1
    if slope_bias_target > 0 and slope_dir_active != 0:
        target_pos = slope_dir_active * slope_bias_target
        delta = target_pos - pos
        if delta > 0:
            for ask_p in sorted(od.sell_orders.keys()):
                if buy_cap <= 0 or delta <= 0:
                    break
                vol = min(abs(od.sell_orders[ask_p]), buy_cap, delta)
                if vol > 0:
                    orders.append(Order(product, ask_p, +vol))
                    buy_cap -= vol
                    delta -= vol
                    pos += vol
        elif delta < 0:
            need = -delta
            for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                if sell_cap <= 0 or need <= 0:
                    break
                vol = min(od.buy_orders[bid_p], sell_cap, need)
                if vol > 0:
                    orders.append(Order(product, bid_p, -vol))
                    sell_cap -= vol
                    need -= vol
                    pos -= vol

    # ========== PHASE 1: AGGRESSIVE TAKING ==========
    tw = cfg.get("take_width", 1)
    # take_edge: minimum edge (in ticks) below wall_mid required to take.
    # Default 1 preserves historical behavior. Raise on toxic-take products
    # (e.g. VEF take_pnl=-733K vs make_pnl=+1.01M → takes are negatively
    # selected; widen threshold to filter for higher-edge fills only).
    take_edge_pegged = cfg.get("take_edge_pegged", 1)

    # R4 informed-counterparty lean: when an informed Mark fires, relax the
    # take threshold by N ticks (capped) so we lean WITH them.
    # Helper no-ops on R3-style data where buyer/seller are None.
    informed_buy_relax = 0
    informed_sell_relax = 0
    if cfg.get("informed_lean", False):
        # Use actual book mid (not wall_mid which is FV-anchored) for fwd-return tracking.
        actual_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        sig = update_informed_counterparty(market_trades, actual_mid, mem, product, ts,
                                            min_obs=cfg.get("informed_min_obs", 20),
                                            bias_thr=cfg.get("informed_bias_thr", 0.7),
                                            fwd_thr=cfg.get("informed_fwd_thr", 0.5))
        if sig:
            side, edge, name = sig
            # Aggressive relax: scale edge by mult, cap at max_relax. With edge≈1.4
            # and mult=5, relax=7 ticks — captures the forward $1-2 mid move.
            relax = min(int(round(edge * cfg.get("informed_relax_mult", 5.0))),
                        cfg.get("informed_max_relax", 10))
            if side == "buy":
                informed_buy_relax = relax
            else:
                informed_sell_relax = relax
            # R4 v9: publish signal to mem so vouchers can lean delta-1 on the
            # same alpha (intel table showed +$5.52 mean move on VEV_4000 with
            # 99.68% win rate — that's Mark 67's signal at delta=1).
            mem["_inf_sig"] = (ts, side, edge, name, product)
    # I8 gated: Thompson bandit picks take_width from configured arm set.
    # Reward = 1 if last fill markout > 0 (good), 0 otherwise.
    if cfg.get("thompson_tw", False):
        arms = cfg.get("thompson_tw_arms", [1, 2, 3])
        bandit_key = f"tsbd_tw_{product}"
        # Reward update from previous tick's fill if available
        if ps and ps.markout_count >= 1:
            last_arm = mem.get(f"tsbd_tw_arm_{product}", arms[0])
            reward = 1.0 if (ps.markout_10 or 0.0) > 0 else 0.0
            thompson_sample_update(mem, bandit_key, last_arm, reward)
        rng_seed = (state.timestamp ^ hash(product)) & 0xFFFFFFFF
        _rng = random.Random(rng_seed)
        chosen = thompson_sample_pick(mem, bandit_key, arms, _rng)
        mem[f"tsbd_tw_arm_{product}"] = chosen
        tw = int(chosen)

    # DRIFT PRODUCTS: aggressively build long position by crossing the spread.
    # Price goes up +0.1/tick. Buying at ask (even above current FV) is profitable
    # because FV will exceed our purchase price within a few ticks.
    if fv_drift and fv_drift > 0 and pos < limit:
        # Cross the spread HARD: buy everything up to FV + 8
        # With drift +0.1/tick, paying FV+8 is profitable in 80 ticks
        max_buy_price = int(wall_mid + 8)
        for ask_p in sorted(od.sell_orders.keys()):
            if buy_cap <= 0:
                break
            if ask_p <= max_buy_price:
                vol = min(abs(od.sell_orders[ask_p]), buy_cap)
                orders.append(Order(product, ask_p, +vol))
                buy_cap -= vol
    else:
        # Stationary: sweep below FV (proven on server)
        # Take-phase inventory throttle (opt-in via throttle_takes_by_inventory):
        # Live test 364694 showed HYDROGEL takes blowing past thr2=160 to pos=200 long,
        # then mid dropped 82pts = -16K drawdown. Gate buy-takes by thr1/thr2.
        thr1_t = cfg.get("inventory_threshold_1", limit)
        thr2_t = cfg.get("inventory_threshold_2", limit)
        gate_takes = cfg.get("throttle_takes_by_inventory", False)
        # v8 SLOPE-DEFENSIVE GATE: live log 412166 showed HYD ended +185 long while
        # mid drifted -51 (~9.5K MTM bleed). Static fv=9990 keeps us BUYING into
        # down-trends. Block alpha buy-takes ONLY WHEN already loaded long (pos > thr1)
        # AND 50-tick slope < -thr. Mirror for sell. This preserves alpha during normal
        # noise (the regression cause) and only fires when loaded-into-trend (the bug).
        # Default thr=0 disables → legacy behavior preserved.
        slope_def_thr = cfg.get("slope_defensive_thr", 0.0)
        slope_def_window = cfg.get("slope_defensive_window", 50)
        slope_def_inv_gate = cfg.get("slope_defensive_inv_gate", thr1_t)
        block_buy_alpha = False
        block_sell_alpha = False
        if slope_def_thr > 0 and ps and getattr(ps, "mid_hist", None) and len(ps.mid_hist) >= slope_def_window:
            sub = ps.mid_hist[-slope_def_window:]
            n = len(sub)
            mx = (n - 1) / 2.0
            sxx = sum((i - mx) ** 2 for i in range(n))
            if sxx > 0:
                slope_v = sum((i - mx) * sub[i] for i in range(n)) / sxx
                if slope_v < -slope_def_thr and pos > slope_def_inv_gate:
                    block_buy_alpha = True
                elif slope_v > slope_def_thr and pos < -slope_def_inv_gate:
                    block_sell_alpha = True
        for ask_p in sorted(od.sell_orders.keys()):
            if buy_cap <= 0:
                break
            # Inventory gate: hard stop at thr2, quartered at thr1
            if gate_takes and pos >= thr2_t:
                break  # stop buying entirely
            if ask_p <= wall_mid - take_edge_pegged + informed_buy_relax and not block_buy_alpha:
                vol = min(abs(od.sell_orders[ask_p]), buy_cap)
                if gate_takes and pos >= thr1_t:
                    vol = min(vol, max(0, thr2_t - pos))  # only add up to thr2
                # R4 v3: per-tick take rate cap. Without this, the take loop can
                # consume entire position cap in 1-2 ticks (200 HYD pin observed
                # in live log 496734). Cap forces gradual build, gives slope_defensive
                # time to engage. Default off (max_take_per_tick=0) preserves legacy.
                mtt = cfg.get("max_take_per_tick", 0)
                if mtt > 0:
                    used_buy = (limit - pos) - buy_cap  # how much we've taken this tick
                    vol = min(vol, max(0, mtt - used_buy))
                if vol > 0:
                    orders.append(Order(product, ask_p, +vol))
                    buy_cap -= vol
            elif ask_p <= wall_mid and pos < 0:
                # Offload: always allowed regardless of throttle (reduces |pos|)
                vol = min(abs(od.sell_orders[ask_p]), buy_cap, abs(pos))
                if vol > 0:
                    orders.append(Order(product, ask_p, +vol))
                    buy_cap -= vol
            elif (cfg.get("overload_offload", False) and pos < 0 and
                  abs(pos) >= int(cfg.get("overload_frac", 0.9) * limit) and
                  ask_p <= wall_mid + cfg.get("overload_tol", 2)):
                # OVERLOADED short: cover at small adverse fill to free position before
                # mean reversion punishes us. Capped to fraction of |pos| so we don't
                # flat-out — keep some exposure for the reversion edge.
                cover_cap = max(1, int(abs(pos) * cfg.get("overload_cut_frac", 0.25)))
                vol = min(abs(od.sell_orders[ask_p]), buy_cap, abs(pos), cover_cap)
                if vol > 0:
                    orders.append(Order(product, ask_p, +vol))
                    buy_cap -= vol

    for bid_p in sorted(od.buy_orders.keys(), reverse=True):
        if sell_cap <= 0:
            break
        # DRIFT: NEVER sell. Pure hold. Proven +7,392 on server.
        if fv_drift and fv_drift > 0:
            if False:
                vol = 0
                if vol > 0:
                    orders.append(Order(product, bid_p, -vol))
                    sell_cap -= vol
        else:
            # Inventory gate (mirror of buy-take): hard stop at -thr2, quartered at -thr1
            gate_takes_s = cfg.get("throttle_takes_by_inventory", False)
            thr1_s = cfg.get("inventory_threshold_1", limit)
            thr2_s = cfg.get("inventory_threshold_2", limit)
            if gate_takes_s and pos <= -thr2_s:
                break  # stop selling entirely
            if bid_p >= wall_mid + take_edge_pegged - informed_sell_relax and not block_sell_alpha:
                vol = min(od.buy_orders[bid_p], sell_cap)
                if gate_takes_s and pos <= -thr1_s:
                    vol = min(vol, max(0, thr2_s + pos))  # only add up to -thr2
                # R4 v3: per-tick take rate cap (sell side mirror)
                mtt_s = cfg.get("max_take_per_tick", 0)
                if mtt_s > 0:
                    used_sell = (limit + pos) - sell_cap  # how much we've sold this tick
                    vol = min(vol, max(0, mtt_s - used_sell))
                if vol > 0:
                    orders.append(Order(product, bid_p, -vol))
                    sell_cap -= vol
            elif bid_p >= wall_mid and pos > 0:
                # Offload: always allowed
                vol = min(od.buy_orders[bid_p], sell_cap, pos)
                if vol > 0:
                    orders.append(Order(product, bid_p, -vol))
                    sell_cap -= vol
            elif (cfg.get("overload_offload", False) and pos > 0 and
                  pos >= int(cfg.get("overload_frac", 0.9) * limit) and
                  bid_p >= wall_mid - cfg.get("overload_tol", 2)):
                # OVERLOADED long: lock in profit at small adverse fill before mean
                # reversion. Mirror of buy-side overload offload above.
                cut_cap = max(1, int(pos * cfg.get("overload_cut_frac", 0.25)))
                vol = min(od.buy_orders[bid_p], sell_cap, pos, cut_cap)
                if vol > 0:
                    orders.append(Order(product, bid_p, -vol))
                    sell_cap -= vol

    # ========== PHASE 2: PASSIVE MAKING ==========
    best_bid = max(od.buy_orders.keys())
    best_ask = min(od.sell_orders.keys())

    # DRIFT: pure hold. Buy to max, never sell. Proven +7,392 on server.
    if fv_drift and fv_drift > 0:
        # Inside-spread passive bid to catch cheaper bid-side fills.
        # Data: bid-side market trades at ts=4100 (11998), ts=6800 (12000) etc. are
        # 4-8 below the ask. Bidding at best_ask-3 catches these at better cost than
        # always crossing at the ask, saving ~3-5 per unit on passive fills.
        dbo = cfg.get("drift_bid_offset", 4)
        dam = cfg.get("drift_ask_margin", 3)
        our_bid = min(int(wall_mid) + dbo, best_ask - dam)
        our_bid = max(our_bid, best_bid + 2)
        our_ask = int(wall_mid) + 999  # never sell
    else:
        # Stationary: penny-jump +2 from book for exclusive queue position.
        # Data shows 83% of ticks our old bid+1 competed with other penny-jumpers.
        # bid+2 ensures we're the exclusive best bid by 2, capturing all fills.
        our_bid = int(bid_wall + 1)
        our_ask = int(ask_wall - 1)
        for bp in sorted(od.buy_orders.keys(), reverse=True):
            bv = od.buy_orders[bp]
            if bv > 1 and bp + 2 < wall_mid:
                our_bid = max(our_bid, bp + 2)   # +2 not +1: exclusive queue
                break
            elif bp < wall_mid:
                our_bid = max(our_bid, bp + 1)
                break
        for ap in sorted(od.sell_orders.keys()):
            av = abs(od.sell_orders[ap])
            if av > 1 and ap - 2 > wall_mid:
                our_ask = min(our_ask, ap - 2)   # -2 not -1: exclusive queue
                break
            elif ap > wall_mid:
                our_ask = min(our_ask, ap - 1)
                break

    # Depth imbalance signal — TIERED response calibrated from 30K-tick analysis:
    # OFI<-0.4 predicts -2.3 price drop (168/1K ticks); OFI>+0.4 predicts +3.1 rise.
    # Old: ±1 tick. New: tiered ±1/±2 with asymmetric bearish bias (stronger signal).
    if not fv_drift and ps and ps.bid_depth > 0 and ps.ask_depth > 0:
        imb = (ps.bid_depth - ps.ask_depth) / (ps.bid_depth + ps.ask_depth)
        if imb > 0.5:
            our_bid += 2    # very strong buy signal
        elif imb > 0.3:
            our_bid += 1    # moderate buy signal
        elif imb < -0.5:
            our_ask -= 2    # very strong sell signal — pull bids back too
            our_bid -= 1
        elif imb < -0.3:
            our_ask -= 1    # moderate sell signal

    # Spread compression aggression: spread<10 = high-activity + 57% bullish bias.
    # Data: 65 compression events in 1K ticks earn 4x normal rate per tick.
    # When market is tight, go 1 tick more aggressive on both sides.
    if not fv_drift:
        cur_spread = best_ask - best_bid
        if cur_spread <= 10:
            our_bid += 1    # more aggressive: 57% chance market moves up


    # A-S inventory skew: shift quotes based on position
    # For drift products: minimal skew — we WANT to be long
    if fv_drift and fv_drift > 0:
        # Only skew when position is extreme (>90% of limit)
        if pos > limit * 0.9:
            as_shift = 2  # push to sell when nearly full
        elif pos < -limit * 0.5:
            as_shift = -2  # push to buy when short (wrong side!)
        else:
            as_shift = 0  # no skew — stay long
    else:
        sigma_sq = max(1.0, (ps.realized_vol if ps else 2.0) ** 2)
        gm = ps.gamma_mult if ps else 1.0
        gamma_as = gm * 2.0 / (limit * sigma_sq)
        as_shift = int(round(max(-3, min(3, gamma_as * pos * sigma_sq))))
    our_bid -= as_shift
    our_ask -= as_shift

    # Safety: never cross wall_mid
    our_bid = min(our_bid, int(wall_mid) - 1)
    our_ask = max(our_ask, int(wall_mid) + 1)

    if our_bid >= our_ask:
        our_bid = int(wall_mid) - 1
        our_ask = int(wall_mid) + 1

    # Inventory-threshold quoting: prevent extreme positions.
    # thr1: reduce the "worsening" side to 25% size (slow accumulation)
    # thr2: fully suppress the "worsening" side (only quote the exit side)
    # Previously configured but never applied — now live.
    thr1 = cfg.get("inventory_threshold_1", limit)
    thr2 = cfg.get("inventory_threshold_2", limit)
    bid_sz = buy_cap
    ask_sz = sell_cap
    if pos >= thr2:
        bid_sz = 0                            # very long: stop buying entirely
    elif pos >= thr1:
        bid_sz = max(0, min(buy_cap, limit // 4))  # long: throttle buys to 25%
    if pos <= -thr2:
        ask_sz = 0                            # very short: stop selling entirely
    elif pos <= -thr1:
        ask_sz = max(0, min(sell_cap, limit // 4))  # short: throttle sells to 25%

    # B3: markout-driven size reduction (opt-in via cfg). R2 ASH collapsed because
    # make_size stayed full while we were being adversely picked off. Enable this
    # flag on products whose live markout turns negative; default OFF to protect
    # backtest parity on products where markout is noisy.
    if cfg.get("markout_sizedown", False):
        cur_mid_for_markout = (best_bid + best_ask) / 2.0
        ms = markout_score(mem, product, cur_mid_for_markout)
        if ms < -0.5:
            bid_sz = max(1, bid_sz // 2) if bid_sz else 0
            ask_sz = max(1, ask_sz // 2) if ask_sz else 0

    # Active unwind when pinned at limit AND bleeding (opt-in).
    # Rationale: tick analysis of live log 364944 showed HYDROGEL pinned at ±200
    # for 98.4% of ticks with zero fills on passive makes — pure MTM bleed.
    # Only fires when BOTH conditions hold: (a) |pos| near limit for N ticks,
    # (b) mid has drifted against position by unwind_mid_move_thr ticks.
    # Without (b), backtest fires on oscillating mid and loses spread unnecessarily.
    if cfg.get("unwind_when_pinned", False) and not fv_drift:
        pin_frac = cfg.get("unwind_pin_frac", 0.98)
        stuck_trigger = cfg.get("unwind_after_stuck", 10)
        unwind_size = cfg.get("unwind_size", 20)
        mid_move_thr = cfg.get("unwind_mid_move_thr", 3)
        stuck_key = f"_stuck_{product}"
        pin_mid_key = f"_stuck_pin_mid_{product}"
        cur_mid = (best_bid + best_ask) / 2.0
        if pos >= int(pin_frac * limit):
            sc = mem.get(stuck_key, 0) + 1
            mem[stuck_key] = sc
            if sc == 1:
                mem[pin_mid_key] = cur_mid  # record mid when we first pinned
            pin_mid = mem.get(pin_mid_key, cur_mid)
            bleeding = (pin_mid - cur_mid) >= mid_move_thr  # mid dropped → long bleeds
            if sc >= stuck_trigger and bleeding and ask_sz > 0:
                cross_qty = min(ask_sz, unwind_size, pos)
                if cross_qty > 0:
                    our_ask = best_bid
                    ask_sz = cross_qty
        elif pos <= -int(pin_frac * limit):
            sc = mem.get(stuck_key, 0) + 1
            mem[stuck_key] = sc
            if sc == 1:
                mem[pin_mid_key] = cur_mid
            pin_mid = mem.get(pin_mid_key, cur_mid)
            bleeding = (cur_mid - pin_mid) >= mid_move_thr  # mid rose → short bleeds
            if sc >= stuck_trigger and bleeding and bid_sz > 0:
                cross_qty = min(bid_sz, unwind_size, -pos)
                if cross_qty > 0:
                    our_bid = best_ask
                    bid_sz = cross_qty
        else:
            mem[stuck_key] = 0
            mem.pop(pin_mid_key, None)

    if bid_sz > 0:
        orders.append(Order(product, our_bid, +bid_sz))
    if ask_sz > 0:
        orders.append(Order(product, our_ask, -ask_sz))

    return orders, mem

# ============================================================
# STRATEGY: AR + OLIVIA (KELP-type)
# ============================================================

def _adaptive_classify(market_trades, mid, mem, product):
    """Auto-detect product behavior: 'ofi' (trade-following) vs 'mr' (mean-reverting).
    Uses rolling trade-direction vs mid-change correlation.
    Returns ('ofi', strength) or ('mr', strength)."""
    # Track trade signs and mid changes
    ts_key = f"adapt_ts_{product}"
    mc_key = f"adapt_mc_{product}"
    trade_signs = mem.get(ts_key, [])
    mid_changes = mem.get(mc_key, [])

    # Record mid change from last tick
    prev_mid_key = f"adapt_pm_{product}"
    prev_mid = mem.get(prev_mid_key, mid)
    mid_change = mid - prev_mid
    mem[prev_mid_key] = mid

    # Compute net trade sign this tick
    net_sign = 0
    trades = market_trades if isinstance(market_trades, list) else market_trades.get(product, [])
    for t in trades:
        if hasattr(t, 'buyer') and t.buyer:
            net_sign += t.quantity
        elif hasattr(t, 'seller') and t.seller:
            net_sign -= t.quantity
        else:
            # Anonymous: use price vs mid
            if t.price >= mid:
                net_sign += t.quantity
            else:
                net_sign -= t.quantity

    trade_signs.append(1 if net_sign > 0 else (-1 if net_sign < 0 else 0))
    mid_changes.append(mid_change)

    # Keep rolling window of 100
    if len(trade_signs) > 100:
        trade_signs = trade_signs[-100:]
        mid_changes = mid_changes[-100:]
    mem[ts_key] = trade_signs
    mem[mc_key] = mid_changes

    # Need at least 30 ticks to classify
    if len(trade_signs) < 30:
        return "unknown", 0.0

    # Only reclassify every 10 ticks (performance optimization)
    classify_counter_key = f"adapt_cc_{product}"
    cc = mem.get(classify_counter_key, 0) + 1
    mem[classify_counter_key] = cc
    cached_mode_key = f"adapt_mode_{product}"
    if cc % 5 != 0 and cached_mode_key in mem:
        return mem[cached_mode_key]

    # Correlation: trade sign at t vs CUMULATIVE mid change from t+1 to t+5
    n = len(trade_signs) - 5
    if n < 20:
        return "unknown", 0.0

    signs = trade_signs[:n]
    # Prefix sum for O(n) cumulative forward changes
    prefix = [0.0]
    for mc in mid_changes:
        prefix.append(prefix[-1] + mc)
    changes = [prefix[i+6] - prefix[i+1] for i in range(n)]
    s_mean = sum(signs) / n
    c_mean = sum(changes) / n
    cov = sum((s - s_mean) * (c - c_mean) for s, c in zip(signs, changes)) / n
    s_std = (sum((s - s_mean)**2 for s in signs) / n) ** 0.5
    c_std = (sum((c - c_mean)**2 for c in changes) / n) ** 0.5

    if s_std < 0.01 or c_std < 0.001:
        return "mr", 0.5  # no variation, default to mean-revert

    corr = cov / (s_std * c_std)

    if corr > 0.25:
        result = ("ofi", min(corr, 1.0))
    elif corr < -0.10:
        result = ("mr", min(abs(corr), 1.0))
    else:
        result = ("mr", 0.3)
    mem[cached_mode_key] = result
    return result


def run_generic_mm(product, cfg, od, pos, mem, ts, market_trades, own_trades, ps=None):
    """Adaptive MM for unknown products."""
    limit = cfg.get("position_limit", 50)
    orders = []
    buy_cap = limit - pos
    sell_cap = limit + pos

    if not od.buy_orders or not od.sell_orders:
        return orders, mem

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    # Track history
    hist_key = f"hist_{product}"
    hist = mem.get(hist_key, [])
    hist.append(mid)
    if len(hist) > 250:
        hist = hist[-250:]
    mem[hist_key] = hist

    # Pre-computed from ProductState
    kf_fv = ps.kf_x if ps else mid
    m_conf = ps.model_confidence if ps else 1.0
    ofi = ps.ofi if ps else 0.0
    toxicity = ps.toxicity if ps else 0.5
    sweep = ps.sweep if ps else 0

    # NO-TRADE ZONE: skip when confidence is very low
    if m_conf < 0.3 and abs(pos) < limit * 0.3:
        return orders, mem

    # ADAPTIVE CLASSIFICATION: auto-detect product behavior
    mode, mode_strength = _adaptive_classify(market_trades, mid, mem, product)

    # Build FV based on detected mode
    mp = ps.microprice if ps and ps.microprice else mid
    w_fv = ps.wall_mid if ps and ps.wall_mid else mid
    fv = 0.5 * mp + 0.5 * w_fv

    if mode == "ofi":
        # Trade-following mode (KELP-like): trust OFI heavily, blend with Kalman
        fv = 0.5 * fv + 0.5 * kf_fv
        ofi_weight = 1.0 + mode_strength  # stronger correlation -> more OFI trust
        fv += ofi * ofi_weight
        if sweep != 0:
            fv += sweep * 1.0  # trust sweeps more in OFI mode
    elif mode == "mr":
        # Mean-revert mode: trust Kalman heavily, actively fade deviations
        fv = 0.3 * fv + 0.7 * kf_fv  # lean more on Kalman
        fv += ofi * 0.3  # dampen OFI (trades are noise)
        # Active mean-reversion: push FV toward Kalman when price deviates
        deviation = mid - kf_fv
        fv -= deviation * 0.5 * mode_strength  # stronger fade
    else:
        # Unknown (warmup): conservative blend
        fv = 0.6 * fv + 0.4 * kf_fv
        fv += ofi * 0.5
        if sweep != 0:
            fv += sweep * 0.5

    # Olivia check — named first, behavioral fallback for P4
    olivia_dir = check_olivia(market_trades, own_trades)
    if olivia_dir is None:
        olivia_dir = detect_informed_bot(market_trades, mid, mem, product)
    if olivia_dir:
        mem[f"olv_{product}"] = olivia_dir
        mem[f"olv_ts_{product}"] = ts

    active_olivia = mem.get(f"olv_{product}")

    # Sizing: base * confidence * toxicity * inventory_pressure
    base_sz = cfg.get("make_size", 15)
    tox_adj = ps.tox_adj if ps else 1.0
    inv_pressure = max(0.2, 1.0 - ps.inv_ratio) if ps else max(0.2, 1.0 - abs(pos) / max(limit, 1))
    make_sz = max(1, int(base_sz * m_conf * tox_adj * inv_pressure))

    # Take mispriced (with tolerance for wide spreads)
    take_width = max(1, int(spread * 0.30))  # take within 30% of spread
    for ask_p in sorted(od.sell_orders.keys()):
        if buy_cap <= 0 or ask_p > fv + take_width: break
        vol = min(abs(od.sell_orders[ask_p]), buy_cap, make_sz)
        orders.append(Order(product, ask_p, +vol))
        buy_cap -= vol

    for bid_p in sorted(od.buy_orders.keys(), reverse=True):
        if sell_cap <= 0 or bid_p < fv - take_width: break
        vol = min(od.buy_orders[bid_p], sell_cap, make_sz)
        orders.append(Order(product, bid_p, -vol))
        sell_cap -= vol

    # A-S reservation price: skew quotes based on inventory
    sigma_est = 1.0
    if len(hist) > 20:
        diffs = [hist[i] - hist[i-1] for i in range(-min(20, len(hist)), 0)]
        sigma_est = max(0.5, (sum(d*d for d in diffs) / len(diffs)) ** 0.5)
    gm_g = ps.gamma_mult if ps else 1.0
    r_price = reservation_price(fv, pos, sigma_est, gamma=0.1 * gm_g, T_remaining=0.5)

    # Passive quotes: for tight spreads penny-jump, for wide spreads quote at book levels
    if spread <= 4:
        our_bid = int(r_price - max(1, spread // 4))
        our_ask = int(r_price + max(1, spread // 4)) + 1
        if spread > 2:
            our_bid = max(our_bid, best_bid + 1)
            our_ask = min(our_ask, best_ask - 1)
    else:
        # Wide spread: quote near the book levels where fills actually happen
        offset = max(2, int(spread * 0.3))
        our_bid = int(r_price - offset)
        our_ask = int(r_price + offset) + 1
        # Don't penny-jump on wide spreads — it puts us in no-man's land
    if our_bid >= our_ask:
        our_bid = best_bid
        our_ask = best_ask

    buy_sz = make_sz
    sell_sz = make_sz

    # Directional tilt based on mode
    if mode == "ofi":
        # In OFI mode, trust directional signals more
        if active_olivia == "LONG" or ofi > 0.2 or sweep > 0:
            buy_sz = make_sz * 2
            sell_sz = max(1, make_sz // 3)
        elif active_olivia == "SHORT" or ofi < -0.2 or sweep < 0:
            sell_sz = make_sz * 2
            buy_sz = max(1, make_sz // 3)
    else:
        # In MR/unknown mode, only tilt on strong signals
        if active_olivia == "LONG" or ofi > 0.5:
            buy_sz = make_sz * 2
            sell_sz = max(1, make_sz // 3)
        elif active_olivia == "SHORT" or ofi < -0.5:
            sell_sz = make_sz * 2
            buy_sz = max(1, make_sz // 3)

    # Inventory pressure
    if abs(pos) > limit * 0.7:
        if pos > 0: sell_sz = min(sell_sz * 2, sell_cap)
        else: buy_sz = min(buy_sz * 2, buy_cap)

    if buy_cap > 0:
        orders.append(Order(product, our_bid, +min(buy_sz, buy_cap)))
    if sell_cap > 0:
        orders.append(Order(product, our_ask, -min(sell_sz, sell_cap)))

    return orders, mem

# ============================================================
# STRATEGY: BASKET ARB (R2 -- Gift Basket / Picnic Basket type)
# ============================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF -- Abramowitz & Stegun, error < 7.5e-8."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    prob = d * math.exp(-x * x / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
        t * (-1.821255978 + t * 1.330274429))))
    )
    return 0.5 + sign * (0.5 - prob)

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(S, K, T, sigma, r=0.0):
    """Black-Scholes European call price."""
    if S <= 0 or K <= 0: return 0.0
    if T <= 0: return max(S - K, 0.0)
    if sigma <= 0: return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_delta(S, K, T, sigma, r=0.0):
    if S <= 0 or K <= 0: return 0.0
    if T <= 0 or sigma <= 0: return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)

def bs_vega(S, K, T, sigma, r=0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * _norm_pdf(d1)

def _ema_update(mem, key, window, value):
    """EMA helper: alpha = 2/(window+1). Returns new EMA value."""
    old = mem.get(key, 0.0)
    alpha = 2.0 / (window + 1)
    new = alpha * value + (1 - alpha) * old
    mem[key] = new
    return new


def _smile_theo(S, K, TTE, smile_coeffs):
    """Compute BS theo price from vol smile fit.
    smile_coeffs = [a, b, c] for quadratic: IV = a*m^2 + b*m + c
    where m = log(K/S) / sqrt(TTE)."""
    if S <= 0 or TTE <= 0:
        return None, None, None
    m = math.log(K / S) / math.sqrt(TTE)
    iv = smile_coeffs[0] * m * m + smile_coeffs[1] * m + smile_coeffs[2]
    if iv <= 0.01:
        return None, None, None
    theo = bs_call(S, K, TTE, iv)
    delta = bs_delta(S, K, TTE, iv)
    vega = bs_vega(S, K, TTE, iv)
    return theo, delta, vega


def run_mean_revert_mm(product, cfg, od, pos, mem, ts=0):
    """Aggressive deviation-sized mean reversion (Frankfurt/CMU pattern).

    Replaces 'pegged' philosophy on high-volatility mean-reverting products.
    Pegged accumulates inventory passively until it gets pinned at limit BEFORE
    the move; this strategy SIZES INTO the move proportional to EMA deviation.

    Diagnostic that motivated this (server log 377029, HYDROGEL_PACK):
      stuck@80%limit=984/1000 ticks, 42 takes/4 makes, worst tick -1600 MTM
      with pos=+200 (already maxed) when mid dropped 8.

    Logic per tick:
      1. Update EMA(window) of mid.
      2. Compute deviation = mid - ema.
      3. target_pos = clip(-deviation / scale * limit, -limit, limit).
         (Negative because dev>0 means mid above EMA → expect down → want short.)
      4. Trade gap = target_pos - pos by crossing BBO (take liquidity).
         Walk multiple price levels if needed.
      5. Optional: passive scratch quotes when |dev| < flat_thr (capture spread).

    Config:
      ema_window:    EMA half-life in ticks (e.g. 20)
      scale:         deviation in ticks at which target=±limit (e.g. p90 of |dev|)
      flat_thr:      |dev| below this triggers unwind to 0 / passive scratch
      min_trade_qty: skip orders below this size (avoid 1-lot churn)
      max_levels:    walk up to N levels deep when crossing
      passive:       if True, also place 1-lot scratch makes when flat
    """
    fv = cfg.get("fair_value")  # optional anchor; defaults to EMA seed
    limit = cfg["position_limit"]
    window = cfg.get("ema_window", 20)
    scale = cfg.get("scale", 30.0)
    flat_thr = cfg.get("flat_thr", 5.0)
    min_qty = cfg.get("min_trade_qty", 2)
    max_levels = cfg.get("max_levels", 3)

    if not od.buy_orders or not od.sell_orders:
        return [], mem

    bids_desc = sorted(od.buy_orders.keys(), reverse=True)
    asks_asc = sorted(od.sell_orders.keys())
    bid = bids_desc[0]
    ask = asks_asc[0]
    mid = (bid + ask) / 2.0

    ema_key = f"_mrm_ema_{product}"
    ema = mem.get(ema_key)
    if ema is None:
        ema = float(fv) if fv is not None else mid
    alpha = 2.0 / (window + 1.0)
    ema = alpha * mid + (1 - alpha) * ema
    mem[ema_key] = ema

    deviation = mid - ema
    raw_target = -deviation / scale * limit
    target = int(max(-limit, min(limit, round(raw_target))))

    orders = []
    used_buy = 0
    used_sell = 0

    # If |dev| small AND pos != 0, force unwind toward zero.
    if abs(deviation) < flat_thr:
        target = 0

    gap = target - pos

    if gap > 0:
        # Need to BUY |gap| more — walk asks
        remaining = gap
        for px in asks_asc[:max_levels]:
            avail = od.sell_orders.get(px, 0)
            if avail <= 0:
                continue
            qty = min(remaining, avail, limit - pos - used_buy)
            if qty >= min_qty:
                orders.append(Order(product, px, qty))
                used_buy += qty
                remaining -= qty
            if remaining <= 0:
                break
    elif gap < 0:
        # Need to SELL |gap| more — walk bids
        remaining = -gap
        for px in bids_desc[:max_levels]:
            avail = od.buy_orders.get(px, 0)
            if avail <= 0:
                continue
            qty = min(remaining, avail, limit + pos - used_sell)
            if qty >= min_qty:
                orders.append(Order(product, px, -qty))
                used_sell += qty
                remaining -= qty
            if remaining <= 0:
                break

    return orders, mem


def run_voucher_intrinsic_mm(product, cfg, state, pos, mem):
    """Deep-ITM voucher MM at intrinsic with delta-hedge to UL.

    Voucher analysis (analysis/voucher_intel.py) shows VEV_4000 has avg spread
    20.8 ticks (mid 1250) and VEV_4500 has 15.9 (mid 750). These are deep ITM
    with TV~0 — they trade at intrinsic. Nobody MMs them; spread = pure alpha.

    Logic per tick:
      1. theo = max(UL_mid - K, 0) (intrinsic; delta=1 for ITM)
      2. Quote at int(theo)-make_edge / int(theo)+make_edge (inside the wide BBO)
      3. Take aggressively when ask <= theo - take_edge OR bid >= theo + take_edge
      4. Record per-strike (pos, delta=1, UL) to mem for skew_delta_agg.

    Config: underlying, strike, position_limit, make_size, make_edge, take_edge,
            max_pos_frac, inv_skew_ticks (optional)
    """
    underlying = cfg["underlying"]
    K = cfg["strike"]
    limit = cfg["position_limit"]
    make_size = cfg.get("make_size", 30)
    make_edge = cfg.get("make_edge", 4)
    take_edge = cfg.get("take_edge", 6)
    max_pos_frac = cfg.get("max_pos_frac", 0.9)
    inv_skew_ticks = cfg.get("inv_skew_ticks", 1)

    od = state.order_depths.get(product)
    ul_od = state.order_depths.get(underlying)
    if not od or not od.buy_orders or not od.sell_orders:
        return [], mem, {}
    if not ul_od or not ul_od.buy_orders or not ul_od.sell_orders:
        return [], mem, {}

    ul_bid = max(ul_od.buy_orders.keys())
    ul_ask = min(ul_od.sell_orders.keys())
    ul_mid = (ul_bid + ul_ask) / 2.0
    theo = max(ul_mid - K, 0.0)
    if theo <= 0:
        return [], mem, {}

    bids_desc = sorted(od.buy_orders.keys(), reverse=True)
    asks_asc = sorted(od.sell_orders.keys())

    orders = []
    used_buy = 0
    used_sell = 0
    max_long = int(max_pos_frac * limit)
    max_short = -max_long

    # R4 v9: informed-counterparty leverage. When informed Mark fired on UL
    # (e.g. Mark 67 buys VEF), buy this voucher aggressively (delta-1 leverage
    # on the same alpha). Intel table showed +$5.52 mean move on VEV_4000 with
    # 99.68% win rate — that's Mark 67's signal at delta=1.
    informed_take_premium = 0
    if cfg.get("informed_voucher_long", False):
        sig = mem.get("_inf_sig")
        if sig and sig[1] == "buy" and sig[4] == underlying:
            sig_ts, _, edge, _, _ = sig
            # Signal valid for 30 ticks after fire
            if state.timestamp - sig_ts <= 3000:
                informed_take_premium = min(int(round(edge * 1.0)),
                                            cfg.get("informed_voucher_max_premium", 3))

    # --- TAKES: cross BBO when far from theo ---
    # informed_take_premium: relax buy threshold by N ticks when informed Mark fired
    for px in asks_asc:
        if px > theo - take_edge + informed_take_premium:
            break
        avail = od.sell_orders.get(px, 0)
        if avail <= 0:
            continue
        room = max_long - pos - used_buy
        qty = min(avail, room)
        if qty > 0:
            orders.append(Order(product, px, qty))
            used_buy += qty

    for px in bids_desc:
        if px < theo + take_edge:
            break
        avail = od.buy_orders.get(px, 0)
        room = max_long + pos - used_sell
        qty = min(avail, room)
        if qty > 0:
            orders.append(Order(product, px, -qty))
            used_sell += qty

    # --- MAKES: peg quote at theo ± make_edge with inventory skew ---
    if cfg.get("make_side", True):
        inv_frac = pos / max(limit, 1)
        skew = int(round(inv_skew_ticks * inv_frac))
        bid_px = int(math.floor(theo - make_edge)) - skew
        ask_px = int(math.ceil(theo + make_edge)) - skew
        # Force valid spread (avoid self-cross)
        if bid_px >= ask_px:
            ask_px = bid_px + 1
        buy_room = max_long - pos - used_buy
        sell_room = max_long + pos - used_sell
        mq_buy = min(make_size, buy_room)
        mq_sell = min(make_size, sell_room)
        if mq_buy > 0:
            orders.append(Order(product, bid_px, mq_buy))
        if mq_sell > 0:
            orders.append(Order(product, ask_px, -mq_sell))

    # --- Record delta for hedge aggregator (delta ≈ 1 for deep ITM) ---
    delta_req = mem.get("_skew_delta_req", {})
    delta_req[product] = {
        "pos": pos, "delta": 1.0, "UL": underlying, "vega": 0.0,
    }
    mem["_skew_delta_req"] = delta_req

    return orders, mem, {}


def run_voucher_chain_dir(product, cfg, state, pos, mem):
    """Directional voucher position-taking on underlying momentum.

    Reverse-engineered from competitor 30K backtest showing per-strike PnL of
    +15 to +83K across VEV chain (vs our skew_mm cap at +740). Hypothesis: top
    teams ride VEF directional drift via voucher chain leverage. They roll
    positions on momentum reversals (~25 round-trips on 30K, each capturing
    ~3.5K gross / ~200 spread cost).

    Config: underlying, strike, dir_window, dir_thr, dir_target_size, position_limit.
    Does NOT hedge — directional exposure is the alpha. No _skew_delta_req write.
    """
    orders = []
    od = state.order_depths.get(product)
    if not od or not od.buy_orders or not od.sell_orders:
        return orders, mem, {}
    underlying = cfg.get("underlying", "VELVETFRUIT_EXTRACT")
    ul_od = state.order_depths.get(underlying)
    ul_hist_key = f"_chain_dir_ul_{underlying}"
    if ul_od and ul_od.buy_orders and ul_od.sell_orders:
        ul_mid = 0.5 * (max(ul_od.buy_orders.keys()) + min(ul_od.sell_orders.keys()))
        hist = mem.get(ul_hist_key, [])
        if not isinstance(hist, list):
            hist = []
        hist.append(ul_mid)
        if len(hist) > 1500:
            hist = hist[-1500:]
        mem[ul_hist_key] = hist
    hist = mem.get(ul_hist_key, [])
    window = cfg.get("dir_window", 500)
    thr = cfg.get("dir_thr", 0.005)
    target_size = cfg.get("dir_target_size", 50)
    if len(hist) < window:
        return orders, mem, {}
    sub = hist[-window:]
    n = len(sub)
    mx = (n - 1) / 2.0
    sxx = sum((i - mx) * (i - mx) for i in range(n))
    if sxx <= 0:
        return orders, mem, {}
    slope = sum((i - mx) * sub[i] for i in range(n)) / sxx
    if slope > thr:
        target = +target_size
    elif slope < -thr:
        target = -target_size
    else:
        target = 0
    delta = target - pos
    if delta > 0:
        for ask_p in sorted(od.sell_orders.keys()):
            if delta <= 0:
                break
            vol = min(abs(od.sell_orders[ask_p]), delta)
            if vol > 0:
                orders.append(Order(product, ask_p, +vol))
                delta -= vol
    elif delta < 0:
        need = -delta
        for bid_p in sorted(od.buy_orders.keys(), reverse=True):
            if need <= 0:
                break
            vol = min(od.buy_orders[bid_p], need)
            if vol > 0:
                orders.append(Order(product, bid_p, -vol))
                need -= vol
    return orders, mem, {}


def run_otm_short(product, cfg, od, pos, mem, ts):
    """OTM voucher theta-decay strategy.

    Rationale (per R4 dashboard intel): OTM vouchers (VEV_5300/5400/5500) trade
    purely on time value which decays to 0 by expiry. Short them and hold = collect
    premium. Theta table:
      VEV_5300: $52 -> $26 -> $0 over 3 days  (-$26 captured per voucher)
      VEV_5400: $17 -> $20 -> $5.5            (-$11.5)
      VEV_5500: $6.5 -> $7 -> $1.5            (-$5)
    At max position 300, total alpha = ~$12.7K per round if held to expiry.

    Strategy:
      - Build short up to target_short by hitting bids
      - Stop-loss: if mid > stop_loss_mid, force-close (VEF spike risk)
      - No buys except at stop-loss
    """
    orders = []
    if not od.buy_orders or not od.sell_orders:
        return orders, mem
    target_short = cfg.get("target_short", 100)  # negative position cap
    stop_loss_mid = cfg.get("stop_loss_mid", 9999)  # if mid exceeds, force-close
    max_sell_per_tick = cfg.get("max_sell_per_tick", 30)

    best_bid = max(od.buy_orders.keys())
    best_ask = min(od.sell_orders.keys())
    cur_mid = (best_bid + best_ask) / 2.0

    # Stop-loss path: if mid exceeds threshold, close any short
    if pos < 0 and cur_mid > stop_loss_mid:
        cover_qty = min(od.sell_orders[best_ask], -pos, max_sell_per_tick)
        if cover_qty > 0:
            orders.append(Order(product, best_ask, +cover_qty))
        return orders, mem

    # Build short: hit best bid
    if pos > -target_short:
        room = -target_short - pos  # negative; we want to sell -room units
        sell_qty = min(od.buy_orders[best_bid], -room, max_sell_per_tick)
        if sell_qty > 0:
            orders.append(Order(product, best_bid, -sell_qty))

    return orders, mem


def run_skew_mm(product, cfg, state, pos, mem):
    """Delta-hedged skew MM around calibrated smile theo.

    No EMA self-reference (which absorbed structural bias in run_options). Pure
    2-sided market making with inventory skew. Delta exposure aggregated across
    strikes and hedged via underlying at end of tick (see run_skew_delta_agg).

    Config: underlying, strike, smile_coeffs, total_option_days, day_offset,
            position_limit, make_size, make_edge, take_edge, max_pos_frac,
            inv_skew_ticks.
    """
    orders = []
    hedge_orders = {}

    od = state.order_depths.get(product)
    if od is None or not od.buy_orders or not od.sell_orders:
        return orders, mem, hedge_orders

    UL = cfg.get("underlying", "")
    ul_od = state.order_depths.get(UL)
    if ul_od is None or not ul_od.buy_orders or not ul_od.sell_orders:
        return orders, mem, hedge_orders

    K = float(cfg.get("strike", 5300))
    smile = cfg.get("smile_coeffs", [0.02671337, 0.00040480, 0.23939739])
    total_days = cfg.get("total_option_days", 8)
    day_offset = cfg.get("day_offset", 0)

    # Track current day via timestamp reset
    opt_day_key = "_skew_current_day"
    prev_ts = mem.get("_skew_prev_ts", -1)
    current_day = mem.get(opt_day_key, 0)
    if state.timestamp < prev_ts:
        current_day += 1
        mem[opt_day_key] = current_day
    mem["_skew_prev_ts"] = state.timestamp
    day = current_day + day_offset

    S = (max(ul_od.buy_orders) + min(ul_od.sell_orders)) / 2.0
    TTE = max(1.0 - (365.0 - total_days + day + state.timestamp / 100.0 / 10000.0) / 365.0, 1e-6)

    theo, delta, vega = _smile_theo(S, K, TTE, smile)
    if theo is None or delta is None or theo < 0.5:
        # Too cheap to trade reliably
        return orders, mem, hedge_orders

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    wall_mid_local = _get_wall_mid(od)
    if wall_mid_local is None:
        wall_mid_local = (best_bid + best_ask) / 2.0

    # Per-strike persistent smile bias: offline fit has up to ±2 systematic
    # residual per strike. Without this, takes fire one-sided forever.
    # EMA tracks (wall_mid - theo) and we trade around (theo + bias).
    # bias_init pre-seeds the EMA so we don't burn ticks waiting for convergence.
    bias_window = cfg.get("bias_window", 1500)
    bias_warmup = cfg.get("bias_warmup", 200)
    bias_init = cfg.get("bias_init", None)
    bias_key = f"_skew_bias_{product}"
    bias_ticks_key = f"_skew_bias_ticks_{product}"
    bias_n = mem.get(bias_ticks_key, 0) + 1
    mem[bias_ticks_key] = bias_n
    if bias_n == 1 and bias_init is not None:
        mem[bias_key] = float(bias_init)
    bias = _ema_update(mem, bias_key, bias_window, wall_mid_local - theo)
    if bias_n < bias_warmup:
        return orders, mem, hedge_orders
    fair = theo + bias

    limit = cfg.get("position_limit", 100)
    make_size = cfg.get("make_size", 15)
    take_edge = cfg.get("take_edge", 1.5)
    make_edge = cfg.get("make_edge", 1.0)
    max_pos_frac = cfg.get("max_pos_frac", 0.9)
    inv_skew_ticks = cfg.get("inv_skew_ticks", 2)

    buy_cap = limit - pos
    sell_cap = limit + pos
    filled_buy = 0
    filled_sell = 0

    # AGGRESSIVE TAKES: ask < fair - take_edge -> buy; bid > fair + take_edge -> sell
    if pos < int(max_pos_frac * limit):
        for ap in sorted(od.sell_orders):
            if buy_cap <= 0:
                break
            if ap <= fair - take_edge:
                vol = min(abs(od.sell_orders[ap]), buy_cap)
                orders.append(Order(product, ap, +vol))
                buy_cap -= vol
                filled_buy += vol
            else:
                break

    if pos > -int(max_pos_frac * limit):
        for bp in sorted(od.buy_orders, reverse=True):
            if sell_cap <= 0:
                break
            if bp >= fair + take_edge:
                vol = min(od.buy_orders[bp], sell_cap)
                orders.append(Order(product, bp, -vol))
                sell_cap -= vol
                filled_sell += vol
            else:
                break

    # 2-SIDED MAKES with inventory skew (tilt quotes away from inventory).
    # v4 first test: takes +7K/11 fills vs makes -3.6K/26 fills. Making was
    # adversely selected as market drifted toward our quotes. Gate via cfg flag.
    if cfg.get("make_side", True):
        inv_frac = pos / max(limit, 1)
        skew = int(round(inv_skew_ticks * inv_frac))

        bid_px = int(fair - make_edge) - skew
        ask_px = int(math.ceil(fair + make_edge)) - skew

        # Respect BBO: don't cross opposite side
        if bid_px >= best_ask:
            bid_px = best_ask - 1
        if ask_px <= best_bid:
            ask_px = best_bid + 1
        bid_px = max(bid_px, 1)

        # Only quote if we have real capacity AND not pinned near limit
        if buy_cap >= make_size and pos < int(max_pos_frac * limit):
            orders.append(Order(product, bid_px, +make_size))
        if sell_cap >= make_size and pos > -int(max_pos_frac * limit):
            orders.append(Order(product, ask_px, -make_size))

    # Record delta contribution for aggregator
    expected_pos = pos + filled_buy - filled_sell
    mem.setdefault("_skew_delta_req", {})
    mem["_skew_delta_req"][product] = {
        "pos": expected_pos, "delta": delta, "UL": UL, "vega": vega,
    }

    # Diagnostics
    mem[f"skew_theo_{product}"] = round(theo, 2)
    mem[f"skew_fair_{product}"] = round(fair, 2)
    mem[f"skew_bias_{product}"] = round(bias, 3)
    mem[f"skew_delta_{product}"] = round(delta, 3)
    mem[f"skew_pos_{product}"] = expected_pos

    return orders, mem, hedge_orders


def run_skew_delta_agg(config_map, state, result, mem):
    """Aggregate per-strike delta contributions, emit single UL hedge order.

    Tracks our own hedge position separately from pegged UL position so we
    don't fight pegged's MM orders. Hedge budget comes from cfg["hedge_budget"]
    on the underlying (default 50).
    """
    hedge_out = {}
    reqs = mem.get("_skew_delta_req", {})
    if not reqs:
        return hedge_out

    # Group by underlying
    by_ul = {}
    for p, info in reqs.items():
        by_ul.setdefault(info["UL"], []).append(info)

    for UL, infos in by_ul.items():
        ul_od = state.order_depths.get(UL)
        if not ul_od or not ul_od.buy_orders or not ul_od.sell_orders:
            continue

        ul_cfg = config_map.get(UL, {})
        hedge_budget = ul_cfg.get("hedge_budget", 50)
        if hedge_budget <= 0:
            continue

        # Total delta exposure from all option strikes
        total_option_delta = sum(info["pos"] * info["delta"] for info in infos)
        # Target UL hedge contribution (opposite sign)
        target_hedge = -int(round(total_option_delta))
        # Clip to hedge budget
        target_hedge = max(-hedge_budget, min(hedge_budget, target_hedge))

        # Track our own hedge pos (separate from pegged MM)
        hedge_key = f"_skew_hedge_pos_{UL}"
        cur_hedge = mem.get(hedge_key, 0)
        adj = target_hedge - cur_hedge

        # Threshold to avoid thrashing on tiny delta changes
        hedge_min_adj = ul_cfg.get("hedge_min_adj", 3)
        if abs(adj) < hedge_min_adj:
            continue

        # Place crossing hedge at BBO (reliable fill)
        if adj > 0:
            px = min(ul_od.sell_orders)
        else:
            px = max(ul_od.buy_orders)

        hedge_out.setdefault(UL, []).append(Order(UL, px, adj))
        mem[hedge_key] = target_hedge  # optimistic: assume fills

    mem["_skew_delta_req"] = {}
    return hedge_out


def validate_orders(orders: List[Order], pos: int, limit: int) -> List[Order]:
    """Enforce IMC's position limit. Trim excess from widest passive quotes
    instead of nuking ALL orders (Frankfurt pattern: never waste a take)."""
    if not orders:
        return orders
    total_buy = sum(o.quantity for o in orders if o.quantity > 0)
    total_sell = sum(abs(o.quantity) for o in orders if o.quantity < 0)
    if pos + total_buy <= limit and pos - total_sell >= -limit:
        return orders  # all good

    # Separate aggressive takes (cross spread) from passive makes
    # Takes are orders that match existing book — higher priority
    # Heuristic: buys at higher prices and sells at lower prices are more aggressive
    buys = sorted([o for o in orders if o.quantity > 0], key=lambda o: -o.price)
    sells = sorted([o for o in orders if o.quantity < 0], key=lambda o: o.price)

    # Trim buys: keep most aggressive (highest price) first, trim widest
    kept = []
    cum_buy = 0
    cum_sell = 0
    for o in buys:
        can_add = min(o.quantity, limit - pos - cum_buy)
        if can_add > 0:
            kept.append(Order(o.symbol, o.price, can_add))
            cum_buy += can_add
    for o in sells:
        can_add = min(abs(o.quantity), limit + pos - cum_sell)
        if can_add > 0:
            kept.append(Order(o.symbol, o.price, -can_add))
            cum_sell += can_add

    return kept

# ============================================================
# TRADER CLASS (IMC interface)
# ============================================================

class Trader:
    def __init__(self):
        self._config = CONFIG
        self._mem = None  # backtest mode: keep mem in memory to skip JSON

    def bid(self):
        # R2 Market Access Fee. Top 50% wins +25% volume access.
        # R1 PnL: ASH 87.9K + IPR 95.5K = 183K. +25% volume ≈ +25-30K gain.
        # Break-even ≈ 25-30K. Bidding 10K = ~97% win prob, ~15K net EV.
        # Missing access = ~25K PnL loss = ~500 rank drop. Aggressive bid
        # trades 3K of EV for near-certain leaderboard position.
        return 10000

    def run(self, state: TradingState) -> tuple:
        # Restore memory from traderData
        if self._mem is not None:
            # Backtest mode: skip JSON round-trip (saves ~80s per 10K ticks)
            mem = self._mem
        else:
            mem = {}
            if state.traderData:
                try:
                    # Try compressed format first (Z: prefix)
                    if state.traderData.startswith("Z:"):
                        compressed = base64.b64decode(state.traderData[2:])
                        mem = json.loads(zlib.decompress(compressed).decode())
                    else:
                        mem = json.loads(state.traderData)
                except Exception as e:
                    print(f"ERR|traderData|{type(e).__name__}|{e}")
            if not isinstance(mem, dict):
                mem = {}

        result = {}
        total_conversions = 0

        # Store config reference so classifier can detect basket components
        mem["_config_ref"] = self._config

        # === TICK-0 CALIBRATION ===
        # First tick of each day: extract initial book state for calibration.
        # This is the purest signal about the exchange's starting conditions.
        tid = tick_in_day(state.timestamp, mem)
        if tid == 0:
            t0_data = {}
            for p, od_tmp in (state.order_depths or {}).items():
                if od_tmp.buy_orders and od_tmp.sell_orders:
                    bb = max(od_tmp.buy_orders)
                    ba = min(od_tmp.sell_orders)
                    t0_data[p] = {
                        "mid": (bb + ba) / 2.0,
                        "spread": ba - bb,
                        "bid_depth": sum(od_tmp.buy_orders.values()),
                        "ask_depth": sum(abs(v) for v in od_tmp.sell_orders.values()),
                    }
            mem["_tick0"] = t0_data
            # Print for server log intelligence
            if self._mem is None:  # production mode only
                print(f"T0|{json.dumps(t0_data)}")

        # === POSITION LIMIT AUTO-DETECTION ===
        # Extract limits from server listings (authoritative).
        # Also probe: if we submitted orders last tick and got ALL rejected,
        # our assumed limit might be wrong.
        if state.listings:
            for sym, listing in state.listings.items():
                lim = getattr(listing, "position_limit", None)
                if lim is not None:
                    if sym in self._config:
                        self._config[sym]["position_limit"] = lim
                    # Store discovered limits for future reference
                    mem[f"_limit_{sym}"] = lim
        # Note: position limits are ONLY set from state.listings (authoritative).
        # Do NOT infer limits from fill rejection patterns — no fills ≠ rejection.

        # Cross-product signal: compute all current mids and mid-changes
        # KELP-SQUID_INK correlation = -0.591 (P3 data). When one moves up, the other tends to move down.
        current_mids = {}
        for p, od_tmp in (state.order_depths or {}).items():
            if od_tmp.buy_orders and od_tmp.sell_orders:
                current_mids[p] = (max(od_tmp.buy_orders) + min(od_tmp.sell_orders)) / 2.0
        # Store mid changes for cross-product signal
        prev_mids = mem.get("_xprod_mids", {})
        mid_changes = {}
        for p, m in current_mids.items():
            if p in prev_mids:
                mid_changes[p] = m - prev_mids[p]
        mem["_xprod_mids"] = current_mids
        mem["_xprod_changes"] = mid_changes

        # Auto-discover negatively correlated pairs every 500 ticks
        xprod_hist = mem.get("_xprod_hist", {})
        for p, chg in mid_changes.items():
            if p not in xprod_hist:
                xprod_hist[p] = []
            xprod_hist[p].append(chg)
            if len(xprod_hist[p]) > 200:
                xprod_hist[p] = xprod_hist[p][-200:]
        mem["_xprod_hist"] = xprod_hist

        # Also track price levels (not just changes) for cointegration detection
        xprod_levels = mem.get("_xprod_levels", {})
        for p, m in current_mids.items():
            if p not in xprod_levels:
                xprod_levels[p] = []
            xprod_levels[p].append(round(m, 1))
            if len(xprod_levels[p]) > 200:
                xprod_levels[p] = xprod_levels[p][-200:]
        mem["_xprod_levels"] = xprod_levels

        xprod_tick = mem.get("_xprod_tick", 0) + 1
        mem["_xprod_tick"] = xprod_tick
        # Run correlation/cointegration at tick 100, then every 500
        xprod_due = (xprod_tick == 100) or (xprod_tick % 500 == 0)
        if xprod_due and len(xprod_hist) >= 2:
            # Compute pairwise correlations on CHANGES (for signal tilt)
            neg_pairs = {}
            products_with_hist = [p for p in xprod_hist if len(xprod_hist[p]) >= 100]
            for i, p1 in enumerate(products_with_hist):
                for p2 in products_with_hist[i+1:]:
                    h1 = xprod_hist[p1][-100:]
                    h2 = xprod_hist[p2][-100:]
                    n = min(len(h1), len(h2))
                    if n < 50:
                        continue
                    m1 = sum(h1[-n:]) / n
                    m2 = sum(h2[-n:]) / n
                    cov = sum((h1[-n+j] - m1) * (h2[-n+j] - m2) for j in range(n)) / n
                    v1 = sum((x - m1)**2 for x in h1[-n:]) / n
                    v2 = sum((x - m2)**2 for x in h2[-n:]) / n
                    if v1 > 0 and v2 > 0:
                        corr = cov / (v1**0.5 * v2**0.5)
                        if corr < -0.3:  # significant negative correlation
                            neg_pairs[p1] = p2
                            neg_pairs[p2] = p1
            mem["_xprod_pairs"] = neg_pairs

            # Detect cointegrated pairs from LEVELS (for pairs arb)
            # Two products are cointegrated if their prices move together (high R²)
            # and the spread between them mean-reverts
            products_with_levels = [p for p in xprod_levels if len(xprod_levels[p]) >= 80]
            # Exclude products already classified as basket/conversion/options
            # Also exclude basket COMPONENTS (they correlate via basket, not genuine pairs)
            # and options underlying (already serving options delta hedge role)
            exclude_types = {"basket_arb", "conversion_arb", "options", "generic_mm"}
            basket_components = set()
            for bcfg in self._config.values():
                if bcfg.get("type") == "basket_arb":
                    for comp in bcfg.get("components", {}):
                        basket_components.add(comp)
            candidates = [p for p in products_with_levels
                         if self._config.get(p, {}).get("type") not in exclude_types
                         and p not in basket_components]
            coint_pairs = mem.get("_coint_pairs", {})

            for i, p1 in enumerate(candidates):
                for p2 in candidates[i+1:]:
                    l1 = xprod_levels[p1][-100:]
                    l2 = xprod_levels[p2][-100:]
                    nn = min(len(l1), len(l2))
                    if nn < 100:
                        continue
                    # Correlation on levels
                    m1 = sum(l1[-nn:]) / nn
                    m2 = sum(l2[-nn:]) / nn
                    cov = sum((l1[-nn+j] - m1) * (l2[-nn+j] - m2) for j in range(nn)) / nn
                    v1 = sum((x - m1)**2 for x in l1[-nn:]) / nn
                    v2 = sum((x - m2)**2 for x in l2[-nn:]) / nn
                    if v1 > 0 and v2 > 0:
                        corr = cov / (v1**0.5 * v2**0.5)
                        if corr > 0.85:  # highly cointegrated (prices move together)
                            coint_pairs[p1] = p2
                            coint_pairs[p2] = p1
                            # Auto-promote to pairs_arb if currently generic_mm
                            for p in [p1, p2]:
                                partner = p2 if p == p1 else p1
                                current_type = self._config.get(p, {}).get("type")
                                if current_type in ("generic_mm", "ar_olivia"):
                                    lim = self._config.get(p, {}).get("position_limit", 50)
                                    self._config[p] = {
                                        "type": "pairs_arb",
                                        "position_limit": lim,
                                        "partner": partner,
                                        "partner_limit": self._config.get(partner, {}).get("position_limit", lim),
                                        "entry_z": 2.0,
                                        "exit_z": 0.3,
                                        "trade_size": min(10, lim),
                                        "spread_mean": 0.0,  # auto-discovered
                                    }
            mem["_coint_pairs"] = coint_pairs

        # === CROSS-PRODUCT INFORMED SIGNAL BUS ===
        # After individual strategies run detect_informed_bot, propagate signals
        # across products. We DON'T run detection here (would double-count).
        # Instead, we read the signals that strategies already computed.
        # This is done AFTER the main product loop below — see "Signal propagation" section.

        # Product states for structured state management
        product_states = {}

        # C1: pre-scan market trades for taker sentiment BEFORE strategies dispatch.
        # Strategies can read mem[f"tick_taker_sentiment_{product}"] to front-run
        # teams that only react to their own fills.
        precompute_taker_sentiment(state, mem)

        for product, od in (state.order_depths or {}).items():
            cfg = self._config.get(product)
            if cfg is None:
                # IMP 4: Auto-classify unknown products at runtime
                cfg = classify_product_live(od, product, mem, state)
                self._config[product] = cfg  # cache for future ticks
                # Override with server limit if available
                if state.listings and product in state.listings:
                    lim = getattr(state.listings[product], "position_limit", None)
                    if lim is not None:
                        cfg["position_limit"] = lim
            pos = state.position.get(product, 0)
            limit = cfg.get("position_limit", 50)
            mt = state.market_trades.get(product, [])
            ot = state.own_trades.get(product, [])
            ptype = cfg.get("type", "generic_mm")

            # Create/restore ProductState and compute all derived fields
            ps = ProductState.from_dict(product, mem.get(f"_ps_{product}", {}))
            # Set Kalman Q/R per product type before update
            kalman_qr = {"pegged": (0.1, 1.0), "ar_olivia": (1.0, 4.0),
                         "olivia_follow": (2.0, 8.0), "wide_spread": (1.5, 6.0)}
            qr = kalman_qr.get(ptype, (1.0, 4.0))
            ps._kalman_Q = cfg.get("kalman_Q", qr[0])
            ps._kalman_R = cfg.get("kalman_R", qr[1])
            ps.update(od, mt, ot, state.timestamp, pos, limit)
            # Sync TO mem so existing strategies can read from mem keys
            ps.sync_to_mem(mem)
            product_states[product] = ps

            hedge_orders = {}

            # END-OF-DAY: smart position management.
            # Last 100 ticks: flatten UNLESS we have a directional edge.
            # DRIFT PRODUCTS: NEVER flatten — holding = unrealized PnL.
            tid = tick_in_day(state.timestamp, mem)
            is_eod = tid >= 995000
            has_drift = cfg.get("fv_drift", 0) != 0 or (ps is not None and ps.is_drift_regime())
            if is_eod and pos != 0 and od.buy_orders and od.sell_orders and not has_drift:
                best_bid = max(od.buy_orders)
                best_ask = min(od.sell_orders)
                mid = (best_bid + best_ask) / 2.0
                # Check if we have a directional edge from recent momentum
                hist = mem.get(f"hist_{product}", [])
                eod_edge = 0  # positive = expect mid to go up
                if len(hist) >= 20:
                    # Short-term momentum: last 20 mids, normalized by vol
                    recent_drift = hist[-1] - hist[-20]
                    rv = ps.realized_vol if ps and ps.realized_vol > 0 else 1.0
                    eod_edge = recent_drift / rv  # normalize so threshold is comparable across products
                # Also check Olivia signal
                olivia = mem.get(f"olv_{product}")
                if olivia == "LONG":
                    eod_edge += ps.realized_vol * 0.5
                elif olivia == "SHORT":
                    eod_edge -= ps.realized_vol * 0.5

                # Decision: if our position ALIGNS with the edge, hold it
                # Position aligns if: pos > 0 and edge > 0, or pos < 0 and edge < 0
                pos_aligns = (pos > 0 and eod_edge > 0.5) or (pos < 0 and eod_edge < -0.5)

                if pos_aligns and tid < 998000:
                    # Edge aligns — keep position, continue normal strategy
                    pass  # fall through to normal strategy dispatch
                else:
                    # No edge or very last ticks — flatten.
                    # Default: aggressive single-tick dump.
                    # With cfg.ac_eod_unwind: Almgren-Chriss optimal slice (smooth schedule).
                    if cfg.get("ac_eod_unwind", False):
                        sigma_eod = ps.realized_vol if ps and ps.realized_vol > 0 else 1.0
                        ticks_left = max(1, DAY_PERIOD - tid)
                        slice_q = almgren_chriss_unwind(pos, ticks_left, sigma_eod,
                                                        eta=cfg.get("ac_eta", 0.01),
                                                        gamma=cfg.get("ac_gamma", 1e-4),
                                                        lam=cfg.get("ac_lam", 1e-5))
                        ac_sz = abs(slice_q)
                    else:
                        ac_sz = None
                    if pos > 0:
                        flat_sz = min(pos, limit + pos) if ac_sz is None else min(pos, ac_sz)
                        if flat_sz > 0:
                            orders = [Order(product, best_bid, -flat_sz)]
                            orders = validate_orders(orders, pos, limit)
                            result[product] = orders
                            ps.sync_from_mem(mem)
                            ps.fills_attempted += len(orders)
                            continue
                    else:
                        flat_sz = min(abs(pos), limit - pos) if ac_sz is None else min(abs(pos), ac_sz)
                        if flat_sz > 0:
                            orders = [Order(product, best_ask, +flat_sz)]
                            orders = validate_orders(orders, pos, limit)
                            result[product] = orders
                            ps.sync_from_mem(mem)
                            ps.fills_attempted += len(orders)
                            continue

            try:
                # Gap 4 (K3 timing-jitter): per-product action-gap gate for directional strategies.
                # cfg.min_action_gap=0 disables (default). cfg.min_action_gap=N suppresses ar_olivia
                # and olivia_follow until N timestamps have elapsed since the last non-empty submission.
                _action_gap = cfg.get("min_action_gap", 0)
                _last_action_ts = mem.get(f"_lat_{product}")
                _gap_blocked = (
                    ptype in ("ar_olivia", "olivia_follow")
                    and _action_gap > 0
                    and _last_action_ts is not None
                    and (state.timestamp - _last_action_ts) < _action_gap
                )
                if _gap_blocked:
                    orders = []
                elif ptype == "pegged":
                    orders, mem = run_pegged(product, cfg, od, pos, mem, mt, ot, ps=ps, ts=state.timestamp)
                elif ptype == "ar_olivia":
                    orders = []
                elif ptype == "olivia_follow":
                    orders = []
                elif ptype == "wide_spread":
                    orders = []
                elif ptype == "pairs_arb":
                    orders = []
                elif ptype == "basket_arb":
                    orders = []
                elif ptype == "conversion_arb":
                    orders = []
                elif ptype == "options":
                    orders = []
                elif ptype == "skew_mm":
                    orders, mem, hedge_orders = run_skew_mm(product, cfg, state, pos, mem)
                elif ptype == "mean_revert_mm":
                    orders, mem = run_mean_revert_mm(product, cfg, od, pos, mem, ts=state.timestamp)
                elif ptype == "voucher_intrinsic_mm":
                    orders, mem, hedge_orders = run_voucher_intrinsic_mm(product, cfg, state, pos, mem)
                elif ptype == "otm_short":
                    orders, mem = run_otm_short(product, cfg, od, pos, mem, state.timestamp)
                elif ptype == "voucher_chain_dir":
                    orders, mem, hedge_orders = run_voucher_chain_dir(product, cfg, state, pos, mem)
                elif ptype == "do_nothing":
                    orders = []  # Explicitly sit out — better than bad generic_mm trades
                else:
                    orders, mem = run_generic_mm(product, cfg, od, pos, mem, state.timestamp, mt, ot, ps=ps)
            except Exception as e:
                if self._mem is None:  # production: log to server log
                    print(f"ERR|{product}|{type(e).__name__}|{e}")
                orders = []

            # Sync FROM mem so ProductState captures any strategy updates
            ps.sync_from_mem(mem)
            ps.fills_attempted += len(orders)

            orders = validate_orders(orders, pos, limit)
            result[product] = orders
            # Gap 4: timestamp the directional submission for the next-tick action-gap check
            if orders and ptype in ("ar_olivia", "olivia_follow"):
                mem[f"_lat_{product}"] = state.timestamp
            # K4 trade log emission (no-op unless IMC_TRADE_LOG_PATH env set)
            if orders:
                _emit_trade_log(state.timestamp, product, ptype,
                                {"olv": mem.get(f"olv_{product}"),
                                 "tox": getattr(ps, "tox_adj", 1.0),
                                 "pos": pos,
                                 "mo10": getattr(ps, "markout_10", 0.0)},
                                orders)

            # Route hedge orders to their respective products
            for hedge_product, h_orders in hedge_orders.items():
                if hedge_product not in result:
                    result[hedge_product] = []
                result[hedge_product].extend(h_orders)

        # SKEW MM delta-hedge aggregator: collects per-strike delta contributions
        # recorded during the main loop and emits a single UL hedge order per
        # underlying. Runs after main loop so it sees all strikes' requests.
        if any(c.get("type") in ("skew_mm", "voucher_intrinsic_mm") for c in self._config.values()):
            skew_hedge = run_skew_delta_agg(self._config, state, result, mem)
            for hp, ho in skew_hedge.items():
                if hp not in result:
                    result[hp] = []
                result[hp].extend(ho)

        # Validate hedge order products too
        for product in list(result.keys()):
            pos = state.position.get(product, 0)
            cfg = self._config.get(product, {})
            limit = cfg.get("position_limit", 50)
            result[product] = validate_orders(result[product], pos, limit)

        # === B3: MARKOUT BUFFER (passive observation) ===
        # Record every fill with current mid so markout_score() can measure
        # adverse selection post-hoc. Gated consumers read `cfg.get("markout_sizedown")`.
        for product in (state.own_trades or {}):
            od_tmp = (state.order_depths or {}).get(product)
            cur_mid = None
            if od_tmp and od_tmp.buy_orders and od_tmp.sell_orders:
                cur_mid = (max(od_tmp.buy_orders) + min(od_tmp.sell_orders)) / 2.0
            update_markout_buffer(mem, product, state.own_trades.get(product, []), cur_mid)

        # === C9: VPIN BUFFER (passive observation) ===
        # Stream signed flow per product. Gated consumers read vpin_score().
        for product in (state.order_depths or {}):
            vpin_update(mem, product,
                        state.own_trades.get(product, []) if state.own_trades else [],
                        state.market_trades.get(product, []) if state.market_trades else [])

        # === Gap 5: per-fill K4 attribution emission ===
        # No-op unless IMC_TRADE_LOG_PATH is set. Snapshots mid/obi/vpin/garch_sigma/pos at fill.
        if _TRADE_LOG_PATH and state.own_trades:
            for product, fills in state.own_trades.items():
                if not fills:
                    continue
                cfg = self._config.get(product, {})
                ptype = cfg.get("type", "generic_mm")
                od_tmp = (state.order_depths or {}).get(product)
                mid_f = None
                obi_f = None
                if od_tmp and od_tmp.buy_orders and od_tmp.sell_orders:
                    mid_f = (max(od_tmp.buy_orders) + min(od_tmp.sell_orders)) / 2.0
                    try:
                        obi_f = compute_obi(od_tmp, levels=3)
                    except Exception:
                        obi_f = None
                try:
                    vpin_f = vpin_score(mem, product)
                except Exception:
                    vpin_f = None
                garch_sigma_f = mem.get(f"garch_sigma_{product}")
                feat = {"mid": mid_f, "obi": obi_f, "vpin": vpin_f,
                        "garch_sigma": garch_sigma_f,
                        "pos": state.position.get(product, 0),
                        "olv": mem.get(f"olv_{product}")}
                _emit_fill_log(state.timestamp, product, ptype, fills, feat)

        # === CROSS-PRODUCT SIGNAL PROPAGATION ===
        # After all strategies ran, collect informed signals and share them.
        # Strategies already computed signals via check_olivia/detect_informed_bot.
        # We just read the results and propagate to basket products.
        informed_signals = {}
        for p in (state.order_depths or {}):
            sig = mem.get(f"olv_{p}")  # olivia signal set by strategies
            if sig:
                informed_signals[p] = sig
        mem["_informed_signals"] = informed_signals
        # Propagate to baskets: if a component has an informed signal, pass to basket
        for bcfg_product, bcfg in self._config.items():
            if bcfg.get("type") != "basket_arb":
                continue
            for comp in bcfg.get("components", {}):
                if comp in informed_signals:
                    mem[f"_basket_informed_{bcfg_product}"] = informed_signals[comp]
                    break

        # Persist ProductState for each product
        for product, ps in product_states.items():
            mem[f"_ps_{product}"] = ps.to_dict()

        # Diagnostic print: every 100 ticks, compact state to stdout (readable in server logs)
        if self._mem is None and state.timestamp % 10000 == 0:
            for p in sorted(result.keys()):
                ps_d = mem.get(f"_ps_{p}", {})
                print(f"D|{state.timestamp}|{p}|{self._config.get(p,{}).get('type','?')}|{state.position.get(p,0)}|{ps_d.get('realized_vol',0):.1f}|{mem.get(f'olv_{p}','-')}")

        # Backtest mode: skip JSON entirely, keep mem in memory
        if self._mem is not None:
            self._mem = mem
            return result, total_conversions, ""

        # Production mode: serialize to traderData string
        # Memory pruning: prevent traderData from exceeding Lambda limits
        MAX_MEM_BYTES = 90000

        # First pass: scrub NaN/Inf + round floats to save bytes
        for k in list(mem.keys()):
            v = mem[k]
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    mem[k] = 0.0
                else:
                    mem[k] = round(v, 4)
            elif isinstance(v, list):
                mem[k] = [round(x, 2) if isinstance(x, float) and not (math.isnan(x) or math.isinf(x))
                          else (0.0 if isinstance(x, float) else x) for x in v]

        # Remove transient keys before serialization
        mem.pop("_config_ref", None)

        try:
            json_str = json.dumps(mem)
            # Compress with zlib + base64 (3-4x size reduction)
            compressed = base64.b64encode(zlib.compress(json_str.encode(), 9)).decode()
            trader_data = "Z:" + compressed

            if len(trader_data) > MAX_MEM_BYTES:
                # A5: whitelist keys that carry long-horizon signal (markout, x-product
                # mids, mid_hist). These get 500-entry cap in the first pruning pass;
                # all other lists drop to 50. If still too large, subsequent passes
                # tighten whitelisted caps then fall back to the legacy 50-cap.
                LONG_HORIZON_PREFIXES = ("hist_", "xprod_mids", "markout_",
                                         "mid_hist_", "_te_buf_", "_vpin_")
                def _is_long_horizon(key):
                    return any(key.startswith(p) for p in LONG_HORIZON_PREFIXES)

                # Pass 1: 500 cap for whitelisted, 50 for everything else.
                for k in list(mem.keys()):
                    v = mem[k]
                    if isinstance(v, list):
                        cap = 500 if _is_long_horizon(k) else 50
                        if len(v) > cap:
                            mem[k] = v[-cap:]
                    elif isinstance(v, dict) and k.startswith("_bdisc_"):
                        for sk in list(v.get("hist", {}).keys()):
                            if isinstance(v["hist"][sk], list) and len(v["hist"][sk]) > 80:
                                v["hist"][sk] = v["hist"][sk][-80:]
                    elif k.startswith("opt_") or k.startswith("dev_hist_"):
                        if isinstance(v, list) and len(v) > 30:
                            mem[k] = v[-30:]
                json_str = json.dumps(mem)
                compressed = base64.b64encode(zlib.compress(json_str.encode(), 9)).decode()
                trader_data = "Z:" + compressed

                # Pass 2: tighten whitelisted cap to 200 if still over.
                if len(trader_data) > MAX_MEM_BYTES:
                    for k in list(mem.keys()):
                        if isinstance(mem[k], list) and _is_long_horizon(k) and len(mem[k]) > 200:
                            mem[k] = mem[k][-200:]
                    json_str = json.dumps(mem)
                    compressed = base64.b64encode(zlib.compress(json_str.encode(), 9)).decode()
                    trader_data = "Z:" + compressed

                # Pass 3: drop bot detection + force all lists to 50.
                if len(trader_data) > MAX_MEM_BYTES:
                    for k in list(mem.keys()):
                        if k.startswith("_bot_det_"):
                            del mem[k]
                        elif isinstance(mem.get(k), list) and len(mem[k]) > 50:
                            mem[k] = mem[k][-50:]
                    json_str = json.dumps(mem)
                    compressed = base64.b64encode(zlib.compress(json_str.encode(), 9)).decode()
                    trader_data = "Z:" + compressed
        except Exception:
            trader_data = "{}"

        return result, total_conversions, trader_data
