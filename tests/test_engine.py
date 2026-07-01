"""Unit tests for the trading engine's core logic.

Covers the components whose correctness is load-bearing for live trading:
  - position-limit enforcement (must never let the bot breach a limit)
  - rolling-statistics / z-score math (the basis of the R5 mean-reversion alpha)
  - order-book best-bid/ask extraction
  - Black-Scholes pricing and Greeks (the R3/R4 options strategies)

Run:  pytest -q
"""
import os
import sys
import math

import pytest

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "engine")
sys.path.insert(0, ENGINE)

import datamodel as dm
import trader_r5 as tr5


# ---------------------------------------------------------------------------
# Position-limit enforcement (trader_r5.validate_orders)
# ---------------------------------------------------------------------------

def test_validate_orders_caps_buys_at_limit():
    # Flat position, limit 10, try to buy 25 -> must be trimmed to 10.
    orders = [dm.Order("X", 100, 25)]
    out = tr5.validate_orders(orders, pos=0, limit=10)
    assert sum(o.quantity for o in out) == 10


def test_validate_orders_caps_sells_at_limit():
    orders = [dm.Order("X", 100, -25)]
    out = tr5.validate_orders(orders, pos=0, limit=10)
    assert sum(o.quantity for o in out) == -10


def test_validate_orders_respects_existing_position():
    # Already long 7, limit 10 -> can only buy 3 more.
    orders = [dm.Order("X", 100, 10)]
    out = tr5.validate_orders(orders, pos=7, limit=10)
    assert sum(o.quantity for o in out) == 3


def test_validate_orders_keeps_aggressive_first():
    # Two buys, only room for one: the higher-priced (more aggressive) survives.
    orders = [dm.Order("X", 100, 8), dm.Order("X", 105, 8)]
    out = tr5.validate_orders(orders, pos=0, limit=10)
    total = sum(o.quantity for o in out)
    assert total == 10
    # the 105 order should be fully present
    assert any(o.price == 105 and o.quantity == 8 for o in out)


def test_validate_orders_empty():
    assert tr5.validate_orders([], pos=0, limit=10) == []


# ---------------------------------------------------------------------------
# Rolling statistics / z-score (trader_r5.update_rolling_stats)
# ---------------------------------------------------------------------------

def test_rolling_stats_warmup_returns_none():
    mem = {}
    m, sd = tr5.update_rolling_stats(mem, "k", 100.0, window=500)
    assert m is None and sd is None  # not enough samples yet


def test_rolling_stats_mean_and_std():
    mem = {}
    window = 40
    # feed a known constant then a known spread
    vals = [10.0] * 30 + [20.0] * 30
    m = sd = None
    for v in vals:
        m, sd = tr5.update_rolling_stats(mem, "k", v, window=window)
    # after 60 samples, buffer holds the last 40 -> 10 tens and 30 twenties
    assert m is not None
    expected_mean = (10 * 10.0 + 30 * 20.0) / 40
    assert abs(m - expected_mean) < 1e-6
    assert sd > 0


def test_rolling_stats_buffer_capped_at_window():
    mem = {}
    for v in range(1000):
        tr5.update_rolling_stats(mem, "k", float(v), window=100)
    assert len(mem["k_buf"]) == 100


# ---------------------------------------------------------------------------
# Order-book extraction (trader_r5.best_bid_ask)
# ---------------------------------------------------------------------------

def test_best_bid_ask_normal():
    od = dm.OrderDepth()
    od.buy_orders = {99: 5, 98: 10}
    od.sell_orders = {101: -5, 102: -10}
    bb, ba, mid = tr5.best_bid_ask(od)
    assert bb == 99 and ba == 101 and mid == 100.0


def test_best_bid_ask_empty_book():
    od = dm.OrderDepth()
    od.buy_orders = {}
    od.sell_orders = {}
    bb, ba, mid = tr5.best_bid_ask(od)
    assert bb is None and ba is None and mid is None


# ---------------------------------------------------------------------------
# Black-Scholes pricing and Greeks (trader.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trader_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("trader_main", os.path.join(ENGINE, "trader.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_norm_cdf_known_values(trader_mod):
    assert abs(trader_mod._norm_cdf(0.0) - 0.5) < 1e-6
    assert trader_mod._norm_cdf(5.0) > 0.999
    assert trader_mod._norm_cdf(-5.0) < 0.001


def test_bs_call_atm_positive(trader_mod):
    # ATM call with positive time and vol must have positive value below spot.
    price = trader_mod.bs_call(S=100, K=100, T=0.25, sigma=0.5)
    assert 0 < price < 100


def test_bs_call_monotonic_in_vol(trader_mod):
    # Call price increases with volatility (positive vega).
    low = trader_mod.bs_call(S=100, K=100, T=0.25, sigma=0.2)
    high = trader_mod.bs_call(S=100, K=100, T=0.25, sigma=0.6)
    assert high > low


def test_bs_delta_bounds(trader_mod):
    # Call delta is in (0, 1); deep ITM -> near 1, deep OTM -> near 0.
    d_itm = trader_mod.bs_delta(S=150, K=100, T=0.25, sigma=0.3)
    d_otm = trader_mod.bs_delta(S=50, K=100, T=0.25, sigma=0.3)
    assert 0.0 < d_otm < d_itm < 1.0
    assert d_itm > 0.9 and d_otm < 0.1


def test_bs_vega_positive(trader_mod):
    assert trader_mod.bs_vega(S=100, K=100, T=0.25, sigma=0.3) > 0


# ---------------------------------------------------------------------------
# Engine smoke test: a full tick must not crash on an empty/odd book
# ---------------------------------------------------------------------------

def _make_state(traderData=""):
    state = dm.TradingState.__new__(dm.TradingState)
    state.traderData = traderData
    state.timestamp = 0
    state.listings = {}
    state.order_depths = {"PEBBLES_XL": dm.OrderDepth()}
    state.order_depths["PEBBLES_XL"].buy_orders = {}
    state.order_depths["PEBBLES_XL"].sell_orders = {}
    state.own_trades = {}
    state.market_trades = {}
    state.position = {}
    state.observations = None
    return state


def test_engine_handles_empty_book_without_crashing():
    trader = tr5.Trader()
    result, conversions, traderData = trader.run(_make_state())
    assert isinstance(result, dict)
    assert isinstance(traderData, str)


def test_engine_state_roundtrips():
    trader = tr5.Trader()
    _, _, td1 = trader.run(_make_state())
    # feeding the produced state back in must not crash
    _, _, td2 = trader.run(_make_state(td1))
    assert isinstance(td2, str)


# ---------------------------------------------------------------------------
# Momentum filter (trader_r5.run_single_mr) — the R5 "key robustness fix"
# per docs/06-round5.md: skip new mean-reversion entries during a strong
# trend, since mean-reversion bleeds money when the price is trending
# rather than oscillating. This was previously untested.
# ---------------------------------------------------------------------------

class _FakeState:
    def __init__(self, mid):
        od = dm.OrderDepth()
        od.buy_orders = {mid - 0.5: 50}
        od.sell_orders = {mid + 0.5: -50}
        self.order_depths = {"X": od}
        self.position = {}


def _feed_calm_history(mem, result, ticks, window, mom_window, cfg, start=100.0, spread=0.1):
    """Feed alternating small oscillations to build up rolling-stats history
    with a small, known standard deviation, without triggering any entry."""
    for i in range(ticks):
        mid = start + (spread if i % 2 == 0 else -spread)
        tr5.run_single_mr("X", cfg, _FakeState(mid), mem, result)


def test_momentum_filter_blocks_entry_during_strong_trend():
    cfg = {"window": 50, "z_in": 1.5, "z_out": 0.3, "max_pos": 10,
           "mom_window": 20, "mom_thresh": 1.0}
    mem, result = {}, {}
    _feed_calm_history(mem, result, 60, cfg["window"], cfg["mom_window"], cfg)
    assert mem.get("sm_X_state", 0) == 0  # calm history must not have entered

    # A jump large enough to exceed both z_in (amplified by the small sd
    # from the calm history) and mom_thresh (measured against sd clamped
    # to a floor of 1.0, so a jump of ~2.0 clears a mom_thresh of 1.0).
    tr5.run_single_mr("X", cfg, _FakeState(102.0), mem, result)

    assert mem["sm_X_state"] == 0, "momentum filter should have blocked the entry"
    assert "X" not in result


def test_small_jump_without_momentum_still_enters():
    """Control case: a jump big enough to trip z_in but too small to trip
    mom_thresh must still be allowed to enter (proves the filter is
    momentum-specific, not just suppressing all large-z entries)."""
    cfg = {"window": 50, "z_in": 1.5, "z_out": 0.3, "max_pos": 10,
           "mom_window": 20, "mom_thresh": 1.0}
    mem, result = {}, {}
    _feed_calm_history(mem, result, 60, cfg["window"], cfg["mom_window"], cfg)

    tr5.run_single_mr("X", cfg, _FakeState(100.5), mem, result)

    assert mem["sm_X_state"] != 0, "a small jump under mom_thresh should not be filtered"
    assert "X" in result
