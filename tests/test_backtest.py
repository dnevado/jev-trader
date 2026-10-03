import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from jevbt.backtest.engine import buy_and_hold, rebalance_dates, run_backtest
from jevbt.backtest.metrics import cagr, max_drawdown, sharpe, summarize
from jevbt.decision.jev import JevDecider, MockJevClassifier
from jevbt.features.fundamentals import compute_ratios
from jevbt.graph import build_graph
from jevbt.llm.summarizer import MockSummaryLLM, Summarizer
from jevbt.strategy import BaselineStrategy, JevRules, JevStrategy


class ScriptedStrategy:
    """Target weights by date; records the date of the data each decision saw."""

    name = "scripted"

    def __init__(self, targets):
        self.targets = {pd.Timestamp(k): v for k, v in targets.items()}
        self.seen = []

    def target(self, ticker, t, technical, current_weight):
        self.seen.append((t, technical.name))
        return self.targets.get(t, current_weight), {"reason": "scripted"}


def flat_prices(n=30, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n, name="date")
    close = pd.Series(np.linspace(100, 129, n), index=idx)
    return pd.DataFrame({"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close, "volume": 1}, index=idx)


def test_rebalance_dates_first_session_of_week():
    cal = pd.bdate_range("2024-01-01", "2024-01-19").delete(0)  # Monday Jan 1 is a holiday
    assert list(rebalance_dates(cal)) == [pd.Timestamp(d) for d in ("2024-01-02", "2024-01-08", "2024-01-15")]


def test_execution_at_open_with_previous_day_data_and_costs():
    px = flat_prices()
    strat = ScriptedStrategy({"2024-01-08": 1.0})
    res = run_backtest({"X": px}, strat, "2024-01-02", "2024-02-09", initial_cash=10_000, cost_bps=10)
    assert all(data_date < t for t, data_date in strat.seen)
    assert all(data_date == px.index[px.index.get_loc(t) - 1] for t, data_date in strat.seen)
    trade = res.trades.iloc[0]
    assert trade["date"] == pd.Timestamp("2024-01-08")
    assert trade["price"] == px.at[pd.Timestamp("2024-01-08"), "open"]
    assert trade["shares"] * trade["price"] == pytest.approx(10_000)
    assert trade["cost"] == pytest.approx(10.0)
    # Before the trade the equity is flat; on the trade day it is marked at the close, minus costs.
    assert res.equity[pd.Timestamp("2024-01-05")] == pytest.approx(10_000)
    day = pd.Timestamp("2024-01-08")
    assert res.equity[day] == pytest.approx(trade["shares"] * px.at[day, "close"] - 10.0)


def test_round_trip_pnl_and_hit_rate():
    px = flat_prices()
    strat = ScriptedStrategy({"2024-01-08": 0.5, "2024-01-22": 0.0})
    res = run_backtest({"X": px}, strat, "2024-01-02", "2024-02-09", initial_cash=10_000, cost_bps=0)
    buy_px, sell_px = px.at[pd.Timestamp("2024-01-08"), "open"], px.at[pd.Timestamp("2024-01-22"), "open"]
    assert len(res.trades) == 2 and res.trades.iloc[1]["side"] == "sell"
    assert res.round_trip_pnl == [pytest.approx(5_000 / buy_px * (sell_px - buy_px))]
    m = summarize(res.equity, res.trades, res.round_trip_pnl)
    assert m["trades"] == 2 and m["hit_rate"] == 1.0
    assert res.equity.iloc[-1] == pytest.approx(10_000 + res.round_trip_pnl[0])


def test_metrics_known_values():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01"])
    assert cagr(pd.Series([100.0, 110.0], index=idx)) == pytest.approx(0.10, abs=1e-3)
    assert max_drawdown(pd.Series([100.0, 120.0, 60.0, 130.0])) == pytest.approx(-0.5)
    assert np.isnan(sharpe(pd.Series([100.0, 100.0, 100.0])))
    up = pd.Series(100 * 1.001 ** np.arange(300) * (1 + 0.001 * np.sin(np.arange(300))))
    assert sharpe(up) > 0


def test_buy_and_hold():
    px = flat_prices()
    bh = buy_and_hold({"X": px}, "2024-01-02", "2024-02-09", initial_cash=10_000, cost_bps=0)
    first = pd.Timestamp("2024-01-02")
    assert bh.iloc[-1] == pytest.approx(10_000 / px.at[first, "open"] * px["close"].iloc[-1])


def test_baseline_runs_on_synthetic_data():
    prices = {"A": make_ohlcv(600, seed=1), "B": make_ohlcv(600, seed=2)}
    res = run_backtest(prices, BaselineStrategy(max_alloc=0.5), "2023-12-01", "2025-04-30")
    assert res.equity.notna().all() and (res.equity > 0).all()
    assert len(res.trades) > 0
    held = res.trades.groupby("ticker")["shares"].sum()
    assert (held >= -1e-9).all()  # long only


def synthetic_ratios(first_known="2023-06-01"):
    idx = pd.to_datetime(["2022-12-31", "2023-12-31"]).rename("date")
    acc = pd.to_datetime([first_known, "2024-02-01"])
    income = pd.DataFrame({"fiscal_year": ["2022", "2023"], "accepted_date": acc, "revenue": [100.0, 120.0],
                           "gross_profit": [40.0, 50.0], "operating_income": [20.0, 25.0],
                           "net_income": [15.0, 18.0], "eps_diluted": [1.0, 1.2]}, index=idx)
    balance = pd.DataFrame({"accepted_date": acc, "total_debt": [10.0, 10.0],
                            "total_stockholders_equity": [50.0, 60.0]}, index=idx)
    cashflow = pd.DataFrame({"accepted_date": acc, "free_cash_flow": [10.0, 12.0]}, index=idx)
    return compute_ratios(income, balance, cashflow)


def test_jev_strategy_end_to_end_with_mocks(tmp_path):
    prices = {"AAPL": make_ohlcv(600, seed=3)}
    summarizer = Summarizer(MockSummaryLLM(), "mock", tmp_path)
    decider = JevDecider(MockJevClassifier(), tmp_path)
    strat = JevStrategy(build_graph(summarizer, decider), {"AAPL": synthetic_ratios()}, max_alloc=1.0)
    log = tmp_path / "decisions.jsonl"
    res = run_backtest(prices, strat, "2023-11-01", "2025-04-30", log_path=log)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records and all("state" in r for r in records if r["reason"] != "warm-up")
    # Anonymized state: no ticker, no dates; fundamentals only once published.
    assert not any("AAPL" in r.get("state", "") for r in records)
    assert not any("2024-" in r.get("state", "") for r in records)
    before = [r for r in records if r["date"] < "2024-02-02" and "state" in r]
    after = [r for r in records if r["date"] > "2024-02-02" and "state" in r]
    assert before and all("FY2023" not in r["state"] for r in before)
    assert after and all("FY2023" in r["state"] for r in after)
    # Summaries are cached per filing: 2 fiscal-year windows → 2 summarizer calls.
    assert summarizer.calls == 2
    assert decider.calls <= len(records)
    for r in records:
        if r["order"] and r["order"]["side"] == "buy":
            j = r["jev"]
            assert j["action"] == "buy" and j["trend_up"] >= 0.75 and j["fundamental_quality"] >= 1.5
    assert res.equity.notna().all()


def test_jev_rules_sizing(tmp_path):
    prices = {"X": make_ohlcv(600, seed=3)}
    graph = build_graph(Summarizer(MockSummaryLLM(), "mock", tmp_path), JevDecider(MockJevClassifier(), tmp_path))
    strict = JevStrategy(graph, {"X": synthetic_ratios()}, rules=JevRules(min_confidence=0.99))
    res = run_backtest(prices, strict, "2023-11-01", "2025-04-30")
    assert res.trades.empty  # mock confidence 0.8 never passes 0.99


# ---------- phase 2b: signals entry mode, liquidation, walk-forward ----------

from jevbt.backtest.walkforward import default_grid, make_folds, walk_forward  # noqa: E402


def test_liquidate_at_end_closes_positions():
    px = flat_prices()
    res = run_backtest({"X": px}, ScriptedStrategy({"2024-01-08": 1.0}), "2024-01-02", "2024-02-09",
                       initial_cash=10_000, cost_bps=10, liquidate_at_end=True)
    last = res.trades.iloc[-1]
    assert last["side"] == "sell" and last["date"] == px.index[px.index <= "2024-02-09"][-1]
    assert len(res.round_trip_pnl) == 1
    assert res.equity.iloc[-1] == pytest.approx(10_000 + res.round_trip_pnl[0])


def test_signals_mode_enters_where_action_mode_does_not(tmp_path):
    class HoldingJev(MockJevClassifier):
        """Bullish signals but always 'hold': the case seen with AMZN."""

        def invoke(self, state):
            r = super().invoke(state)
            action = r.answers["action"].model_copy(update={"choice": "hold",
                                                            "probabilities": {"buy": 0.3, "hold": 0.6, "sell": 0.1}})
            return r.model_copy(update={"answers": {**r.answers, "action": action}})

    prices = {"X": make_ohlcv(600, seed=3)}
    graph = build_graph(Summarizer(MockSummaryLLM(), "mock", tmp_path), JevDecider(HoldingJev(), tmp_path))
    action = run_backtest(prices, JevStrategy(graph, {"X": synthetic_ratios()}, JevRules(entry_mode="action")),
                          "2023-11-01", "2025-04-30")
    signals = run_backtest(prices, JevStrategy(graph, {"X": synthetic_ratios()}, JevRules(entry_mode="signals")),
                           "2023-11-01", "2025-04-30")
    assert action.trades.empty
    assert not signals.trades.empty


def test_make_folds():
    folds = make_folds("2023-01-01", "2024-02-15", train_months=12, test_months=1)
    assert folds[0] == (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"),
                        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))
    assert folds[-1][2:] == (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-15"))
    assert make_folds("2023-01-01", "2023-06-01", 12, 3) == []


def test_walk_forward_is_out_of_sample_and_calls_jev_once_per_week(tmp_path):
    prices = {"X": make_ohlcv(700, seed=3)}
    decider = JevDecider(MockJevClassifier(), tmp_path)
    strat = JevStrategy(build_graph(Summarizer(MockSummaryLLM(), "mock", tmp_path), decider),
                        {"X": synthetic_ratios()}, max_alloc=1.0)
    grid = default_grid()[:4] + default_grid()[-4:]
    wf = walk_forward(prices, strat, "2023-11-01", "2025-08-29", grid=grid, train_months=6, test_months=3)
    assert len(wf.folds) >= 5
    for f in wf.folds:
        assert f.train_end < f.test_start and f.rules in grid
    assert wf.equity.index.min() >= wf.folds[0].test_start and wf.equity.index.is_monotonic_increasing
    weeks = len(rebalance_dates(pd.bdate_range("2023-11-01", "2025-08-29")))
    # Memoized across the whole grid search: at most one call per week, plus the first session of
    # windows that start mid-week (train and test starts).
    assert decider.calls <= weeks + 2 * len(wf.folds)
    assert len(wf.round_trip_pnl) == int((wf.trades["side"] == "sell").sum())


def test_rank_prefers_enough_trades_over_lucky_sharpe():
    from jevbt.backtest.walkforward import _rank

    lucky = {"sharpe": 2.7, "trades": 2}
    active = {"sharpe": 0.9, "trades": 12}
    better = {"sharpe": 1.2, "trades": 10}
    flat = {"sharpe": float("nan"), "trades": 0}
    assert _rank(active, 10) > _rank(lucky, 10)
    assert _rank(better, 10) > _rank(active, 10)
    # Nobody reaches the minimum → the most active wins; a flat rule set is last.
    assert _rank({"sharpe": 0.1, "trades": 6}, 10) > _rank(lucky, 10) > _rank(flat, 10)


def test_overbought_gate_and_sell_flags():
    from types import SimpleNamespace

    class OneShotGraph:
        def __init__(self, **answers):
            self.answers = answers

        def invoke(self, state):
            d = SimpleNamespace(trend_up=0.9, overbought=0.9, fundamental_quality=2.0, valuation_risk=0.0,
                                action="sell", action_confidence=0.5, p_buy=0.1)
            d.__dict__.update(self.answers)
            d.model_dump = lambda: dict(d.__dict__)
            return {"decision": d, "state_text": "s"}

    tech = pd.Series({"sma200": 1.0})
    t = pd.Timestamp("2024-01-08")

    def target(graph, weight=0.0, **flags):
        return JevStrategy(graph, {"X": None}, JevRules(entry_mode="signals", **flags), max_alloc=1.0).target(
            "X", t, tech, weight)[0]

    overbought_sell = OneShotGraph()
    assert target(overbought_sell) == 0.0                                       # both gates block
    assert target(overbought_sell, use_overbought=False) == 0.0                 # sell veto still blocks
    assert target(overbought_sell, sell_veto=False) == 0.0                      # overbought still blocks
    assert target(overbought_sell, use_overbought=False, sell_veto=False) == pytest.approx(0.9)
    assert target(overbought_sell, weight=0.5) == 0.0                           # exit on sell
    assert target(overbought_sell, weight=0.5, exit_on_sell=False) == 0.5       # hold instead


def test_default_grid_without_overbought_drops_duplicates():
    grid = default_grid(use_overbought=False, sell_veto=False)
    assert len(grid) == 16 and len(default_grid()) == 32
    assert all(not r.use_overbought and not r.sell_veto for r in grid)


def test_default_grid_fixed_values_replace_dimensions():
    grid = default_grid(min_quality=0.0, valuation_penalty=0.0)
    assert len(grid) == 16
    assert {r.min_quality for r in grid} == {0.0} and {r.valuation_penalty for r in grid} == {0.0}


# ---------- shorts ----------

def test_short_round_trip_pnl_costs_and_borrow():
    px = flat_prices()  # rising prices: a short loses
    strat = ScriptedStrategy({"2024-01-08": -1.0, "2024-01-22": 0.0})
    res = run_backtest({"X": px}, strat, "2024-01-02", "2024-02-09", initial_cash=10_000, cost_bps=10, borrow_bps=0)
    open_px, close_px = px.at[pd.Timestamp("2024-01-08"), "open"], px.at[pd.Timestamp("2024-01-22"), "open"]
    n = 10_000 / open_px
    expected = n * (open_px - close_px) - 10.0 - n * close_px * 10 / 1e4
    assert list(res.trades["side"]) == ["sell", "buy"]
    assert res.round_trip_pnl == [pytest.approx(expected)]
    assert expected < 0
    assert res.equity.iloc[-1] == pytest.approx(10_000 + expected)
    # The borrow fee only costs money while short.
    with_fee = run_backtest({"X": px}, ScriptedStrategy({"2024-01-08": -1.0, "2024-01-22": 0.0}),
                            "2024-01-02", "2024-02-09", initial_cash=10_000, cost_bps=10, borrow_bps=100)
    fee = res.equity.iloc[-1] - with_fee.equity.iloc[-1]
    assert 0 < fee < 10_000 * 0.01 * 15 / 252 * 1.2
    long_only = run_backtest({"X": px}, ScriptedStrategy({"2024-01-08": 1.0}), "2024-01-02", "2024-02-09",
                             initial_cash=10_000, cost_bps=10, borrow_bps=100)
    no_fee = run_backtest({"X": px}, ScriptedStrategy({"2024-01-08": 1.0}), "2024-01-02", "2024-02-09",
                          initial_cash=10_000, cost_bps=10, borrow_bps=0)
    assert long_only.equity.iloc[-1] == pytest.approx(no_fee.equity.iloc[-1])


def test_flip_long_to_short_is_close_then_open():
    px = flat_prices()
    res = run_backtest({"X": px}, ScriptedStrategy({"2024-01-08": 1.0, "2024-01-15": -1.0}), "2024-01-02", "2024-02-09",
                       initial_cash=10_000, cost_bps=0, liquidate_at_end=True, borrow_bps=0)
    flip =res.trades[res.trades["date"] == pd.Timestamp("2024-01-15")]
    assert list(flip["side"]) == ["sell", "sell"]
    assert len(res.round_trip_pnl) == 2  # the long, then the short closed by the liquidation
    assert res.round_trip_pnl[0] > 0 > res.round_trip_pnl[1]
    assert res.equity.iloc[-1] == pytest.approx(10_000 + sum(res.round_trip_pnl))


def _tech(close, sma200, rsi):
    return pd.Series({"close": close, "sma200": sma200, "rsi14": rsi})


def test_baseline_directions():
    long_, short, both = (BaselineStrategy(max_alloc=0.5, direction=d) for d in ("long", "short", "both"))
    down, up = _tech(90, 100, 50), _tech(110, 100, 50)
    assert long_.target("X", None, down, 0.0)[0] == 0.0
    assert short.target("X", None, down, 0.0)[0] == -0.5
    assert short.target("X", None, _tech(90, 100, 25), 0.0)[0] == 0.0     # oversold: no new short
    assert short.target("X", None, down, -0.5)[0] == -0.5                 # hold short
    assert short.target("X", None, up, -0.5)[0] == 0.0                    # cover
    assert short.target("X", None, up, 0.0)[0] == 0.0                     # never long
    assert both.target("X", None, down, 0.5)[0] == -0.5                   # flip to short
    assert both.target("X", None, up, -0.5)[0] == 0.5                     # flip to long
    assert long_.target("X", None, up, 0.0)[0] == 0.5


def test_jev_short_mirrors_long_rules():
    from types import SimpleNamespace

    class OneShotGraph:
        def __init__(self, **answers):
            self.answers = answers

        def invoke(self, state):
            d = SimpleNamespace(trend_up=0.1, overbought=0.1, fundamental_quality=0.4, valuation_risk=0.8,
                                action="sell", action_confidence=0.8, action_probabilities={"sell": 0.7})
            d.__dict__.update(self.answers)
            d.p_buy = d.action_probabilities.get("buy", 0.0)
            d.p_sell = d.action_probabilities.get("sell", 0.0)
            d.model_dump = lambda: {}
            return {"decision": d, "state_text": "s"}

    tech = pd.Series({"sma200": 1.0})

    def target(graph, weight=0.0, **rules):
        rules = {"entry_mode": "signals", "min_trend_up": 0.75, "min_quality": 1.5, **rules}
        return JevStrategy(graph, {"X": None}, JevRules(**rules), max_alloc=1.0).target("X", None, tech, weight)[0]

    bearish = OneShotGraph()
    assert target(bearish) == 0.0                                           # long-only: no short
    assert target(bearish, direction="short") == pytest.approx(-(0.9 * (1 - 0.5 * 0.2)))
    assert target(bearish, direction="short", entry_mode="action") == pytest.approx(-(0.7 * (1 - 0.5 * 0.2)))
    assert target(OneShotGraph(fundamental_quality=1.0), direction="short") == 0.0   # quality not weak enough
    assert target(OneShotGraph(action="buy"), direction="short") == 0.0               # buy veto
    assert target(OneShotGraph(action="buy"), direction="short", sell_veto=False) < 0
    assert target(OneShotGraph(trend_up=0.7, action="hold"), weight=-0.5, direction="short") == 0.0  # cover
    assert target(OneShotGraph(action="hold"), weight=-0.5, direction="short") == -0.5             # hold short
    assert target(bearish, weight=0.5, direction="both") < 0                                        # flip


# ---------- trailing stop / take-profit ----------

def bars(opens, highs, lows, closes, start="2024-01-01"):
    """Sessions from `start`, plus one warm-up session before it (decisions need data ≤ t-1)."""
    idx = pd.bdate_range(end=start, periods=2)[:1].append(pd.bdate_range(start, periods=len(opens))).rename("date")
    first = lambda xs: [xs[0]] + list(xs)  # noqa: E731
    return pd.DataFrame({"open": first(opens), "high": first(opens), "low": first(opens), "close": first(opens),
                         "volume": 1}, index=idx).assign(high=first(highs), low=first(lows), close=first(closes))


def test_trailing_stop_long_trails_the_high_and_fills_at_level():
    # Mon 01-01 entry at 100; the high reaches 110 on Wed; Thu's low 106.6 touches 110 × 0.97 = 106.7.
    px = bars([100, 104, 108, 108, 106, 105], [101, 106, 110, 109, 107, 106],
              [99, 103, 107, 106.6, 104, 104], [100, 105, 109, 107, 105, 105])
    res = run_backtest({"X": px}, ScriptedStrategy({"2024-01-01": 1.0}), "2024-01-01", "2024-01-08",
                       initial_cash=10_000, cost_bps=0, trailing_stop=0.03)
    stop = res.trades.iloc[-1]
    assert stop["reason"] == "stop" and stop["date"] == pd.Timestamp("2024-01-04")
    assert stop["price"] == pytest.approx(110 * 0.97)
    assert res.round_trip_pnl == [pytest.approx(100 * (110 * 0.97 - 100))]


def test_trailing_stop_fills_at_open_on_a_gap_and_short_mirror():
    # Long gapped below the stop: filled at the open (95), not at the level (97).
    px = bars([100, 95, 96], [101, 96, 97], [99.5, 94, 95], [100, 95, 96])
    res = run_backtest({"X": px}, ScriptedStrategy({"2024-01-01": 1.0}), "2024-01-01", "2024-01-03",
                       initial_cash=10_000, cost_bps=0, trailing_stop=0.03)
    assert res.trades.iloc[-1]["price"] == 95 and res.trades.iloc[-1]["reason"] == "stop"
    # Short: the low falls to 90, then the high touches 90 × 1.03 = 92.7 → cover at 92.7 (a gain).
    px = bars([100, 95, 91, 92], [100.5, 96, 92, 93], [99, 94, 90, 91], [99, 95, 91, 92.5])
    res = run_backtest({"X": px}, ScriptedStrategy({"2024-01-01": -1.0}), "2024-01-01", "2024-01-04",
                       initial_cash=10_000, cost_bps=0, borrow_bps=0, trailing_stop=0.03)
    cover = res.trades.iloc[-1]
    assert cover["side"] == "buy" and cover["reason"] == "stop" and cover["price"] == pytest.approx(92.7)
    assert res.round_trip_pnl[0] == pytest.approx(100 * (100 - 92.7))


def test_take_profit_and_stop_first_when_both_trigger():
    px = bars([100, 102, 104], [101, 103, 111], [99, 101, 103], [100, 102, 110])
    res = run_backtest({"X": px}, ScriptedStrategy({"2024-01-01": 1.0}), "2024-01-01", "2024-01-03",
                       initial_cash=10_000, cost_bps=0, take_profit=0.10)
    tp = res.trades.iloc[-1]
    assert tp["reason"] == "take_profit" and tp["price"] == pytest.approx(110)
    # One wide session hits both the 3% stop and the 10% take-profit: the stop wins.
    px = bars([100, 100], [101, 112], [99, 95], [100, 105])
    res = run_backtest({"X": px}, ScriptedStrategy({"2024-01-01": 1.0}), "2024-01-01", "2024-01-02",
                       initial_cash=10_000, cost_bps=0, trailing_stop=0.03, take_profit=0.10)
    assert res.trades.iloc[-1]["reason"] == "stop"


def test_strategy_can_reenter_after_a_stop():
    px = flat_prices(30, start="2023-12-29")
    px.loc[pd.Timestamp("2024-01-03"), "low"] = 90  # one-day spike down stops the long
    res = run_backtest({"X": px}, BaselineLikeAlwaysLong(), "2024-01-01", "2024-02-09",
                       initial_cash=10_000, cost_bps=0, trailing_stop=0.03)
    assert list(res.trades["reason"][:3]) == ["signal", "stop", "signal"]
    assert res.trades.iloc[2]["date"] == pd.Timestamp("2024-01-08")


class BaselineLikeAlwaysLong:
    name = "always-long"

    def target(self, ticker, t, technical, current_weight):
        return (current_weight or 1.0), {}


def test_atr_trailing_stop_uses_previous_session_atr():
    from jevbt.features.technical import compute_technical

    px = flat_prices(60, start="2023-12-01")
    day = pd.Timestamp("2024-02-15")
    px.loc[day, "low"] = px.at[day, "close"] - 8  # ≈ 6% intraday drop
    res = run_backtest({"X": px}, BaselineLikeAlwaysLong(), "2024-01-15", "2024-02-22",
                       initial_cash=10_000, cost_bps=0, trailing_stop_atr=2)
    stop = res.trades[res.trades["reason"] == "stop"].iloc[0]
    assert stop["date"] == day
    atr_prev = compute_technical(px)["atr_pct"].shift(1)[day]
    highest = px.loc[:day, "high"].iloc[:-1].loc["2024-01-15":].max()
    assert stop["price"] == pytest.approx(highest * (1 - 2 * atr_prev))


def test_stop_rearm_waits_for_a_fresh_signal_and_cooldown_waits_weeks():
    px = flat_prices(40, start="2023-12-29")
    px.loc[pd.Timestamp("2024-01-03"), "low"] = 90  # stop on Wednesday
    common = dict(initial_cash=10_000, cost_bps=0, trailing_stop=0.03)
    # Signal stays on: rearm never re-enters; plain stop re-enters next Monday.
    always = run_backtest({"X": px}, BaselineLikeAlwaysLong(), "2024-01-01", "2024-02-23", **common)
    rearm = run_backtest({"X": px}, BaselineLikeAlwaysLong(), "2024-01-01", "2024-02-23", stop_rearm=True, **common)
    assert len(always.trades) > 2 and list(rearm.trades["reason"]) == ["signal", "stop"]
    # Signal switches off on 01-15, back on 01-22 → rearm re-enters on 01-22.
    targets = {"2024-01-01": 1.0, "2024-01-08": 1.0, "2024-01-15": 0.0, "2024-01-22": 1.0}
    r = run_backtest({"X": px}, ScriptedStrategy(targets), "2024-01-01", "2024-02-23", stop_rearm=True, **common)
    assert r.trades.iloc[2]["date"] == pd.Timestamp("2024-01-22")
    # Cooldown of 2 weeks: blocked on 01-08 and 01-15, allowed again from 01-22 (03 + 14 days = 01-17).
    c = run_backtest({"X": px}, BaselineLikeAlwaysLong(), "2024-01-01", "2024-02-23", stop_cooldown_weeks=2, **common)
    assert c.trades.iloc[2]["date"] == pd.Timestamp("2024-01-22")


# ---------- regime switch ----------

def test_regime_switch_delegates_by_index_trend_and_sets_per_position_exits():
    from jevbt.strategy import RegimeSwitchStrategy

    idx = pd.bdate_range("2023-01-02", periods=320, name="date")
    up = np.linspace(100, 200, 260)
    index_close = pd.Series(np.concatenate([up, np.linspace(200, 120, 60)]), index=idx)  # falls below SMA200 late
    index_px = pd.DataFrame({"open": index_close, "high": index_close + 1, "low": index_close - 1,
                             "close": index_close, "volume": 1})

    class Fixed:
        def __init__(self, w):
            self.w, self.name = w, "fixed"

        def target(self, ticker, t, technical, current_weight):
            return self.w, {}

    strat = RegimeSwitchStrategy(index_px, Fixed(0.5), Fixed(-0.5), exits={("bear", 1): {"trailing_stop_atr": 2}})
    early, late = idx[250], idx[-1]
    assert strat.regime(early) == "bull" and strat.regime(late) == "bear"
    assert strat.target("X", early, None, 0.0) == (0.5, {"regime": "bull"})
    assert strat.target("X", late, None, 0.0) == (-0.5, {"regime": "bear"})
    from jevbt.strategy import NO_EXITS
    assert strat.position_exits(late, 1) == {**NO_EXITS, "trailing_stop_atr": 2}
    assert strat.position_exits(late, -1) == NO_EXITS and strat.position_exits(early, 1) == NO_EXITS


def test_engine_uses_per_position_exits_from_the_strategy():
    px = bars([100, 100, 100], [101, 101, 101], [99, 96, 99], [100, 100, 100])

    from jevbt.strategy import NO_EXITS

    class LongWithStop(ScriptedStrategy):
        def position_exits(self, t, sign):
            return {"trailing_stop": 0.03} if sign > 0 else NO_EXITS

    res = run_backtest({"X": px}, LongWithStop({"2024-01-01": 1.0}), "2024-01-01", "2024-01-03",
                       initial_cash=10_000, cost_bps=0)
    assert list(res.trades["reason"]) == ["signal", "stop"]  # stop only from the strategy, no global option
    # NO_EXITS for shorts overrides a global stop that would trigger (high 101 ≥ 100 × 1.005).
    short = run_backtest({"X": px}, LongWithStop({"2024-01-01": -1.0}), "2024-01-01", "2024-01-03",
                         initial_cash=10_000, cost_bps=0, borrow_bps=0, trailing_stop=0.005)
    assert list(short.trades["reason"]) == ["signal"]
    plain = run_backtest({"X": px}, ScriptedStrategy({"2024-01-01": -1.0}), "2024-01-01", "2024-01-03",
                         initial_cash=10_000, cost_bps=0, borrow_bps=0, trailing_stop=0.005)
    assert list(plain.trades["reason"]) == ["signal", "stop"]


def test_strategy_can_release_a_stop_block():
    px = flat_prices(40, start="2023-12-29")
    px.loc[pd.Timestamp("2024-01-03"), "low"] = 90

    class ReleaseAfterTwoWeeks(BaselineLikeAlwaysLong):
        def release_stop_block(self, t, stopped_at):
            return t - stopped_at >= pd.Timedelta(days=14)

    common = dict(initial_cash=10_000, cost_bps=0, trailing_stop=0.03, stop_rearm=True)
    held_out = run_backtest({"X": px}, BaselineLikeAlwaysLong(), "2024-01-01", "2024-02-23", **common)
    released = run_backtest({"X": px}, ReleaseAfterTwoWeeks(), "2024-01-01", "2024-02-23", **common)
    assert list(held_out.trades["reason"]) == ["signal", "stop"]
    assert released.trades.iloc[2]["date"] == pd.Timestamp("2024-01-22")
