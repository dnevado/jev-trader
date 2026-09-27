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
