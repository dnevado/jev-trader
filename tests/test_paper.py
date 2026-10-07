import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import API_KEY
from jevbt.broker.alpaca_paper import AlpacaPaperBroker, BrokerError
from jevbt.paper import first_session_of_week, plan_orders, run_paper
from jevbt.strategy import TrendConfidenceStrategy


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


class FakeTradingSession:
    """Stands in for requests.Session against the Alpaca paper Trading API."""

    def __init__(self, positions=None, sessions=None, equity=100_000.0, shortable=True):
        self.calls = []
        self.positions = positions or []
        self.sessions = sessions or ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]
        self.equity, self.shortable = equity, shortable

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        path = url.split("paper-api.alpaca.markets/", 1)[1]
        self.calls.append((method, path, json, headers))
        if path == "v2/account":
            return Resp(200, {"equity": str(self.equity), "cash": str(self.equity), "status": "ACTIVE",
                              "shorting_enabled": True, "trading_blocked": False})
        if path == "v2/positions":
            return Resp(200, self.positions)
        if path == "v2/calendar":
            return Resp(200, [{"date": d} for d in self.sessions])
        if path.startswith("v2/assets/"):
            return Resp(200, {"shortable": self.shortable, "easy_to_borrow": self.shortable})
        if path == "v2/orders":
            return Resp(200, {"id": f"oid-{len(self.calls)}", "status": "accepted"})
        return Resp(404, {"message": "not found"})


class FakePrices:
    def __init__(self, frames):
        self.frames = frames

    def prices(self, tk, start, end):
        df = self.frames[tk]
        return df[df.index < pd.Timestamp(end)]  # like AlpacaClient: never today's bar


def trending_up(n=500):
    """Rising with wiggles (RSI < 70, efficiency ratio > 0.15): a confident up-trend for TrendConfidenceStrategy."""
    idx = pd.bdate_range("2024-11-01", periods=n, name="date")
    i = np.arange(n)
    close = pd.Series(100 + 0.3 * i + 3 * np.sin(i / 4), index=idx)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1})


TODAY = date(2026, 9, 28)  # a Monday, first session of the week


def paper_settings(settings, **kw):
    return settings.model_copy(update=kw)


def test_broker_refuses_non_paper_endpoint(settings):
    with pytest.raises(BrokerError, match="non-paper"):
        AlpacaPaperBroker(paper_settings(settings, alpaca_paper_url="https://api.alpaca.markets"))
    with pytest.raises(BrokerError, match="non-paper"):
        AlpacaPaperBroker(paper_settings(settings, alpaca_paper_url="https://paper-api.alpaca.markets.evil.com"))
    broker = AlpacaPaperBroker(settings, session=FakeTradingSession())
    assert broker.base_url == "https://paper-api.alpaca.markets"


def test_plan_orders_whole_shares_flip_and_closes_first():
    orders = plan_orders({"A": 0.10, "B": -0.10, "C": 0.0, "D": 0.10},
                         {"B": 50.0, "C": -20.0, "D": 30.0}, {"A": 33.0, "B": 10.0, "C": 5.0, "D": 100.0}, 10_000)
    as_tuples = [(o.ticker, o.side, o.qty, o.reason) for o in orders]
    assert as_tuples[:3] == [("B", "sell", 50, "close"), ("C", "buy", 20, "close")] + as_tuples[2:3]
    assert ("A", "buy", 30, "open") in as_tuples          # 1000 / 33 = 30.3 → 30 whole shares
    assert ("B", "sell", 100, "open") in as_tuples        # flip: close 50 long, then short 100
    assert ("D", "sell", 20, "adjust") in as_tuples       # 30 → 10 shares
    assert all(o.qty > 0 for o in orders)


def test_first_session_of_week():
    sessions = ["2026-09-29", "2026-09-30"]  # Monday holiday
    assert first_session_of_week(sessions, date(2026, 9, 29))
    assert not first_session_of_week(sessions, date(2026, 9, 30))


def test_dry_run_sends_no_orders_and_logs(settings):
    session = FakeTradingSession()
    broker = AlpacaPaperBroker(settings, session=session)
    run = run_paper(settings, ["UPCO"], TrendConfidenceStrategy(0.5, "long"), today=TODAY, broker=broker,
                    prices_client=FakePrices({"UPCO": trending_up()}))
    assert not run["submitted"]
    assert [c for c in session.calls if c[0] == "POST"] == []
    assert run["orders"] and run["orders"][0]["status"] == "dry-run" and run["orders"][0]["side"] == "buy"
    assert run["decisions"][0]["data_until"] < TODAY.isoformat()
    logged = json.loads(open(run["log_path"], encoding="utf-8").read())
    assert logged["orders"] == run["orders"]
    assert all(c[3]["APCA-API-SECRET-KEY"] == API_KEY for c in session.calls)


def test_submit_posts_market_on_open_orders(settings):
    session = FakeTradingSession()
    broker = AlpacaPaperBroker(settings, session=session)
    run = run_paper(settings, ["UPCO"], TrendConfidenceStrategy(0.5, "long"), submit=True, today=TODAY,
                    broker=broker, prices_client=FakePrices({"UPCO": trending_up()}))
    posts = [c for c in session.calls if c[0] == "POST"]
    assert len(posts) == 1
    body = posts[0][2]
    assert body["side"] == "buy" and body["type"] == "market" and body["time_in_force"] == "day"
    assert int(body["qty"]) == run["orders"][0]["qty"] > 0
    assert run["orders"][0]["status"] == "accepted"


def test_submit_refused_when_not_first_session_unless_forced(settings):
    broker = AlpacaPaperBroker(settings, session=FakeTradingSession())
    frames = FakePrices({"UPCO": trending_up()})
    with pytest.raises(SystemExit, match="first session"):
        run_paper(settings, ["UPCO"], TrendConfidenceStrategy(0.5), submit=True, today=date(2026, 9, 30),
                  broker=broker, prices_client=frames)
    run = run_paper(settings, ["UPCO"], TrendConfidenceStrategy(0.5), submit=True, force=True,
                    today=date(2026, 9, 30), broker=broker, prices_client=frames)
    assert run["submitted"]


def test_outside_positions_count_in_the_gross_cap(settings):
    outside = [{"symbol": "ZZZ", "qty": "100", "side": "long", "market_value": "90000"}]
    broker = AlpacaPaperBroker(settings, session=FakeTradingSession(positions=outside))
    run = run_paper(settings, ["UPCO"], TrendConfidenceStrategy(0.5, "long"), today=TODAY, broker=broker,
                    prices_client=FakePrices({"UPCO": trending_up()}))
    assert run["decisions"][0]["target_weight"] == pytest.approx(0.10)  # only 10% room left under the 100% cap
    assert any("outside the universe" in n for n in run["notes"])


def test_daily_rebalance_submits_midweek_without_force(settings):
    broker = AlpacaPaperBroker(settings, session=FakeTradingSession())
    run = run_paper(settings, ["UPCO"], TrendConfidenceStrategy(0.5), submit=True, today=date(2026, 9, 30),
                    broker=broker, prices_client=FakePrices({"UPCO": trending_up()}), rebalance="daily")
    assert run["submitted"] and run["rebalance"] == "daily" and run["notes"] == []
