from datetime import date
from pathlib import Path

import pytest

from jevbt import aws_job
from jevbt.aws_job import Deps, handler

MONDAY, WEDNESDAY = date(2026, 10, 5), date(2026, 10, 7)
SESSIONS = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02",
            "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]


class FakeBroker:
    def __init__(self, settings=None, orders=None):
        self._orders = orders or []

    def calendar(self, start, end):
        return SESSIONS

    def orders(self, after):
        self.after = after
        return self._orders

    def account(self):
        return {"equity": 101_234.5}

    def positions(self):
        return {"XOM": {"qty": 50, "market_value": 8000}}


@pytest.fixture(autouse=True)
def lambda_env(monkeypatch, tmp_path):
    monkeypatch.setenv("JEVBT_TICKERS", "XOM, nke ,KO")
    monkeypatch.setenv("JEVBT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ALPACA_API_KEY_ID", "kid")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")


def make_deps(today, broker=None, run_result=None):
    sent, calls = [], []

    def fake_run_paper(settings, tickers, strategy, **kw):
        calls.append((tickers, strategy, kw))
        log = Path(settings.data_dir) / "run.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("{}")
        return {"equity": 100_000.0, "orders": [], "positions_before": {}, "notes": [], "log_path": str(log),
                **(run_result or {})}

    deps = Deps(publish=lambda s, m: sent.append((s, m)), upload=lambda p: f"s3://bucket/paper/{p.name}",
                broker_factory=lambda settings: broker or FakeBroker(), run_paper=fake_run_paper, today=today)
    return deps, sent, calls


def test_trade_skips_quietly_when_not_first_session():
    deps, sent, calls = make_deps(WEDNESDAY)
    out = handler({"mode": "trade"}, None, deps)
    assert "skipped" in out and sent == [] and calls == []


def test_trade_submits_on_first_session_and_emails_orders():
    orders = [{"ticker": "XOM", "side": "buy", "qty": 50, "reason": "open", "ref_price": 160.0, "status": "accepted"}]
    deps, sent, calls = make_deps(MONDAY, run_result={"orders": orders})
    out = handler({"mode": "trade"}, None, deps)
    tickers, strategy, kw = calls[0]
    assert tickers == ["XOM", "NKE", "KO"]
    assert kw["submit"] is True and kw["time_in_force"] == "opg" and kw["today"] == MONDAY
    assert strategy.name == "trend" and strategy.direction == "long"
    assert out["orders"] == 1 and out["log"] == "s3://bucket/paper/run.json"
    subject, body = sent[0]
    assert "1 order(s) submitted" in subject and "BUY" in body and "XOM" in body


def test_trade_without_orders_still_emails_a_heartbeat():
    deps, sent, _ = make_deps(MONDAY)
    handler({"mode": "trade"}, None, deps)
    assert sent and "no orders" in sent[0][0]


def test_report_emails_fills_opened_and_closed():
    orders = [
        {"client_order_id": "jevbt-20261005-XOM-open-buy", "symbol": "XOM", "status": "filled",
         "filled_qty": "50", "filled_avg_price": "161.20", "qty": "50"},
        {"client_order_id": "jevbt-20261005-NKE-close-sell", "symbol": "NKE", "status": "filled",
         "filled_qty": "30", "filled_avg_price": "34.10", "qty": "30"},
        {"client_order_id": "jevbt-20261005-KO-open-buy", "symbol": "KO", "status": "rejected", "qty": "10"},
        {"client_order_id": "manual-order", "symbol": "AAPL", "status": "filled", "filled_qty": "1",
         "filled_avg_price": "200", "qty": "1"},
    ]
    broker = FakeBroker(orders=orders)
    deps, sent, _ = make_deps(MONDAY, broker=broker)
    out = handler({"mode": "report"}, None, deps)
    assert out == {"mode": "report", "filled": 2, "not_filled": 1}
    subject, body = sent[0]
    assert subject == "jevbt paper 2026-10-05: 1 opened, 1 closed, 1 NOT filled"
    assert "OPENED LONG" in body and "XOM" in body and "161.20" in body
    assert "CLOSED LONG" in body and "NKE" in body
    assert "rejected" in body and "AAPL" not in body
    assert broker.after.startswith("2026-10-05T04:00:00")  # midnight New York (EDT) in UTC


def test_report_silent_without_jevbt_orders():
    deps, sent, _ = make_deps(WEDNESDAY, broker=FakeBroker(orders=[]))
    assert handler({"mode": "report"}, None, deps) == {"mode": "report", "orders": 0}
    assert sent == []


def test_errors_are_emailed_and_reraised():
    deps, sent, _ = make_deps(MONDAY)
    deps.broker_factory = lambda settings: (_ for _ in ()).throw(RuntimeError("alpaca down"))
    with pytest.raises(RuntimeError, match="alpaca down"):
        handler({"mode": "trade"}, None, deps)
    assert sent[0][0] == "jevbt paper: ERROR in trade" and "alpaca down" in sent[0][1]


def test_secrets_loaded_from_ssm(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID")
    monkeypatch.delenv("ALPACA_API_SECRET_KEY")

    class FakeSSM:
        def get_parameters(self, Names, WithDecryption):
            assert WithDecryption is True
            return {"Parameters": [{"Name": "/jevbt/alpaca_api_key_id", "Value": "K"},
                                   {"Name": "/jevbt/alpaca_api_secret_key", "Value": "S"}]}

    aws_job.load_secrets_from_ssm(FakeSSM())
    import os
    assert os.environ["ALPACA_API_KEY_ID"] == "K" and os.environ["ALPACA_API_SECRET_KEY"] == "S"


def test_placeholder_secrets_are_rejected(monkeypatch):
    monkeypatch.delenv("ALPACA_API_SECRET_KEY")

    class FakeSSM:
        def get_parameters(self, Names, WithDecryption):
            return {"Parameters": [{"Name": "/jevbt/alpaca_api_key_id", "Value": "K"},
                                   {"Name": "/jevbt/alpaca_api_secret_key", "Value": "CHANGE_ME"}]}

    with pytest.raises(RuntimeError, match="alpaca_api_secret_key"):
        aws_job.load_secrets_from_ssm(FakeSSM())


def test_main_returns_exit_code(monkeypatch):
    monkeypatch.setattr(aws_job, "handler", lambda event, context: {"mode": event["mode"], "ok": True})
    assert aws_job.main(["report"]) == 0

    def boom(event, context):
        raise RuntimeError("x")

    monkeypatch.setattr(aws_job, "handler", boom)
    assert aws_job.main(["trade"]) == 1


def test_daily_mode_trades_midweek_and_emails_only_with_orders(monkeypatch):
    monkeypatch.setenv("JEVBT_REBALANCE", "daily")
    deps, sent, calls = make_deps(WEDNESDAY)
    out = handler({"mode": "trade"}, None, deps)
    assert calls and calls[0][2]["rebalance"] == "daily"
    assert out["orders"] == 0 and sent == []                    # no empty email midweek
    orders = [{"ticker": "XOM", "side": "buy", "qty": 5, "reason": "open", "ref_price": 160.0, "status": "accepted"}]
    deps, sent, _ = make_deps(WEDNESDAY, run_result={"orders": orders})
    handler({"mode": "trade"}, None, deps)
    assert sent and "1 order(s) submitted" in sent[0][0] and "daily rebalance" in sent[0][1]
    deps, sent, _ = make_deps(MONDAY)                           # weekly heartbeat on the first session
    handler({"mode": "trade"}, None, deps)
    assert sent and "no orders" in sent[0][0]


def test_daily_mode_skips_non_sessions(monkeypatch):
    monkeypatch.setenv("JEVBT_REBALANCE", "daily")
    deps, sent, calls = make_deps(date(2026, 10, 10))  # Saturday
    out = handler({"mode": "trade"}, None, deps)
    assert "not a trading session" in out["skipped"] and calls == [] and sent == []
