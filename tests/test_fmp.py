import json

import pandas as pd
import pytest

from conftest import API_KEY, FIXED_TODAY, FakeResponse, FakeSession
from jevbt.ingest.fmp import FMPBudgetError, FMPClient, FMPError, FMPPlanError, normalize

PAYWALL = "Restricted Endpoint: This endpoint is not available under your current subscription"


def statement_rows(dates: list[str], legacy_typo: bool = False) -> list[dict]:
    key = "fillingDate" if legacy_typo else "filingDate"
    return [
        {"date": d, "symbol": "AAPL", key: d, "acceptedDate": f"{d} 06:01:02", "fiscalYear": d[:4],
         "period": "Q1", "revenue": 100.0, "netIncome": 10.0}
        for d in sorted(dates, reverse=True)
    ]


def statements_handler(path, params):
    assert path == "income-statement"
    return FakeResponse(200, statement_rows(["2024-03-30", "2024-06-29"]))


def make_client(settings, handler, **kw):
    session = FakeSession(handler)
    client = FMPClient(settings, session=session, sleep=lambda s: None, today=lambda: FIXED_TODAY, **kw)
    return client, session


# ---------- normalization ----------

def test_normalize_fixes_typo_snake_case_dates_and_order():
    df = normalize(statement_rows(["2024-03-30", "2024-06-29"], legacy_typo=True))
    assert {"filing_date", "accepted_date", "fiscal_year", "net_income"} <= set(df.columns)
    assert "filling_date" not in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_datetime64_any_dtype(df["accepted_date"])
    assert df["date"].is_monotonic_increasing


def test_normalize_empty():
    assert normalize([]).empty


# ---------- statements ----------

def test_statements_cached_and_merged_on_refresh(settings):
    batches = iter([
        statement_rows(["2025-03-29", "2025-06-28"]),
        statement_rows(["2025-06-28", "2025-09-27"]),  # the free plan only returns the latest periods
    ])
    client, session = make_client(settings, lambda path, params: FakeResponse(200, next(batches)))
    first = client.statements("AAPL", "income", "quarter")
    assert session.calls[0] == ("income-statement",
                                {"symbol": "AAPL", "period": "quarter", "limit": 5, "apikey": API_KEY})
    client.statements("AAPL", "income", "quarter")
    assert len(session.calls) == 1
    merged = client.statements("AAPL", "income", "quarter", refresh=True)
    assert len(first) == 2
    assert list(merged.index.strftime("%Y-%m-%d")) == ["2025-03-29", "2025-06-28", "2025-09-27"]
    assert "accepted_date" in merged.columns


def test_statements_unknown_kind(settings):
    client, _ = make_client(settings, statements_handler)
    with pytest.raises(ValueError):
        client.statements("AAPL", "ratios")


# ---------- errors, retries, budget, secrets ----------

def test_paywall_raises_plan_error_without_leaking_key(settings):
    client, _ = make_client(settings, lambda path, params: FakeResponse(402, f"{PAYWALL} apikey={API_KEY}"))
    with pytest.raises(FMPPlanError) as exc:
        client.transcript("AAPL", 2024, 1)
    assert API_KEY not in str(exc.value)
    assert "Restricted Endpoint" in str(exc.value)


def test_retries_on_429_then_succeeds(settings):
    responses = iter([FakeResponse(429, "slow down"), FakeResponse(503, "busy"),
                      FakeResponse(200, statement_rows(["2024-03-30", "2024-06-29"]))])
    client, session = make_client(settings, lambda path, params: next(responses))
    assert len(client.statements("AAPL", "income")) == 2
    assert len(session.calls) == 3


def test_retries_exhausted(settings):
    client, session = make_client(settings, lambda path, params: FakeResponse(500, "boom"), max_retries=2)
    with pytest.raises(FMPError, match="HTTP 500"):
        client.statements("AAPL", "income")
    assert len(session.calls) == 3


def test_error_message_in_200_body(settings):
    client, _ = make_client(settings, lambda path, params: FakeResponse(200, {"Error Message": f"bad key {API_KEY}"}))
    with pytest.raises(FMPError) as exc:
        client.statements("AAPL", "income")
    assert API_KEY not in str(exc.value)


def test_budget_exhausted_blocks_request(settings):
    budget = settings.raw_dir / "_budget.json"
    budget.parent.mkdir(parents=True)
    budget.write_text(json.dumps({"date": FIXED_TODAY.isoformat(), "count": settings.fmp_daily_budget}))
    client, session = make_client(settings, statements_handler)
    with pytest.raises(FMPBudgetError):
        client.statements("AAPL", "income")
    assert session.calls == []


def test_budget_resets_on_a_new_day_and_counts(settings):
    budget = settings.raw_dir / "_budget.json"
    budget.parent.mkdir(parents=True)
    budget.write_text(json.dumps({"date": "2026-09-26", "count": settings.fmp_daily_budget}))
    client, _ = make_client(settings, statements_handler)
    client.statements("AAPL", "income")
    assert json.loads(budget.read_text()) == {"date": FIXED_TODAY.isoformat(), "count": 1}


def test_missing_api_key(settings):
    client, session = make_client(settings.model_copy(update={"fmp_api_key": ""}), statements_handler)
    with pytest.raises(FMPError, match="FMP_API_KEY"):
        client.statements("AAPL", "income")
    assert session.calls == []
