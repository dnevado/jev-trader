import pytest
from fastapi.testclient import TestClient

import jevbt.api as api
from jevbt.ingest.fmp import FMPBudgetError
from jevbt.research.mcp_agent import Candidate, ResearchResult


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.delenv("JEVBT_RESEARCH_MOCK", raising=False)
    return TestClient(api.create_app(settings))


def test_mock_mode_makes_no_agent_call(client, monkeypatch):
    monkeypatch.setenv("JEVBT_RESEARCH_MOCK", "1")

    async def boom(*a, **k):
        raise AssertionError("agent must not run in mock mode")

    monkeypatch.setattr(api, "run_research", boom)
    r = client.post("/api/research", json={"criteria": "large caps", "seeds": ["amzn", " msft "], "max_tickers": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["mock"] is True and body["id"].endswith("_mock")
    assert [c["ticker"] for c in body["result"]["candidates"]] == ["AMZN"]


def test_research_run_is_saved_and_listed(client, settings, monkeypatch):
    seen = {}

    async def fake(settings_, criteria, max_tickers, seeds):
        seen.update(criteria=criteria, max_tickers=max_tickers, seeds=seeds)
        return ResearchResult(criteria=criteria, candidates=[Candidate(ticker="NVDA", rationale="growth +65%")],
                              notes="ok")

    monkeypatch.setattr(api, "run_research", fake)
    r = client.post("/api/research", json={"criteria": "AI hardware", "seeds": ["nvda"]})
    assert r.status_code == 200
    assert seen == {"criteria": "AI hardware", "max_tickers": 8, "seeds": ["NVDA"]}
    run_id = r.json()["id"]
    assert (settings.data_dir / "research" / f"{run_id}.json").exists()

    listing = client.get("/api/research").json()
    assert listing[0]["id"] == run_id and listing[0]["tickers"] == ["NVDA"] and listing[0]["mock"] is False
    detail = client.get(f"/api/research/{run_id}").json()
    assert detail["result"]["candidates"][0]["rationale"] == "growth +65%"


def test_validation_and_not_found(client):
    assert client.post("/api/research", json={"criteria": "x"}).status_code == 422
    assert client.post("/api/research", json={"criteria": "large caps", "seeds": ["AMZN;rm"]}).status_code == 422
    assert client.post("/api/research", json={"criteria": "large caps", "max_tickers": 50}).status_code == 422
    assert client.get("/api/research/20260101-000000").status_code == 404
    assert client.get("/api/research/..%2F..%2Fsecret").status_code == 404


def test_budget_error_maps_to_429(client, monkeypatch):
    async def exhausted(*a, **k):
        raise FMPBudgetError("FMP daily budget exhausted (240/240)")

    monkeypatch.setattr(api, "run_research", exhausted)
    r = client.post("/api/research", json={"criteria": "large caps"})
    assert r.status_code == 429 and "exhausted" in r.json()["detail"]


def test_budget_endpoint(client, settings):
    body = client.get("/api/budget").json()
    assert body["count"] == 0 and body["limit"] == settings.fmp_daily_budget
