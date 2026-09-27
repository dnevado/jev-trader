"""HTTP API for the React UI (ui/): prompt the discovery (research) agent and browse past runs.

Run: python -m jevbt serve [--port 8000]      (the UI dev server proxies /api here)
Mock mode: set JEVBT_RESEARCH_MOCK=1 to answer with a canned result (no FMP / OpenAI calls) while
working on the UI.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from jevbt.config import PROJECT_ROOT, Settings, load_settings
from jevbt.ingest.fmp import FMPBudgetError
from jevbt.research.mcp_agent import Candidate, ResearchResult, run_research

TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,9}$")
UI_DIST = PROJECT_ROOT / "ui" / "dist"


class ResearchRequest(BaseModel):
    criteria: str = Field(min_length=3, max_length=500)
    seeds: list[str] = Field(default_factory=list, max_length=10)
    max_tickers: int = Field(default=8, ge=1, le=20)

    @field_validator("seeds")
    @classmethod
    def _tickers(cls, seeds: list[str]) -> list[str]:
        out = [s.strip().upper() for s in seeds if s.strip()]
        bad = [s for s in out if not TICKER_RE.match(s)]
        if bad:
            raise ValueError(f"invalid ticker(s): {', '.join(bad)}")
        return out


class ResearchRun(BaseModel):
    id: str
    created_at: datetime
    mock: bool
    result: ResearchResult


class ResearchSummary(BaseModel):
    id: str
    created_at: datetime
    mock: bool
    criteria: str
    tickers: list[str]


class Budget(BaseModel):
    date: str | None
    count: int
    limit: int


def research_dir(settings: Settings) -> Path:
    return settings.data_dir / "research"


def save_research(settings: Settings, result: ResearchResult, mock: bool = False) -> ResearchRun:
    """Same file format as `python -m jevbt research`: data/research/<YYYYmmdd-HHMMSS>[_mock].json."""
    now = datetime.now()
    run_id = f"{now:%Y%m%d-%H%M%S}" + ("_mock" if mock else "")
    path = research_dir(settings) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return ResearchRun(id=run_id, created_at=now.replace(microsecond=0), mock=mock, result=result)


def load_research(path: Path) -> ResearchRun:
    run_id = path.stem
    return ResearchRun(id=run_id, created_at=datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S"),
                       mock=run_id.endswith("_mock"),
                       result=ResearchResult.model_validate_json(path.read_text(encoding="utf-8")))


def mock_research(req: ResearchRequest) -> ResearchResult:
    tickers = (req.seeds or ["AAPL", "MSFT", "GOOGL"])[: req.max_tickers]
    return ResearchResult(
        criteria=req.criteria,
        candidates=[Candidate(ticker=t, rationale="Mock result (JEVBT_RESEARCH_MOCK=1): no data was retrieved.")
                    for t in tickers],
        notes="Mock mode: no FMP or OpenAI calls were made.",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="jev-backtest API")
    lock = asyncio.Lock()  # one agent run at a time: every run spends the shared FMP daily budget

    @app.get("/api/budget", response_model=Budget)
    def budget() -> Budget:
        path = settings.raw_dir / "_budget.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        today = datetime.now().date().isoformat()
        count = state.get("count", 0) if state.get("date") == today else 0
        return Budget(date=today, count=count, limit=settings.fmp_daily_budget)

    @app.get("/api/research", response_model=list[ResearchSummary])
    def list_research() -> list[ResearchSummary]:
        runs = []
        for path in sorted(research_dir(settings).glob("*.json"), reverse=True):
            try:
                run = load_research(path)
            except ValueError:
                continue  # unreadable or foreign file
            runs.append(ResearchSummary(id=run.id, created_at=run.created_at, mock=run.mock,
                                        criteria=run.result.criteria,
                                        tickers=[c.ticker for c in run.result.candidates]))
        return runs

    @app.get("/api/research/{run_id}", response_model=ResearchRun)
    def get_research(run_id: str) -> ResearchRun:
        path = research_dir(settings) / f"{run_id}.json"
        if not re.fullmatch(r"\d{8}-\d{6}(_mock)?", run_id) or not path.exists():
            raise HTTPException(404, "research run not found")
        return load_research(path)

    @app.post("/api/research", response_model=ResearchRun)
    async def post_research(req: ResearchRequest) -> ResearchRun:
        if os.getenv("JEVBT_RESEARCH_MOCK") == "1":
            return save_research(settings, mock_research(req), mock=True)
        if lock.locked():
            raise HTTPException(409, "a research run is already in progress")
        async with lock:
            try:
                result = await run_research(settings, req.criteria, req.max_tickers, req.seeds)
            except FMPBudgetError as e:
                raise HTTPException(429, str(e)) from None
        return save_research(settings, result)

    if UI_DIST.exists():  # production build of the React UI, served from the same origin
        app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
    return app
