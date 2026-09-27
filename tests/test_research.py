import asyncio
import json

import pytest
from langchain_core.tools import StructuredTool

from jevbt.ingest.fmp import FMPBudgetError, FMPClient
from jevbt.research.mcp_agent import ALLOWED_TOOLS, _budgeted


def make_tool(name="company"):
    async def run(endpoint: str, symbol: str = "") -> str:
        return f"{endpoint}:{symbol}"

    return StructuredTool.from_function(coroutine=run, name=name, description="fake MCP tool")


def test_budgeted_tool_counts_in_fmp_budget(settings):
    spend = FMPClient(settings)._spend_budget
    tool = _budgeted(make_tool(), spend)
    assert asyncio.run(tool.ainvoke({"endpoint": "peers", "symbol": "AMZN"})) == "peers:AMZN"
    assert asyncio.run(tool.ainvoke({"endpoint": "profile-symbol", "symbol": "MSFT"})) == "profile-symbol:MSFT"
    assert json.loads((settings.raw_dir / "_budget.json").read_text())["count"] == 2


def test_budgeted_tool_stops_when_budget_is_exhausted(settings):
    settings = settings.model_copy(update={"fmp_daily_budget": 1})
    calls = []

    async def run(endpoint: str) -> str:
        calls.append(endpoint)
        return "ok"

    tool = _budgeted(StructuredTool.from_function(coroutine=run, name="quote", description="x"),
                     FMPClient(settings)._spend_budget)
    asyncio.run(tool.ainvoke({"endpoint": "a"}))
    with pytest.raises(FMPBudgetError):
        asyncio.run(tool.ainvoke({"endpoint": "b"}))
    assert calls == ["a"]


def test_denied_tools_are_not_offered():
    assert "search" not in ALLOWED_TOOLS and "indexes" not in ALLOWED_TOOLS
