"""Research agent: ChatOpenAI (full model) + FMP's official MCP tools → proposed ticker universe.

Separate from the backtest (CLAUDE.md §1.2): *MCP to explore and choose, REST to measure.*
Free FMP plan (verified 2026-09-27, scripts/probe_fmp_mcp.py): `search` (screener) and `indexes`
(S&P 500 constituents) are denied; `company` (profile, peers), `statements` (key-metrics, statements)
and `marketPerformance` work. The agent only gets the allowed tools, and every tool call is counted
in the same daily FMP budget as the REST client.

Caveat: the agent picks tickers with today's information. Backtesting those tickers over past periods
has selection (hindsight/survivorship) bias → use the proposed universe for forward / recent-period tests.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from jevbt.config import Settings
from jevbt.ingest.fmp import FMPClient

ALLOWED_TOOLS = ("company", "statements", "marketPerformance", "quote")
MAX_TOOL_CALLS = 25

SYSTEM_PROMPT = f"""You are an equity research assistant that proposes a universe of US stocks for a backtest.
You have Financial Modeling Prep tools. The account is on the FREE plan: the stock screener and index
constituent lists are NOT available. Build candidates from:
- the seed tickers given by the user, expanded with `company` endpoint `peers`;
- `marketPerformance` (e.g. `most-active`, `biggest-gainers`, sector snapshots) for ideas;
- `company` `profile-symbol` to check sector, market cap, country and that the stock is actively traded;
- `statements` `key-metrics` / `income-statement-growth` (period=annual, limit<=5) to check the criteria.
Rules:
- Only US-listed common stocks (no ETFs, funds, ADRs unless the user asks), actively trading, with at least
  5 years of history.
- Use at most {MAX_TOOL_CALLS} tool calls in total; every call spends a daily API quota. Prefer one call per
  candidate check. If a tool answers ACCESS DENIED, do not call it again.
- Base each rationale on the data you retrieved (cite the figures), not on your own memory.
- Return fewer tickers rather than tickers you could not check."""


class Candidate(BaseModel):
    ticker: str = Field(description="US ticker symbol, upper case")
    rationale: str = Field(description="Why it fits the criteria, citing the retrieved figures")


class ResearchResult(BaseModel):
    criteria: str
    candidates: list[Candidate]
    notes: str = Field(description="Limitations, denied tools, candidates rejected and why")


def _budgeted(tool, spend):
    """Count every MCP call in the FMP daily budget before it runs."""
    original = tool.coroutine

    async def call(*args, **kwargs):
        spend()
        return await original(*args, **kwargs)

    tool.coroutine = call
    return tool


async def get_fmp_tools(settings: Settings):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({"fmp": {
        "transport": "streamable_http",
        "url": f"https://financialmodelingprep.com/mcp?apikey={settings.fmp_api_key}",
    }})
    spend = FMPClient(settings)._spend_budget
    return [_budgeted(t, spend) for t in await client.get_tools() if t.name in ALLOWED_TOOLS]


def build_agent(model, tools):
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT, response_format=ResearchResult)


async def run_research(settings: Settings, criteria: str, max_tickers: int = 10, seeds: list[str] | None = None,
                       model=None, tools=None) -> ResearchResult:
    if model is None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=settings.openai_model_full, temperature=0)
    if tools is None:
        tools = await get_fmp_tools(settings)
    agent = build_agent(model, tools)
    request = f"Criteria: {criteria}\nPropose at most {max_tickers} tickers."
    if seeds:
        request += f"\nSeed tickers: {', '.join(s.upper() for s in seeds)}"
    out = await agent.ainvoke({"messages": [("user", request)]},
                              config={"recursion_limit": 2 * MAX_TOOL_CALLS + 5})
    result: ResearchResult = out["structured_response"]
    result.criteria = criteria
    result.candidates = result.candidates[:max_tickers]
    for c in result.candidates:
        c.ticker = c.ticker.strip().upper()
    return result
