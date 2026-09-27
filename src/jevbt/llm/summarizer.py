"""Fundamentals summarizer: OpenAI (ChatOpenAI, mini model) → FundSummary, cached per (ticker, fiscal year filing).

The model is only called for a prompt that is not in the disk cache; the cache key is the hash of
model + system prompt + user prompt, so changing any of them invalidates it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

SYSTEM_PROMPT = (
    "You are a fundamental equity analyst. Use ONLY the documents provided in the user message. "
    "Do not use your own knowledge about the company, its industry, its stock price or any event "
    "after the analysis date. If a figure is missing, write 'no data'. Be factual and concise; "
    "do not give investment recommendations."
)


class FundSummary(BaseModel):
    growth: str = Field(description="Revenue growth level and trend across the fiscal years provided")
    margins: str = Field(description="Gross, operating, net and FCF margin levels and trend")
    balance_sheet: str = Field(description="Leverage (debt to equity) and balance sheet strength")
    catalysts: list[str] = Field(description="Positive factors visible in the documents")
    risks: list[str] = Field(description="Negative factors or deteriorations visible in the documents")
    data_quality: Literal["good", "partial", "poor"] = Field(
        description="good: 3 fiscal years with all ratios; partial: some missing; poor: mostly missing")


class StructuredLLM(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> FundSummary: ...


def openai_summarizer_llm(model: str) -> StructuredLLM:
    from langchain_openai import ChatOpenAI  # imported lazily: tests and offline runs do not need it

    return ChatOpenAI(model=model, temperature=0).with_structured_output(FundSummary)


class Summarizer:
    def __init__(self, llm: StructuredLLM, model: str, cache_dir: Path) -> None:
        self.llm = llm
        self.model = model
        self.cache_dir = cache_dir / "summaries"
        self.calls = 0

    def _key(self, user_prompt: str) -> str:
        blob = json.dumps([self.model, SYSTEM_PROMPT, user_prompt])
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def summarize(self, ticker: str, fiscal_year: str, user_prompt: str) -> FundSummary:
        path = self.cache_dir / f"{ticker.upper()}_FY{fiscal_year}_{self._key(user_prompt)}.json"
        if path.exists():
            return FundSummary.model_validate_json(path.read_text(encoding="utf-8"))
        summary = self.llm.invoke([("system", SYSTEM_PROMPT), ("user", user_prompt)])
        self.calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        return summary


class MockSummaryLLM:
    """Offline stand-in: deterministic summary derived from the prompt, no network."""

    def invoke(self, messages: list[tuple[str, str]]) -> FundSummary:
        prompt = messages[-1][1]
        n_years = prompt.count("\nFY")
        missing = prompt.count("no data")
        quality = "good" if n_years >= 3 and missing == 0 else "partial" if n_years >= 1 else "poor"
        return FundSummary(growth="mock", margins="mock", balance_sheet="mock",
                           catalysts=[], risks=[], data_quality=quality)
