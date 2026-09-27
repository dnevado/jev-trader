"""LangGraph per (ticker, rebalance date t): fundamentals → build_state → decide.

Everything passed in is already point-in-time (technical row ≤ t-1, ratios with known_at < t).
The state sent to Jev and the summarizer prompt are anonymized by default (no ticker, no date) to limit
what the models can recall from training about a specific company and period.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from jevbt.decision.jev import JevDecider, JevDecision
from jevbt.features.fundamentals import build_user_prompt, pe_ratio, point_in_time, ratios_table
from jevbt.llm.summarizer import FundSummary, Summarizer

JEV_MAX_STATE_CHARS = 100_000  # well below Jev 1.13's 32K-token limit


class TickerState(TypedDict, total=False):
    ticker: str
    t: pd.Timestamp
    technical: pd.Series      # features_asof(features, t)
    ratios: pd.DataFrame      # compute_ratios(...) for this ticker
    history: pd.DataFrame     # point-in-time fiscal years
    summary: FundSummary | None
    pe: float
    state_text: str
    decision: JevDecision


def _v(x: float, pct: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "no data"
    return f"{x:.2%}" if pct else f"{x:.2f}"


def render_state(technical: pd.Series, history: pd.DataFrame, pe: float, summary: FundSummary | None,
                 label: str = "the company") -> str:
    tech = technical
    lines = [
        f"Stock: {label} (US equity). Decision horizon: 4 weeks.",
        "## Technical indicators (last close)",
        f"close: {_v(tech['close'])}",
        f"sma50: {_v(tech['sma50'])}",
        f"sma200: {_v(tech['sma200'])}",
        f"rsi14: {_v(tech['rsi14'])}",
        f"mom_3m: {_v(tech['mom_3m'], True)}",
        f"mom_12m: {_v(tech['mom_12m'], True)}",
        f"atr_pct: {_v(tech['atr_pct'], True)}",
        f"drawdown: {_v(tech['drawdown'], True)}",
        "## Valuation",
        f"pe_ratio: {_v(pe)}",
        "## Annual fundamentals (published fiscal years)",
    ]
    if history.empty:
        lines.append("no data")
    else:
        latest = history.iloc[-1]
        lines += [
            f"revenue_growth_yoy: {_v(latest['revenue_growth_yoy'], True)}",
            f"operating_margin: {_v(latest['operating_margin'], True)}",
            f"fcf_margin: {_v(latest['fcf_margin'], True)}",
            f"debt_to_equity: {_v(latest['debt_to_equity'])}",
            ratios_table(history),
        ]
    if summary is not None:
        lines += [
            "## Analyst summary of the fundamentals",
            f"growth: {summary.growth}",
            f"margins: {summary.margins}",
            f"balance_sheet: {summary.balance_sheet}",
            "catalysts: " + ("; ".join(summary.catalysts) or "none"),
            "risks: " + ("; ".join(summary.risks) or "none"),
            f"data_quality: {summary.data_quality}",
        ]
    text = "\n".join(lines)
    if len(text) > JEV_MAX_STATE_CHARS:
        raise ValueError(f"Jev state too long ({len(text)} chars)")
    return text


def build_graph(summarizer: Summarizer, decider: JevDecider, anonymize: bool = True):
    def fundamentals(s: TickerState) -> TickerState:
        history = point_in_time(s["ratios"], s["t"])
        summary = None
        if not history.empty:
            company = "the company" if anonymize else s["ticker"]
            latest = history.iloc[-1]
            summary = summarizer.summarize(s["ticker"], latest["fiscal_year"], build_user_prompt(company, history))
        eps = history.iloc[-1]["eps_diluted"] if not history.empty else float("nan")
        return {"history": history, "summary": summary, "pe": pe_ratio(s["technical"]["close"], eps)}

    def build_state(s: TickerState) -> TickerState:
        label = "the company" if anonymize else s["ticker"]
        return {"state_text": render_state(s["technical"], s["history"], s["pe"], s["summary"], label)}

    def decide(s: TickerState) -> TickerState:
        return {"decision": decider.decide(s["state_text"])}

    g = StateGraph(TickerState)
    g.add_node("fundamentals", fundamentals)
    g.add_node("build_state", build_state)
    g.add_node("decide", decide)
    g.add_edge(START, "fundamentals")
    g.add_edge("fundamentals", "build_state")
    g.add_edge("build_state", "decide")
    g.add_edge("decide", END)
    return g.compile()
