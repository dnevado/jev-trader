"""Entry/exit/sizing rules. Strategies return a target weight per ticker at each rebalance date.

Weights are fractions of portfolio equity, capped at `max_alloc` per position. A held position that
triggers no exit keeps its current weight (no resizing) to limit turnover.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd


class Strategy(Protocol):
    name: str

    def target(self, ticker: str, t: pd.Timestamp, technical: pd.Series, current_weight: float) -> tuple[float, dict]:
        """(target weight, log record) for `ticker` at rebalance date t, from data ≤ t-1."""


@dataclass
class BaselineStrategy:
    """No-LLM baseline (CLAUDE.md §1.9): enter when close > SMA200 and RSI14 < 70; exit when close < SMA200."""

    max_alloc: float = 0.1
    name: str = "baseline"

    def target(self, ticker, t, technical, current_weight):
        close, sma200, rsi = technical["close"], technical["sma200"], technical["rsi14"]
        if not np.isfinite(sma200):
            return 0.0, {"reason": "warm-up"}
        if current_weight > 0:
            if close < sma200:
                return 0.0, {"reason": "exit: close < sma200"}
            return current_weight, {"reason": "hold"}
        if close > sma200 and rsi < 70:
            return self.max_alloc, {"reason": "entry: close > sma200 and rsi < 70"}
        return 0.0, {"reason": "no entry"}


@dataclass(frozen=True)
class JevRules:
    """CLAUDE.md §5 thresholds (optimized with walk-forward validation, see backtest/walkforward.py).

    entry_mode:
      "action"  — original §5 rule: requires Jev's action == buy with confidence ≥ min_confidence;
                  size = p(buy) × (1 − penalty × valuation_risk).
      "signals" — entry from the Noul/Score answers only (trend, overbought, quality), vetoed by action == sell;
                  size = trend_up × (1 − penalty × valuation_risk). Jev's action is still used to exit.
    """

    entry_mode: Literal["action", "signals"] = "action"
    min_confidence: float = 0.70
    min_trend_up: float = 0.75
    max_overbought: float = 0.50
    min_quality: float = 1.5      # expected Score level: 0 weak, 1 medium, 2 strong
    exit_trend_up: float = 0.40
    valuation_penalty: float = 0.5


class JevStrategy:
    """Jev decisions do not depend on the position or the rules, so they are memoized per (ticker, t):
    changing `rules` (walk-forward grid search) re-evaluates the rules without calling the graph again."""

    name = "jev"

    def __init__(self, graph, ratios: dict[str, pd.DataFrame], rules: JevRules | None = None,
                 max_alloc: float = 0.1) -> None:
        self.graph = graph
        self.ratios = ratios
        self.rules = rules or JevRules()
        self.max_alloc = max_alloc
        self._memo: dict[tuple[str, pd.Timestamp], dict] = {}

    def decision(self, ticker: str, t: pd.Timestamp, technical: pd.Series) -> dict:
        key = (ticker, t)
        if key not in self._memo:
            self._memo[key] = self.graph.invoke(
                {"ticker": ticker, "t": t, "technical": technical, "ratios": self.ratios[ticker]})
        return self._memo[key]

    def target(self, ticker, t, technical, current_weight):
        if not np.isfinite(technical["sma200"]):
            return 0.0, {"reason": "warm-up"}
        out = self.decision(ticker, t, technical)
        d = out["decision"]
        r = self.rules
        log = {"state": out["state_text"], "jev": d.model_dump(), "rules": asdict(r)}
        if current_weight > 0:
            if d.action == "sell" or d.trend_up < r.exit_trend_up:
                return 0.0, {**log, "reason": "exit"}
            return current_weight, {**log, "reason": "hold"}
        signals_ok = (d.trend_up >= r.min_trend_up and d.overbought < r.max_overbought
                      and d.fundamental_quality >= r.min_quality)
        if r.entry_mode == "action":
            if signals_ok and d.action == "buy" and d.action_confidence >= r.min_confidence:
                size = d.p_buy * (1 - r.valuation_penalty * d.valuation_risk)
                return size * self.max_alloc, {**log, "reason": "entry"}
        elif signals_ok and d.action != "sell":
            size = d.trend_up * (1 - r.valuation_penalty * d.valuation_risk)
            return size * self.max_alloc, {**log, "reason": "entry"}
        return 0.0, {**log, "reason": "no entry"}
