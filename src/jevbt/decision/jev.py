"""Jev (TypeSafe) decisions via OpenRouter, with a disk cache and an offline mock.

API verified against the installed langchain-typesafe 0.0.1a2: questions and model are constructor
arguments, `invoke(state)` posts to `{base_url}/v1/systemone`, and `Score.score` is an expected value
(a float such as 1.35), not a level index.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from langchain_typesafe import Choice, Noul, Score

QUALITY_LEVELS = ["weak", "medium", "strong"]

QUESTIONS = {
    "trend_up": Noul(instructions="Is the medium-term price trend clearly bullish?"),
    "overbought": Noul(instructions="Is the stock overbought, with a risk of a short-term correction?"),
    "fundamental_quality": Score(
        instructions="How strong are the company's fundamentals (growth, margins, balance sheet)?",
        criteria=QUALITY_LEVELS,
    ),
    "valuation_risk": Noul(instructions="Does the valuation pose a relevant risk within the next 4 weeks?"),
    "action": Choice(
        instructions="For a 4-week horizon, what should a disciplined investor do with this stock?",
        criteria={
            "buy": "Open or keep a long position: trend, momentum and fundamentals support it.",
            "hold": "No clear edge: do not open a new position.",
            "sell": "Close the position: trend broken, fundamentals deteriorating or clear overvaluation.",
        },
    ),
}


class JevDecision(BaseModel):
    trend_up: float
    overbought: float
    fundamental_quality: float  # expected level: 0 = weak … 2 = strong
    valuation_risk: float
    action: str
    action_confidence: float
    action_probabilities: dict[str, float]
    model: str
    request_id: str | None = None

    @property
    def p_buy(self) -> float:
        return self.action_probabilities.get("buy", 0.0)

    @property
    def p_sell(self) -> float:
        return self.action_probabilities.get("sell", 0.0)


class Classifier(Protocol):
    model: str

    def invoke(self, state: str): ...


def questions_payload() -> dict:
    return {name: q.model_dump(mode="json", exclude_none=True) for name, q in QUESTIONS.items()}


def to_decision(response) -> JevDecision:
    action = response.choices["action"]
    return JevDecision(
        trend_up=response.nouls["trend_up"].noul,
        overbought=response.nouls["overbought"].noul,
        fundamental_quality=response.scores["fundamental_quality"].score,
        valuation_risk=response.nouls["valuation_risk"].noul,
        action=action.choice,
        action_confidence=action.confidence,
        action_probabilities=action.probabilities,
        model=response.model,
        request_id=response.request_id,
    )


def typesafe_classifier(model: str, api_key: str, base_url: str) -> Classifier:
    from langchain_typesafe import TypeSafeClassifier

    return TypeSafeClassifier(questions=QUESTIONS, model=model, api_key=api_key, base_url=base_url, timeout=60)


class JevDecider:
    """One Jev call per distinct (state, questions, model); repeated states come from the disk cache."""

    def __init__(self, classifier: Classifier, cache_dir: Path) -> None:
        self.classifier = classifier
        self.cache_dir = cache_dir / "jev"
        self.calls = 0

    def key(self, state: str) -> str:
        blob = json.dumps([self.classifier.model, questions_payload(), state], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def decide(self, state: str) -> JevDecision:
        path = self.cache_dir / f"{self.key(state)}.json"
        if path.exists():
            return JevDecision.model_validate_json(path.read_text(encoding="utf-8"))
        decision = to_decision(self.classifier.invoke(state))
        self.calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")
        return decision


# ---------- offline mock ----------

def _number(state: str, name: str) -> float | None:
    m = re.search(rf"^{name}: (-?[\d.,]+)(%?)", state, re.MULTILINE)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return value / 100 if m.group(2) else value


class MockJevClassifier:
    """Deterministic rule-of-thumb answers read from the state text. For tests and dry runs, never for results."""

    model = "mock/jev"

    def invoke(self, state: str):
        from langchain_typesafe.types import ClassificationResponse

        close = _number(state, "close") or 0.0
        sma200 = _number(state, "sma200")
        rsi = _number(state, "rsi14") or 50.0
        mom = _number(state, "mom_3m") or 0.0
        growth = _number(state, "revenue_growth_yoy")
        pe = _number(state, "pe_ratio")

        trend = 0.85 if sma200 and close > sma200 and mom > 0 else 0.3
        overbought = 0.8 if rsi > 70 else 0.2
        quality = 2.0 if growth is not None and growth > 0.05 else 1.0 if growth is not None else 0.5
        val_risk = 0.7 if pe is not None and pe > 40 else 0.3
        if trend > 0.5 and overbought < 0.5:
            probs = {"buy": 0.8, "hold": 0.15, "sell": 0.05}
        elif trend < 0.5:
            probs = {"buy": 0.05, "hold": 0.25, "sell": 0.7}
        else:
            probs = {"buy": 0.2, "hold": 0.7, "sell": 0.1}
        choice = max(probs, key=probs.get)
        return ClassificationResponse.model_validate({
            "model": self.model,
            "answers": {
                "trend_up": {"type": "noul", "noul": trend},
                "overbought": {"type": "noul", "noul": overbought},
                "fundamental_quality": {
                    "type": "score", "score": quality, "legend": dict(enumerate(QUALITY_LEVELS)),
                    "probabilities": {0: 0.0, 1: 0.0, 2: 1.0}, "confidence": 0.9},
                "valuation_risk": {"type": "noul", "noul": val_risk},
                "action": {"type": "choice", "choice": choice, "probabilities": probs, "confidence": 0.8},
            },
        })
