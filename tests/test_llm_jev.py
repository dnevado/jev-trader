import json

import httpx2
import pytest

from jevbt.decision.jev import (
    QUESTIONS, JevDecider, MockJevClassifier, questions_payload, to_decision, typesafe_classifier,
)
from jevbt.llm.summarizer import FundSummary, MockSummaryLLM, Summarizer

JEV_RESPONSE = {
    "model": "typesafe/jev-1.13",
    "answers": {
        "trend_up": {"type": "noul", "noul": 0.9},
        "overbought": {"type": "noul", "noul": 0.1},
        "fundamental_quality": {"type": "score", "score": 1.7, "legend": {"0": "weak", "1": "medium", "2": "strong"},
                                "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75}, "confidence": 0.6},
        "valuation_risk": {"type": "noul", "noul": 0.4},
        "action": {"type": "choice", "choice": "buy", "probabilities": {"buy": 0.8, "hold": 0.15, "sell": 0.05},
                   "confidence": 0.75},
    },
    "usage": {"input_tokens": 500, "output_tokens": 5},
}


class CountingLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return MockSummaryLLM().invoke(messages)


def test_summarizer_caches_per_prompt(tmp_path):
    llm = CountingLLM()
    s = Summarizer(llm, "gpt-test", tmp_path)
    a = s.summarize("aapl", "2024", "prompt A")
    b = s.summarize("AAPL", "2024", "prompt A")
    assert isinstance(a, FundSummary) and a == b
    assert llm.calls == 1 and s.calls == 1
    s.summarize("AAPL", "2024", "prompt B")
    assert llm.calls == 2
    # A different model must not reuse the cache.
    Summarizer(llm, "gpt-other", tmp_path).summarize("AAPL", "2024", "prompt A")
    assert llm.calls == 3


def test_typesafe_classifier_request_and_parsing():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=JEV_RESPONSE, headers={"x-typesafe-request-id": "req-1"})

    clf = typesafe_classifier("typesafe/jev-1.13", "sk-or-test", "https://openrouter.ai/api")
    clf.client = httpx2.Client(transport=httpx2.MockTransport(handler))
    d = to_decision(clf.invoke("state text"))
    assert seen["url"] == "https://openrouter.ai/api/v1/systemone"
    assert seen["auth"] == "Bearer sk-or-test"
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert seen["body"]["state"] == "state text"
    assert set(seen["body"]["questions"]) == set(QUESTIONS)
    assert seen["body"]["questions"]["fundamental_quality"]["type"] == "score"
    assert d.action == "buy" and d.p_buy == pytest.approx(0.8)
    assert d.fundamental_quality == pytest.approx(1.7)
    assert d.request_id == "req-1"


class FakeClassifier:
    model = "typesafe/jev-1.13"

    def __init__(self):
        self.calls = 0

    def invoke(self, state):
        from langchain_typesafe.types import ClassificationResponse

        self.calls += 1
        return ClassificationResponse.model_validate(JEV_RESPONSE)


def test_jev_decider_cache(tmp_path):
    clf = FakeClassifier()
    decider = JevDecider(clf, tmp_path)
    first = decider.decide("state 1")
    assert decider.decide("state 1") == first
    assert clf.calls == 1
    decider.decide("state 2")
    assert clf.calls == 2
    assert decider.key("state 1") != JevDecider(MockJevClassifier(), tmp_path).key("state 1")


def test_questions_payload_is_stable():
    assert json.dumps(questions_payload(), sort_keys=True) == json.dumps(questions_payload(), sort_keys=True)


def test_mock_jev_reads_state():
    bull = "close: 120.00\nsma200: 100.00\nrsi14: 55.00\nmom_3m: 8.00%\nrevenue_growth_yoy: 12.00%\npe_ratio: 25.00"
    bear = "close: 80.00\nsma200: 100.00\nrsi14: 40.00\nmom_3m: -8.00%\nrevenue_growth_yoy: no data\npe_ratio: no data"
    b = to_decision(MockJevClassifier().invoke(bull))
    assert b.action == "buy" and b.trend_up > 0.75 and b.fundamental_quality == 2.0
    s = to_decision(MockJevClassifier().invoke(bear))
    assert s.action == "sell" and s.trend_up < 0.4
