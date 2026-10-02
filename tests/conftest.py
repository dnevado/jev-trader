from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from jevbt.config import Settings

API_KEY = "test-secret-key-123"


def make_ohlcv(n: int = 400, seed: int = 7, start: str = "2023-01-02") -> pd.DataFrame:
    """Deterministic synthetic OHLCV on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


class FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else str(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Stands in for requests.Session: `handler(path, params) -> FakeResponse`, records every call."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, dict]] = []
        self.headers: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        # FMP: path after /stable/; Alpaca: path after the host (v2/stocks/...).
        path = url.split("/stable/", 1)[1] if "/stable/" in url else url.split("://", 1)[1].split("/", 1)[1]
        self.calls.append((path, dict(params or {})))
        self.headers.append(dict(headers or {}))
        return self.handler(path, params or {})


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(fmp_api_key=API_KEY, alpaca_api_key_id="test-key-id", alpaca_api_secret_key=API_KEY,
                    data_dir=tmp_path / "data", fmp_daily_budget=50)


FIXED_TODAY = date(2026, 9, 27)
