"""Alpaca Market Data v2 client for daily stock bars, with a local Parquet cache.

Prices only: fundamentals stay on FMP (ingest/fmp.py). Bars are requested with `adjustment=split`, like FMP's
EOD prices, because the statements are restated for splits too → `close / eps_diluted` stays consistent.
The feed (sip / iex) is part of the cache path: the two feeds give different bars.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import requests

from jevbt.config import Settings
from jevbt.ingest.cache import load_range

BARS_PATH = "v2/stocks/{symbol}/bars"
BAR_COLUMNS = {"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
               "n": "trade_count", "vw": "vwap"}
MARKET_TZ = "America/New_York"
PAGE_LIMIT = 10_000
RETRY_STATUSES = {429, 500, 502, 503, 504}


class AlpacaError(RuntimeError):
    """Unexpected Alpaca response."""


class AlpacaPlanError(AlpacaError):
    """Bad credentials (401) or data not allowed by the subscription (403), e.g. SIP data < 15 min old."""


def normalize_bars(bars: list[dict] | None) -> pd.DataFrame:
    """Alpaca bars → OHLCV frame with a naive `date` column (the session's calendar date), sorted."""
    df = pd.DataFrame(bars or [])
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "date" else "float64")
                             for c in BAR_COLUMNS.values()})
    df = df.rename(columns=BAR_COLUMNS)
    df = df[[c for c in BAR_COLUMNS.values() if c in df.columns]]
    # Daily bars are stamped at midnight New York time (04:00 or 05:00 UTC) → keep the session date.
    df["date"] = (pd.to_datetime(df["date"], utc=True).dt.tz_convert(MARKET_TZ)
                  .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"))
    return df.sort_values("date", kind="stable").reset_index(drop=True)


class AlpacaClient:
    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], date] = date.today,
        max_retries: int = 4,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self._sleep = sleep
        self._today = today
        self.max_retries = max_retries
        self.raw_dir = settings.raw_dir

    def prices(self, symbol: str, start: str | date, end: str | date, refresh: bool = False) -> pd.DataFrame:
        """Daily split-adjusted OHLCV in [start, end], indexed by date (same shape as FMPClient.prices)."""
        symbol = symbol.upper()
        start = pd.Timestamp(start).date()
        # Never mark today or the future as covered: today's bar may still be incomplete.
        end = min(pd.Timestamp(end).date(), self._today() - timedelta(days=1))
        if end < start:
            raise ValueError(f"empty range: {start} .. {end}")
        path = self.raw_dir / "prices_alpaca" / self.settings.alpaca_feed / f"{symbol}.parquet"
        return load_range(path, start, end, lambda a, b: self._bars(symbol, a, b), refresh)

    # ---------- internals ----------

    def _bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "timeframe": "1Day",
            "start": start.isoformat(),
            # A date-only `end` means 00:00 UTC, before that day's bar (stamped 04:00/05:00 UTC) → ask for end + 1.
            "end": (end + timedelta(days=1)).isoformat(),
            "adjustment": "split",
            "feed": self.settings.alpaca_feed,
            "limit": PAGE_LIMIT,
            "sort": "asc",
        }
        bars: list[dict] = []
        token = None
        while True:
            body = self._get(BARS_PATH.format(symbol=symbol), {**params, **({"page_token": token} if token else {})})
            bars.extend(body.get("bars") or [])
            token = body.get("next_page_token")
            if not token:
                break
        df = normalize_bars(bars)
        return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)

    def _redact(self, text: str) -> str:
        secret = self.settings.alpaca_api_secret_key
        return text.replace(secret, "***") if secret else text

    def _get(self, path: str, params: dict) -> dict:
        if not (self.settings.alpaca_api_key_id and self.settings.alpaca_api_secret_key):
            raise AlpacaError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set")
        url = f"{self.settings.alpaca_data_url.rstrip('/')}/{path}"
        headers = {"APCA-API-KEY-ID": self.settings.alpaca_api_key_id,
                   "APCA-API-SECRET-KEY": self.settings.alpaca_api_secret_key}
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise AlpacaError(self._redact(f"{path}: {e}")) from None
                self._sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                try:
                    body = r.json()
                except ValueError:
                    raise AlpacaError(self._redact(f"{path}: non-JSON body: {r.text[:200]}")) from None
                if not isinstance(body, dict):
                    raise AlpacaError(self._redact(f"{path}: unexpected body: {str(body)[:200]}"))
                return body
            if r.status_code in (401, 403):
                raise AlpacaPlanError(self._redact(f"{path} (HTTP {r.status_code}): {r.text[:300]}"))
            if r.status_code in RETRY_STATUSES and attempt < self.max_retries:
                # Free plan: 200 requests/min → a 429 clears within the minute.
                self._sleep(15 * (attempt + 1) if r.status_code == 429 else 2 ** attempt)
                continue
            raise AlpacaError(self._redact(f"{path}: HTTP {r.status_code}: {r.text[:300]}"))
        raise AssertionError("unreachable")
