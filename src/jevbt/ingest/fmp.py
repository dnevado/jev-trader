"""FMP "stable" REST client with a local Parquet cache.

Endpoints and field names are the ones verified in docs/fmp_endpoints.md.
Every request goes through a daily budget counter; cached data never hits the network.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import requests

from jevbt.config import Settings

StatementKind = Literal["income", "balance", "cashflow"]
Period = Literal["quarter", "annual"]

PRICES_PATH = "historical-price-eod/full"
STATEMENT_PATHS: dict[str, str] = {
    "income": "income-statement",
    "balance": "balance-sheet-statement",
    "cashflow": "cash-flow-statement",
}
TRANSCRIPT_DATES_PATH = "earning-call-transcript-dates"
TRANSCRIPT_PATH = "earning-call-transcript"

DATE_COLUMNS = ("date", "filing_date", "accepted_date")
RETRY_STATUSES = {429, 500, 502, 503, 504}


class FMPError(RuntimeError):
    """Unexpected FMP response."""


class FMPPlanError(FMPError):
    """Endpoint or parameter not available on the current FMP subscription (HTTP 402/403)."""


class FMPBudgetError(FMPError):
    """The daily request budget is exhausted."""


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def normalize(records: list[dict]) -> pd.DataFrame:
    """Records → DataFrame: legacy typo fixed, snake_case columns, parsed dates, sorted by date."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.rename(columns={"fillingDate": "filingDate"}).rename(columns=_snake)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if "date" in df.columns:
        df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class FMPClient:
    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], date] = date.today,
        max_retries: int = 3,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self._sleep = sleep
        self._today = today
        self.max_retries = max_retries
        self.raw_dir = settings.raw_dir

    # ---------- public API ----------

    def prices(self, symbol: str, start: str | date, end: str | date, refresh: bool = False) -> pd.DataFrame:
        """Daily split-adjusted OHLCV in [start, end], indexed by date."""
        symbol = symbol.upper()
        start = pd.Timestamp(start).date()
        # Never mark today or the future as covered: today's bar may still be incomplete.
        end = min(pd.Timestamp(end).date(), self._today() - timedelta(days=1))
        if end < start:
            raise ValueError(f"empty range: {start} .. {end}")
        path = self.raw_dir / "prices" / f"{symbol}.parquet"
        meta_path = path.with_suffix(".meta.json")

        cached = pd.read_parquet(path) if path.exists() and not refresh else None
        covered = json.loads(meta_path.read_text()) if cached is not None and meta_path.exists() else None

        if covered is None:
            missing = [(start, end)]
            cov_start, cov_end = start, end
        else:
            cov_start = date.fromisoformat(covered["start"])
            cov_end = date.fromisoformat(covered["end"])
            missing = []
            if start < cov_start:
                missing.append((start, cov_start - timedelta(days=1)))
            if end > cov_end:
                missing.append((cov_end + timedelta(days=1), end))
            cov_start, cov_end = min(start, cov_start), max(end, cov_end)

        if missing:
            frames = [] if cached is None else [cached]
            for a, b in missing:
                frames.append(normalize(self._get(PRICES_PATH, {"symbol": symbol, "from": a.isoformat(), "to": b.isoformat()})))
            frames = [f for f in frames if not f.empty]
            merged = (
                pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
                if frames else pd.DataFrame(columns=["date"])
            )
            _write_parquet(merged.reset_index(drop=True), path)
            meta_path.write_text(json.dumps({"start": cov_start.isoformat(), "end": cov_end.isoformat()}))
            cached = merged

        df = cached.set_index("date")
        return df.loc[pd.Timestamp(start):pd.Timestamp(end)]

    def statements(self, symbol: str, kind: StatementKind, period: Period = "quarter",
                   refresh: bool = False) -> pd.DataFrame:
        """Financial statements, oldest first, indexed by period-end date.

        FMP only returns the latest `fmp_statement_limit` periods on the free plan, so refreshes are
        merged into the cache: history accumulates over time instead of being overwritten.
        """
        symbol = symbol.upper()
        if kind not in STATEMENT_PATHS:
            raise ValueError(f"unknown statement kind: {kind}")
        path = self.raw_dir / f"{kind}_{period}" / f"{symbol}.parquet"
        cached = pd.read_parquet(path) if path.exists() else None
        if cached is None or refresh:
            fresh = normalize(self._get(STATEMENT_PATHS[kind], {
                "symbol": symbol, "period": period, "limit": self.settings.fmp_statement_limit}))
            frames = [f for f in (cached, fresh) if f is not None and not f.empty]
            merged = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(["date", "period"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
                if frames else fresh
            )
            _write_parquet(merged, path)
            cached = merged
        return cached.set_index("date") if "date" in cached.columns else cached

    def transcript_dates(self, symbol: str, refresh: bool = False) -> pd.DataFrame:
        """Earning-call transcript dates (not available on the free plan → FMPPlanError)."""
        symbol = symbol.upper()
        path = self.raw_dir / "transcript_dates" / f"{symbol}.parquet"
        if path.exists() and not refresh:
            return pd.read_parquet(path)
        df = normalize(self._get(TRANSCRIPT_DATES_PATH, {"symbol": symbol}))
        _write_parquet(df, path)
        return df

    def transcript(self, symbol: str, year: int, quarter: int) -> pd.DataFrame:
        """One earning-call transcript (not available on the free plan → FMPPlanError). Immutable once cached."""
        symbol = symbol.upper()
        path = self.raw_dir / "transcripts" / f"{symbol}_{year}Q{quarter}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        df = normalize(self._get(TRANSCRIPT_PATH, {"symbol": symbol, "year": year, "quarter": quarter}))
        _write_parquet(df, path)
        return df

    # ---------- internals ----------

    def _redact(self, text: str) -> str:
        key = self.settings.fmp_api_key
        return text.replace(key, "***") if key else text

    def _spend_budget(self) -> None:
        path = self.raw_dir / "_budget.json"
        today = self._today().isoformat()
        state = json.loads(path.read_text()) if path.exists() else {}
        count = state.get("count", 0) if state.get("date") == today else 0
        if count >= self.settings.fmp_daily_budget:
            raise FMPBudgetError(f"FMP daily budget exhausted ({count}/{self.settings.fmp_daily_budget} requests on {today})")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"date": today, "count": count + 1}))

    def _get(self, path: str, params: dict) -> list[dict]:
        if not self.settings.fmp_api_key:
            raise FMPError("FMP_API_KEY is not set")
        url = f"{self.settings.fmp_base_url}/{path}"
        for attempt in range(self.max_retries + 1):
            self._spend_budget()
            try:
                r = self.session.get(url, params={**params, "apikey": self.settings.fmp_api_key}, timeout=30)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise FMPError(self._redact(f"{path}: {e}")) from None
                self._sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                try:
                    body = r.json()
                except ValueError:
                    raise FMPError(self._redact(f"{path}: non-JSON body: {r.text[:200]}")) from None
                if isinstance(body, dict) and "Error Message" in body:
                    raise FMPError(self._redact(f"{path}: {body['Error Message']}"))
                if not isinstance(body, list):
                    raise FMPError(self._redact(f"{path}: unexpected body: {str(body)[:200]}"))
                return body
            if r.status_code in (402, 403):
                raise FMPPlanError(self._redact(f"{path} (HTTP {r.status_code}): {r.text[:300]}"))
            if r.status_code in RETRY_STATUSES and attempt < self.max_retries:
                self._sleep(2 ** attempt)
                continue
            raise FMPError(self._redact(f"{path}: HTTP {r.status_code}: {r.text[:300]}"))
        raise AssertionError("unreachable")
