"""Alpaca Trading API v2 client restricted to the PAPER endpoint (no real money).

The constructor refuses any base URL that is not a paper-trading host, so this module can never place orders
on a live account. Keys: ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY, falling back to the data keys.
"""

from __future__ import annotations

from urllib.parse import urlparse

import requests

from jevbt.config import Settings

PAPER_HOST = "paper-api.alpaca.markets"


class BrokerError(RuntimeError):
    """Unexpected or rejected Alpaca Trading API response."""


class AlpacaPaperBroker:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        url = settings.alpaca_paper_url.rstrip("/")
        if urlparse(url).hostname != PAPER_HOST:
            raise BrokerError(f"refusing a non-paper trading endpoint: {url!r} (only {PAPER_HOST} is allowed)")
        self.base_url = url
        self.key_id = settings.alpaca_paper_key_id or settings.alpaca_api_key_id
        self.secret = settings.alpaca_paper_secret_key or settings.alpaca_api_secret_key
        if not (self.key_id and self.secret):
            raise BrokerError("ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY (or the data keys) are not set")
        self.session = session or requests.Session()

    # ---------- read-only ----------

    def account(self) -> dict:
        a = self._request("GET", "v2/account")
        return {"equity": float(a["equity"]), "cash": float(a["cash"]), "status": a.get("status"),
                "shorting_enabled": bool(a.get("shorting_enabled")), "trading_blocked": bool(a.get("trading_blocked"))}

    def positions(self) -> dict[str, dict]:
        """By symbol: {"qty": signed shares (negative = short), "market_value": signed value}."""
        out = {}
        for p in self._request("GET", "v2/positions"):
            sign = -1 if p.get("side") == "short" else 1
            out[p["symbol"]] = {"qty": sign * abs(float(p["qty"])), "market_value": sign * abs(float(p["market_value"]))}
        return out

    def calendar(self, start: str, end: str) -> list[str]:
        """Trading session dates (YYYY-MM-DD) in [start, end]."""
        return [d["date"] for d in self._request("GET", "v2/calendar", params={"start": start, "end": end})]

    def asset(self, symbol: str) -> dict:
        return self._request("GET", f"v2/assets/{symbol}")

    # ---------- orders ----------

    def submit_market_order(self, symbol: str, qty: int, side: str, time_in_force: str, client_order_id: str) -> dict:
        if qty <= 0 or side not in ("buy", "sell"):
            raise BrokerError(f"invalid order: {side} {qty} {symbol}")
        return self._request("POST", "v2/orders", json={
            "symbol": symbol, "qty": str(int(qty)), "side": side, "type": "market",
            "time_in_force": time_in_force, "client_order_id": client_order_id})

    # ---------- internals ----------

    def _request(self, method: str, path: str, params: dict | None = None, json: dict | None = None):
        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}
        url = f"{self.base_url}/{path}"
        try:
            r = self.session.request(method, url, params=params, json=json, headers=headers, timeout=30)
        except requests.RequestException as e:
            raise BrokerError(f"{method} {path}: {e}") from None
        if r.status_code not in (200, 201):
            raise BrokerError(f"{method} {path}: HTTP {r.status_code}: {r.text[:300].replace(self.secret, '***')}")
        return r.json()
