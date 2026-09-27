"""One-off probe of the FMP "stable" endpoints used by the backtest (~7 requests).

Saves raw responses to data/probe/ and prints status, row count, columns, date range
and which publication-date fields exist. The API key is never printed.
"""

from __future__ import annotations

import json
import sys

import requests

from jevbt.config import load_settings

SYMBOL = "AAPL"
DATE_FIELDS = ("date", "filingDate", "acceptedDate", "fillingDate")

PROBES = {
    "prices": ("historical-price-eod/full", {"from": "2024-01-01", "to": "2024-03-31"}),
    "income": ("income-statement", {"period": "quarter", "limit": 8}),
    "balance": ("balance-sheet-statement", {"period": "quarter", "limit": 8}),
    "cashflow": ("cash-flow-statement", {"period": "quarter", "limit": 8}),
    "income_annual": ("income-statement", {"period": "annual", "limit": 5}),
    "transcript_dates": ("earning-call-transcript-dates", {}),
    "transcript": ("earning-call-transcript", {"year": 2024, "quarter": 1}),
}


def main() -> int:
    s = load_settings()
    if not s.fmp_api_key:
        print("FMP_API_KEY not found in .env")
        return 1
    out = s.data_dir / "probe"
    out.mkdir(parents=True, exist_ok=True)
    for name, (path, params) in PROBES.items():
        r = requests.get(f"{s.fmp_base_url}/{path}",
                         params={"symbol": SYMBOL, **params, "apikey": s.fmp_api_key}, timeout=30)
        text = r.text.replace(s.fmp_api_key, "***")
        (out / f"{name}.json").write_text(text, encoding="utf-8")
        print(f"\n== {name}: /{path} -> HTTP {r.status_code}")
        try:
            body = json.loads(text)
        except ValueError:
            print("   non-JSON body:", text[:200])
            continue
        if not isinstance(body, list) or not body or not isinstance(body[0], dict):
            print("   body:", str(body)[:300])
            continue
        cols = list(body[0].keys())
        print(f"   rows={len(body)} columns={cols}")
        for f in DATE_FIELDS:
            if f in body[0]:
                vals = sorted(str(row.get(f)) for row in body)
                print(f"   {f}: {vals[0]} .. {vals[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
