"""Point-in-time fundamentals from annual statements (free FMP plan: latest 5 fiscal years).

Ratios are computed here with pandas; FMP's "current" TTM endpoints are never used.
A fiscal year is known at rebalance date t only if all three statements were accepted before t.
FMP restates EPS and share counts for splits, so `close / eps_diluted` is consistent with
the split-adjusted prices (verified on NVDA's 2024 split, see docs/fmp_endpoints.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_COLUMNS = [
    "revenue", "revenue_growth_yoy", "gross_margin", "operating_margin", "net_margin",
    "debt_to_equity", "fcf_margin", "eps_diluted",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b != 0)


def compute_ratios(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> pd.DataFrame:
    """One row per fiscal year (index = period-end date), oldest first.

    `known_at` is the latest `accepted_date` of the three statements: the moment the full year is public.
    """
    cols = ["fiscal_year", "accepted_date"]
    df = (
        income[cols + ["revenue", "gross_profit", "operating_income", "net_income", "eps_diluted"]]
        .join(balance[["accepted_date", "total_debt", "total_stockholders_equity"]], rsuffix="_bs", how="inner")
        .join(cashflow[["accepted_date", "free_cash_flow"]], rsuffix="_cf", how="inner")
        .sort_index()
    )
    out = pd.DataFrame(index=df.index)
    out["fiscal_year"] = df["fiscal_year"].astype(str)
    out["known_at"] = df[["accepted_date", "accepted_date_bs", "accepted_date_cf"]].max(axis=1)
    out["revenue"] = df["revenue"]
    out["revenue_growth_yoy"] = df["revenue"].pct_change(fill_method=None)
    out["gross_margin"] = _safe_div(df["gross_profit"], df["revenue"])
    out["operating_margin"] = _safe_div(df["operating_income"], df["revenue"])
    out["net_margin"] = _safe_div(df["net_income"], df["revenue"])
    # Negative equity makes D/E meaningless → NaN instead of a misleading negative ratio.
    out["debt_to_equity"] = _safe_div(df["total_debt"], df["total_stockholders_equity"].where(df["total_stockholders_equity"] > 0))
    out["fcf_margin"] = _safe_div(df["free_cash_flow"], df["revenue"])
    out["eps_diluted"] = df["eps_diluted"]
    return out


def point_in_time(ratios: pd.DataFrame, t: str | pd.Timestamp, years: int = 3) -> pd.DataFrame:
    """Fiscal years fully published before t (at most `years`, oldest first). Empty if none."""
    known = ratios.loc[ratios["known_at"] < pd.Timestamp(t)]
    return known.tail(years)


def pe_ratio(close_prev: float, eps_diluted: float) -> float:
    """P/E with the previous session's close; NaN when EPS is not positive."""
    if eps_diluted is None or not np.isfinite(eps_diluted) or eps_diluted <= 0:
        return float("nan")
    return float(close_prev / eps_diluted)


def _fmt(value: float, pct: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "no data"
    return f"{value:.1%}" if pct else f"{value:,.2f}"


def ratios_table(history: pd.DataFrame) -> str:
    """Compact text table of the published fiscal years, for prompts and the Jev state."""
    lines = ["fiscal_year | revenue_usd_bn | revenue_growth_yoy | gross_margin | operating_margin | "
             "net_margin | debt_to_equity | fcf_margin | eps_diluted"]
    for _, r in history.iterrows():
        lines.append(" | ".join([
            f"FY{r['fiscal_year']}", _fmt(r["revenue"] / 1e9), _fmt(r["revenue_growth_yoy"], True),
            _fmt(r["gross_margin"], True), _fmt(r["operating_margin"], True), _fmt(r["net_margin"], True),
            _fmt(r["debt_to_equity"]), _fmt(r["fcf_margin"], True), _fmt(r["eps_diluted"]),
        ]))
    return "\n".join(lines)


def build_user_prompt(company: str, history: pd.DataFrame) -> str:
    """Summarizer user prompt (CLAUDE.md §6) for the latest fiscal year in `history`.

    The analysis date is the publication date of that fiscal year, so the prompt (and its cache entry)
    is the same for every rebalance date until the next annual filing. No earnings-call transcripts
    on the free plan, so the prompt says so explicitly.
    """
    latest = history.iloc[-1]
    return (
        f"Company: {company}\n"
        f"Analysis date: {latest['known_at']:%Y-%m-%d}\n"
        "(All the following documents predate this date.)\n"
        f"<ratios_annual>\n{ratios_table(history)}\n</ratios_annual>\n"
        "<earnings_call_extract>not available</earnings_call_extract>\n"
        "Generate the structured summary."
    )
