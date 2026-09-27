"""Performance metrics: CAGR, Sharpe, max drawdown, number of trades, hit rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe(equity: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe of daily returns (risk-free rate as an annual rate)."""
    rets = equity.pct_change().dropna() - risk_free / TRADING_DAYS
    std = rets.std()
    if len(rets) < 2 or not std > 0:
        return float("nan")
    return float(rets.mean() / std * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summarize(equity: pd.Series, trades: pd.DataFrame | None = None,
              round_trip_pnl: list[float] | None = None) -> dict:
    out = {
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": cagr(equity),
        "sharpe": sharpe(equity),
        "max_drawdown": max_drawdown(equity),
    }
    if trades is not None:
        out["trades"] = int(len(trades))
    if round_trip_pnl is not None:
        out["round_trips"] = len(round_trip_pnl)
        out["hit_rate"] = float(np.mean([p > 0 for p in round_trip_pnl])) if round_trip_pnl else float("nan")
    return out
