"""Own backtest engine: weekly rebalance, decisions from data ≤ t-1, execution at the open of t,
costs in basis points of traded notional, multi-ticker, long-only, fractional shares.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from jevbt.features.technical import compute_technical, features_asof
from jevbt.strategy import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    round_trip_pnl: list[float] = field(default_factory=list)


def rebalance_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First session of each calendar week."""
    weeks = calendar.to_period("W")
    return calendar[~weeks.duplicated()]


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def run_backtest(
    prices: dict[str, pd.DataFrame],
    strategy: Strategy,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    initial_cash: float = 100_000.0,
    cost_bps: float = 10.0,
    log_path: Path | None = None,
    liquidate_at_end: bool = False,
) -> BacktestResult:
    """`liquidate_at_end` sells every open position at the last session's close (with costs),
    so a window's result and round trips are complete (used by walk-forward test windows)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    features = {tk: compute_technical(df) for tk, df in prices.items()}
    calendar = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    if calendar.empty:
        raise ValueError("no sessions in the backtest range")
    closes = pd.DataFrame({tk: df["close"] for tk, df in prices.items()}).sort_index().ffill()
    rebalances = set(rebalance_dates(calendar))

    cash = initial_cash
    shares = {tk: 0.0 for tk in prices}
    basis = {tk: 0.0 for tk in prices}
    trades, round_trips, equity = [], [], {}
    log = log_path.open("w", encoding="utf-8") if log_path else None

    try:
        for d in calendar:
            if d in rebalances:
                # Mark at today's open where available, else yesterday's close.
                marks = {}
                for tk, df in prices.items():
                    if d in df.index:
                        marks[tk] = df.at[d, "open"]
                    else:
                        prev = closes.loc[:d].iloc[:-1][tk].dropna()
                        marks[tk] = prev.iloc[-1] if len(prev) else np.nan
                eq_open = cash + sum(shares[tk] * marks[tk] for tk in prices if shares[tk])
                for tk, df in prices.items():
                    if d not in df.index:
                        continue
                    try:
                        technical = features_asof(features[tk], d)
                    except ValueError:
                        continue
                    price = marks[tk]
                    current_w = shares[tk] * price / eq_open if eq_open > 0 else 0.0
                    target_w, record = strategy.target(tk, d, technical, current_w)
                    order = None
                    if target_w != current_w:
                        delta = target_w * eq_open / price - shares[tk]
                        if target_w == 0:
                            delta = -shares[tk]
                        if abs(delta) > 1e-9:
                            notional = delta * price
                            cost = abs(notional) * cost_bps / 1e4
                            cash -= notional + cost
                            if delta > 0:
                                basis[tk] += notional + cost
                            else:
                                sold_frac = -delta / shares[tk]
                                released = basis[tk] * sold_frac
                                basis[tk] -= released
                                if target_w == 0:
                                    round_trips.append(-notional - cost - released)
                            shares[tk] += delta
                            if target_w == 0:
                                shares[tk] = 0.0
                            order = {"side": "buy" if delta > 0 else "sell", "shares": delta, "price": price, "cost": cost}
                            trades.append({"date": d, "ticker": tk, **order})
                    if log:
                        log.write(json.dumps({"date": d, "ticker": tk, "strategy": strategy.name,
                                              "current_weight": current_w, "target_weight": target_w,
                                              "order": order, **record}, default=_json_default) + "\n")
            row = closes.loc[d]
            if liquidate_at_end and d == calendar[-1]:
                for tk in prices:
                    if shares[tk]:
                        notional = shares[tk] * row[tk]
                        cost = abs(notional) * cost_bps / 1e4
                        cash += notional - cost
                        round_trips.append(notional - cost - basis[tk])
                        trades.append({"date": d, "ticker": tk, "side": "sell", "shares": -shares[tk],
                                       "price": row[tk], "cost": cost})
                        shares[tk], basis[tk] = 0.0, 0.0
            equity[d] = cash + sum(shares[tk] * row[tk] for tk in prices if shares[tk])
    finally:
        if log:
            log.close()

    return BacktestResult(
        equity=pd.Series(equity, name="equity"),
        trades=pd.DataFrame(trades, columns=["date", "ticker", "side", "shares", "price", "cost"]),
        round_trip_pnl=round_trips,
    )


def buy_and_hold(prices: dict[str, pd.DataFrame], start, end, initial_cash: float = 100_000.0,
                 cost_bps: float = 10.0) -> pd.Series:
    """Equal-weight buy & hold benchmark, bought at the first session's open."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    closes = pd.DataFrame({tk: df["close"] for tk, df in prices.items()}).sort_index().ffill()
    closes = closes.loc[start:end]
    first = closes.index[0]
    alloc = initial_cash * (1 - cost_bps / 1e4) / len(prices)
    units = {tk: alloc / prices[tk].at[first, "open"] for tk in prices if first in prices[tk].index}
    cash = initial_cash - alloc * len(units) / (1 - cost_bps / 1e4)
    return (closes[list(units)] * pd.Series(units)).sum(axis=1).rename("equity") + cash
