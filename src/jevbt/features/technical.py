"""Technical indicators with pandas (never an LLM).

All functions are causal: the value at date d only uses data up to and including d.
Use `features_asof(features, t)` to get what is known *before* the rebalance session t (data ≤ t-1).
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS_3M = 63
TRADING_DAYS_12M = 252


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]; 100 when there were no losses, 50 on a completely flat window."""
    delta = close.diff()
    avg_gain = _wilder(delta.clip(lower=0), n)
    avg_loss = _wilder(-delta.clip(upper=0), n)
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def momentum(close: pd.Series, n: int) -> pd.Series:
    """Simple return over the last n sessions."""
    return close / close.shift(n) - 1


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)
    true_range.iloc[0] = high.iloc[0] - low.iloc[0]
    return _wilder(true_range, n)


def drawdown(close: pd.Series) -> pd.Series:
    """Distance to the running maximum (≤ 0)."""
    return close / close.cummax() - 1


def compute_technical(prices: pd.DataFrame) -> pd.DataFrame:
    """OHLCV indexed by date (columns open/high/low/close) → indicator table with the same index."""
    prices = prices.sort_index()
    close = prices["close"]
    return pd.DataFrame(
        {
            "close": close,
            "sma50": sma(close, 50),
            "sma200": sma(close, 200),
            "rsi14": rsi(close, 14),
            "mom_3m": momentum(close, TRADING_DAYS_3M),
            "mom_12m": momentum(close, TRADING_DAYS_12M),
            "atr_pct": atr(prices["high"], prices["low"], close, 14) / close,
            "drawdown": drawdown(close),
        },
        index=prices.index,
    )


def features_asof(features: pd.DataFrame, t: str | pd.Timestamp) -> pd.Series:
    """Last row strictly before session t (anti look-ahead: decisions at t use data ≤ t-1)."""
    past = features.loc[features.index < pd.Timestamp(t)]
    if past.empty:
        raise ValueError(f"no data before {t}")
    return past.iloc[-1]
