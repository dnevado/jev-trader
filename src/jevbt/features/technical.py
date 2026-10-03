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


def slope(s: pd.Series, n: int) -> pd.Series:
    """Relative change of a series over n sessions (e.g. of the SMA200: > 0 rising, < 0 falling)."""
    return s / s.shift(n) - 1


def share_above(close: pd.Series, level: pd.Series, n: int) -> pd.Series:
    """Fraction of the last n sessions that closed above `level` (NaN until n valid sessions)."""
    above = (close > level).astype(float).where(level.notna())
    return above.rolling(n, min_periods=n).mean()


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman's efficiency ratio in [0, 1]: |net move over n| / sum of |daily moves|. High = clean trend."""
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return (close - close.shift(n)).abs() / path


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (trend strength, direction-free; > 20-25 = trending)."""
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(high, low, close, n)
    plus_di = 100 * _wilder(plus_dm, n) / atr_
    minus_di = 100 * _wilder(minus_dm, n) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _wilder(dx, n)


def compute_technical(prices: pd.DataFrame) -> pd.DataFrame:
    """OHLCV indexed by date (columns open/high/low/close) → indicator table with the same index."""
    prices = prices.sort_index()
    close = prices["close"]
    sma200 = sma(close, 200)
    return pd.DataFrame(
        {
            "close": close,
            "sma50": sma(close, 50),
            "sma200": sma200,
            "rsi14": rsi(close, 14),
            "mom_3m": momentum(close, TRADING_DAYS_3M),
            "mom_12m": momentum(close, TRADING_DAYS_12M),
            "atr_pct": atr(prices["high"], prices["low"], close, 14) / close,
            "drawdown": drawdown(close),
            # Trend-confidence inputs (strategy.TrendConfidenceStrategy); not part of the Jev state.
            "sma200_slope": slope(sma200, 20),
            "above200_share": share_above(close, sma200, 60),
            "eff_ratio": efficiency_ratio(close, 60),
            "adx14": adx(prices["high"], prices["low"], close, 14),
        },
        index=prices.index,
    )


def features_asof(features: pd.DataFrame, t: str | pd.Timestamp) -> pd.Series:
    """Last row strictly before session t (anti look-ahead: decisions at t use data ≤ t-1)."""
    past = features.loc[features.index < pd.Timestamp(t)]
    if past.empty:
        raise ValueError(f"no data before {t}")
    return past.iloc[-1]
