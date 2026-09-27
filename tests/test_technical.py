import numpy as np
import pandas as pd
import pytest

from jevbt.features.technical import (
    atr, compute_technical, drawdown, features_asof, momentum, rsi, sma,
)


def test_sma_matches_manual_mean(ohlcv):
    s = sma(ohlcv["close"], 50)
    assert s.iloc[:49].isna().all()
    assert s.iloc[49] == pytest.approx(ohlcv["close"].iloc[:50].mean())
    assert s.iloc[-1] == pytest.approx(ohlcv["close"].iloc[-50:].mean())


def test_rsi_bounds_and_extremes(ohlcv):
    r = rsi(ohlcv["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    up = pd.Series(np.arange(1.0, 40.0))
    assert rsi(up).dropna().eq(100).all()
    down = pd.Series(np.arange(40.0, 1.0, -1))
    assert rsi(down).dropna().eq(0).all()
    flat = pd.Series(np.full(30, 10.0))
    assert rsi(flat).dropna().eq(50).all()
    assert rsi(up).iloc[:14].isna().all()


def test_rsi_known_value():
    # Alternating +1/-1 moves → equal average gain and loss → RSI 50 (after warm-up converges).
    close = pd.Series(100 + np.tile([0.0, 1.0], 100))
    assert rsi(close).iloc[-1] == pytest.approx(50, abs=2)


def test_momentum():
    close = pd.Series([100.0, 110.0, 121.0])
    m = momentum(close, 2)
    assert np.isnan(m.iloc[1])
    assert m.iloc[2] == pytest.approx(0.21)


def test_atr_constant_range():
    n = 30
    close = pd.Series(np.full(n, 100.0))
    a = atr(close + 1, close - 1, close, 14)
    assert a.iloc[:13].isna().all()
    assert a.dropna().eq(2.0).all()


def test_atr_uses_gap_from_previous_close():
    close = pd.Series([100.0] * 20 + [110.0])
    high = close + 0.5
    low = close - 0.5
    tr_last = 110.5 - 100.0  # gap up: |high - prev close|
    a = atr(high, low, close, 14)
    assert a.iloc[-1] == pytest.approx(a.iloc[-2] + (tr_last - a.iloc[-2]) / 14)


def test_drawdown():
    d = drawdown(pd.Series([100.0, 120.0, 90.0, 130.0]))
    assert list(d) == pytest.approx([0.0, 0.0, -0.25, 0.0])


def test_compute_technical_columns_and_warmup(ohlcv):
    f = compute_technical(ohlcv)
    assert list(f.columns) == ["close", "sma50", "sma200", "rsi14", "mom_3m", "mom_12m", "atr_pct", "drawdown"]
    assert f.index.equals(ohlcv.index)
    assert f["sma200"].iloc[:199].isna().all() and f["sma200"].iloc[199:].notna().all()
    assert f["mom_12m"].iloc[:252].isna().all() and f["mom_12m"].iloc[252:].notna().all()
    assert (f["drawdown"] <= 0).all()
    assert (f["atr_pct"].dropna() > 0).all()


def test_compute_technical_sorts_input(ohlcv):
    shuffled = ohlcv.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(compute_technical(shuffled), compute_technical(ohlcv), check_freq=False)


def test_features_asof_uses_previous_session(ohlcv):
    f = compute_technical(ohlcv)
    t = f.index[300]
    row = features_asof(f, t)
    assert row.name == f.index[299]
    # A calendar date that is not a session (weekend) → last session before it.
    saturday = pd.Timestamp("2024-06-08")
    assert features_asof(f, saturday).name == pd.Timestamp("2024-06-07")
    with pytest.raises(ValueError):
        features_asof(f, f.index[0])


def test_no_look_ahead(ohlcv):
    """Changing data from t onwards must not change what is known at t."""
    t = ohlcv.index[300]
    before = features_asof(compute_technical(ohlcv), t)
    tampered = ohlcv.copy()
    tampered.loc[tampered.index >= t, ["open", "high", "low", "close"]] *= 3.0
    after = features_asof(compute_technical(tampered), t)
    pd.testing.assert_series_equal(before, after)
