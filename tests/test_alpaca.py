import pandas as pd
import pytest

from conftest import API_KEY, FIXED_TODAY, FakeResponse, FakeSession
from jevbt.ingest.alpaca import AlpacaClient, AlpacaError, AlpacaPlanError, normalize_bars


def bar_rows(start: str, end_exclusive: str) -> list[dict]:
    """Business-day bars in Alpaca's shape: stamped at midnight New York time, in UTC."""
    days = pd.bdate_range(start, pd.Timestamp(end_exclusive) - pd.Timedelta(days=1))
    return [
        {"t": d.tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
         "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100, "n": 10, "vw": 1.3}
        for d in days
    ]


def bars_handler(path, params):
    assert path == "v2/stocks/AAPL/bars"
    return FakeResponse(200, {"bars": bar_rows(params["start"], params["end"]), "symbol": "AAPL",
                              "next_page_token": None})


def make_client(settings, handler, **kw):
    session = FakeSession(handler)
    client = AlpacaClient(settings, session=session, sleep=lambda s: None, today=lambda: FIXED_TODAY, **kw)
    return client, session


# ---------- normalization ----------

def test_normalize_bars_uses_new_york_session_date():
    # 05:00Z in winter (EST) and 04:00Z in summer (EDT) are both midnight in New York.
    df = normalize_bars([{"t": "2024-07-01T04:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1},
                         {"t": "2024-01-02T05:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}])
    assert list(df["date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-07-01")]
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns)
    assert df["date"].dt.tz is None


def test_normalize_bars_empty():
    df = normalize_bars(None)
    assert df.empty and "close" in df.columns


# ---------- prices + cache ----------

def test_prices_request_shape_and_headers(settings):
    client, session = make_client(settings, bars_handler)
    df = client.prices("aapl", "2024-01-01", "2024-03-31")
    path, params = session.calls[0]
    assert params["timeframe"] == "1Day" and params["adjustment"] == "split" and params["feed"] == "sip"
    assert (params["start"], params["end"]) == ("2024-01-01", "2024-04-01")  # end + 1: bars are stamped after 00:00Z
    assert session.headers[0] == {"APCA-API-KEY-ID": "test-key-id", "APCA-API-SECRET-KEY": API_KEY}
    assert API_KEY not in str(params)
    assert df.index.name == "date" and df.index.is_monotonic_increasing
    assert df.index.max() == pd.Timestamp("2024-03-29")
    assert len(df) == len(pd.bdate_range("2024-01-01", "2024-03-31"))
    assert (settings.raw_dir / "prices_alpaca" / "sip" / "AAPL.parquet").exists()


def test_prices_second_call_uses_cache(settings):
    client, session = make_client(settings, bars_handler)
    client.prices("AAPL", "2024-01-01", "2024-03-31")
    second = client.prices("AAPL", "2024-02-01", "2024-02-29")
    assert len(session.calls) == 1
    assert second.index.min() >= pd.Timestamp("2024-02-01") and second.index.max() <= pd.Timestamp("2024-02-29")


def test_prices_extends_only_missing_ranges(settings):
    client, session = make_client(settings, bars_handler)
    client.prices("AAPL", "2024-03-01", "2024-03-31")
    df = client.prices("AAPL", "2024-01-01", "2024-05-31")
    ranges = [(p["start"], p["end"]) for _, p in session.calls]
    assert ranges == [("2024-03-01", "2024-04-01"), ("2024-01-01", "2024-03-01"), ("2024-04-01", "2024-06-01")]
    assert df.index.is_unique
    assert len(df) == len(pd.bdate_range("2024-01-01", "2024-05-31"))


def test_prices_follow_pagination(settings):
    def handler(path, params):
        rows = bar_rows(params["start"], params["end"])
        if "page_token" not in params:
            return FakeResponse(200, {"bars": rows[:10], "next_page_token": "abc"})
        assert params["page_token"] == "abc"
        return FakeResponse(200, {"bars": rows[10:], "next_page_token": None})

    client, session = make_client(settings, handler)
    df = client.prices("AAPL", "2024-01-01", "2024-01-31")
    assert len(session.calls) == 2
    assert len(df) == len(pd.bdate_range("2024-01-01", "2024-01-31"))


def test_prices_never_cover_today_or_future(settings):
    client, session = make_client(settings, bars_handler)
    client.prices("AAPL", "2026-09-01", "2026-12-31")
    assert session.calls[0][1]["end"] == "2026-09-27"  # yesterday (09-26) + 1


def test_feed_is_part_of_cache(settings):
    client, session = make_client(settings, bars_handler)
    client.prices("AAPL", "2024-01-01", "2024-01-31")
    iex, iex_session = make_client(settings.model_copy(update={"alpaca_feed": "iex"}), bars_handler)
    iex.prices("AAPL", "2024-01-01", "2024-01-31")
    assert len(iex_session.calls) == 1 and iex_session.calls[0][1]["feed"] == "iex"


def test_empty_range_with_no_bars(settings):
    client, _ = make_client(settings, lambda p, q: FakeResponse(200, {"bars": None, "next_page_token": None}))
    assert client.prices("AAPL", "2024-01-01", "2024-01-31").empty


# ---------- errors, retries, secrets ----------

def test_missing_keys(settings):
    client, session = make_client(settings.model_copy(update={"alpaca_api_secret_key": ""}), bars_handler)
    with pytest.raises(AlpacaError, match="ALPACA_API_KEY_ID"):
        client.prices("AAPL", "2024-01-01", "2024-01-31")
    assert session.calls == []


@pytest.mark.parametrize("status", [401, 403])
def test_auth_and_plan_errors_are_not_retried(settings, status):
    client, session = make_client(settings, lambda p, q: FakeResponse(status, f'{{"message": "denied {API_KEY}"}}'))
    with pytest.raises(AlpacaPlanError) as exc:
        client.prices("AAPL", "2024-01-01", "2024-01-31")
    assert len(session.calls) == 1
    assert API_KEY not in str(exc.value)


def test_rate_limit_is_retried(settings):
    responses = iter([FakeResponse(429, "too many requests")])

    def handler(path, params):
        return next(responses, None) or bars_handler(path, params)

    client, session = make_client(settings, handler)
    assert len(client.prices("AAPL", "2024-01-01", "2024-01-31")) > 0
    assert len(session.calls) == 2


def test_persistent_server_error(settings):
    client, session = make_client(settings, lambda p, q: FakeResponse(500, "boom"), max_retries=2)
    with pytest.raises(AlpacaError, match="HTTP 500"):
        client.prices("AAPL", "2024-01-01", "2024-01-31")
    assert len(session.calls) == 3
