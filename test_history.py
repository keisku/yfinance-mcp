"""Tests for history tool — market state handling and cache behavior."""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
import history as history_mod
from cache import Cache
from history import (
    _exchange_today,
    _exchange_tz,
    _is_ohlcv_finalized,
    fetch_ohlcv,
    history,
)


@pytest.fixture(autouse=True)
def _clear_exchange_tz_cache():
    _exchange_tz.cache_clear()


def _make_api_df(bars: list[tuple]) -> pd.DataFrame:
    """Build a DataFrame that mimics yfinance's history() output.

    Args:
        bars: list of (date, open, high, low, close, volume).
    """
    index = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") for d, *_ in bars],
        name="Date",
    )
    return pd.DataFrame(
        {
            "Open": [b[1] for b in bars],
            "High": [b[2] for b in bars],
            "Low": [b[3] for b in bars],
            "Close": [b[4] for b in bars],
            "Volume": [b[5] for b in bars],
        },
        index=index,
    )


class TestIsOhlcvFinalized:
    """_is_ohlcv_finalized returns True only for finalized states."""

    @pytest.mark.parametrize("state", ["POST", "POSTPOST", "CLOSED"])
    @patch("history.yf.Ticker")
    def test_finalized_states(self, mock_cls, state):
        mock_cls.return_value.info = {"marketState": state}
        assert _is_ohlcv_finalized("TEST") is True

    @pytest.mark.parametrize("state", ["PREPRE", "PRE", "REGULAR"])
    @patch("history.yf.Ticker")
    def test_non_finalized_states(self, mock_cls, state):
        mock_cls.return_value.info = {"marketState": state}
        assert _is_ohlcv_finalized("TEST") is False

    @patch("history.yf.Ticker")
    def test_missing_key_returns_false(self, mock_cls):
        mock_cls.return_value.info = {}
        assert _is_ohlcv_finalized("TEST") is False

    @patch("history.yf.Ticker")
    def test_exception_returns_false(self, mock_cls):
        mock_cls.side_effect = Exception("network error")
        assert _is_ohlcv_finalized("TEST") is False


class TestExchangeTz:
    """_exchange_tz returns the IANA tz string and is process-cached."""

    @patch("history.yf.Ticker")
    def test_returns_tz_from_fast_info(self, mock_cls):
        mock_cls.return_value.fast_info = {"timezone": "Asia/Tokyo"}
        assert _exchange_tz("7974.T") == "Asia/Tokyo"

    @patch("history.yf.Ticker")
    def test_returns_none_when_missing(self, mock_cls):
        mock_cls.return_value.fast_info = {}
        assert _exchange_tz("X") is None

    @patch("history.yf.Ticker")
    def test_returns_none_on_exception(self, mock_cls):
        mock_cls.side_effect = RuntimeError("network down")
        assert _exchange_tz("X") is None


class TestExchangeToday:
    """_exchange_today returns a date and falls back gracefully."""

    @patch("history.yf.Ticker")
    def test_returns_date(self, mock_cls):
        mock_cls.return_value.fast_info = {"timezone": "Asia/Tokyo"}
        result = _exchange_today("TEST")
        assert isinstance(result, date)

    @patch("history.yf.Ticker")
    def test_fallback_on_exception(self, mock_cls):
        mock_cls.side_effect = Exception("network error")
        assert _exchange_today("TEST") == date.today()

    @patch("history.yf.Ticker")
    def test_fallback_on_no_timezone(self, mock_cls):
        mock_cls.return_value.fast_info = {"timezone": None}
        assert _exchange_today("TEST") == date.today()


# A fixed "today" for deterministic tests.
EX_TODAY = date(2026, 3, 2)  # Monday

# A bar for today.
TODAY_BAR = (EX_TODAY, 100.0, 105.0, 95.0, 102.0, 5000)

# A bar for the previous trading day (Friday).
PREV_BAR = (date(2026, 2, 27), 98.0, 103.0, 93.0, 100.0, 4000)


class TestFetchOhlcvMarketClosed:
    """When OHLCV is finalized, today's data is cached."""

    @patch("history._is_ohlcv_finalized", return_value=True)
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    def test_today_data_cached(self, mock_api, _et, _fin, tmp_path):
        cache = Cache(path=tmp_path / "test.parquet")
        mock_api.return_value = _make_api_df([TODAY_BAR])

        with patch("history._get_cache", return_value=cache):
            df = fetch_ohlcv("TEST", "1d", "2026-03-02", "2026-03-03")

        assert len(df) == 1
        cached = cache.get("TEST", "1d", EX_TODAY, EX_TODAY)
        assert len(cached) == 1
        assert cached[0][4] == pytest.approx(102.0)

    @patch("history._is_ohlcv_finalized", return_value=True)
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    def test_past_and_today_both_cached(self, mock_api, _et, _fin, tmp_path):
        cache = Cache(path=tmp_path / "test.parquet")
        mock_api.return_value = _make_api_df([PREV_BAR, TODAY_BAR])

        with patch("history._get_cache", return_value=cache):
            df = fetch_ohlcv("TEST", "1d", "2026-02-27", "2026-03-03")

        assert len(df) == 2
        cached = cache.get("TEST", "1d", date(2026, 2, 27), EX_TODAY)
        assert len(cached) == 2


class TestFetchOhlcvMarketOpen:
    """When OHLCV is NOT finalized, today's data is returned but NOT cached."""

    @patch("history._is_ohlcv_finalized", return_value=False)
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    def test_today_data_returned_not_cached(self, mock_api, _et, _fin, tmp_path):
        cache = Cache(path=tmp_path / "test.parquet")
        mock_api.return_value = _make_api_df([TODAY_BAR])

        with patch("history._get_cache", return_value=cache):
            df = fetch_ohlcv("TEST", "1d", "2026-03-02", "2026-03-03")

        # Data is returned.
        assert len(df) == 1
        assert df.iloc[0]["Close"] == pytest.approx(102.0)
        # But NOT persisted in cache.
        cached = cache.get("TEST", "1d", EX_TODAY, EX_TODAY)
        assert len(cached) == 0

    @patch("history._is_ohlcv_finalized", return_value=False)
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    def test_past_data_cached_today_not(self, mock_api, _et, _fin, tmp_path):
        cache = Cache(path=tmp_path / "test.parquet")
        mock_api.return_value = _make_api_df([PREV_BAR, TODAY_BAR])

        with patch("history._get_cache", return_value=cache):
            df = fetch_ohlcv("TEST", "1d", "2026-02-27", "2026-03-03")

        # Both rows returned.
        assert len(df) == 2
        # Only past day is in cache.
        cached = cache.get("TEST", "1d", date(2026, 2, 27), EX_TODAY)
        assert len(cached) == 1
        assert cached[0][0] == date(2026, 2, 27)


class TestFetchOhlcvHistoricalRange:
    """When today is outside the requested range, market state is not queried."""

    @patch("history._is_ohlcv_finalized")
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    def test_no_market_state_query(self, mock_api, _et, mock_fin, tmp_path):
        cache = Cache(path=tmp_path / "test.parquet")
        past_bar = (date(2026, 2, 20), 90.0, 95.0, 85.0, 92.0, 3000)
        mock_api.return_value = _make_api_df([past_bar])

        with patch("history._get_cache", return_value=cache):
            df = fetch_ohlcv("TEST", "1d", "2026-02-20", "2026-02-21")

        assert len(df) == 1
        # _is_ohlcv_finalized should NOT have been called.
        mock_fin.assert_not_called()


class TestHistoryTzReporting:
    """Regression: ``_build_response_from_cache`` used to hardcode ``tz: 'UTC'``,
    mislabeling daily bars from non-UTC exchanges (Tokyo, London, etc.).
    """

    @patch("history._is_ohlcv_finalized", return_value=True)
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    @patch("history.yf.Ticker")
    def test_cache_path_returns_exchange_tz(
        self, mock_ticker_cls, mock_api, _et, _fin, tmp_path, monkeypatch
    ):
        cache = Cache(path=tmp_path / "test.parquet")
        monkeypatch.setattr(history_mod, "_get_cache", lambda: cache)
        mock_ticker_cls.return_value.fast_info = {"timezone": "Asia/Tokyo"}
        mock_api.return_value = _make_api_df([PREV_BAR])

        result = history("7974.T", "1d", "2026-02-27", "2026-02-28")
        assert result["tz"] == "Asia/Tokyo"


class TestFetchOhlcvFutureBarSkipped:
    """Bars with date > exchange_today are silently dropped."""

    @patch("history._is_ohlcv_finalized", return_value=True)
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    def test_future_bar_dropped(self, mock_api, _et, _fin, tmp_path):
        cache = Cache(path=tmp_path / "test.parquet")
        future_bar = (date(2026, 3, 3), 110.0, 115.0, 105.0, 112.0, 6000)
        mock_api.return_value = _make_api_df([TODAY_BAR, future_bar])

        with patch("history._get_cache", return_value=cache):
            df = fetch_ohlcv("TEST", "1d", "2026-03-02", "2026-03-04")

        assert len(df) == 1
        assert df.iloc[0]["Close"] == pytest.approx(102.0)


class TestFetchOhlcvHolidayDetection:
    """Holiday detection respects finalized state."""

    @patch("history._is_ohlcv_finalized", return_value=True)
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    def test_holiday_on_today_when_finalized(self, mock_api, _et, _fin, tmp_path):
        """If market is finalized and today has no data, mark as holiday."""
        cache = Cache(path=tmp_path / "test.parquet")
        # API returns no bar for today (it's a holiday).
        mock_api.return_value = _make_api_df([PREV_BAR])

        with patch("history._get_cache", return_value=cache):
            df = fetch_ohlcv("TEST", "1d", "2026-02-27", "2026-03-03")

        # Only the prev bar is in the result (holiday rows are excluded by cache.get).
        assert len(df) == 1
        # But today should be in cached_dates (as a holiday sentinel).
        all_dates = cache.cached_dates("TEST", "1d", date(2026, 2, 27), EX_TODAY)
        assert EX_TODAY in all_dates

    @patch("history._is_ohlcv_finalized", return_value=False)
    @patch("history._exchange_today", return_value=EX_TODAY)
    @patch("history._fetch_api")
    def test_no_holiday_on_today_when_not_finalized(
        self, mock_api, _et, _fin, tmp_path
    ):
        """If market is not finalized, today must NOT be marked as holiday."""
        cache = Cache(path=tmp_path / "test.parquet")
        mock_api.return_value = _make_api_df([PREV_BAR])

        with patch("history._get_cache", return_value=cache):
            df = fetch_ohlcv("TEST", "1d", "2026-02-27", "2026-03-03")

        assert len(df) == 1
        all_dates = cache.cached_dates("TEST", "1d", date(2026, 2, 27), EX_TODAY)
        assert EX_TODAY not in all_dates
