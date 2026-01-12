"""Tests for fetch_japan_etf_expense helper function."""

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

from yfinance_mcp.helpers import fetch_japan_etf_expense

CACHE_BACKEND_PATH = "yfinance_mcp.cache_duckdb.DuckDBCacheBackend"


@pytest.fixture
def mock_cache():
    """Create a mock DuckDBCacheBackend."""
    cache = MagicMock()
    cache.get_etf_expense.return_value = (None, False)
    return cache


class TestCacheHit:
    """Tests for cache hit scenarios."""

    @pytest.mark.parametrize(
        "cached_value,expected",
        [
            pytest.param(0.15, 0.15, id="returns_cached_value"),
            pytest.param(None, None, id="returns_cached_none"),
        ],
    )
    def test_cache_hit(self, mock_cache, cached_value, expected):
        """When cache has valid data, return it without HTTP request."""
        mock_cache.get_etf_expense.return_value = (cached_value, True)

        with patch(CACHE_BACKEND_PATH, return_value=mock_cache):
            result = fetch_japan_etf_expense("282A.T")

        assert result == expected
        mock_cache.close.assert_called_once()


class TestScraping:
    """Tests for web scraping scenarios."""

    @pytest.mark.parametrize(
        "html,expected",
        [
            pytest.param(
                "<div>信託報酬</div><span>0.11%</span>",
                0.11,
                id="with_percent_sign",
            ),
            pytest.param(
                "<div>信託報酬</div><td>0.649%</td>",
                0.649,
                id="different_tag",
            ),
            pytest.param(
                "<div>信託報酬</div><span>0.11</span><",
                0.11,
                id="without_percent_sign",
            ),
        ],
    )
    def test_parses_expense_ratio(self, mock_cache, html, expected):
        """Should parse expense ratio from various HTML formats."""
        resp = MagicMock()
        resp.status_code = 200
        resp.text = html

        with (
            patch(CACHE_BACKEND_PATH, return_value=mock_cache),
            patch("yfinance_mcp.helpers.requests.get", return_value=resp),
        ):
            result = fetch_japan_etf_expense("282A.T")

        assert result == expected
        mock_cache.store_etf_expense.assert_called_once_with(
            "282A.T", expected, exchange="JPX", source="yahoo_japan"
        )

    def test_pattern_not_found_caches_none(self, mock_cache):
        """When expense ratio pattern not found, cache None."""
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body>No expense ratio here</body></html>"

        with (
            patch(CACHE_BACKEND_PATH, return_value=mock_cache),
            patch("yfinance_mcp.helpers.requests.get", return_value=resp),
        ):
            result = fetch_japan_etf_expense("282A.T")

        assert result is None
        mock_cache.store_etf_expense.assert_called_once_with(
            "282A.T", None, exchange="JPX", source="yahoo_japan"
        )


class TestRetry:
    """Tests for retry logic."""

    def test_retries_on_non_200_status(self, mock_cache):
        """Should retry with different UA on non-200 responses."""
        error_resp = MagicMock()
        error_resp.status_code = 503
        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.text = "<div>信託報酬</div><span>0.11</span><"

        with (
            patch(CACHE_BACKEND_PATH, return_value=mock_cache),
            patch(
                "yfinance_mcp.helpers.requests.get",
                side_effect=[error_resp, error_resp, success_resp],
            ),
        ):
            result = fetch_japan_etf_expense("282A.T")

        assert result == 0.11

    def test_retries_on_network_error(self, mock_cache):
        """Should retry on network exceptions."""
        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.text = "<div>信託報酬</div><span>0.11</span><"

        with (
            patch(CACHE_BACKEND_PATH, return_value=mock_cache),
            patch(
                "yfinance_mcp.helpers.requests.get",
                side_effect=[
                    requests.RequestException("Connection error"),
                    success_resp,
                ],
            ),
        ):
            result = fetch_japan_etf_expense("282A.T")

        assert result == 0.11

    def test_all_retries_fail_caches_none(self, mock_cache):
        """After exhausting retries, cache None and return None."""
        error_resp = MagicMock()
        error_resp.status_code = 503

        with (
            patch(CACHE_BACKEND_PATH, return_value=mock_cache),
            patch("yfinance_mcp.helpers.requests.get", return_value=error_resp),
        ):
            result = fetch_japan_etf_expense("282A.T")

        assert result is None
        mock_cache.store_etf_expense.assert_called_once_with(
            "282A.T", None, exchange="JPX", source="yahoo_japan"
        )


class TestSymbolNormalization:
    """Tests for symbol normalization."""

    @pytest.mark.parametrize(
        "input_symbol,expected_cache_key",
        [
            pytest.param("282A", "282A.T", id="without_suffix"),
            pytest.param("282A.T", "282A.T", id="with_suffix"),
            pytest.param("282a.t", "282A.T", id="lowercase"),
            pytest.param("1343", "1343.T", id="numeric_code"),
        ],
    )
    def test_normalizes_symbol(self, mock_cache, input_symbol, expected_cache_key):
        """Symbol should be normalized to uppercase with .T suffix."""
        mock_cache.get_etf_expense.return_value = (0.11, True)

        with patch(CACHE_BACKEND_PATH, return_value=mock_cache):
            fetch_japan_etf_expense(input_symbol)

        mock_cache.get_etf_expense.assert_called_with(expected_cache_key)


class TestResourceManagement:
    """Tests for resource management and error handling."""

    def test_cache_always_closed(self, mock_cache):
        """Cache should be closed even on exception."""
        mock_cache.get_etf_expense.side_effect = RuntimeError("DB error")

        with (
            patch(CACHE_BACKEND_PATH, return_value=mock_cache),
            pytest.raises(RuntimeError),
        ):
            fetch_japan_etf_expense("282A.T")

        mock_cache.close.assert_called_once()

    def test_logger_called_on_cache_hit(self, mock_cache):
        """Logger should be called on cache hit when provided."""
        mock_cache.get_etf_expense.return_value = (0.11, True)
        logger = MagicMock(spec=logging.Logger)

        with patch(CACHE_BACKEND_PATH, return_value=mock_cache):
            fetch_japan_etf_expense("282A.T", logger=logger)

        logger.debug.assert_called()

    def test_url_construction(self, mock_cache):
        """URL should be constructed correctly for Yahoo Finance Japan."""
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<div>信託報酬</div><span>0.11</span><"

        with (
            patch(CACHE_BACKEND_PATH, return_value=mock_cache),
            patch("yfinance_mcp.helpers.requests.get", return_value=resp) as mock_get,
        ):
            fetch_japan_etf_expense("1343.T")

        call_args = mock_get.call_args_list[0]
        assert call_args[0][0] == "https://finance.yahoo.co.jp/quote/1343.T"


class TestFmtToon:
    """Tests for fmt_toon function - row-oriented TOON encoding."""

    @pytest.fixture
    def decode(self):
        """Provide toon_decode for tests."""
        from toon_format import decode as toon_decode

        return toon_decode

    @pytest.fixture
    def fmt_toon(self):
        """Provide fmt_toon for tests."""
        from yfinance_mcp.helpers import fmt_toon

        return fmt_toon

    @pytest.mark.parametrize(
        "interval",
        [
            pytest.param("1m", id="1min"),
            pytest.param("5m", id="5min"),
            pytest.param("15m", id="15min"),
            pytest.param("30m", id="30min"),
            pytest.param("1h", id="1hour"),
            pytest.param("1d", id="daily"),
            pytest.param("1wk", id="weekly"),
        ],
    )
    def test_interval_included_when_provided(self, fmt_toon, decode, interval):
        """interval parameter should be included in output for all valid intervals."""
        import pandas as pd

        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        result = decode(fmt_toon(df, interval=interval))

        assert result["interval"] == interval

    def test_interval_not_included_when_none(self, fmt_toon, decode):
        """interval should be omitted when not provided."""
        import pandas as pd

        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        result = decode(fmt_toon(df))

        assert "interval" not in result

    def test_columns_include_ts_first(self, fmt_toon, decode):
        """columns should include 'ts' as first element."""
        import pandas as pd

        df = pd.DataFrame(
            {"open": [100], "close": [101]},
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )
        result = decode(fmt_toon(df))

        assert result["columns"][0] == "ts"
        assert result["columns"][1:] == ["open", "close"]

    def test_rows_contain_timestamp_and_values(self, fmt_toon, decode):
        """Each row should be [timestamp, val1, val2, ...]."""
        import pandas as pd

        df = pd.DataFrame(
            {"open": [100, 101], "close": [102, 103]},
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )
        result = decode(fmt_toon(df))

        assert len(result["rows"]) == 2
        assert result["rows"][0] == ["2024-01-01", 100, 102]
        assert result["rows"][1] == ["2024-01-02", 101, 103]

    def test_intraday_timestamp_format(self, fmt_toon, decode):
        """Intraday data should use YYYY-MM-DDTHH:MM format."""
        import pandas as pd

        df = pd.DataFrame(
            {"close": [100, 101]},
            index=pd.date_range("2024-01-15 09:30", periods=2, freq="30min"),
        )
        result = decode(fmt_toon(df))

        assert result["rows"][0][0] == "2024-01-15T09:30"
        assert result["rows"][1][0] == "2024-01-15T10:00"

    def test_daily_timestamp_format(self, fmt_toon, decode):
        """Daily data should use YYYY-MM-DD format."""
        import pandas as pd

        df = pd.DataFrame(
            {"close": [100, 101]},
            index=pd.date_range("2024-01-15", periods=2, freq="D"),
        )
        result = decode(fmt_toon(df))

        assert result["rows"][0][0] == "2024-01-15"
        assert result["rows"][1][0] == "2024-01-16"

    @pytest.mark.parametrize(
        "interval,expected_interval",
        [
            pytest.param("1h", "1h", id="with_interval"),
            pytest.param(None, None, id="no_interval"),
        ],
    )
    def test_empty_dataframe(self, fmt_toon, decode, interval, expected_interval):
        """Empty DataFrame should return empty rows."""
        import pandas as pd

        df = pd.DataFrame(columns=["close"])
        df.index = pd.DatetimeIndex([])
        result = decode(fmt_toon(df, interval=interval))

        if expected_interval:
            assert result["interval"] == expected_interval
        else:
            assert "interval" not in result
        assert result["rows"] == []
        assert result["columns"] == ["ts", "close"]

    def test_single_row(self, fmt_toon, decode):
        """Single row DataFrame should produce single row output."""
        import pandas as pd

        df = pd.DataFrame({"close": [100]}, index=pd.date_range("2024-01-15 09:30", periods=1))
        result = decode(fmt_toon(df, interval="5m"))

        assert len(result["rows"]) == 1
        assert result["rows"][0] == ["2024-01-15T09:30", 100]

    @pytest.mark.parametrize(
        "tz,should_include_tz",
        [
            pytest.param("America/New_York", True, id="new_york"),
            pytest.param("America/Los_Angeles", True, id="los_angeles"),
            pytest.param("Asia/Tokyo", True, id="tokyo"),
            pytest.param(None, False, id="no_timezone"),
            pytest.param("invalid", False, id="invalid_timezone"),
        ],
    )
    def test_timezone_handling(self, fmt_toon, decode, tz, should_include_tz):
        """Valid IANA timezone should be included as tz field."""
        import pandas as pd

        df = pd.DataFrame(
            {"close": [100, 101]},
            index=pd.date_range("2024-01-15 09:30", periods=2, freq="1h"),
        )
        result = decode(fmt_toon(df, interval="1h", tz=tz))

        if should_include_tz:
            assert result["tz"] == tz
        else:
            assert "tz" not in result

    def test_wrapper_key_wraps_data(self, fmt_toon, decode):
        """wrapper_key should wrap the TOON data in a nested structure."""
        import pandas as pd

        df = pd.DataFrame({"close": [100, 101]}, index=pd.date_range("2024-01-01", periods=2))
        result = decode(fmt_toon(df, wrapper_key="bars", interval="1d"))

        assert "bars" in result
        assert result["bars"]["interval"] == "1d"
        assert "rows" in result["bars"]
        assert result["bars"]["columns"] == ["ts", "close"]

    def test_issues_included_when_provided(self, fmt_toon, decode):
        """issues dict should be included as _issues in output."""
        import pandas as pd

        df = pd.DataFrame({"close": [100, 101]}, index=pd.date_range("2024-01-01", periods=2))
        issues = {"ac": "Adj Close unavailable"}
        result = decode(fmt_toon(df, issues=issues))

        assert "_issues" in result
        assert result["_issues"]["ac"] == "Adj Close unavailable"

    def test_summaries_included_at_top_level(self, fmt_toon, decode):
        """summaries dict should be merged at top level."""
        import pandas as pd

        df = pd.DataFrame({"close": [100, 101]}, index=pd.date_range("2024-01-01", periods=2))
        summaries = {"latest": {"close": 101}}
        result = decode(fmt_toon(df, summaries=summaries))

        assert "latest" in result
        assert result["latest"]["close"] == 101

    def test_non_datetime_index_raises(self, fmt_toon):
        """Non-DatetimeIndex should raise TypeError."""
        import pandas as pd

        df = pd.DataFrame({"close": [100, 101]}, index=[0, 1])

        with pytest.raises(TypeError, match="fmt_toon expects DatetimeIndex"):
            fmt_toon(df)
