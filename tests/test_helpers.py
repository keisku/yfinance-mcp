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
