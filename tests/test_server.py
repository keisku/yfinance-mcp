"""Tests for MCP server - focuses on public interface (list_tools, call_tool).

Tests are structured from an MCP client's perspective:
1. Tool Discovery - list_tools() returns expected tools
2. Tool Execution - call_tool() returns expected responses
3. Error Handling - call_tool() returns structured errors
4. Resilience - circuit breaker protects against cascading failures
5. Edge Cases - robustness against malformed or unusual inputs
6. Data Edge Cases - property-based fuzzing with hypothesis to discover:
   - Ultra-penny stocks: prices < $0.01 need dynamic decimal precision
   - Series values: yfinance sometimes returns Series instead of scalars
   - Missing data: sparse fundamentals should not crash
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from yfinance_mcp.server import (
    TOOLS,
    call_tool,
    list_tools,
    open_circuit_breaker_for_testing,
)


@pytest.fixture
def call():
    """Fixture to call tool and parse JSON response."""

    def _call(name: str, args: dict) -> dict:
        result = asyncio.run(call_tool(name, args))
        return json.loads(result[0].text)

    return _call


class TestToolDiscovery:
    """Test list_tools() - how agents discover available tools."""

    def test_list_tools_returns_all_tools(self) -> None:
        """Agent should see all 7 tools."""
        result = asyncio.run(list_tools())
        assert len(result) == 6
        names = {t.name for t in result}
        assert names == {
            "summary",
            "history",
            "technicals",
            "fundamentals",
            "financials",
            "search",
        }

    def test_each_tool_has_required_schema(self) -> None:
        """Each tool should have name, description, and inputSchema."""
        for tool in TOOLS:
            assert tool.name
            assert tool.description
            assert "properties" in tool.inputSchema


class TestSummaryTool:
    """Test summary tool - the recommended starting point for stock analysis."""

    def _mock_ticker(self, **overrides) -> MagicMock:
        """Create mock with sensible defaults."""
        mock = MagicMock()
        mock.fast_info.last_price = overrides.get("price", 150.0)
        mock.fast_info.previous_close = 148.0
        mock.fast_info.market_cap = 3e12
        mock.info = {
            "regularMarketPrice": overrides.get("price", 150.0),  # Required for validation
            "returnOnAssets": 0.15,
            "operatingCashflow": 100e9,
            "netIncomeToCommon": 80e9,
            "currentRatio": 1.5,
            "debtToEquity": 50,
            "grossMargins": 0.4,
            "returnOnEquity": 0.2,
            "trailingPE": overrides.get("pe", 25),
            "earningsGrowth": overrides.get("growth", 0.15),
        }
        return mock

    def test_returns_key_metrics(self, call) -> None:
        """Summary should return price, quality score, PEG, trend."""
        mock = self._mock_ticker()
        df = pd.DataFrame({"Close": [150] * 60}, index=pd.date_range("2024-01-01", periods=60))

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=df),
        ):
            parsed = call("summary", {"symbol": "AAPL"})

        assert "price" in parsed
        assert "quality_score" in parsed
        assert "quality_max" in parsed  # Shows scale (7, static thresholds)
        assert "quality_signal" in parsed
        assert "trend" in parsed
        assert "_hint" in parsed  # Guides next action

    def test_quality_strong_signal(self, call) -> None:
        """Company with strong fundamentals should get quality_signal=strong."""
        mock = self._mock_ticker()
        df = pd.DataFrame({"Close": [150] * 60}, index=pd.date_range("2024-01-01", periods=60))

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=df),
        ):
            parsed = call("summary", {"symbol": "AAPL"})

        assert parsed["quality_score"] >= 6  # 6-7 is strong on our 0-7 scale
        assert parsed["quality_max"] == 7
        assert parsed["quality_signal"] == "strong"

    def test_peg_undervalued_signal(self, call) -> None:
        """Low PE with high growth should signal undervalued."""
        mock = self._mock_ticker(pe=10, growth=0.20)  # PEG = 10/20 = 0.5
        df = pd.DataFrame({"Close": [150] * 60}, index=pd.date_range("2024-01-01", periods=60))

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=df),
        ):
            parsed = call("summary", {"symbol": "AAPL"})

        assert parsed["peg"] == 0.5
        assert parsed["peg_signal"] == "undervalued"

    def test_trend_based_on_sma50(self, call) -> None:
        """Trend should reflect price vs 50-day SMA."""
        mock = self._mock_ticker(price=160.0)  # Above SMA50 of 150
        df = pd.DataFrame({"Close": [150] * 60}, index=pd.date_range("2024-01-01", periods=60))

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=df),
        ):
            parsed = call("summary", {"symbol": "AAPL"})

        assert parsed["trend"] == "uptrend"


class TestHistoryTool:
    """Test price tool - historical OHLCV data."""

    def _mock_history(self, n: int = 50) -> MagicMock:
        mock = MagicMock()
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": [1000000] * n,
            },
            index=pd.date_range("2024-01-01", periods=n),
        )
        mock.history.return_value = df
        return mock

    def test_returns_bars_with_hint(self, call) -> None:
        """History should return bars dict with navigation hint."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call("history", {"symbol": "AAPL"})

        assert "bars" in parsed
        assert "_hint" in parsed
        first_bar = list(parsed["bars"].values())[0]
        assert set(first_bar.keys()) == {"o", "h", "l", "c", "v"}

    def test_limit_restricts_bars(self, call) -> None:
        """Limit parameter should control number of bars."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call("history", {"symbol": "AAPL", "limit": 5})

        assert len(parsed["bars"]) == 5

    def test_detailed_format(self, call) -> None:
        """format=detailed should use full column names."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call("history", {"symbol": "AAPL", "format": "detailed"})

        first_bar = list(parsed["bars"].values())[0]
        assert "Open" in first_bar and "Close" in first_bar

    def test_start_and_end_date_range(self, call) -> None:
        """start/end should fetch specific date range."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call(
                "history",
                {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-01-31"},
            )

        assert "bars" in parsed
        assert len(parsed["bars"]) > 0

    def test_start_only_defaults_end_to_today(self, call) -> None:
        """start without end should fetch from start to today."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call("history", {"symbol": "AAPL", "start": "2024-01-01"})

        assert "bars" in parsed
        assert len(parsed["bars"]) > 0

    def test_end_only_computes_start_from_period(self, call) -> None:
        """end without start should compute start = end - period."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call(
                "history",
                {"symbol": "AAPL", "end": "2024-06-30", "period": "3mo"},
            )

        assert "bars" in parsed
        assert len(parsed["bars"]) > 0

    def test_intraday_datetime_strings(self, call) -> None:
        """Intraday intervals should accept datetime strings."""
        mock = self._mock_history()
        # Create intraday-style index
        df = mock.history.return_value.copy()
        df.index = pd.date_range("2024-01-15 09:30", periods=len(df), freq="5min")
        mock.history.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call(
                "history",
                {
                    "symbol": "AAPL",
                    "start": "2024-01-15 09:30",
                    "end": "2024-01-15 16:00",
                    "interval": "5m",
                },
            )

        assert "bars" in parsed
        # Intraday format includes time
        first_date = list(parsed["bars"].keys())[0]
        assert " " in first_date  # Contains time component


class TestTechnicalsTool:
    """Test technicals tool - trading signals."""

    def _mock_prices(self) -> MagicMock:
        mock = MagicMock()
        np.random.seed(42)
        n = 100  # Need 78+ for Ichimoku (52 + 26)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": [1000000] * n,
            },
            index=pd.date_range("2024-01-01", periods=n),
        )
        mock.history.return_value = df
        return mock

    def test_rsi_with_signal(self, call) -> None:
        """RSI should include overbought/oversold signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["rsi"]})

        assert "rsi" in parsed
        assert "rsi_signal" in parsed
        assert parsed["rsi_signal"] in ["overbought", "oversold", "neutral"]

    def test_macd_with_trend(self, call) -> None:
        """MACD should include bullish/bearish trend."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["macd"]})

        assert "macd" in parsed
        assert "macd_trend" in parsed
        assert parsed["macd_trend"] in ["bullish", "bearish"]

    def test_sma_with_position(self, call) -> None:
        """SMA should show price position (above/below)."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["sma_20"]})

        assert "sma_20" in parsed
        assert "sma_20_pos" in parsed

    def test_wma_with_position(self, call) -> None:
        """WMA should show price position (above/below)."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["wma_20"]})

        assert "wma_20" in parsed
        assert "wma_20_pos" in parsed

    def test_momentum_with_signal(self, call) -> None:
        """Momentum should return value and bullish/bearish signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["momentum"]})

        assert "momentum" in parsed
        assert "momentum_signal" in parsed
        assert parsed["momentum_signal"] in ["bullish", "bearish"]

    def test_cci_with_signal(self, call) -> None:
        """CCI should return value and overbought/oversold/neutral signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["cci"]})

        assert "cci" in parsed
        assert "cci_signal" in parsed
        assert parsed["cci_signal"] in ["overbought", "oversold", "neutral"]

    def test_dmi_with_signal(self, call) -> None:
        """DMI should return +DI, -DI, ADX and trend signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["dmi"]})

        assert "dmi_plus" in parsed
        assert "dmi_minus" in parsed
        assert "adx" in parsed
        assert "dmi_signal" in parsed

    def test_williams_r_with_signal(self, call) -> None:
        """Williams %R should return value and overbought/oversold signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["williams"]})

        assert "williams_r" in parsed
        assert "williams_signal" in parsed
        assert parsed["williams_signal"] in ["overbought", "oversold", "neutral"]

    def test_fast_stochastic(self, call) -> None:
        """Fast Stochastic should return %K, %D and signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["fast_stoch"]})

        assert "fast_stoch_k" in parsed
        assert "fast_stoch_d" in parsed
        assert "fast_stoch_signal" in parsed
        assert parsed["fast_stoch_signal"] in ["overbought", "oversold", "neutral"]

    def test_ichimoku(self, call) -> None:
        """Ichimoku should return components and cloud signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["ichimoku"]})

        assert "ichimoku_conversion" in parsed
        assert "ichimoku_base" in parsed
        assert "ichimoku_leading_a" in parsed
        assert "ichimoku_leading_b" in parsed
        assert "ichimoku_signal" in parsed

    def test_volume_profile(self, call) -> None:
        """Volume Profile should return POC, value area, and signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["volume_profile"]})

        assert "vp_poc" in parsed
        assert "vp_value_area_high" in parsed
        assert "vp_value_area_low" in parsed
        assert "vp_signal" in parsed
        assert parsed["vp_signal"] in ["above_value_area", "below_value_area", "in_value_area"]

    def test_price_change(self, call) -> None:
        """Price Change should return change, percentage, and signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["price_change"]})

        assert "price_change" in parsed
        assert "price_change_pct" in parsed
        assert "price_change_signal" in parsed
        assert parsed["price_change_signal"] in ["up", "down", "flat"]

    def test_bollinger_bands(self, call) -> None:
        """BB should return bands and %B signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["bb"]})

        assert "bb_upper" in parsed
        assert "bb_lower" in parsed
        assert "bb_signal" in parsed

    def test_all_indicators(self, call) -> None:
        """All supported indicators should work."""
        indicators = ["rsi", "macd", "sma_20", "ema_12", "wma_10", "momentum", "cci", "dmi", "williams", "fast_stoch", "ichimoku", "volume_profile", "price_change", "bb", "stoch", "atr", "obv"]
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": indicators})

        assert "_hint" in parsed
        assert "rsi" in parsed
        assert "macd" in parsed
        assert "bb_upper" in parsed

    def test_empty_indicators_error(self, call) -> None:
        """Empty indicators should return validation error."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": []})

        assert parsed["err"] == "VALIDATION_ERROR"


class TestFundamentalsTool:
    """Test fundamentals tool - valuation metrics."""

    def _mock_fundamentals(self) -> MagicMock:
        mock = MagicMock()
        mock.info = {
            "trailingPE": 25.5,
            "forwardPE": 22.0,
            "pegRatio": 1.5,
            "trailingEps": 6.0,
            "forwardEps": 7.0,
            "grossMargins": 0.45,
            "operatingMargins": 0.30,
            "profitMargins": 0.25,
            "revenueGrowth": 0.15,
            "earningsGrowth": 0.20,
            "priceToBook": 35.0,
            "priceToSalesTrailing12Months": 7.5,
            "enterpriseToEbitda": 20.0,
            "dividendYield": 0.005,
            "dividendRate": 0.96,
            "payoutRatio": 0.15,
        }
        return mock

    def test_pe_eps_margins(self, call) -> None:
        """Basic metrics should be returned."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_fundamentals()):
            parsed = call("fundamentals", {"symbol": "AAPL", "metrics": ["pe", "eps", "margins"]})

        assert "pe" in parsed
        assert "eps" in parsed
        assert "margin_gross" in parsed
        assert "_hint" in parsed

    def test_growth_metrics(self, call) -> None:
        """Growth option should return growth rates."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_fundamentals()):
            parsed = call("fundamentals", {"symbol": "AAPL", "metrics": ["growth"]})

        assert "growth_rev" in parsed
        assert "growth_earn" in parsed

    def test_valuation_metrics(self, call) -> None:
        """Valuation option should return ratios."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_fundamentals()):
            parsed = call("fundamentals", {"symbol": "AAPL", "metrics": ["valuation"]})

        assert "pb" in parsed
        assert "ps" in parsed
        assert "ev_ebitda" in parsed

    def test_dividends_metrics(self, call) -> None:
        """Dividends option should return yield, rate, payout."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_fundamentals()):
            parsed = call("fundamentals", {"symbol": "AAPL", "metrics": ["dividends"]})

        assert "div_yield" in parsed
        assert "div_rate" in parsed
        assert "payout_ratio" in parsed

    def test_empty_metrics_error(self, call) -> None:
        """Empty metrics should return validation error."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_fundamentals()):
            parsed = call("fundamentals", {"symbol": "AAPL", "metrics": []})

        assert parsed["err"] == "VALIDATION_ERROR"


class TestFinancialsTool:
    """Test financials tool - financial statements."""

    def _mock_financials(self) -> MagicMock:
        mock = MagicMock()
        for stmt, data in [
            ("get_income_stmt", {"TotalRevenue": [100000, 90000]}),
            ("get_balance_sheet", {"TotalAssets": [500000, 450000]}),
            ("get_cashflow", {"OperatingCashFlow": [25000, 22000]}),
        ]:
            df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=2, freq="YE")).T
            getattr(mock, stmt).return_value = df
        return mock

    def test_income_statement(self, call) -> None:
        """Income statement should return data with hint."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_financials()):
            parsed = call("financials", {"symbol": "AAPL", "statement": "income"})

        assert "_hint" in parsed
        assert len(parsed) > 1

    def test_balance_sheet(self, call) -> None:
        """Balance sheet should work."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_financials()):
            parsed = call("financials", {"symbol": "AAPL", "statement": "balance"})

        assert len(parsed) > 1

    def test_cashflow(self, call) -> None:
        """Cash flow should work."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_financials()):
            parsed = call("financials", {"symbol": "AAPL", "statement": "cashflow"})

        assert len(parsed) > 1


class TestSearchTool:
    """Test search tool - symbol discovery."""

    def test_returns_matches_with_hint(self, call) -> None:
        """Search should return matches and guide to next action."""
        with patch("yfinance.Search") as mock_search:
            mock_search.return_value.quotes = [
                {"symbol": "AAPL", "shortname": "Apple Inc.", "quoteType": "EQUITY"},
            ]
            parsed = call("search", {"query": "Apple"})

        assert "matches" in parsed
        assert "count" in parsed
        assert "_hint" in parsed
        assert parsed["matches"][0]["symbol"] == "AAPL"

    def test_limit_restricts_results(self, call) -> None:
        """Limit should control number of matches."""
        with patch("yfinance.Search") as mock_search:
            mock_search.return_value.quotes = [
                {"symbol": "AAPL", "shortname": "Apple"},
                {"symbol": "AAPL.BA", "shortname": "Apple"},
                {"symbol": "AAPL.MX", "shortname": "Apple"},
            ]
            parsed = call("search", {"query": "Apple", "limit": 2})

        assert parsed["count"] == 2

    def test_empty_query_error(self, call) -> None:
        """Empty query should error."""
        parsed = call("search", {"query": ""})
        assert parsed["err"] == "VALIDATION_ERROR"


class TestErrorHandling:
    """Test error responses - how agents handle failures."""

    def test_invalid_symbol_error(self, call) -> None:
        """Invalid symbol should return SYMBOL_NOT_FOUND."""
        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.return_value.fast_info = None
            mock_yf.return_value.info = {"regularMarketPrice": None}
            parsed = call("summary", {"symbol": "INVALID123"})

        assert parsed["err"] == "SYMBOL_NOT_FOUND"

    def test_unknown_tool_error(self, call) -> None:
        """Unknown tool should return VALIDATION_ERROR."""
        parsed = call("nonexistent_tool", {})
        assert parsed["err"] == "VALIDATION_ERROR"

    def test_network_error_wrapped(self, call) -> None:
        """Network errors should be wrapped cleanly."""
        with patch("yfinance_mcp.server._ticker") as mock:
            mock.side_effect = ConnectionError("Network failed")
            parsed = call("summary", {"symbol": "AAPL"})

        assert parsed["err"] == "ERROR"
        assert "msg" in parsed


class TestCircuitBreaker:
    """Test circuit breaker - protects against cascading failures.

    Uses public test hooks to verify observable behavior without
    exposing internal state.
    """

    def test_rejects_requests_when_open(self, call) -> None:
        """When circuit is open, requests should be rejected immediately."""
        # Open the circuit using public test hook
        open_circuit_breaker_for_testing()

        # Request should be rejected without hitting yfinance
        parsed = call("summary", {"symbol": "AAPL"})
        assert parsed["err"] == "DATA_UNAVAILABLE"
        assert "retry later" in parsed["msg"].lower()

    def test_allows_requests_when_closed(self, call) -> None:
        """When circuit is closed, requests should proceed normally."""
        # Circuit starts closed (conftest resets it via public hook)
        mock = MagicMock()
        mock.fast_info.last_price = 150.0
        mock.fast_info.previous_close = 148.0
        mock.fast_info.last_volume = 1000000
        mock.fast_info.market_cap = 3e12
        mock.fast_info.year_high = 200.0
        mock.fast_info.year_low = 100.0
        mock.info = {"regularMarketPrice": 150.0}
        mock.history.return_value = pd.DataFrame()

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("summary", {"symbol": "AAPL"})

        assert "err" not in parsed
        assert "price" in parsed


class TestEdgeCases:
    """Test edge cases - robustness against malformed or unusual inputs."""

    def test_empty_symbol_returns_validation_error(self, call) -> None:
        """Empty symbol should return VALIDATION_ERROR."""
        parsed = call("summary", {"symbol": ""})
        assert parsed["err"] == "VALIDATION_ERROR"
        assert "symbol" in parsed["msg"].lower()

    def test_sql_injection_in_symbol_is_safe(self, call) -> None:
        """SQL injection attempt should be treated as invalid symbol."""
        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.return_value.fast_info = None
            parsed = call("summary", {"symbol": "'; DROP TABLE stocks;--"})

        assert parsed["err"] == "SYMBOL_NOT_FOUND"

    def test_xss_in_search_is_safe(self, call) -> None:
        """XSS attempt in search should return empty results, not execute."""
        with patch("yfinance.Search") as mock_search:
            mock_search.return_value.quotes = []
            parsed = call("search", {"query": "<script>alert('xss')</script>"})

        assert "matches" in parsed
        assert parsed["count"] == 0

    def test_newline_in_symbol_is_safe(self, call) -> None:
        """Newline in symbol should be treated as invalid."""
        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.return_value.fast_info = None
            parsed = call("summary", {"symbol": "AAPL\nMSFT"})

        assert parsed["err"] == "SYMBOL_NOT_FOUND"

    def test_negative_limit_clamped_to_one(self, call) -> None:
        """Negative limit should be clamped to 1, not cause errors."""
        mock = MagicMock()
        df = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        mock.history.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("history", {"symbol": "AAPL", "limit": -10})

        assert "bars" in parsed
        assert len(parsed["bars"]) >= 1  # At least 1 bar returned

    @pytest.mark.parametrize(
        "indicators,expected_unknown,should_have_valid",
        [
            (["sma_999"], None, False),  # Large period returns null
            (["sma_abc", "rsi"], "sma_abc", True),  # Invalid format
            (["sma_-5"], "sma_-5", False),  # Negative period
            (["nonexistent", "rsi"], "nonexistent", True),  # Unknown indicator
        ],
    )
    def test_invalid_indicators_handled_gracefully(
        self, call, indicators, expected_unknown, should_have_valid
    ) -> None:
        """Invalid indicators should be added to _unknown list or return null."""
        mock = MagicMock()
        df = pd.DataFrame(
            {
                "Open": [100] * 50,
                "High": [101] * 50,
                "Low": [99] * 50,
                "Close": [100] * 50,
                "Volume": [1000] * 50,
            },
            index=pd.date_range("2024-01-01", periods=50),
        )
        mock.history.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": indicators})

        if expected_unknown:
            assert "_unknown" in parsed
            assert expected_unknown in parsed["_unknown"]
        if should_have_valid:
            assert "rsi" in parsed

    def test_invalid_statement_defaults_to_cashflow(self, call) -> None:
        """Invalid statement type defaults to cashflow (graceful fallback)."""
        mock = MagicMock()
        df = pd.DataFrame(
            {"OperatingCashFlow": [100000]}, index=pd.date_range("2024-01-01", periods=1, freq="YE")
        ).T
        mock.get_cashflow.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("financials", {"symbol": "AAPL", "statement": "invalid"})

        # Should not error, falls back to cashflow
        assert "err" not in parsed

    def test_search_limit_clamped(self, call) -> None:
        """Search limit should be clamped to 1-20."""
        with patch("yfinance.Search") as mock_search:
            mock_search.return_value.quotes = [{"symbol": f"S{i}"} for i in range(30)]
            parsed = call("search", {"query": "test", "limit": 100})

        # Should not exceed 20
        assert parsed["count"] <= 20


class TestDataEdgeCases:
    """Property-based fuzzing with hypothesis to discover data edge cases.

    Fuzzes:
    - Prices from 1e-5 to 1e6 (ultra-penny to high-value stocks)
    - Info values as None, float, Series, or empty Series
    - Missing PE, growth, and net income combinations
    """

    # Prices from $0.00001 to $1,000,000
    price_strategy = st.floats(min_value=1e-5, max_value=1e6, allow_nan=False, allow_infinity=False)

    # yfinance info values can be None, scalar, Series, or empty Series
    info_value_strategy = st.one_of(
        st.none(),
        st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False).map(
            lambda x: pd.Series([x])
        ),
        st.just(pd.Series([], dtype=float)),
    )

    def _call(self, name: str, args: dict) -> dict:
        """Call tool and parse JSON response."""
        result = asyncio.run(call_tool(name, args))
        return json.loads(result[0].text)

    def _mock_ticker(self, price: float, info_overrides: dict | None = None) -> MagicMock:
        """Create mock ticker with given price and optional info overrides."""
        mock = MagicMock()
        mock.fast_info.last_price = price
        mock.fast_info.previous_close = price * 0.99
        mock.fast_info.market_cap = price * 1e6
        info = {"regularMarketPrice": price}
        if info_overrides:
            info.update(info_overrides)
        mock.info = info
        return mock

    @given(price=price_strategy)
    @settings(max_examples=50)
    def test_summary_handles_any_price(self, price: float) -> None:
        """Any valid price should not crash and should preserve non-zero."""
        mock = self._mock_ticker(price)
        df = pd.DataFrame({"Close": [price] * 60}, index=pd.date_range("2024-01-01", periods=60))

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=df),
        ):
            result = self._call("summary", {"symbol": "TEST"})

        assert "price" in result
        assert result["price"] > 0, f"Price {price} rounded to zero or negative"

    @given(
        roa=info_value_strategy,
        ocf=info_value_strategy,
        pe=info_value_strategy,
        growth=info_value_strategy,
    )
    @settings(max_examples=50)
    def test_summary_handles_any_info_types(
        self,
        roa: float | pd.Series | None,
        ocf: float | pd.Series | None,
        pe: float | pd.Series | None,
        growth: float | pd.Series | None,
    ) -> None:
        """Summary should handle any combination of info value types."""
        mock = self._mock_ticker(
            100.0,
            {
                "returnOnAssets": roa,
                "operatingCashflow": ocf,
                "trailingPE": pe,
                "earningsGrowth": growth,
            },
        )
        df = pd.DataFrame({"Close": [100.0] * 60}, index=pd.date_range("2024-01-01", periods=60))

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=df),
        ):
            result = self._call("summary", {"symbol": "TEST"})

        # Should not crash, should return valid response
        assert "price" in result
        assert "quality_score" in result

    @given(
        has_pe=st.booleans(),
        has_growth=st.booleans(),
        net_income=st.one_of(st.none(), st.floats(min_value=-1e9, max_value=1e9)),
    )
    @settings(max_examples=30)
    def test_summary_explains_missing_metrics(
        self, has_pe: bool, has_growth: bool, net_income: float | None
    ) -> None:
        """Summary should gracefully handle and explain missing metrics."""
        info: dict = {"regularMarketPrice": 50.0}
        if has_pe:
            info["trailingPE"] = 25.0
        if has_growth:
            info["earningsGrowth"] = 0.15
        if net_income is not None:
            info["netIncomeToCommon"] = net_income

        mock = self._mock_ticker(50.0, info)
        df = pd.DataFrame({"Close": [50.0] * 60}, index=pd.date_range("2024-01-01", periods=60))

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=df),
        ):
            result = self._call("summary", {"symbol": "TEST"})

        # Should not crash
        assert "price" in result

        # Should explain why PEG is missing if applicable
        if not has_pe and not has_growth:
            assert "peg_note" in result

    @given(price=st.floats(min_value=1e-5, max_value=0.009, allow_nan=False))
    @settings(max_examples=30)
    def test_price_tool_preserves_sub_cent(self, price: float) -> None:
        """Price tool should preserve sub-cent values in bars."""
        mock = MagicMock()
        df = pd.DataFrame(
            {
                "Open": [price * 0.9, price],
                "High": [price * 1.1, price * 1.2],
                "Low": [price * 0.8, price * 0.9],
                "Close": [price, price * 1.1],
                "Volume": [1000000, 1200000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        mock.history.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            result = self._call("history", {"symbol": "TEST", "limit": 5})

        bars = result["bars"]
        first_bar = list(bars.values())[0]
        assert first_bar["c"] > 0, f"Close price {price} rounded to zero"
