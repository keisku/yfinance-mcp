"""Tests for MCP server - focuses on public interface (list_tools, call_tool)."""

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
        """Agent should see all 5 tools."""
        result = asyncio.run(list_tools())
        assert len(result) == 5
        names = {t.name for t in result}
        assert names == {
            "search_stock",
            "history",
            "technicals",
            "valuation",
            "financials",
        }

    def test_each_tool_has_required_schema(self) -> None:
        """Each tool should have name, description, and inputSchema."""
        for tool in TOOLS:
            assert tool.name
            assert tool.description
            assert "properties" in tool.inputSchema


class TestSearchStockTool:
    """Test search_stock tool - entry point for stock identification."""

    def _mock_ticker(self, **overrides) -> MagicMock:
        """Create mock with sensible defaults."""
        mock = MagicMock()
        mock.fast_info.last_price = overrides.get("price", 150.0)
        mock.fast_info.previous_close = 148.0
        mock.fast_info.market_cap = 3e12
        mock.fast_info.day_high = 152.0
        mock.fast_info.day_low = 147.0
        mock.fast_info.last_volume = 50000000
        mock.info = {
            "regularMarketPrice": overrides.get("price", 150.0),
            "shortName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchange": "NMS",
            "currency": "USD",
        }
        return mock

    def test_returns_identity_and_price(self, call) -> None:
        """search_stock should return identity + current price only."""
        mock = self._mock_ticker()

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("search_stock", {"symbol": "AAPL"})

        assert "symbol" in parsed
        assert "name" in parsed
        assert "sector" in parsed
        assert "industry" in parsed
        assert "exchange" in parsed
        assert "price" in parsed
        assert "change_pct" in parsed
        assert "market_cap" in parsed
        assert "volume" in parsed
        # Should NOT contain valuation or technical analysis
        assert "pe" not in parsed
        assert "peg" not in parsed
        assert "trend" not in parsed
        assert "quality_score" not in parsed

    def test_search_by_query(self, call) -> None:
        """search_stock should work with company name query."""
        mock = self._mock_ticker()

        with (
            patch("yfinance.Search") as mock_search,
            patch("yfinance_mcp.server._ticker", return_value=mock),
        ):
            mock_search.return_value.quotes = [{"symbol": "AAPL"}]
            parsed = call("search_stock", {"query": "Apple"})

        assert parsed["symbol"] == "AAPL"
        assert "price" in parsed

    def test_requires_symbol_or_query(self, call) -> None:
        """search_stock should error if neither symbol nor query provided."""
        parsed = call("search_stock", {})
        assert parsed["err"] == "VALIDATION_ERROR"

    def test_query_not_found(self, call) -> None:
        """search_stock should error if query returns no results."""
        with patch("yfinance.Search") as mock_search:
            mock_search.return_value.quotes = []
            parsed = call("search_stock", {"query": "NonexistentCompany12345"})

        assert parsed["err"] == "SYMBOL_NOT_FOUND"

    def test_smart_search_strips_suffix(self, call) -> None:
        """search_stock should strip common suffixes and retry (e.g., 'DBS Bank' -> 'DBS')."""
        mock = self._mock_ticker()

        def search_side_effect(query, **kwargs):
            result = MagicMock()
            # "DBS Bank" returns empty, but "DBS" (stripped) succeeds
            if query.lower() == "dbs":
                result.quotes = [{"symbol": "D05.SI"}]
            else:
                result.quotes = []
            return result

        with (
            patch("yfinance.Search", side_effect=search_side_effect),
            patch("yfinance_mcp.server._ticker", return_value=mock),
        ):
            parsed = call("search_stock", {"query": "DBS Bank"})

        assert parsed["symbol"] == "D05.SI"
        assert "price" in parsed


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

    def test_returns_bars(self, call) -> None:
        """History should return bars dict."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call("history", {"symbol": "AAPL"})

        assert "bars" in parsed
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

    @pytest.mark.parametrize(
        "indicator,expected_keys,signal_key,valid_signals",
        [
            ("rsi", ["rsi"], "rsi_signal", ["overbought", "oversold", "neutral"]),
            ("macd", ["macd"], "macd_trend", ["bullish", "bearish"]),
            ("sma_20", ["sma_20", "sma_20_pos"], None, None),
            ("wma_20", ["wma_20", "wma_20_pos"], None, None),
            ("ema_12", ["ema_12"], None, None),
            ("momentum", ["momentum"], "momentum_signal", ["bullish", "bearish"]),
            ("cci", ["cci"], "cci_signal", ["overbought", "oversold", "neutral"]),
            ("dmi", ["dmi_plus", "dmi_minus", "adx"], "dmi_signal", None),
            ("williams", ["williams_r"], "williams_signal", ["overbought", "oversold", "neutral"]),
            (
                "fast_stoch",
                ["fast_stoch_k", "fast_stoch_d"],
                "fast_stoch_signal",
                ["overbought", "oversold", "neutral"],
            ),
            (
                "stoch",
                ["stoch_k", "stoch_d"],
                "stoch_signal",
                ["overbought", "oversold", "neutral"],
            ),
            ("ichimoku", ["ichimoku_conversion", "ichimoku_base"], "ichimoku_signal", None),
            (
                "volume_profile",
                ["vp_poc", "vp_value_area_high"],
                "vp_signal",
                ["above_value_area", "below_value_area", "in_value_area"],
            ),
            (
                "price_change",
                ["price_change", "price_change_pct"],
                "price_change_signal",
                ["up", "down", "flat"],
            ),
            ("bb", ["bb_upper", "bb_lower"], "bb_signal", None),
            ("trend", ["sma50", "trend"], None, None),
            ("atr", ["atr"], None, None),
            ("obv", ["obv"], "obv_trend", ["bullish", "bearish"]),
        ],
    )
    def test_indicator_returns_expected_structure(
        self, call, indicator, expected_keys, signal_key, valid_signals
    ) -> None:
        """Each indicator should return expected keys and valid signal values."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": [indicator]})

        for key in expected_keys:
            assert key in parsed, f"Missing key {key} for indicator {indicator}"

        if signal_key:
            assert signal_key in parsed, f"Missing signal key {signal_key} for {indicator}"
            if valid_signals:
                assert parsed[signal_key] in valid_signals, (
                    f"Invalid signal {parsed[signal_key]} for {indicator}"
                )

    def test_trend_insufficient_data(self, call) -> None:
        """Trend with <50 bars should return error message."""
        mock = MagicMock()
        df = pd.DataFrame(
            {
                "Open": [100] * 30,
                "High": [101] * 30,
                "Low": [99] * 30,
                "Close": [100] * 30,
                "Volume": [1000000] * 30,
            },
            index=pd.date_range("2024-01-01", periods=30),
        )
        mock.history.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": ["trend"]})

        assert parsed["trend"] is None
        assert "_trend_error" in parsed

    def test_all_indicators(self, call) -> None:
        """All supported indicators should work."""
        indicators = [
            "trend",
            "rsi",
            "macd",
            "sma_20",
            "ema_12",
            "wma_10",
            "momentum",
            "cci",
            "dmi",
            "williams",
            "fast_stoch",
            "ichimoku",
            "volume_profile",
            "price_change",
            "bb",
            "stoch",
            "atr",
            "obv",
        ]
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": indicators})

        assert "rsi" in parsed
        assert "macd" in parsed
        assert "bb_upper" in parsed
        assert "trend" in parsed

    def test_empty_indicators_error(self, call) -> None:
        """Empty indicators should return validation error."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call("technicals", {"symbol": "AAPL", "indicators": []})

        assert parsed["err"] == "VALIDATION_ERROR"


class TestValuationTool:
    """Test valuation tool - valuation metrics and quality score."""

    def _mock_valuation(self) -> MagicMock:
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
            "returnOnAssets": 0.15,
            "operatingCashflow": 100e9,
            "netIncomeToCommon": 80e9,
            "currentRatio": 1.5,
            "debtToEquity": 50,
            "returnOnEquity": 0.2,
        }
        return mock

    def test_pe_eps_margins(self, call) -> None:
        """Basic metrics should be returned."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_valuation()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["pe", "eps", "margins"]})

        assert "pe" in parsed
        assert "eps" in parsed
        assert "margin_gross" in parsed

    def test_growth_metrics(self, call) -> None:
        """Growth option should return growth rates."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_valuation()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["growth"]})

        assert "growth_rev" in parsed
        assert "growth_earn" in parsed

    def test_ratios_metrics(self, call) -> None:
        """Ratios option should return valuation ratios."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_valuation()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["ratios"]})

        assert "pb" in parsed
        assert "ps" in parsed
        assert "ev_ebitda" in parsed

    def test_dividends_metrics(self, call) -> None:
        """Dividends option should return yield, rate, payout."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_valuation()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["dividends"]})

        assert "div_yield" in parsed
        assert "div_rate" in parsed
        assert "payout_ratio" in parsed

    def test_quality_metric(self, call) -> None:
        """Quality returns score, signal, and details."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_valuation()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["quality"]})

        # Structure assertions
        assert "quality_score" in parsed
        assert "quality_max" in parsed
        assert parsed["quality_max"] == 7
        assert "quality_signal" in parsed
        assert parsed["quality_signal"] in ["strong", "neutral", "weak"]
        assert "quality_details" in parsed
        # Value assertions (mock has strong fundamentals)
        assert parsed["quality_score"] >= 6
        assert parsed["quality_signal"] == "strong"

    def test_peg_metric(self, call) -> None:
        """PEG returns value, source, and signal."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_valuation()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["peg"]})

        assert "peg" in parsed
        assert "peg_source" in parsed
        assert parsed["peg_source"] in ["earnings", "revenue"]
        assert "peg_signal" in parsed
        assert parsed["peg_signal"] in ["undervalued", "fair", "overvalued"]

    def test_peg_undervalued_signal(self, call) -> None:
        """Low PE with high growth should signal undervalued."""
        mock = self._mock_valuation()
        mock.info["trailingPE"] = 10
        mock.info["earningsGrowth"] = 0.20  # PEG = 10/20 = 0.5

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["peg"]})

        assert parsed["peg"] == 0.5
        assert parsed["peg_signal"] == "undervalued"

    def test_empty_metrics_error(self, call) -> None:
        """Empty metrics should return validation error."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_valuation()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": []})

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
        """Income statement should return data."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_financials()):
            parsed = call("financials", {"symbol": "AAPL", "statement": "income"})

        assert len(parsed) >= 1

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


class TestErrorHandling:
    """Test error responses - how agents handle failures."""

    def test_invalid_symbol_error(self, call) -> None:
        """Invalid symbol should return SYMBOL_NOT_FOUND."""
        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.return_value.fast_info = None
            mock_yf.return_value.info = {"regularMarketPrice": None}
            parsed = call("search_stock", {"symbol": "INVALID123"})

        assert parsed["err"] == "SYMBOL_NOT_FOUND"

    def test_unknown_tool_error(self, call) -> None:
        """Unknown tool should return VALIDATION_ERROR."""
        parsed = call("nonexistent_tool", {})
        assert parsed["err"] == "VALIDATION_ERROR"

    def test_network_error_wrapped(self, call) -> None:
        """Network errors should be wrapped cleanly."""
        with patch("yfinance_mcp.server._ticker") as mock:
            mock.side_effect = ConnectionError("Network failed")
            parsed = call("search_stock", {"symbol": "AAPL"})

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
        parsed = call("search_stock", {"symbol": "AAPL"})
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
        mock.fast_info.day_high = 152.0
        mock.fast_info.day_low = 147.0
        mock.info = {"regularMarketPrice": 150.0, "shortName": "Apple Inc."}

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("search_stock", {"symbol": "AAPL"})

        assert "err" not in parsed
        assert "price" in parsed


class TestEdgeCases:
    """Test edge cases - robustness against malformed or unusual inputs."""

    def test_empty_symbol_returns_validation_error(self, call) -> None:
        """Empty symbol should return VALIDATION_ERROR."""
        parsed = call("search_stock", {"symbol": ""})
        assert parsed["err"] == "VALIDATION_ERROR"

    @pytest.mark.parametrize(
        "input_type,value,description",
        [
            ("symbol", "'; DROP TABLE stocks;--", "SQL injection"),
            ("query", "<script>alert('xss')</script>", "XSS attempt"),
            ("symbol", "AAPL\nMSFT", "Newline injection"),
            ("symbol", "../../../etc/passwd", "Path traversal"),
        ],
    )
    def test_malicious_input_is_safe(self, call, input_type, value, description) -> None:
        """Malicious input should return SYMBOL_NOT_FOUND, not execute."""
        if input_type == "symbol":
            with patch("yfinance.Ticker") as mock_yf:
                mock_yf.return_value.fast_info = None
                parsed = call("search_stock", {"symbol": value})
        else:
            with patch("yfinance.Search") as mock_search:
                mock_search.return_value.quotes = []
                parsed = call("search_stock", {"query": value})

        assert parsed["err"] == "SYMBOL_NOT_FOUND", f"{description} should be rejected"

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


class TestDataEdgeCases:
    """Property-based fuzzing with hypothesis to discover data edge cases.

    Fuzzes:
    - Prices from 1e-5 to 1e6 (ultra-penny to high-value stocks)
    - Info values as None, float, Series, or empty Series
    """

    price_strategy = st.floats(min_value=1e-5, max_value=1e6, allow_nan=False, allow_infinity=False)

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
        mock.fast_info.day_high = price * 1.01
        mock.fast_info.day_low = price * 0.99
        mock.fast_info.last_volume = 1000000
        info = {"regularMarketPrice": price, "shortName": "Test Stock"}
        if info_overrides:
            info.update(info_overrides)
        mock.info = info
        return mock

    @given(price=price_strategy)
    @settings(max_examples=50)
    def test_search_stock_handles_any_price(self, price: float) -> None:
        """Any valid price should not crash and should preserve non-zero."""
        mock = self._mock_ticker(price)

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            result = self._call("search_stock", {"symbol": "TEST"})

        assert "price" in result
        assert result["price"] > 0, f"Price {price} rounded to zero or negative"

    @given(
        roa=info_value_strategy,
        ocf=info_value_strategy,
        pe=info_value_strategy,
        growth=info_value_strategy,
    )
    @settings(max_examples=50)
    def test_valuation_quality_handles_any_info_types(
        self,
        roa: float | pd.Series | None,
        ocf: float | pd.Series | None,
        pe: float | pd.Series | None,
        growth: float | pd.Series | None,
    ) -> None:
        """Valuation quality should handle any combination of info value types."""
        mock = self._mock_ticker(
            100.0,
            {
                "returnOnAssets": roa,
                "operatingCashflow": ocf,
                "trailingPE": pe,
                "earningsGrowth": growth,
            },
        )

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            result = self._call("valuation", {"symbol": "TEST", "metrics": ["quality"]})

        assert "quality_score" in result
        assert "quality_signal" in result

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
