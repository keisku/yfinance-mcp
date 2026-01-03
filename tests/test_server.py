"""Tests for MCP server - focuses on public interface (list_tools, call_tool)."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from toon_format import decode as toon_decode

from yfinance_mcp.server import (
    ALL_INDICATORS,
    TOOLS,
    call_tool,
    list_tools,
    open_circuit_breaker_for_testing,
)


def _parse_response(text: str) -> dict:
    """Parse response, auto-detecting TOON or JSON format."""
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)

    if "\nmeta: " in text:
        toon_part, meta_part = text.split("\nmeta: ", 1)
        result = toon_decode(toon_part)
        result["meta"] = json.loads(meta_part)
        return result

    return toon_decode(text)


@pytest.fixture
def call():
    """Fixture to call tool and parse JSON response."""

    def _call(name: str, args: dict) -> dict:
        result = asyncio.run(call_tool(name, args))
        return json.loads(result[0].text)

    return _call


@pytest.fixture
def call_toon():
    """Fixture to call tool and parse TOON or JSON response."""

    def _call(name: str, args: dict) -> dict:
        result = asyncio.run(call_tool(name, args))
        return _parse_response(result[0].text)

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

    def test_returns_bars(self, call_toon) -> None:
        """History should return bars list in TOON format."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call_toon("history", {"symbol": "AAPL"})

        assert "bars" in parsed
        first_bar = parsed["bars"][0]
        assert set(first_bar.keys()) == {"d", "o", "h", "l", "c", "v"}

    def test_start_and_end_date_range(self, call_toon) -> None:
        """start/end should fetch specific date range."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call_toon(
                "history",
                {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-01-31"},
            )

        assert "bars" in parsed
        assert len(parsed["bars"]) > 0

    def test_start_only_defaults_end_to_today(self, call_toon) -> None:
        """start without end should fetch from start to today."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call_toon("history", {"symbol": "AAPL", "start": "2024-01-01"})

        assert "bars" in parsed
        assert len(parsed["bars"]) > 0

    def test_end_only_computes_start_from_period(self, call_toon) -> None:
        """end without start should compute start = end - period."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_history()):
            parsed = call_toon(
                "history",
                {"symbol": "AAPL", "end": "2024-06-30", "period": "3mo"},
            )

        assert "bars" in parsed
        assert len(parsed["bars"]) > 0

    def test_intraday_datetime_strings(self, call_toon) -> None:
        """Short time spans should auto-select intraday interval."""
        mock = self._mock_history()
        df = mock.history.return_value.copy()
        df.index = pd.date_range("2024-01-15 09:30", periods=len(df), freq="5min")
        mock.history.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call_toon(
                "history",
                {
                    "symbol": "AAPL",
                    "start": "2024-01-15 09:30",
                    "end": "2024-01-15 16:00",
                },
            )

        assert "bars" in parsed
        first_bar = parsed["bars"][0]
        assert " " in first_bar["d"]  # Contains time component


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
        "indicator,expected_data_keys,expected_meta_keys",
        [
            ("rsi", ["rsi"], None),
            ("macd", ["macd", "macd_signal", "macd_hist"], None),
            ("sma_20", ["sma_20"], None),
            ("wma_20", ["wma_20"], None),
            ("ema_12", ["ema_12"], None),
            ("momentum", ["momentum"], None),
            ("cci", ["cci"], None),
            ("dmi", ["dmi_plus", "dmi_minus", "adx"], None),
            ("williams", ["williams_r"], None),
            ("fast_stoch", ["fast_stoch_k", "fast_stoch_d"], None),
            ("stoch", ["stoch_k", "stoch_d"], None),
            ("ichimoku", ["ichimoku_conversion", "ichimoku_base"], None),
            ("volume_profile", None, ["volume_profile"]),
            ("price_change", None, ["price_change"]),
            ("bb", ["bb_upper", "bb_lower", "bb_middle", "bb_pctb"], None),
            ("trend", ["sma50"], None),
            ("atr", ["atr", "atr_pct"], None),
            ("obv", ["obv"], None),
        ],
    )
    def test_indicator_returns_expected_structure(
        self, call_toon, indicator, expected_data_keys, expected_meta_keys
    ) -> None:
        """Each indicator should return time series data or meta info."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": [indicator]})

        assert "data" in parsed, "Response should contain 'data' key"

        if expected_data_keys:
            first_row = parsed["data"][0]
            for key in expected_data_keys:
                assert key in first_row, f"Missing key {key} in data for indicator {indicator}"

        if expected_meta_keys:
            assert "meta" in parsed, f"Missing 'meta' for indicator {indicator}"
            for key in expected_meta_keys:
                assert key in parsed["meta"], f"Missing key {key} in meta for indicator {indicator}"

    def test_trend_insufficient_data(self, call_toon) -> None:
        """Trend with <50 bars should return error message in meta."""
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
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": ["trend"]})

        assert "meta" in parsed
        assert "_trend_error" in parsed["meta"]

    def test_all_indicators(self, call_toon) -> None:
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
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": indicators})

        assert "data" in parsed
        first_row = parsed["data"][0]
        assert "rsi" in first_row
        assert "macd" in first_row
        assert "bb_upper" in first_row
        assert "sma50" in first_row

    def test_all_keyword_expands_to_all_indicators(self, call_toon) -> None:
        """indicators=['all'] should expand to ALL_INDICATORS constant."""
        mock = MagicMock()
        np.random.seed(42)
        n = 250
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

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": ["all"]})

        assert "data" in parsed
        first_row = parsed["data"][0]

        data_indicator_keys = {
            "rsi": "rsi",
            "macd": "macd",
            "bb": "bb_upper",
            "stoch": "stoch_k",
            "fast_stoch": "fast_stoch_k",
            "trend": "sma50",
            "dmi": "dmi_plus",
            "ichimoku": "ichimoku_conversion",
            "williams": "williams_r",
            "cci": "cci",
            "atr": "atr",
            "obv": "obv",
            "momentum": "momentum",
        }

        meta_indicator_keys = {
            "volume_profile": "volume_profile",
            "price_change": "price_change",
            "fibonacci": "fibonacci",
            "pivot": "pivot",
        }

        for ind in ALL_INDICATORS:
            if ind.startswith("sma_") or ind.startswith("ema_") or ind.startswith("wma_"):
                assert ind in first_row, f"Missing MA indicator {ind} in data"
            elif ind in data_indicator_keys:
                expected_key = data_indicator_keys[ind]
                assert expected_key in first_row, (
                    f"Missing {expected_key} in data for indicator {ind}"
                )
            elif ind in meta_indicator_keys:
                expected_key = meta_indicator_keys[ind]
                assert "meta" in parsed, f"Missing meta for indicator {ind}"
                assert expected_key in parsed["meta"], (
                    f"Missing {expected_key} in meta for indicator {ind}"
                )

    def test_empty_indicators_defaults_to_all(self, call_toon) -> None:
        """Empty or omitted indicators should default to all."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": []})

        assert "data" in parsed
        first_row = parsed["data"][0]
        assert "rsi" in first_row
        assert "macd" in first_row

    def test_start_end_historical_range(self, call_toon) -> None:
        """start/end should fetch specific historical date range."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_prices()):
            parsed = call_toon(
                "technicals",
                {
                    "symbol": "AAPL",
                    "indicators": ["rsi"],
                    "start": "2020-01-01",
                    "end": "2020-12-31",
                },
            )

        assert "data" in parsed
        first_row = parsed["data"][0]
        assert "rsi" in first_row


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

    def test_empty_metrics_defaults_to_all(self, call) -> None:
        """Empty or omitted metrics should default to all."""
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_valuation()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": []})

        assert "pe" in parsed or "trailing_pe" in parsed
        assert "eps" in parsed or "trailing_eps" in parsed

    def _mock_historical_valuation(self) -> MagicMock:
        """Create mock with historical statements using relative dates."""
        from datetime import datetime

        mock = MagicMock()

        # Dynamic year computation - never hardcode years
        current_year = datetime.now().year
        prev_year = current_year - 1

        # Fiscal year end dates (September 30 like Apple)
        fy_current = pd.Timestamp(f"{current_year}-09-30")
        fy_prev = pd.Timestamp(f"{prev_year}-09-30")

        # Annual income statement
        income_data = {
            fy_current: {"Net Income": 93e9, "Total Revenue": 391e9},
            fy_prev: {"Net Income": 97e9, "Total Revenue": 383e9},
        }
        income_df = pd.DataFrame(income_data).T
        income_df.columns = pd.Index(["Net Income", "Total Revenue"])
        mock.income_stmt = income_df.T

        # Annual balance sheet
        balance_data = {
            fy_current: {"Stockholders Equity": 57e9, "Ordinary Shares Number": 15.1e9},
            fy_prev: {"Stockholders Equity": 62e9, "Ordinary Shares Number": 15.5e9},
        }
        balance_df = pd.DataFrame(balance_data).T
        balance_df.columns = pd.Index(["Stockholders Equity", "Ordinary Shares Number"])
        mock.balance_sheet = balance_df.T

        # Quarterly statements (5 quarters for TTM testing)
        q_dates = [
            pd.Timestamp(f"{current_year}-09-30"),
            pd.Timestamp(f"{current_year}-06-30"),
            pd.Timestamp(f"{current_year}-03-31"),
            pd.Timestamp(f"{prev_year}-12-31"),
            pd.Timestamp(f"{prev_year}-09-30"),
        ]
        q_income_data = {
            q_dates[0]: {"Net Income": 27e9, "Total Revenue": 102e9},
            q_dates[1]: {"Net Income": 23e9, "Total Revenue": 98e9},
            q_dates[2]: {"Net Income": 24e9, "Total Revenue": 100e9},
            q_dates[3]: {"Net Income": 35e9, "Total Revenue": 118e9},
            q_dates[4]: {"Net Income": 22e9, "Total Revenue": 90e9},
        }
        q_income_df = pd.DataFrame(q_income_data).T
        q_income_df.columns = pd.Index(["Net Income", "Total Revenue"])
        mock.quarterly_income_stmt = q_income_df.T

        q_balance_data = {
            q_dates[0]: {"Stockholders Equity": 73e9, "Ordinary Shares Number": 14.8e9},
            q_dates[1]: {"Stockholders Equity": 66e9, "Ordinary Shares Number": 14.9e9},
            q_dates[2]: {"Stockholders Equity": 67e9, "Ordinary Shares Number": 15.0e9},
            q_dates[3]: {"Stockholders Equity": 66e9, "Ordinary Shares Number": 15.0e9},
            q_dates[4]: {"Stockholders Equity": 62e9, "Ordinary Shares Number": 15.1e9},
        }
        q_balance_df = pd.DataFrame(q_balance_data).T
        q_balance_df.columns = pd.Index(["Stockholders Equity", "Ordinary Shares Number"])
        mock.quarterly_balance_sheet = q_balance_df.T

        # Mock info for current valuation
        mock.info = {"regularMarketPrice": 150.0, "shortName": "Test"}

        return mock

    def _mock_history_for_date(self, target_date: pd.Timestamp, price: float) -> pd.DataFrame:
        """Create mock price history around a target date."""
        dates = pd.date_range(target_date - pd.Timedelta(days=3), target_date, freq="D")
        return pd.DataFrame(
            {
                "Open": [price * 0.99] * len(dates),
                "High": [price * 1.01] * len(dates),
                "Low": [price * 0.98] * len(dates),
                "Close": [price] * len(dates),
                "Volume": [1000000] * len(dates),
            },
            index=dates,
        )

    def test_periods_now_uses_current_info(self, call) -> None:
        """periods='now' should use existing t.info path."""
        mock = self._mock_valuation()
        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["pe"], "periods": "now"})

        assert "pe" in parsed
        assert parsed["pe"] == 25.5  # From mock.info

    def test_periods_single_year(self, call) -> None:
        """periods='YYYY' should return valuation for that fiscal year."""
        from datetime import datetime

        prev_year = datetime.now().year - 1
        mock = self._mock_historical_valuation()
        fy_date = pd.Timestamp(f"{prev_year}-09-30")
        price_df = self._mock_history_for_date(fy_date, 170.0)

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=price_df),
        ):
            parsed = call(
                "valuation",
                {"symbol": "AAPL", "metrics": ["pe", "ratios"], "periods": str(prev_year)},
            )

        # Assert on structure, not specific year values
        assert len([k for k in parsed.keys() if not k.startswith("_")]) == 1
        date_key = [k for k in parsed.keys() if not k.startswith("_")][0]
        assert date_key.endswith("-09-30")
        assert "pe" in parsed[date_key]
        assert "pb" in parsed[date_key]
        assert "ps" in parsed[date_key]

    def test_periods_year_range(self, call) -> None:
        """periods='YYYY:YYYY' should return valuations for both years."""
        from datetime import datetime

        current_year = datetime.now().year
        prev_year = current_year - 1
        mock = self._mock_historical_valuation()

        def mock_history(symbol, start=None, end=None, interval="1d", **kwargs):
            # Return price data for any date range
            if start:
                start_date = pd.Timestamp(start)
                return self._mock_history_for_date(start_date + pd.Timedelta(days=3), 180.0)
            return pd.DataFrame()

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", side_effect=mock_history),
        ):
            parsed = call(
                "valuation",
                {"symbol": "AAPL", "metrics": ["pe"], "periods": f"{prev_year}:{current_year}"},
            )

        # Should have 2 date entries
        date_keys = [k for k in parsed.keys() if not k.startswith("_")]
        assert len(date_keys) == 2

    def test_periods_single_quarter(self, call) -> None:
        """periods='YYYY-QN' should return quarterly valuation with TTM."""
        from datetime import datetime

        current_year = datetime.now().year
        mock = self._mock_historical_valuation()
        q_date = pd.Timestamp(f"{current_year}-09-30")
        price_df = self._mock_history_for_date(q_date, 250.0)

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=price_df),
        ):
            parsed = call(
                "valuation",
                {"symbol": "AAPL", "metrics": ["pe"], "periods": f"{current_year}-Q3"},
            )

        date_keys = [k for k in parsed.keys() if not k.startswith("_")]
        assert len(date_keys) == 1
        date_key = date_keys[0]
        assert "pe" in parsed[date_key]
        # TTM note should be present for quarterly
        assert "_note" in parsed[date_key]

    def test_periods_unavailable_year_error(self, call) -> None:
        """Unavailable year should return error with available years."""
        mock = self._mock_historical_valuation()
        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["pe"], "periods": "2005"})

        assert parsed["err"] == "VALIDATION_ERROR"
        assert "2005" in parsed["msg"]
        assert "Available" in parsed["msg"]

    def test_periods_invalid_format_error(self, call) -> None:
        """Invalid format should return validation error."""
        mock = self._mock_historical_valuation()
        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["pe"], "periods": "invalid"})

        assert parsed["err"] == "VALIDATION_ERROR"

    def test_historical_unsupported_metrics(self, call) -> None:
        """Metrics like 'quality' should be listed as unsupported in historical mode."""
        from datetime import datetime

        prev_year = datetime.now().year - 1
        mock = self._mock_historical_valuation()
        fy_date = pd.Timestamp(f"{prev_year}-09-30")
        price_df = self._mock_history_for_date(fy_date, 170.0)

        with (
            patch("yfinance_mcp.server._ticker", return_value=mock),
            patch("yfinance_mcp.history.get_history", return_value=price_df),
        ):
            parsed = call(
                "valuation",
                {"symbol": "AAPL", "metrics": ["pe", "quality"], "periods": str(prev_year)},
            )

        assert "_unsupported" in parsed
        assert "quality" in parsed["_unsupported"]


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

    def _mock_financials_with_years(self) -> MagicMock:
        """Create mock with multiple fiscal years for period filtering tests."""
        from datetime import datetime

        mock = MagicMock()
        current_year = datetime.now().year
        dates = pd.DatetimeIndex(
            [
                pd.Timestamp(f"{current_year}-12-31"),
                pd.Timestamp(f"{current_year - 1}-12-31"),
                pd.Timestamp(f"{current_year - 2}-12-31"),
            ]
        )
        for stmt, row_data in [
            ("get_income_stmt", {"TotalRevenue": [120000, 100000, 90000]}),
            ("get_balance_sheet", {"TotalAssets": [600000, 500000, 450000]}),
            ("get_cashflow", {"OperatingCashFlow": [30000, 25000, 22000]}),
        ]:
            df = pd.DataFrame(row_data, index=dates).T
            getattr(mock, stmt).return_value = df
        return mock

    def test_periods_single_year_filters(self, call) -> None:
        """periods='YYYY' should filter to that fiscal year only."""
        from datetime import datetime

        prev_year = datetime.now().year - 1
        with patch("yfinance_mcp.server._ticker", return_value=self._mock_financials_with_years()):
            parsed = call("financials", {"symbol": "AAPL", "periods": str(prev_year)})

        date_cols = [k for k in parsed.keys() if not k.startswith("_")]
        assert len(date_cols) == 1
        assert str(prev_year) in date_cols[0]


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

    @pytest.mark.parametrize(
        "indicators,expected_unknown,should_have_valid",
        [
            (["sma_999"], None, False),  # Large period returns null in data
            (["sma_abc", "rsi"], "sma_abc", True),  # Invalid format
            (["sma_-5"], "sma_-5", False),  # Negative period
            (["nonexistent", "rsi"], "nonexistent", True),  # Unknown indicator
        ],
    )
    def test_invalid_indicators_handled_gracefully(
        self, call_toon, indicators, expected_unknown, should_have_valid
    ) -> None:
        """Invalid indicators should be added to _unknown list in meta or return null."""
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
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": indicators})

        if expected_unknown:
            assert "meta" in parsed
            assert "_unknown" in parsed["meta"]
            assert expected_unknown in parsed["meta"]["_unknown"]
        if should_have_valid:
            assert "data" in parsed
            first_row = parsed["data"][0]
            assert "rsi" in first_row

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
        """Call tool and parse TOON or JSON response."""
        result = asyncio.run(call_tool(name, args))
        return _parse_response(result[0].text)

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
            result = self._call("history", {"symbol": "TEST"})

        bars = result["bars"]
        first_bar = bars[0]
        assert first_bar["c"] > 0, f"Close price {price} rounded to zero"


class TestAutoInterval:
    """Test auto-interval selection and downsampling logic."""

    @pytest.mark.parametrize(
        "period,expected",
        [
            ("1d", 1),
            ("5d", 5),
            ("1mo", 30),
            ("3mo", 90),
            ("1y", 365),
            ("5y", 1825),
            ("ytd", 180),
            ("max", 7300),
            ("unknown", 90),
            (None, 90),  # fallbacks
        ],
    )
    def test_calculate_span_days_period(self, period, expected) -> None:
        from yfinance_mcp.helpers import calculate_span_days

        assert calculate_span_days(period) == expected

    @pytest.mark.parametrize(
        "start,end,expected",
        [
            ("2024-01-01", "2024-01-31", 30),
            ("2024-01-01", "2024-12-31", 365),
            ("2024-01-01", "2024-01-01", 0),
        ],
    )
    def test_calculate_span_days_dates(self, start, end, expected) -> None:
        from yfinance_mcp.helpers import calculate_span_days

        assert calculate_span_days(start=start, end=end) == expected

    @pytest.mark.parametrize(
        "span_days,expected_interval",
        [
            # Boundaries
            (0, "5m"),
            (3, "15m"),
            (12, "1h"),
            (80, "1d"),
            (400, "1wk"),
            (1600, "1mo"),
            # Within ranges
            (1, "5m"),
            (5, "15m"),
            (20, "1h"),
            (100, "1d"),
            (500, "1wk"),
            (2000, "1mo"),
            # Floats
            (0.5, "5m"),
            (2.9, "5m"),
            (79.9, "1h"),
            (80.1, "1d"),
        ],
    )
    def test_auto_interval(self, span_days, expected_interval) -> None:
        from yfinance_mcp.helpers import auto_interval

        assert auto_interval(span_days) == expected_interval

    @pytest.mark.parametrize(
        "size_multiplier,should_downsample",
        [(0, False), (0.5, False), (1, False), (2, True), (5, True)],
    )
    def test_auto_downsample(self, size_multiplier, should_downsample) -> None:
        import pandas as pd

        from yfinance_mcp.helpers import TARGET_POINTS, auto_downsample

        size = max(0, int(TARGET_POINTS * size_multiplier))
        df = pd.DataFrame({"a": range(size)}) if size > 0 else pd.DataFrame()
        result = auto_downsample(df)

        if should_downsample:
            assert len(result) == TARGET_POINTS
            assert result.iloc[0]["a"] == 0  # first preserved
            assert result.iloc[-1]["a"] == size - 1  # last preserved
        else:
            assert len(result) == size

    @pytest.mark.parametrize(
        "period,start,end,should_error",
        [
            ("1y", None, None, False),  # 1 year period - within limit
            ("2y", None, None, False),  # 2 year period - within limit
            ("5y", None, None, True),  # 5 year period - exceeds limit
            ("max", None, None, True),  # max period - exceeds limit
            (None, "2023-01-01", "2025-01-01", False),  # 2y date range - within limit
            (None, "2015-01-01", "2025-01-01", True),  # 10y date range - exceeds
            (None, "2024-01-01", "2024-06-01", False),  # 5mo date range - within
        ],
    )
    def test_validate_date_range(self, period, start, end, should_error) -> None:
        from yfinance_mcp.helpers import DateRangeExceededError, validate_date_range

        if should_error:
            with pytest.raises(DateRangeExceededError) as exc_info:
                validate_date_range(period, start, end)
            assert exc_info.value.max_days > 0
            assert exc_info.value.suggested_period in ("1y", "2y")
        else:
            validate_date_range(period, start, end)  # Should not raise

    @pytest.mark.parametrize(
        "max_days,expected_count,must_include,must_exclude",
        [
            (30, 5, ["1d", "5d", "1w", "2w", "1mo"], ["ytd", "1y"]),
            (100, 7, ["1d", "3mo"], ["ytd", "1y"]),
            (200, 7, ["1d", "ytd"], ["1y", "2y"]),
            (400, 7, ["1d", "ytd", "1y"], ["2y", "5y"]),
            (1120, 7, ["1d", "ytd", "3y"], ["5y", "10y"]),
            (2000, 7, ["1d", "ytd", "5y"], ["10y", "max"]),
            (4000, 7, ["1d", "ytd", "10y"], ["max"]),
            (8000, 7, ["1d", "ytd", "max"], []),
        ],
    )
    def test_get_valid_periods(self, max_days, expected_count, must_include, must_exclude) -> None:
        from yfinance_mcp.helpers import MAX_PERIOD_OPTIONS, get_valid_periods

        periods = get_valid_periods(max_days)

        assert len(periods) <= MAX_PERIOD_OPTIONS
        assert len(periods) == expected_count

        for p in must_include:
            assert p in periods, f"Expected {p} in {periods} for max_days={max_days}"

        for p in must_exclude:
            assert p not in periods, f"Expected {p} NOT in {periods} for max_days={max_days}"

    def test_get_valid_periods_sorted_by_duration(self) -> None:
        from yfinance_mcp.helpers import PERIOD_TO_DAYS, get_valid_periods

        periods = get_valid_periods(1120)
        durations = [PERIOD_TO_DAYS.get(p, 0) for p in periods]
        assert durations == sorted(durations), "Periods should be sorted by duration"
