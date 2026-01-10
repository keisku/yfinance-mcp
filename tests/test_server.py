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


def split_to_row(data: dict, row_idx: int = 0) -> dict:
    """Reconstruct row dict from delta-encoded split format."""
    cols = data["columns"]
    row = data["values"][row_idx]
    return dict(zip(cols, row))


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

    def test_returns_identity_and_price(self, call, mock_ticker_factory) -> None:
        """search_stock should return identity + current price only."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()):
            parsed = call("search_stock", {"symbol": "AAPL"})

        expected_keys = [
            "symbol",
            "name",
            "quote_type",
            "sector",
            "industry",
            "exchange",
            "price",
            "change_pct",
            "market_cap",
            "volume",
        ]
        for key in expected_keys:
            assert key in parsed
        excluded_keys = ["pe", "peg", "trend", "quality_score"]
        for key in excluded_keys:
            assert key not in parsed

    def test_search_by_query(self, call, mock_ticker_factory) -> None:
        """search_stock should work with company name query."""
        with (
            patch("yfinance.Search") as mock_search,
            patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()),
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

    def test_smart_search_strips_suffix(self, call, mock_ticker_factory) -> None:
        """search_stock should strip common suffixes and retry (e.g., 'DBS Bank' -> 'DBS')."""

        def search_side_effect(query, **kwargs):
            result = MagicMock()
            result.quotes = [{"symbol": "D05.SI"}] if query.lower() == "dbs" else []
            return result

        with (
            patch("yfinance.Search", side_effect=search_side_effect),
            patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()),
        ):
            parsed = call("search_stock", {"query": "DBS Bank"})

        assert parsed["symbol"] == "D05.SI"
        assert "price" in parsed

    @pytest.mark.parametrize(
        "exchange_filter,expected_symbol",
        [
            ("JPX", "2871.T"),  # exact match
            ("jpx", "2871.T"),  # case insensitive
        ],
    )
    def test_exchange_filter(
        self, call, mock_ticker_factory, exchange_filter, expected_symbol
    ) -> None:
        """search_stock should filter results by exchange when specified."""

        def search_side_effect(query, **kwargs):
            result = MagicMock()
            result.quotes = [
                {"symbol": "NI3.F", "exchange": "FRA"},
                {"symbol": "NI3.SG", "exchange": "STU"},
                {"symbol": "2871.T", "exchange": "JPX"},
            ]
            return result

        with (
            patch("yfinance.Search", side_effect=search_side_effect),
            patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()),
        ):
            parsed = call("search_stock", {"query": "Nichirei", "exchange": exchange_filter})

        assert parsed["symbol"] == expected_symbol

    def test_exchange_filter_no_match_returns_actionable_error(self, call) -> None:
        """search_stock should return actionable error with available exchanges."""

        def search_side_effect(query, **kwargs):
            result = MagicMock()
            result.quotes = [
                {"symbol": "NI3.F", "exchange": "FRA"},
                {"symbol": "2871.T", "exchange": "JPX"},
            ]
            return result

        with patch("yfinance.Search", side_effect=search_side_effect):
            parsed = call("search_stock", {"query": "Nichirei", "exchange": "NYSE"})

        assert parsed["err"] == "SYMBOL_NOT_FOUND"
        assert "hint" in parsed
        assert "NYSE" in parsed["hint"]
        assert "FRA" in parsed["hint"] or "JPX" in parsed["hint"]
        assert "Available:" in parsed["hint"]


class TestFilterByExchange:
    """Unit tests for _filter_by_exchange helper."""

    @pytest.mark.parametrize(
        "quotes,exchange,expected_len,expected_first",
        [
            # exact match
            (
                [{"symbol": "NI3.F", "exchange": "FRA"}, {"symbol": "2871.T", "exchange": "JPX"}],
                "JPX",
                1,
                "2871.T",
            ),
            # case insensitive
            (
                [{"symbol": "2871.T", "exchange": "JPX"}],
                "jpx",
                1,
                "2871.T",
            ),
            # no match returns empty list
            (
                [{"symbol": "NI3.F", "exchange": "FRA"}, {"symbol": "2871.T", "exchange": "JPX"}],
                "NYSE",
                0,
                None,
            ),
            # None exchange returns original
            (
                [{"symbol": "AAPL", "exchange": "NMS"}],
                None,
                1,
                "AAPL",
            ),
            # empty quotes
            (
                [],
                "JPX",
                0,
                None,
            ),
        ],
    )
    def test_filter_by_exchange(self, quotes, exchange, expected_len, expected_first) -> None:
        from yfinance_mcp.helpers import _filter_by_exchange

        result = _filter_by_exchange(quotes, exchange)
        assert len(result) == expected_len
        if expected_first:
            assert result[0]["symbol"] == expected_first


class TestHistoryTool:
    """Test price tool - historical OHLCV data."""

    def test_returns_bars(self, call_toon, mock_ticker_with_history) -> None:
        """History should return bars in delta-encoded split format."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history()):
            parsed = call_toon("history", {"symbol": "AAPL"})

        assert "bars" in parsed
        assert set(parsed["bars"]["columns"]) == {"o", "h", "l", "c", "ac", "v"}
        assert "base_ts" in parsed["bars"]
        assert "deltas" in parsed["bars"]
        assert "values" in parsed["bars"]
        assert len(parsed["bars"]["values"]) > 0

    @pytest.mark.parametrize(
        "args,description",
        [
            ({"start": "2024-01-01", "end": "2024-01-31"}, "date range"),
            ({"start": "2024-01-01"}, "start only defaults end to today"),
            ({"end": "2024-06-30", "period": "3mo"}, "end only computes start from period"),
        ],
    )
    def test_date_range_variations(
        self, call_toon, mock_ticker_with_history, args, description
    ) -> None:
        """Various date range specifications should return bars."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history()):
            parsed = call_toon("history", {"symbol": "AAPL", **args})

        assert "bars" in parsed
        assert len(parsed["bars"]["values"]) > 0

    def test_intraday_datetime_strings(self, call_toon, mock_ticker_with_history) -> None:
        """Short time spans should auto-select intraday interval."""
        mock = mock_ticker_with_history()
        df = mock.history.return_value.copy()
        df.index = pd.date_range("2024-01-15 09:30", periods=len(df), freq="5min")
        mock.history.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call_toon(
                "history",
                {"symbol": "AAPL", "start": "2024-01-15 09:30", "end": "2024-01-15 16:00"},
            )

        assert "bars" in parsed
        assert "T" in parsed["bars"]["base_ts"]  # Contains time component (ISO 8601 format)

    @pytest.mark.parametrize(
        "close,adj_close,check",
        [
            # Dividend stock: ac < c (~6% adjustment)
            ([100.0, 101.0, 102.0], [94.0, 94.94, 95.88], {"ac_lt_c": True}),
            # Stock split: ac << c (4:1 ratio)
            ([120.0, 121.0, 122.0], [30.0, 30.25, 30.5], {"ratio_gt": 3.5}),
            # Index: ac == c (no dividends)
            ([4500.0, 4510.0, 4520.0], [4500.0, 4510.0, 4520.0], {"ac_eq_c": True}),
        ],
        ids=["dividend", "split", "index"],
    )
    def test_adj_close_vs_close_relationship(
        self, call_toon, mock_ticker_with_history, close, adj_close, check
    ) -> None:
        """Verify ac/c relationship for dividends, splits, and indices."""
        mock = mock_ticker_with_history()
        df = pd.DataFrame(
            {
                "Open": [c - 0.5 for c in close],
                "High": [c + 1 for c in close],
                "Low": [c - 1 for c in close],
                "Close": close,
                "Adj Close": adj_close,
                "Volume": [1000000] * len(close),
            },
            index=pd.date_range("2024-01-01", periods=len(close), freq="D"),
        )
        mock.history.return_value = df

        # Use start/end to bypass cache and hit direct API path
        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call_toon(
                "history", {"symbol": "TEST", "start": "2024-01-01", "end": "2024-01-10"}
            )

        bars = parsed["bars"]
        cols = bars["columns"]
        c_val = bars["values"][0][cols.index("c")]
        ac_val = bars["values"][0][cols.index("ac")]

        if check.get("ac_lt_c"):
            assert ac_val < c_val, "ac should be lower for dividend stocks"
        if check.get("ratio_gt"):
            assert c_val / ac_val > check["ratio_gt"], f"Expected ratio > {check['ratio_gt']}"
        if check.get("ac_eq_c"):
            assert ac_val == c_val, "Index should have ac == c"
            assert "_issues" not in parsed, "No warning when Adj Close exists"

    def test_missing_trading_days_handled(self, call_toon, mock_ticker_with_history) -> None:
        """Weekends/holidays should not fill gaps; dt array shows skips."""
        mock = mock_ticker_with_history()
        n = 10
        df = pd.DataFrame(
            {
                col: [100 + i for i in range(n)]
                for col in ["Open", "High", "Low", "Close", "Adj Close"]
            }
            | {"Volume": [1000000] * n},
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        mock.history.return_value = df

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call_toon(
                "history", {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-01-15"}
            )

        assert len(parsed["bars"]["values"]) == n
        assert any(d > 1 for d in parsed["bars"]["deltas"]), "Weekend gaps expected"

    def test_adj_close_fallback_shows_warning(self, call_toon, mock_ticker_with_history) -> None:
        """Missing Adj Close column should trigger fallback with _issues warning."""
        mock = mock_ticker_with_history()
        mock.history.return_value = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100.5], "Volume": [1000000]},
            index=pd.date_range("2024-01-01", periods=1),
        )

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call_toon("history", {"symbol": "WEIRD"})

        assert "_issues" in parsed and "ac" in parsed["_issues"]
        assert "using Close" in parsed["_issues"]["ac"]
        assert "ac" in parsed["bars"]["columns"]

    def test_empty_history_returns_error(self, call_toon, mock_ticker_with_history) -> None:
        """Empty history data should return error."""
        mock = mock_ticker_with_history()
        mock.history.return_value = pd.DataFrame()

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call_toon("history", {"symbol": "DELISTED"})

        assert "err" in parsed


class TestTechnicalsTool:
    """Test technicals tool - trading signals."""

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
        self, call_toon, mock_ticker_with_history, indicator, expected_data_keys, expected_meta_keys
    ) -> None:
        """Each indicator should return time series data or extras at top level."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=100)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": [indicator]})

        if expected_data_keys:
            assert "data" in parsed
            first_row = split_to_row(parsed["data"])
            for key in expected_data_keys:
                assert key in first_row, f"Missing {key} for {indicator}"
        if expected_meta_keys:
            # Non-timeseries extras are at top level (no data[] when only extras)
            for key in expected_meta_keys:
                assert key in parsed, f"Missing {key} at top level for {indicator}"

    def test_trend_insufficient_data(self, call_toon, mock_ticker_with_history) -> None:
        """Trend with <50 bars should return error in _issues."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=30)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": ["trend"]})

        assert "_issues" in parsed
        assert "insufficient_data" in parsed["_issues"]
        assert "trend" in parsed["_issues"]["insufficient_data"]

    def test_all_indicators(self, call_toon, mock_ticker_with_history) -> None:
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
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=100)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": indicators})

        assert "data" in parsed
        first_row = split_to_row(parsed["data"])
        for key in ["rsi", "macd", "bb_upper", "sma50"]:
            assert key in first_row

    def test_all_keyword_expands_to_all_indicators(
        self, call_toon, mock_ticker_with_history
    ) -> None:
        """indicators=['all'] should expand to ALL_INDICATORS constant."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=250)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": ["all"]})

        assert "data" in parsed
        first_row = split_to_row(parsed["data"])

        data_keys = {
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
        meta_keys = {
            "volume_profile": "volume_profile",
            "price_change": "price_change",
            "fibonacci": "fibonacci",
            "pivot": "pivot",
        }

        for ind in ALL_INDICATORS:
            if ind.startswith(("sma_", "ema_", "wma_")):
                assert ind in first_row, f"Missing MA {ind}"
            elif ind in data_keys:
                assert data_keys[ind] in first_row, f"Missing {data_keys[ind]} for {ind}"
            elif ind in meta_keys:
                # Extras are now at top level, not nested under "meta"
                assert meta_keys[ind] in parsed, f"Missing {meta_keys[ind]} at top level"

    def test_empty_indicators_defaults_to_all(self, call_toon, mock_ticker_with_history) -> None:
        """Empty or omitted indicators should default to all."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=100)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": []})

        assert "data" in parsed
        first_row = split_to_row(parsed["data"])
        assert all(k in first_row for k in ["rsi", "macd"])

    def test_start_end_historical_range(self, call_toon, mock_ticker_with_history) -> None:
        """start/end should fetch specific historical date range."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=100)):
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
        first_row = split_to_row(parsed["data"])
        assert "rsi" in first_row


class TestTechnicalsActionableFeedback:
    """Test actionable feedback in technicals tool responses."""

    @pytest.mark.parametrize(
        "n_bars,indicators,issue_type,expected_keys,expected_hints",
        [
            # Insufficient data with specific period hints
            (10, ["sma_200"], "insufficient_data", ["sma_200"], {"sma_200": "1y"}),
            (10, ["sma_100"], "insufficient_data", ["sma_100"], {"sma_100": "6mo"}),
            (10, ["sma_50"], "insufficient_data", ["sma_50"], {"sma_50": "3mo"}),
            # Unknown indicators
            (50, ["fake_ind", "also_fake"], "unknown", ["fake_ind", "also_fake"], {}),
        ],
    )
    def test_issues_structure(
        self,
        call_toon,
        mock_ticker_with_history,
        n_bars,
        indicators,
        issue_type,
        expected_keys,
        expected_hints,
    ) -> None:
        """Issues should be grouped by type with actionable hints."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=n_bars)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": indicators})

        assert "_issues" in parsed
        assert issue_type in parsed["_issues"]
        for key in expected_keys:
            assert key in parsed["_issues"][issue_type]
        for key, hint in expected_hints.items():
            assert hint in parsed["_issues"][issue_type][key]

    def test_partial_data_shows_when_warmup_dominates(
        self, call_toon, mock_ticker_with_history
    ) -> None:
        """Warmup nulls should warn only when valid data < 50%."""
        # 80 bars with SMA50: 49 nulls, 31 valid = 38.75% → warning shown
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=80)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": ["sma_50"]})

        assert "_issues" in parsed
        assert "partial_data" in parsed["_issues"]
        assert "period='3mo'" in parsed["_issues"]["partial_data"]["sma_50"]

    @pytest.mark.parametrize(
        "n_bars,indicators,expect_data,expect_issues",
        [
            (10, ["sma_200", "sma_100"], False, True),  # All fail → issues only
            (50, ["rsi", "sma_200", "unknown"], True, True),  # Mixed → both
            (200, ["rsi", "macd"], True, False),  # All succeed → data only
        ],
    )
    def test_data_and_issues_presence(
        self, call_toon, mock_ticker_with_history, n_bars, indicators, expect_data, expect_issues
    ) -> None:
        """Response should have data/issues based on indicator success."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=n_bars)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": indicators})

        assert ("data" in parsed) == expect_data
        assert ("_issues" in parsed) == expect_issues

    @pytest.mark.parametrize(
        "indicator,top_level_key,nested_key",
        [
            ("volume_profile", "volume_profile", "poc"),
            ("price_change", "price_change", "change_pct"),
            ("fibonacci", "fibonacci", "levels"),
            ("pivot", "pivot", "levels"),
        ],
    )
    def test_summaries_at_top_level(
        self, call_toon, mock_ticker_with_history, indicator, top_level_key, nested_key
    ) -> None:
        """Single-value results should be at top level, not in data[]."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=50)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": [indicator]})

        assert top_level_key in parsed
        assert nested_key in parsed[top_level_key]

    def test_all_null_rows_excluded(self, call_toon, mock_ticker_with_history) -> None:
        """Rows with all null values should be excluded from data[] to save tokens."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=80)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": ["sma_50"]})

        if "data" in parsed:
            for row_values in parsed["data"]["values"]:
                # Each row should have at least one non-null value
                assert any(v is not None for v in row_values), "All-null row should be excluded"

    def test_all_failed_returns_toon_format(self, mock_ticker_with_history) -> None:
        """When all indicators fail, response should still be TOON format (not JSON)."""
        import asyncio

        from yfinance_mcp.server import call_tool

        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=10)):
            result = asyncio.run(
                call_tool("technicals", {"symbol": "AAPL", "indicators": ["sma_500", "sma_300"]})
            )

        text = result[0].text
        # TOON format uses colons for key-value, not JSON's {"key": "value"}
        assert "_issues:" in text or "insufficient_data:" in text


class TestValuationTool:
    """Test valuation tool - valuation metrics and quality score."""

    @pytest.mark.parametrize(
        "metrics,expected_keys",
        [
            (["pe", "eps", "margins"], ["pe", "eps", "margin_gross"]),
            (["growth"], ["growth_rev", "growth_earn"]),
            (["ratios"], ["pb", "ps", "ev_ebitda"]),
            (["dividends"], ["div_yield", "div_rate", "payout_ratio"]),
            (["peg"], ["peg", "peg_source", "peg_signal"]),
        ],
    )
    def test_valuation_metrics(self, call, mock_ticker_factory, metrics, expected_keys) -> None:
        """Valuation metrics should return expected keys."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": metrics})

        for key in expected_keys:
            assert key in parsed, f"Missing {key} for metrics {metrics}"

    def test_quality_metric(self, call, mock_ticker_factory) -> None:
        """Quality returns score, signal, and details."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["quality"]})

        assert parsed["quality_max"] == 7
        assert parsed["quality_signal"] in ["strong", "neutral", "weak"]
        assert all(k in parsed for k in ["quality_score", "quality_details"])
        assert parsed["quality_score"] >= 6
        assert parsed["quality_signal"] == "strong"

    def test_peg_undervalued_signal(self, call, mock_ticker_factory) -> None:
        """Low PE with high growth should signal undervalued."""
        mock = mock_ticker_factory(trailingPE=10, earningsGrowth=0.20)

        with patch("yfinance_mcp.server._ticker", return_value=mock):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["peg"]})

        assert parsed["peg"] == 0.5
        assert parsed["peg_signal"] == "undervalued"

    def test_empty_metrics_defaults_to_all(self, call, mock_ticker_factory) -> None:
        """Empty or omitted metrics should default to all."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()):
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

    def test_periods_now_uses_current_info(self, call, mock_ticker_factory) -> None:
        """periods='now' should use existing t.info path."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()):
            parsed = call("valuation", {"symbol": "AAPL", "metrics": ["pe"], "periods": "now"})

        assert "pe" in parsed
        assert parsed["pe"] == 25.5  # From DEFAULT_VALUATION_INFO

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

    @pytest.mark.parametrize(
        "statement",
        ["income", "balance", "cashflow"],
    )
    def test_financial_statements(self, call, mock_financials_factory, statement) -> None:
        """Financial statements should return data."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_financials_factory()):
            parsed = call("financials", {"symbol": "AAPL", "statement": statement})

        assert len(parsed) >= 1

    def test_periods_single_year_filters(self, call) -> None:
        """periods='YYYY' should filter to that fiscal year only."""
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

        prev_year = current_year - 1
        with patch("yfinance_mcp.server._ticker", return_value=mock):
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
    """Test circuit breaker - protects against cascading failures."""

    def test_rejects_requests_when_open(self, call) -> None:
        """When circuit is open, requests should be rejected immediately."""
        open_circuit_breaker_for_testing()
        parsed = call("search_stock", {"symbol": "AAPL"})
        assert parsed["err"] == "DATA_UNAVAILABLE"
        assert "retry later" in parsed["msg"].lower()

    def test_allows_requests_when_closed(self, call, mock_ticker_factory) -> None:
        """When circuit is closed, requests should proceed normally."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_factory()):
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
        self, call_toon, mock_ticker_with_history, indicators, expected_unknown, should_have_valid
    ) -> None:
        """Invalid indicators should be added to _issues.unknown or return null."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_ticker_with_history(n=50)):
            parsed = call_toon("technicals", {"symbol": "AAPL", "indicators": indicators})

        if expected_unknown:
            assert "_issues" in parsed and "unknown" in parsed["_issues"]
            assert expected_unknown in parsed["_issues"]["unknown"]
        if should_have_valid:
            assert "data" in parsed and "rsi" in split_to_row(parsed["data"])

    def test_invalid_statement_defaults_to_cashflow(self, call, mock_financials_factory) -> None:
        """Invalid statement type defaults to cashflow (graceful fallback)."""
        with patch("yfinance_mcp.server._ticker", return_value=mock_financials_factory()):
            parsed = call("financials", {"symbol": "AAPL", "statement": "invalid"})

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
        first_bar = split_to_row(bars, 0)
        assert first_bar["c"] > 0, f"Close price {price} rounded to zero"


class TestAutoInterval:
    """Test auto-interval selection and downsampling logic."""

    @pytest.mark.parametrize(
        "period,expected_interval",
        [
            # Coarsest interval where bars >= 100 (TARGET_POINTS/2)
            # Calendar days converted to trading days via × 5/7
            # US markets (390 min): 5m=78, 15m=26, 30m=13, 1h=6.5, 1d=1, 1wk=0.2 per trading day
            ("1d", "5m"),  # 1 × 5/7 × 78 = 55.7 < 100 → 5m fallback
            ("1mo", "1h"),  # 30 × 5/7 × 6.5 = 139 ≥ 100
            ("3mo", "1h"),  # 90 × 5/7 × 6.5 = 418 ≥ 100
            ("2y", "1wk"),  # 730 × 5/7 × 0.2 = 104.3 ≥ 100
            ("5y", "1wk"),  # 1825 × 5/7 × 0.2 = 260.7 ≥ 100 (but exceeds MAX_PERIOD)
            # Fallbacks
            ("unknown_period", "1h"),  # Unknown → 90 days default → 1h
            (None, "1h"),  # None → 90 days default → 1h
        ],
    )
    def test_select_interval_period(self, period, expected_interval) -> None:
        from yfinance_mcp.helpers import select_interval

        assert select_interval(period) == expected_interval

    @pytest.mark.parametrize(
        "start,end,expected_interval",
        [
            # Coarsest interval where trading_days × ppd >= 100 (TARGET_POINTS/2)
            # trading_days = calendar_days × 5/7
            # US markets (390 min): 5m=78, 15m=26, 30m=13, 1h=6.5, 1d=1, 1wk=0.2 ppd
            ("2024-01-01", "2024-01-01", "5m"),  # 0d → all fail → 5m fallback
            (
                "2024-01-01",
                "2024-01-05",
                "5m",
            ),  # 4 × 5/7 × 78 = 223 ≥ 100 → 5m (coarsest that works)
            ("2024-01-01", "2024-01-09", "15m"),  # 8 × 5/7 × 26 = 149 ≥ 100
            ("2024-01-01", "2024-01-17", "30m"),  # 16 × 5/7 × 13 = 149 ≥ 100
            ("2024-01-01", "2024-04-20", "1h"),  # 110 × 5/7 × 6.5 = 511 ≥ 100
            ("2024-01-01", "2024-05-20", "1d"),  # 140 × 5/7 × 1 = 100 ≥ 100
            ("2024-01-01", "2025-12-02", "1wk"),  # 701 × 5/7 × 0.2 = 100.1 ≥ 100
            ("2020-01-01", "2024-01-01", "1wk"),  # 1461 × 5/7 × 0.2 = 208.7 ≥ 100
            # Edge cases
            ("2024-01-01", "2024-01-02", "5m"),  # 1 × 5/7 × 78 = 55.7 < 100 → 5m fallback
            ("2024-01-01", "2024-01-03", "5m"),  # 2 × 5/7 × 78 = 111 ≥ 100 → 5m
        ],
    )
    def test_select_interval_dates(self, start, end, expected_interval) -> None:
        from yfinance_mcp.helpers import select_interval

        assert select_interval(start=start, end=end) == expected_interval

    def test_select_interval_start_without_end_uses_now(self) -> None:
        """Start without end should use current timestamp."""
        from datetime import datetime, timedelta

        from yfinance_mcp.helpers import select_interval

        # 30 days ago: 30 × 6.5 = 195 ≥ 100 → 1h
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        assert select_interval(start=start) == "1h"

    def test_select_interval_start_overrides_period(self) -> None:
        """When both start and period provided, start takes precedence."""
        from yfinance_mcp.helpers import select_interval

        # period=5y would give 1wk, but 30-day date range gives 1h
        result = select_interval(
            period="5y",
            start="2024-01-01",
            end="2024-01-31",
        )
        assert result == "1h"

    def test_select_interval_negative_span_returns_5m(self) -> None:
        """End before start (negative span) should return finest interval."""
        from yfinance_mcp.helpers import select_interval

        result = select_interval(start="2024-12-31", end="2024-01-01")
        assert result == "5m"

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
            # Period-based tests (MAX_PERIOD_DAYS = 1000 trading days for TARGET_POINTS=200)
            # Calendar days converted to trading days via × 5/7
            ("1y", None, None, False),  # 365 × 5/7 = 261 trading days - within limit
            ("2y", None, None, False),  # 730 × 5/7 = 521 trading days - within limit
            ("3y", None, None, False),  # 1095 × 5/7 = 782 trading days - within limit
            ("5y", None, None, True),  # 1825 × 5/7 = 1304 trading days - exceeds 1000 limit
            ("10y", None, None, True),  # 3650 × 5/7 = 2607 trading days - exceeds limit
            ("max", None, None, True),  # 7300 × 5/7 = 5214 trading days - exceeds limit
            ("unknown_period", None, None, False),  # defaults to 90 days
            # Date-based tests
            (None, "2023-01-01", "2025-01-01", False),  # 731 × 5/7 = 522 trading days - within
            (None, "2022-03-01", "2026-01-01", True),  # 1402 × 5/7 = 1001 trading days - exceeds
            (None, "2024-01-01", "2024-06-01", False),  # 152 × 5/7 = 109 trading days - within
            (None, "2024-01-01", "2024-01-01", False),  # same day (0 days)
            (None, "2024-01-01", None, False),  # end defaults to today
            # Boundary tests (~1000 trading day limit = ~1400 calendar days)
            (None, "2021-03-29", "2025-01-01", False),  # 1374 × 5/7 = 981 trading days - under
            (None, "2021-03-01", "2025-01-01", True),  # 1402 × 5/7 = 1001 trading days - exceeds
        ],
    )
    def test_validate_date_range(self, period, start, end, should_error) -> None:
        from yfinance_mcp.helpers import DateRangeExceededError, validate_date_range

        if should_error:
            with pytest.raises(DateRangeExceededError) as exc_info:
                validate_date_range(period, start, end)
            assert exc_info.value.max_days > 0
            assert exc_info.value.requested_days > exc_info.value.max_days
        else:
            validate_date_range(period, start, end)  # Should not raise

    def test_date_range_error_contains_actionable_splits(self) -> None:
        """Verify error message provides specific date ranges for sequential requests."""
        from yfinance_mcp.helpers import DateRangeExceededError, validate_date_range

        with pytest.raises(DateRangeExceededError) as exc_info:
            validate_date_range(start="2010-01-01", end="2025-01-01")

        msg = str(exc_info.value)
        assert "Split into" in msg
        assert "start=" in msg and "end=" in msg
        assert "YFINANCE_TARGET_POINTS" in msg

    def test_period_error_suggests_alternative(self) -> None:
        """Verify period-based error provides helpful suggestion."""
        from yfinance_mcp.helpers import DateRangeExceededError, validate_date_range

        with pytest.raises(DateRangeExceededError) as exc_info:
            validate_date_range(period="5y")

        msg = str(exc_info.value)
        assert "3y" in msg or "split" in msg.lower()

    @pytest.mark.parametrize(
        "max_trading_days,expected_count,must_include,must_exclude",
        [
            # max_trading_days compared against calendar_days × 5/7
            (21, 5, ["1d", "5d", "1w", "2w", "1mo"], ["ytd", "1y"]),
            (70, 7, ["1d", "3mo"], ["ytd", "1y"]),
            (140, 9, ["1d", "ytd", "6mo"], ["1y", "2y"]),
            (280, 11, ["1d", "ytd", "1y"], ["2y", "5y"]),
            (800, 14, ["1d", "ytd", "3y"], ["5y", "10y"]),
            (1400, 15, ["1d", "ytd", "5y"], ["10y", "max"]),
            (2800, 16, ["1d", "ytd", "10y"], ["max"]),
            (5500, 17, ["1d", "ytd", "max"], []),
        ],
    )
    def test_get_valid_periods(
        self, max_trading_days, expected_count, must_include, must_exclude
    ) -> None:
        from yfinance_mcp.helpers import get_valid_periods

        periods = get_valid_periods(max_trading_days)

        assert len(periods) == expected_count

        for p in must_include:
            assert p in periods, (
                f"Expected {p} in {periods} for max_trading_days={max_trading_days}"
            )

        for p in must_exclude:
            assert p not in periods, (
                f"Expected {p} NOT in {periods} for max_trading_days={max_trading_days}"
            )

    def test_get_valid_periods_sorted_by_duration(self) -> None:
        from yfinance_mcp.helpers import PERIOD_TO_DAYS, get_valid_periods

        periods = get_valid_periods(800)  # 800 trading days ≈ 3y
        durations = [PERIOD_TO_DAYS.get(p, 0) for p in periods]
        assert durations == sorted(durations), "Periods should be sorted by duration"

    def test_get_valid_periods_with_max_period_days(self) -> None:
        """Verify tool schema shows correct periods for current MAX_PERIOD_DAYS."""
        from yfinance_mcp.helpers import MAX_PERIOD_DAYS, get_valid_periods

        periods = get_valid_periods(MAX_PERIOD_DAYS)

        # MAX_PERIOD_DAYS = 1000 trading days (TARGET_POINTS=200 × 5)
        assert "3y" in periods  # 1095 calendar days = 782 trading days < 1000
        assert "5y" not in periods  # 1825 calendar days = 1304 trading days > 1000
        assert "ytd" in periods
        assert len(periods) == 14


class TestOHLCResample:
    """Test OHLC resampling for history tool."""

    @pytest.mark.parametrize(
        "n_rows,target,expected_min,expected_max",
        [
            (0, 10, 0, 0),  # empty
            (1, 10, 1, 1),  # single row unchanged
            (1, 1, 1, 1),  # single row, target=1
            (2, 10, 2, 2),  # two rows unchanged
            (100, 10, 5, 10),  # reduces to ~target
            (100, 5, 3, 5),  # reduces to ~target
            (100, 1, 1, 1),  # extreme reduction
            (100, 200, 100, 100),  # target > input
        ],
    )
    def test_output_size(self, n_rows, target, expected_min, expected_max) -> None:
        from yfinance_mcp.helpers import ohlc_resample

        if n_rows == 0:
            df = pd.DataFrame(columns=["o", "h", "l", "c", "v"])
        else:
            df = pd.DataFrame(
                {
                    "o": [100] * n_rows,
                    "h": [101] * n_rows,
                    "l": [99] * n_rows,
                    "c": [100] * n_rows,
                    "v": [1000] * n_rows,
                },
                index=pd.date_range("2024-01-01", periods=n_rows, freq="D"),
            )
        result = ohlc_resample(df, target_points=target)
        assert expected_min <= len(result) <= expected_max

    def test_aggregation_preserves_extremes(self) -> None:
        """High=max, Low=min within buckets."""
        from yfinance_mcp.helpers import ohlc_resample

        df = pd.DataFrame(
            {
                "o": list(range(100, 200)),
                "h": list(range(200, 300)),
                "l": list(range(50, 150)),
                "c": list(range(150, 250)),
                "v": [1000] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="D"),
        )
        result = ohlc_resample(df, target_points=5)
        assert result["h"].iloc[0] > 200, "High aggregates to max"
        assert result["l"].iloc[0] < 100, "Low aggregates to min"

    def test_handles_nan_values(self) -> None:
        """NaN values should be handled gracefully."""
        from yfinance_mcp.helpers import ohlc_resample

        df = pd.DataFrame(
            {
                "o": [100, np.nan, 102],
                "h": [101, np.nan, 103],
                "l": [99, np.nan, 101],
                "c": [100, np.nan, 102],
                "v": [1000, np.nan, 1200],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        result = ohlc_resample(df, target_points=10)
        assert len(result) <= 3


class TestLTTBDownsample:
    """Test LTTB downsampling for technicals tool."""

    @pytest.mark.parametrize(
        "n_rows,target,expected_len",
        [
            (0, 10, 0),  # empty
            (1, 10, 1),  # single row
            (2, 10, 2),  # two rows
            (3, 10, 3),  # small input unchanged
            (10, 5, 5),  # reduces to target
            (10, 3, 3),  # minimum for LTTB buckets
            (10, 2, 2),  # first + last only
            (10, 1, 1),  # just last point
            (5, 100, 5),  # target > input
        ],
    )
    def test_output_size(self, n_rows, target, expected_len) -> None:
        from yfinance_mcp.helpers import lttb_downsample

        df = pd.DataFrame(
            {"rsi": list(range(n_rows))},
            index=pd.date_range("2024-01-01", periods=max(1, n_rows))[:n_rows],
        )
        result = lttb_downsample(df, target_points=target)
        assert len(result) == expected_len

    def test_preserves_first_and_last(self) -> None:
        from yfinance_mcp.helpers import lttb_downsample

        values = [10, 11, 50, 12, 10, 9, 5, 10, 11, 12]
        df = pd.DataFrame({"rsi": values}, index=pd.date_range("2024-01-01", periods=10))
        result = lttb_downsample(df, target_points=5)
        assert result["rsi"].iloc[0] == 10
        assert result["rsi"].iloc[-1] == 12

    def test_preserves_all_columns(self) -> None:
        from yfinance_mcp.helpers import lttb_downsample

        df = pd.DataFrame(
            {"rsi": [10, 50, 20, 80, 30], "macd": [1, 2, 3, 4, 5]},
            index=pd.date_range("2024-01-01", periods=5),
        )
        result = lttb_downsample(df, target_points=3)
        assert set(result.columns) == {"rsi", "macd"}

    def test_handles_nan_values(self) -> None:
        """NaN in data should not crash."""
        from yfinance_mcp.helpers import lttb_downsample

        df = pd.DataFrame(
            {"rsi": [10, np.nan, 30, 40, 50]},
            index=pd.date_range("2024-01-01", periods=5),
        )
        result = lttb_downsample(df, target_points=3)
        assert len(result) == 3
