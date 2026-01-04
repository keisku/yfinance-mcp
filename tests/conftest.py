"""Pytest configuration for Yahoo Finance MCP tests."""

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

os.environ["YFINANCE_CACHE_DISABLED"] = "1"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "crosscheck: cross-validation tests against pandas-ta (may be slow)",
    )


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset circuit breaker state before each test to prevent test pollution."""
    from yfinance_mcp.server import reset_circuit_breaker_for_testing

    reset_circuit_breaker_for_testing()
    yield
    reset_circuit_breaker_for_testing()


# Default valuation info for mock tickers
DEFAULT_VALUATION_INFO = {
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


@pytest.fixture
def mock_ticker_factory():
    """Factory fixture for creating mock tickers with customizable properties."""

    def _create(price: float = 150.0, **info_overrides) -> MagicMock:
        mock = MagicMock()
        mock.fast_info.last_price = price
        mock.fast_info.previous_close = price * 0.99
        mock.fast_info.market_cap = price * 1e6
        mock.fast_info.day_high = price * 1.01
        mock.fast_info.day_low = price * 0.99
        mock.fast_info.last_volume = 1000000
        info = {
            "regularMarketPrice": price,
            "shortName": "Test Stock",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchange": "NMS",
            "currency": "USD",
            **DEFAULT_VALUATION_INFO,
        }
        info.update(info_overrides)
        mock.info = info
        return mock

    return _create


@pytest.fixture
def mock_ohlcv_factory():
    """Factory fixture for creating OHLCV DataFrames."""

    def _create(n: int = 50, seed: int = 42, freq: str = "D") -> pd.DataFrame:
        np.random.seed(seed)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": [1000000] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq=freq),
        )

    return _create


@pytest.fixture
def mock_ticker_with_history(mock_ticker_factory, mock_ohlcv_factory):
    """Factory for mock ticker with history data attached."""

    def _create(n: int = 50, price: float = 150.0, **info_overrides) -> MagicMock:
        mock = mock_ticker_factory(price=price, **info_overrides)
        mock.history.return_value = mock_ohlcv_factory(n=n)
        return mock

    return _create


@pytest.fixture
def mock_financials_factory():
    """Factory fixture for creating mock financials."""

    def _create() -> MagicMock:
        mock = MagicMock()
        for stmt, data in [
            ("get_income_stmt", {"TotalRevenue": [100000, 90000]}),
            ("get_balance_sheet", {"TotalAssets": [500000, 450000]}),
            ("get_cashflow", {"OperatingCashFlow": [25000, 22000]}),
        ]:
            df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=2, freq="YE")).T
            getattr(mock, stmt).return_value = df
        return mock

    return _create
