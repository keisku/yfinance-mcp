"""Pytest configuration and fixtures for benchmarks."""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

_benchmark_dir = tempfile.mkdtemp(prefix="yfinance-mcp-bench-")
_benchmark_db_path = os.path.join(_benchmark_dir, "benchmark.duckdb")
os.environ["YFINANCE_CACHE_DB"] = _benchmark_db_path

os.environ.pop("YFINANCE_CACHE_DISABLED", None)


def generate_ohlcv_data(
    symbol: str,
    start_date: datetime,
    num_days: int,
    base_price: float = 100.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate realistic OHLCV data for benchmarking.

    Args:
        symbol: Stock symbol (for reproducible random seed)
        start_date: Starting date for the data
        num_days: Number of trading days to generate
        base_price: Starting price
        volatility: Daily volatility (default 2%)

    Returns:
        DataFrame with OHLC data and DatetimeIndex
    """
    np.random.seed(hash(symbol) % (2**32))

    dates = pd.date_range(start=start_date, periods=num_days, freq="B")

    returns = np.random.normal(0, volatility, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    data = []
    for date, close in zip(dates, prices):
        daily_range = close * 0.01
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = np.random.uniform(low, high)

        base_volume = 10_000_000
        volume = int(base_volume * np.random.uniform(0.5, 2.0))

        data.append(
            {
                "o": open_price,
                "h": high,
                "l": low,
                "c": close,
                "ac": close * 0.98,  # simulate dividend adjustment
                "v": volume,
            }
        )

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture(scope="session")
def benchmark_symbols():
    """Fake symbols with diverse exchange formats for comprehensive testing."""
    return [
        "TEST.US",
        "1234.T",
        "BENCH.DE",
        "MOCK.PA",
        "FAKE.L",
        "9999.HK",
        "DEMO.TO",
    ]


@pytest.fixture(scope="session")
def seeded_cache(benchmark_symbols):
    """Pre-populate cache with deterministic synthetic data."""
    from yfinance_mcp.cache import CachedPriceFetcher

    fetcher = CachedPriceFetcher()

    symbol_params = {
        "TEST.US": {"base_price": 150.0, "volatility": 0.018},
        "1234.T": {"base_price": 2000.0, "volatility": 0.015},
        "BENCH.DE": {"base_price": 120.0, "volatility": 0.017},
        "MOCK.PA": {"base_price": 700.0, "volatility": 0.019},
        "FAKE.L": {"base_price": 6.0, "volatility": 0.016},
        "9999.HK": {"base_price": 350.0, "volatility": 0.022},
        "DEMO.TO": {"base_price": 100.0, "volatility": 0.014},
    }

    start_date = datetime(2023, 1, 1)

    for symbol in benchmark_symbols:
        params = symbol_params.get(symbol, {"base_price": 100.0, "volatility": 0.02})

        daily_df = generate_ohlcv_data(
            symbol,
            start_date,
            num_days=730,
            base_price=params["base_price"],
            volatility=params["volatility"],
        )
        fetcher.cache.store_prices(symbol, daily_df, interval="1d")

        weekly_df = daily_df.resample("W").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "ac": "last", "v": "sum"}
        )
        fetcher.cache.store_prices(symbol, weekly_df, interval="1wk")

        monthly_df = daily_df.resample("ME").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "ac": "last", "v": "sum"}
        )
        fetcher.cache.store_prices(symbol, monthly_df, interval="1mo")

    yield fetcher

    fetcher.clear()
    fetcher.close()

    import shutil

    try:
        shutil.rmtree(_benchmark_dir)
    except OSError:
        pass
