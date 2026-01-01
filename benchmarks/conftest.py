"""Pytest configuration and fixtures for benchmarks."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Use isolated database to avoid DuckDB lock conflicts with running MCP server
_benchmark_dir = tempfile.mkdtemp(prefix="yfinance-mcp-bench-")
_benchmark_db_path = os.path.join(_benchmark_dir, "benchmark.duckdb")
os.environ["YFINANCE_CACHE_DB"] = _benchmark_db_path

# Ensure cache is ENABLED for benchmarks (opposite of tests)
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
    # Use symbol as seed for reproducibility
    np.random.seed(hash(symbol) % (2**32))

    dates = pd.date_range(start=start_date, periods=num_days, freq="D")
    # Filter to weekdays only (trading days)
    dates = dates[dates.dayofweek < 5]

    # Generate price series with random walk
    returns = np.random.normal(0, volatility, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Intraday variation (±1%)
        daily_range = close * 0.01
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = np.random.uniform(low, high)

        # Volume with realistic variation
        base_volume = 10_000_000
        volume = int(base_volume * np.random.uniform(0.5, 2.0))

        data.append(
            {
                "o": open_price,
                "h": high,
                "l": low,
                "c": close,
                "v": volume,
            }
        )

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture(scope="session")
def benchmark_symbols():
    """Symbols used across benchmarks - synthetic test symbols with diverse formats.

    These are FAKE symbols to avoid confusion with real market data.
    They represent different exchange formats for comprehensive testing.
    """
    return [
        "TEST.US",  # US-style symbol
        "1234.T",  # Tokyo-style symbol
        "BENCH.DE",  # Germany/XETRA-style
        "MOCK.PA",  # Paris-style symbol
        "FAKE.L",  # London-style symbol
        "9999.HK",  # Hong Kong-style
        "DEMO.TO",  # Toronto-style symbol
    ]


@pytest.fixture(scope="session")
def seeded_cache(benchmark_symbols):
    """Pre-populate cache with deterministic synthetic data.

    Generates fake market data on-the-fly for benchmark testing. This provides:
    - Deterministic, repeatable results (using fixed random seeds)
    - Network-independent benchmarks (no API calls)
    - Fast benchmark execution (no file I/O)
    - Clean repo (no large fixture files committed)

    Data is generated using realistic OHLCV patterns with proper volatility.
    """
    from yfinance_mcp.cache import CachedPriceFetcher

    fetcher = CachedPriceFetcher()

    # Symbol parameters for realistic price generation
    symbol_params = {
        "TEST.US": {"base_price": 150.0, "volatility": 0.018},
        "1234.T": {"base_price": 2000.0, "volatility": 0.015},
        "BENCH.DE": {"base_price": 120.0, "volatility": 0.017},
        "MOCK.PA": {"base_price": 700.0, "volatility": 0.019},
        "FAKE.L": {"base_price": 6.0, "volatility": 0.016},
        "9999.HK": {"base_price": 350.0, "volatility": 0.022},
        "DEMO.TO": {"base_price": 100.0, "volatility": 0.014},
    }

    # Generate 2 years of data for each symbol and interval
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2023, 1, 1)

    for symbol in benchmark_symbols:
        params = symbol_params.get(symbol, {"base_price": 100.0, "volatility": 0.02})

        # Generate daily data
        daily_df = generate_ohlcv_data(
            symbol,
            start_date,
            num_days=730,  # ~2 years
            base_price=params["base_price"],
            volatility=params["volatility"],
        )
        fetcher.cache.store_prices(symbol, daily_df, interval="1d")

        # Generate weekly data (resample from daily)
        weekly_df = daily_df.resample("W").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
        )
        fetcher.cache.store_prices(symbol, weekly_df, interval="1wk")

        # Generate monthly data (resample from daily)
        monthly_df = daily_df.resample("ME").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
        )
        fetcher.cache.store_prices(symbol, monthly_df, interval="1mo")

    yield fetcher

    # Cleanup
    fetcher.clear()
    fetcher.close()

    import shutil

    try:
        shutil.rmtree(_benchmark_dir)
    except OSError:
        pass
