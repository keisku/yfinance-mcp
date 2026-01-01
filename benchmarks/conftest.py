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
    """Pre-populate cache with real market data from fixture files.

    Uses pre-fetched real data committed to the repo. This provides:
    - Real, accurate market data (fetched from Yahoo Finance)
    - Network-independent benchmarks (no API calls during tests)
    - Deterministic, repeatable results
    - Fast benchmark execution

    To update fixture data, run: python benchmarks/fetch_fixture_data.py
    """
    import json

    from yfinance_mcp.cache import CachedPriceFetcher

    fetcher = CachedPriceFetcher()

    fixtures_dir = Path(__file__).parent / "fixtures"

    # Load fixture data from JSON files
    for symbol in benchmark_symbols:
        for interval in ["1d", "1wk", "1mo"]:
            fixture_file = fixtures_dir / f"{symbol}_{interval}.json"

            if not fixture_file.exists():
                # Generate synthetic data as fallback
                print(f"Warning: Fixture file {fixture_file} not found, using synthetic data")
                df = generate_ohlcv_data(
                    symbol,
                    datetime(2023, 1, 1),
                    num_days=500,
                    base_price=150.0,
                )
            else:
                # Load real data from fixture file
                with open(fixture_file) as f:
                    records = json.load(f)

                # Convert to DataFrame
                data = []
                dates = []
                for record in records:
                    dates.append(pd.Timestamp(record["date"]))
                    data.append(
                        {
                            "o": record["o"],
                            "h": record["h"],
                            "l": record["l"],
                            "c": record["c"],
                            "v": record["v"],
                        }
                    )

                df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
                df.index.name = "Date"

            # Store in cache
            fetcher.cache.store_prices(symbol, df, interval=interval)

    yield fetcher

    # Cleanup
    fetcher.clear()
    fetcher.close()

    import shutil

    try:
        shutil.rmtree(_benchmark_dir)
    except OSError:
        pass
