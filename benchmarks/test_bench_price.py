"""Benchmark tests for price data fetching.

Run with pytest-benchmark:
    uv run pytest benchmarks/test_bench_price.py --benchmark-only
    uv run pytest benchmarks/test_bench_price.py --benchmark-json=benchmark-results.json

For CI integration with github-action-benchmark:
    https://github.com/benchmark-action/github-action-benchmark/tree/master/examples/pytest
"""

import os
import tempfile

import pytest
import yfinance as yf

# Use isolated database to avoid DuckDB lock conflicts with running MCP server
_benchmark_dir = tempfile.mkdtemp(prefix="yfinance-mcp-bench-")
_benchmark_db_path = os.path.join(_benchmark_dir, "benchmark.duckdb")
os.environ["YFINANCE_CACHE_DB"] = _benchmark_db_path

# Ensure cache is ENABLED for benchmarks (opposite of tests)
os.environ.pop("YFINANCE_CACHE_DISABLED", None)

from yahoo_finance_mcp import prices  # noqa: E402 - must import after env setup

# Global stocks across different exchanges
SYMBOLS = [
    "AAPL",  # NYSE (US)
    "7203.T",  # Toyota - Tokyo
    "SAP.DE",  # SAP - Germany/XETRA
    "MC.PA",  # LVMH - Paris
    "HSBA.L",  # HSBC - London
    "0700.HK",  # Tencent - Hong Kong
    "RY.TO",  # Royal Bank - Toronto
]

PERIODS = ["5d", "1mo", "3mo", "1y"]


@pytest.fixture(scope="module")
def _primed_cache():
    """Prime cache with data for all symbols and intervals."""
    prices.clear()
    # Prime the cache with 2y data for all intervals
    for symbol in SYMBOLS:
        prices.get_history(symbol, "2y", "1d")
        prices.get_history(symbol, "2y", "1wk")
        prices.get_history(symbol, "2y", "1mo")
    yield
    prices.clear()
    # Cleanup temp database directory
    import shutil

    try:
        shutil.rmtree(_benchmark_dir)
    except OSError:
        pass


class TestCacheHit:
    """Benchmark cache hit performance via public prices.get_history() API."""

    def test_cache_hit_us(self, benchmark, _primed_cache):
        """Cache hit for US stock (AAPL)."""
        benchmark(prices.get_history, "AAPL", "1y", "1d")

    def test_cache_hit_japan(self, benchmark, _primed_cache):
        """Cache hit for Japan stock (Toyota)."""
        benchmark(prices.get_history, "7203.T", "1y", "1d")

    def test_cache_hit_europe(self, benchmark, _primed_cache):
        """Cache hit for European stock (SAP)."""
        benchmark(prices.get_history, "SAP.DE", "1y", "1d")

    def test_cache_hit_varied_periods(self, benchmark, _primed_cache):
        """Cache hit with varied periods."""

        def fetch_all_periods():
            for period in PERIODS:
                prices.get_history("AAPL", period, "1d")

        benchmark(fetch_all_periods)


class TestPortfolio:
    """Benchmark portfolio-style queries."""

    def test_portfolio_scan(self, benchmark, _primed_cache):
        """Scan all global stocks with 1y period."""

        def scan_portfolio():
            for symbol in SYMBOLS:
                prices.get_history(symbol, "1y", "1d")

        benchmark(scan_portfolio)


class TestWeeklyMonthly:
    """Benchmark weekly/monthly cache hit performance (directly cached from API)."""

    def test_weekly_cache_hit(self, benchmark, _primed_cache):
        """Weekly bars from DuckDB cache."""
        benchmark(prices.get_history, "AAPL", "1y", "1wk")

    def test_monthly_cache_hit(self, benchmark, _primed_cache):
        """Monthly bars from DuckDB cache."""
        benchmark(prices.get_history, "AAPL", "1y", "1mo")

    def test_weekly_portfolio_scan(self, benchmark, _primed_cache):
        """Weekly data for all stocks from cache."""

        def scan_weekly():
            for symbol in SYMBOLS:
                prices.get_history(symbol, "1y", "1wk")

        benchmark(scan_weekly)


class TestIntradayCache:
    """Benchmark intraday TTL cache performance.

    Note: First call hits API, subsequent calls hit cache.
    """

    def test_intraday_cache_hit(self, benchmark, _primed_cache):
        """Intraday cache hit after priming."""
        # Prime the intraday cache first
        prices.get_history("AAPL", "5d", "15m")

        # Benchmark cache hits
        benchmark(prices.get_history, "AAPL", "5d", "15m")


class TestBaseline:
    """Baseline comparison - direct yfinance calls.

    Note: These are slow (network calls) and may be skipped in CI.
    Run with: pytest benchmarks/ --benchmark-only -k "not baseline"
    """

    @pytest.mark.slow
    def test_baseline_single(self, benchmark):
        """Direct yfinance call without cache."""

        def direct_fetch():
            t = yf.Ticker("AAPL")
            return t.history(period="1mo")

        benchmark(direct_fetch)
