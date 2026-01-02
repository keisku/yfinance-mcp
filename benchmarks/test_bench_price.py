"""Benchmark tests for price data fetching.

These benchmarks are deterministic and network-independent.
They use pre-seeded fixture data instead of making API calls.

Run with pytest-benchmark:
    uv run pytest benchmarks/ -m benchmark --benchmark-only
    uv run pytest benchmarks/ -m benchmark --benchmark-json=benchmark-results.json

For CI integration with github-action-benchmark:
    https://github.com/benchmark-action/github-action-benchmark/tree/master/examples/pytest
"""

import pytest

from yfinance_mcp import history


@pytest.mark.benchmark
class TestCacheHitDaily:
    """Benchmark daily data cache hit performance."""

    def test_cache_hit_us_symbol(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hit for US-style symbol - daily data."""
        benchmark(history.get_history, "TEST.US", "1y", "1d")

    def test_cache_hit_tokyo_symbol(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hit for Tokyo-style symbol - daily data."""
        benchmark(history.get_history, "1234.T", "1y", "1d")

    def test_cache_hit_europe_symbol(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hit for European-style symbol - daily data."""
        benchmark(history.get_history, "BENCH.DE", "1y", "1d")

    def test_cache_hit_varied_periods(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hits with varied periods (5d, 1mo, 3mo, 1y)."""

        def fetch_all_periods():
            for period in ["5d", "1mo", "3mo", "1y"]:
                history.get_history("TEST.US", period, "1d")

        benchmark(fetch_all_periods)


@pytest.mark.benchmark
class TestCacheHitWeeklyMonthly:
    """Benchmark weekly/monthly cache hit performance."""

    def test_cache_hit_weekly(self, benchmark, seeded_cache, benchmark_symbols):
        """Weekly bars from cache."""
        benchmark(history.get_history, "TEST.US", "1y", "1wk")

    def test_cache_hit_monthly(self, benchmark, seeded_cache, benchmark_symbols):
        """Monthly bars from cache."""
        benchmark(history.get_history, "TEST.US", "1y", "1mo")


@pytest.mark.benchmark
class TestPortfolioScans:
    """Benchmark portfolio-style queries (multiple symbols)."""

    def test_portfolio_scan_daily(self, benchmark, seeded_cache, benchmark_symbols):
        """Scan all global stocks with 1y daily data."""

        def scan_portfolio():
            for symbol in benchmark_symbols:
                history.get_history(symbol, "1y", "1d")

        benchmark(scan_portfolio)

    def test_portfolio_scan_weekly(self, benchmark, seeded_cache, benchmark_symbols):
        """Scan all global stocks with 1y weekly data."""

        def scan_weekly():
            for symbol in benchmark_symbols:
                history.get_history(symbol, "1y", "1wk")

        benchmark(scan_weekly)
