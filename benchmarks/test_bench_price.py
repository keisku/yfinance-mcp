"""Benchmark tests for price data fetching.

These benchmarks are designed to be deterministic and network-independent.
They use pre-seeded fixture data instead of making API calls.

Run with pytest-benchmark:
    uv run pytest benchmarks/ --benchmark-only
    uv run pytest benchmarks/ --benchmark-json=benchmark-results.json

For CI integration with github-action-benchmark:
    https://github.com/benchmark-action/github-action-benchmark/tree/master/examples/pytest
"""


from yfinance_mcp import prices


class TestCacheHitDaily:
    """Benchmark daily data cache hit performance."""

    def test_cache_hit_us_symbol(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hit for US-style symbol - daily data."""
        benchmark(prices.get_history, "TEST.US", "1y", "1d")

    def test_cache_hit_tokyo_symbol(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hit for Tokyo-style symbol - daily data."""
        benchmark(prices.get_history, "1234.T", "1y", "1d")

    def test_cache_hit_europe_symbol(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hit for European-style symbol - daily data."""
        benchmark(prices.get_history, "BENCH.DE", "1y", "1d")

    def test_cache_hit_varied_periods(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hits with varied periods (5d, 1mo, 3mo, 1y)."""

        def fetch_all_periods():
            for period in ["5d", "1mo", "3mo", "1y"]:
                prices.get_history("TEST.US", period, "1d")

        benchmark(fetch_all_periods)

    def test_cache_hit_short_period(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hit for short period (5 days)."""
        benchmark(prices.get_history, "TEST.US", "5d", "1d")

    def test_cache_hit_long_period(self, benchmark, seeded_cache, benchmark_symbols):
        """Cache hit for long period (5 years)."""
        benchmark(prices.get_history, "TEST.US", "5y", "1d")


class TestCacheHitWeeklyMonthly:
    """Benchmark weekly/monthly cache hit performance."""

    def test_cache_hit_weekly(self, benchmark, seeded_cache, benchmark_symbols):
        """Weekly bars from cache."""
        benchmark(prices.get_history, "TEST.US", "1y", "1wk")

    def test_cache_hit_monthly(self, benchmark, seeded_cache, benchmark_symbols):
        """Monthly bars from cache."""
        benchmark(prices.get_history, "TEST.US", "1y", "1mo")

    def test_cache_hit_weekly_long_period(self, benchmark, seeded_cache, benchmark_symbols):
        """Weekly data for 5 years."""
        benchmark(prices.get_history, "TEST.US", "5y", "1wk")

    def test_cache_hit_monthly_long_period(self, benchmark, seeded_cache, benchmark_symbols):
        """Monthly data for 5 years."""
        benchmark(prices.get_history, "TEST.US", "5y", "1mo")


class TestPortfolioScans:
    """Benchmark portfolio-style queries (multiple symbols)."""

    def test_portfolio_scan_daily(self, benchmark, seeded_cache, benchmark_symbols):
        """Scan all global stocks with 1y daily data."""

        def scan_portfolio():
            for symbol in benchmark_symbols:
                prices.get_history(symbol, "1y", "1d")

        benchmark(scan_portfolio)

    def test_portfolio_scan_weekly(self, benchmark, seeded_cache, benchmark_symbols):
        """Scan all global stocks with 1y weekly data."""

        def scan_weekly():
            for symbol in benchmark_symbols:
                prices.get_history(symbol, "1y", "1wk")

        benchmark(scan_weekly)

    def test_portfolio_scan_short_period(self, benchmark, seeded_cache, benchmark_symbols):
        """Quick portfolio scan (5 days)."""

        def scan_short():
            for symbol in benchmark_symbols:
                prices.get_history(symbol, "5d", "1d")

        benchmark(scan_short)

    def test_portfolio_scan_long_period(self, benchmark, seeded_cache, benchmark_symbols):
        """Deep portfolio analysis (5 years)."""

        def scan_long():
            for symbol in benchmark_symbols:
                prices.get_history(symbol, "5y", "1d")

        benchmark(scan_long)


class TestDateRangeQueries:
    """Benchmark date-range based queries."""

    def test_date_range_1month(self, benchmark, seeded_cache, benchmark_symbols):
        """Date range query for 1 month."""
        benchmark(
            prices.get_history,
            "TEST.US",
            interval="1d",
            start="2024-11-01",
            end="2024-12-01",
        )

    def test_date_range_1year(self, benchmark, seeded_cache, benchmark_symbols):
        """Date range query for 1 year."""
        benchmark(
            prices.get_history,
            "TEST.US",
            interval="1d",
            start="2024-01-01",
            end="2024-12-31",
        )

    def test_date_range_multi_year(self, benchmark, seeded_cache, benchmark_symbols):
        """Date range query for 2 years."""
        benchmark(
            prices.get_history,
            "TEST.US",
            interval="1d",
            start="2023-01-01",
            end="2024-12-31",
        )


class TestCacheOperations:
    """Benchmark cache-specific operations."""

    def test_cache_stats_retrieval(self, benchmark, seeded_cache):
        """Benchmark retrieving cache statistics."""
        from yfinance_mcp.cache import get_cache_stats

        # Warm up
        prices.get_history("TEST.US", "1y", "1d")

        benchmark(get_cache_stats)

    def test_repeated_same_query(self, benchmark, seeded_cache):
        """Repeated identical queries (best-case cache scenario)."""

        def repeated_query():
            for _ in range(10):
                prices.get_history("TEST.US", "1mo", "1d")

        benchmark(repeated_query)


class TestConcurrentAccess:
    """Benchmark concurrent-like access patterns."""

    def test_interleaved_symbols(self, benchmark, seeded_cache, benchmark_symbols):
        """Interleaved queries across different symbols."""

        def interleaved_access():
            periods = ["5d", "1mo", "3mo"]
            for period in periods:
                for symbol in benchmark_symbols:
                    prices.get_history(symbol, period, "1d")

        benchmark(interleaved_access)

    def test_mixed_intervals(self, benchmark, seeded_cache):
        """Mixed intervals for same symbol."""

        def mixed_intervals():
            for interval in ["1d", "1wk", "1mo"]:
                prices.get_history("TEST.US", "1y", interval)

        benchmark(mixed_intervals)


class TestRealWorldPatterns:
    """Benchmark real-world usage patterns."""

    def test_typical_analysis_workflow(self, benchmark, seeded_cache):
        """Typical analysis: compare multiple periods for decision making."""

        def analysis_workflow():
            # Short-term trend
            prices.get_history("TEST.US", "5d", "1d")
            # Medium-term trend
            prices.get_history("TEST.US", "3mo", "1d")
            # Long-term trend
            prices.get_history("TEST.US", "1y", "1d")
            # Historical comparison
            prices.get_history("TEST.US", "5y", "1wk")

        benchmark(analysis_workflow)

    def test_multi_stock_comparison(self, benchmark, seeded_cache):
        """Compare same period across multiple test symbols (global formats)."""

        def stock_comparison():
            symbols = ["TEST.US", "1234.T", "BENCH.DE"]
            for symbol in symbols:
                prices.get_history(symbol, "1y", "1d")

        benchmark(stock_comparison)

    def test_dashboard_load(self, benchmark, seeded_cache, benchmark_symbols):
        """Simulate dashboard loading multiple widgets (diverse symbol formats)."""

        def dashboard_load():
            # Widget 1: Recent price movement
            prices.get_history("TEST.US", "5d", "1d")
            # Widget 2: Monthly trend
            prices.get_history("TEST.US", "1mo", "1d")
            # Widget 3: Portfolio overview (diverse formats)
            for symbol in ["TEST.US", "1234.T", "BENCH.DE"]:
                prices.get_history(symbol, "1mo", "1d")

        benchmark(dashboard_load)
