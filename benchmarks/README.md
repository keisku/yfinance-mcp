# Benchmarks

This directory contains performance benchmarks for the yfinance-mcp server.

## Philosophy

These benchmarks are designed to be:

- **Deterministic**: Same results every run, no flakiness
- **Network-independent**: No API calls during benchmark execution
- **Fast**: Quick setup and execution for rapid iteration
- **Representative**: Use real market data patterns (or real data when available)

## Architecture

The benchmarks use a **synthetic data generation approach**:

1. **Data generation**: Fake market data is generated on-the-fly using deterministic random seeds
2. **Cache seeding**: Before benchmarks run, synthetic data is loaded into DuckDB cache
3. **Benchmark execution**: Tests measure cache hit performance, not network performance

This eliminates the flakiness caused by:
- Network timeouts and failures
- API rate limiting
- Variable network latency
- Market hours and holidays
- Retry logic timing
- Large fixture files in the repository

## Running Benchmarks

```bash
# Run all benchmarks
uv run pytest benchmarks/ --benchmark-only

# Run specific benchmark class
uv run pytest benchmarks/ --benchmark-only -k TestCacheHitDaily

# Save results to JSON for CI tracking
uv run pytest benchmarks/ --benchmark-only --benchmark-json=results.json

# Compare with previous results
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=0001
```

## Data Generation

Benchmarks use **synthetic data generated on-the-fly** with deterministic random seeds. This approach:

- **Deterministic**: Same data every run (using symbol name as random seed)
- **Realistic**: Follows proper OHLCV patterns with volatility
- **Fast**: No file I/O or network calls
- **Clean**: No large fixture files in the repository

### Using Real Market Data (Optional)

If you want to benchmark with actual historical market data instead of synthetic data:

```bash
python benchmarks/fetch_fixture_data.py
```

This script:
- Fetches real data from Yahoo Finance for real ticker symbols
- Saves data as JSON files in `benchmarks/fixtures/`
- Requires network access
- Updates `conftest.py` to load from fixture files instead of generating

**Note**: By default, benchmarks use synthetic data with fake symbols (TEST.US, 1234.T, etc.). Only use real data if you specifically need to benchmark against actual market patterns.

## Benchmark Categories

### TestCacheHitDaily
Measures cache performance for daily data queries:
- Single symbol lookups
- Various time periods (5d, 1mo, 3mo, 1y, 5y)
- Different stocks

### TestCacheHitWeeklyMonthly
Measures cache performance for aggregated intervals:
- Weekly bars
- Monthly bars
- Long-period queries

### TestPortfolioScans
Simulates portfolio analysis workflows:
- Multi-symbol queries
- Different time periods
- Cross-sectional analysis

### TestDateRangeQueries
Measures performance of date-range based queries:
- Specific start/end dates
- Various period lengths
- Subset queries within cached data

### TestCacheOperations
Benchmarks cache-specific functionality:
- Cache stats retrieval
- Repeated identical queries
- Cache hit optimization

### TestConcurrentAccess
Simulates concurrent-like access patterns:
- Interleaved symbol queries
- Mixed interval queries
- Cache thrashing scenarios

### TestRealWorldPatterns
Benchmarks realistic usage scenarios:
- Typical analysis workflows
- Multi-stock comparisons
- Dashboard loading patterns

## CI Integration

For tracking benchmark performance over time, use [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark):

```yaml
- name: Run benchmarks
  run: uv run pytest benchmarks/ --benchmark-json=output.json

- name: Store benchmark result
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: output.json
    github-token: ${{ secrets.GITHUB_TOKEN }}
    auto-push: true
```

## Troubleshooting

### Benchmarks are slow
- Check that `YFINANCE_CACHE_DB` is set to a fast storage location
- Verify synthetic data generation completes successfully
- Ensure DuckDB has write permissions to temp directory

### Inconsistent results
- Check that no other process is using the benchmark database
- Ensure system load is consistent between runs
- Verify numpy/pandas versions are consistent (affects random seed behavior)
