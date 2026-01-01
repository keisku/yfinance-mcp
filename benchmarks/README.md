# Benchmarks

This directory contains performance benchmarks for the yfinance-mcp server.

## Philosophy

These benchmarks are designed to be:

- **Deterministic**: Same results every run, no flakiness
- **Network-independent**: No API calls during benchmark execution
- **Fast**: Quick setup and execution for rapid iteration
- **Representative**: Use real market data patterns (or real data when available)

## Architecture

The benchmarks use a **fixture-based approach**:

1. **Fixture data**: Pre-generated market data stored in `fixtures/` directory
2. **Cache seeding**: Before benchmarks run, data is loaded into DuckDB cache
3. **Benchmark execution**: Tests measure cache hit performance, not network performance

This eliminates the flakiness caused by:
- Network timeouts and failures
- API rate limiting
- Variable network latency
- Market hours and holidays
- Retry logic timing

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

## Fixture Data

Fixture data is stored in `benchmarks/fixtures/` as JSON files with timestamped OHLCV records.

### Generating Fixture Data

**Option 1: Deterministic synthetic data (recommended for consistency)**
```bash
python benchmarks/generate_fixtures.py
```

This creates realistic market data using mathematical models. Data is:
- Deterministic (same output every time)
- Realistic (follows market patterns)
- Fast to generate (no network calls)

**Option 2: Real Yahoo Finance data (requires network access)**
```bash
python benchmarks/fetch_fixture_data.py
```

This fetches actual market data from Yahoo Finance. Use when:
- You want to benchmark with real historical data
- Network access is available and reliable
- You're willing to accept some variation in fixture data updates

### Fixture Format

Each fixture file is a JSON array of OHLCV records:

```json
[
  {
    "date": "2023-01-03T00:00:00",
    "o": 150.25,
    "h": 152.30,
    "l": 149.80,
    "c": 151.75,
    "v": 45678900
  },
  ...
]
```

- `date`: ISO 8601 timestamp
- `o`: Open price
- `h`: High price
- `l`: Low price
- `c`: Close price
- `v`: Volume

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

### "Fixture file not found" warnings
Run `python benchmarks/generate_fixtures.py` to create fixture data.

### Benchmarks are slow
- Check that `YFINANCE_CACHE_DB` is set to a fast storage location
- Verify fixture data is properly cached (check setup output)
- Ensure DuckDB has write permissions to temp directory

### Inconsistent results
- Verify fixture data hasn't changed
- Check that no other process is using the benchmark database
- Ensure system load is consistent between runs
