# Benchmarks

Performance benchmarks for the yfinance-mcp server.

## Philosophy

These benchmarks are designed to be:

- **Deterministic**: Same results every run, no flakiness
- **Network-independent**: No API calls during benchmark execution
- **Fast**: Quick setup and execution for rapid iteration
- **Representative**: Realistic market data patterns for meaningful results

## How It Works

Benchmarks measure **cache performance**, not network performance.

The process:
1. Generate synthetic market data with realistic patterns
2. Seed the cache with this data
3. Run benchmarks against the cached data
4. Measure retrieval performance

This approach eliminates flakiness from network issues, API rate limits, and variable latency while still providing meaningful performance metrics.

## Running Benchmarks

```bash
# Run all benchmarks
uv run pytest benchmarks/ --benchmark-only

# Save results for CI tracking
uv run pytest benchmarks/ --benchmark-only --benchmark-json=results.json

# Compare with previous results
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=0001
```

## What's Being Measured

- **Cache hit performance**: How fast can we retrieve data from DuckDB?
- **Portfolio queries**: Multi-symbol lookups
- **Date range queries**: Specific time period requests
- **Real-world patterns**: Typical analysis workflows

All benchmarks use fake symbols (TEST.US, 1234.T, etc.) with diverse exchange formats to test symbol parsing across different markets.

## For CI Integration

Example with [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark):

```yaml
- name: Run benchmarks
  run: uv run pytest benchmarks/ --benchmark-json=output.json

- name: Store benchmark result
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: output.json
```
