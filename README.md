# Yahoo Finance MCP

MCP server providing real-time stock data, technicals, and fundamentals via Yahoo Finance.

## Usage

```json
{
  "mcpServers": {
    "yfinance": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "ghcr.io/keisku/yfinance-mcp:latest"]
    }
  }
}
```

## Tools

| Tool | Purpose |
|------|---------|
| `summary` | Stock overview with Quality Score, PEG, trend |
| `history` | OHLCV bars with date range support |
| `technicals` | Technical indicators (see below) |
| `fundamentals` | P/E, EPS, margins, dividends |
| `financials` | Income/balance/cashflow statements |
| `search` | Find ticker by company name |

### Technical Indicators

| Indicator | Description |
|-----------|-------------|
| `rsi` | Relative Strength Index (14-period) |
| `macd` | Moving Average Convergence Divergence |
| `sma_N` | Simple Moving Average (e.g., `sma_20`) |
| `ema_N` | Exponential Moving Average (e.g., `ema_12`) |
| `wma_N` | Weighted Moving Average (e.g., `wma_10`) |
| `momentum` | Price momentum (10-period) |
| `cci` | Commodity Channel Index |
| `dmi` | Directional Movement Index (+DI, -DI, ADX) |
| `williams` | Williams %R oscillator |
| `bb` | Bollinger Bands |
| `stoch` | Slow Stochastic |
| `fast_stoch` | Fast Stochastic |
| `ichimoku` | Ichimoku Cloud components |
| `atr` | Average True Range |
| `obv` | On-Balance Volume |
| `volume_profile` | Volume Profile (POC, Value Area) |
| `price_change` | Price change and percentage |
| `fibonacci` | Fibonacci retracement levels |
| `pivot` | Pivot points (Standard, Fibonacci, Camarilla, Woodie) |

## Development

```bash
uv sync                                # Install dependencies
uv run pytest                          # Run tests
uv run pytest benchmarks/ --benchmark-only  # Run benchmarks
uv run ruff check .                    # Lint
uv run pip-audit                       # Security scan
```

### Cache

DuckDB-based local cache to reduce Yahoo Finance API calls and speed up repeated queries.

- Location: `~/.cache/yfinance-mcp/market.duckdb`
- Disable: `YFINANCE_CACHE_DISABLED=1`
- Custom path: `YFINANCE_CACHE_DB=/path/to/cache.db`

### Yahoo Finance API Constraints

- Intraday data (1m-1h): max 60 days history
- Financial statements: max 4 years annual / 5 quarters
- Rate limiting: HTTP 429 handled by circuit breaker

## License

MIT
