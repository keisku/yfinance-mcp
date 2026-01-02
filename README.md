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
| `diagnose` | Stock health diagnosis with intelligent analysis recommendations |
| `history` | OHLCV bars with date range support |
| `technicals` | Technical indicators (see below) |
| `fundamentals` | P/E, EPS, margins, dividends |
| `financials` | Income/balance/cashflow statements |
| `search` | Find ticker by company name |

### Technical Indicators

| Indicator | Description |
|-----------|-------------|
| `rsi` | [Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp) (14-period) |
| `macd` | [Moving Average Convergence Divergence](https://www.investopedia.com/terms/m/macd.asp) |
| `sma_N` | [Simple Moving Average](https://www.investopedia.com/terms/s/sma.asp) (e.g., `sma_20`) |
| `ema_N` | [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp) (e.g., `ema_12`) |
| `wma_N` | [Weighted Moving Average](https://www.investopedia.com/articles/technical/060401.asp) (e.g., `wma_10`) |
| `momentum` | [Momentum](https://www.investopedia.com/terms/m/momentum.asp) (10-period) |
| `cci` | [Commodity Channel Index](https://www.investopedia.com/terms/c/commoditychannelindex.asp) |
| `dmi` | [Directional Movement Index](https://www.investopedia.com/terms/d/dmi.asp) (+DI, -DI, ADX) |
| `williams` | [Williams %R](https://www.investopedia.com/terms/w/williamsr.asp) oscillator |
| `bb` | [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp) |
| `stoch` | [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp) (Slow) |
| `fast_stoch` | [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp) (Fast) |
| `ichimoku` | [Ichimoku Cloud](https://www.investopedia.com/terms/i/ichimoku-cloud.asp) components |
| `atr` | [Average True Range](https://www.investopedia.com/terms/a/atr.asp) |
| `obv` | [On-Balance Volume](https://www.investopedia.com/terms/o/onbalancevolume.asp) |
| `volume_profile` | [Volume Profile](https://www.investopedia.com/terms/v/volume-analysis.asp) (POC, Value Area) |
| `price_change` | [Price Change](https://www.investopedia.com/terms/p/price-change.asp) and percentage |
| `fibonacci` | [Fibonacci Retracement](https://www.investopedia.com/terms/f/fibonacciretracement.asp) levels |
| `pivot` | [Pivot Points](https://www.investopedia.com/terms/p/pivotpoint.asp) (Standard, Fibonacci, Camarilla, Woodie) |

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
