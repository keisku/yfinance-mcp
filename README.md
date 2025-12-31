# Yahoo Finance MCP (Python)

A Python MCP server for Yahoo Finance data, optimized for AI agents.

## Design Philosophy

- **Compact JSON responses** - no indentation, short keys
- **Batched operations** - multiple indicators/symbols in one call
- **No redundancy** - each data point returned once
- **Flexible date ranges** - arbitrary historical data via `start`/`end` parameters

Based on [Anthropic's Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) and [Snyk MCP best practices](https://snyk.io/articles/5-best-practices-for-building-mcp-servers/).

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Usage

```bash
uv run yfinance-mcp
```

### Cursor/Claude Configuration

```json
{
  "mcpServers": {
    "yfinance": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yfinance-mcp", "yfinance-mcp"]
    }
  }
}
```

## Recommended Workflow

```python
# 1. Get complete overview with Quality Score
summary(symbol="SHOP")

# 2. Compare with peers
peers(symbols=["SHOP", "META"], metrics=["price", "pe", "market_cap"])

# 3. Get technicals for trading signals
technicals(symbol="SHOP", indicators=["rsi", "macd"])

# 4. Get price history (period-based or date range)
price(symbol="SHOP", period="ytd")
price(symbol="SHOP", start="2020-01-01", end="2024-12-31")
```

## Tools

| Tool | Purpose | Key Args |
|------|---------|----------|
| `summary` | **Best starting point** - Quality Score, PEG, trend | `symbol` |
| `price` | OHLCV bars with date range support | `symbol`, `period`, `start`, `end`, `interval` |
| `technicals` | RSI, MACD, SMA, BB, etc | `symbol`, `indicators[]` |
| `fundamentals` | P/E, EPS, margins, dividends | `symbol`, `metrics[]` |
| `financials` | Income/balance/cashflow | `symbol`, `statement`, `freq` |
| `peers` | Compare multiple stocks | `symbols[]`, `metrics[]` |
| `search` | Find ticker by name | `query` |

### Response Keys

- `o/h/l/c/v` = Open/High/Low/Close/Volume
- `pe/pb/ps` = Price-to-Earnings/Book/Sales
- `mcap` = Market Cap
- `pct` = Percent change
- `fwd` = Forward
- `pos` = Position (above/below)

## Docker

```bash
docker build -t yfinance-mcp .
docker run -i yfinance-mcp
```

```json
{
  "mcpServers": {
    "yfinance": {
      "command": "docker",
      "args": ["run", "-i", "yfinance-mcp"]
    }
  }
}
```

## Development

```bash
uv run pytest                            # Run all tests
uv run pytest -m "not crosscheck"        # Skip pandas-ta crosscheck tests
uv run mypy src/yfinance_mcp        # Type check
uv run ruff check src/yfinance_mcp  # Lint
uv run pip-audit                         # Security scan
```

## Environment Variables

```bash
# Logging
MCP_LOG_LEVEL=DEBUG                  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)
MCP_LOG_FILE=/tmp/yfinance-mcp.log   # Log file path (default: /tmp/yfinance-mcp.log)
MCP_LOG_CONSOLE=1                    # Enable console logging to stderr (default: disabled)

# Cache
YFINANCE_CACHE_DISABLED=1            # Disable DuckDB cache (always fetch from API)
YFINANCE_CACHE_DB=/path/to/cache.db  # Custom cache path (default: ~/.cache/yfinance-mcp/market.duckdb)
YFINANCE_INTRADAY_TTL_MINUTES=30     # Intraday data cache TTL in minutes (default: 30)
```

## Logging

The server provides structured logging for debugging and monitoring:

```bash
# Enable debug logging
MCP_LOG_LEVEL=DEBUG uv run yfinance-mcp

# View logs
tail -f /tmp/yfinance-mcp.log
```

Log messages use structured format with key=value pairs:
- `tool_call_start/success/error` - Tool execution with timing
- `cache_hit/miss` - Cache performance
- `api_fetch` - API calls with retry info
- `circuit_breaker` - Service protection state

## Limitations

These are Yahoo Finance API constraints, not yfinance or this MCP.

- **Intraday intervals** (1m-1h): max 60 days history
- **Financial statements**: max 4 years annual / 5 quarters
- **Rate limiting**: Yahoo enforces HTTP 429; circuit breaker handles gracefully

## License

MIT
