# Yahoo Finance MCP

A Python MCP server for Yahoo Finance data, optimized for AI agents.

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
| `price` | OHLCV bars with date range support |
| `technicals` | RSI, MACD, SMA, Bollinger Bands, etc |
| `fundamentals` | P/E, EPS, margins, dividends |
| `financials` | Income/balance/cashflow statements |
| `peers` | Compare multiple stocks |
| `search` | Find ticker by company name |

## Development

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests
uv run ruff check .        # Lint
uv run pip-audit           # Security scan
```

## License

MIT
