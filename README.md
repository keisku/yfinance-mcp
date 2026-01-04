# Yahoo Finance MCP

MCP server providing real-time stock data, technicals, and fundamentals via Yahoo Finance.

## Quick Start

Add to your MCP client configuration:

uv:

```json
{
  "mcpServers": {
    "yfinance": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/keisku/yfinance-mcp",
        "yfinance-mcp"
      ]
    }
  }
}
```

Docker:

```json
{
  "mcpServers": {
    "yfinance": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "yfinance-cache:/home/app/.cache/yfinance-mcp",
        "ghcr.io/keisku/yfinance-mcp:latest"
      ]
    }
  }
}
```

## Tools

- `search_stock` - Find stock by symbol or company name, returns identity + current price
- `history` - Historical OHLCV bars (up to 1680 days ≈ 14 days * 120 points, chosen for biweekly resolution at 120 points)
- `technicals` - Technical indicators: RSI, MACD, Bollinger Bands, Ichimoku, and [more](#technical-indicators)
- `valuation` - P/E, PEG, margins, growth, quality score, and [more](#valuation-metrics)
- `financials` - Income, balance sheet, and cash flow statements

## Configuration

- `YFINANCE_TARGET_POINTS` (default: `120`) - Data points per response. Lower = fewer tokens. Range: 50-200.
- `YFINANCE_MAX_SPAN_DAYS` (default: `1680`) - Maximum history in days (~4.6 years).
- `YFINANCE_CACHE_DISABLED` (default: unset) - Set to `1` to disable caching.
- `YFINANCE_CACHE_DB` (default: `~/.cache/yfinance-mcp/market.duckdb`) - Cache database path.
- `YFINANCE_INTRADAY_TTL_MINUTES` (default: `30`) - Intraday cache TTL in minutes.

## Limitations

- Intraday data (1m-1h): max 60 days history
- Financial statements: ~4-5 years annual / ~5 quarters
- Rate limiting: HTTP 429 handled by circuit breaker

---

## Reference

### Technical Indicators

Available via the `technicals` tool:

- `trend` - SMA50-based trend direction
- `rsi` - [Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp) (14-period)
- `macd` - [Moving Average Convergence Divergence](https://www.investopedia.com/terms/m/macd.asp)
- `sma_N`, `ema_N`, `wma_N` - Moving averages (e.g., `sma_20`, `ema_12`)
- `bb` - [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)
- `stoch`, `fast_stoch` - [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp)
- `ichimoku` - [Ichimoku Cloud](https://www.investopedia.com/terms/i/ichimoku-cloud.asp)
- `atr` - [Average True Range](https://www.investopedia.com/terms/a/atr.asp)
- `obv` - [On-Balance Volume](https://www.investopedia.com/terms/o/onbalancevolume.asp)
- `cci` - [Commodity Channel Index](https://www.investopedia.com/terms/c/commoditychannelindex.asp)
- `dmi` - [Directional Movement Index](https://www.investopedia.com/terms/d/dmi.asp)
- `williams` - [Williams %R](https://www.investopedia.com/terms/w/williamsr.asp)
- `momentum` - [Momentum](https://www.investopedia.com/terms/m/momentum.asp) (10-period)
- `volume_profile` - [Volume Profile](https://www.investopedia.com/terms/v/volume-analysis.asp)
- `price_change` - Price change and percentage
- `fibonacci` - [Fibonacci Retracement](https://www.investopedia.com/terms/f/fibonacciretracement.asp) levels
- `pivot` - [Pivot Points](https://www.investopedia.com/terms/p/pivotpoint.asp) (Standard, Fibonacci, Camarilla, Woodie)

### Valuation Metrics

Available via the `valuation` tool:

- `pe` - P/E ratios (trailing and forward)
- `eps` - Earnings per share
- `peg` - [PEG ratio](https://www.investopedia.com/terms/p/pegratio.asp) (<1 undervalued, >2 overvalued)
- `margins` - Gross, operating, and net margins
- `growth` - Revenue and earnings growth
- `ratios` - P/B, P/S, EV/EBITDA
- `dividends` - Yield, rate, payout ratio
- `quality` - 0-7 score based on ROA, cash flow, liquidity, leverage, margins, ROE

**Historical valuation** via `periods` parameter:

- `now` (default) - Current valuation
- `YYYY` - Fiscal year (e.g., `2024`)
- `YYYY-QN` - Quarter (e.g., `2024-Q3`)
- `YYYY:YYYY` - Year range (e.g., `2023:2024`)

---

## Development

```bash
uv sync                                # Install dependencies
uv run pytest                          # Run tests
uv run pytest benchmarks/ --benchmark-only  # Run benchmarks
uv run ruff check .                    # Lint
uv run pip-audit                       # Security scan
```

### Architecture

**Caching** - Local cache reduces Yahoo Finance API calls. The cache detects gaps in cached data and fetches only missing ranges. Consecutive gaps are merged to minimize API calls while respecting `YFINANCE_MAX_SPAN_DAYS`.

**Token optimization** - Data is returned in [TOON format](https://github.com/toon-format/toon), cutting token usage by ~45% vs JSON. Large datasets are downsampled:

- `history` uses OHLC resampling (preserves support/resistance levels)
- `technicals` uses [LTTB algorithm](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf) (preserves trend reversals)

Each TOON bar (e.g., `2024-01-04,100.5,101.2,99.8,100.9,1000000`) uses ~30 tokens:

- 50 points (~1.6K tokens) - Quick, low-cost queries
- 80 points (~2.6K tokens) - Compact analysis
- 120 points (~3.8K tokens) - Balanced detail (default)
- 150 points (~4.8K tokens) - In-depth analysis

## License

MIT
