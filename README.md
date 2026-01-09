# Yahoo Finance MCP

MCP server providing real-time stock data, technicals, and fundamentals via Yahoo Finance.

## Quick Start

Add to your MCP client configuration:

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

Or run the HTTP server directly and connect via URL:

```bash
YFINANCE_TRANSPORT=http uvx --from git+https://github.com/keisku/yfinance-mcp yfinance-mcp
```

```json
{
  "mcpServers": {
    "yfinance": {
      "url": "http://127.0.0.1:9246/mcp"
    }
  }
}
```

## Tools

- `search_stock` - Find stock by symbol or company name, returns identity + current price
- `history` - Historical OHLCV bars with adjusted close (`c` for price-return, `ac` for total-return). [more](#architecture)
- `technicals` - Technical indicators using adjusted close: RSI, MACD, Bollinger Bands, Ichimoku, and [more](#technical-indicators)
- `valuation` - P/E, PEG, margins, growth, quality score, and [more](#valuation-metrics)
- `financials` - Income, balance sheet, cash flow (Yahoo Finance API provides ~4-5 years annual, ~5 quarters)

## Configuration

Defaults work well for most use cases. Override only if needed:

Transport:
- `YFINANCE_TRANSPORT` (default: `stdio`) - Transport mode: `stdio` or `http`.
- `YFINANCE_HTTP_HOST` (default: `127.0.0.1`) - HTTP server bind address (only for `http` transport).
- `YFINANCE_HTTP_PORT` (default: `9246`) - HTTP server port (only for `http` transport).

Data:
- `YFINANCE_TARGET_POINTS` (default: `200`) - Data points per response. [more](#architecture).
- `YFINANCE_CACHE_DISABLED` (default: unset) - Set to `1` to disable caching.
- `YFINANCE_CACHE_DB` (default: `~/.cache/yfinance-mcp/market.duckdb`) - Cache database path.
- `YFINANCE_INTRADAY_TTL_MINUTES` (default: `30`) - Intraday cache TTL in minutes.

Logging:
- `YFINANCE_LOG_LEVEL` (default: `WARNING`) - Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `YFINANCE_LOG_FILE` (default: `/tmp/yfinance-mcp.log` on Unix, `%LOCALAPPDATA%/yfinance-mcp/logs/yfinance-mcp.log` on Windows) - Log file path. Set to empty string to disable file logging.
- `YFINANCE_LOG_CONSOLE` (default: disabled) - Set to `1`, `true`, or `yes` to enable console logging.
- `YFINANCE_LOG_STREAM` (default: `stderr`) - Set to `stdout` to log to stdout instead of stderr.

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

**Data points & tokens** - Each response returns ~200 data points by default, using ~3.3K tokens. This stays near the [1-3K ideal range](https://github.com/adobe-research/NoLiMa) where top LLMs maintain high accuracy. The server auto-selects resolution based on date range:

| Date Range | Interval | Why |
|------------|----------|-----|
| 1-5 days | 5-minute | 78 bars/day × 2.5 days ≈ 200 |
| 5-20 days | 15-30 min | 13-26 bars/day × 10 days ≈ 200 |
| 20-130 days | Hourly | 6.5 bars/day × 30 days ≈ 200 |
| 130+ days | Daily/Weekly | 1 bar/day × 200 days = 200 |

Configure via `YFINANCE_TARGET_POINTS` (100-400 range):

- 100 points → ~1.6K tokens (quick queries)
- 200 points → ~3.3K tokens (default, near ideal range)
- 400 points → ~6.5K tokens (detailed analysis, within [10K practical limit](https://cookbook.openai.com/examples/gpt-5/gpt-5-1-codex-max_prompting_guide))

Data uses delta-encoded split format in [TOON](https://github.com/toon-format/toon), achieving ~56% token reduction vs JSON by eliminating repeated keys and delta-encoding dates. Downsampling preserves key features: `history` uses OHLC resampling (support/resistance), `technicals` uses [LTTB](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf) (trend reversals).

**Caching** - Local cache reduces Yahoo Finance API calls. The cache detects gaps in cached data and fetches only missing ranges.

## License

MIT
