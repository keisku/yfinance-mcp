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

Each tool has a single responsibility with no overlap.

- `search_stock` - Find stock by symbol or company name, returns identity + current price
- `history` - Historical OHLCV bars
- `technicals` - Technical indicators and signals
- `valuation` - Valuation metrics and quality score
- `financials` - Income/balance/cashflow statements

### Technical Indicators

- `trend` - SMA50-based trend direction (uptrend/downtrend)
- `rsi` - [Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp) (14-period)
- `macd` - [Moving Average Convergence Divergence](https://www.investopedia.com/terms/m/macd.asp)
- `sma_N` - [Simple Moving Average](https://www.investopedia.com/terms/s/sma.asp) (e.g., `sma_20`)
- `ema_N` - [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp) (e.g., `ema_12`)
- `wma_N` - [Weighted Moving Average](https://www.investopedia.com/articles/technical/060401.asp) (e.g., `wma_10`)
- `momentum` - [Momentum](https://www.investopedia.com/terms/m/momentum.asp) (10-period)
- `cci` - [Commodity Channel Index](https://www.investopedia.com/terms/c/commoditychannelindex.asp)
- `dmi` - [Directional Movement Index](https://www.investopedia.com/terms/d/dmi.asp) (+DI, -DI, ADX)
- `williams` - [Williams %R](https://www.investopedia.com/terms/w/williamsr.asp) oscillator
- `bb` - [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)
- `stoch` - [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp) (Slow)
- `fast_stoch` - [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp) (Fast)
- `ichimoku` - [Ichimoku Cloud](https://www.investopedia.com/terms/i/ichimoku-cloud.asp) components
- `atr` - [Average True Range](https://www.investopedia.com/terms/a/atr.asp)
- `obv` - [On-Balance Volume](https://www.investopedia.com/terms/o/onbalancevolume.asp)
- `volume_profile` - [Volume Profile](https://www.investopedia.com/terms/v/volume-analysis.asp) (POC, Value Area)
- `price_change` - [Price Change](https://www.investopedia.com/terms/p/price-change.asp) and percentage
- `fibonacci` - [Fibonacci Retracement](https://www.investopedia.com/terms/f/fibonacciretracement.asp) levels
- `pivot` - [Pivot Points](https://www.investopedia.com/terms/p/pivotpoint.asp) (Standard, Fibonacci, Camarilla, Woodie)

### Valuation Metrics

- `pe` - P/E ratios (trailing and forward)
- `eps` - Earnings per share
- `peg` - [PEG ratio](https://www.investopedia.com/terms/p/pegratio.asp) (<1 undervalued, >2 overvalued)
- `margins` - Gross, operating, and net margins
- `growth` - Revenue and earnings growth
- `ratios` - P/B, P/S, EV/EBITDA
- `dividends` - Yield, rate, payout ratio
- `quality` - 0-7 score based on ROA, cash flow, liquidity, leverage, margins, ROE

#### Historical Valuation

The `valuation` tool supports historical data via the `periods` parameter:

- `now` (default) - Current valuation from real-time data
- `YYYY` - Fiscal year (e.g., `2024`)
- `YYYY-QN` - Quarter (e.g., `2024-Q3`)
- `YYYY:YYYY` - Year range (e.g., `2023:2024`)
- `YYYY-QN:YYYY-QN` - Quarter range (e.g., `2024-Q1:2024-Q3`)

Historical mode computes P/E, P/B, P/S from financial statements. Data availability is limited to recent years (~4-5 years for annual, ~5 quarters for quarterly).

## Development

```bash
uv sync                                # Install dependencies
uv run pytest                          # Run tests
uv run pytest benchmarks/ --benchmark-only  # Run benchmarks
uv run ruff check .                    # Lint
uv run pip-audit                       # Security scan
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `YFINANCE_TARGET_POINTS` | `120` | Target data points for `history` and `technicals`. Lower = fewer tokens, faster LLM processing. Range: 50-200. |
| `YFINANCE_MAX_SPAN_DAYS` | `1680` | Maximum time range in days (~4.6 years). Optimized for 120 points at biweekly resolution. Longer ranges are truncated to keep most recent data. |
| `YFINANCE_CACHE_DISABLED` | `0` | Set to `1` to disable caching. Local cache to reduce Yahoo Finance API calls and speed up repeated queries. |
| `YFINANCE_CACHE_DB` | `~/.cache/yfinance-mcp/market.duckdb` | Custom cache path. |

#### Data Point Optimization

Both `history` and `technicals` return data in [TOON format](https://github.com/toon-format/toon), a compact notation that cuts token usage by about 45% compared to JSON. Large datasets are downsampled to fit within `YFINANCE_TARGET_POINTS`.

**How each tool samples data**

The `history` tool uses **OHLC resampling**, which combines multiple bars into one while keeping the semantics intact: the first Open, highest High, lowest Low, last Close, and total Volume. This approach preserves support and resistance levels that simple subsampling would miss.

The `technicals` tool uses the **[LTTB algorithm](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf)** (Largest-Triangle-Three-Buckets), which picks the points that best preserve the visual shape of the data. It's particularly good at keeping trend reversals and indicator crossovers.

**Why default to 120 points?**

Each TOON-formatted bar uses roughly 25–35 tokens, so 120 points comes out to about 3,600 tokens. That's enough detail for meaningful analysis while leaving plenty of room in the context window for instructions and responses.

| Points | Tokens | Best for |
|--------|--------|----------|
| 50 | ~1.5–2K | Quick, low-cost queries |
| 80 | ~2.5–4K | Compact analysis |
| 120 | ~3.5–5K | Balanced detail (default) |
| 150 | ~5–7K | In-depth analysis |

**Further reading**
- [TOON format](https://github.com/toon-format/toon) — Token-efficient serialization for LLMs
- [LTTB paper](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf) — Sveinn Steinarsson's thesis on time-series downsampling
- [Anthropic: Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)

### Yahoo Finance API Constraints

- Intraday data (1m-1h): max 60 days history
- Financial statements: ~4-5 years annual / ~5 quarters (limits historical valuation)
- Rate limiting: HTTP 429 handled by circuit breaker

## License

MIT
