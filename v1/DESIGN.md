# yfinance-mcp Design

## Design Principles

1. **Plain JSON** — Responses are standard JSON objects.
2. **Explicit over magic** — The caller controls interval, date range, and output. No auto-selection or downsampling.

## Tools

### `history`

Get OHLCV price history for a symbol.

**Input**

| Parameter  | Type   | Required | Default | Description |
|------------|--------|----------|---------|-------------|
| `symbol`   | string | yes      |         | Ticker symbol (e.g., "AAPL", "7203.T") |
| `interval` | string | yes      |         | Bar granularity (see interval constraints below) |
| `start`    | string | yes      |         | Start date (YYYY-MM-DD) |
| `end`      | string | yes      |         | End date (YYYY-MM-DD) |
| `adjust`   | boolean | no      | false   | Return adjusted prices. See [Adjusted vs Unadjusted](#adjusted-vs-unadjusted) |

**Output**

Columnar format — keys appear once, arrays hold values per bar.
`tz` is the exchange timezone (IANA name). Timestamps are ISO 8601 without offset — interpret using `tz`.
For `interval >= 1d`, timestamps use date-only format (`YYYY-MM-DD`). For intraday, `YYYY-MM-DDTHH:MM:SS`.

Unadjusted example (`adjust=false`, default):

```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "tz": "America/New_York",
  "t": ["2026-02-23", "2026-02-24", "2026-02-25"],
  "o": [263.49, 267.86, 271.78],
  "h": [269.43, 274.89, 274.94],
  "l": [263.38, 267.71, 271.05],
  "c": [266.18, 272.14, 274.23],
  "v": [37308200, 47014600, 33714300]
}
```

Adjusted example (`adjust=true`):

```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "tz": "America/New_York",
  "t": ["2026-02-23", "2026-02-24", "2026-02-25"],
  "o": [263.49, 267.86, 271.78],
  "h": [269.43, 274.89, 274.94],
  "l": [263.38, 267.71, 271.05],
  "c": [266.18, 272.14, 274.23],
  "ac": [265.90, 271.85, 273.94],
  "v": [37308200, 47014600, 33714300]
}
```

### Adjusted vs Unadjusted

| Mode | Columns | Use case |
|------|---------|----------|
| `adjust=false` (default) | o, h, l, c, v | Actual trade prices. Values never change for completed bars. |
| `adjust=true` | o, h, l, c, ac, v | Time-series analysis, return calculations, backtesting. Values are retroactively recalculated on stock splits and dividends. |

**Interval Constraints**

Each interval has a maximum date range imposed by Yahoo Finance:

| Interval | Max range |
|----------|-----------|
| `1m` | 8 days |
| `2m`, `5m`, `15m`, `30m`, `90m` | 60 days |
| `1h` | 730 days |
| `1d`, `5d`, `1wk`, `1mo`, `3mo` | unlimited |

## Cache

Unadjusted OHLC prices for completed bars are immutable — they represent the actual trade prices recorded on that day and never change. The cache exploits this property to eliminate redundant API calls.

### What Is Cached

- **Cached**: Unadjusted OHLCV (o, h, l, c, v) for daily and longer intervals (1d, 5d, 1wk, 1mo, 3mo), excluding today's bar.
- **Not cached**: Intraday intervals (short retention, low reuse). Adjusted prices (retroactively change on corporate actions). Today's bar (still trading).

### Storage

Single Apache Parquet file: `~/.cache/yfinance-mcp/unadjusted_ohlc.parquet`

Override path via `YFINANCE_CACHE_PATH` environment variable.

Parquet columns: `symbol`, `interval`, `date`, `o`, `h`, `l`, `c`, `v`.

DuckDB (in-memory) is used as the query engine to read and write the Parquet file.

### Gap Detection

When a request partially overlaps with cached data, the cache identifies missing date ranges (gaps) and fetches only those from the API.

Algorithm:
1. Query cached dates for the requested (symbol, interval, start, end).
2. Walk weekdays from start to end. Group uncached weekdays into contiguous ranges.
3. Fetch each gap from the API and store in cache (excluding today).
4. Return the full range from cache.

## Architecture

```
server.py      # MCP server: tool registration, request dispatch, error handling
history.py     # history tool logic (wraps yfinance Ticker.history), gap detection
cache.py       # Parquet cache using DuckDB as query engine
```

`server.py` owns the MCP protocol layer. `history.py` is a pure data module with no MCP awareness — it returns Python dicts, and `server.py` serializes to JSON. `cache.py` handles Parquet I/O via DuckDB.

### Error Handling

All errors return a JSON object with `error`:

```json
{
  "error": "No data for 'INVALID' from 2026-02-01 to 2026-02-28 at 1d"
}
```

Simple dict-based errors. No custom exception hierarchy.

## Dependencies

- yfinance
- pandas
- duckdb
