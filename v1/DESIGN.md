# yfinance-mcp Design

## Design Principles

1. **Plain JSON** — Responses are standard JSON objects.
2. **Explicit over magic** — The caller controls interval, date range, and output. No auto-selection or downsampling.

## Tools

### `history`

Get OHLCV price history for a symbol.

**Input**

| Parameter  | Type   | Required | Description |
|------------|--------|----------|-------------|
| `symbol`   | string | yes      | Ticker symbol (e.g., "AAPL", "7203.T") |
| `interval` | string | yes      | Bar granularity (see interval constraints below) |
| `start`    | string | yes      | Start date (YYYY-MM-DD) |
| `end`      | string | yes      | End date (YYYY-MM-DD) |

**Output**

Columnar format — keys appear once, arrays hold values per bar.
`tz` is the exchange timezone (IANA name). Timestamps are ISO 8601 without offset — interpret using `tz`.
For `interval >= 1d`, timestamps use date-only format (`YYYY-MM-DD`). For intraday, `YYYY-MM-DDTHH:MM:SS`.

Daily example (`interval=1d`):

```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "tz": "America/New_York",
  "t": ["2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27"],
  "o": [263.49, 267.86, 271.78, 274.95, 272.81],
  "h": [269.43, 274.89, 274.94, 276.11, 272.81],
  "l": [263.38, 267.71, 271.05, 270.8, 262.89],
  "c": [266.18, 272.14, 274.23, 272.95, 264.18],
  "v": [37308200, 47014600, 33714300, 32345100, 71592273]
}
```

Intraday example (`interval=5m`):

```json
{
  "symbol": "AAPL",
  "interval": "5m",
  "tz": "America/New_York",
  "t": ["2026-02-27T09:30:00", "2026-02-27T09:35:00", "2026-02-27T09:40:00", "2026-02-27T09:45:00"],
  "o": [272.81, 269.47, 269.56, 268.93],
  "h": [272.81, 269.96, 269.74, 270.91],
  "l": [269.2, 269.04, 268.87, 268.76],
  "c": [269.47, 269.55, 268.98, 270.89],
  "v": [1968544, 1005719, 1429758, 490275]
}
```

**Interval Constraints**

Each interval has a maximum date range imposed by Yahoo Finance:

| Interval | Max range |
|----------|-----------|
| `1m` | 8 days |
| `2m`, `5m`, `15m`, `30m`, `90m` | 60 days |
| `1h` | 730 days |
| `1d`, `5d`, `1wk`, `1mo`, `3mo` | unlimited |

## Architecture

```
server.py      # MCP server: tool registration, request dispatch, error handling
history.py     # history tool logic (wraps yfinance Ticker.history)
```

`server.py` owns the MCP protocol layer. `history.py` is a pure data module with no MCP awareness — it returns Python dicts, and `server.py` serializes to JSON.

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
