# yfinance-mcp

An [MCP](https://modelcontextprotocol.io/) server that exposes Yahoo Finance market data to LLM agents.

## Tools

### `history`

Returns OHLCV price bars for a given symbol and date range.

| Parameter  | Type   | Required | Description                                |
|------------|--------|----------|--------------------------------------------|
| `symbol`   | string | yes      | Ticker symbol (e.g. `AAPL`, `7203.T`)     |
| `interval` | string | yes      | Bar granularity (`1m` … `3mo`)             |
| `start`    | string | yes      | Start date (`YYYY-MM-DD`)                  |
| `end`      | string | yes      | End date (`YYYY-MM-DD`)                    |
| `adjust`   | bool   | no       | Return split/dividend-adjusted prices      |

Response columns: `t` (timestamps), `o`, `h`, `l`, `c`, `v`, and `ac` (adjusted close, when `adjust=true`).

### `oscillator`

Returns daily momentum oscillators for a given symbol and date range. Warmup data is fetched automatically so indicators are valid from the first returned date.

| Parameter | Type   | Required | Description                            |
|-----------|--------|----------|----------------------------------------|
| `symbol`  | string | yes      | Ticker symbol (e.g. `AAPL`, `7203.T`) |
| `start`   | string | yes      | Start date (`YYYY-MM-DD`)              |
| `end`     | string | yes      | End date (`YYYY-MM-DD`)                |

Response columns: `t`, `rsi`, `stoch_k`, `stoch_d`.

### `trend`

Returns daily trend-following indicators for a given symbol and date range. Warmup data is fetched automatically so indicators are valid from the first returned date.

| Parameter | Type   | Required | Description                            |
|-----------|--------|----------|----------------------------------------|
| `symbol`  | string | yes      | Ticker symbol (e.g. `AAPL`, `7203.T`) |
| `start`   | string | yes      | Start date (`YYYY-MM-DD`)              |
| `end`     | string | yes      | End date (`YYYY-MM-DD`)                |

Response columns: `t`, `macd`, `macd_signal`, `macd_hist`, `plus_di`, `minus_di`, `adx`.

## Setup

Requires Python 3.13+.

```bash
uv sync
```

## Usage

### Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "yfinance": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yfinance-mcp", "python", "server.py"]
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "yfinance": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yfinance-mcp", "python", "server.py"]
    }
  }
}
```

## License

MIT
