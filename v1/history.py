"""History tool — get OHLCV price history for a symbol."""

from typing import Any

import yfinance as yf

INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "90m", "1h"}

VALID_INTERVALS = INTRADAY_INTERVALS | {"1d", "5d", "1wk", "1mo", "3mo"}


def history(symbol: str, interval: str, start: str, end: str) -> dict[str, Any]:
    """Get OHLCV price history in columnar format.

    Args:
        symbol: Ticker symbol.
        interval: Bar granularity (e.g., "1d", "5m").
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Columnar dict with symbol, interval, tz, t, o, h, l, c, v.

    Raises:
        ValueError: If interval is invalid or no data is returned.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Valid: {', '.join(sorted(VALID_INTERVALS))}"
        )

    t = yf.Ticker(symbol)
    df = t.history(start=start, end=end, interval=interval, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data for '{symbol}' from {start} to {end} at {interval}")

    tz = str(df.index.tz) if df.index.tz else "UTC"

    if interval in INTRADAY_INTERVALS:
        timestamps = [ts.strftime("%Y-%m-%dT%H:%M:%S") for ts in df.index]
    else:
        timestamps = [ts.strftime("%Y-%m-%d") for ts in df.index]

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "tz": tz,
        "t": timestamps,
        "o": [round(v, 2) for v in df["Open"]],
        "h": [round(v, 2) for v in df["High"]],
        "l": [round(v, 2) for v in df["Low"]],
        "c": [round(v, 2) for v in df["Close"]],
        "v": [int(v) for v in df["Volume"]],
    }
