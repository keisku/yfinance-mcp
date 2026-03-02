"""Volume tool — volume moving averages for a symbol."""

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
from history import fetch_ohlcv

logger = logging.getLogger(__name__)

WARMUP_CALENDAR_DAYS = 112

SMA_PERIODS = (5, 10, 20, 50)


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period, min_periods=period).mean()


def volume(
    symbol: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Compute volume moving averages for a symbol using daily bars.

    Fetches extra warmup bars before *start* so all indicators are valid
    from the first returned timestamp onward.

    Args:
        symbol: Ticker symbol.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Columnar dict with symbol, interval, tz, t, volume, and volume SMA values.

    Raises:
        ValueError: If no data is returned.
    """
    warmup_start = (
        date.fromisoformat(start) - timedelta(days=WARMUP_CALENDAR_DAYS)
    ).isoformat()
    logger.debug("volume %s warmup=%s..%s", symbol, warmup_start, end)

    df = fetch_ohlcv(symbol, "1d", warmup_start, end, adjust=False)

    vol = df["Volume"].astype(float)

    smas = {f"vol_sma_{p}": _sma(vol, p) for p in SMA_PERIODS}

    indicators = pd.DataFrame(
        {"volume": vol, **smas},
        index=df.index,
    )

    if "Date" in df.columns:
        timestamps = [d.isoformat() for d in df["Date"]]
        mask = df["Date"] >= date.fromisoformat(start)
    else:
        timestamps = [ts.strftime("%Y-%m-%d") for ts in df.index]
        mask = df.index >= pd.Timestamp(start, tz=df.index.tz)
    tz = str(df.index.tz) if hasattr(df.index, "tz") and df.index.tz else "UTC"

    indicators["_t"] = timestamps
    trimmed = indicators.loc[mask].dropna()

    if trimmed.empty:
        raise ValueError(f"No volume data for '{symbol}' from {start} to {end}")

    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": "1d",
        "tz": tz,
        "t": trimmed["_t"].tolist(),
        "volume": [int(v) for v in trimmed["volume"]],
    }
    for col in smas:
        result[col] = [round(v, 2) for v in trimmed[col]]

    return result
