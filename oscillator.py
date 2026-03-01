"""Oscillator tool — momentum indicators for a symbol."""

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
from history import fetch_ohlcv

logger = logging.getLogger(__name__)

WARMUP_CALENDAR_DAYS = 112


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    k = 100 * (close - lowest) / (highest - lowest)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def oscillator(
    symbol: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Compute momentum oscillators for a symbol using daily bars.

    Fetches extra warmup bars before *start* so all indicators are valid
    from the first returned timestamp onward.

    Args:
        symbol: Ticker symbol.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Columnar dict with symbol, interval, tz, t, and oscillator values.

    Raises:
        ValueError: If no data is returned.
    """
    warmup_start = (
        date.fromisoformat(start) - timedelta(days=WARMUP_CALENDAR_DAYS)
    ).isoformat()
    logger.debug("oscillator %s warmup=%s..%s", symbol, warmup_start, end)

    df = fetch_ohlcv(symbol, "1d", warmup_start, end)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    rsi = _rsi(close)
    stoch_k, stoch_d = _stochastic(high, low, close)

    indicators = pd.DataFrame(
        {
            "rsi": rsi,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
        },
        index=df.index,
    )

    if "Date" in df.columns:
        timestamps = [d.isoformat() for d in df["Date"]]
        mask = df["Date"] >= date.fromisoformat(start)
    else:
        timestamps = [ts.strftime("%Y-%m-%d") for ts in df.index]
        mask = df.index >= pd.Timestamp(start)
    tz = str(df.index.tz) if hasattr(df.index, "tz") and df.index.tz else "UTC"

    indicators["_t"] = timestamps
    trimmed = indicators.loc[mask].dropna()

    if trimmed.empty:
        raise ValueError(f"No oscillator data for '{symbol}' from {start} to {end}")

    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": "1d",
        "tz": tz,
        "t": trimmed["_t"].tolist(),
    }
    indicator_cols = [
        "rsi",
        "stoch_k",
        "stoch_d",
    ]
    for col in indicator_cols:
        result[col] = [round(v, 2) for v in trimmed[col]]

    return result
