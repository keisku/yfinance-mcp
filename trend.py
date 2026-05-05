"""Trend tool — trend-following indicators for a symbol."""

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
from history import fetch_ohlcv

logger = logging.getLogger(__name__)

WARMUP_CALENDAR_DAYS = 300


SMA_PERIODS = (20, 50, 200)
EMA_PERIODS = (20, 50, 200)


def _sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(period, min_periods=period).mean()


def _ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=period, min_periods=period, adjust=False).mean()


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _adx_dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX, +DI, and -DI using Wilder's smoothing."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100 * smooth_plus / atr
    minus_di = 100 * smooth_minus / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return plus_di, minus_di, adx


def trend(
    symbol: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Compute trend-following indicators for a symbol using daily bars.

    Fetches extra warmup bars before *start* so all indicators are valid
    from the first returned timestamp onward.

    Args:
        symbol: Ticker symbol.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Columnar dict with symbol, interval, tz, t, and trend indicator values.

    Raises:
        ValueError: If no data is returned.
    """
    warmup_start = (
        date.fromisoformat(start) - timedelta(days=WARMUP_CALENDAR_DAYS)
    ).isoformat()
    logger.debug("trend %s warmup=%s..%s", symbol, warmup_start, end)

    df = fetch_ohlcv(symbol, "1d", warmup_start, end, adjust=True)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    smas = {f"sma_{p}": _sma(close, p) for p in SMA_PERIODS}
    emas = {f"ema_{p}": _ema(close, p) for p in EMA_PERIODS}
    macd_line, macd_signal, macd_hist = _macd(close)
    plus_di, minus_di, adx = _adx_dmi(high, low, close)

    indicators = pd.DataFrame(
        {
            **smas,
            **emas,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": adx,
        },
        index=df.index,
    )

    timestamps = [ts.strftime("%Y-%m-%d") for ts in df.index]
    mask = df.index >= pd.Timestamp(start, tz=df.index.tz)
    tz = str(df.index.tz)

    indicators["_t"] = timestamps
    trimmed = indicators.loc[mask].dropna()

    if trimmed.empty:
        raise ValueError(f"No trend data for '{symbol}' from {start} to {end}")

    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": "1d",
        "tz": tz,
        "t": trimmed["_t"].tolist(),
    }
    indicator_cols = [
        *smas,
        *emas,
        "macd",
        "macd_signal",
        "macd_hist",
        "plus_di",
        "minus_di",
        "adx",
    ]
    for col in indicator_cols:
        result[col] = [round(v, 2) for v in trimmed[col]]

    return result
