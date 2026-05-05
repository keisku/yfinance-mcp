"""Volume tool — volume moving averages for a symbol."""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from history import fetch_ohlcv

logger = logging.getLogger(__name__)

WARMUP_CALENDAR_DAYS = 112

SMA_PERIODS = (5, 10, 20, 50)


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period, min_periods=period).mean()


def _epoch_to_iso_date(ts: Any) -> str | None:
    """Convert a Unix epoch timestamp (seconds, UTC) to YYYY-MM-DD."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _fetch_short_interest(symbol: str) -> dict[str, Any] | None:
    """Fetch a short-interest snapshot from yfinance info.

    Returns None for tickers that do not report short interest
    (non-US equities, most ETFs, crypto).
    """
    # Yahoo Finance sources short interest from FINRA, which only covers
    # US-listed equities. yfinance convention uses "." as an exchange
    # suffix (e.g. ".T", ".HK", ".L"), so skip the network call.
    if "." in symbol:
        return None

    try:
        info = yf.Ticker(symbol).info
    except Exception as e:
        logger.warning("short interest fetch failed for %s: %s", symbol, e)
        return None

    shares_short = info.get("sharesShort")
    if shares_short is None:
        return None

    snapshot: dict[str, Any] = {
        "as_of": _epoch_to_iso_date(info.get("dateShortInterest")),
        "shares_short": int(shares_short),
        "pct_of_float": info.get("shortPercentOfFloat"),
        "days_to_cover": info.get("shortRatio"),
    }

    prior = info.get("sharesShortPriorMonth")
    if prior is not None:
        snapshot["prior_month"] = {
            "as_of": _epoch_to_iso_date(info.get("sharesShortPreviousMonthDate")),
            "shares_short": int(prior),
        }

    return snapshot


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
        Columnar dict with symbol, interval, tz, t, volume, volume SMA values,
        and a ``short_interest`` snapshot (or ``None`` when the ticker does
        not report short interest, e.g. non-US equities, ETFs, crypto).

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

    timestamps = [ts.strftime("%Y-%m-%d") for ts in df.index]
    mask = df.index >= pd.Timestamp(start, tz=df.index.tz)
    tz = str(df.index.tz)

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

    result["short_interest"] = _fetch_short_interest(symbol)

    return result
