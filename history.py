"""History tool — get OHLCV price history for a symbol."""

import logging
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from cache import Cache

logger = logging.getLogger(__name__)

INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "90m", "1h"}

CACHEABLE_INTERVALS = {"1d", "5d", "1wk", "1mo", "3mo"}

VALID_INTERVALS = INTRADAY_INTERVALS | CACHEABLE_INTERVALS

_cache: Cache | None = None


def _get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


def _find_gaps(start: date, end: date, cached: set[date]) -> list[tuple[date, date]]:
    """Identify contiguous weekday ranges not covered by *cached*.

    Walks every calendar day from *start* to *end*, skips weekends,
    and groups consecutive uncached weekdays into (gap_start, gap_end) pairs.
    """
    gaps: list[tuple[date, date]] = []
    gap_start: date | None = None
    last_missing: date | None = None
    day = start
    while day <= end:
        if day.weekday() < 5:  # weekday
            if day not in cached:
                if gap_start is None:
                    gap_start = day
                last_missing = day
            else:
                if gap_start is not None:
                    gaps.append((gap_start, last_missing))  # type: ignore[arg-type]
                    gap_start = None
                    last_missing = None
        day += timedelta(days=1)

    if gap_start is not None:
        gaps.append((gap_start, last_missing))  # type: ignore[arg-type]

    return gaps


def _exchange_today(symbol: str) -> date:
    """Return today's date in the exchange's local timezone."""
    try:
        tz_name = yf.Ticker(symbol).fast_info.get("timezone")
        if tz_name:
            return datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        pass
    return date.today()


def _is_ohlcv_finalized(symbol: str) -> bool:
    """Return True if today's regular-session OHLCV data is finalized.

    Yahoo Finance ``marketState`` values and their meanings:

    - ``PREPRE``  – before pre-market opens; today's session has not started.
    - ``PRE``     – pre-market trading; regular session has not started.
    - ``REGULAR`` – regular session is active; data still changing.
    - ``POST``    – after-hours trading; regular-session OHLCV is final.
    - ``POSTPOST``– all sessions ended for the day.
    - ``CLOSED``  – market never opened today (weekend / holiday).

    ``POST``, ``POSTPOST``, and ``CLOSED`` all mean regular-session data
    will not change, so it is safe to cache.
    """
    try:
        state = yf.Ticker(symbol).info.get("marketState", "")
        return state in ("POST", "POSTPOST", "CLOSED")
    except Exception:
        return False


def _fetch_api(
    symbol: str, interval: str, start: str, end: str, *, auto_adjust: bool
) -> Any:
    """Call yfinance and return the raw DataFrame."""
    logger.debug("fetch %s %s %s..%s", symbol, interval, start, end)
    t = yf.Ticker(symbol)
    return t.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)


def _build_response(
    symbol: str, interval: str, df: Any, *, include_ac: bool
) -> dict[str, Any]:
    """Build the columnar JSON response from a DataFrame."""
    tz = str(df.index.tz) if df.index.tz else "UTC"

    if interval in INTRADAY_INTERVALS:
        timestamps = [ts.strftime("%Y-%m-%dT%H:%M:%S") for ts in df.index]
    else:
        timestamps = [ts.strftime("%Y-%m-%d") for ts in df.index]

    result: dict[str, Any] = {
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
    if include_ac and "Adj Close" in df.columns:
        result["ac"] = [round(v, 2) for v in df["Adj Close"]]
    return result


def _build_response_from_cache(
    symbol: str, interval: str, rows: list[tuple]
) -> dict[str, Any]:
    """Build columnar JSON from cached rows.

    Each row is (date, o, h, l, c, v).
    """
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "tz": "UTC",
        "t": [r[0].strftime("%Y-%m-%d") for r in rows],
        "o": [round(r[1], 2) for r in rows],
        "h": [round(r[2], 2) for r in rows],
        "l": [round(r[3], 2) for r in rows],
        "c": [round(r[4], 2) for r in rows],
        "v": [int(r[5]) for r in rows],
    }


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    adjust: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data as a DataFrame.

    Uses cache for unadjusted daily+ intervals.  Intraday and adjusted
    requests always hit the yfinance API.

    Args:
        symbol: Ticker symbol.
        interval: Bar granularity (e.g., "1d", "5m").
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        adjust: If True, fetch adjusted prices.

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns
        (and Adj Close when adjust=True).

    Raises:
        ValueError: If interval is invalid or no data is returned.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Valid: {', '.join(sorted(VALID_INTERVALS))}"
        )

    if adjust or interval in INTRADAY_INTERVALS:
        df = _fetch_api(symbol, interval, start, end, auto_adjust=adjust)
        if df.empty:
            raise ValueError(
                f"No data for '{symbol}' from {start} to {end} at {interval}"
            )
        return df

    cache = _get_cache()
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    ex_today = _exchange_today(symbol)
    # Only query marketState when today falls in the requested range.
    if start_date <= ex_today < end_date:
        ohlcv_finalized = _is_ohlcv_finalized(symbol)
    else:
        # Today is outside the range; all data is historical.
        ohlcv_finalized = True

    cached = cache.cached_dates(symbol, interval, start_date, end_date)
    gaps = _find_gaps(start_date, end_date, cached)
    if gaps:
        logger.debug("gaps=%d for %s %s", len(gaps), symbol, interval)

    today_rows: list[tuple] = []

    for gap_start, gap_end in gaps:
        fetch_end = (gap_end + timedelta(days=1)).isoformat()
        df = _fetch_api(
            symbol,
            interval,
            gap_start.isoformat(),
            fetch_end,
            auto_adjust=False,
        )

        fetched_dates: set[date] = set()
        rows: list[tuple] = []
        if not df.empty:
            for ts, row in df.iterrows():
                bar_date = ts.date() if hasattr(ts, "date") else ts
                bar_tuple = (
                    bar_date,
                    round(float(row["Open"]), 2),
                    round(float(row["High"]), 2),
                    round(float(row["Low"]), 2),
                    round(float(row["Close"]), 2),
                    int(row["Volume"]),
                )
                if bar_date > ex_today:
                    continue
                if bar_date == ex_today:
                    if ohlcv_finalized:
                        fetched_dates.add(bar_date)
                        rows.append(bar_tuple)
                    else:
                        today_rows.append(bar_tuple)
                    continue
                fetched_dates.add(bar_date)
                rows.append(bar_tuple)

        # Detect holidays: weekdays with no data that are finalized.
        holiday_cutoff = ex_today + timedelta(days=1) if ohlcv_finalized else ex_today
        holidays: list[date] = []
        d = gap_start
        while d <= gap_end:
            if d.weekday() < 5 and d < holiday_cutoff and d not in fetched_dates:
                rows.append((d, 0, 0, 0, 0, -1))
                holidays.append(d)
            d += timedelta(days=1)
        if holidays:
            logger.debug("holidays cached: %s %s %s", symbol, interval, holidays)

        cache.put(symbol, interval, rows)

    all_rows = cache.get(symbol, interval, start_date, end_date)
    all_rows.extend(today_rows)
    if not all_rows:
        raise ValueError(f"No data for '{symbol}' from {start} to {end} at {interval}")

    return pd.DataFrame(
        all_rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
    )


def history(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    adjust: bool = False,
) -> dict[str, Any]:
    """Get OHLCV price history in columnar format.

    Args:
        symbol: Ticker symbol.
        interval: Bar granularity (e.g., "1d", "5m").
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        adjust: If True, return adjusted prices (o, h, l, c, ac, v).
                If False (default), return unadjusted prices and use cache.

    Returns:
        Columnar dict with symbol, interval, tz, t, o, h, l, c, v
        (and ac when adjust=True).

    Raises:
        ValueError: If interval is invalid or no data is returned.
    """
    df = fetch_ohlcv(symbol, interval, start, end, adjust=adjust)

    if adjust or interval in INTRADAY_INTERVALS:
        return _build_response(symbol, interval, df, include_ac=adjust)

    return _build_response_from_cache(
        symbol,
        interval,
        list(df.itertuples(index=False, name=None)),
    )
