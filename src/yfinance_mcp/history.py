"""Price data access layer for Yahoo Finance."""

import logging
import os
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from .helpers import INTRADAY_INTERVALS, OHLCV_COLS_TO_LONG, PERIOD_DELTAS

if TYPE_CHECKING:
    from .cache import CachedPriceFetcher

logger = logging.getLogger("yfinance_mcp.history")

_fetcher: "CachedPriceFetcher | None" = None
_initialized = False

# server.py injects its ticker function for validation
_ticker_fn: Callable[[str], yf.Ticker] | None = None


class _IntradayCache:
    """In-memory cache for intraday data with TTL expiration."""

    def __init__(self, ttl_minutes: int = 30):
        self.ttl = timedelta(minutes=ttl_minutes)
        self._cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, symbol: str, period: str, interval: str) -> str:
        return f"{symbol.upper()}:{period}:{interval}"

    def get(self, symbol: str, period: str, interval: str) -> pd.DataFrame | None:
        key = self._make_key(symbol, period, interval)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        df, fetched_at = entry
        if datetime.now() - fetched_at > self.ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return df.copy()

    def set(self, symbol: str, period: str, interval: str, df: pd.DataFrame) -> None:
        key = self._make_key(symbol, period, interval)
        self._cache[key] = (df.copy(), datetime.now())

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "entries": len(self._cache)}


_intraday_cache: _IntradayCache | None = None


def _get_intraday_cache() -> _IntradayCache:
    """Get or create the intraday cache singleton."""
    global _intraday_cache
    if _intraday_cache is None:
        ttl = int(os.getenv("YFINANCE_INTRADAY_TTL_MINUTES", "30"))
        _intraday_cache = _IntradayCache(ttl_minutes=ttl)
        logger.debug("intraday_cache_init ttl_minutes=%d", ttl)
    return _intraday_cache


def _is_cache_disabled() -> bool:
    """Check if caching is disabled (checked at runtime for testability)."""
    return os.getenv("YFINANCE_CACHE_DISABLED", "").lower() in ("1", "true", "yes")


def _init() -> None:
    """Initialize the data layer (lazy, called on first use)."""
    global _fetcher, _initialized

    # early return if initialization state is unchanged
    if _initialized and (_fetcher is None) == _is_cache_disabled():
        return

    _initialized = True

    if _is_cache_disabled():
        logger.debug("history_init cache=disabled")
        _fetcher = None
        return

    try:
        from .cache import CachedPriceFetcher

        _fetcher = CachedPriceFetcher()
        logger.debug("history_init cache=enabled")
    except ImportError as e:
        logger.warning("history_init cache=unavailable error=%s", e)
        _fetcher = None


def set_ticker_fn(fn: Callable[[str], yf.Ticker] | None) -> None:
    """Set the ticker function (used by server.py to inject _ticker)."""
    global _ticker_fn
    _ticker_fn = fn


def get_history(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    ticker: yf.Ticker | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Get OHLCV price history for a symbol."""
    _init()

    # yfinance expects start when end is provided
    if start is None and end is not None:
        end_date = date.fromisoformat(end[:10])  # Handle datetime strings too
        if period == "ytd":
            start = date(end_date.year, 1, 1).isoformat()
        else:
            delta = PERIOD_DELTAS.get(period, relativedelta(months=1))
            start = (end_date - delta).isoformat()
        logger.debug(
            "get_history computed start=%s from end=%s period=%s",
            start,
            end,
            period,
        )

    logger.debug(
        "get_history symbol=%s start=%s end=%s period=%s interval=%s cache=%s",
        symbol,
        start,
        end,
        period,
        interval,
        _fetcher is not None,
    )

    if _fetcher is not None and interval in ("1d", "1wk", "1mo"):
        if start is not None:
            df = _fetcher.get_history_by_dates(symbol, start, end, interval)
        else:
            df = _fetcher.get_history(symbol, period, interval)
        result = df.rename(columns=OHLCV_COLS_TO_LONG)
        logger.debug(
            "get_history symbol=%s interval=%s bars=%d source=cache",
            symbol,
            interval,
            len(result),
        )
        return result

    if interval in INTRADAY_INTERVALS:
        if ticker is not None:
            t = ticker
        elif _ticker_fn is not None:
            t = _ticker_fn(symbol)
        else:
            t = yf.Ticker(symbol.upper())

        if start is not None:
            result = t.history(start=start, end=end, interval=interval)
            logger.debug(
                "get_history symbol=%s interval=%s bars=%d source=api (date-range)",
                symbol,
                interval,
                len(result),
            )
            return result

        cache = _get_intraday_cache()
        cached = cache.get(symbol, period, interval)
        if cached is not None:
            logger.debug(
                "get_history symbol=%s interval=%s bars=%d source=intraday_cache",
                symbol,
                interval,
                len(cached),
            )
            return cached

        result = t.history(period=period, interval=interval)
        cache.set(symbol, period, interval, result)
        logger.debug(
            "get_history symbol=%s interval=%s bars=%d source=api+intraday_cache",
            symbol,
            interval,
            len(result),
        )
        return result

    if ticker is not None:
        t = ticker
    elif _ticker_fn is not None:
        t = _ticker_fn(symbol)
    else:
        t = yf.Ticker(symbol.upper())

    if start is not None:
        result = t.history(start=start, end=end, interval=interval)
    else:
        result = t.history(period=period, interval=interval)
    logger.debug(
        "get_history symbol=%s interval=%s bars=%d source=api", symbol, interval, len(result)
    )
    return result


def clear(symbol: str | None = None) -> None:
    """Clear stored price data for a symbol or all symbols."""
    _init()
    if _fetcher is not None:
        _fetcher.clear() if symbol is None else _fetcher.cache.clear(symbol)
    if _intraday_cache is not None:
        _intraday_cache.clear()
        logger.debug("intraday_cache_cleared")


def get_intraday_cache_stats() -> dict[str, int]:
    """Get intraday cache statistics (for debugging/monitoring)."""
    if _intraday_cache is None:
        return {"hits": 0, "misses": 0, "entries": 0}
    return _intraday_cache.stats()
