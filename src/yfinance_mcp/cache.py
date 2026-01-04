"""Pluggable cache layer for Yahoo Finance data."""

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from .helpers import OHLCV_COLS_TO_SHORT, PERIOD_DELTAS, normalize_tz

logger = logging.getLogger("yfinance_mcp.cache")

_cache_stats = {"hits": 0, "misses": 0}


def get_cache_stats() -> dict[str, int]:
    """Get current cache hit/miss counts."""
    return _cache_stats.copy()


def reset_cache_stats() -> None:
    """Reset cache stats (for testing)."""
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0


class PriceCacheBackend(Protocol):
    """Protocol for price cache backends."""

    def get_prices(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        """Get cached prices for a date range and interval."""
        ...

    def store_prices(self, symbol: str, df: pd.DataFrame, interval: str = "1d") -> None:
        """Store price data in cache for a specific interval."""
        ...

    def clear(self, symbol: str | None = None) -> None:
        """Clear cache for a symbol or all data."""
        ...

    def close(self) -> None:
        """Close any connections."""
        ...


class NullCacheBackend:
    """No-op cache backend - always returns empty/None."""

    def get_prices(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        return pd.DataFrame()

    def store_prices(self, symbol: str, df: pd.DataFrame, interval: str = "1d") -> None:
        pass

    def clear(self, symbol: str | None = None) -> None:
        pass

    def close(self) -> None:
        pass


def create_cache_backend() -> PriceCacheBackend:
    """Create a cache backend based on configuration."""
    if os.getenv("YFINANCE_CACHE_DISABLED", "").lower() in ("1", "true", "yes"):
        logger.info("cache_disabled using NullCacheBackend")
        return NullCacheBackend()

    from .cache_duckdb import DuckDBCacheBackend

    backend = DuckDBCacheBackend()
    logger.info("cache_enabled backend=DuckDB path=%s", backend.db_path)
    return backend


class CachedPriceFetcher:
    """Price fetcher with pluggable caching backend."""

    def __init__(
        self,
        backend: PriceCacheBackend | None = None,
        db_path: Path | None = None,
    ):
        if backend is not None:
            self.cache = backend
        elif db_path is not None:
            from .cache_duckdb import DuckDBCacheBackend

            self.cache = DuckDBCacheBackend(db_path)
        else:
            self.cache = create_cache_backend()

    def fetch(self, symbol: str, period: str) -> int:
        """Fetch price data, using cache when possible."""
        df = self.get_history(symbol, period)
        return len(df)

    def get_history_by_dates(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get price history for a specific date range."""
        symbol = symbol.upper()
        today = date.today()

        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end) if end else today

        return self._get_history_internal(symbol, start_date, end_date, interval)

    def get_history(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get price history, fetching only missing data from API."""
        symbol = symbol.upper()
        today = date.today()

        # Calculate date range from period
        start_date, end_date = self._period_to_dates(period, today)

        return self._get_history_internal(symbol, start_date, end_date, interval)

    def _get_history_internal(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Internal method to get price history for a date range."""
        today = date.today()
        start_time = time.time()
        logger.debug(
            "cache_get_history symbol=%s interval=%s range=%s..%s",
            symbol,
            interval,
            start_date,
            end_date,
        )

        cached_df = self.cache.get_prices(symbol, start_date, end_date, interval)

        if cached_df.empty:
            _cache_stats["misses"] += 1
            logger.debug(
                "cache_miss symbol=%s interval=%s reason=no_cache range=%s..%s",
                symbol,
                interval,
                start_date,
                end_date,
            )
            df = self._fetch_from_api(symbol, start_date, end_date, interval)
            self.cache.store_prices(symbol, df, interval)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                "cache_fetch_complete symbol=%s interval=%s bars=%d elapsed_ms=%.1f source=api",
                symbol,
                interval,
                len(df),
                elapsed_ms,
            )
            return self._format_output(df)

        _cache_stats["hits"] += 1

        if hasattr(cached_df.index, "date"):
            cached_dates = set(cached_df.index.date)
        else:
            cached_dates = set(pd.to_datetime(cached_df.index).date)

        cached_start = min(cached_dates)
        cached_end = max(cached_dates)

        logger.debug(
            "cache_hit symbol=%s interval=%s bars=%d cached=%s..%s requested=%s..%s",
            symbol,
            interval,
            len(cached_df),
            cached_start,
            cached_end,
            start_date,
            end_date,
        )

        result_parts = [cached_df]
        fetched_any = False

        # early gap: requested start is before cached start
        should_fetch_early = start_date < cached_start
        if should_fetch_early and interval == "1wk":
            gap_days = (cached_start - start_date).days
            if gap_days < 7:
                should_fetch_early = False
        elif should_fetch_early and interval == "1mo":
            same_month = (
                start_date.year == cached_start.year and start_date.month == cached_start.month
            )
            if same_month:
                should_fetch_early = False

        if should_fetch_early:
            fetched_any = True
            logger.debug(
                "cache_fetch_early symbol=%s interval=%s range=%s..%s",
                symbol,
                interval,
                start_date,
                cached_start - timedelta(days=1),
            )
            early_df = self._fetch_from_api(
                symbol, start_date, cached_start - timedelta(days=1), interval
            )
            if not early_df.empty:
                self.cache.store_prices(symbol, early_df, interval)
                result_parts.append(early_df)

        # interior gaps: missing dates within cached range (daily interval only)
        if interval == "1d":
            gaps = self._find_gaps(cached_dates, cached_start, cached_end)
            for gap_start, gap_end in gaps:
                fetched_any = True
                logger.debug(
                    "cache_fetch_gap symbol=%s interval=%s range=%s..%s",
                    symbol,
                    interval,
                    gap_start,
                    gap_end,
                )
                gap_df = self._fetch_from_api(symbol, gap_start, gap_end, interval)
                if not gap_df.empty:
                    self.cache.store_prices(symbol, gap_df, interval)
                    result_parts.append(gap_df)

        # late gap: requested end is after cached end
        if end_date > cached_end and self._has_completed_periods_since(cached_end, today, interval):
            fetched_any = True
            fetch_start = cached_end + timedelta(days=1)
            logger.debug(
                "cache_fetch_late symbol=%s interval=%s range=%s..%s",
                symbol,
                interval,
                fetch_start,
                end_date,
            )
            late_df = self._fetch_from_api(symbol, fetch_start, end_date, interval)
            if not late_df.empty:
                self.cache.store_prices(symbol, late_df, interval)
                result_parts.append(late_df)

        df = pd.concat(result_parts)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        if hasattr(df.index, "date"):
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        else:
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        df = df[mask]

        elapsed_ms = (time.time() - start_time) * 1000
        source = "cache" if not fetched_any else "cache+api"
        logger.debug(
            "cache_fetch_complete symbol=%s bars=%d ms=%.1f src=%s",
            symbol,
            len(df),
            elapsed_ms,
            source,
        )
        return self._format_output(df)

    def _period_to_dates(self, period: str, today: date) -> tuple[date, date]:
        """Convert period string to start/end dates."""
        end_date = today

        if period == "ytd":
            start_date = date(today.year, 1, 1)
        elif period in PERIOD_DELTAS:
            start_date = today - PERIOD_DELTAS[period]
        else:
            # Default to 1 month
            start_date = today - relativedelta(months=1)

        return start_date, end_date

    def _fetch_from_api(self, symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
        """Fetch data from yfinance API with retry and exponential backoff."""
        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                logger.debug(
                    "api_fetch symbol=%s range=%s..%s attempt=%d/%d",
                    symbol,
                    start,
                    end,
                    attempt + 1,
                    max_retries,
                )
                fetch_start = time.time()
                t = yf.Ticker(symbol)
                # yfinance end date is exclusive
                df = t.history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    interval=interval,
                )
                df = normalize_tz(df)
                if not df.empty:
                    df = df.rename(columns=OHLCV_COLS_TO_SHORT)
                    df = df[[c for c in ["o", "h", "l", "c", "v"] if c in df.columns]]
                elapsed_ms = (time.time() - fetch_start) * 1000
                logger.debug(
                    "api_fetch_success symbol=%s bars=%d elapsed_ms=%.1f",
                    symbol,
                    len(df),
                    elapsed_ms,
                )
                return df
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(
                        "api_fetch_failed symbol=%s attempts=%d error=%s",
                        symbol,
                        max_retries,
                        e,
                    )
                    return pd.DataFrame()

                delay = base_delay * (2**attempt)
                logger.debug(
                    "api_fetch_retry symbol=%s attempt=%d delay=%.1fs error=%s",
                    symbol,
                    attempt + 1,
                    delay,
                    e,
                )
                time.sleep(delay)

        return pd.DataFrame()

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format output with standard column names."""
        if df.empty:
            return df

        df = df.rename(columns=OHLCV_COLS_TO_SHORT)
        keep_cols = ["o", "h", "l", "c", "v"]
        df = df[[c for c in keep_cols if c in df.columns]]
        return df

    def _find_gaps(
        self,
        cached_dates: set[date],
        start: date,
        end: date,
    ) -> list[tuple[date, date]]:
        """Find date ranges not covered by cache.

        Skips weekends but does not use holiday calendars.
        Trusts the API to return actual trading days for each market.
        """
        if not cached_dates:
            return [(start, end)]

        sorted_dates = sorted(cached_dates)
        gaps: list[tuple[date, date]] = []

        for i in range(len(sorted_dates) - 1):
            curr_date = sorted_dates[i]
            next_date = sorted_dates[i + 1]

            gap_start = curr_date + timedelta(days=1)
            gap_end = next_date - timedelta(days=1)

            if gap_start > gap_end:
                continue

            has_weekday = any(
                (gap_start + timedelta(days=d)).weekday() < 5
                for d in range((gap_end - gap_start).days + 1)
            )

            if has_weekday:
                gaps.append((gap_start, gap_end))

        return gaps

    def _has_completed_periods_since(self, cached_end: date, today: date, interval: str) -> bool:
        """Check if there are any completed periods after cached_end."""
        yesterday = today - timedelta(days=1)
        if cached_end >= yesterday:
            return False

        if interval == "1wk":
            return (today - cached_end).days >= 7
        elif interval == "1mo":
            return today.month != cached_end.month or today.year != cached_end.year
        else:
            current = cached_end + timedelta(days=1)
            while current <= yesterday:
                if current.weekday() < 5:
                    return True
                current += timedelta(days=1)
            return False

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()

    def close(self) -> None:
        """Close cache connection."""
        self.cache.close()
