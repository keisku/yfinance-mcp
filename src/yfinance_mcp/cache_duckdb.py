"""DuckDB cache backend implementation."""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .helpers import normalize_tz

logger = logging.getLogger("yfinance_mcp.cache.duckdb")


def get_cache_path() -> Path:
    """Get the cache database path."""
    default = Path.home() / ".cache" / "yfinance-mcp" / "market.duckdb"
    path = Path(os.getenv("YFINANCE_CACHE_DB", default))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class DuckDBCacheBackend:
    """DuckDB-backed cache for OHLCV price data."""

    def __init__(self, db_path: Path | None = None):
        # avoids hard dependency when using other backends
        import duckdb

        self._duckdb = duckdb
        self.db_path = db_path or get_cache_path()
        self._conn: Any = None
        self._init_db()
        logger.debug("duckdb_init path=%s", self.db_path)

    def _get_conn(self) -> Any:
        """Get or create database connection."""
        if self._conn is None:
            logger.debug("duckdb_connect path=%s", self.db_path)
            self._conn = self._duckdb.connect(str(self.db_path))
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL DEFAULT '1d',
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (symbol, interval, date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_symbol_interval_date 
            ON prices (symbol, interval, date)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS etf_expense (
                symbol VARCHAR PRIMARY KEY,
                exchange VARCHAR,
                expense_ratio DOUBLE,
                source VARCHAR NOT NULL,
                fetched_at DATE NOT NULL
            )
        """)
        logger.debug("duckdb_schema_ready")

    def get_prices(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        """Get cached prices for a date range and interval."""
        conn = self._get_conn()
        df = conn.execute(
            """
            SELECT date, open, high, low, close, adj_close, volume 
            FROM prices 
            WHERE symbol = ? AND interval = ? AND date >= ? AND date <= ?
            ORDER BY date
        """,
            [symbol.upper(), interval, start, end],
        ).fetchdf()

        if not df.empty:
            df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
            df = normalize_tz(df)
            df = df.rename(
                columns={
                    "open": "o",
                    "high": "h",
                    "low": "l",
                    "close": "c",
                    "adj_close": "ac",
                    "volume": "v",
                }
            )
            logger.debug(
                "duckdb_get symbol=%s interval=%s range=%s..%s rows=%d",
                symbol.upper(),
                interval,
                start,
                end,
                len(df),
            )
        else:
            logger.debug(
                "duckdb_get symbol=%s interval=%s range=%s..%s rows=0",
                symbol.upper(),
                interval,
                start,
                end,
            )
        return df

    def store_prices(self, symbol: str, df: pd.DataFrame, interval: str = "1d") -> None:
        """Store price data in cache for a specific interval."""
        if df.empty:
            logger.debug("duckdb_store symbol=%s interval=%s skipped=empty", symbol, interval)
            return

        conn = self._get_conn()
        symbol = symbol.upper()

        records = []
        for idx, row in df.iterrows():
            dt = idx.date() if hasattr(idx, "date") else idx
            close_val = float(row.get("Close", row.get("c", 0)))
            adj_close_val = float(row.get("Adj Close", row.get("ac", close_val)))
            records.append(
                (
                    symbol,
                    interval,
                    dt,
                    float(row.get("Open", row.get("o", 0))),
                    float(row.get("High", row.get("h", 0))),
                    float(row.get("Low", row.get("l", 0))),
                    close_val,
                    adj_close_val,
                    int(row.get("Volume", row.get("v", 0))),
                )
            )

        conn.executemany(
            """
            INSERT OR REPLACE INTO prices 
            (symbol, interval, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            records,
        )
        logger.debug("duckdb_store symbol=%s interval=%s rows=%d", symbol, interval, len(records))

    def clear(self, symbol: str | None = None, interval: str | None = None) -> None:
        """Clear cache for a symbol/interval or all data."""
        conn = self._get_conn()
        if symbol and interval:
            conn.execute(
                "DELETE FROM prices WHERE symbol = ? AND interval = ?",
                [symbol.upper(), interval],
            )
            logger.info("duckdb_clear symbol=%s interval=%s", symbol.upper(), interval)
        elif symbol:
            conn.execute("DELETE FROM prices WHERE symbol = ?", [symbol.upper()])
            logger.info("duckdb_clear symbol=%s", symbol.upper())
        else:
            conn.execute("DELETE FROM prices")
            logger.info("duckdb_clear all=true")

    def get_etf_expense(self, symbol: str, ttl_days: int = 1) -> tuple[float | None, bool]:
        """Get cached ETF expense ratio if not expired.

        Returns:
            Tuple of (expense_ratio, found). If found is False, cache miss.
            If found is True but expense_ratio is None, the value was cached as unavailable.
        """
        conn = self._get_conn()
        result = conn.execute(
            """
            SELECT expense_ratio, fetched_at FROM etf_expense
            WHERE symbol = ?
            """,
            [symbol.upper()],
        ).fetchone()

        if result is None:
            logger.debug("duckdb_etf_expense_miss symbol=%s", symbol.upper())
            return None, False

        expense_ratio, fetched_at = result
        if (date.today() - fetched_at).days >= ttl_days:
            conn.execute("DELETE FROM etf_expense WHERE symbol = ?", [symbol.upper()])
            logger.debug("duckdb_etf_expense_expired symbol=%s", symbol.upper())
            return None, False

        logger.debug("duckdb_etf_expense_hit symbol=%s value=%s", symbol.upper(), expense_ratio)
        return expense_ratio, True

    def store_etf_expense(
        self,
        symbol: str,
        expense_ratio: float | None,
        exchange: str | None = None,
        source: str = "yahoo_japan",
    ) -> None:
        """Store ETF expense ratio in cache."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO etf_expense (symbol, exchange, expense_ratio, source, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [symbol.upper(), exchange, expense_ratio, source, date.today()],
        )
        logger.debug(
            "duckdb_etf_expense_store symbol=%s exchange=%s value=%s source=%s",
            symbol.upper(),
            exchange,
            expense_ratio,
            source,
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("duckdb_close")
