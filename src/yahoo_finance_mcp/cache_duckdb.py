"""DuckDB cache backend implementation."""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("yf-mcp.cache.duckdb")


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
        # migrate from old schema without interval column
        try:
            conn.execute("SELECT interval FROM prices LIMIT 1")
        except Exception:
            conn.execute("DROP TABLE IF EXISTS prices")
            logger.info("duckdb_schema_migrated dropping old table without interval")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL DEFAULT '1d',
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (symbol, interval, date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_symbol_interval_date 
            ON prices (symbol, interval, date)
        """)
        logger.debug("duckdb_schema_ready")

    def get_cached_range(
        self, symbol: str, interval: str = "1d"
    ) -> tuple[date | None, date | None]:
        """Get the date range we have cached for a symbol and interval."""
        conn = self._get_conn()
        result = conn.execute(
            """
            SELECT MIN(date), MAX(date) FROM prices 
            WHERE symbol = ? AND interval = ?
        """,
            [symbol.upper(), interval],
        ).fetchone()
        if result and result[0]:
            logger.debug(
                "duckdb_range symbol=%s interval=%s start=%s end=%s",
                symbol.upper(),
                interval,
                result[0],
                result[1],
            )
            return result[0], result[1]
        logger.debug("duckdb_range symbol=%s interval=%s empty=true", symbol.upper(), interval)
        return None, None

    def get_prices(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        """Get cached prices for a date range and interval."""
        conn = self._get_conn()
        df = conn.execute(
            """
            SELECT date, open, high, low, close, volume 
            FROM prices 
            WHERE symbol = ? AND interval = ? AND date >= ? AND date <= ?
            ORDER BY date
        """,
            [symbol.upper(), interval, start, end],
        ).fetchdf()

        if not df.empty:
            df = df.set_index("date")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.rename(
                columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
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
            records.append(
                (
                    symbol,
                    interval,
                    dt,
                    float(row.get("Open", row.get("o", 0))),
                    float(row.get("High", row.get("h", 0))),
                    float(row.get("Low", row.get("l", 0))),
                    float(row.get("Close", row.get("c", 0))),
                    int(row.get("Volume", row.get("v", 0))),
                )
            )

        conn.executemany(
            """
            INSERT OR REPLACE INTO prices (symbol, interval, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("duckdb_close")
