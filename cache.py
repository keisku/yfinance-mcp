"""Parquet cache for OHLCV data, using DuckDB as query engine."""

import os
import time
from datetime import date
from pathlib import Path

import duckdb

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "yfinance-mcp"


def _cache_dir() -> Path:
    return Path(os.getenv("YFINANCE_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))


class Cache:
    """Read/write unadjusted OHLCV bars from a single Parquet file.

    DuckDB is used in-memory to query and merge data.
    File paths are embedded in SQL (not parameterized) because DuckDB does
    not support ``?`` placeholders inside ``read_parquet()`` / ``COPY TO``.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or _cache_dir() / "unadjusted_ohlc.parquet"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _exists(self) -> bool:
        return self._path.exists() and self._path.stat().st_size > 0

    @property
    def _pq(self) -> str:
        """Escaped path string for embedding in SQL."""
        return str(self._path).replace("'", "''")

    def cached_dates(
        self, symbol: str, interval: str, start: date, end: date
    ) -> set[date]:
        """Return the set of dates present in cache for the given range."""
        if not self._exists():
            return set()
        conn = duckdb.connect()
        try:
            rows = conn.execute(
                f"SELECT date FROM read_parquet('{self._pq}') "
                "WHERE symbol = ? AND interval = ? AND date >= ? AND date <= ?",
                [symbol.upper(), interval, start, end],
            ).fetchall()
            return {row[0] for row in rows}
        finally:
            conn.close()

    def get(self, symbol: str, interval: str, start: date, end: date) -> list[tuple]:
        """Return cached rows for the given range as list of (date, o, h, l, c, v)."""
        if not self._exists():
            return []
        conn = duckdb.connect()
        try:
            return conn.execute(
                f"SELECT date, o, h, l, c, v FROM read_parquet('{self._pq}') "
                "WHERE symbol = ? AND interval = ? AND date >= ? AND date <= ? "
                "AND v != -1 "
                "ORDER BY date",
                [symbol.upper(), interval, start, end],
            ).fetchall()
        finally:
            conn.close()

    def put(self, symbol: str, interval: str, rows: list[tuple]) -> None:
        """Merge rows into the Parquet file.

        Each row is (date, o, h, l, c, v).  Duplicates are resolved by
        keeping the new row.
        """
        if not rows:
            return

        symbol = symbol.upper()
        conn = duckdb.connect()
        try:
            conn.execute(
                """
                CREATE TABLE new_rows (
                    symbol  VARCHAR,
                    interval VARCHAR,
                    date    DATE,
                    o       DOUBLE,
                    h       DOUBLE,
                    l       DOUBLE,
                    c       DOUBLE,
                    v       BIGINT
                )
                """
            )
            conn.executemany(
                "INSERT INTO new_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(symbol, interval, *r) for r in rows],
            )

            if self._exists():
                conn.execute(
                    f"""
                    COPY (
                        SELECT * FROM read_parquet('{self._pq}')
                        WHERE NOT (symbol = ? AND interval = ? AND date IN (
                            SELECT date FROM new_rows
                        ))
                        UNION ALL
                        SELECT * FROM new_rows
                        ORDER BY symbol, interval, date
                    ) TO '{self._pq}' (FORMAT PARQUET)
                    """,
                    [symbol, interval],
                )
            else:
                conn.execute(
                    f"""
                    COPY (
                        SELECT * FROM new_rows ORDER BY symbol, interval, date
                    ) TO '{self._pq}' (FORMAT PARQUET)
                    """
                )
        finally:
            conn.close()

    def close(self) -> None:
        """No-op — DuckDB connections are transient."""


class AdjustedCache:
    """TTL-based Parquet cache for adjusted OHLCV data.

    Uses absolute TTL — entries expire based on creation time,
    not last access time.  Each entry is keyed by
    (symbol, interval, start_date, end_date).
    """

    def __init__(self, path: Path | None = None, ttl: int = 3600):
        self._path = path or _cache_dir() / "adjusted_ohlc.parquet"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl

    def _exists(self) -> bool:
        return self._path.exists() and self._path.stat().st_size > 0

    @property
    def _pq(self) -> str:
        """Escaped path string for embedding in SQL."""
        return str(self._path).replace("'", "''")

    def get(
        self, symbol: str, interval: str, start: str, end: str
    ) -> list[tuple] | None:
        """Return cached rows if TTL not expired, else None.

        Each row is (ts_epoch, o, h, l, c, v, tz).
        Expired entries are evicted from the file as a side effect.
        """
        if not self._exists():
            return None
        conn = duckdb.connect()
        try:
            cutoff = time.time() - self._ttl
            rows = conn.execute(
                f"SELECT ts, o, h, l, c, v, tz FROM read_parquet('{self._pq}') "
                "WHERE symbol = ? AND interval = ? AND start_date = ? AND end_date = ? "
                "AND created_at > ? "
                "ORDER BY ts",
                [symbol.upper(), interval, start, end, cutoff],
            ).fetchall()
            self._evict_expired(conn, cutoff)
            return rows if rows else None
        finally:
            conn.close()

    def _evict_expired(self, conn: duckdb.DuckDBPyConnection, cutoff: float) -> None:
        """Remove expired entries from the Parquet file."""
        if not self._exists():
            return
        has_expired = conn.execute(
            f"SELECT 1 FROM read_parquet('{self._pq}') WHERE created_at <= ? LIMIT 1",
            [cutoff],
        ).fetchone()
        if not has_expired:
            return
        remaining = conn.execute(
            f"SELECT count(*) FROM read_parquet('{self._pq}') WHERE created_at > ?",
            [cutoff],
        ).fetchone()[0]
        if remaining > 0:
            conn.execute(
                f"""
                COPY (
                    SELECT * FROM read_parquet('{self._pq}')
                    WHERE created_at > ?
                    ORDER BY symbol, interval, start_date, end_date, ts
                ) TO '{self._pq}' (FORMAT PARQUET)
                """,
                [cutoff],
            )
        else:
            self._path.unlink(missing_ok=True)

    def put(
        self, symbol: str, interval: str, start: str, end: str, rows: list[tuple]
    ) -> None:
        """Store adjusted OHLCV rows with current timestamp.

        Each row is (ts_epoch, o, h, l, c, v, tz).
        Replaces any existing entry for the same key and evicts expired entries.
        """
        if not rows:
            return
        symbol = symbol.upper()
        now = time.time()
        conn = duckdb.connect()
        try:
            conn.execute(
                """
                CREATE TABLE new_rows (
                    symbol     VARCHAR,
                    interval   VARCHAR,
                    start_date VARCHAR,
                    end_date   VARCHAR,
                    created_at DOUBLE,
                    ts         DOUBLE,
                    o          DOUBLE,
                    h          DOUBLE,
                    l          DOUBLE,
                    c          DOUBLE,
                    v          BIGINT,
                    tz         VARCHAR
                )
                """
            )
            conn.executemany(
                "INSERT INTO new_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(symbol, interval, start, end, now, *r) for r in rows],
            )

            if self._exists():
                conn.execute(
                    f"""
                    COPY (
                        SELECT * FROM read_parquet('{self._pq}')
                        WHERE NOT (
                            symbol = ? AND interval = ? AND start_date = ? AND end_date = ?
                        )
                        AND created_at > ?
                        UNION ALL
                        SELECT * FROM new_rows
                        ORDER BY symbol, interval, start_date, end_date, ts
                    ) TO '{self._pq}' (FORMAT PARQUET)
                    """,
                    [symbol, interval, start, end, now - self._ttl],
                )
            else:
                conn.execute(
                    f"""
                    COPY (
                        SELECT * FROM new_rows
                        ORDER BY symbol, interval, start_date, end_date, ts
                    ) TO '{self._pq}' (FORMAT PARQUET)
                    """
                )
        finally:
            conn.close()

    def close(self) -> None:
        """No-op — DuckDB connections are transient."""
