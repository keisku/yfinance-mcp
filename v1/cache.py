"""Parquet cache for unadjusted OHLCV data, using DuckDB as query engine."""

import os
from datetime import date
from pathlib import Path

import duckdb

_DEFAULT_PATH = Path.home() / ".cache" / "yfinance-mcp" / "unadjusted_ohlc.parquet"


class Cache:
    """Read/write unadjusted OHLCV bars from a single Parquet file.

    DuckDB is used in-memory to query and merge data.
    File paths are embedded in SQL (not parameterized) because DuckDB does
    not support ``?`` placeholders inside ``read_parquet()`` / ``COPY TO``.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or Path(os.getenv("YFINANCE_CACHE_PATH", _DEFAULT_PATH))
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _exists(self) -> bool:
        return self._path.exists() and self._path.stat().st_size > 0

    @property
    def _pq(self) -> str:
        """Escaped path string for embedding in SQL."""
        return str(self._path).replace("'", "''")

    def cached_dates(self, symbol: str, interval: str, start: date, end: date) -> set[date]:
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
