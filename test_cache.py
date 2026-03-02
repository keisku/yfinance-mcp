"""Tests for gap detection and Parquet cache."""

from datetime import date
from unittest.mock import patch

import pytest
from cache import AdjustedCache, Cache
from history import _find_gaps


class TestFindGaps:
    """Tests for the _find_gaps() pure function."""

    def test_empty_cache(self):
        """No cached dates → single gap spanning the whole range."""
        gaps = _find_gaps(date(2025, 1, 6), date(2025, 1, 10), set())
        assert gaps == [(date(2025, 1, 6), date(2025, 1, 10))]

    def test_full_cache(self):
        """All weekdays cached → no gaps."""
        cached = {
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
            date(2025, 1, 9),
            date(2025, 1, 10),
        }
        gaps = _find_gaps(date(2025, 1, 6), date(2025, 1, 10), cached)
        assert gaps == []

    def test_partial_first_half_cached(self):
        """First half cached, second half missing."""
        # Mon-Fri week: 6,7,8,9,10
        cached = {date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)}
        gaps = _find_gaps(date(2025, 1, 6), date(2025, 1, 10), cached)
        assert gaps == [(date(2025, 1, 9), date(2025, 1, 10))]

    def test_interior_hole(self):
        """Cached data with a hole in the middle."""
        # Mon 6, Tue 7, [gap Wed 8], Thu 9, Fri 10
        cached = {
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 9),
            date(2025, 1, 10),
        }
        gaps = _find_gaps(date(2025, 1, 6), date(2025, 1, 10), cached)
        assert gaps == [(date(2025, 1, 8), date(2025, 1, 8))]

    def test_skips_weekends(self):
        """Fri-to-Mon gap with no missing weekday → no gap."""
        # Fri 10 and Mon 13 both cached; Sat 11 and Sun 12 are weekends.
        cached = {date(2025, 1, 10), date(2025, 1, 13)}
        gaps = _find_gaps(date(2025, 1, 10), date(2025, 1, 13), cached)
        assert gaps == []

    def test_multiple_holes(self):
        """Multiple non-contiguous gaps."""
        # Mon 6 .. Fri 17 (two weeks)
        # Cached: Mon 6, Tue 7, Thu 9, Fri 10, Wed 15, Thu 16, Fri 17
        # Missing: Wed 8, Mon 13, Tue 14
        cached = {
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 15),
            date(2025, 1, 16),
            date(2025, 1, 17),
        }
        gaps = _find_gaps(date(2025, 1, 6), date(2025, 1, 17), cached)
        assert gaps == [
            (date(2025, 1, 8), date(2025, 1, 8)),
            (date(2025, 1, 13), date(2025, 1, 14)),
        ]

    def test_weekend_only_range(self):
        """Range that covers only a weekend → no gaps."""
        gaps = _find_gaps(date(2025, 1, 11), date(2025, 1, 12), set())
        assert gaps == []

    def test_single_day_cached(self):
        """Single weekday, cached → no gap."""
        gaps = _find_gaps(date(2025, 1, 6), date(2025, 1, 6), {date(2025, 1, 6)})
        assert gaps == []

    def test_single_day_missing(self):
        """Single weekday, not cached → one gap."""
        gaps = _find_gaps(date(2025, 1, 6), date(2025, 1, 6), set())
        assert gaps == [(date(2025, 1, 6), date(2025, 1, 6))]

    def test_second_half_cached(self):
        """Second half cached, first half missing."""
        cached = {date(2025, 1, 9), date(2025, 1, 10)}
        gaps = _find_gaps(date(2025, 1, 6), date(2025, 1, 10), cached)
        assert gaps == [(date(2025, 1, 6), date(2025, 1, 8))]

    def test_gap_closes_after_weekend(self):
        """Fri missing, Mon cached → gap end should be Fri, not Sun."""
        # Fri 10 missing, Mon 13 cached.
        cached = {date(2025, 1, 13)}
        gaps = _find_gaps(date(2025, 1, 10), date(2025, 1, 13), cached)
        assert len(gaps) == 1
        gap_start, gap_end = gaps[0]
        assert gap_start == date(2025, 1, 10)
        # gap_end must be a weekday (Fri), not Sun.
        assert gap_end.weekday() < 5

    def test_start_after_end(self):
        """start > end → no gaps."""
        gaps = _find_gaps(date(2025, 1, 10), date(2025, 1, 6), set())
        assert gaps == []


class TestCache:
    """Tests for the Parquet-backed Cache class."""

    def test_put_get_roundtrip(self, tmp_path):
        """Store rows and read them back."""
        cache = Cache(path=tmp_path / "test.parquet")
        rows = [
            # (date, o, h, l, c, v)
            (date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10),
            (date(2025, 1, 7), 3.0, 4.0, 2.5, 3.5, 20),
        ]
        cache.put("TEST", "1d", rows)

        result = cache.get("TEST", "1d", date(2025, 1, 6), date(2025, 1, 7))
        assert len(result) == 2
        assert result[0][0] == date(2025, 1, 6)
        assert result[0][1] == pytest.approx(1.0)
        assert result[1][5] == 20

    def test_put_merge_dedup(self, tmp_path):
        """Two overlapping puts keep the latest values."""
        cache = Cache(path=tmp_path / "test.parquet")

        cache.put(
            "TEST",
            "1d",
            [
                (date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10),
                (date(2025, 1, 7), 3.0, 4.0, 2.5, 3.5, 20),
            ],
        )

        # Overwrite Jan 7 with new values.
        cache.put(
            "TEST",
            "1d",
            [
                (date(2025, 1, 7), 5.0, 6.0, 4.5, 5.5, 30),
                (date(2025, 1, 8), 7.0, 8.0, 6.5, 7.5, 40),
            ],
        )

        result = cache.get("TEST", "1d", date(2025, 1, 6), date(2025, 1, 8))
        assert len(result) == 3
        # Jan 7 should have the updated values.
        jan7 = result[1]
        assert jan7[1] == pytest.approx(5.0)
        assert jan7[5] == 30

    def test_cached_dates(self, tmp_path):
        """cached_dates returns the correct set of dates."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put(
            "TEST",
            "1d",
            [
                (date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10),
                (date(2025, 1, 7), 3.0, 4.0, 2.5, 3.5, 20),
                (date(2025, 1, 8), 5.0, 6.0, 4.5, 5.5, 30),
            ],
        )

        dates = cache.cached_dates("TEST", "1d", date(2025, 1, 6), date(2025, 1, 10))
        assert dates == {date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)}

    def test_empty_cache_returns_empty(self, tmp_path):
        """get and cached_dates return empty results when file doesn't exist."""
        cache = Cache(path=tmp_path / "nonexistent.parquet")
        assert cache.get("TEST", "1d", date(2025, 1, 6), date(2025, 1, 10)) == []
        assert (
            cache.cached_dates("TEST", "1d", date(2025, 1, 6), date(2025, 1, 10))
            == set()
        )

    def test_different_symbols_isolated(self, tmp_path):
        """Different symbols are stored and retrieved independently."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put(
            "AAA",
            "1d",
            [
                (date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10),
            ],
        )
        cache.put(
            "BBB",
            "1d",
            [
                (date(2025, 1, 6), 11.0, 12.0, 10.5, 11.5, 110),
            ],
        )

        aaa = cache.get("AAA", "1d", date(2025, 1, 6), date(2025, 1, 6))
        bbb = cache.get("BBB", "1d", date(2025, 1, 6), date(2025, 1, 6))
        assert len(aaa) == 1
        assert len(bbb) == 1
        assert aaa[0][1] == pytest.approx(1.0)
        assert bbb[0][1] == pytest.approx(11.0)

    def test_different_intervals_isolated(self, tmp_path):
        """Different intervals for the same symbol are isolated."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put(
            "TEST",
            "1d",
            [
                (date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10),
            ],
        )
        cache.put(
            "TEST",
            "1wk",
            [
                (date(2025, 1, 6), 11.0, 12.0, 10.5, 11.5, 110),
            ],
        )

        daily = cache.get("TEST", "1d", date(2025, 1, 6), date(2025, 1, 6))
        weekly = cache.get("TEST", "1wk", date(2025, 1, 6), date(2025, 1, 6))
        assert daily[0][1] == pytest.approx(1.0)
        assert weekly[0][1] == pytest.approx(11.0)

    def test_case_insensitive_symbol(self, tmp_path):
        """put with lowercase, get with uppercase should hit."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put("test", "1d", [(date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10)])
        result = cache.get("TEST", "1d", date(2025, 1, 6), date(2025, 1, 6))
        assert len(result) == 1

    def test_put_empty_rows(self, tmp_path):
        """put with empty list should not create a file or error."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put("TEST", "1d", [])
        assert not (tmp_path / "test.parquet").exists()
        assert cache.get("TEST", "1d", date(2025, 1, 6), date(2025, 1, 10)) == []

    def test_get_narrower_range(self, tmp_path):
        """Put 3 days, get only the middle day."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put(
            "TEST",
            "1d",
            [
                (date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10),
                (date(2025, 1, 7), 3.0, 4.0, 2.5, 3.5, 20),
                (date(2025, 1, 8), 5.0, 6.0, 4.5, 5.5, 30),
            ],
        )
        result = cache.get("TEST", "1d", date(2025, 1, 7), date(2025, 1, 7))
        assert len(result) == 1
        assert result[0][0] == date(2025, 1, 7)

    def test_symbol_with_dot(self, tmp_path):
        """Dotted symbol like 'FOO.X' works correctly."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put("FOO.X", "1d", [(date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10)])
        result = cache.get("FOO.X", "1d", date(2025, 1, 6), date(2025, 1, 6))
        assert len(result) == 1

    def test_holiday_sentinel_excluded_from_get(self, tmp_path):
        """Rows with v=-1 (holiday markers) are not returned by get()."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put(
            "TEST",
            "1d",
            [
                (date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10),
                (date(2025, 1, 7), 0, 0, 0, 0, -1),
                (date(2025, 1, 8), 3.0, 4.0, 2.5, 3.5, 20),
            ],
        )
        result = cache.get("TEST", "1d", date(2025, 1, 6), date(2025, 1, 8))
        assert len(result) == 2
        assert result[0][0] == date(2025, 1, 6)
        assert result[1][0] == date(2025, 1, 8)

    def test_holiday_sentinel_included_in_cached_dates(self, tmp_path):
        """Holiday markers (v=-1) count as cached so gaps are not re-fetched."""
        cache = Cache(path=tmp_path / "test.parquet")
        cache.put(
            "TEST",
            "1d",
            [
                (date(2025, 1, 6), 1.0, 2.0, 0.5, 1.5, 10),
                (date(2025, 1, 7), 0, 0, 0, 0, -1),
            ],
        )
        dates = cache.cached_dates("TEST", "1d", date(2025, 1, 6), date(2025, 1, 7))
        assert dates == {date(2025, 1, 6), date(2025, 1, 7)}


# Sample rows for AdjustedCache: (ts_epoch, o, h, l, c, v, tz)
_T0 = 1_000_000.0

_ADJ_ROWS = [
    (_T0 + 0, 100.0, 105.0, 95.0, 102.0, 5000, "America/New_York"),
    (_T0 + 86400, 103.0, 108.0, 98.0, 106.0, 6000, "America/New_York"),
]


class TestAdjustedCache:
    """Tests for the TTL-based AdjustedCache class."""

    def test_put_get_roundtrip(self, tmp_path):
        cache = AdjustedCache(path=tmp_path / "adj.parquet")
        cache.put("TEST", "1d", "2025-01-06", "2025-01-08", _ADJ_ROWS)

        result = cache.get("TEST", "1d", "2025-01-06", "2025-01-08")
        assert result is not None
        assert len(result) == 2
        assert result[0][0] == pytest.approx(_T0)
        assert result[0][1] == pytest.approx(100.0)
        assert result[1][5] == 6000
        assert result[0][6] == "America/New_York"

    def test_empty_cache_returns_none(self, tmp_path):
        cache = AdjustedCache(path=tmp_path / "adj.parquet")
        assert cache.get("TEST", "1d", "2025-01-06", "2025-01-08") is None

    def test_put_empty_rows(self, tmp_path):
        cache = AdjustedCache(path=tmp_path / "adj.parquet")
        cache.put("TEST", "1d", "2025-01-06", "2025-01-08", [])
        assert not (tmp_path / "adj.parquet").exists()

    def test_different_symbols_isolated(self, tmp_path):
        cache = AdjustedCache(path=tmp_path / "adj.parquet")
        rows_a = [(_T0, 1.0, 2.0, 0.5, 1.5, 10, "UTC")]
        rows_b = [(_T0, 11.0, 12.0, 10.5, 11.5, 110, "UTC")]
        cache.put("AAA", "1d", "2025-01-06", "2025-01-07", rows_a)
        cache.put("BBB", "1d", "2025-01-06", "2025-01-07", rows_b)

        aaa = cache.get("AAA", "1d", "2025-01-06", "2025-01-07")
        bbb = cache.get("BBB", "1d", "2025-01-06", "2025-01-07")
        assert aaa is not None and aaa[0][1] == pytest.approx(1.0)
        assert bbb is not None and bbb[0][1] == pytest.approx(11.0)

    def test_different_ranges_isolated(self, tmp_path):
        cache = AdjustedCache(path=tmp_path / "adj.parquet")
        rows_a = [(_T0, 1.0, 2.0, 0.5, 1.5, 10, "UTC")]
        rows_b = [(_T0, 11.0, 12.0, 10.5, 11.5, 110, "UTC")]
        cache.put("TEST", "1d", "2025-01-06", "2025-01-07", rows_a)
        cache.put("TEST", "1d", "2025-02-01", "2025-02-02", rows_b)

        a = cache.get("TEST", "1d", "2025-01-06", "2025-01-07")
        b = cache.get("TEST", "1d", "2025-02-01", "2025-02-02")
        assert a is not None and a[0][1] == pytest.approx(1.0)
        assert b is not None and b[0][1] == pytest.approx(11.0)

    def test_case_insensitive_symbol(self, tmp_path):
        cache = AdjustedCache(path=tmp_path / "adj.parquet")
        cache.put("test", "1d", "2025-01-06", "2025-01-07", _ADJ_ROWS)
        result = cache.get("TEST", "1d", "2025-01-06", "2025-01-07")
        assert result is not None
        assert len(result) == 2

    def test_put_replaces_same_key(self, tmp_path):
        cache = AdjustedCache(path=tmp_path / "adj.parquet")
        old_rows = [(_T0, 1.0, 2.0, 0.5, 1.5, 10, "UTC")]
        new_rows = [(_T0, 99.0, 99.0, 99.0, 99.0, 99, "UTC")]

        cache.put("TEST", "1d", "2025-01-06", "2025-01-07", old_rows)
        cache.put("TEST", "1d", "2025-01-06", "2025-01-07", new_rows)

        result = cache.get("TEST", "1d", "2025-01-06", "2025-01-07")
        assert result is not None
        assert len(result) == 1
        assert result[0][1] == pytest.approx(99.0)

    @patch("cache.time.time")
    def test_ttl_expiry(self, mock_time, tmp_path):
        """Entry expires after TTL elapses."""
        cache = AdjustedCache(path=tmp_path / "adj.parquet", ttl=60)

        mock_time.return_value = 1000.0
        cache.put("TEST", "1d", "2025-01-06", "2025-01-07", _ADJ_ROWS)

        # At 1059s (59s later) — still valid.
        mock_time.return_value = 1059.0
        assert cache.get("TEST", "1d", "2025-01-06", "2025-01-07") is not None

        # At 1060s (exactly TTL) — expired.
        mock_time.return_value = 1060.0
        assert cache.get("TEST", "1d", "2025-01-06", "2025-01-07") is None

    @patch("cache.time.time")
    def test_ttl_is_absolute_not_sliding(self, mock_time, tmp_path):
        """Accessing the cache does NOT extend the TTL."""
        cache = AdjustedCache(path=tmp_path / "adj.parquet", ttl=60)

        mock_time.return_value = 1000.0
        cache.put("TEST", "1d", "2025-01-06", "2025-01-07", _ADJ_ROWS)

        # Access at 1030s — hit, but does NOT reset the timer.
        mock_time.return_value = 1030.0
        assert cache.get("TEST", "1d", "2025-01-06", "2025-01-07") is not None

        # At 1060s — expired based on original put time (1000), not last get (1030).
        mock_time.return_value = 1060.0
        assert cache.get("TEST", "1d", "2025-01-06", "2025-01-07") is None

    @patch("cache.time.time")
    def test_expired_entries_evicted_on_put(self, mock_time, tmp_path):
        """Expired entries for other keys are cleaned up during put."""
        cache = AdjustedCache(path=tmp_path / "adj.parquet", ttl=60)

        mock_time.return_value = 1000.0
        cache.put(
            "OLD",
            "1d",
            "2025-01-01",
            "2025-01-02",
            [(_T0, 1.0, 2.0, 0.5, 1.5, 10, "UTC")],
        )

        # 120s later, OLD is expired. Put a new entry — OLD should be evicted.
        mock_time.return_value = 1120.0
        cache.put(
            "NEW",
            "1d",
            "2025-02-01",
            "2025-02-02",
            [(_T0, 9.0, 9.0, 9.0, 9.0, 99, "UTC")],
        )

        # NEW is fresh.
        assert cache.get("NEW", "1d", "2025-02-01", "2025-02-02") is not None
        # OLD is gone.
        assert cache.get("OLD", "1d", "2025-01-01", "2025-01-02") is None
