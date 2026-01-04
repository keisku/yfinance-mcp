"""Tests for cache layer - gap detection and filling."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from yfinance_mcp.cache import CachedPriceFetcher, NullCacheBackend


@pytest.fixture
def fetcher():
    """Create fetcher with null backend for unit testing."""
    return CachedPriceFetcher(backend=NullCacheBackend())


class TestFindGaps:
    """Test _find_gaps method."""

    @pytest.mark.parametrize(
        "cached_dates,start,end,expected_gaps",
        [
            pytest.param(
                {date(2024, 6, 3)},
                date(2024, 6, 3),
                date(2024, 6, 3),
                [],
                id="single_day_no_gaps",
            ),
            pytest.param(
                set(),
                date(2024, 6, 3),
                date(2024, 6, 3),
                [(date(2024, 6, 3), date(2024, 6, 3))],
                id="empty_cache_single_day",
            ),
            pytest.param(
                set(),
                date(2024, 6, 3),
                date(2024, 6, 10),
                [(date(2024, 6, 3), date(2024, 6, 10))],
                id="empty_cache_range",
            ),
            pytest.param(
                {date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 5)},
                date(2024, 6, 3),
                date(2024, 6, 5),
                [],
                id="consecutive_days_no_gaps",
            ),
            pytest.param(
                {date(2024, 6, 3), date(2024, 6, 10)},
                date(2024, 6, 3),
                date(2024, 6, 10),
                [(date(2024, 6, 4), date(2024, 6, 9))],
                id="single_gap_middle",
            ),
            pytest.param(
                {date(2024, 6, 3), date(2024, 6, 6), date(2024, 6, 10)},
                date(2024, 6, 3),
                date(2024, 6, 10),
                [(date(2024, 6, 4), date(2024, 6, 5)), (date(2024, 6, 7), date(2024, 6, 9))],
                id="multiple_gaps",
            ),
            pytest.param(
                {date(2024, 10, 4), date(2024, 10, 7)},
                date(2024, 10, 4),
                date(2024, 10, 7),
                [],
                id="weekend_only_no_gap",
            ),
            pytest.param(
                {date(2023, 12, 29), date(2024, 1, 3)},
                date(2023, 12, 29),
                date(2024, 1, 3),
                [(date(2023, 12, 30), date(2024, 1, 2))],
                id="year_boundary",
            ),
            pytest.param(
                {date(2024, 2, 28), date(2024, 3, 1)},
                date(2024, 2, 28),
                date(2024, 3, 1),
                [(date(2024, 2, 29), date(2024, 2, 29))],
                id="leap_year_feb_29",
            ),
            pytest.param(
                {date(2023, 2, 28), date(2023, 3, 1)},
                date(2023, 2, 28),
                date(2023, 3, 1),
                [],
                id="non_leap_year_no_gap",
            ),
        ],
    )
    def test_find_gaps(self, fetcher, cached_dates, start, end, expected_gaps):
        gaps = fetcher._find_gaps(cached_dates, start, end)
        assert gaps == expected_gaps

    def test_find_gaps_ten_year_range(self, fetcher):
        """10-year range with sparse cache points produces gaps between each pair."""
        cached_dates = {
            date(2015, 1, 2),
            date(2015, 7, 1),
            date(2016, 1, 4),
            date(2016, 7, 1),
            date(2017, 1, 3),
            date(2017, 7, 3),
            date(2018, 1, 2),
            date(2018, 7, 2),
            date(2019, 1, 2),
            date(2019, 7, 1),
            date(2020, 1, 2),
            date(2020, 7, 1),
            date(2021, 1, 4),
            date(2021, 7, 1),
            date(2022, 1, 3),
            date(2022, 7, 1),
            date(2023, 1, 3),
            date(2023, 7, 3),
            date(2024, 1, 2),
            date(2024, 7, 1),
            date(2025, 1, 2),
        }
        gaps = fetcher._find_gaps(cached_dates, date(2015, 1, 2), date(2025, 1, 2))

        assert len(gaps) == 20
        assert all(start <= end for start, end in gaps)
        assert gaps[0] == (date(2015, 1, 3), date(2015, 6, 30))
        assert gaps[-1] == (date(2024, 7, 2), date(2025, 1, 1))


class TestCacheFillGaps:
    """Test that _get_history_internal fills gaps correctly."""

    @pytest.mark.parametrize(
        "interval,should_detect_gaps",
        [
            ("1d", True),
            ("1wk", False),
            ("1mo", False),
        ],
    )
    def test_gap_detection_interval(self, interval, should_detect_gaps):
        """Gap detection only applies to daily interval."""
        backend = MagicMock()
        apr = pd.DataFrame(
            {"o": [100], "h": [101], "l": [99], "c": [100.5], "v": [1000]},
            index=pd.to_datetime([date(2024, 4, 15)]),
        )
        oct = pd.DataFrame(
            {"o": [110], "h": [111], "l": [109], "c": [110.5], "v": [1100]},
            index=pd.to_datetime([date(2024, 10, 21)]),
        )
        backend.get_prices.return_value = pd.concat([apr, oct])
        backend.store_prices = MagicMock()

        fetcher = CachedPriceFetcher(backend=backend)

        with patch.object(fetcher, "_find_gaps", wraps=fetcher._find_gaps) as spy:
            with patch.object(fetcher, "_fetch_from_api", return_value=pd.DataFrame()):
                fetcher._get_history_internal(
                    "TEST", date(2024, 4, 15), date(2024, 10, 21), interval
                )

            assert spy.called == should_detect_gaps

    def test_no_api_call_when_fully_cached(self):
        """Should not call API when cache has all consecutive days."""
        backend = MagicMock()
        dates = pd.date_range(date(2024, 5, 6), date(2024, 5, 10), freq="D")
        cached_df = pd.DataFrame(
            {
                "o": [100] * len(dates),
                "h": [101] * len(dates),
                "l": [99] * len(dates),
                "c": [100.5] * len(dates),
                "v": [1000] * len(dates),
            },
            index=dates,
        )
        backend.get_prices.return_value = cached_df
        backend.store_prices = MagicMock()

        fetcher = CachedPriceFetcher(backend=backend)

        with patch.object(fetcher, "_fetch_from_api") as mock_api:
            fetcher._get_history_internal("TEST", date(2024, 5, 6), date(2024, 5, 10), "1d")
            mock_api.assert_not_called()


class TestGapFilling:
    """Test gap detection and filling behavior.

    These scenarios test the cache's ability to:
    - Detect gaps when API returns 0 bars (holidays, market closures)
    - Handle early/interior/late gap combinations
    - Merge cached and fetched data correctly
    - Store fetched data in cache
    """

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend with common setup."""
        backend = MagicMock()
        backend.store_prices = MagicMock()
        return backend

    @pytest.mark.parametrize(
        "request_start,request_end,cache_start,cache_end,min_api_calls",
        [
            pytest.param(
                date(2024, 8, 1),
                date(2025, 1, 3),
                date(2024, 10, 1),
                date(2024, 12, 31),
                2,
                id="early_and_late_gaps",
            ),
            pytest.param(
                date(2024, 10, 1),
                date(2024, 12, 31),
                date(2024, 10, 1),
                date(2024, 10, 31),
                1,
                id="late_gap_only",
            ),
            pytest.param(
                date(2024, 7, 1),
                date(2024, 9, 30),
                date(2024, 8, 1),
                date(2024, 9, 30),
                1,
                id="early_gap_only",
            ),
            pytest.param(
                date(2024, 7, 12),
                date(2024, 7, 16),
                date(2024, 7, 12),
                date(2024, 7, 12),
                1,
                id="interior_gap_api_returns_zero",
            ),
        ],
    )
    def test_gap_detection_triggers_api_calls(
        self, mock_backend, request_start, request_end, cache_start, cache_end, min_api_calls
    ):
        """Gap detection triggers API calls; system handles 0-bar responses."""
        cache_dates = pd.date_range(cache_start, cache_end, freq="B")
        mock_backend.get_prices.return_value = pd.DataFrame(
            {
                "o": [100] * len(cache_dates),
                "h": [101] * len(cache_dates),
                "l": [99] * len(cache_dates),
                "c": [100.5] * len(cache_dates),
                "v": [1000] * len(cache_dates),
            },
            index=cache_dates,
        )

        fetcher = CachedPriceFetcher(backend=mock_backend)
        api_calls = []

        def track_api_calls(symbol, start, end, interval):
            api_calls.append((start, end))
            return pd.DataFrame()  # API returns 0 bars (simulating holidays)

        with patch.object(fetcher, "_fetch_from_api", side_effect=track_api_calls):
            fetcher._get_history_internal("TEST", request_start, request_end, "1d")
            assert len(api_calls) >= min_api_calls

    @pytest.mark.parametrize(
        "cached_dates,gap_dates,expected_total",
        [
            pytest.param(
                [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
                [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)],
                7,
                id="4_cached_3_fetched",
            ),
            pytest.param(
                [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 5)],
                [date(2024, 6, 6), date(2024, 6, 7)],
                5,
                id="3_cached_2_fetched",
            ),
            pytest.param(
                [date(2024, 10, 1)],
                [date(2024, 10, 2), date(2024, 10, 3), date(2024, 10, 4)],
                4,
                id="1_cached_3_fetched",
            ),
        ],
    )
    def test_merges_cached_and_fetched_data(
        self, mock_backend, cached_dates, gap_dates, expected_total
    ):
        """Result correctly merges cached data with newly fetched data."""
        cached_df = pd.DataFrame(
            {
                "o": [100 + i for i in range(len(cached_dates))],
                "h": [101 + i for i in range(len(cached_dates))],
                "l": [99 + i for i in range(len(cached_dates))],
                "c": [100.5 + i for i in range(len(cached_dates))],
                "v": [1000] * len(cached_dates),
            },
            index=pd.to_datetime(cached_dates),
        )
        mock_backend.get_prices.return_value = cached_df

        gap_df = pd.DataFrame(
            {
                "o": [200 + i for i in range(len(gap_dates))],
                "h": [201 + i for i in range(len(gap_dates))],
                "l": [199 + i for i in range(len(gap_dates))],
                "c": [200.5 + i for i in range(len(gap_dates))],
                "v": [2000] * len(gap_dates),
            },
            index=pd.to_datetime(gap_dates),
        )

        fetcher = CachedPriceFetcher(backend=mock_backend)

        with patch.object(fetcher, "_fetch_from_api", return_value=gap_df):
            request_start = min(cached_dates + gap_dates)
            request_end = max(cached_dates + gap_dates)
            result = fetcher._get_history_internal("TEST", request_start, request_end, "1d")

            assert len(result) == expected_total
            assert result.index.is_monotonic_increasing

    def test_cache_stores_fetched_data(self, mock_backend):
        """Fetched data is stored in cache for future use."""
        mock_backend.get_prices.return_value = pd.DataFrame()  # Empty cache

        num_days = 5
        dates = [date(2024, 10, 1 + i) for i in range(num_days)]
        api_data = pd.DataFrame(
            {
                "o": [100 + i for i in range(num_days)],
                "h": [101 + i for i in range(num_days)],
                "l": [99 + i for i in range(num_days)],
                "c": [100.5 + i for i in range(num_days)],
                "v": [1000] * num_days,
            },
            index=pd.to_datetime(dates),
        )

        fetcher = CachedPriceFetcher(backend=mock_backend)

        with patch.object(fetcher, "_fetch_from_api", return_value=api_data):
            fetcher._get_history_internal("TEST", dates[0], dates[-1], "1d")

            mock_backend.store_prices.assert_called()
            stored_df = mock_backend.store_prices.call_args[0][1]
            assert len(stored_df) == num_days
