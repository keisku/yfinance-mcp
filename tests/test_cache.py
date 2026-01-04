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
    """Test _find_gaps method directly."""

    @pytest.mark.parametrize(
        "cached_dates,start,end,expected_gaps",
        [
            pytest.param(
                {date(2024, 6, 3)},
                date(2024, 6, 3),
                date(2024, 6, 3),
                [],
                id="single_day_range_cached",
            ),
            pytest.param(
                set(),
                date(2024, 6, 3),
                date(2024, 6, 3),
                [(date(2024, 6, 3), date(2024, 6, 3))],
                id="single_day_range_missing",
            ),
            pytest.param(
                {date(2024, 8, 7)},
                date(2024, 8, 5),
                date(2024, 8, 9),
                [(date(2024, 8, 5), date(2024, 8, 6)), (date(2024, 8, 8), date(2024, 8, 9))],
                id="only_one_cached_gaps_both_sides",
            ),
            pytest.param(
                {date(2024, 4, 8), date(2024, 4, 9), date(2024, 4, 10)},
                date(2024, 4, 1),
                date(2024, 4, 10),
                [(date(2024, 4, 1), date(2024, 4, 5))],
                id="gap_at_start",
            ),
            pytest.param(
                {date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 3)},
                date(2024, 4, 1),
                date(2024, 4, 10),
                [(date(2024, 4, 4), date(2024, 4, 5)), (date(2024, 4, 8), date(2024, 4, 10))],
                id="gap_at_end",
            ),
            pytest.param(
                set(),
                date(2024, 9, 14),
                date(2024, 9, 15),
                [],
                id="range_entirely_weekend",
            ),
            pytest.param(
                {date(2024, 10, 4), date(2024, 10, 7)},
                date(2024, 10, 4),
                date(2024, 10, 7),
                [],
                id="friday_to_monday_no_gap",
            ),
            pytest.param(
                {date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6)},
                date(2024, 3, 4),
                date(2024, 3, 6),
                [],
                id="no_gaps_mar_2024",
            ),
            pytest.param(
                {date(2023, 7, 10), date(2023, 7, 14)},
                date(2023, 7, 10),
                date(2023, 7, 14),
                [(date(2023, 7, 11), date(2023, 7, 13))],
                id="single_gap_jul_2023",
            ),
            pytest.param(
                {date(2024, 11, 4), date(2024, 11, 8), date(2024, 11, 15)},
                date(2024, 11, 4),
                date(2024, 11, 15),
                [(date(2024, 11, 5), date(2024, 11, 7)), (date(2024, 11, 11), date(2024, 11, 14))],
                id="multiple_gaps_nov_2024",
            ),
            pytest.param(
                {date(2023, 12, 28), date(2023, 12, 29), date(2024, 1, 3)},
                date(2023, 12, 28),
                date(2024, 1, 3),
                [(date(2024, 1, 2), date(2024, 1, 2))],
                id="year_boundary_2023_2024",
            ),
            pytest.param(
                {date(2024, 12, 30), date(2024, 12, 31), date(2025, 1, 6)},
                date(2024, 12, 30),
                date(2025, 1, 6),
                [(date(2025, 1, 2), date(2025, 1, 3))],
                id="year_boundary_2024_2025",
            ),
            pytest.param(
                set(),
                date(2022, 8, 15),
                date(2022, 8, 17),
                [(date(2022, 8, 15), date(2022, 8, 17))],
                id="empty_cache_aug_2022",
            ),
            pytest.param(
                {date(2024, 2, 1), date(2024, 2, 29)},
                date(2024, 2, 1),
                date(2024, 2, 29),
                [
                    (date(2024, 2, 2), date(2024, 2, 2)),
                    (date(2024, 2, 5), date(2024, 2, 9)),
                    (date(2024, 2, 12), date(2024, 2, 16)),
                    (date(2024, 2, 20), date(2024, 2, 23)),
                    (date(2024, 2, 26), date(2024, 2, 28)),
                ],
                id="leap_year_feb_2024",
            ),
            pytest.param(
                {date(2023, 2, 1), date(2023, 2, 28)},
                date(2023, 2, 1),
                date(2023, 2, 28),
                [
                    (date(2023, 2, 2), date(2023, 2, 3)),
                    (date(2023, 2, 6), date(2023, 2, 10)),
                    (date(2023, 2, 13), date(2023, 2, 17)),
                    (date(2023, 2, 21), date(2023, 2, 24)),
                    (date(2023, 2, 27), date(2023, 2, 27)),
                ],
                id="non_leap_year_feb_2023",
            ),
            pytest.param(
                {date(2024, 6, 3), date(2024, 6, 7)},
                date(2024, 6, 3),
                date(2024, 6, 7),
                [(date(2024, 6, 4), date(2024, 6, 6))],
                id="week_range_single_gap",
            ),
        ],
    )
    def test_find_gaps(self, fetcher, cached_dates, start, end, expected_gaps):
        gaps = fetcher._find_gaps(cached_dates, start, end, symbol="AAPL")
        assert gaps == expected_gaps

    def test_find_gaps_ten_year_range(self, fetcher):
        """10-year range with sparse cache points produces many gaps."""
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
        gaps = fetcher._find_gaps(cached_dates, date(2015, 1, 2), date(2025, 1, 2), symbol="AAPL")

        assert len(gaps) == 545
        assert all(start <= end for start, end in gaps)
        assert gaps[0][0] == date(2015, 1, 5)
        assert gaps[-1][1] == date(2024, 12, 31)

    @pytest.mark.parametrize(
        "symbol",
        [
            pytest.param("AAPL", id="us_market"),
            pytest.param("7203.T", id="japan_market"),
            pytest.param("VOD.L", id="uk_market"),
        ],
    )
    def test_find_gaps_different_markets(self, fetcher, symbol):
        """Gap detection works with different market calendars."""
        cached_dates = {date(2024, 5, 6), date(2024, 5, 10)}
        gaps = fetcher._find_gaps(cached_dates, date(2024, 5, 6), date(2024, 5, 10), symbol=symbol)
        assert len(gaps) == 1


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
                    "AAPL", date(2024, 4, 15), date(2024, 10, 21), interval
                )

            assert spy.called == should_detect_gaps

    def test_fills_detected_gaps(self):
        """Should call API for each detected gap."""
        backend = MagicMock()
        jan = pd.DataFrame(
            {"o": [100], "h": [101], "l": [99], "c": [100.5], "v": [1000]},
            index=pd.to_datetime([date(2024, 1, 8)]),
        )
        mar = pd.DataFrame(
            {"o": [110], "h": [111], "l": [109], "c": [110.5], "v": [1100]},
            index=pd.to_datetime([date(2024, 3, 11)]),
        )
        backend.get_prices.return_value = pd.concat([jan, mar])
        backend.store_prices = MagicMock()

        fetcher = CachedPriceFetcher(backend=backend)

        with patch.object(fetcher, "_fetch_from_api", return_value=pd.DataFrame()) as mock_api:
            fetcher._get_history_internal("AAPL", date(2024, 1, 8), date(2024, 3, 11), "1d")

            assert mock_api.called
            calls = mock_api.call_args_list
            fetched_starts = [c[0][1] for c in calls]
            assert any(d.month == 1 for d in fetched_starts)

    def test_no_api_call_when_fully_cached(self):
        """Should not call API when cache has all trading days."""
        backend = MagicMock()
        dates = pd.bdate_range(date(2024, 5, 6), date(2024, 5, 10))
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
            fetcher._get_history_internal("AAPL", date(2024, 5, 6), date(2024, 5, 10), "1d")
            mock_api.assert_not_called()
