"""Tests for volume tool — implementation-agnostic.

Tests only the public volume() function via mocked OHLCV data.
Assertions target mathematical properties, not internal representations.
"""

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from volume import volume

START = "2025-06-01"
END = "2025-07-31"
DATA_START = date(2024, 1, 1)

EXPECTED_KEYS = {
    "symbol",
    "interval",
    "tz",
    "t",
    "volume",
    "vol_sma_5",
    "vol_sma_10",
    "vol_sma_20",
    "vol_sma_50",
    "short_interest",
}


@pytest.fixture(autouse=True)
def _mock_ticker():
    """Default: yfinance Ticker returns an empty info dict.

    Individual tests override ``m.return_value.info`` to supply
    short-interest data, or make ``m`` raise to simulate a fetch failure.
    """
    with patch("volume.yf.Ticker") as m:
        m.return_value.info = {}
        yield m


def _trading_days() -> list[date]:
    days: list[date] = []
    d = DATA_START
    end = date.fromisoformat(END)
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _make_ohlcv(volumes: list[float]) -> pd.DataFrame:
    dates = _trading_days()
    n = len(dates)
    assert len(volumes) == n
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0] * n,
            "High": [102.0] * n,
            "Low": [98.0] * n,
            "Close": [101.0] * n,
            "Volume": volumes,
        }
    )


N = len(_trading_days())


def _increasing_volume() -> pd.DataFrame:
    return _make_ohlcv([1000.0 + i * 100 for i in range(N)])


def _decreasing_volume() -> pd.DataFrame:
    return _make_ohlcv([60000.0 - i * 100 for i in range(N)])


def _constant_volume() -> pd.DataFrame:
    return _make_ohlcv([5000.0] * N)


def _spike_volume() -> pd.DataFrame:
    vols = [1000.0] * N
    vols[-1] = 100000.0
    return _make_ohlcv(vols)


class TestOutputStructure:
    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_has_all_keys(self, mock_fetch):
        result = volume("TEST", START, END)
        assert set(result.keys()) == EXPECTED_KEYS

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_symbol_uppercased(self, mock_fetch):
        result = volume("test", START, END)
        assert result["symbol"] == "TEST"

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_interval_is_1d(self, mock_fetch):
        result = volume("TEST", START, END)
        assert result["interval"] == "1d"

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_arrays_same_length(self, mock_fetch):
        result = volume("TEST", START, END)
        n = len(result["t"])
        assert n > 0
        series_keys = EXPECTED_KEYS - {
            "symbol",
            "interval",
            "tz",
            "t",
            "short_interest",
        }
        for key in series_keys:
            assert len(result[key]) == n

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_timestamps_within_range(self, mock_fetch):
        result = volume("TEST", START, END)
        for t in result["t"]:
            assert t >= START
            assert t <= END

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_volume_values_are_ints(self, mock_fetch):
        result = volume("TEST", START, END)
        for v in result["volume"]:
            assert isinstance(v, int)


class TestPositiveValues:
    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_all_sma_positive(self, mock_fetch):
        result = volume("TEST", START, END)
        for key in ("vol_sma_5", "vol_sma_10", "vol_sma_20", "vol_sma_50"):
            for v in result[key]:
                assert v > 0, f"{key}={v} not positive"

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_volume_positive(self, mock_fetch):
        result = volume("TEST", START, END)
        for v in result["volume"]:
            assert v > 0


class TestIncreasingVolume:
    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_short_sma_above_long(self, mock_fetch):
        result = volume("TEST", START, END)
        assert result["vol_sma_5"][-1] > result["vol_sma_50"][-1]


class TestDecreasingVolume:
    @patch("volume.fetch_ohlcv", return_value=_decreasing_volume())
    def test_short_sma_below_long(self, mock_fetch):
        result = volume("TEST", START, END)
        assert result["vol_sma_5"][-1] < result["vol_sma_50"][-1]


class TestConstantVolume:
    @patch("volume.fetch_ohlcv", return_value=_constant_volume())
    def test_all_sma_equal(self, mock_fetch):
        result = volume("TEST", START, END)
        for key in ("vol_sma_5", "vol_sma_10", "vol_sma_20", "vol_sma_50"):
            assert result[key][-1] == 5000.0


class TestSpikeVolume:
    @patch("volume.fetch_ohlcv", return_value=_spike_volume())
    def test_short_sma_reacts_more(self, mock_fetch):
        result = volume("TEST", START, END)
        assert result["vol_sma_5"][-1] > result["vol_sma_50"][-1]


# 2026-03-31 and 2026-02-27 as UTC epochs (seconds).
_AS_OF_EPOCH = 1774915200
_PRIOR_AS_OF_EPOCH = 1772150400

_FULL_SHORT_INFO = {
    "sharesShort": 126771284,
    "shortPercentOfFloat": 0.0086,
    "shortRatio": 3.11,
    "sharesShortPriorMonth": 129553812,
    "dateShortInterest": _AS_OF_EPOCH,
    "sharesShortPreviousMonthDate": _PRIOR_AS_OF_EPOCH,
}


class TestShortInterestMissing:
    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_none_when_info_empty(self, mock_fetch):
        result = volume("TEST", START, END)
        assert result["short_interest"] is None

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_none_when_ticker_raises(self, mock_fetch, _mock_ticker):
        _mock_ticker.side_effect = RuntimeError("network down")
        result = volume("TEST", START, END)
        assert result["short_interest"] is None

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_none_when_shares_short_missing(self, mock_fetch, _mock_ticker):
        _mock_ticker.return_value.info = {
            "shortPercentOfFloat": 0.01,
            "shortRatio": 2.5,
        }
        result = volume("TEST", START, END)
        assert result["short_interest"] is None

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_skips_network_for_non_us_tickers(self, mock_fetch, _mock_ticker):
        """Tickers with a '.' exchange suffix short-circuit before the info call."""
        _mock_ticker.return_value.info = dict(_FULL_SHORT_INFO)
        for sym in ("7203.T", "0700.HK", "BP.L"):
            _mock_ticker.reset_mock()
            result = volume(sym, START, END)
            assert result["short_interest"] is None
            _mock_ticker.assert_not_called()


class TestShortInterestPresent:
    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_full_snapshot(self, mock_fetch, _mock_ticker):
        _mock_ticker.return_value.info = dict(_FULL_SHORT_INFO)
        si = volume("TEST", START, END)["short_interest"]
        assert si["shares_short"] == 126771284
        assert si["pct_of_float"] == 0.0086
        assert si["days_to_cover"] == 3.11
        assert si["as_of"] == "2026-03-31"
        assert si["prior_month"] == {
            "as_of": "2026-02-27",
            "shares_short": 129553812,
        }

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_no_prior_month(self, mock_fetch, _mock_ticker):
        _mock_ticker.return_value.info = {
            "sharesShort": 100,
            "shortPercentOfFloat": 0.01,
            "shortRatio": 2.0,
            "dateShortInterest": _AS_OF_EPOCH,
        }
        si = volume("TEST", START, END)["short_interest"]
        assert si["shares_short"] == 100
        assert "prior_month" not in si

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_invalid_epoch_yields_none_date(self, mock_fetch, _mock_ticker):
        _mock_ticker.return_value.info = {
            "sharesShort": 100,
            "dateShortInterest": "not-an-epoch",
        }
        si = volume("TEST", START, END)["short_interest"]
        assert si["as_of"] is None

    @patch("volume.fetch_ohlcv", return_value=_increasing_volume())
    def test_called_with_symbol(self, mock_fetch, _mock_ticker):
        _mock_ticker.return_value.info = dict(_FULL_SHORT_INFO)
        volume("aapl", START, END)
        _mock_ticker.assert_any_call("aapl")
