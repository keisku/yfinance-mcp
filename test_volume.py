"""Tests for volume tool — implementation-agnostic.

Tests only the public volume() function via mocked OHLCV data.
Assertions target mathematical properties, not internal representations.
"""

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
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
}


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
        for key in EXPECTED_KEYS - {"symbol", "interval", "tz", "t"}:
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
