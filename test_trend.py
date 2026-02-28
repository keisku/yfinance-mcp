"""Tests for trend tool — implementation-agnostic.

Tests only the public trend() function via mocked OHLCV data.
Assertions target mathematical properties, not internal representations.
"""

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from trend import trend

START = "2025-06-01"
END = "2025-06-30"
WARMUP_START = date(2025, 6, 1) - timedelta(days=112)

EXPECTED_KEYS = {
    "symbol",
    "interval",
    "tz",
    "t",
    "macd",
    "macd_signal",
    "macd_hist",
    "plus_di",
    "minus_di",
    "adx",
}

BOUNDED_KEYS = ["plus_di", "minus_di", "adx"]


def _trading_days(start: date, n: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _make_ohlcv(closes: list[float], spread: float = 2.0) -> pd.DataFrame:
    n = len(closes)
    dates = _trading_days(WARMUP_START, n)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [c - 0.5 for c in closes],
            "High": [c + spread for c in closes],
            "Low": [c - spread for c in closes],
            "Close": closes,
            "Volume": [1000] * n,
        }
    )


def _uptrend_ohlcv() -> pd.DataFrame:
    return _make_ohlcv([100.0 + i for i in range(100)])


def _downtrend_ohlcv() -> pd.DataFrame:
    return _make_ohlcv([200.0 - i for i in range(100)])


def _oscillating_ohlcv() -> pd.DataFrame:
    return _make_ohlcv([100.0 + (3.0 if i % 2 == 0 else -3.0) for i in range(100)])


def _constant_ohlcv() -> pd.DataFrame:
    return _make_ohlcv([100.0] * 100, spread=0.0)


class TestOutputStructure:
    @patch("trend.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_has_all_keys(self, mock_fetch):
        result = trend("TEST", START, END)
        assert set(result.keys()) == EXPECTED_KEYS

    @patch("trend.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_symbol_uppercased(self, mock_fetch):
        result = trend("test", START, END)
        assert result["symbol"] == "TEST"

    @patch("trend.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_interval_is_1d(self, mock_fetch):
        result = trend("TEST", START, END)
        assert result["interval"] == "1d"

    @patch("trend.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_arrays_same_length(self, mock_fetch):
        result = trend("TEST", START, END)
        n = len(result["t"])
        assert n > 0
        for key in EXPECTED_KEYS - {"symbol", "interval", "tz", "t"}:
            assert len(result[key]) == n

    @patch("trend.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_timestamps_within_range(self, mock_fetch):
        result = trend("TEST", START, END)
        for t in result["t"]:
            assert t >= START
            assert t <= END


class TestBoundedRange:
    @patch("trend.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_uptrend_bounded(self, mock_fetch):
        result = trend("TEST", START, END)
        for key in BOUNDED_KEYS:
            for v in result[key]:
                assert 0 <= v <= 100, f"{key}={v} out of [0, 100]"

    @patch("trend.fetch_ohlcv", return_value=_downtrend_ohlcv())
    def test_downtrend_bounded(self, mock_fetch):
        result = trend("TEST", START, END)
        for key in BOUNDED_KEYS:
            for v in result[key]:
                assert 0 <= v <= 100, f"{key}={v} out of [0, 100]"

    @patch("trend.fetch_ohlcv", return_value=_oscillating_ohlcv())
    def test_oscillating_bounded(self, mock_fetch):
        result = trend("TEST", START, END)
        for key in BOUNDED_KEYS:
            for v in result[key]:
                assert 0 <= v <= 100, f"{key}={v} out of [0, 100]"


class TestUptrend:
    @patch("trend.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_macd_positive(self, mock_fetch):
        result = trend("TEST", START, END)
        assert result["macd"][-1] > 0

    @patch("trend.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_plus_di_gt_minus_di(self, mock_fetch):
        result = trend("TEST", START, END)
        assert result["plus_di"][-1] > result["minus_di"][-1]


class TestDowntrend:
    @patch("trend.fetch_ohlcv", return_value=_downtrend_ohlcv())
    def test_macd_negative(self, mock_fetch):
        result = trend("TEST", START, END)
        assert result["macd"][-1] < 0

    @patch("trend.fetch_ohlcv", return_value=_downtrend_ohlcv())
    def test_minus_di_gt_plus_di(self, mock_fetch):
        result = trend("TEST", START, END)
        assert result["minus_di"][-1] > result["plus_di"][-1]


class TestConstantPrice:
    @patch("trend.fetch_ohlcv", return_value=_constant_ohlcv())
    def test_raises_value_error(self, mock_fetch):
        with pytest.raises(ValueError):
            trend("TEST", START, END)
