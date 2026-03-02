"""Tests for oscillator tool — implementation-agnostic.

Tests only the public oscillator() function via mocked OHLCV data.
Assertions target mathematical properties, not internal representations.
"""

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from oscillator import oscillator

START = "2025-06-01"
END = "2025-06-30"
DATA_START = date(2024, 1, 1)

EXPECTED_KEYS = {
    "symbol",
    "interval",
    "tz",
    "t",
    "rsi",
    "stoch_k",
    "stoch_d",
}

BOUNDED_KEYS = ["rsi", "stoch_k", "stoch_d"]


def _trading_days() -> list[date]:
    days: list[date] = []
    d = DATA_START
    end = date.fromisoformat(END)
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _make_ohlcv(closes: list[float], spread: float = 2.0) -> pd.DataFrame:
    dates = _trading_days()
    n = len(dates)
    assert len(closes) == n
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


N = len(_trading_days())


def _uptrend_ohlcv() -> pd.DataFrame:
    return _make_ohlcv([100.0 + i for i in range(N)])


def _downtrend_ohlcv() -> pd.DataFrame:
    return _make_ohlcv([600.0 - i for i in range(N)])


def _oscillating_ohlcv() -> pd.DataFrame:
    return _make_ohlcv([100.0 + (3.0 if i % 2 == 0 else -3.0) for i in range(N)])


def _constant_ohlcv() -> pd.DataFrame:
    return _make_ohlcv([100.0] * N, spread=0.0)


class TestOutputStructure:
    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_has_all_keys(self, mock_fetch):
        result = oscillator("TEST", START, END)
        assert set(result.keys()) == EXPECTED_KEYS

    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_symbol_uppercased(self, mock_fetch):
        result = oscillator("test", START, END)
        assert result["symbol"] == "TEST"

    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_interval_is_1d(self, mock_fetch):
        result = oscillator("TEST", START, END)
        assert result["interval"] == "1d"

    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_arrays_same_length(self, mock_fetch):
        result = oscillator("TEST", START, END)
        n = len(result["t"])
        assert n > 0
        for key in EXPECTED_KEYS - {"symbol", "interval", "tz", "t"}:
            assert len(result[key]) == n

    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_timestamps_within_range(self, mock_fetch):
        result = oscillator("TEST", START, END)
        for t in result["t"]:
            assert t >= START
            assert t <= END


class TestBoundedRange:
    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_uptrend_bounded(self, mock_fetch):
        result = oscillator("TEST", START, END)
        for key in BOUNDED_KEYS:
            for v in result[key]:
                assert 0 <= v <= 100, f"{key}={v} out of [0, 100]"

    @patch("oscillator.fetch_ohlcv", return_value=_downtrend_ohlcv())
    def test_downtrend_bounded(self, mock_fetch):
        result = oscillator("TEST", START, END)
        for key in BOUNDED_KEYS:
            for v in result[key]:
                assert 0 <= v <= 100, f"{key}={v} out of [0, 100]"

    @patch("oscillator.fetch_ohlcv", return_value=_oscillating_ohlcv())
    def test_oscillating_bounded(self, mock_fetch):
        result = oscillator("TEST", START, END)
        for key in BOUNDED_KEYS:
            for v in result[key]:
                assert 0 <= v <= 100, f"{key}={v} out of [0, 100]"


class TestUptrend:
    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_rsi_above_50(self, mock_fetch):
        result = oscillator("TEST", START, END)
        assert result["rsi"][-1] > 50


class TestDowntrend:
    @patch("oscillator.fetch_ohlcv", return_value=_downtrend_ohlcv())
    def test_rsi_below_50(self, mock_fetch):
        result = oscillator("TEST", START, END)
        assert result["rsi"][-1] < 50


class TestConstantPrice:
    @patch("oscillator.fetch_ohlcv", return_value=_constant_ohlcv())
    def test_raises_value_error(self, mock_fetch):
        with pytest.raises(ValueError):
            oscillator("TEST", START, END)
