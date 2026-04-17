"""Tests for oscillator tool — implementation-agnostic.

Tests only the public oscillator() function via mocked OHLCV data.
Assertions target mathematical properties, not internal representations.
"""

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    "put_call_ratio",
}

BOUNDED_KEYS = ["rsi", "stoch_k", "stoch_d"]


@pytest.fixture(autouse=True)
def _mock_ticker():
    """Default: yfinance Ticker reports no listed options.

    Individual tests override ``_mock_ticker.return_value`` to supply
    option-chain data, or set ``side_effect`` to simulate failures.
    """
    with patch("oscillator.yf.Ticker") as m:
        default = MagicMock()
        default.options = ()
        m.return_value = default
        yield m


def _make_chain(
    call_vol: float, put_vol: float, call_oi: float, put_oi: float
) -> SimpleNamespace:
    return SimpleNamespace(
        calls=pd.DataFrame({"volume": [call_vol], "openInterest": [call_oi]}),
        puts=pd.DataFrame({"volume": [put_vol], "openInterest": [put_oi]}),
    )


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
        series_keys = EXPECTED_KEYS - {
            "symbol",
            "interval",
            "tz",
            "t",
            "put_call_ratio",
        }
        for key in series_keys:
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


class TestPutCallRatio:
    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_skips_network_for_non_us_tickers(self, mock_fetch, _mock_ticker):
        """'.' suffix short-circuits before any yfinance call — cost guarantee."""
        for sym in ("7203.T", "0700.HK", "BP.L"):
            _mock_ticker.reset_mock()
            result = oscillator(sym, START, END)
            assert result["put_call_ratio"] is None
            _mock_ticker.assert_not_called()

    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_aggregates_across_expirations(self, mock_fetch, _mock_ticker):
        tkr = MagicMock()
        tkr.options = ("2026-04-17", "2026-04-24")
        tkr.option_chain.side_effect = lambda exp: {
            "2026-04-17": _make_chain(100, 50, 200, 100),
            "2026-04-24": _make_chain(200, 40, 300, 120),
        }[exp]
        _mock_ticker.return_value = tkr

        pcr = oscillator("TEST", START, END)["put_call_ratio"]
        assert pcr["call_volume"] == 300
        assert pcr["put_volume"] == 90
        assert pcr["call_oi"] == 500
        assert pcr["put_oi"] == 220
        assert pcr["volume_based"] == round(90 / 300, 4)
        assert pcr["oi_based"] == round(220 / 500, 4)
        assert pcr["expirations"] == 2
        assert pcr["as_of"] == date.today().isoformat()

    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_partial_chain_failure_counts_successes(self, mock_fetch, _mock_ticker):
        tkr = MagicMock()
        tkr.options = ("2026-04-17", "2026-04-24", "2026-05-01")

        def _side(exp: str):
            if exp == "2026-04-24":
                raise RuntimeError("flaky")
            return {
                "2026-04-17": _make_chain(100, 50, 200, 100),
                "2026-05-01": _make_chain(200, 60, 300, 120),
            }[exp]

        tkr.option_chain.side_effect = _side
        _mock_ticker.return_value = tkr

        pcr = oscillator("TEST", START, END)["put_call_ratio"]
        assert pcr["expirations"] == 2
        assert pcr["call_volume"] == 300
        assert pcr["put_volume"] == 110

    @patch("oscillator.fetch_ohlcv", return_value=_uptrend_ohlcv())
    def test_zero_call_volume_yields_none_volume_ratio(self, mock_fetch, _mock_ticker):
        tkr = MagicMock()
        tkr.options = ("2026-04-17",)
        tkr.option_chain.return_value = _make_chain(0, 10, 100, 50)
        _mock_ticker.return_value = tkr

        pcr = oscillator("TEST", START, END)["put_call_ratio"]
        assert pcr["volume_based"] is None
        assert pcr["oi_based"] == round(50 / 100, 4)
