"""Tests for technical indicators - error handling and unique functions.

Note: Accuracy tests are in test_indicators_crosscheck.py (validates against pandas-ta).
This file covers error handling and functions not available in pandas-ta.
"""

import pandas as pd
import pytest

from yfinance_mcp.errors import CalculationError
from yfinance_mcp.indicators import (
    calculate_cci,
    calculate_dmi,
    calculate_fast_stochastic,
    calculate_fibonacci_levels,
    calculate_ichimoku,
    calculate_momentum,
    calculate_pivot_points,
    calculate_price_change,
    calculate_sma,
    calculate_volume_profile,
    calculate_williams_r,
    calculate_wma,
)


class TestErrorHandling:
    """Test CalculationError is raised for insufficient data."""

    def test_sma_insufficient_data(self) -> None:
        """SMA should raise CalculationError when data < period."""
        prices = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(CalculationError):
            calculate_sma(prices, 10)

    def test_wma_insufficient_data(self) -> None:
        """WMA should raise CalculationError when data < period."""
        prices = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(CalculationError):
            calculate_wma(prices, 10)

    def test_momentum_insufficient_data(self) -> None:
        """Momentum should raise CalculationError when data < period + 1."""
        prices = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(CalculationError):
            calculate_momentum(prices, 10)

    def test_cci_insufficient_data(self) -> None:
        """CCI should raise CalculationError when data < period."""
        high = pd.Series([1, 2, 3, 4, 5])
        low = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        close = pd.Series([0.8, 1.8, 2.8, 3.8, 4.8])
        with pytest.raises(CalculationError):
            calculate_cci(high, low, close, 20)

    def test_dmi_insufficient_data(self) -> None:
        """DMI should raise CalculationError when data < period * 2."""
        high = pd.Series([1, 2, 3, 4, 5])
        low = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        close = pd.Series([0.8, 1.8, 2.8, 3.8, 4.8])
        with pytest.raises(CalculationError):
            calculate_dmi(high, low, close, 14)

    def test_williams_r_insufficient_data(self) -> None:
        """Williams %R should raise CalculationError when data < period."""
        high = pd.Series([1, 2, 3, 4, 5])
        low = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        close = pd.Series([0.8, 1.8, 2.8, 3.8, 4.8])
        with pytest.raises(CalculationError):
            calculate_williams_r(high, low, close, 14)

    def test_fast_stochastic_insufficient_data(self) -> None:
        """Fast Stochastic should raise CalculationError when data < k + d."""
        high = pd.Series([1, 2, 3, 4, 5])
        low = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        close = pd.Series([0.8, 1.8, 2.8, 3.8, 4.8])
        with pytest.raises(CalculationError):
            calculate_fast_stochastic(high, low, close, 14, 3)

    def test_ichimoku_insufficient_data(self) -> None:
        """Ichimoku should raise CalculationError when data < senkou_b + kijun."""
        high = pd.Series([1, 2, 3, 4, 5])
        low = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        close = pd.Series([0.8, 1.8, 2.8, 3.8, 4.8])
        with pytest.raises(CalculationError):
            calculate_ichimoku(high, low, close)

    def test_volume_profile_insufficient_data(self) -> None:
        """Volume Profile should raise CalculationError when data < 10."""
        close = pd.Series([1, 2, 3, 4, 5])
        volume = pd.Series([100, 200, 300, 400, 500])
        with pytest.raises(CalculationError):
            calculate_volume_profile(close, volume)

    def test_price_change_insufficient_data(self) -> None:
        """Price Change should raise CalculationError when data < period + 1."""
        close = pd.Series([100.0])
        with pytest.raises(CalculationError):
            calculate_price_change(close, period=1)

    def test_price_change_division_by_zero(self) -> None:
        """Price Change should return None for change_pct when previous price is 0."""
        close = pd.Series([0.0, 100.0])
        result = calculate_price_change(close, period=1)
        assert result["change"] == 100.0
        assert result["change_pct"] is None


class TestPriceChangeFullPeriod:
    """Test price_change with full period (first-to-last bar comparison).

    This addresses the issue where users expect "1 month" change to reflect
    the full period change, not just the last single bar change.
    """

    def test_full_period_change_basic(self) -> None:
        """Full period should compare first and last bars."""
        close = pd.Series([100.0, 105.0, 110.0, 108.0, 115.0])
        result = calculate_price_change(close, period=len(close) - 1)

        expected_change = 115.0 - 100.0  # last - first
        expected_pct = (expected_change / 100.0) * 100

        assert result["change"] == expected_change
        assert abs(result["change_pct"] - expected_pct) < 0.0001

    def test_full_period_vs_single_bar(self) -> None:
        """Full period change should differ from single-bar change.

        This is the exact bug scenario: single-bar shows ~0.05% while
        full-period shows the actual month-over-month change.
        """
        close = pd.Series([100.0, 101.0, 102.0, 101.5, 101.6])

        single_bar = calculate_price_change(close, period=1)
        full_period = calculate_price_change(close, period=len(close) - 1)

        assert single_bar["change"] == pytest.approx(0.1, abs=0.01)
        assert single_bar["change_pct"] == pytest.approx(0.0985, abs=0.01)

        assert full_period["change"] == pytest.approx(1.6, abs=0.01)
        assert full_period["change_pct"] == pytest.approx(1.6, abs=0.01)

        assert abs(full_period["change"]) > abs(single_bar["change"])

    def test_full_period_negative_change(self) -> None:
        """Full period should correctly show negative returns."""
        close = pd.Series([100.0, 98.0, 95.0, 96.0, 94.0])
        result = calculate_price_change(close, period=len(close) - 1)

        assert result["change"] == -6.0
        assert result["change_pct"] == pytest.approx(-6.0, abs=0.01)

    def test_full_period_with_volatility(self) -> None:
        """Full period ignores intermediate volatility, only compares endpoints."""
        close = pd.Series([100.0, 150.0, 80.0, 120.0, 105.0])
        result = calculate_price_change(close, period=len(close) - 1)

        assert result["change"] == 5.0
        assert result["change_pct"] == pytest.approx(5.0, abs=0.01)


class TestFibonacci:
    """Test Fibonacci retracement levels (not in pandas-ta)."""

    def test_uptrend_levels(self) -> None:
        """Uptrend retracements from high to low."""
        levels = calculate_fibonacci_levels(100, 80, is_uptrend=True)
        assert levels["level_0"] == 100
        assert levels["level_100"] == 80
        assert levels["level_500"] == 90  # 50% retracement

    def test_downtrend_levels(self) -> None:
        """Downtrend retracements from low to high."""
        levels = calculate_fibonacci_levels(100, 80, is_uptrend=False)
        assert levels["level_0"] == 80
        assert levels["level_100"] == 100


class TestPivotPoints:
    """Test pivot point calculations (not in pandas-ta)."""

    def test_standard_method(self) -> None:
        """Standard pivot = (H + L + C) / 3."""
        pivots = calculate_pivot_points(105, 95, 102, method="standard")
        assert "pivot" in pivots
        assert "r1" in pivots
        assert "s1" in pivots
        expected_pivot = (105 + 95 + 102) / 3
        assert abs(pivots["pivot"] - expected_pivot) < 0.01

    def test_all_methods(self) -> None:
        """All pivot methods should return valid structure."""
        for method in ["standard", "fibonacci", "camarilla", "woodie"]:
            pivots = calculate_pivot_points(105, 95, 102, method=method)
            assert "pivot" in pivots
            assert "r1" in pivots and "r2" in pivots and "r3" in pivots
            assert "s1" in pivots and "s2" in pivots and "s3" in pivots

    def test_invalid_method(self) -> None:
        """Invalid method should raise CalculationError."""
        with pytest.raises(CalculationError):
            calculate_pivot_points(105, 95, 102, method="invalid")
