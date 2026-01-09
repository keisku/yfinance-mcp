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
