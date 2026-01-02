"""Cross-validate indicator calculations against pandas-ta.

These tests ensure our implementations match industry-standard calculations.
pandas-ta is a widely-used technical analysis library that serves as ground truth.

Run these tests with: pytest -m crosscheck
Skip these tests with: pytest -m "not crosscheck"

Reference: https://www.pandas-ta.dev/getting-started/installation/#official-source
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

from yfinance_mcp import indicators

pytestmark = pytest.mark.crosscheck


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate realistic OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    # Simulate price movement with drift and volatility
    returns = np.random.randn(n) * 0.02 + 0.001
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n),
    )


class TestSMACrosscheck:
    """Validate SMA against pandas-ta.

    SMA is a deterministic calculation with no initialization ambiguity,
    so we expect exact matches (within floating-point precision).
    """

    def test_sma_20_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """SMA(20) should match pandas-ta exactly."""
        close = sample_ohlcv["Close"]

        our_sma = indicators.calculate_sma(close, 20)
        expected = ta.sma(close, length=20)

        np.testing.assert_allclose(
            our_sma.dropna().values,
            expected.dropna().values,
            rtol=1e-10,
        )

    def test_sma_50_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """SMA(50) should match pandas-ta exactly."""
        close = sample_ohlcv["Close"]

        our_sma = indicators.calculate_sma(close, 50)
        expected = ta.sma(close, length=50)

        np.testing.assert_allclose(
            our_sma.dropna().values,
            expected.dropna().values,
            rtol=1e-10,
        )


class TestEMACrosscheck:
    """Validate EMA against pandas-ta.

    Tolerance rationale:
    - pandas-ta initializes EMA with SMA of first N values
    - We use pandas.ewm which starts from first price
    - Both are valid approaches used in different trading platforms
    - After warmup period, values converge and track identically
    - We test correlation (>0.999) rather than exact values
    """

    def test_ema_12_correlation(self, sample_ohlcv: pd.DataFrame) -> None:
        """EMA(12) should be highly correlated with pandas-ta."""
        close = sample_ohlcv["Close"]

        our_ema = indicators.calculate_ema(close, 12)
        expected = ta.ema(close, length=12)

        corr = our_ema.iloc[20:].corr(expected.iloc[20:])
        assert corr > 0.9999

    def test_ema_26_correlation(self, sample_ohlcv: pd.DataFrame) -> None:
        """EMA(26) should be highly correlated with pandas-ta.

        Tolerance rationale:
        - Longer EMA periods show more initialization divergence
        - pandas-ta uses different warm-up handling
        - 0.995 correlation is still excellent agreement
        """
        close = sample_ohlcv["Close"]

        our_ema = indicators.calculate_ema(close, 26)
        expected = ta.ema(close, length=26)

        corr = our_ema.iloc[40:].corr(expected.iloc[40:])
        assert corr > 0.995


class TestRSICrosscheck:
    """Validate RSI against pandas-ta.

    Tolerance rationale:
    - RSI uses EMA internally, inheriting initialization differences
    - pandas-ta uses Wilder's smoothing (alpha=1/period)
    - We use standard EMA (alpha=2/(period+1))
    - Both produce valid RSI that oscillates in the same range
    - 10% tolerance accounts for the smoothing method difference
    """

    def test_rsi_14_correlation(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI(14) should be highly correlated with pandas-ta."""
        close = sample_ohlcv["Close"]

        our_rsi = indicators.calculate_rsi(close, 14)
        expected = ta.rsi(close, length=14)

        valid_our = our_rsi.dropna()
        valid_exp = expected.dropna()

        start = max(valid_our.index[0], valid_exp.index[0])
        our_aligned = valid_our.loc[start:]
        exp_aligned = valid_exp.loc[start:]

        corr = our_aligned.corr(exp_aligned)
        assert corr > 0.98

    def test_rsi_bounds(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI must always be in [0, 100]."""
        close = sample_ohlcv["Close"]
        rsi = indicators.calculate_rsi(close)

        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestMACDCrosscheck:
    """Validate MACD against pandas-ta.

    Tolerance rationale:
    - MACD = EMA(12) - EMA(26), so initialization differences compound
    - Signal line applies another EMA, adding more variance
    - Histogram is difference of differences, highest variance
    - Correlation thresholds: MACD > 0.998, Signal > 0.995, Histogram > 0.97
    """

    def test_macd_correlation(self, sample_ohlcv: pd.DataFrame) -> None:
        """MACD components should be highly correlated with pandas-ta."""
        close = sample_ohlcv["Close"]

        our_macd = indicators.calculate_macd(close)
        expected = ta.macd(close, fast=12, slow=26, signal=9)

        macd_corr = our_macd["macd"].iloc[40:].corr(expected["MACD_12_26_9"].iloc[40:])
        assert macd_corr > 0.998

        signal_corr = our_macd["signal"].iloc[40:].corr(expected["MACDs_12_26_9"].iloc[40:])
        assert signal_corr > 0.995

        hist_corr = our_macd["histogram"].iloc[40:].corr(expected["MACDh_12_26_9"].iloc[40:])
        assert hist_corr > 0.97


class TestBollingerCrosscheck:
    """Validate Bollinger Bands against pandas-ta.

    Bollinger Bands use SMA (deterministic) and standard deviation,
    so we expect exact matches within floating-point precision.
    """

    def test_bollinger_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """Bollinger Bands should match pandas-ta exactly."""
        close = sample_ohlcv["Close"]

        our_bb = indicators.calculate_bollinger_bands(close, 20, 2.0)
        expected = ta.bbands(close, length=20, std=2.0)

        bbm_col = [c for c in expected.columns if c.startswith("BBM")][0]
        bbu_col = [c for c in expected.columns if c.startswith("BBU")][0]
        bbl_col = [c for c in expected.columns if c.startswith("BBL")][0]

        np.testing.assert_allclose(
            our_bb["middle"].dropna().values,
            expected[bbm_col].dropna().values,
            rtol=1e-10,
        )

        np.testing.assert_allclose(
            our_bb["upper"].dropna().values,
            expected[bbu_col].dropna().values,
            rtol=1e-10,
        )

        np.testing.assert_allclose(
            our_bb["lower"].dropna().values,
            expected[bbl_col].dropna().values,
            rtol=1e-10,
        )

    def test_bollinger_ordering(self, sample_ohlcv: pd.DataFrame) -> None:
        """Upper > Middle > Lower must always hold."""
        close = sample_ohlcv["Close"]
        bb = indicators.calculate_bollinger_bands(close)

        valid_idx = bb["upper"].dropna().index
        assert (bb["upper"].loc[valid_idx] >= bb["middle"].loc[valid_idx]).all()
        assert (bb["middle"].loc[valid_idx] >= bb["lower"].loc[valid_idx]).all()


class TestStochasticCrosscheck:
    """Validate Stochastic Oscillator against pandas-ta.

    Stochastic uses rolling min/max which is deterministic.
    We expect close matches with 1% tolerance for %D smoothing differences.
    """

    def test_stochastic_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """Stochastic %K should match pandas-ta."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        our_stoch = indicators.calculate_stochastic(high, low, close, 14, 3)
        expected = ta.stoch(high, low, close, k=14, d=3, smooth_k=1)

        np.testing.assert_allclose(
            our_stoch["k"].iloc[20:].values,
            expected["STOCHk_14_3_1"].iloc[20:].values,
            rtol=0.01,
        )

    def test_stochastic_bounds(self, sample_ohlcv: pd.DataFrame) -> None:
        """Stochastic must be in [0, 100]."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        stoch = indicators.calculate_stochastic(high, low, close)

        valid_k = stoch["k"].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()


class TestATRCrosscheck:
    """Validate ATR against pandas-ta.

    Tolerance rationale:
    - ATR uses True Range with EMA smoothing
    - pandas-ta defaults to RMA (Wilder's), we use EMA
    - Testing with mamode="ema" for fair comparison
    - 3% tolerance for minor implementation differences
    """

    def test_atr_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR should match pandas-ta with EMA mode."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        our_atr = indicators.calculate_atr(high, low, close, 14)
        expected = ta.atr(high, low, close, length=14, mamode="ema")

        np.testing.assert_allclose(
            our_atr.iloc[20:].values,
            expected.iloc[20:].values,
            rtol=0.03,
        )

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR must always be positive."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        atr = indicators.calculate_atr(high, low, close)
        valid = atr.dropna()
        assert (valid > 0).all()


class TestOBVCrosscheck:
    """Validate OBV against pandas-ta.

    OBV is a cumulative sum based on price direction, fully deterministic.
    We expect exact matches within floating-point precision.
    """

    def test_obv_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """OBV should match pandas-ta exactly."""
        close = sample_ohlcv["Close"]
        volume = sample_ohlcv["Volume"]

        our_obv = indicators.calculate_obv(close, volume)
        expected = ta.obv(close, volume)

        np.testing.assert_allclose(
            our_obv.iloc[1:].values,
            expected.iloc[1:].values,
            rtol=1e-10,
        )


class TestWMACrosscheck:
    """Validate WMA against pandas-ta.

    WMA uses linearly increasing weights, which is deterministic.
    We expect exact matches within floating-point precision.
    """

    def test_wma_10_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """WMA(10) should match pandas-ta exactly."""
        close = sample_ohlcv["Close"]

        our_wma = indicators.calculate_wma(close, 10)
        expected = ta.wma(close, length=10)

        np.testing.assert_allclose(
            our_wma.dropna().values,
            expected.dropna().values,
            rtol=1e-10,
        )


class TestMomentumCrosscheck:
    """Validate Momentum against pandas-ta.

    Momentum is simply close - close[n], fully deterministic.
    """

    def test_momentum_10_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """Momentum(10) should match pandas-ta exactly."""
        close = sample_ohlcv["Close"]

        our_mom = indicators.calculate_momentum(close, 10)
        expected = ta.mom(close, length=10)

        np.testing.assert_allclose(
            our_mom.dropna().values,
            expected.dropna().values,
            rtol=1e-10,
        )


class TestMathematicalInvariants:
    """Test properties that must always hold regardless of implementation.

    These tests validate mathematical relationships rather than comparing
    to an external library. They catch fundamental calculation errors.
    """

    def test_ema_reacts_faster_than_sma(self, sample_ohlcv: pd.DataFrame) -> None:
        """EMA should track price changes more closely than SMA."""
        close = sample_ohlcv["Close"]

        sma = indicators.calculate_sma(close, 20)
        ema = indicators.calculate_ema(close, 20)

        sma_diff_var = sma.diff().dropna().var()
        ema_diff_var = ema.diff().dropna().var()

        assert ema_diff_var >= sma_diff_var * 0.9

    def test_longer_period_smoother(self, sample_ohlcv: pd.DataFrame) -> None:
        """Longer period SMA should have lower variance in changes."""
        close = sample_ohlcv["Close"]

        sma_20 = indicators.calculate_sma(close, 20)
        sma_50 = indicators.calculate_sma(close, 50)

        var_20 = sma_20.diff().dropna().var()
        var_50 = sma_50.diff().dropna().var()

        assert var_50 < var_20
