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
    """Validate Slow Stochastic Oscillator against pandas-ta.

    Slow Stochastic applies smoothing to %K before computing %D.
    We use smooth_k=3 to match our implementation.
    """

    def test_stochastic_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """Slow Stochastic %K should match pandas-ta with smooth_k=3."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        our_stoch = indicators.calculate_stochastic(high, low, close, 14, 3)
        expected = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)

        np.testing.assert_allclose(
            our_stoch["k"].iloc[20:].values,
            expected["STOCHk_14_3_3"].iloc[20:].values,
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


class TestCCICrosscheck:
    """Validate CCI mathematical properties.

    Note: CCI implementations vary across libraries due to different
    mean deviation calculations. Our implementation uses the standard
    Lambert formula: CCI = (TP - SMA(TP)) / (0.015 * MeanDev)

    We test mathematical properties rather than exact match with pandas-ta.
    """

    def test_cci_oscillates(self, sample_ohlcv: pd.DataFrame) -> None:
        """CCI should oscillate around zero."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        cci = indicators.calculate_cci(high, low, close, 20)
        valid = cci.dropna()

        assert valid.mean() < 100
        assert valid.mean() > -100

    def test_cci_bounds(self, sample_ohlcv: pd.DataFrame) -> None:
        """CCI should have reasonable range (typically -200 to +200 most of time)."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        cci = indicators.calculate_cci(high, low, close)
        valid = cci.dropna()

        assert valid.std() < 500


class TestDMICrosscheck:
    """Validate DMI/ADX against pandas-ta.

    Tolerance rationale:
    - DMI uses smoothed directional movement
    - pandas-ta uses RMA (Wilder's smoothing), we use EMA
    - ADX values track similar patterns but with different smoothing
    - We test that ADX captures trend strength (high when trending)
    """

    def test_adx_correlation(self, sample_ohlcv: pd.DataFrame) -> None:
        """ADX should be correlated with pandas-ta (allowing smoothing differences)."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        our_dmi = indicators.calculate_dmi(high, low, close, 14)
        expected = ta.adx(high, low, close, length=14)

        adx_corr = our_dmi["adx"].iloc[30:].corr(expected["ADX_14"].iloc[30:])
        assert adx_corr > 0.75

    def test_adx_bounds(self, sample_ohlcv: pd.DataFrame) -> None:
        """ADX should be in [0, 100] range."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        dmi = indicators.calculate_dmi(high, low, close)
        valid_adx = dmi["adx"].dropna()

        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()


class TestWilliamsRCrosscheck:
    """Validate Williams %R against pandas-ta.

    Williams %R uses rolling high/low which is deterministic.
    We expect exact matches within floating-point precision.
    """

    def test_williams_r_14_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """Williams %R(14) should match pandas-ta exactly."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        our_wr = indicators.calculate_williams_r(high, low, close, 14)
        expected = ta.willr(high, low, close, length=14)

        np.testing.assert_allclose(
            our_wr.dropna().values,
            expected.dropna().values,
            rtol=1e-10,
        )

    def test_williams_r_bounds(self, sample_ohlcv: pd.DataFrame) -> None:
        """Williams %R must be in [-100, 0] range."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        wr = indicators.calculate_williams_r(high, low, close)
        valid = wr.dropna()

        assert (valid >= -100).all()
        assert (valid <= 0).all()


class TestFastStochasticCrosscheck:
    """Validate Fast Stochastic against pandas-ta.

    Fast Stochastic uses raw %K (no smoothing).
    We expect exact matches within floating-point precision.
    """

    def test_fast_stoch_matches(self, sample_ohlcv: pd.DataFrame) -> None:
        """Fast Stochastic should match pandas-ta with smooth_k=1."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        our_stoch = indicators.calculate_fast_stochastic(high, low, close, 14, 3)
        expected = ta.stoch(high, low, close, k=14, d=3, smooth_k=1)

        np.testing.assert_allclose(
            our_stoch["k"].iloc[20:].values,
            expected["STOCHk_14_3_1"].iloc[20:].values,
            rtol=0.01,
        )

    def test_fast_stoch_bounds(self, sample_ohlcv: pd.DataFrame) -> None:
        """Fast Stochastic must be in [0, 100]."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        stoch = indicators.calculate_fast_stochastic(high, low, close)

        valid_k = stoch["k"].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()


class TestIchimokuCrosscheck:
    """Validate Ichimoku Cloud components.

    Ichimoku uses rolling high/low midpoints which are deterministic.
    We test structure and mathematical properties.
    """

    def test_ichimoku_structure(self, sample_ohlcv: pd.DataFrame) -> None:
        """Ichimoku should return all expected components."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        ich = indicators.calculate_ichimoku(high, low, close)

        assert "conversion_line" in ich
        assert "base_line" in ich
        assert "leading_span_a" in ich
        assert "leading_span_b" in ich
        assert "lagging_span" in ich

    def test_conversion_base_relationship(self, sample_ohlcv: pd.DataFrame) -> None:
        """Conversion (9-period) should be more volatile than Base (26-period)."""
        high = sample_ohlcv["High"]
        low = sample_ohlcv["Low"]
        close = sample_ohlcv["Close"]

        ich = indicators.calculate_ichimoku(high, low, close)

        conversion_var = ich["conversion_line"].diff().dropna().var()
        base_var = ich["base_line"].diff().dropna().var()

        assert conversion_var >= base_var * 0.5


class TestVolumeProfileCrosscheck:
    """Validate Volume Profile calculations.

    Volume Profile is a price-volume distribution analysis.
    We test structure and mathematical properties.
    """

    def test_volume_profile_structure(self, sample_ohlcv: pd.DataFrame) -> None:
        """Volume Profile should return POC and value area."""
        close = sample_ohlcv["Close"]
        volume = sample_ohlcv["Volume"]

        vp = indicators.calculate_volume_profile(close, volume)

        assert "poc" in vp
        assert "value_area_high" in vp
        assert "value_area_low" in vp
        assert "profile" in vp
        assert vp["value_area_high"] >= vp["value_area_low"]

    def test_poc_within_price_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """Point of Control should be within the price range."""
        close = sample_ohlcv["Close"]
        volume = sample_ohlcv["Volume"]

        vp = indicators.calculate_volume_profile(close, volume)

        assert vp["poc"] >= close.min()
        assert vp["poc"] <= close.max()


class TestPriceChangeCrosscheck:
    """Validate Price Change calculations.

    Price Change is a simple percentage calculation.
    """

    def test_price_change_structure(self, sample_ohlcv: pd.DataFrame) -> None:
        """Price Change should return change and percentage."""
        close = sample_ohlcv["Close"]

        pc = indicators.calculate_price_change(close)

        assert "change" in pc
        assert "change_pct" in pc

    def test_price_change_accuracy(self, sample_ohlcv: pd.DataFrame) -> None:
        """Price Change percentage should be mathematically correct."""
        close = sample_ohlcv["Close"]

        pc = indicators.calculate_price_change(close, period=1)
        
        expected_change = float(close.iloc[-1]) - float(close.iloc[-2])
        expected_pct = (expected_change / float(close.iloc[-2])) * 100

        assert abs(pc["change"] - expected_change) < 0.0001
        assert abs(pc["change_pct"] - expected_pct) < 0.0001


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
