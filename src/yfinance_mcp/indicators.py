"""Technical indicator calculations."""

import logging

import numpy as np
import pandas as pd

from .errors import CalculationError

logger = logging.getLogger("yfinance_mcp.indicators")


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        logger.warning(
            "indicator_insufficient_data type=sma required=%d available=%d", period, len(prices)
        )
        raise CalculationError(
            f"Insufficient data: need {period} periods, got {len(prices)}",
            {"required": period, "available": len(prices)},
        )
    logger.debug("calculate_sma period=%d data_points=%d", period, len(prices))
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        logger.warning(
            "indicator_insufficient_data type=ema required=%d available=%d", period, len(prices)
        )
        raise CalculationError(
            f"Insufficient data: need {period} periods, got {len(prices)}",
            {"required": period, "available": len(prices)},
        )
    logger.debug("calculate_ema period=%d data_points=%d", period, len(prices))
    return prices.ewm(span=period, adjust=False).mean()


def calculate_wma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Weighted Moving Average.
    
    Uses linearly increasing weights: for period 5, weights are [1, 2, 3, 4, 5].
    More recent prices have higher weight.
    """
    if len(prices) < period:
        logger.warning(
            "indicator_insufficient_data type=wma required=%d available=%d", period, len(prices)
        )
        raise CalculationError(
            f"Insufficient data: need {period} periods, got {len(prices)}",
            {"required": period, "available": len(prices)},
        )
    logger.debug("calculate_wma period=%d data_points=%d", period, len(prices))
    
    weights = np.arange(1, period + 1)
    
    def weighted_avg(x: np.ndarray) -> float:
        return np.sum(weights * x) / np.sum(weights)
    
    return prices.rolling(window=period).apply(weighted_avg, raw=True)


def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Momentum indicator.
    
    Momentum = current_close - close_n_periods_ago
    Positive values indicate upward momentum, negative indicates downward.
    """
    if len(prices) < period + 1:
        logger.warning(
            "indicator_insufficient_data type=momentum required=%d available=%d",
            period + 1,
            len(prices),
        )
        raise CalculationError(
            f"Insufficient data: need {period + 1} periods, got {len(prices)}",
            {"required": period + 1, "available": len(prices)},
        )
    logger.debug("calculate_momentum period=%d data_points=%d", period, len(prices))
    return prices - prices.shift(period)


def calculate_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Calculate Commodity Channel Index.
    
    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    where TP = (High + Low + Close) / 3
    
    Values > 100 indicate overbought, < -100 indicate oversold.
    """
    if len(close) < period:
        logger.warning(
            "indicator_insufficient_data type=cci required=%d available=%d",
            period,
            len(close),
        )
        raise CalculationError(
            f"Insufficient data: need {period} periods, got {len(close)}",
            {"required": period, "available": len(close)},
        )
    logger.debug("calculate_cci period=%d data_points=%d", period, len(close))
    
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci


def calculate_dmi(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> dict[str, pd.Series]:
    """Calculate Directional Movement Index.
    
    Returns +DI, -DI, and ADX (Average Directional Index).
    ADX > 25 indicates strong trend, < 20 indicates weak/no trend.
    +DI > -DI suggests uptrend, -DI > +DI suggests downtrend.
    """
    min_periods = period * 2
    if len(close) < min_periods:
        logger.warning(
            "indicator_insufficient_data type=dmi required=%d available=%d",
            min_periods,
            len(close),
        )
        raise CalculationError(
            f"Insufficient data: need {min_periods} periods, got {len(close)}",
            {"required": min_periods, "available": len(close)},
        )
    logger.debug("calculate_dmi period=%d data_points=%d", period, len(close))
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return {"plus_di": plus_di, "minus_di": minus_di, "adx": adx}


def calculate_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate Williams %R.
    
    Williams %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100
    
    Values range from -100 to 0. Above -20 is overbought, below -80 is oversold.
    """
    if len(close) < period:
        logger.warning(
            "indicator_insufficient_data type=williams required=%d available=%d",
            period,
            len(close),
        )
        raise CalculationError(
            f"Insufficient data: need {period} periods, got {len(close)}",
            {"required": period, "available": len(close)},
        )
    logger.debug("calculate_williams period=%d data_points=%d", period, len(close))
    
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    return williams_r


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1:
        logger.warning(
            "indicator_insufficient_data type=rsi required=%d available=%d",
            period + 1,
            len(prices),
        )
        raise CalculationError(
            f"Insufficient data: need {period + 1} periods, got {len(prices)}",
            {"required": period + 1, "available": len(prices)},
        )
    logger.debug("calculate_rsi period=%d data_points=%d", period, len(prices))

    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    min_periods = slow_period + signal_period
    if len(prices) < min_periods:
        logger.warning(
            "indicator_insufficient_data type=macd required=%d available=%d",
            min_periods,
            len(prices),
        )
        raise CalculationError(
            f"Insufficient data: need {min_periods} periods, got {len(prices)}",
            {"required": min_periods, "available": len(prices)},
        )
    logger.debug(
        "calculate_macd fast=%d slow=%d signal=%d data_points=%d",
        fast_period,
        slow_period,
        signal_period,
        len(prices),
    )

    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2.0
) -> dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        logger.warning(
            "indicator_insufficient_data type=bb required=%d available=%d", period, len(prices)
        )
        raise CalculationError(
            f"Insufficient data: need {period} periods, got {len(prices)}",
            {"required": period, "available": len(prices)},
        )
    logger.debug("calculate_bb period=%d std_dev=%.1f data_points=%d", period, std_dev, len(prices))

    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    bandwidth = (upper - lower) / middle * 100
    percent_b = (prices - lower) / (upper - lower)

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b,
    }


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, pd.Series]:
    """Calculate Slow Stochastic Oscillator.
    
    Slow Stochastic applies smoothing to %K before computing %D.
    - %K = SMA of Fast %K (raw stochastic)
    - %D = SMA of %K
    """
    min_periods = k_period + d_period
    if len(close) < min_periods:
        logger.warning(
            "indicator_insufficient_data type=stoch required=%d available=%d",
            min_periods,
            len(close),
        )
        raise CalculationError(
            f"Insufficient data: need {min_periods} periods, got {len(close)}",
            {"required": min_periods, "available": len(close)},
        )
    logger.debug("calculate_stoch k=%d d=%d data_points=%d", k_period, d_period, len(close))

    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(window=d_period).mean()
    d = k.rolling(window=d_period).mean()

    return {"k": k, "d": d}


def calculate_fast_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, pd.Series]:
    """Calculate Fast Stochastic Oscillator.
    
    Fast Stochastic uses the raw (unsmoothed) %K.
    - %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    - %D = SMA of %K
    
    More responsive to price changes than Slow Stochastic.
    """
    min_periods = k_period + d_period
    if len(close) < min_periods:
        logger.warning(
            "indicator_insufficient_data type=fast_stoch required=%d available=%d",
            min_periods,
            len(close),
        )
        raise CalculationError(
            f"Insufficient data: need {min_periods} periods, got {len(close)}",
            {"required": min_periods, "available": len(close)},
        )
    logger.debug("calculate_fast_stoch k=%d d=%d data_points=%d", k_period, d_period, len(close))

    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return {"k": k, "d": d}


def calculate_ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    conversion_period: int = 9,
    base_period: int = 26,
    leading_b_period: int = 52,
) -> dict[str, pd.Series]:
    """Calculate Ichimoku Kinko Hyo (Equilibrium Chart) components.
    
    Returns:
    - conversion_line: 9-period mid-price (fast signal line)
    - base_line: 26-period mid-price (slow signal line)
    - leading_span_a: Average of Conversion and Base, shifted 26 periods ahead
    - leading_span_b: 52-period mid-price, shifted 26 periods ahead
    - lagging_span: Close shifted back 26 periods
    
    Cloud is bullish when Leading Span A > Leading Span B, bearish otherwise.
    """
    min_periods = leading_b_period + base_period
    if len(close) < min_periods:
        logger.warning(
            "indicator_insufficient_data type=ichimoku required=%d available=%d",
            min_periods,
            len(close),
        )
        raise CalculationError(
            f"Insufficient data: need {min_periods} periods, got {len(close)}",
            {"required": min_periods, "available": len(close)},
        )
    logger.debug(
        "calculate_ichimoku conversion=%d base=%d leading_b=%d data_points=%d",
        conversion_period,
        base_period,
        leading_b_period,
        len(close),
    )
    
    def midprice(h: pd.Series, l: pd.Series, period: int) -> pd.Series:
        return (h.rolling(window=period).max() + l.rolling(window=period).min()) / 2
    
    conversion_line = midprice(high, low, conversion_period)
    base_line = midprice(high, low, base_period)
    leading_span_a = ((conversion_line + base_line) / 2).shift(base_period)
    leading_span_b = midprice(high, low, leading_b_period).shift(base_period)
    lagging_span = close.shift(-base_period)
    
    return {
        "conversion_line": conversion_line,
        "base_line": base_line,
        "leading_span_a": leading_span_a,
        "leading_span_b": leading_span_b,
        "lagging_span": lagging_span,
    }


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    if len(close) < period + 1:
        logger.warning(
            "indicator_insufficient_data type=atr required=%d available=%d",
            period + 1,
            len(close),
        )
        raise CalculationError(
            f"Insufficient data: need {period + 1} periods, got {len(close)}",
            {"required": period + 1, "available": len(close)},
        )
    logger.debug("calculate_atr period=%d data_points=%d", period, len(close))

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On Balance Volume."""
    if len(close) != len(volume):
        logger.warning(
            "indicator_data_mismatch type=obv close_len=%d volume_len=%d", len(close), len(volume)
        )
        raise CalculationError(
            "Close and volume series must have same length",
            {"close_len": len(close), "volume_len": len(volume)},
        )

    # NaN volume contributes nothing to OBV
    nan_count = volume.isna().sum()
    clean_volume = volume.fillna(0)

    logger.debug("calculate_obv data_points=%d nan_filled=%d", len(close), nan_count)
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * clean_volume).cumsum()


def calculate_fibonacci_levels(
    high: float, low: float, is_uptrend: bool = True
) -> dict[str, float]:
    """Calculate Fibonacci retracement and extension levels."""
    diff = high - low

    if is_uptrend:
        levels = {
            "level_0": high,
            "level_236": high - diff * 0.236,
            "level_382": high - diff * 0.382,
            "level_500": high - diff * 0.500,
            "level_618": high - diff * 0.618,
            "level_786": high - diff * 0.786,
            "level_100": low,
            "ext_1272": low - diff * 0.272,
            "ext_1618": low - diff * 0.618,
            "ext_200": low - diff * 1.0,
            "ext_2618": low - diff * 1.618,
        }
    else:
        levels = {
            "level_0": low,
            "level_236": low + diff * 0.236,
            "level_382": low + diff * 0.382,
            "level_500": low + diff * 0.500,
            "level_618": low + diff * 0.618,
            "level_786": low + diff * 0.786,
            "level_100": high,
            "ext_1272": high + diff * 0.272,
            "ext_1618": high + diff * 0.618,
            "ext_200": high + diff * 1.0,
            "ext_2618": high + diff * 1.618,
        }

    return levels


def calculate_pivot_points(
    high: float, low: float, close: float, method: str = "standard"
) -> dict[str, float]:
    """Calculate pivot points and support/resistance levels."""
    if method == "standard":
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
    elif method == "fibonacci":
        pivot = (high + low + close) / 3
        diff = high - low
        r1 = pivot + diff * 0.382
        s1 = pivot - diff * 0.382
        r2 = pivot + diff * 0.618
        s2 = pivot - diff * 0.618
        r3 = pivot + diff
        s3 = pivot - diff
    elif method == "camarilla":
        pivot = (high + low + close) / 3
        diff = high - low
        r1 = close + diff * 1.1 / 12
        s1 = close - diff * 1.1 / 12
        r2 = close + diff * 1.1 / 6
        s2 = close - diff * 1.1 / 6
        r3 = close + diff * 1.1 / 4
        s3 = close - diff * 1.1 / 4
    elif method == "woodie":
        pivot = (high + low + 2 * close) / 4
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
    else:
        logger.warning("indicator_invalid_method type=pivot method=%s", method)
        raise CalculationError(
            f"Unknown pivot point method: {method}",
            {"valid_methods": ["standard", "fibonacci", "camarilla", "woodie"]},
        )
    logger.debug(
        "calculate_pivot method=%s high=%.2f low=%.2f close=%.2f", method, high, low, close
    )

    return {
        "pivot": pivot,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }
