"""Oscillator tool — momentum indicators for a symbol."""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from history import fetch_ohlcv

logger = logging.getLogger(__name__)

WARMUP_CALENDAR_DAYS = 112

OPTION_CHAIN_MAX_WORKERS = 8


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    k = 100 * (close - lowest) / (highest - lowest)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def _fetch_put_call_ratio(symbol: str) -> dict[str, Any] | None:
    """Aggregate Put/Call Ratio across all listed expirations.

    Returns None for tickers with no listed options (non-US equities,
    most indexes, crypto) or when every expiration fetch fails.
    """
    # yfinance option chains exist only for US-listed equities/ETFs.
    # An exchange suffix (e.g. ".T", ".HK", ".L") reliably rules out options.
    if "." in symbol:
        return None

    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
    except Exception as e:
        logger.warning("put/call ratio fetch failed for %s: %s", symbol, e)
        return None

    if not expirations:
        return None

    def _fetch_one(exp: str) -> Any:
        try:
            return ticker.option_chain(exp)
        except Exception as e:
            logger.warning("option chain fetch failed for %s %s: %s", symbol, exp, e)
            return None

    with ThreadPoolExecutor(max_workers=OPTION_CHAIN_MAX_WORKERS) as pool:
        chains = list(pool.map(_fetch_one, expirations))

    call_vol = put_vol = 0
    call_oi = put_oi = 0
    counted = 0
    for chain in chains:
        if chain is None:
            continue
        call_vol += int(chain.calls["volume"].fillna(0).sum())
        put_vol += int(chain.puts["volume"].fillna(0).sum())
        call_oi += int(chain.calls["openInterest"].fillna(0).sum())
        put_oi += int(chain.puts["openInterest"].fillna(0).sum())
        counted += 1

    if counted == 0:
        return None

    return {
        "as_of": date.today().isoformat(),
        "volume_based": round(put_vol / call_vol, 4) if call_vol else None,
        "oi_based": round(put_oi / call_oi, 4) if call_oi else None,
        "call_volume": call_vol,
        "put_volume": put_vol,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "expirations": counted,
    }


def oscillator(
    symbol: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Compute momentum oscillators for a symbol using daily bars.

    Fetches extra warmup bars before *start* so all indicators are valid
    from the first returned timestamp onward.

    Args:
        symbol: Ticker symbol.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Columnar dict with symbol, interval, tz, t, oscillator values,
        and a ``put_call_ratio`` snapshot (or ``None`` for tickers
        without listed options — non-US equities, indexes, crypto).

    Raises:
        ValueError: If no data is returned.
    """
    warmup_start = (
        date.fromisoformat(start) - timedelta(days=WARMUP_CALENDAR_DAYS)
    ).isoformat()
    logger.debug("oscillator %s warmup=%s..%s", symbol, warmup_start, end)

    df = fetch_ohlcv(symbol, "1d", warmup_start, end, adjust=True)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    rsi = _rsi(close)
    stoch_k, stoch_d = _stochastic(high, low, close)

    indicators = pd.DataFrame(
        {
            "rsi": rsi,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
        },
        index=df.index,
    )

    if "Date" in df.columns:
        timestamps = [d.isoformat() for d in df["Date"]]
        mask = df["Date"] >= date.fromisoformat(start)
    else:
        timestamps = [ts.strftime("%Y-%m-%d") for ts in df.index]
        mask = df.index >= pd.Timestamp(start, tz=df.index.tz)
    tz = str(df.index.tz) if hasattr(df.index, "tz") and df.index.tz else "UTC"

    indicators["_t"] = timestamps
    trimmed = indicators.loc[mask].dropna()

    if trimmed.empty:
        raise ValueError(f"No oscillator data for '{symbol}' from {start} to {end}")

    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": "1d",
        "tz": tz,
        "t": trimmed["_t"].tolist(),
    }
    indicator_cols = [
        "rsi",
        "stoch_k",
        "stoch_d",
    ]
    for col in indicator_cols:
        result[col] = [round(v, 2) for v in trimmed[col]]

    result["put_call_ratio"] = _fetch_put_call_ratio(symbol)

    return result
