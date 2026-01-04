"""Yahoo Finance MCP Server"""

import contextvars
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import history, indicators
from .cache import get_cache_stats
from .errors import (
    CalculationError,
    DataUnavailableError,
    MCPError,
    SymbolNotFoundError,
    ValidationError,
)
from .helpers import (
    INTRADAY_INTERVALS,
    MAX_SPAN_DAYS,
    OHLCV_COLS_TO_SHORT,
    TARGET_POINTS,
    DateRangeExceededError,
    adaptive_decimals,
    add_unknown,
    calculate_quality,
    configure_logging,
    err,
    fmt,
    fmt_toon,
    get_default_log_path,
    get_valid_periods,
    lttb_downsample,
    normalize_df,
    normalize_tz,
    ohlc_resample,
    parse_moving_avg_period,
    period_to_date_range,
    round_result,
    safe_get,
    safe_gt,
    safe_scalar,
    select_interval,
    smart_search,
    summarize_args,
    to_scalar,
    validate_date_range,
)

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_stats_lock = threading.Lock()
_debug_stats: dict[str, int | float] = {
    "calls": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_time_ms": 0.0,
}


def _get_stats_snapshot() -> dict:
    """Return a thread-safe copy of debug stats."""
    with _stats_lock:
        return _debug_stats.copy()


def _update_stats(
    increment: dict[str, int | float] | None = None, set_values: dict[str, int] | None = None
) -> None:
    """Thread-safe stats update. Supports both incrementing and setting values."""
    with _stats_lock:
        if increment:
            for key, value in increment.items():
                if key in _debug_stats:
                    _debug_stats[key] += value
        if set_values:
            for key, value in set_values.items():
                if key in _debug_stats:
                    _debug_stats[key] = value


logger = configure_logging(
    request_id_getter=lambda: _request_id.get(),
    stats_getter=_get_stats_snapshot,
)

server = Server("yfinance-mcp")

_cb: dict = {
    "fails": 0,
    "open": False,
    "threshold": 5,
    "recovery_timeout": 30,
    "opened_at": None,
}


def _check_cb() -> None:
    """Check circuit breaker state, with auto-recovery after timeout."""
    if not _cb["open"]:
        return

    if _cb["opened_at"] is not None:
        elapsed = (datetime.now() - _cb["opened_at"]).total_seconds()
        if elapsed >= _cb["recovery_timeout"]:
            logger.info(
                "circuit_breaker recovery_attempt elapsed=%.1fs threshold=%d",
                elapsed,
                _cb["threshold"],
            )
            _cb["open"] = False
            _cb["opened_at"] = None
            return

    logger.debug("circuit_breaker blocked request fails=%d", _cb["fails"])
    raise DataUnavailableError("Service temporarily unavailable, retry later")


def _record_fail() -> None:
    """Record a failure and potentially open circuit breaker."""
    _cb["fails"] += 1
    if _cb["fails"] >= _cb["threshold"]:
        _cb["open"] = True
        _cb["opened_at"] = datetime.now()
        logger.warning(
            "circuit_breaker_opened fails=%d threshold=%d", _cb["fails"], _cb["threshold"]
        )


def _reset_cb() -> None:
    """Reset circuit breaker on success."""
    _cb["fails"] = 0
    _cb["open"] = False
    _cb["opened_at"] = None


def reset_circuit_breaker_for_testing() -> None:
    """Test-only hook to reset circuit breaker state.

    This provides a clean interface for tests without exposing internal state.
    """
    _reset_cb()


def open_circuit_breaker_for_testing() -> None:
    """Test-only hook to open circuit breaker.

    Simulates threshold failures without actually calling the API.
    """
    for _ in range(_cb["threshold"]):
        _record_fail()


def _ticker(symbol: str) -> yf.Ticker:
    """Get validated ticker."""
    _check_cb()
    if not symbol or not isinstance(symbol, str):
        logger.warning("ticker_validation failed symbol=%r", symbol)
        raise ValidationError("Invalid symbol")

    symbol_upper = symbol.upper().strip()
    logger.debug("ticker_lookup symbol=%s", symbol_upper)

    t = yf.Ticker(symbol_upper)
    try:
        fi = t.fast_info
    except Exception as e:
        if isinstance(e, KeyError):
            logger.warning("ticker_not_found symbol=%s keyerror=%s", symbol_upper, e)
            raise SymbolNotFoundError(symbol)
        if "symbol" in str(e).lower() or "not found" in str(e).lower():
            logger.warning("ticker_not_found symbol=%s error=%s", symbol_upper, e)
            raise SymbolNotFoundError(symbol)
        logger.warning("ticker_api_error symbol=%s error=%s", symbol_upper, e)
        _record_fail()
        raise

    if fi is None:
        logger.warning("ticker_not_found symbol=%s fast_info=None", symbol_upper)
        raise SymbolNotFoundError(symbol)

    try:
        _ = fi.last_price
    except KeyError as e:
        logger.warning("ticker_not_found symbol=%s keyerror=%s", symbol_upper, e)
        raise SymbolNotFoundError(symbol)

    _reset_cb()
    logger.debug("ticker_validated symbol=%s", symbol_upper)
    return t


def _require_symbol(args: dict) -> tuple[str, yf.Ticker]:
    """Validate symbol argument and return (symbol, ticker) tuple."""
    symbol = args.get("symbol")
    if not symbol:
        raise ValidationError("symbol required")
    return symbol, _ticker(symbol)


def _add_unknown(result: dict, indicator: str) -> None:
    """Add indicator to _unknown list in result dict."""
    add_unknown(result, indicator, logger)


TOOLS = [
    Tool(
        name="search_stock",
        description=(
            "Find stock by symbol or company name. "
            "Returns identity (name, sector, industry, exchange, currency) "
            "and current price snapshot (price, change, change_pct, market_cap, volume)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": (
                        "Stock ticker (e.g., AAPL, MSFT, 7203.T). "
                        "Required if query is not provided."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Company name to search (e.g., 'Apple', 'Tesla'). "
                        "Required if symbol is not provided. "
                        "Works globally: US, Japan (.T), Germany (.DE), Singapore (.SI), etc. "
                        "Tip: Use core name without suffixes for best results "
                        "(e.g., 'Toyota' instead of 'Toyota Motor Corporation')."
                    ),
                },
            },
        },
    ),
    Tool(
        name="history",
        description=(
            f"Historical OHLCV bars. Returns ~{TARGET_POINTS} data points. "
            f"Max range: {MAX_SPAN_DAYS} days (~{round(MAX_SPAN_DAYS / 365, 1)} years). "
            "For longer periods, split into multiple sequential requests."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker"},
                "period": {
                    "type": "string",
                    "default": "3mo",
                    "enum": get_valid_periods(MAX_SPAN_DAYS),
                    "description": "Relative period. Ignored if start provided.",
                },
                "start": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD).",
                },
                "end": {
                    "type": "string",
                    "description": "End date. Defaults to today.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="technicals",
        description=(
            f"Technical indicators and signals. Returns ~{TARGET_POINTS} data points. "
            f"Max range: {MAX_SPAN_DAYS} days (~{round(MAX_SPAN_DAYS / 365, 1)} years). "
            "For longer periods, split into multiple sequential requests. "
            "trend: SMA50-based trend direction. "
            "rsi: >70 overbought, <30 oversold. "
            "macd: histogram>0 bullish. "
            "sma_N, ema_N, wma_N: moving averages. "
            "bb: Bollinger Bands. "
            "stoch, fast_stoch: Stochastic oscillators, >80 overbought, <20 oversold. "
            "cci: >100 overbought, <-100 oversold. "
            "dmi: ADX>25 strong trend. "
            "williams: >-20 overbought, <-80 oversold. "
            "ichimoku: cloud analysis. "
            "atr: volatility. obv: volume trend. "
            "momentum, price_change: rate of change. "
            "volume_profile: price-level activity. "
            "fibonacci, pivot: support/resistance levels."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker"},
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["all"],
                    "description": (
                        "Default: ['all']. "
                        "Options: all, trend, rsi, macd, sma_N, ema_N, wma_N, bb, stoch, "
                        "fast_stoch, cci, dmi, williams, ichimoku, atr, obv, "
                        "momentum, volume_profile, price_change, fibonacci, pivot"
                    ),
                },
                "period": {
                    "type": "string",
                    "default": "1y",
                    "enum": get_valid_periods(MAX_SPAN_DAYS),
                    "description": "Relative period. Ignored if start provided.",
                },
                "start": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD).",
                },
                "end": {
                    "type": "string",
                    "description": "End date. Defaults to today.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="valuation",
        description=(
            "Valuation metrics and financial quality. "
            "pe: P/E ratios. eps: earnings per share. "
            "peg: PE-to-growth ratio (<1 undervalued, >2 overvalued). "
            "margins: gross/operating/net. growth: revenue/earnings. "
            "ratios: P/B, P/S, EV/EBITDA. dividends: yield, rate, payout. "
            "quality: 0-7 score based on ROA, cash flow, liquidity, leverage, margins, ROE. "
            "Use periods parameter for historical valuation at fiscal year or quarter ends."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["all"],
                    "description": (
                        "Default: ['all']. "
                        "Options: all, pe, eps, peg, margins, growth, ratios, dividends, quality"
                    ),
                },
                "periods": {
                    "type": "string",
                    "default": "now",
                    "description": (
                        '"now" for current (default), "YYYY" for fiscal year, '
                        '"YYYY-QN" for quarter, or ranges like "YYYY:YYYY"'
                    ),
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="financials",
        description=(
            "Financial statements: income, balance, cashflow. "
            "Raw data rows. Use 'fields' to filter specific items."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker"},
                "statement": {
                    "type": "string",
                    "default": "income",
                    "enum": ["income", "balance", "cashflow"],
                    "description": "Statement type to retrieve.",
                },
                "freq": {
                    "type": "string",
                    "default": "annual",
                    "enum": ["annual", "quarterly"],
                    "description": "Reporting frequency.",
                },
                "periods": {
                    "type": "string",
                    "default": "now",
                    "description": (
                        '"now" for current (default), "YYYY" for fiscal year, '
                        '"YYYY-QN" for quarter, or ranges like "YYYY:YYYY"'
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of rows (max 100).",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter rows (e.g., ['TotalRevenue', 'NetIncome'])",
                },
            },
            "required": ["symbol"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    req_id = uuid.uuid4().hex[:12]
    _request_id.set(req_id)

    start_time = time.time()
    _update_stats(increment={"calls": 1})
    stats = _get_stats_snapshot()
    logger.debug(
        "tool_call_start name=%s args=%s call=%d",
        name,
        summarize_args(arguments),
        stats["calls"],
    )

    try:
        result = await _execute(name, arguments)
        elapsed_ms = (time.time() - start_time) * 1000
        cache_stats = get_cache_stats()
        _update_stats(
            increment={"total_time_ms": elapsed_ms},
            set_values={"cache_hits": cache_stats["hits"], "cache_misses": cache_stats["misses"]},
        )
        hit_rate = round(
            cache_stats["hits"] / max(1, cache_stats["hits"] + cache_stats["misses"]) * 100,
            1,
        )
        logger.info(
            "tool_call_success name=%s elapsed_ms=%.1f result_len=%d hit_rate=%.1f%%",
            name,
            elapsed_ms,
            len(result),
            hit_rate,
        )
        return [TextContent(type="text", text=result)]
    except DateRangeExceededError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        max_years = e.max_days / 365
        logger.warning(
            "tool_call_error name=%s code=DATE_RANGE_EXCEEDED elapsed_ms=%.1f msg=%s",
            name,
            elapsed_ms,
            str(e),
        )
        validation_err = ValidationError(
            str(e),
            {
                "requested_days": e.requested_days,
                "max_days": e.max_days,
                "max_years": round(max_years, 1),
                "hint": f"Use period='{e.suggested_period}' or "
                f"specify start/end within {round(max_years, 1)} years.",
            },
        )
        return [TextContent(type="text", text=err(validation_err))]
    except MCPError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning(
            "tool_call_error name=%s code=%s elapsed_ms=%.1f msg=%s",
            name,
            e.code,
            elapsed_ms,
            e.message,
        )
        return [TextContent(type="text", text=err(e))]
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "tool_call_exception name=%s elapsed_ms=%.1f error=%s",
            name,
            elapsed_ms,
            e,
            exc_info=True,
        )
        return [TextContent(type="text", text=err(e))]
    finally:
        _request_id.set(None)


def _handle_search_stock(args: dict) -> str:
    """Handle search_stock tool - find stock and return identity + current price."""
    symbol = args.get("symbol")
    query = args.get("query")

    if not symbol and not query:
        raise ValidationError("Either symbol or query required")

    if query and not symbol:
        quotes = smart_search(query, max_results=1, logger=logger)
        if not quotes:
            raise SymbolNotFoundError(query)
        symbol = quotes[0].get("symbol")
        if not symbol:
            raise SymbolNotFoundError(query)

    logger.debug("search_stock symbol=%s query=%s", symbol, query)

    t = _ticker(symbol)
    try:
        fi = t.fast_info
        info = t.info
        if not info or safe_get(info, "regularMarketPrice") is None:
            raise SymbolNotFoundError(symbol)
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("search_stock_invalid symbol=%s error=%s", symbol, e)
        raise SymbolNotFoundError(symbol)

    try:
        last_price = safe_scalar(fi.last_price)
        prev_close = safe_scalar(fi.previous_close)
        market_cap = safe_scalar(fi.market_cap)
        day_high = safe_scalar(fi.day_high)
        day_low = safe_scalar(fi.day_low)
        volume = safe_scalar(fi.last_volume)
    except Exception:
        last_price = None
        prev_close = None
        market_cap = None
        day_high = None
        day_low = None
        volume = None

    price_decimals = adaptive_decimals(float(last_price)) if last_price else 2

    result = {
        "symbol": symbol.upper(),
        "name": safe_get(info, "shortName") or safe_get(info, "longName"),
        "sector": safe_get(info, "sector"),
        "industry": safe_get(info, "industry"),
        "exchange": safe_get(info, "exchange"),
        "currency": safe_get(info, "currency"),
        "price": round(last_price, price_decimals) if last_price else None,
        "change_pct": round((last_price / prev_close - 1) * 100, 2)
        if last_price and prev_close
        else None,
        "day_high": round(day_high, price_decimals) if day_high else None,
        "day_low": round(day_low, price_decimals) if day_low else None,
        "volume": int(volume) if volume else None,
        "market_cap": int(market_cap) if market_cap else None,
    }
    return fmt({k: v for k, v in result.items() if v is not None})


def _handle_history(args: dict) -> str:
    """Handle history tool - historical OHLCV data."""
    symbol, t = _require_symbol(args)
    period = args.get("period", "3mo")
    start = args.get("start")
    end = args.get("end")

    # Convert custom periods (e.g., "1w", "9mo") to date ranges
    if not start and period:
        period, start, end = period_to_date_range(period)

    validate_date_range(period, start, end)

    interval = select_interval(period, start, end)

    logger.debug(
        "price_fetch symbol=%s start=%s end=%s period=%s interval=%s",
        symbol,
        start,
        end,
        period,
        interval,
    )

    df = history.get_history(symbol, period, interval, ticker=t, start=start, end=end)
    if df.empty:
        logger.warning("price_no_data symbol=%s period=%s", symbol, period)
        raise DataUnavailableError(f"No price data for {symbol}. Try different period.")
    logger.debug("price_fetched symbol=%s bars=%d", symbol, len(df))

    last_close = df["Close"].iloc[-1] if not df.empty else 1.0
    decimals = adaptive_decimals(float(last_close))

    df = df.rename(columns=OHLCV_COLS_TO_SHORT)
    df = df[["o", "h", "l", "c", "v"]].round(decimals)

    df = ohlc_resample(df)
    logger.debug("price_resampled symbol=%s bars=%d", symbol, len(df))

    if interval in INTRADAY_INTERVALS:
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d %H:%M")
    else:
        df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")

    return fmt_toon(df, wrapper_key="bars")


ALL_INDICATORS = [
    "trend",
    "rsi",
    "macd",
    "bb",
    "stoch",
    "fast_stoch",
    "cci",
    "dmi",
    "williams",
    "ichimoku",
    "atr",
    "obv",
    "momentum",
    "volume_profile",
    "price_change",
    "fibonacci",
    "pivot",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_9",
    "ema_12",
    "ema_26",
    "ema_50",
    "wma_20",
]

ALL_METRICS = [
    "pe",
    "eps",
    "peg",
    "margins",
    "growth",
    "ratios",
    "dividends",
    "quality",
]


def _handle_technicals(args: dict) -> str:
    """Handle technicals tool - returns time series of technical indicators."""
    symbol, t = _require_symbol(args)
    period = args.get("period", "1y")
    start = args.get("start")
    end = args.get("end")

    # Convert custom periods (e.g., "1w", "9mo") to date ranges
    if not start and period:
        period, start, end = period_to_date_range(period)

    validate_date_range(period, start, end)

    inds = args.get("indicators") or ["all"]

    if "all" in inds:
        inds = ALL_INDICATORS

    interval = select_interval(period, start, end)

    logger.debug(
        "technicals_fetch symbol=%s period=%s start=%s end=%s interval=%s indicators=%s",
        symbol,
        period,
        start,
        end,
        interval,
        inds,
    )

    df = history.get_history(symbol, period, interval, ticker=t, start=start, end=end)
    if df.empty:
        logger.warning("technicals_no_data symbol=%s period=%s", symbol, period)
        raise DataUnavailableError(f"No price data for {symbol}. Try period='6mo' for more data.")
    df = normalize_df(df)
    logger.debug("technicals_data_ready symbol=%s bars=%d", symbol, len(df))

    result_df = pd.DataFrame(index=df.index)
    meta: dict[str, Any] = {}

    for ind in inds:
        try:
            if ind == "rsi":
                result_df["rsi"] = indicators.calculate_rsi(df["Close"]).round(1)

            elif ind == "macd":
                m = indicators.calculate_macd(df["Close"])
                result_df["macd"] = m["macd"].round(3)
                result_df["macd_signal"] = m["signal"].round(3)
                result_df["macd_hist"] = m["histogram"].round(3)

            elif ind.startswith("sma_"):
                p = parse_moving_avg_period(ind)
                if p is None:
                    _add_unknown(meta, ind)
                    continue
                result_df[ind] = indicators.calculate_sma(df["Close"], p).round(2)

            elif ind.startswith("ema_"):
                p = parse_moving_avg_period(ind)
                if p is None:
                    _add_unknown(meta, ind)
                    continue
                result_df[ind] = indicators.calculate_ema(df["Close"], p).round(2)

            elif ind.startswith("wma_"):
                p = parse_moving_avg_period(ind)
                if p is None:
                    _add_unknown(meta, ind)
                    continue
                result_df[ind] = indicators.calculate_wma(df["Close"], p).round(2)

            elif ind == "momentum":
                result_df["momentum"] = indicators.calculate_momentum(df["Close"]).round(2)

            elif ind == "cci":
                result_df["cci"] = indicators.calculate_cci(
                    df["High"], df["Low"], df["Close"]
                ).round(1)

            elif ind == "dmi":
                dmi = indicators.calculate_dmi(df["High"], df["Low"], df["Close"])
                result_df["dmi_plus"] = dmi["plus_di"].round(1)
                result_df["dmi_minus"] = dmi["minus_di"].round(1)
                result_df["adx"] = dmi["adx"].round(1)

            elif ind == "williams":
                result_df["williams_r"] = indicators.calculate_williams_r(
                    df["High"], df["Low"], df["Close"]
                ).round(1)

            elif ind == "bb":
                bb = indicators.calculate_bollinger_bands(df["Close"])
                result_df["bb_upper"] = bb["upper"].round(2)
                result_df["bb_middle"] = bb["middle"].round(2)
                result_df["bb_lower"] = bb["lower"].round(2)
                result_df["bb_pctb"] = bb["percent_b"].round(2)

            elif ind == "stoch":
                s = indicators.calculate_stochastic(df["High"], df["Low"], df["Close"])
                result_df["stoch_k"] = s["k"].round(1)
                result_df["stoch_d"] = s["d"].round(1)

            elif ind == "fast_stoch":
                s = indicators.calculate_fast_stochastic(df["High"], df["Low"], df["Close"])
                result_df["fast_stoch_k"] = s["k"].round(1)
                result_df["fast_stoch_d"] = s["d"].round(1)

            elif ind == "ichimoku":
                ich = indicators.calculate_ichimoku(df["High"], df["Low"], df["Close"])
                result_df["ichimoku_conversion"] = ich["conversion_line"].round(2)
                result_df["ichimoku_base"] = ich["base_line"].round(2)
                result_df["ichimoku_leading_a"] = ich["leading_span_a"].round(2)
                result_df["ichimoku_leading_b"] = ich["leading_span_b"].round(2)
                result_df["ichimoku_lagging"] = ich["lagging_span"].round(2)

            elif ind == "atr":
                atr_series = indicators.calculate_atr(df["High"], df["Low"], df["Close"])
                result_df["atr"] = atr_series.round(3)
                result_df["atr_pct"] = (atr_series / df["Close"] * 100).round(2)

            elif ind == "obv":
                result_df["obv"] = indicators.calculate_obv(df["Close"], df["Volume"]).round(0)

            elif ind == "trend":
                if len(df) >= 50:
                    sma50 = indicators.calculate_sma(df["Close"], 50)
                    result_df["sma50"] = sma50.round(2)
                else:
                    meta["_trend_error"] = f"need_50_bars_have_{len(df)}"

            elif ind == "volume_profile":
                vp = indicators.calculate_volume_profile(df["Close"], df["Volume"])
                meta["volume_profile"] = {
                    "poc": vp["poc"],
                    "value_area_high": vp["value_area_high"],
                    "value_area_low": vp["value_area_low"],
                }

            elif ind == "price_change":
                pc = indicators.calculate_price_change(df["Close"])
                meta["price_change"] = {
                    "change": round(pc["change"], 2),
                    "change_pct": round(pc["change_pct"], 2),
                }

            elif ind == "fibonacci":
                period_high = float(to_scalar(df["High"].max()))
                period_low = float(to_scalar(df["Low"].min()))
                current_close = float(to_scalar(df["Close"].iloc[-1]))
                is_uptrend = current_close > (period_high + period_low) / 2
                fib = indicators.calculate_fibonacci_levels(period_high, period_low, is_uptrend)
                meta["fibonacci"] = {
                    "trend": "uptrend" if is_uptrend else "downtrend",
                    "levels": {k: round(v, 2) for k, v in fib.items()},
                }

            elif ind == "pivot" or ind.startswith("pivot_"):
                method = "standard"
                if ind.startswith("pivot_"):
                    method = ind.split("_", 1)[1]
                prev_high = float(to_scalar(df["High"].iloc[-2]))
                prev_low = float(to_scalar(df["Low"].iloc[-2]))
                prev_close = float(to_scalar(df["Close"].iloc[-2]))
                pivot = indicators.calculate_pivot_points(prev_high, prev_low, prev_close, method)
                meta["pivot"] = {
                    "method": method,
                    "levels": {k: round(v, 2) for k, v in pivot.items()},
                }

            else:
                _add_unknown(meta, ind)

        except CalculationError:
            meta[f"_{ind}_error"] = "calculation_error"
        except (ValueError, TypeError) as e:
            logger.warning("technicals_conversion_error indicator=%s error=%s", ind, e)
            meta[f"_{ind}_error"] = str(e)
        except Exception as e:
            error_msg = str(e)
            if "blk ref_locs" in error_msg or "internal" in error_msg.lower():
                logger.warning("technicals_data_quality indicator=%s error=%s", ind, e)
                meta[f"_{ind}_error"] = "data_quality_issue"
            else:
                raise

    result_df = lttb_downsample(result_df)
    logger.debug("technicals_downsampled symbol=%s points=%d", symbol, len(result_df))

    toon_output = fmt_toon(result_df, wrapper_key="data")
    if meta:
        toon_output += "\nmeta: " + fmt(meta)
    return toon_output


def _parse_valuation_period(
    period_str: str,
    available_annual: dict[int, pd.Timestamp],
    available_quarterly: dict[str, pd.Timestamp],
) -> tuple[str, list[pd.Timestamp]]:
    """Parse period string and validate against available data.

    Returns (period_type, target_dates) where period_type is 'annual' or 'quarterly'.
    Raises ValidationError if format is invalid or period not available.
    """
    if period_str == "now":
        return ("now", [])

    # Range pattern: YYYY:YYYY or YYYY-QN:YYYY-QN
    if ":" in period_str:
        start_str, end_str = period_str.split(":", 1)
        if re.match(r"^\d{4}-?Q\d$", start_str, re.I):
            # Quarterly range
            start_match = re.match(r"^(\d{4})-?Q(\d)$", start_str, re.I)
            end_match = re.match(r"^(\d{4})-?Q(\d)$", end_str, re.I)
            if not start_match or not end_match:
                raise ValidationError(
                    f"Invalid quarter range format: {period_str}. "
                    "Use YYYY-QN:YYYY-QN (e.g., 2023-Q1:2024-Q3)"
                )
            start_key = f"{start_match.group(1)}-Q{start_match.group(2)}"
            end_key = f"{end_match.group(1)}-Q{end_match.group(2)}"
            dates = []
            for key in sorted(available_quarterly.keys()):
                if start_key <= key <= end_key:
                    dates.append(available_quarterly[key])
            if not dates:
                raise ValidationError(
                    f"No quarters available in range {period_str}. "
                    f"Available: {sorted(available_quarterly.keys())}"
                )
            logger.debug(
                "valuation_period_parsed periods=%s type=quarterly dates=%s",
                period_str,
                [str(d.date()) for d in dates],
            )
            return ("quarterly", dates)
        else:
            # Annual range
            try:
                start_year = int(start_str)
                end_year = int(end_str)
            except ValueError:
                raise ValidationError(
                    f"Invalid year range format: {period_str}. Use YYYY:YYYY (e.g., 2023:2024)"
                )
            dates = []
            for year in sorted(available_annual.keys()):
                if start_year <= year <= end_year:
                    dates.append(available_annual[year])
            if not dates:
                raise ValidationError(
                    f"No years available in range {period_str}. "
                    f"Available: {sorted(available_annual.keys())}"
                )
            logger.debug(
                "valuation_period_parsed periods=%s type=annual dates=%s",
                period_str,
                [str(d.date()) for d in dates],
            )
            return ("annual", dates)

    # Single quarter: 2024-Q3 or 2024Q3
    quarter_match = re.match(r"^(\d{4})-?Q(\d)$", period_str, re.I)
    if quarter_match:
        year, q = quarter_match.group(1), quarter_match.group(2)
        key = f"{year}-Q{q}"
        if key not in available_quarterly:
            raise ValidationError(
                f"Quarter {key} not available. Available: {sorted(available_quarterly.keys())}"
            )
        target_date = available_quarterly[key]
        logger.debug(
            "valuation_period_parsed periods=%s type=quarterly dates=%s",
            period_str,
            [str(target_date.date())],
        )
        return ("quarterly", [target_date])

    # Single year: 2024
    if re.match(r"^\d{4}$", period_str):
        year = int(period_str)
        if year not in available_annual:
            raise ValidationError(
                f"Year {year} not available. Available: {sorted(available_annual.keys())}"
            )
        target_date = available_annual[year]
        logger.debug(
            "valuation_period_parsed periods=%s type=annual dates=%s",
            period_str,
            [str(target_date.date())],
        )
        return ("annual", [target_date])

    raise ValidationError(
        f"Invalid period format: {period_str}. "
        'Use "now", "YYYY", "YYYY-QN", "YYYY:YYYY", or "YYYY-QN:YYYY-QN"'
    )


def _compute_historical_valuation(
    symbol: str,
    ticker: Any,
    period_type: str,
    target_dates: list[pd.Timestamp],
    metrics: list[str],
) -> str:
    """Compute historical valuation metrics from financial statements."""
    if period_type == "annual":
        income = ticker.income_stmt
        balance = ticker.balance_sheet
    else:
        income = ticker.quarterly_income_stmt
        balance = ticker.quarterly_balance_sheet

    logger.debug(
        "valuation_statements symbol=%s type=%s income_cols=%d balance_cols=%d",
        symbol,
        period_type,
        len(income.columns),
        len(balance.columns),
    )

    results: dict[str, dict[str, Any]] = {}
    unsupported = []

    # Check for unsupported metrics in historical mode
    historical_supported = {"pe", "eps", "ratios"}
    for m in metrics:
        if m not in historical_supported and m != "all":
            if m not in unsupported:
                unsupported.append(m)

    for target_date in target_dates:
        date_key = str(target_date.date())

        # Find matching column in statements
        col = None
        for c in income.columns:
            if c == target_date:
                col = c
                break
        if col is None:
            logger.warning("historical_valuation_no_data date=%s", target_date)
            continue

        try:
            net_income = float(income.loc["Net Income", col])
            revenue = float(income.loc["Total Revenue", col])
            equity = float(balance.loc["Stockholders Equity", col])
            shares = float(balance.loc["Ordinary Shares Number", col])
        except (KeyError, TypeError) as e:
            logger.warning("historical_valuation_missing_field date=%s error=%s", target_date, e)
            continue

        if shares <= 0:
            logger.warning("historical_valuation_invalid_shares date=%s", target_date)
            continue

        # Check for official EPS fields (more accurate than manual calculation)
        has_diluted_eps = "Diluted EPS" in income.index
        has_basic_eps = "Basic EPS" in income.index

        # For quarterly, compute TTM if we have 4 quarters
        if period_type == "quarterly":
            col_idx = list(income.columns).index(col)
            if col_idx + 4 <= len(income.columns):
                ttm_cols = income.columns[col_idx : col_idx + 4]
                ttm_revenue = sum(float(income.loc["Total Revenue", c]) for c in ttm_cols)
                # Use official EPS if available (sum of 4 quarters)
                if has_diluted_eps:
                    eps = sum(float(income.loc["Diluted EPS", c]) for c in ttm_cols)
                elif has_basic_eps:
                    eps = sum(float(income.loc["Basic EPS", c]) for c in ttm_cols)
                else:
                    ttm_net_income = sum(float(income.loc["Net Income", c]) for c in ttm_cols)
                    eps = ttm_net_income / shares
                rev_per_share = ttm_revenue / shares
                ttm_note = "ttm"
            else:
                # Annualize single quarter
                if has_diluted_eps:
                    eps = float(income.loc["Diluted EPS", col]) * 4
                elif has_basic_eps:
                    eps = float(income.loc["Basic EPS", col]) * 4
                else:
                    eps = (net_income * 4) / shares
                rev_per_share = (revenue * 4) / shares
                ttm_note = "annualized"
                logger.debug(
                    "valuation_ttm_fallback date=%s need=4 have=%d",
                    target_date.date(),
                    len(income.columns) - col_idx,
                )
        else:
            # Annual: use official EPS if available
            if has_diluted_eps:
                eps = float(income.loc["Diluted EPS", col])
            elif has_basic_eps:
                eps = float(income.loc["Basic EPS", col])
            else:
                eps = net_income / shares
            rev_per_share = revenue / shares
            ttm_note = None

        book_per_share = equity / shares

        # Get price using history module (benefits from cache)
        try:
            start_str = (target_date.date() - timedelta(days=5)).isoformat()
            end_str = (target_date.date() + timedelta(days=1)).isoformat()
            price_df = history.get_history(symbol, start=start_str, end=end_str, interval="1d")

            if price_df.empty:
                logger.warning("historical_valuation_no_price date=%s", target_date)
                continue

            price_df = normalize_tz(price_df)

            # Find closest price on or before target date
            target_ts = pd.Timestamp(target_date.date())
            mask = price_df.index <= target_ts
            if mask.any():
                close_price = float(price_df.loc[price_df.index[mask][-1], "Close"])
            else:
                close_price = float(price_df["Close"].iloc[0])
        except Exception as e:
            logger.warning("historical_valuation_price_error date=%s error=%s", target_date, e)
            continue

        # Compute ratios
        entry: dict[str, Any] = {"price": round(close_price, 2)}

        notes = []
        if ttm_note:
            notes.append(ttm_note)

        if "all" in metrics or "pe" in metrics or "eps" in metrics:
            entry["eps"] = round(eps, 2)
            if eps > 0:
                entry["pe"] = round(close_price / eps, 1)
            else:
                entry["pe"] = None
                notes.append("pe:null (negative earnings)")

        if "all" in metrics or "ratios" in metrics:
            if book_per_share > 0:
                entry["pb"] = round(close_price / book_per_share, 1)
            else:
                entry["pb"] = None
                notes.append("pb:null (negative book value)")
            entry["ps"] = round(close_price / rev_per_share, 1) if rev_per_share > 0 else None

        if notes:
            entry["_note"] = ", ".join(notes)

        results[date_key] = entry

    if not results:
        raise DataUnavailableError("No historical valuation data available for requested periods")

    if unsupported:
        results["_unsupported"] = unsupported  # type: ignore[assignment]

    logger.debug(
        "valuation_historical_result symbol=%s attempted=%d succeeded=%d",
        symbol,
        len(target_dates),
        len([k for k in results if not k.startswith("_")]),
    )

    return fmt(results)


def _handle_valuation(args: dict) -> str:
    """Handle valuation tool - valuation metrics and quality score."""
    symbol = args.get("symbol")
    if not symbol:
        raise ValidationError("symbol required")

    t = _ticker(symbol)
    metrics = args.get("metrics") or ["all"]

    if "all" in metrics:
        metrics = ALL_METRICS

    periods = args.get("periods", "now")

    # Historical valuation mode
    if periods != "now":
        # Build available dates from statements
        try:
            income_annual = t.income_stmt
            income_quarterly = t.quarterly_income_stmt
        except Exception as e:
            raise DataUnavailableError(f"Cannot fetch financial statements: {e}")

        available_annual: dict[int, pd.Timestamp] = {}
        for c in income_annual.columns:
            available_annual[c.year] = c

        available_quarterly: dict[str, pd.Timestamp] = {}
        for c in income_quarterly.columns:
            month = c.month
            if month <= 3:
                q = 1
            elif month <= 6:
                q = 2
            elif month <= 9:
                q = 3
            else:
                q = 4
            available_quarterly[f"{c.year}-Q{q}"] = c

        period_type, target_dates = _parse_valuation_period(
            periods, available_annual, available_quarterly
        )

        if period_type == "now":
            pass  # Fall through to current valuation
        else:
            return _compute_historical_valuation(symbol, t, period_type, target_dates, metrics)

    # Current valuation mode (periods == "now")
    info = t.info
    result: dict[str, Any] = {}

    if "all" in metrics or "pe" in metrics:
        result["pe"] = safe_get(info, "trailingPE")
        result["pe_fwd"] = safe_get(info, "forwardPE")

    if "all" in metrics or "peg" in metrics:
        pe = safe_get(info, "trailingPE")
        forward_pe = safe_get(info, "forwardPE")
        earnings_growth = safe_get(info, "earningsGrowth")
        revenue_growth = safe_get(info, "revenueGrowth")

        peg = None
        peg_source = None
        pe_for_peg = pe if pe else forward_pe

        if pe_for_peg:
            if safe_gt(earnings_growth, 0):
                peg = round(float(pe_for_peg) / (float(earnings_growth) * 100), 2)
                peg_source = "earnings"
            elif safe_gt(revenue_growth, 0):
                peg = round(float(pe_for_peg) / (float(revenue_growth) * 100), 2)
                peg_source = "revenue"

        result["peg"] = peg
        if peg_source:
            result["peg_source"] = peg_source
        if peg:
            if peg < 1:
                result["peg_signal"] = "undervalued"
            elif peg > 2:
                result["peg_signal"] = "overvalued"
            else:
                result["peg_signal"] = "fair"

    if "all" in metrics or "eps" in metrics:
        result["eps"] = safe_get(info, "trailingEps")
        result["eps_fwd"] = safe_get(info, "forwardEps")

    if "all" in metrics or "margins" in metrics:
        result["margin_gross"] = safe_get(info, "grossMargins")
        result["margin_op"] = safe_get(info, "operatingMargins")
        result["margin_net"] = safe_get(info, "profitMargins")

    if "all" in metrics or "growth" in metrics:
        result["growth_rev"] = safe_get(info, "revenueGrowth")
        result["growth_earn"] = safe_get(info, "earningsGrowth")

    if "all" in metrics or "ratios" in metrics:
        result["pb"] = safe_get(info, "priceToBook")
        result["ps"] = safe_get(info, "priceToSalesTrailing12Months")
        result["ev_ebitda"] = safe_get(info, "enterpriseToEbitda")

    if "all" in metrics or "dividends" in metrics:
        result["div_yield"] = safe_get(info, "dividendYield")
        result["div_rate"] = safe_get(info, "dividendRate")

    result = round_result(result, 3)

    if "all" in metrics or "dividends" in metrics:
        payout = safe_get(info, "payoutRatio")
        if payout is not None:
            result["payout_ratio"] = round(payout * 100, 1)

    if "all" in metrics or "quality" in metrics:
        score, details = calculate_quality(info)
        result["quality_score"] = score
        result["quality_max"] = 7
        if score >= 6:
            result["quality_signal"] = "strong"
        elif score >= 3:
            result["quality_signal"] = "neutral"
        else:
            result["quality_signal"] = "weak"
        result["quality_details"] = ",".join(details) if details else None

    return fmt(result)


def _handle_financials(args: dict) -> str:
    """Handle financials tool - financial statements."""
    symbol = args.get("symbol")
    if not symbol:
        raise ValidationError("symbol required")

    t = _ticker(symbol)
    stmt = args.get("statement", "income")
    freq = args.get("freq", "annual")
    limit = max(1, args.get("limit", 10))
    periods = args.get("periods", "now")
    fields = args.get("fields", [])

    freq_param = "quarterly" if freq == "quarterly" else "yearly"
    if stmt == "income":
        df = t.get_income_stmt(freq=freq_param)
    elif stmt == "balance":
        df = t.get_balance_sheet(freq=freq_param)
    else:
        df = t.get_cashflow(freq=freq_param)

    if df.empty:
        raise DataUnavailableError("No data")

    if periods != "now":
        available_annual: dict[int, pd.Timestamp] = {}
        available_quarterly: dict[str, pd.Timestamp] = {}
        for c in df.columns:
            available_annual[c.year] = c
            month = c.month
            if month <= 3:
                q = 1
            elif month <= 6:
                q = 2
            elif month <= 9:
                q = 3
            else:
                q = 4
            available_quarterly[f"{c.year}-Q{q}"] = c

        period_type, target_dates = _parse_valuation_period(
            periods, available_annual, available_quarterly
        )

        if period_type != "now" and target_dates:
            target_set = set(target_dates)
            selected_cols = [c for c in df.columns if c in target_set]
            if selected_cols:
                df = df[selected_cols]

    if fields:
        available = set(df.index)
        found = available & set(fields)
        if found:
            df = df.loc[list(found)]
        else:
            data = {
                "_available_fields": list(df.index)[:20],
                "_hint": "None of requested fields found. See available fields above.",
            }
            return fmt(data)

    total_rows = len(df)
    df = df.head(limit)
    df.columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c) for c in df.columns]

    data = df.to_dict()
    if total_rows > limit:
        data["_truncated"] = f"Showing {limit} of {total_rows}. Increase limit for more."
    return fmt(data)


_TOOL_HANDLERS: dict[str, Any] = {
    "search_stock": _handle_search_stock,
    "history": _handle_history,
    "technicals": _handle_technicals,
    "valuation": _handle_valuation,
    "financials": _handle_financials,
}


async def _execute(name: str, args: dict) -> str:
    """Execute tool via dispatch table."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValidationError(f"Unknown tool: {name}")
    return handler(args)


async def run_server() -> None:
    """Run the MCP server."""
    log_file = os.environ.get("MCP_LOG_FILE") or get_default_log_path()
    logger.info(
        "server_start log_level=%s log_file=%s",
        logging.getLevelName(logger.getEffectiveLevel()),
        log_file,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    logger.info("server_stop")


def main() -> None:
    """Entry point."""
    import asyncio

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
