"""Yahoo Finance MCP Server"""

import json
import logging
import os
import platform
import sys
import tempfile
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import indicators, prices
from .cache import get_cache_stats
from .errors import (
    CalculationError,
    DataUnavailableError,
    MCPError,
    SymbolNotFoundError,
    ValidationError,
)

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


def _get_default_log_path() -> str:
    """Return platform-appropriate default log path."""
    log_name = "yfinance-mcp.log"
    if platform.system() == "Windows":
        # Use %LOCALAPPDATA%/yfinance-mcp or fallback to temp
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            log_dir = Path(local_app_data) / "yfinance-mcp" / "logs"
        else:
            log_dir = Path(tempfile.gettempdir()) / "yfinance-mcp"
    else:
        # Unix-like: use /tmp or XDG_STATE_HOME
        xdg_state = os.environ.get("XDG_STATE_HOME")
        if xdg_state:
            log_dir = Path(xdg_state) / "yfinance-mcp"
        else:
            log_dir = Path("/tmp")
    return str(log_dir / log_name)


def _get_log_level() -> int:
    """Determine log level from environment."""
    level_str = os.environ.get("MCP_LOG_LEVEL", "").upper()
    if level_str in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return getattr(logging, level_str)
    if os.environ.get("MCP_DEBUG"):
        return logging.DEBUG
    return logging.WARNING


class NDJSONFormatter(logging.Formatter):
    """Formatter that outputs NDJSON for structured analysis."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "loc": f"{record.filename}:{record.lineno}",
            "msg": record.getMessage(),
        }
        # stats are expensive to serialize, only include at DEBUG
        if record.levelno <= logging.DEBUG:
            entry["stats"] = _get_stats_snapshot()
        if hasattr(record, "data"):
            entry["data"] = record.data
        if hasattr(record, "request_id"):
            entry["request_id"] = record.request_id
        return json.dumps(entry)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        return f"{ts} {color}{record.levelname:7}{reset} {record.name}: {record.getMessage()}"


def _configure_logging() -> logging.Logger:
    """Configure logging with platform support and rotation."""
    log_file = os.environ.get("MCP_LOG_FILE") or _get_default_log_path()
    log_level = _get_log_level()
    enable_console = os.environ.get("MCP_LOG_CONSOLE", "").lower() in ("1", "true", "yes")

    handlers: list[logging.Handler] = []

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(NDJSONFormatter())
        handlers.append(file_handler)

    # stderr avoids interfering with MCP protocol on stdout
    if enable_console:
        console_handler = logging.StreamHandler(sys.stderr)
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            console_handler.setFormatter(ConsoleFormatter())
        else:
            console_handler.setFormatter(NDJSONFormatter())
        handlers.append(console_handler)

    if not handlers:
        handlers.append(logging.NullHandler())

    logging.root.handlers.clear()
    for handler in handlers:
        logging.root.addHandler(handler)
    logging.root.setLevel(log_level)

    # MCP library adds handlers dynamically, so we reconfigure them here
    for name in ("mcp", "mcp.server", "mcp.server.lowlevel", "mcp.server.lowlevel.server"):
        mcp_logger = logging.getLogger(name)
        mcp_logger.handlers.clear()
        mcp_logger.propagate = False
        for handler in handlers:
            mcp_logger.addHandler(handler)
        mcp_logger.setLevel(logging.WARNING)

    return logging.getLogger("yfinance-mcp")


logger = _configure_logging()

server = Server("yfinance-mcp")

_cb: dict = {
    "fails": 0,
    "open": False,
    "threshold": 5,
    "recovery_timeout": 30,  # seconds
    "opened_at": None,
}


def _check_cb() -> None:
    """Check circuit breaker state, with auto-recovery after timeout."""
    if not _cb["open"]:
        return

    # Check if recovery timeout has passed
    if _cb["opened_at"] is not None:
        elapsed = (datetime.now() - _cb["opened_at"]).total_seconds()
        if elapsed >= _cb["recovery_timeout"]:
            # Half-open: allow one request through
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
        logger.warning(f"Circuit breaker opened after {_cb['fails']} failures")


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

    # accessing last_price raises KeyError for invalid tickers
    try:
        _ = fi.last_price
    except KeyError as e:
        logger.warning("ticker_not_found symbol=%s keyerror=%s", symbol_upper, e)
        raise SymbolNotFoundError(symbol)

    _reset_cb()
    logger.debug("ticker_validated symbol=%s", symbol_upper)
    return t


def _fmt(data: Any) -> str:
    """Format result as compact JSON (no indentation)."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.copy()
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.strftime("%Y-%m-%d")
        return json.dumps(data.to_dict(), default=str, separators=(",", ":"))
    return json.dumps(data, default=str, separators=(",", ":"))


def _err(e: Exception) -> str:
    """Format error compactly."""
    if isinstance(e, MCPError):
        return json.dumps({"err": e.code, "msg": e.message}, separators=(",", ":"))
    return json.dumps({"err": "ERROR", "msg": str(e)}, separators=(",", ":"))


def _to_scalar(val: Any) -> Any:
    """Extract scalar from value (handles Series from multi-index edge cases)."""
    if val is None:
        return None
    if isinstance(val, pd.Series):
        return val.iloc[0] if len(val) > 0 else None
    if isinstance(val, pd.DataFrame):
        return val.iloc[0, 0] if val.size > 0 else None
    return val


def _safe_scalar(val: Any) -> Any:
    """Safely extract scalar, catching any ambiguous truth value errors."""
    try:
        return _to_scalar(val)
    except (ValueError, TypeError):
        # pandas Series raises "truth value is ambiguous" on direct comparison
        if hasattr(val, "iloc"):
            try:
                return val.iloc[0] if len(val) > 0 else None
            except Exception:
                return None
        return None


def _safe_gt(a: Any, b: Any) -> bool:
    """Safe greater-than comparison that handles None and Series."""
    try:
        if a is None or b is None:
            return False
        return float(a) > float(b)
    except (ValueError, TypeError):
        return False


def _safe_lt(a: Any, b: Any) -> bool:
    """Safe less-than comparison that handles None and Series."""
    try:
        if a is None or b is None:
            return False
        return float(a) < float(b)
    except (ValueError, TypeError):
        return False


def _safe_get(info: dict, key: str) -> Any:
    """Safely get a value from info dict, converting Series to scalar."""
    val = info.get(key)
    return _safe_scalar(val)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame for indicator calculations."""
    if df.empty:
        return df

    # yfinance returns multi-index for some queries
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)

    # contiguous copy avoids pandas internal block errors
    df = df.copy()

    # sparse data sometimes has object-typed OHLCV columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _require_symbol(args: dict) -> tuple[str, yf.Ticker]:
    """Validate symbol argument and return (symbol, ticker) tuple."""
    symbol = args.get("symbol")
    if not symbol:
        raise ValidationError("symbol required")
    return symbol, _ticker(symbol)


def _signal_level(value: float, high: float, low: float) -> str:
    """Return overbought/oversold/neutral based on thresholds."""
    if value > high:
        return "overbought"
    if value < low:
        return "oversold"
    return "neutral"


def _add_unknown(result: dict, indicator: str) -> None:
    """Add indicator to _unknown list in result dict."""
    result.setdefault("_unknown", []).append(indicator)
    logger.warning("technicals_unknown_indicator indicator=%s", indicator)


def _round_result(data: dict, decimals: int = 2) -> dict:
    """Round float values and remove None values from result dict."""
    return {
        k: (round(v, decimals) if isinstance(v, float) else v)
        for k, v in data.items()
        if v is not None
    }


def _safe_round(value: float, decimals: int = 2) -> float | None:
    """Round value, returning None if NaN or infinite."""
    if pd.isna(value) or not pd.api.types.is_number(value):
        return None
    return round(value, decimals)


def _adaptive_decimals(price: float) -> int:
    """Determine decimal places based on price magnitude."""
    import math

    if price <= 0 or not math.isfinite(price):
        return 2
    if price >= 1.0:
        return 2
    # penny stocks need more decimals to preserve precision
    return max(2, int(-math.log10(price)) + 1)


def _parse_moving_avg_period(indicator: str) -> int | None:
    """Parse period from sma_N or ema_N format. Returns None on invalid format."""
    try:
        period = int(indicator.split("_")[1])
        if period < 1:
            return None
        return period
    except (ValueError, IndexError):
        return None


TOOLS = [
    Tool(
        name="summary",
        description=(
            "Get complete stock overview. Best starting point for analysis. "
            "Returns PEG ratio (valuation), 50-day trend (momentum), and "
            "quality score (0-7, financial health). PEG<1 undervalued, >2 overvalued. "
            "Quality 6-7 strong, 0-2 weak."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker (e.g., AAPL, MSFT, 7203.T)",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="price",
        description=(
            "Get historical price data (Open, High, Low, Close, Volume). "
            "Supports arbitrary date ranges back to 1990s for daily data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker (e.g., AAPL, MSFT)"},
                "period": {
                    "type": "string",
                    "default": "1mo",
                    "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
                    "description": "Relative period from today. Ignored if start provided.",
                },
                "start": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD) or datetime (YYYY-MM-DD HH:MM for intraday).",
                },
                "end": {
                    "type": "string",
                    "description": "End date. Defaults to today.",
                },
                "interval": {
                    "type": "string",
                    "default": "1d",
                    "enum": ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"],
                    "description": "Bar size. Intraday (1m-1h) limited to 60 days.",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Number of bars to return (max 500).",
                },
                "format": {
                    "type": "string",
                    "enum": ["concise", "detailed"],
                    "default": "concise",
                    "description": "concise: o/h/l/c/v keys. detailed: full names.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="technicals",
        description=(
            "Calculate technical indicators for trading signals. "
            "RSI: >70 overbought, <30 oversold. MACD: histogram>0 bullish. "
            "SMA/EMA: price above = uptrend. BB: volatility bands. "
            "Use 'summary' first for fundamental context."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker"},
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Options: rsi, macd, sma_N, ema_N, bb, stoch, atr, obv",
                },
                "period": {"type": "string", "default": "3mo"},
            },
            "required": ["symbol", "indicators"],
        },
    ),
    Tool(
        name="fundamentals",
        description=(
            "Get detailed fundamental valuation metrics. "
            "pe: P/E ratio. eps: earnings per share. margins: gross/operating/net. "
            "growth: revenue/earnings growth. valuation: P/B, P/S, EV/EBITDA. "
            "dividends: yield, rate, payout ratio. "
            "Use 'summary' for quick PEG/quality analysis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Options: pe, eps, margins, growth, valuation, dividends",
                },
            },
            "required": ["symbol", "metrics"],
        },
    ),
    Tool(
        name="financials",
        description=(
            "Get financial statements: income, balance, or cashflow. "
            "income: revenue, net income. balance: assets, liabilities. "
            "cashflow: operating/investing/financing. Use 'fields' to filter rows. "
            "Note: Yahoo API limits to 4 years annual or 5 quarters."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker"},
                "statement": {
                    "type": "string",
                    "enum": ["income", "balance", "cashflow"],
                    "default": "income",
                },
                "freq": {"type": "string", "enum": ["annual", "quarterly"], "default": "annual"},
                "periods": {
                    "type": "integer",
                    "default": 4,
                    "description": "Number of time periods/years to return (default 4, max ~10)",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max rows to return (for token efficiency)",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific rows (e.g., ['TotalRevenue', 'NetIncome'])",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="peers",
        description=(
            "Compare multiple stocks side-by-side on selected metrics. "
            "Max 10 symbols. Use for sector analysis or finding best-in-class. "
            "Metrics: price, pe, market_cap, pb, ps, yield, beta."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tickers to compare",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Options: price, pe, market_cap, pb, ps, yield, beta",
                },
            },
            "required": ["symbols", "metrics"],
        },
    ),
    Tool(
        name="search",
        description=(
            "Find stock ticker symbols by company name. "
            "Enter company name, returns matching tickers. "
            "Use returned symbol with 'summary' for full analysis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Company name to search"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    start_time = time.time()
    _update_stats(increment={"calls": 1})
    stats = _get_stats_snapshot()
    logger.debug(
        "tool_call_start name=%s args=%s call=%d",
        name,
        _summarize_args(arguments),
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
    except MCPError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning(
            "tool_call_error name=%s code=%s elapsed_ms=%.1f msg=%s",
            name,
            e.code,
            elapsed_ms,
            e.message,
        )
        return [TextContent(type="text", text=_err(e))]
    except Exception as e:
        import traceback

        elapsed_ms = (time.time() - start_time) * 1000
        tb_str = traceback.format_exc()
        logger.error(
            "tool_call_exception name=%s elapsed_ms=%.1f error=%s\n%s",
            name,
            elapsed_ms,
            e,
            tb_str,
        )
        return [TextContent(type="text", text=_err(e))]


def _summarize_args(args: dict) -> str:
    """Create a compact summary of arguments for logging."""
    summary = {}
    for k, v in args.items():
        if isinstance(v, list) and len(v) > 3:
            summary[k] = f"[{len(v)} items]"
        elif isinstance(v, str) and len(v) > 50:
            summary[k] = v[:50] + "..."
        else:
            summary[k] = v
    return json.dumps(summary, separators=(",", ":"))


def _handle_summary(args: dict) -> str:
    """Handle summary tool - quick stock overview with PEG, trend, and quality score."""
    symbol, t = _require_symbol(args)
    logger.debug("summary_fetch symbol=%s", symbol)

    try:
        fi = t.fast_info
        info = t.info
        if not info or _safe_get(info, "regularMarketPrice") is None:
            raise SymbolNotFoundError(symbol)
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("summary_invalid_symbol symbol=%s error=%s", symbol, e)
        raise SymbolNotFoundError(symbol)
    logger.debug("summary_info symbol=%s has_pe=%s", symbol, "trailingPE" in info)

    quality_score = 0
    quality_details = []

    roa = _safe_get(info, "returnOnAssets")
    if _safe_gt(roa, 0):
        quality_score += 1
        quality_details.append("roa+")

    ocf = _safe_get(info, "operatingCashflow")
    if _safe_gt(ocf, 0):
        quality_score += 1
        quality_details.append("ocf+")

    net_income = _safe_get(info, "netIncomeToCommon")
    if _safe_gt(ocf, net_income):
        quality_score += 1
        quality_details.append("accrual+")

    current_ratio = _safe_get(info, "currentRatio")
    if _safe_gt(current_ratio, 1):
        quality_score += 1
        quality_details.append("liquidity+")

    debt_equity = _safe_get(info, "debtToEquity")
    if _safe_lt(debt_equity, 100) and debt_equity is not None:
        quality_score += 1
        quality_details.append("lowdebt+")

    gross_margin = _safe_get(info, "grossMargins")
    if _safe_gt(gross_margin, 0.2):
        quality_score += 1
        quality_details.append("margin+")

    roe = _safe_get(info, "returnOnEquity")
    if _safe_gt(roe, 0.1):
        quality_score += 1
        quality_details.append("roe+")

    pe = _safe_get(info, "trailingPE")
    forward_pe = _safe_get(info, "forwardPE")
    earnings_growth = _safe_get(info, "earningsGrowth")
    revenue_growth = _safe_get(info, "revenueGrowth")

    peg = None
    peg_source = None
    pe_for_peg = pe if pe else forward_pe

    if pe_for_peg:
        if _safe_gt(earnings_growth, 0):
            peg = round(float(pe_for_peg) / (float(earnings_growth) * 100), 2)
            peg_source = "earnings"
        elif _safe_gt(revenue_growth, 0):
            peg = round(float(pe_for_peg) / (float(revenue_growth) * 100), 2)
            peg_source = "revenue"

    pe_note = None
    if not pe:
        net_income_for_pe = _safe_get(info, "netIncomeToCommon")
        if _safe_lt(net_income_for_pe, 0):
            pe_note = "unprofitable"
        elif forward_pe:
            pe_note = "use_forward_pe"

    peg_note = None
    if not peg:
        if not pe_for_peg:
            peg_note = "no_pe"
        elif earnings_growth is not None and not _safe_gt(earnings_growth, 0):
            peg_note = "negative_earnings_growth"
        elif revenue_growth is not None and not _safe_gt(revenue_growth, 0):
            peg_note = "negative_growth"
        else:
            peg_note = "no_growth_data"

    # fast_info returns Series for some stocks
    try:
        last_price = _safe_scalar(fi.last_price)
        prev_close = _safe_scalar(fi.previous_close)
        market_cap = _safe_scalar(fi.market_cap)
    except Exception:
        last_price = None
        prev_close = None
        market_cap = None

    df = prices.get_history(symbol, "3mo", "1d", ticker=t)
    trend = "unknown"
    sma50 = None
    if not df.empty and len(df) >= 50:
        sma50 = df["Close"].tail(50).mean()
        trend = "uptrend" if last_price and last_price > sma50 else "downtrend"

    if quality_score >= 6:
        quality_signal = "strong"
    elif quality_score >= 3:
        quality_signal = "neutral"
    else:
        quality_signal = "weak"

    price_decimals = _adaptive_decimals(float(last_price)) if last_price else 2

    result = {
        "price": round(last_price, price_decimals) if last_price else None,
        "change_pct": round((last_price / prev_close - 1) * 100, 2)
        if last_price and prev_close
        else None,
        "market_cap": int(market_cap) if market_cap else None,
        "pe": round(pe, 1) if pe else None,
        "pe_forward": round(forward_pe, 1) if forward_pe and not pe else None,
        "pe_note": pe_note,
        "peg": peg,
        "peg_source": peg_source,
        "peg_note": peg_note if not peg else None,
        "peg_signal": "undervalued"
        if peg and peg < 1
        else "overvalued"
        if peg and peg > 2
        else "fair"
        if peg
        else None,
        "trend": trend,
        "sma50": round(sma50, price_decimals) if sma50 else None,
        "quality_score": quality_score,
        "quality_max": 7,
        "quality_signal": quality_signal,
        "quality_details": ",".join(quality_details) if quality_details else None,
        "roe": round(roe * 100, 1) if roe else None,
        "_hint": "peg_signal + trend = primary view. quality_score = financial health check.",
    }
    return _fmt({k: v for k, v in result.items() if v is not None})


def _handle_price(args: dict) -> str:
    """Handle price tool - historical OHLCV data."""
    symbol, t = _require_symbol(args)
    start = args.get("start")
    end = args.get("end")
    period = args.get("period", "1mo")
    interval = args.get("interval", "1d")
    raw_limit = args.get("limit", 20)

    try:
        limit = max(1, min(int(raw_limit), 500))
    except (TypeError, ValueError):
        raise ValidationError(f"limit must be an integer, got {type(raw_limit).__name__}")

    fmt = args.get("format", "concise")

    logger.debug(
        "price_fetch symbol=%s start=%s end=%s period=%s interval=%s limit=%d",
        symbol,
        start,
        end,
        period,
        interval,
        limit,
    )

    df = prices.get_history(symbol, period, interval, ticker=t, start=start, end=end)
    if df.empty:
        logger.warning("price_no_data symbol=%s period=%s", symbol, period)
        raise DataUnavailableError(
            f"No price data for {symbol}. Try different period or check with 'search'."
        )
    logger.debug("price_fetched symbol=%s bars=%d", symbol, len(df))

    total_bars = len(df)
    df = df.tail(limit)

    last_close = df["Close"].iloc[-1] if not df.empty else 1.0
    decimals = _adaptive_decimals(float(last_close))

    if fmt == "concise":
        col_map = {"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"}
        df = df.rename(columns=col_map)
        df = df[["o", "h", "l", "c", "v"]].round(decimals)
    else:
        df = df[["Open", "High", "Low", "Close", "Volume"]].round(decimals)

    intraday_intervals = {"1m", "5m", "15m", "1h"}
    if interval in intraday_intervals:
        df.index = df.index.strftime("%Y-%m-%d %H:%M")
    else:
        df.index = df.index.strftime("%Y-%m-%d")

    result: dict[str, Any] = {"bars": df.to_dict("index")}
    if total_bars > limit:
        result["_truncated"] = f"Showing {limit} of {total_bars}. Increase limit for more."
    result["_hint"] = "Use 'technicals' for indicators or 'summary' for fundamentals"
    return _fmt(result)


def _handle_technicals(args: dict) -> str:
    """Handle technicals tool - trading signals and indicators."""
    symbol, t = _require_symbol(args)
    period = args.get("period", "3mo")
    inds = args.get("indicators", [])

    if not inds:
        logger.debug("technicals_no_indicators symbol=%s", symbol)
        raise ValidationError(
            "indicators required. Options: rsi, macd, sma_N, ema_N, bb, stoch, atr, obv"
        )

    logger.debug("technicals_fetch symbol=%s period=%s indicators=%s", symbol, period, inds)

    df = prices.get_history(symbol, period, "1d", ticker=t)
    if df.empty:
        logger.warning("technicals_no_data symbol=%s period=%s", symbol, period)
        raise DataUnavailableError(f"No price data for {symbol}. Try period='6mo' for more data.")
    df = _normalize_df(df)
    logger.debug("technicals_data_ready symbol=%s bars=%d", symbol, len(df))

    result: dict[str, Any] = {}

    for ind in inds:
        try:
            if ind == "rsi":
                rsi = indicators.calculate_rsi(df["Close"])
                v = float(_to_scalar(rsi.iloc[-1]))
                result["rsi"] = round(v, 1)
                result["rsi_signal"] = _signal_level(v, 70, 30)

            elif ind == "macd":
                m = indicators.calculate_macd(df["Close"])
                hist = float(_to_scalar(m["histogram"].iloc[-1]))
                result["macd"] = round(float(_to_scalar(m["macd"].iloc[-1])), 3)
                result["macd_signal"] = round(float(_to_scalar(m["signal"].iloc[-1])), 3)
                result["macd_hist"] = round(hist, 3)
                result["macd_trend"] = "bullish" if hist > 0 else "bearish"

            elif ind.startswith("sma_"):
                p = _parse_moving_avg_period(ind)
                if p is None:
                    _add_unknown(result, ind)
                    continue
                sma = indicators.calculate_sma(df["Close"], p)
                v = float(_to_scalar(sma.iloc[-1]))
                result[ind] = round(v, 2) if not pd.isna(v) else None
                if not pd.isna(v):
                    close = float(_to_scalar(df["Close"].iloc[-1]))
                    result[f"{ind}_pos"] = "above" if close > v else "below"

            elif ind.startswith("ema_"):
                p = _parse_moving_avg_period(ind)
                if p is None:
                    _add_unknown(result, ind)
                    continue
                ema = indicators.calculate_ema(df["Close"], p)
                v = float(_to_scalar(ema.iloc[-1]))
                result[ind] = round(v, 2) if not pd.isna(v) else None

            elif ind == "bb":
                bb = indicators.calculate_bollinger_bands(df["Close"])
                upper = float(_to_scalar(bb["upper"].iloc[-1]))
                lower = float(_to_scalar(bb["lower"].iloc[-1]))
                pctb = float(_to_scalar(bb["percent_b"].iloc[-1]))

                result["bb_upper"] = _safe_round(upper, 2)
                result["bb_lower"] = _safe_round(lower, 2)

                if pd.isna(pctb):
                    result["bb_pctb"] = None
                    result["bb_signal"] = "unavailable"
                else:
                    result["bb_pctb"] = round(pctb, 2)
                    result["bb_signal"] = _signal_level(pctb, 1, 0)

            elif ind == "stoch":
                s = indicators.calculate_stochastic(df["High"], df["Low"], df["Close"])
                k = float(_to_scalar(s["k"].iloc[-1]))
                result["stoch_k"] = round(k, 1)
                result["stoch_d"] = round(float(_to_scalar(s["d"].iloc[-1])), 1)
                result["stoch_signal"] = _signal_level(k, 80, 20)

            elif ind == "atr":
                atr = indicators.calculate_atr(df["High"], df["Low"], df["Close"])
                atr_val = float(_to_scalar(atr.iloc[-1]))
                close = float(_to_scalar(df["Close"].iloc[-1]))
                result["atr"] = round(atr_val, 3)
                result["atr_pct"] = round(atr_val / close * 100, 2)

            elif ind == "obv":
                obv = indicators.calculate_obv(df["Close"], df["Volume"])
                obv_val = float(_to_scalar(obv.iloc[-1]))
                obv_sma = float(_to_scalar(obv.rolling(20).mean().iloc[-1]))

                if pd.isna(obv_val):
                    result["obv"] = None
                    result["obv_trend"] = None
                else:
                    result["obv"] = int(obv_val)
                    result["obv_trend"] = "bullish" if obv_val > obv_sma else "bearish"

            else:
                _add_unknown(result, ind)

        except CalculationError:
            result[ind] = None
        except (ValueError, TypeError) as e:
            # NaN values cause conversion errors
            logger.warning("technicals_conversion_error indicator=%s error=%s", ind, e)
            result[ind] = None
        except Exception as e:
            # pandas internal errors need user-friendly messages
            error_msg = str(e)
            if "blk ref_locs" in error_msg or "internal" in error_msg.lower():
                logger.warning("technicals_data_quality indicator=%s error=%s", ind, e)
                result[ind] = None
                result[f"_{ind}_error"] = "data_quality_issue"
            else:
                raise

    result["_hint"] = (
        "Use 'summary' for PEG/quality analysis or 'peers' to compare with competitors"
    )
    return _fmt(result)


def _handle_fundamentals(args: dict) -> str:
    """Handle fundamentals tool - valuation metrics."""
    symbol = args.get("symbol")
    if not symbol:
        raise ValidationError("symbol required")

    t = _ticker(symbol)
    info = t.info
    metrics = args.get("metrics", [])

    if not metrics:
        raise ValidationError(
            "metrics required. Options: pe, eps, margins, growth, valuation, dividends"
        )

    result: dict[str, Any] = {}

    if "all" in metrics or "pe" in metrics:
        result["pe"] = _safe_get(info, "trailingPE")
        result["pe_fwd"] = _safe_get(info, "forwardPE")
        result["peg"] = _safe_get(info, "pegRatio")

    if "all" in metrics or "eps" in metrics:
        result["eps"] = _safe_get(info, "trailingEps")
        result["eps_fwd"] = _safe_get(info, "forwardEps")

    if "all" in metrics or "margins" in metrics:
        result["margin_gross"] = _safe_get(info, "grossMargins")
        result["margin_op"] = _safe_get(info, "operatingMargins")
        result["margin_net"] = _safe_get(info, "profitMargins")

    if "all" in metrics or "growth" in metrics:
        result["growth_rev"] = _safe_get(info, "revenueGrowth")
        result["growth_earn"] = _safe_get(info, "earningsGrowth")

    if "all" in metrics or "valuation" in metrics:
        result["pb"] = _safe_get(info, "priceToBook")
        result["ps"] = _safe_get(info, "priceToSalesTrailing12Months")
        result["ev_ebitda"] = _safe_get(info, "enterpriseToEbitda")

    if "all" in metrics or "dividends" in metrics:
        result["div_yield"] = _safe_get(info, "dividendYield")
        result["div_rate"] = _safe_get(info, "dividendRate")

    result = _round_result(result, 3)

    if "all" in metrics or "dividends" in metrics:
        payout = _safe_get(info, "payoutRatio")
        if payout is not None:
            result["payout_ratio"] = round(payout * 100, 1)

    result["_hint"] = "Use 'financials' for full statements or 'peers' to compare with competitors"
    return _fmt(result)


def _handle_financials(args: dict) -> str:
    """Handle financials tool - financial statements."""
    symbol = args.get("symbol")
    if not symbol:
        raise ValidationError("symbol required")

    t = _ticker(symbol)
    stmt = args.get("statement", "income")
    freq = args.get("freq", "annual")
    limit = max(1, args.get("limit", 10))
    periods = max(1, args.get("periods", 4))
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

    total_periods = df.shape[1]
    if periods < total_periods:
        df = df.iloc[:, :periods]

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
            return _fmt(data)

    total_rows = len(df)
    df = df.head(limit)
    df.columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c) for c in df.columns]

    data = df.to_dict()
    if total_rows > limit:
        data["_truncated"] = f"Showing {limit} of {total_rows}. Increase limit for more."
    data["_hint"] = "Use 'fundamentals' for key ratios or try 'balance'/'cashflow' statements"
    return _fmt(data)


def _handle_peers(args: dict) -> str:
    """Handle peers tool - comparative analysis."""
    symbols = args.get("symbols", [])
    metrics = args.get("metrics", [])

    if not symbols:
        raise ValidationError("symbols required. Example: ['AAPL', 'MSFT', 'GOOGL']")
    if not metrics:
        raise ValidationError(
            "metrics required. Options: price, pe, market_cap, pb, ps, yield, beta"
        )
    if len(symbols) > 10:
        logger.warning("peers_too_many_symbols count=%d", len(symbols))
        raise ValidationError("Max 10 symbols per call. Split into multiple calls for more.")

    logger.debug("peers_compare symbols=%s metrics=%s", symbols, metrics)
    result: dict[str, Any] = {}
    failed_symbols = []

    for sym in symbols:
        try:
            t = _ticker(sym)
            info = t.info
            fi = t.fast_info

            last_price = _safe_scalar(fi.last_price)
            mcap = _safe_scalar(fi.market_cap)

            row: dict[str, Any] = {}
            price_value = None
            for m in metrics:
                if m == "price":
                    if last_price:
                        decimals = _adaptive_decimals(float(last_price))
                        price_value = round(last_price, decimals)
                elif m == "pe":
                    row["pe"] = _safe_get(info, "trailingPE")
                elif m == "market_cap":
                    row["mcap"] = int(mcap) if mcap else None
                elif m == "pb":
                    row["pb"] = _safe_get(info, "priceToBook")
                elif m == "ps":
                    row["ps"] = _safe_get(info, "priceToSalesTrailing12Months")
                elif m == "yield":
                    div_yield = _safe_get(info, "dividendYield")
                    row["yield"] = round(div_yield, 2) if div_yield else None
                elif m == "beta":
                    row["beta"] = _safe_get(info, "beta")

            rounded_row = _round_result(row)
            if price_value is not None:
                rounded_row["price"] = price_value
            result[sym.upper()] = rounded_row
        except Exception as e:
            logger.debug("peers_symbol_failed symbol=%s error=%s", sym, e)
            failed_symbols.append(sym)
            result[sym.upper()] = {"err": "failed"}

    if failed_symbols:
        logger.warning(
            "peers_partial_failure success=%d failed=%d symbols=%s",
            len(symbols) - len(failed_symbols),
            len(failed_symbols),
            failed_symbols,
        )
    result["_hint"] = "Use 'summary' on individual symbols for detailed PEG/quality analysis"
    return _fmt(result)


def _handle_search(args: dict) -> str:
    """Handle search tool - symbol discovery."""
    query = args.get("query", "")
    limit = max(1, min(args.get("limit", 5), 20))

    if not query:
        raise ValidationError("query required. Example: 'Apple' or 'Tesla'")

    search = yf.Search(query, max_results=limit)
    quotes = search.quotes[:limit]

    matches = [
        {
            "symbol": q.get("symbol"),
            "name": q.get("shortname") or q.get("longname"),
            "type": q.get("quoteType"),
        }
        for q in quotes
    ]
    return _fmt(
        {
            "matches": matches,
            "count": len(matches),
            "_hint": f"Found {len(matches)}. Use 'summary' with symbol for analysis.",
        }
    )


_TOOL_HANDLERS: dict[str, Any] = {
    "summary": _handle_summary,
    "price": _handle_price,
    "technicals": _handle_technicals,
    "fundamentals": _handle_fundamentals,
    "financials": _handle_financials,
    "peers": _handle_peers,
    "search": _handle_search,
}


async def _execute(name: str, args: dict) -> str:
    """Execute tool via dispatch table."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValidationError(f"Unknown tool: {name}")
    return handler(args)


async def run_server() -> None:
    """Run the MCP server."""
    log_file = os.environ.get("MCP_LOG_FILE") or _get_default_log_path()
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
