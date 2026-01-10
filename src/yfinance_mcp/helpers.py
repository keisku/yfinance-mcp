"""Helper functions for Yahoo Finance MCP Server."""

import json
import logging
import math
import os
import platform
import re
import sys
import tempfile
from datetime import date, timedelta
from zoneinfo import ZoneInfo
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta
from toon_format import encode as toon_encode
from yfinance.const import USER_AGENTS

from . import LOGGER_NAME
from .errors import MCPError


def get_default_log_path() -> str:
    """Return platform-appropriate default log path."""
    log_name = "yfinance-mcp.log"
    if platform.system() == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            log_dir = Path(local_app_data) / "yfinance-mcp" / "logs"
        else:
            log_dir = Path(tempfile.gettempdir()) / "yfinance-mcp"
    else:
        xdg_state = os.environ.get("XDG_STATE_HOME")
        if xdg_state:
            log_dir = Path(xdg_state) / "yfinance-mcp"
        else:
            log_dir = Path("/tmp")
    return str(log_dir / log_name)


def get_log_level() -> int:
    """Determine log level from environment."""
    level_str = os.environ.get("YFINANCE_LOG_LEVEL", "").upper()
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if level_str in valid_levels:
        return getattr(logging, level_str)
    if level_str:
        sys.stderr.write(
            f"[yfinance-mcp] Invalid YFINANCE_LOG_LEVEL='{level_str}', "
            f"expected one of {valid_levels}. Using WARNING.\n"
        )
    return logging.WARNING


class NDJSONFormatter(logging.Formatter):
    """Formatter that outputs NDJSON for structured analysis."""

    def __init__(self, request_id_getter: Any = None, stats_getter: Any = None):
        super().__init__()
        self._get_request_id = request_id_getter
        self._get_stats = stats_getter

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "loc": f"{record.filename}:{record.lineno}",
            "msg": record.getMessage(),
        }
        if self._get_request_id:
            req_id = self._get_request_id()
            if req_id:
                entry["request_id"] = req_id
        if record.levelno <= logging.INFO and self._get_stats:
            entry["stats"] = self._get_stats()
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry)


def configure_logging(request_id_getter: Any = None, stats_getter: Any = None) -> logging.Logger:
    """Configure logging with platform support and rotation."""
    log_file_env = os.environ.get("YFINANCE_LOG_FILE")
    log_file = log_file_env if log_file_env is not None else get_default_log_path()
    log_level = get_log_level()
    enable_console = os.environ.get("YFINANCE_LOG_CONSOLE", "").lower() in ("1", "true", "yes")

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
        file_handler.setFormatter(NDJSONFormatter(request_id_getter, stats_getter))
        handlers.append(file_handler)

    if enable_console:
        log_stream = (
            sys.stdout
            if os.environ.get("YFINANCE_LOG_STREAM", "").lower() == "stdout"
            else sys.stderr
        )
        console_handler = logging.StreamHandler(log_stream)
        console_handler.setFormatter(NDJSONFormatter(request_id_getter, stats_getter))
        handlers.append(console_handler)

    if not handlers:
        handlers.append(logging.NullHandler())

    logging.root.handlers.clear()
    for handler in handlers:
        logging.root.addHandler(handler)
    logging.root.setLevel(log_level)

    app_logger = logging.getLogger(LOGGER_NAME)
    app_logger.setLevel(log_level)

    mcp_log_level = max(log_level, logging.WARNING)
    for name in ("mcp", "mcp.server", "mcp.server.lowlevel", "mcp.server.lowlevel.server"):
        mcp_logger = logging.getLogger(name)
        mcp_logger.handlers.clear()
        mcp_logger.propagate = False
        for handler in handlers:
            mcp_logger.addHandler(handler)
        mcp_logger.setLevel(mcp_log_level)

    return app_logger


def fmt(data: Any) -> str:
    """Format result as compact JSON (no indentation)."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.copy()
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.strftime("%Y-%m-%d")
        return json.dumps(data.to_dict(), default=str, separators=(",", ":"))
    return json.dumps(data, default=str, separators=(",", ":"))


def fmt_toon(
    df: pd.DataFrame,
    wrapper_key: str | None = None,
    issues: dict | None = None,
    summaries: dict | None = None,
    tz: str | None = None,
) -> str:
    """Format DataFrame as TOON for token-efficient LLM responses.

    Uses delta-encoded split format for ~56% token reduction vs row-oriented JSON:
    - _: schema hint for LLM comprehension ("ts[i] = t0 + sum(dt[0..i])")
    - cols: column names (excluding timestamp)
    - t0: first timestamp as ISO 8601 with timezone offset (e.g., "2024-01-15-05:00"
      for daily, "2024-01-15T09:30-05:00" for intraday)
    - dt: time deltas from previous row (first element is always 0)
    - dt_unit: "day" for daily data, "min" for intraday
    - rows: value tuples in column order

    If issues is provided, it's included as _issues in the TOON structure.
    If summaries is provided, each key is added at the top level.
    """
    df = df.copy()
    cols = df.columns.tolist()

    # Schema hint for LLM comprehension (adds ~15 tokens, improves interpretability)
    schema = "ts[i] = t0 + sum(dt[0..i])"

    if len(df) == 0:
        data: dict = {
            "_": schema,
            "cols": cols,
            "t0": None,
            "dt": [],
            "dt_unit": "day",
            "rows": [],
        }
    elif isinstance(df.index, pd.DatetimeIndex):
        first_ts = df.index[0]
        has_time = first_ts.hour != 0 or first_ts.minute != 0

        # Validate tz is a proper IANA timezone string (e.g., "America/New_York")
        valid_tz = isinstance(tz, str) and "/" in tz

        if has_time:
            if valid_tz:
                localized = first_ts.to_pydatetime().replace(tzinfo=ZoneInfo(tz))
                t0 = localized.strftime("%Y-%m-%dT%H:%M%z")
                t0 = t0[:-2] + ":" + t0[-2:]
            else:
                t0 = first_ts.strftime("%Y-%m-%dT%H:%M")
            minutes = ((df.index - first_ts).total_seconds() / 60).astype(int)
            dt = [0] + [int(minutes[i] - minutes[i - 1]) for i in range(1, len(minutes))]
            dt_unit = "min"
        else:
            if valid_tz:
                localized = first_ts.to_pydatetime().replace(tzinfo=ZoneInfo(tz))
                t0 = localized.strftime("%Y-%m-%dT00:00%z")
                t0 = t0[:-2] + ":" + t0[-2:]
            else:
                t0 = first_ts.strftime("%Y-%m-%d")
            days = (df.index - first_ts).days
            dt = [0] + [int(days[i] - days[i - 1]) for i in range(1, len(days))]
            dt_unit = "day"
        rows = df.values.tolist()
        data = {
            "_": schema,
            "cols": cols,
            "t0": t0,
            "dt": dt,
            "dt_unit": dt_unit,
            "rows": rows,
        }
    else:
        # Non-DatetimeIndex is unexpected - fail explicitly rather than produce broken data
        raise TypeError(
            f"fmt_toon expects DatetimeIndex, got {type(df.index).__name__}. "
            "Ensure df.index = pd.to_datetime(df.index) before calling."
        )

    if wrapper_key:
        result: dict = {wrapper_key: data}
    else:
        result = data

    if summaries:
        result.update(summaries)
    if issues:
        result["_issues"] = issues

    return toon_encode(result)


def fmt_toon_dict(data: dict) -> str:
    """Format dict as TOON (for non-DataFrame results)."""
    return toon_encode(data)


def err(e: Exception) -> str:
    """Format error compactly."""
    if isinstance(e, MCPError):
        result = {"err": e.code, "msg": e.message}
        if e.hint:
            result["hint"] = e.hint
        return json.dumps(result, separators=(",", ":"))
    return json.dumps({"err": "ERROR", "msg": str(e)}, separators=(",", ":"))


def to_scalar(val: Any) -> Any:
    """Extract scalar from value (handles Series from multi-index edge cases)."""
    if val is None:
        return None
    if isinstance(val, pd.Series):
        return val.iloc[0] if len(val) > 0 else None
    if isinstance(val, pd.DataFrame):
        return val.iloc[0, 0] if val.size > 0 else None
    return val


def safe_scalar(val: Any) -> Any:
    """Safely extract scalar, catching any ambiguous truth value errors."""
    try:
        return to_scalar(val)
    except (ValueError, TypeError):
        if hasattr(val, "iloc"):
            try:
                return val.iloc[0] if len(val) > 0 else None
            except Exception:
                return None
        return None


def safe_gt(a: Any, b: Any) -> bool:
    """Safe greater-than comparison that handles None and Series."""
    try:
        if a is None or b is None:
            return False
        return float(a) > float(b)
    except (ValueError, TypeError):
        return False


def safe_lt(a: Any, b: Any) -> bool:
    """Safe less-than comparison that handles None and Series."""
    try:
        if a is None or b is None:
            return False
        return float(a) < float(b)
    except (ValueError, TypeError):
        return False


def safe_get(info: dict, key: str) -> Any:
    """Safely get a value from info dict, converting Series to scalar."""
    val = info.get(key)
    return safe_scalar(val)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame for indicator calculations."""
    if df.empty:
        return df

    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)

    df = df.copy()

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone info from DataFrame index for consistent storage/comparison."""
    if df.empty:
        return df
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def add_unknown(result: dict, indicator: str, logger: logging.Logger) -> None:
    """Add indicator to _unknown list in result dict."""
    result.setdefault("_unknown", []).append(indicator)
    logger.warning("technicals_unknown_indicator indicator=%s", indicator)


def round_result(data: dict, decimals: int = 2) -> dict:
    """Round float values and remove None values from result dict."""
    return {
        k: (round(v, decimals) if isinstance(v, float) else v)
        for k, v in data.items()
        if v is not None
    }


def adaptive_decimals(price: float) -> int:
    """Determine decimal places based on price magnitude."""
    if price <= 0 or not math.isfinite(price):
        return 2
    if price >= 1.0:
        return 2
    return max(2, int(-math.log10(price)) + 1)


def parse_moving_avg_period(indicator: str) -> int | None:
    """Parse period from sma_N or ema_N format. Returns None on invalid format."""
    try:
        period = int(indicator.split("_")[1])
        if period < 1:
            return None
        return period
    except (ValueError, IndexError):
        return None


def summarize_args(args: dict) -> str:
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


def calculate_quality(info: dict) -> tuple[int, list[str]]:
    """Calculate quality score (0-7) based on fundamental metrics."""
    score = 0
    details = []

    roa = safe_get(info, "returnOnAssets")
    if safe_gt(roa, 0):
        score += 1
        details.append("ROA>0")

    ocf = safe_get(info, "operatingCashflow")
    if safe_gt(ocf, 0):
        score += 1
        details.append("CashFlow>0")

    net_income = safe_get(info, "netIncomeToCommon")
    if safe_gt(ocf, net_income):
        score += 1
        details.append("CashFlow>NetIncome")

    current_ratio = safe_get(info, "currentRatio")
    if safe_gt(current_ratio, 1):
        score += 1
        details.append("CurrentRatio>1")

    debt_equity = safe_get(info, "debtToEquity")
    if safe_lt(debt_equity, 100) and debt_equity is not None:
        score += 1
        details.append("DebtEquity<100%")

    gross_margin = safe_get(info, "grossMargins")
    if safe_gt(gross_margin, 0.2):
        score += 1
        details.append("GrossMargin>20%")

    roe = safe_get(info, "returnOnEquity")
    if safe_gt(roe, 0.1):
        score += 1
        details.append("ROE>10%")

    return score, details


SEARCH_STRIP_SUFFIXES = (
    "bank",
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "ltd",
    "ltd.",
    "limited",
    "group",
    "holdings",
    "holding",
    "ag",
    "sa",
    "plc",
    "co",
    "co.",
    "company",
)


def _filter_by_exchange(quotes: list[dict], exchange: str | None) -> list[dict]:
    """Filter search results by exchange if specified.

    Returns empty list if exchange filter is specified but no results match,
    rather than falling back to unfiltered results.
    """
    if not exchange or not quotes:
        return quotes
    exchange_upper = exchange.upper()
    return [q for q in quotes if q.get("exchange", "").upper() == exchange_upper]


class SearchResult:
    """Result from smart_search with optional metadata about available exchanges."""

    def __init__(self, quotes: list[dict], available_exchanges: list[str] | None = None):
        self.quotes = quotes
        self.available_exchanges = available_exchanges

    def __bool__(self) -> bool:
        return bool(self.quotes)

    def __iter__(self):
        return iter(self.quotes)

    def __getitem__(self, index):
        return self.quotes[index]


def smart_search(
    query: str,
    max_results: int = 1,
    exchange: str | None = None,
    logger: logging.Logger | None = None,
) -> SearchResult:
    """Search with fallback strategies for better results.

    1. Try original query
    2. If no results, strip common suffixes (Bank, Inc, Corp, etc.)
    3. If still no results, try first word only

    When exchange is specified, filters results to match that exchange.
    Returns SearchResult with quotes and available_exchanges if exchange filter had no matches.
    """
    fetch_count = max(max_results, 10) if exchange else max_results
    all_quotes: list[dict] = []

    search = yf.Search(query, max_results=fetch_count)
    if search.quotes:
        all_quotes.extend(search.quotes)
        results = _filter_by_exchange(search.quotes, exchange)[:max_results]
        if results:
            if logger:
                logger.debug(
                    "search_found query=%r exchange=%r results=%d",
                    query,
                    exchange,
                    len(results),
                )
            return SearchResult(results)

    words = query.lower().split()
    stripped = [w for w in words if w not in SEARCH_STRIP_SUFFIXES]
    if stripped and stripped != words:
        stripped_query = " ".join(stripped)
        search = yf.Search(stripped_query, max_results=fetch_count)
        if search.quotes:
            all_quotes.extend(search.quotes)
            results = _filter_by_exchange(search.quotes, exchange)[:max_results]
            if results:
                if logger:
                    logger.debug(
                        "search_found_stripped query=%r stripped=%r exchange=%r results=%d",
                        query,
                        stripped_query,
                        exchange,
                        len(results),
                    )
                return SearchResult(results)

    if len(words) > 1:
        first_word = words[0]
        if first_word not in SEARCH_STRIP_SUFFIXES and len(first_word) >= 3:
            search = yf.Search(first_word, max_results=fetch_count)
            if search.quotes:
                all_quotes.extend(search.quotes)
                results = _filter_by_exchange(search.quotes, exchange)[:max_results]
                if results:
                    if logger:
                        logger.debug(
                            "search_found_first_word query=%r first=%r exchange=%r results=%d",
                            query,
                            first_word,
                            exchange,
                            len(results),
                        )
                    return SearchResult(results)

    # No matches found - collect available exchanges if we had quotes but exchange filter failed
    available_exchanges = None
    if exchange and all_quotes:
        exchanges_found = sorted(
            set(q.get("exchange", "") for q in all_quotes if q.get("exchange"))
        )
        if exchanges_found:
            available_exchanges = exchanges_found
            if logger:
                logger.debug(
                    "search_exchange_mismatch query=%r requested=%r available=%r",
                    query,
                    exchange,
                    exchanges_found,
                )

    return SearchResult([], available_exchanges)


PERIOD_TO_DAYS = {
    "1d": 1,
    "5d": 5,
    "1w": 7,
    "2w": 14,
    "1mo": 30,
    "2mo": 60,
    "3mo": 90,
    "6mo": 180,
    "9mo": 270,
    "ytd": 180,  # approximate, varies by date
    "1y": 365,
    "18mo": 548,
    "2y": 730,
    "3y": 1095,
    "5y": 1825,
    "10y": 3650,
    "max": 7300,  # ~20 years as fallback
}

PERIOD_DELTAS = {
    "1d": relativedelta(days=1),
    "5d": relativedelta(days=5),
    "1mo": relativedelta(months=1),
    "3mo": relativedelta(months=3),
    "6mo": relativedelta(months=6),
    "1y": relativedelta(years=1),
    "2y": relativedelta(years=2),
    "5y": relativedelta(years=5),
    "10y": relativedelta(years=10),
    "ytd": None,  # Special case: handled separately
    "max": relativedelta(years=99),
}

OHLCV_COLS_TO_SHORT = {
    "Open": "o",
    "High": "h",
    "Low": "l",
    "Close": "c",
    "Adj Close": "ac",
    "Volume": "v",
}
OHLCV_COLS_TO_LONG = {
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "ac": "Adj Close",
    "v": "Volume",
}

INTRADAY_INTERVALS = {"1m", "5m", "15m", "30m", "1h"}

# Periods natively supported by yfinance (others convert to start/end dates)
YFINANCE_NATIVE_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}

MAX_PERIOD_OPTIONS = 7


def get_valid_periods(max_trading_days: int | None = None) -> list[str]:
    """Return up to MAX_PERIOD_OPTIONS period options that fit within max_trading_days.

    Always includes "ytd" if it fits. Selects evenly distributed options.
    """
    if max_trading_days is None:
        max_trading_days = MAX_PERIOD_DAYS

    periods_by_duration = sorted(PERIOD_TO_DAYS.keys(), key=lambda p: PERIOD_TO_DAYS[p])
    valid = [
        p
        for p in periods_by_duration
        if int(PERIOD_TO_DAYS.get(p, 0) * TRADING_DAYS_PER_WEEK / 7) <= max_trading_days
    ]

    if len(valid) <= MAX_PERIOD_OPTIONS:
        return valid

    # Select evenly distributed options, always including ytd and the longest
    result = []
    step = len(valid) / (MAX_PERIOD_OPTIONS - 1)  # -1 to ensure we include last

    for i in range(MAX_PERIOD_OPTIONS - 1):
        idx = int(i * step)
        if valid[idx] not in result:
            result.append(valid[idx])

    # Always include the longest valid period
    if valid[-1] not in result:
        result.append(valid[-1])

    # Ensure ytd is included if valid
    if "ytd" in valid and "ytd" not in result:
        # Replace the closest duration option with ytd
        ytd_days = PERIOD_TO_DAYS["ytd"]
        closest_idx = 0
        closest_diff = float("inf")
        for i, p in enumerate(result):
            if p != result[-1]:  # don't replace the longest
                diff = abs(PERIOD_TO_DAYS.get(p, 0) - ytd_days)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_idx = i
        result[closest_idx] = "ytd"

    # Sort by duration
    return sorted(result, key=lambda p: PERIOD_TO_DAYS.get(p, 0))


def period_to_date_range(period: str) -> tuple[str | None, str | None, str | None]:
    """Convert period to (period, start, end) tuple.

    Native yfinance periods return (period, None, None).
    Custom periods convert to (None, start_date, end_date).
    """
    if period in YFINANCE_NATIVE_PERIODS:
        return (period, None, None)

    days = PERIOD_TO_DAYS.get(period)
    if days is None:
        return (period, None, None)  # let yfinance handle unknown periods

    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    return (None, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))


TARGET_POINTS = int(os.environ.get("YFINANCE_TARGET_POINTS", "200"))

# Trading minutes by exchange (yfinance exchange codes)
# Calculated as: trading_hours × 60 - lunch_break_minutes
EXCHANGE_TRADING_MINUTES: dict[str, int] = {
    # Americas (no lunch break)
    "NYQ": 390,
    "NYSE": 390,  # NYSE: 9:30-16:00 = 6.5h
    "NMS": 390,
    "NGM": 390,
    "NCM": 390,
    "NASDAQ": 390,  # NASDAQ variants
    "PCX": 390,
    "NYQ ARCA": 390,  # NYSE Arca
    "ASE": 390,
    "AMEX": 390,  # NYSE American (AMEX)
    "BTS": 390,  # BATS
    "TOR": 390,
    "TSX": 390,
    "CNQ": 390,
    "NEO": 390,  # Toronto: 9:30-16:00
    "MEX": 390,  # Mexico BMV: 8:30-15:00
    "SAO": 420,  # Brazil B3: 10:00-17:00 = 7h
    # Europe (no lunch break, mostly 8.5h)
    "LSE": 510,
    "IOB": 510,  # London: 8:00-16:30
    "AMS": 510,
    "PAR": 510,
    "BRU": 510,
    "LIS": 510,  # Euronext: 9:00-17:30
    "FRA": 510,
    "GER": 510,
    "XETRA": 510,
    "STU": 510,
    "MUN": 510,  # Germany
    "SIX": 510,
    "EBS": 510,
    "VTX": 510,  # Switzerland: 9:00-17:30
    "MCE": 510,
    "BME": 510,  # Spain: 9:00-17:30
    "MIL": 510,
    "BIT": 510,  # Italy: 9:00-17:30
    "VIE": 510,
    "WBAG": 510,  # Austria: 9:00-17:30
    "CPH": 510,
    "STO": 510,
    "HEL": 510,
    "ICE": 510,
    "OSL": 510,  # Nordic
    "WSE": 510,  # Warsaw: 9:00-17:00
    "ATH": 330,  # Athens: 10:00-17:20
    "IST": 420,  # Istanbul: 10:00-18:00 (with breaks)
    # Asia-Pacific (with lunch breaks)
    "JPX": 300,
    "TYO": 300,
    "OSA": 300,
    "NGO": 300,
    "SAP": 300,
    "FKA": 300,  # Tokyo: 9:00-15:00 - 1h lunch = 5h
    "SHH": 240,
    "SHG": 240,  # Shanghai: 9:30-15:00 - 1.5h lunch = 4h
    "SHZ": 240,
    "SZS": 240,  # Shenzhen: same as Shanghai
    "HKG": 330,
    "HKSE": 330,  # Hong Kong: 9:30-16:00 - 1h lunch = 5.5h
    "KSC": 390,
    "KOE": 390,
    "KOSDAQ": 390,  # Korea: 9:00-15:30 = 6.5h
    "TAI": 270,
    "TWO": 270,  # Taiwan: 9:00-13:30 = 4.5h
    "SGX": 420,
    "SES": 420,  # Singapore: 9:00-17:00 - 1h lunch = 7h
    "ASX": 360,
    "AXS": 360,  # Australia: 10:00-16:00 = 6h
    "NZE": 360,  # New Zealand: 10:00-16:45
    "NSI": 375,
    "BSE": 375,
    "NSE": 375,  # India: 9:15-15:30 = 6.25h
    "BKK": 330,
    "SET": 330,  # Thailand: 10:00-16:30 - 1h lunch = 5.5h
    "JKT": 330,
    "IDX": 330,  # Indonesia: 9:00-16:00 - 1.5h lunch = 5.5h
    "KLS": 330,
    "KLSE": 330,  # Malaysia: 9:00-17:00 - 2.5h lunch = 5.5h
    "PHS": 270,  # Philippines: 9:30-12:00, 13:30-15:30 = 4.5h
    # Middle East
    "SAU": 300,
    "TADAWUL": 300,  # Saudi: 10:00-15:00 = 5h
    "DFM": 240,
    "ADX": 240,  # UAE: 10:00-14:00 = 4h
    "TLV": 330,
    "TASE": 330,  # Israel: 9:25-17:25 (Sun-Thu) = 8h but ~5.5h effective
    # Africa
    "JNB": 300,
    "JSE": 300,  # Johannesburg: 9:00-17:00 but ~5h effective
}

DEFAULT_TRADING_MINUTES = 390  # US markets as fallback
TRADING_DAYS_PER_WEEK = 5

# Symbol suffix to trading minutes (for quick lookup without API call)
SUFFIX_TRADING_MINUTES: dict[str, int] = {
    ".T": 300,  # Tokyo
    ".L": 510,  # London
    ".SS": 240,  # Shanghai
    ".SZ": 240,  # Shenzhen
    ".HK": 330,  # Hong Kong
    ".KS": 390,  # Korea
    ".TW": 270,  # Taiwan
    ".SI": 420,  # Singapore
    ".AX": 360,  # Australia
    ".NZ": 360,  # New Zealand
    ".NS": 375,  # India NSE
    ".BO": 375,  # India BSE
    ".BK": 330,  # Thailand
    ".JK": 330,  # Indonesia
    ".KL": 330,  # Malaysia
    ".TO": 390,  # Toronto
    ".V": 390,  # TSX Venture
    ".MX": 390,  # Mexico
    ".SA": 420,  # Brazil
    ".DE": 510,  # Germany
    ".F": 510,  # Frankfurt
    ".PA": 510,  # Paris
    ".AS": 510,  # Amsterdam
    ".BR": 510,  # Brussels
    ".MI": 510,  # Milan
    ".MC": 510,  # Madrid
    ".SW": 510,  # Switzerland
    ".VI": 510,  # Vienna
    ".CO": 510,  # Copenhagen
    ".ST": 510,  # Stockholm
    ".HE": 510,  # Helsinki
    ".OL": 510,  # Oslo
    ".IS": 420,  # Istanbul
    ".TA": 330,  # Tel Aviv
    ".SR": 300,  # Saudi
}


def get_trading_minutes(symbol: str, exchange: str | None = None) -> int:
    """Get trading minutes for a symbol based on its exchange.

    Tries symbol suffix first (fast), then exchange code lookup.
    Falls back to US market hours (390 min) if unknown.
    """
    symbol_upper = symbol.upper()

    for suffix, minutes in SUFFIX_TRADING_MINUTES.items():
        if symbol_upper.endswith(suffix.upper()):
            return minutes

    if exchange:
        return EXCHANGE_TRADING_MINUTES.get(exchange, DEFAULT_TRADING_MINUTES)

    return DEFAULT_TRADING_MINUTES


def get_intervals(trading_minutes: int) -> list[tuple[str, float, int | None]]:
    """Build interval config for given trading minutes."""
    return [
        ("5m", trading_minutes / 5, 60),
        ("15m", trading_minutes / 15, 60),
        ("30m", trading_minutes / 30, 60),
        ("1h", trading_minutes / 60, 730),
        ("1d", 1, None),
        ("1wk", 1 / TRADING_DAYS_PER_WEEK, None),
    ]


# Default intervals for US markets (used when symbol not provided)
INTERVALS: list[tuple[str, float, int | None]] = get_intervals(DEFAULT_TRADING_MINUTES)

# Max period in trading days (1wk = 5 trading days is minimum resolution)
MAX_PERIOD_DAYS = TARGET_POINTS * TRADING_DAYS_PER_WEEK


def select_interval(
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    symbol: str | None = None,
    exchange: str | None = None,
) -> str:
    """Select the coarsest interval where bars >= TARGET_POINTS/2.

    Iterates from coarsest (1wk) to finest (5m) and picks the first
    interval that produces at least half of TARGET_POINTS bars while
    respecting Yahoo API limits.

    Trading hours are determined by the symbol's exchange (via suffix or
    exchange code). Falls back to US market hours if unknown.
    """
    if start:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end) if end else pd.Timestamp.now()
        calendar_days = float((end_date - start_date).days)
    elif period:
        calendar_days = float(PERIOD_TO_DAYS.get(period, 90))
    else:
        calendar_days = 90.0

    trading_days = calendar_days * TRADING_DAYS_PER_WEEK / 7

    if symbol:
        trading_minutes = get_trading_minutes(symbol, exchange)
        intervals = get_intervals(trading_minutes)
    else:
        intervals = INTERVALS

    min_bars = TARGET_POINTS / 2

    for interval, ppd, max_days in reversed(intervals):
        if max_days is not None and calendar_days > max_days:
            continue
        if trading_days * ppd >= min_bars:
            return interval

    return intervals[0][0]


class DateRangeExceededError(Exception):
    """Raised when requested date range exceeds MAX_PERIOD_DAYS (in trading days)."""

    def __init__(
        self,
        requested_trading_days: int,
        max_trading_days: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        self.requested_days = requested_trading_days
        self.max_days = max_trading_days

        max_weeks = max_trading_days // TRADING_DAYS_PER_WEEK
        max_calendar_days = int(max_trading_days * 7 / TRADING_DAYS_PER_WEEK)
        max_years = round(max_calendar_days / 365, 1)
        num_calls = math.ceil(requested_trading_days / max_trading_days)

        msg_parts = [
            f"Date range too long: exceeds {max_weeks}-week limit "
            f"(~{max_years} years, derived from YFINANCE_TARGET_POINTS={TARGET_POINTS})."
        ]

        if start_date and end_date and num_calls > 1:
            msg_parts.append(f"Split into {num_calls} sequential requests:")
            chunk_calendar_days = max_calendar_days
            current = start_date
            for i in range(num_calls):
                chunk_end = min(current + timedelta(days=chunk_calendar_days), end_date)
                msg_parts.append(
                    f"  {i + 1}. start={current.isoformat()}, end={chunk_end.isoformat()}"
                )
                current = chunk_end + timedelta(days=1)
                if current > end_date:
                    break
        else:
            msg_parts.append(
                f"Use period='3y' or split into {num_calls} requests of ~{max_years} years each."
            )

        super().__init__(" ".join(msg_parts))


def validate_date_range(
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> None:
    """Validate that date range does not exceed MAX_PERIOD_DAYS (in trading days)."""
    if start:
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date() if end else date.today()
        calendar_days = (end_date - start_date).days
        trading_days = int(calendar_days * TRADING_DAYS_PER_WEEK / 7)

        if trading_days > MAX_PERIOD_DAYS:
            raise DateRangeExceededError(trading_days, MAX_PERIOD_DAYS, start_date, end_date)
    elif period:
        calendar_days = PERIOD_TO_DAYS.get(period, 90)
        trading_days = int(calendar_days * TRADING_DAYS_PER_WEEK / 7)
        if trading_days > MAX_PERIOD_DAYS:
            raise DateRangeExceededError(trading_days, MAX_PERIOD_DAYS)


def auto_downsample(
    df: pd.DataFrame,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Downsample DataFrame to approximately TARGET_POINTS rows.

    Selects evenly spaced rows to maintain trend visibility while
    reducing token usage for LLM processing.
    """
    _ = period, start, end  # unused, kept for API consistency

    if len(df) <= TARGET_POINTS:
        return df

    step = len(df) / TARGET_POINTS
    indices = [int(i * step) for i in range(TARGET_POINTS)]
    indices[-1] = len(df) - 1  # always include the last row

    return df.iloc[indices]


def ohlc_resample(df: pd.DataFrame, target_points: int | None = None) -> pd.DataFrame:
    """Resample OHLCV data preserving price semantics.

    Aggregates multiple bars into fewer bars while preserving:
    - Open: first open in bucket (entry price)
    - High: max of all highs (resistance level)
    - Low: min of all lows (support level)
    - Close: last close in bucket (exit price)
    - Adj Close: last adjusted close in bucket (dividend-adjusted exit price)
    - Volume: sum of all volumes (total activity)

    Uses data-point-based bucketing (not time-based) to correctly handle
    trading gaps like weekends and non-market hours.
    """
    if target_points is None:
        target_points = TARGET_POINTS

    if len(df) <= target_points:
        return df

    col_lower = {c.lower(): c for c in df.columns}
    o_col = col_lower.get("o")
    h_col = col_lower.get("h")
    l_col = col_lower.get("l")
    c_col = col_lower.get("c")
    ac_col = col_lower.get("ac")
    v_col = col_lower.get("v")

    if not any([o_col, h_col, l_col, c_col]):
        return auto_downsample(df)

    bucket_size = len(df) / target_points
    rows = []
    indices = []

    for i in range(target_points):
        start_idx = int(i * bucket_size)
        end_idx = int((i + 1) * bucket_size) if i < target_points - 1 else len(df)
        bucket = df.iloc[start_idx:end_idx]

        if bucket.empty:
            continue

        row = {}
        if o_col:
            row[o_col] = bucket[o_col].iloc[0]
        if h_col:
            row[h_col] = bucket[h_col].max()
        if l_col:
            row[l_col] = bucket[l_col].min()
        if c_col:
            row[c_col] = bucket[c_col].iloc[-1]
        if ac_col:
            row[ac_col] = bucket[ac_col].iloc[-1]
        if v_col:
            row[v_col] = bucket[v_col].sum()

        rows.append(row)
        indices.append(bucket.index[0])

    return pd.DataFrame(rows, index=indices)


def _lttb_indices(data: list[float], target_points: int) -> list[int]:
    """Compute indices for LTTB (Largest-Triangle-Three-Buckets) downsampling.

    LTTB preserves visual shape by selecting the point in each bucket that
    forms the largest triangle with its neighbors. This keeps trend reversals,
    extremes, and significant changes while discarding redundant points.

    Reference: https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf
    """
    n = len(data)
    if n <= target_points or target_points <= 0:
        return list(range(n))
    if target_points == 1:
        return [n - 1]  # just last point
    if target_points == 2:
        return [0, n - 1]  # first and last

    indices = [0]
    bucket_size = (n - 2) / (target_points - 2)

    a_idx = 0
    for i in range(target_points - 2):
        bucket_start = int((i + 1) * bucket_size) + 1
        bucket_end = int((i + 2) * bucket_size) + 1
        bucket_end = min(bucket_end, n - 1)

        next_bucket_start = int((i + 2) * bucket_size) + 1
        next_bucket_end = int((i + 3) * bucket_size) + 1
        next_bucket_end = min(next_bucket_end, n)

        if next_bucket_start < n:
            avg_next = sum(data[next_bucket_start:next_bucket_end]) / max(
                1, next_bucket_end - next_bucket_start
            )
        else:
            avg_next = data[-1]

        max_area = -1.0
        max_idx = bucket_start

        a_val = data[a_idx]
        for j in range(bucket_start, bucket_end):
            area = abs(
                (a_idx - next_bucket_start) * (data[j] - a_val) - (a_idx - j) * (avg_next - a_val)
            )
            if area > max_area:
                max_area = area
                max_idx = j

        indices.append(max_idx)
        a_idx = max_idx

    indices.append(n - 1)
    return indices


def lttb_downsample(df: pd.DataFrame, target_points: int | None = None) -> pd.DataFrame:
    """Downsample DataFrame using LTTB algorithm.

    LTTB (Largest-Triangle-Three-Buckets) preserves visual shape by selecting
    points that form the largest triangles with neighbors. This keeps:
    - Trend reversals and direction changes
    - Local extremes (peaks and troughs)
    - Significant value changes

    Ideal for indicator time-series where preserving crossovers and
    signal patterns matters more than exact OHLC semantics.
    """
    if target_points is None:
        target_points = TARGET_POINTS

    if len(df) <= target_points:
        return df

    if df.empty:
        return df

    ref_col = df.columns[0]
    ref_data = df[ref_col].tolist()

    indices = _lttb_indices(ref_data, target_points)

    return df.iloc[indices]


def fetch_japan_etf_expense(symbol: str, logger: logging.Logger | None = None) -> float | None:
    """Fetch expense ratio from Yahoo Finance Japan for Japanese ETFs.

    Scrapes the trust fee (信託報酬) from the Yahoo Finance Japan page.
    Results are cached in DuckDB with 1-day TTL to avoid excessive requests.

    Args:
        symbol: Stock symbol (e.g., "282A.T" or "282A")
        logger: Optional logger for debug output

    Returns:
        Expense ratio as a decimal (e.g., 0.11 for 0.11%), or None if unavailable
    """
    # cache_duckdb imports helpers, so import here to avoid circular dependency
    from .cache_duckdb import DuckDBCacheBackend

    code = symbol.upper().replace(".T", "")
    cache_key = f"{code}.T"

    cache = DuckDBCacheBackend()
    try:
        cached_value, found = cache.get_etf_expense(cache_key)
        if found:
            if logger:
                logger.debug("japan_etf_expense_cache_hit symbol=%s value=%s", symbol, cached_value)
            return cached_value

        url = f"https://finance.yahoo.co.jp/quote/{code}.T"
        max_retries = min(3, len(USER_AGENTS))

        for attempt in range(max_retries):
            ua = USER_AGENTS[attempt % len(USER_AGENTS)]
            headers = {"User-Agent": ua}

            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    text = resp.text
                    match = re.search(r"信託報酬.*?>(\d+\.\d+)%?<", text, re.DOTALL)
                    if match:
                        expense_ratio = float(match.group(1))
                        cache.store_etf_expense(
                            cache_key, expense_ratio, exchange="JPX", source="yahoo_japan"
                        )
                        if logger:
                            logger.debug(
                                "japan_etf_expense_fetched symbol=%s value=%s",
                                symbol,
                                expense_ratio,
                            )
                        return expense_ratio

                    if logger:
                        logger.debug("japan_etf_expense_not_found symbol=%s", symbol)
                    cache.store_etf_expense(cache_key, None, exchange="JPX", source="yahoo_japan")
                    return None

                if logger:
                    logger.debug(
                        "japan_etf_expense_retry symbol=%s status=%d attempt=%d/%d",
                        symbol,
                        resp.status_code,
                        attempt + 1,
                        max_retries,
                    )

            except requests.RequestException as e:
                if logger:
                    logger.debug(
                        "japan_etf_expense_retry symbol=%s error=%s attempt=%d/%d",
                        symbol,
                        e,
                        attempt + 1,
                        max_retries,
                    )

        if logger:
            logger.warning("japan_etf_expense_failed symbol=%s attempts=%d", symbol, max_retries)
        cache.store_etf_expense(cache_key, None, exchange="JPX", source="yahoo_japan")
        return None
    finally:
        cache.close()
