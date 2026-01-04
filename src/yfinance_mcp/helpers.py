"""Helper functions for Yahoo Finance MCP Server."""

import json
import logging
import math
import os
import platform
import sys
import tempfile
from datetime import date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from toon_format import encode as toon_encode

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
    level_str = os.environ.get("MCP_LOG_LEVEL", "").upper()
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if level_str in valid_levels:
        return getattr(logging, level_str)
    if level_str:
        sys.stderr.write(
            f"[yfinance-mcp] Invalid MCP_LOG_LEVEL='{level_str}', "
            f"expected one of {valid_levels}. Using WARNING.\n"
        )
    if os.environ.get("MCP_DEBUG"):
        return logging.DEBUG
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


def configure_logging(request_id_getter: Any = None, stats_getter: Any = None) -> logging.Logger:
    """Configure logging with platform support and rotation."""
    log_file = os.environ.get("MCP_LOG_FILE") or get_default_log_path()
    log_level = get_log_level()
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
        file_handler.setFormatter(NDJSONFormatter(request_id_getter, stats_getter))
        handlers.append(file_handler)

    if enable_console:
        console_handler = logging.StreamHandler(sys.stderr)
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            console_handler.setFormatter(ConsoleFormatter())
        else:
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


def fmt_toon(df: pd.DataFrame, wrapper_key: str | None = None) -> str:
    """Format DataFrame as TOON for token-efficient LLM responses.

    Converts DataFrame to tabular TOON format with date index as 'd' column.
    TOON eliminates repeated keys by using a header line that declares the schema,
    followed by comma-separated rows, achieving ~45% token reduction vs JSON.
    """
    df = df.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime("%Y-%m-%d")

    records = []
    for idx, row in df.iterrows():
        record = {"d": idx}
        record.update(row.to_dict())
        records.append(record)

    if wrapper_key:
        return toon_encode({wrapper_key: records})
    return toon_encode(records)


def err(e: Exception) -> str:
    """Format error compactly."""
    if isinstance(e, MCPError):
        return json.dumps({"err": e.code, "msg": e.message}, separators=(",", ":"))
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

    for col in ["Open", "High", "Low", "Close", "Volume"]:
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
        exchanges_found = sorted(set(q.get("exchange", "") for q in all_quotes if q.get("exchange")))
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

OHLCV_COLS_TO_SHORT = {"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"}
OHLCV_COLS_TO_LONG = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}

INTRADAY_INTERVALS = {"1m", "5m", "15m", "30m", "1h"}

# Ordered by duration for generating valid options
PERIOD_OPTIONS_ORDERED = [
    "1d",
    "5d",
    "1w",
    "2w",
    "1mo",
    "2mo",
    "3mo",
    "6mo",
    "9mo",
    "ytd",
    "1y",
    "18mo",
    "2y",
    "3y",
    "5y",
    "10y",
    "max",
]

# Periods natively supported by yfinance (others convert to start/end dates)
YFINANCE_NATIVE_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}

MAX_PERIOD_OPTIONS = 7


def get_valid_periods(max_span_days: int) -> list[str]:
    """Return up to MAX_PERIOD_OPTIONS period options that fit within max_span_days.

    Always includes "ytd" if it fits. Selects evenly distributed options.
    """
    valid = [p for p in PERIOD_OPTIONS_ORDERED if PERIOD_TO_DAYS.get(p, 0) <= max_span_days]

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


TARGET_POINTS = int(os.environ.get("YFINANCE_TARGET_POINTS", "120"))

# Interval base config: (interval, points_per_day, max_days, switch_multiplier)
#
# points_per_day: trading hours (6.5h) × intervals per hour
# max_days: Yahoo Finance API limit (None = unlimited)
# switch_multiplier: switch when prev interval produces this × TARGET_POINTS
#   - Intraday: lower multiplier (minimize wasted API fetches)
#   - Daily+: higher multiplier (downsampling cached data is cheap)
#
# min_span_days is computed as: (TARGET_POINTS × multiplier) / prev_points_per_day
_INTERVAL_BASE: list[tuple[str, float, int | None, float]] = [
    ("5m", 78, 60, 1.0),
    ("15m", 26, 60, 1.95),
    ("30m", 13, 60, 1.083),
    ("1h", 6.5, 730, 1.3),
    ("1d", 1, None, 4.33),
    ("1wk", 0.2, None, 3.33),
    ("1mo", 0.048, None, 8 / 3),
]


def build_interval_config(
    target_points: int,
) -> list[tuple[str, float, int | None, float]]:
    """Build interval config with min_span_days computed from target_points."""
    result: list[tuple[str, float, int | None, float]] = []
    for i, (interval, ppd, max_days, mult) in enumerate(_INTERVAL_BASE):
        min_span = 0.0 if i == 0 else (target_points * mult) / _INTERVAL_BASE[i - 1][1]
        result.append((interval, ppd, max_days, min_span))
    return result


INTERVAL_CONFIG = build_interval_config(TARGET_POINTS)

MAX_SPAN_DAYS = int(TARGET_POINTS / _INTERVAL_BASE[-1][1])


def select_interval(
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    *,
    _config: list[tuple[str, float, int | None, float]] | None = None,
) -> str:
    """Select optimal interval to produce approximately TARGET_POINTS data points."""
    config = _config if _config is not None else INTERVAL_CONFIG

    if start:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end) if end else pd.Timestamp.now()
        span_days = float((end_date - start_date).days)
    elif period:
        span_days = float(PERIOD_TO_DAYS.get(period, 90))
    else:
        span_days = 90.0

    for interval, _ppd, max_days, min_span in reversed(config):
        if span_days >= min_span:
            if max_days is None or span_days <= max_days:
                return interval
    return "5m"


class DateRangeExceededError(Exception):
    """Raised when requested date range exceeds MAX_SPAN_DAYS."""

    def __init__(
        self,
        requested_days: int,
        max_days: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        self.requested_days = requested_days
        self.max_days = max_days

        max_years = round(max_days / 365, 1)
        num_calls = math.ceil(requested_days / max_days)

        # Build actionable message with specific date ranges
        msg_parts = [
            f"Date range too long: {requested_days} days exceeds {max_days}-day limit "
            f"(~{max_years} years, derived from YFINANCE_TARGET_POINTS={TARGET_POINTS})."
        ]

        if start_date and end_date and num_calls > 1:
            msg_parts.append(f"Split into {num_calls} sequential requests:")
            chunk_days = max_days
            current = start_date
            for i in range(num_calls):
                chunk_end = min(current + timedelta(days=chunk_days), end_date)
                msg_parts.append(
                    f"  {i + 1}. start={current.isoformat()}, end={chunk_end.isoformat()}"
                )
                current = chunk_end + timedelta(days=1)
                if current > end_date:
                    break
        else:
            msg_parts.append(
                f"Use period='5y' or split into {num_calls} requests of ~{max_years} years each."
            )

        super().__init__(" ".join(msg_parts))


def validate_date_range(
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> None:
    """Validate that date range does not exceed MAX_SPAN_DAYS."""
    if start:
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date() if end else date.today()
        span_days = (end_date - start_date).days

        if span_days > MAX_SPAN_DAYS:
            raise DateRangeExceededError(span_days, MAX_SPAN_DAYS, start_date, end_date)
    elif period:
        period_days = PERIOD_TO_DAYS.get(period, 90)
        if period_days > MAX_SPAN_DAYS:
            raise DateRangeExceededError(period_days, MAX_SPAN_DAYS)


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
