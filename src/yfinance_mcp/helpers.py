"""Helper functions for Yahoo Finance MCP Server."""

import json
import logging
import math
import os
import platform
import sys
import tempfile
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from . import LOGGER_NAME
from .errors import MCPError, ValidationError


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


def require_symbol(args: dict, ticker_fn: Any) -> tuple[str, yf.Ticker]:
    """Validate symbol argument and return (symbol, ticker) tuple."""
    symbol = args.get("symbol")
    if not symbol:
        raise ValidationError("symbol required")
    return symbol, ticker_fn(symbol)


def signal_level(value: float, high: float, low: float) -> str:
    """Return overbought/oversold/neutral based on thresholds."""
    if value > high:
        return "overbought"
    if value < low:
        return "oversold"
    return "neutral"


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


def safe_round(value: float, decimals: int = 2) -> float | None:
    """Round value, returning None if NaN or infinite."""
    if pd.isna(value) or not pd.api.types.is_number(value):
        return None
    return round(value, decimals)


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


def smart_search(
    query: str, max_results: int = 1, logger: logging.Logger | None = None
) -> list[dict]:
    """Search with fallback strategies for better results.

    1. Try original query
    2. If no results, strip common suffixes (Bank, Inc, Corp, etc.)
    3. If still no results, try first word only
    """
    search = yf.Search(query, max_results=max_results)
    if search.quotes:
        if logger:
            logger.debug("search_found query=%r results=%d", query, len(search.quotes))
        return search.quotes

    words = query.lower().split()
    stripped = [w for w in words if w not in SEARCH_STRIP_SUFFIXES]
    if stripped and stripped != words:
        stripped_query = " ".join(stripped)
        search = yf.Search(stripped_query, max_results=max_results)
        if search.quotes:
            if logger:
                logger.debug(
                    "search_found_stripped query=%r stripped=%r results=%d",
                    query,
                    stripped_query,
                    len(search.quotes),
                )
            return search.quotes

    if len(words) > 1:
        first_word = words[0]
        if first_word not in SEARCH_STRIP_SUFFIXES and len(first_word) >= 3:
            search = yf.Search(first_word, max_results=max_results)
            if search.quotes:
                if logger:
                    logger.debug(
                        "search_found_first_word query=%r first=%r results=%d",
                        query,
                        first_word,
                        len(search.quotes),
                    )
                return search.quotes

    if logger:
        logger.debug("search_not_found query=%r", query)
    return []
