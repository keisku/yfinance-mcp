"""Error codes and structured error handling."""

from enum import Enum


class ErrorCode(str, Enum):
    """Error codes for structured error responses."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    SYMBOL_NOT_FOUND = "SYMBOL_NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    CALCULATION_ERROR = "CALCULATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class MCPError(Exception):
    """Base exception for MCP errors."""

    def __init__(self, code: ErrorCode, message: str, details: dict | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
            }
        }


class ValidationError(MCPError):
    """Input validation error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(ErrorCode.VALIDATION_ERROR, message, details)


class SymbolNotFoundError(MCPError):
    """Symbol not found error."""

    def __init__(self, symbol: str):
        super().__init__(
            ErrorCode.SYMBOL_NOT_FOUND,
            f"Try search tool with company name to find correct ticker for '{symbol}'",
            {"symbol": symbol, "suggestion": "use search tool"},
        )


class DataUnavailableError(MCPError):
    """Data unavailable error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(ErrorCode.DATA_UNAVAILABLE, message, details)


class CalculationError(MCPError):
    """Calculation error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(ErrorCode.CALCULATION_ERROR, message, details)
