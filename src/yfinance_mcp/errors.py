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

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict | None = None,
        hint: str | None = None,
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.hint = hint
        super().__init__(message)

    def to_dict(self) -> dict:
        result = {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
            }
        }
        if self.hint:
            result["error"]["hint"] = self.hint
        return result


class ValidationError(MCPError):
    """Input validation error."""

    def __init__(self, message: str, details: dict | None = None, hint: str | None = None):
        super().__init__(ErrorCode.VALIDATION_ERROR, message, details, hint)


class SymbolNotFoundError(MCPError):
    """Symbol not found error."""

    def __init__(self, symbol: str, hint: str | None = None):
        message = f"No results for '{symbol}'."
        default_hint = "Try a more specific query."
        super().__init__(
            ErrorCode.SYMBOL_NOT_FOUND,
            message,
            {"symbol": symbol},
            hint or default_hint,
        )


class DataUnavailableError(MCPError):
    """Data unavailable error."""

    def __init__(self, message: str, details: dict | None = None, hint: str | None = None):
        super().__init__(ErrorCode.DATA_UNAVAILABLE, message, details, hint)


class CalculationError(MCPError):
    """Calculation error."""

    def __init__(self, message: str, details: dict | None = None, hint: str | None = None):
        super().__init__(ErrorCode.CALCULATION_ERROR, message, details, hint)
