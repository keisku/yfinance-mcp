"""Yahoo Finance MCP Server."""

__version__ = "0.1.0"
LOGGER_NAME = "yfinance_mcp"

from .server import (  # noqa: E402
    MCPEndpoint,
    create_session_manager,
    create_starlette_app,
    server,
)

__all__ = [
    "MCPEndpoint",
    "create_session_manager",
    "create_starlette_app",
    "server",
]
