"""Yahoo Finance MCP Server."""

import json
import logging

from history import VALID_INTERVALS, history
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from oscillator import oscillator
from trend import trend
from volume import volume

logger = logging.getLogger(__name__)

server = Server("yfinance-mcp")

TOOLS = [
    Tool(
        name="history",
        description="Get OHLCV price history for a symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": 'Ticker symbol (e.g., "AAPL", "7203.T")',
                },
                "interval": {
                    "type": "string",
                    "enum": sorted(VALID_INTERVALS),
                    "description": "Bar granularity",
                },
                "start": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD)",
                },
                "end": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD)",
                },
                "adjust": {
                    "type": "boolean",
                    "description": "Return adjusted prices (default: false). "
                    "Use true for return calculations, performance comparison, "
                    "backtesting, and technical analysis — adjusted prices "
                    "account for splits and dividends so returns are continuous. "
                    "Use false for actual trade prices, support/resistance levels, "
                    "and order book reference.",
                    "default": False,
                },
            },
            "required": ["symbol", "interval", "start", "end"],
        },
    ),
    Tool(
        name="oscillator",
        description=(
            "Daily momentum oscillators for a symbol. "
            "Returns RSI(14) and Stochastic %K(14)/%D(3), plus a current "
            "put_call_ratio snapshot aggregated across all listed "
            "expirations (volume_based, oi_based) — null for tickers "
            "without listed options (non-US equities, indexes, crypto). "
            "Uses daily bars; warmup data is fetched automatically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": 'Ticker symbol (e.g., "AAPL", "7203.T")',
                },
                "start": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD)",
                },
                "end": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD)",
                },
            },
            "required": ["symbol", "start", "end"],
        },
    ),
    Tool(
        name="trend",
        description=(
            "Daily trend-following indicators for a symbol. "
            "Returns SMA(20,50,200), EMA(20,50,200), MACD(12,26,9), "
            "and ADX(14)/DMI. Uses daily bars; warmup data is fetched "
            "automatically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": 'Ticker symbol (e.g., "AAPL", "7203.T")',
                },
                "start": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD)",
                },
                "end": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD)",
                },
            },
            "required": ["symbol", "start", "end"],
        },
    ),
    Tool(
        name="volume",
        description=(
            "Daily volume moving averages for a symbol. "
            "Returns raw volume and SMA(5,10,20,50), plus a current "
            "short-interest snapshot (shares_short, pct_of_float, "
            "days_to_cover, prior_month) when available — null for "
            "tickers that don't report it (non-US equities, ETFs, crypto). "
            "Uses daily bars; warmup data is fetched automatically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": 'Ticker symbol (e.g., "AAPL", "7203.T")',
                },
                "start": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD)",
                },
                "end": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD)",
                },
            },
            "required": ["symbol", "start", "end"],
        },
    ),
]

HANDLERS = {
    "history": lambda args: history(
        args["symbol"],
        args["interval"],
        args["start"],
        args["end"],
        adjust=args.get("adjust", False),
    ),
    "oscillator": lambda args: oscillator(
        args["symbol"],
        args["start"],
        args["end"],
    ),
    "trend": lambda args: trend(
        args["symbol"],
        args["start"],
        args["end"],
    ),
    "volume": lambda args: volume(
        args["symbol"],
        args["start"],
        args["end"],
    ),
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.debug("tool=%s args=%s", name, arguments)
    handler = HANDLERS.get(name)
    if not handler:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Unknown tool: {name}"})
            )
        ]

    try:
        result = handler(arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    except (ValueError, KeyError) as e:
        logger.warning("tool=%s error: %s", name, e)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    except Exception as e:
        logger.warning("tool=%s error: %s", name, e)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
