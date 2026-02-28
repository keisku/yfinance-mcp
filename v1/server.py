"""Yahoo Finance MCP Server."""

import json

from history import VALID_INTERVALS, history
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from oscillator import oscillator

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
            "Returns RSI(14), Stochastic %K(14)/%D(3), MACD(12,26,9), "
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
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    try:
        result = handler(arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    except (ValueError, KeyError) as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
