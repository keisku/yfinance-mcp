#!/usr/bin/env python3
"""Fetch real market data and save as fixture files for benchmarks.

This script should be run occasionally to update benchmark fixture data.
The generated files are committed to the repo for deterministic benchmarks.

Usage:
    python benchmarks/fetch_fixture_data.py
"""

import json
from pathlib import Path

import yfinance as yf

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SYMBOLS = [
    "AAPL",  # NYSE (US)
    "MSFT",  # NASDAQ (US) - more reliable than international symbols
    "GOOGL",  # NASDAQ (US)
]

INTERVALS = ["1d", "1wk", "1mo"]


def main():
    FIXTURES_DIR.mkdir(exist_ok=True)

    print("Fetching real market data from Yahoo Finance...")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Intervals: {', '.join(INTERVALS)}")
    print(f"Period: 2 years\n")

    for symbol in SYMBOLS:
        print(f"Fetching {symbol}...")
        ticker = yf.Ticker(symbol)

        for interval in INTERVALS:
            try:
                print(f"  {interval}...", end=" ")
                df = ticker.history(period="2y", interval=interval)

                if df.empty:
                    print("EMPTY")
                    continue

                # Normalize to cache format
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Convert to records format for JSON storage
                records = []
                for date, row in df.iterrows():
                    records.append(
                        {
                            "date": date.isoformat(),
                            "o": float(row.get("Open", 0)),
                            "h": float(row.get("High", 0)),
                            "l": float(row.get("Low", 0)),
                            "c": float(row.get("Close", 0)),
                            "v": int(row.get("Volume", 0)),
                        }
                    )

                # Save to JSON file
                fixture_file = FIXTURES_DIR / f"{symbol}_{interval}.json"
                with open(fixture_file, "w") as f:
                    json.dump(records, f, indent=2)

                print(f"✓ ({len(records)} bars)")

            except Exception as e:
                print(f"ERROR: {e}")

    print(f"\n✓ Fixture data saved to {FIXTURES_DIR}/")
    print("\nThese files are committed to the repo for deterministic benchmarks.")


if __name__ == "__main__":
    main()
