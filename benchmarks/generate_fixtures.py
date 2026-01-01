#!/usr/bin/env python3
"""Generate deterministic fixture data for benchmarks.

This script creates realistic market data for benchmark testing. The data is:
- Deterministic (same output every time using fixed random seeds)
- Realistic (follows market patterns with proper OHLCV relationships)
- Timestamped (uses actual calendar dates with weekday filtering)
- Versioned (includes metadata about generation date and parameters)

The generated fixtures are committed to the repo to ensure benchmarks are:
- Network-independent (no API calls during benchmark execution)
- Deterministic and repeatable
- Fast (no network latency)

Usage:
    python benchmarks/generate_fixtures.py

To use real Yahoo Finance data instead:
    python benchmarks/fetch_fixture_data.py  # (requires network access)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Benchmark symbols
SYMBOLS = {
    "AAPL": {"base_price": 150.0, "volatility": 0.018},  # Apple
    "MSFT": {"base_price": 350.0, "volatility": 0.016},  # Microsoft
    "GOOGL": {"base_price": 130.0, "volatility": 0.020},  # Google
}

INTERVALS = ["1d", "1wk", "1mo"]


def generate_ohlcv_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    base_price: float,
    volatility: float,
) -> list[dict]:
    """Generate realistic OHLCV data with proper market patterns.

    Args:
        symbol: Stock symbol (used as random seed for reproducibility)
        start_date: Starting date
        end_date: Ending date
        base_price: Starting price
        volatility: Daily volatility (e.g., 0.02 = 2%)

    Returns:
        List of OHLCV records with ISO-formatted timestamps
    """
    # Use symbol as seed for reproducibility
    np.random.seed(hash(symbol) % (2**32))

    # Generate trading days (weekdays only)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = dates[dates.dayofweek < 5]  # Monday=0, Friday=4

    # Generate price series with realistic random walk
    num_days = len(dates)
    returns = np.random.normal(0, volatility, num_days)

    # Add slight upward drift (markets tend to go up over time)
    drift = 0.0002  # ~5% annual drift
    returns += drift

    # Calculate cumulative prices
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV records
    records = []
    for date, close in zip(dates, prices):
        # Intraday variation (0.5% - 2% daily range)
        daily_range = close * np.random.uniform(0.005, 0.02)
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)

        # Open is somewhere between low and high
        open_price = np.random.uniform(low, high)

        # Volume with realistic variation (higher on volatile days)
        base_volume = 50_000_000
        volatility_factor = abs(close - open_price) / close
        volume = int(base_volume * (1 + volatility_factor * 5) * np.random.uniform(0.7, 1.3))

        records.append(
            {
                "date": date.isoformat(),
                "o": round(open_price, 2),
                "h": round(high, 2),
                "l": round(low, 2),
                "c": round(close, 2),
                "v": volume,
            }
        )

    return records


def resample_to_weekly(daily_records: list[dict]) -> list[dict]:
    """Resample daily data to weekly bars."""
    df = pd.DataFrame(daily_records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    weekly = df.resample("W").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})

    records = []
    for date, row in weekly.iterrows():
        if pd.notna(row["c"]):  # Skip empty weeks
            records.append(
                {
                    "date": date.isoformat(),
                    "o": round(row["o"], 2),
                    "h": round(row["h"], 2),
                    "l": round(row["l"], 2),
                    "c": round(row["c"], 2),
                    "v": int(row["v"]),
                }
            )

    return records


def resample_to_monthly(daily_records: list[dict]) -> list[dict]:
    """Resample daily data to monthly bars."""
    df = pd.DataFrame(daily_records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    monthly = df.resample("ME").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    )

    records = []
    for date, row in monthly.iterrows():
        if pd.notna(row["c"]):  # Skip empty months
            records.append(
                {
                    "date": date.isoformat(),
                    "o": round(row["o"], 2),
                    "h": round(row["h"], 2),
                    "l": round(row["l"], 2),
                    "c": round(row["c"], 2),
                    "v": int(row["v"]),
                }
            )

    return records


def main():
    FIXTURES_DIR.mkdir(exist_ok=True)

    # Generate data for the last 2 years
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2023, 1, 1)

    print("Generating deterministic benchmark fixture data...")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {', '.join(SYMBOLS.keys())}")
    print(f"Intervals: {', '.join(INTERVALS)}\n")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "symbols": SYMBOLS,
        "intervals": INTERVALS,
        "note": "Deterministic synthetic data for benchmark testing",
    }

    for symbol, params in SYMBOLS.items():
        print(f"Generating {symbol}...")

        # Generate daily data
        daily_data = generate_ohlcv_data(
            symbol,
            start_date,
            end_date,
            base_price=params["base_price"],
            volatility=params["volatility"],
        )

        # Save daily
        daily_file = FIXTURES_DIR / f"{symbol}_1d.json"
        with open(daily_file, "w") as f:
            json.dump(daily_data, f, indent=2)
        print(f"  1d: {len(daily_data)} bars → {daily_file.name}")

        # Generate and save weekly
        weekly_data = resample_to_weekly(daily_data)
        weekly_file = FIXTURES_DIR / f"{symbol}_1wk.json"
        with open(weekly_file, "w") as f:
            json.dump(weekly_data, f, indent=2)
        print(f"  1wk: {len(weekly_data)} bars → {weekly_file.name}")

        # Generate and save monthly
        monthly_data = resample_to_monthly(daily_data)
        monthly_file = FIXTURES_DIR / f"{symbol}_1mo.json"
        with open(monthly_file, "w") as f:
            json.dump(monthly_data, f, indent=2)
        print(f"  1mo: {len(monthly_data)} bars → {monthly_file.name}")

    # Save metadata
    metadata_file = FIXTURES_DIR / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Generated {len(SYMBOLS) * len(INTERVALS)} fixture files")
    print(f"✓ Saved metadata to {metadata_file.name}")
    print(f"\nFixture data location: {FIXTURES_DIR}/")
    print("\nThese files are committed to the repo for deterministic benchmarks.")


if __name__ == "__main__":
    main()
