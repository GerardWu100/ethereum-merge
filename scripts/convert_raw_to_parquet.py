"""Convert the original project inputs into filtered parquet files.

This script records the one-off conversion used to build the parquet runtime
inputs under ``data/processed/``. It expects the notebook-source files to be
restored under ``data/raw/`` before execution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ethereum_merge.config import project_root

PROJECT_ROOT = project_root()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

ANALYSIS_START = pd.Timestamp("2021-06-01")
ANALYSIS_END = pd.Timestamp("2025-08-31 23:59:59")
SPOT_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def filter_analysis_window(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows inside the notebook analysis calendar."""
    return data.loc[
        (data["datetime"] >= ANALYSIS_START) & (data["datetime"] <= ANALYSIS_END)
    ].copy()


def write_filtered_spot_parquet(ticker: str) -> None:
    """Convert a raw 1-minute spot text file into a filtered parquet file."""
    source_path = RAW_DATA_DIR / f"{ticker}_full_1min.txt"
    target_path = PROCESSED_DATA_DIR / f"{ticker}_full_1min.parquet"

    data = pd.read_csv(
        source_path,
        names=SPOT_COLUMNS,
        parse_dates=["datetime"],
    )
    filtered = filter_analysis_window(data)
    filtered.to_parquet(target_path, index=False)
    print(f"Wrote {target_path.name}: {len(filtered):,} rows")


def write_filtered_futures_parquet(source_path: Path) -> None:
    """Convert one quarterly futures CSV file into a filtered parquet file."""
    target_path = PROCESSED_DATA_DIR / f"{source_path.stem}.parquet"

    data = pd.read_csv(source_path)
    data["datetime"] = pd.to_datetime(data["datetime"])
    filtered = filter_analysis_window(data)
    filtered.to_parquet(target_path, index=False)
    print(f"Wrote {target_path.name}: {len(filtered):,} rows")


def main() -> None:
    """Rebuild the project's processed parquet inputs from restored source files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Start from a clean processed directory so stale contracts do not linger.
    for existing_file in PROCESSED_DATA_DIR.iterdir():
        if existing_file.is_file():
            existing_file.unlink()

    for ticker in ["BTC", "ETH"]:
        write_filtered_spot_parquet(ticker)

    quarterly_files = sorted(
        RAW_DATA_DIR.glob("DA-16_BTC_USDT_*_binance_quarterly.csv")
    )
    quarterly_files += sorted(
        RAW_DATA_DIR.glob("DA-16_ETH_USDT_*_binance_quarterly.csv")
    )
    for source_path in quarterly_files:
        write_filtered_futures_parquet(source_path)


if __name__ == "__main__":
    main()
