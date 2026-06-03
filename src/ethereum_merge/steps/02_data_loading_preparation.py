"""Notebook section: load spot and quarterly futures parquet inputs."""

import glob
import os

import pandas as pd

CRYPTO_DIR = str(PROCESSED_DATA_DIR)


def smoke_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the inclusive smoke-test window configured by the pipeline runner."""
    smoke_start = pd.Timestamp(OVERRIDES.get('smoke_start', '2022-09-01'))
    smoke_end = pd.Timestamp(OVERRIDES.get('smoke_end', '2022-10-15 23:59:59'))
    return smoke_start, smoke_end


def apply_smoke_window(df: pd.DataFrame, datetime_col: str | None = None) -> pd.DataFrame:
    """Trim a dataframe to the smoke window when ``SMOKE_TEST_MODE`` is active."""
    if not SMOKE_TEST_MODE:
        return df

    smoke_start, smoke_end = smoke_date_bounds()
    if datetime_col is None:
        # Spot files are indexed by timestamp after loading.
        return df.loc[(df.index >= smoke_start) & (df.index <= smoke_end)].copy()

    return df.loc[(df[datetime_col] >= smoke_start) & (df[datetime_col] <= smoke_end)].copy()


def load_frd_spot(ticker: str, directory: str) -> pd.DataFrame:
    """Load one-minute spot prices from a processed parquet file."""
    file_path = os.path.join(directory, f'{ticker}_full_1min.parquet')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    print(f"Loading {ticker} Spot from {file_path}...")
    df = pd.read_parquet(file_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    return apply_smoke_window(df)


def load_futures_contracts(file_pattern: str) -> pd.DataFrame:
    """Load and concatenate quarterly futures contracts matched by glob pattern."""
    contracts_list = []
    for file in sorted(glob.glob(file_pattern)):
        df = pd.read_parquet(file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = apply_smoke_window(df, datetime_col='datetime')

        # Expiry token is embedded in the filename after the hourly marker.
        expiry = file.split('_hourly_')[1].split('_')[1]
        df['expiry'] = expiry
        contracts_list.append(df)

    if not contracts_list:
        return pd.DataFrame()

    return pd.concat(contracts_list, ignore_index=True)


if os.path.exists(CRYPTO_DIR):
    print(f"Crypto Directory Found: {CRYPTO_DIR}")
else:
    print(f"WARNING: Could not find crypto directory at: {CRYPTO_DIR}")

btc_spot_hf = load_frd_spot('BTC', CRYPTO_DIR)
eth_spot_hf = load_frd_spot('ETH', CRYPTO_DIR)
print(f"BTC Spot Data: {len(btc_spot_hf):,} rows")
print(f"ETH Spot Data: {len(eth_spot_hf):,} rows")

print("Loading Futures Data...")
btc_futures_pattern = os.path.join(CRYPTO_DIR, 'DA-16_BTC_USDT_*_binance_quarterly.parquet')
eth_futures_pattern = os.path.join(CRYPTO_DIR, 'DA-16_ETH_USDT_*_binance_quarterly.parquet')

btc_all_contracts = load_futures_contracts(btc_futures_pattern)
eth_all_contracts = load_futures_contracts(eth_futures_pattern)
print(f"BTC Futures Loaded: {len(btc_all_contracts):,} rows")
print(f"ETH Futures Loaded: {len(eth_all_contracts):,} rows")
