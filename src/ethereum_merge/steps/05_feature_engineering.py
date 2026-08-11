"""Notebook section: basis, forward volatility, and descriptive statistics."""

import os

from IPython.display import display


def calculate_rolling_rv_vectorized(hf_data, target_timestamps, window_hours=24):
    """Compute forward annualized realized volatility aligned to hourly targets."""
    # Resample to a strict 1-minute grid so rolling windows count minutes, not rows.
    df = hf_data[~hf_data.index.duplicated(keep="last")].copy()
    resampled_close = df["close"].resample("1min").ffill()

    log_return = np.log(resampled_close / resampled_close.shift(1))
    squared_returns = log_return**2

    minutes = window_hours * 60
    rolling_variance = squared_returns.rolling(window=minutes).sum()
    forward_variance = rolling_variance.shift(-minutes)

    annualization_factor = np.sqrt(525600 / minutes)
    forward_realized_vol = np.sqrt(forward_variance) * annualization_factor

    aligned_rv = forward_realized_vol.reindex(
        target_timestamps,
        method="nearest",
        tolerance=pd.Timedelta("5min"),
    )
    return aligned_rv.values


def add_asset_metadata(
    merged_df: pd.DataFrame, asset_label: str, is_ethereum: int
) -> pd.DataFrame:
    """Attach DiD treatment flags and asset label used in pooled regressions."""
    merged_df["PostMerge"] = (merged_df["datetime"] >= MERGE_DATE).astype(int)
    merged_df["Ethereum"] = is_ethereum
    merged_df["asset"] = asset_label
    return merged_df


def summary_stats_row(df: pd.DataFrame, label: str) -> pd.Series:
    """Build one descriptive-statistics row for Table 1."""
    return pd.Series(
        {
            "Obs": len(df),
            "Mean Basis (%)": df["basis"].mean() * 100,
            "Std Basis (%)": df["basis"].std() * 100,
            "Mean Vol (Ann %)": df["forward_vol_24h"].mean() * 100,
            "Std Vol (Ann %)": df["forward_vol_24h"].std() * 100,
        },
        name=label,
    )


# Hourly spot anchors are joined to hourly continuous futures timestamps.
btc_spot_hourly = (
    btc_spot_hf["close"].resample("h").last().dropna().rename("spot_price")
)
eth_spot_hourly = (
    eth_spot_hf["close"].resample("h").last().dropna().rename("spot_price")
)

btc_merged = btc_continuous.merge(btc_spot_hourly, on="datetime", how="inner")
eth_merged = eth_continuous.merge(eth_spot_hourly, on="datetime", how="inner")

# Basis uses raw contract prices, not backward-adjusted stitch prices.
btc_merged["basis"] = (btc_merged["raw_price"] - btc_merged["spot_price"]) / btc_merged[
    "spot_price"
]
eth_merged["basis"] = (eth_merged["raw_price"] - eth_merged["spot_price"]) / eth_merged[
    "spot_price"
]
print(
    "BTC Basis stats: "
    f"Mean={btc_merged['basis'].mean():.4f}, "
    f"Min={btc_merged['basis'].min():.4f}, "
    f"Max={btc_merged['basis'].max():.4f}"
)

btc_merged["forward_vol_24h"] = calculate_rolling_rv_vectorized(
    btc_spot_hf, btc_merged["datetime"], 24
)
btc_merged["forward_vol_168h"] = calculate_rolling_rv_vectorized(
    btc_spot_hf, btc_merged["datetime"], 168
)
eth_merged["forward_vol_24h"] = calculate_rolling_rv_vectorized(
    eth_spot_hf, eth_merged["datetime"], 24
)
eth_merged["forward_vol_168h"] = calculate_rolling_rv_vectorized(
    eth_spot_hf, eth_merged["datetime"], 168
)

btc_merged = add_asset_metadata(btc_merged, "BTC", is_ethereum=0)
eth_merged = add_asset_metadata(eth_merged, "ETH", is_ethereum=1)

pooled_data = pd.concat([btc_merged, eth_merged], ignore_index=True)
pooled_data_clean = pooled_data.dropna(subset=["forward_vol_24h", "forward_vol_168h"])

pre_merge = pooled_data_clean[pooled_data_clean["PostMerge"] == 0]
post_merge = pooled_data_clean[pooled_data_clean["PostMerge"] == 1]

table1 = pd.concat(
    [
        summary_stats_row(pre_merge[pre_merge["asset"] == "BTC"], "BTC (Pre-Merge)"),
        summary_stats_row(post_merge[post_merge["asset"] == "BTC"], "BTC (Post-Merge)"),
        summary_stats_row(pre_merge[pre_merge["asset"] == "ETH"], "ETH (Pre-Merge)"),
        summary_stats_row(post_merge[post_merge["asset"] == "ETH"], "ETH (Post-Merge)"),
    ],
    axis=1,
).T.round(4)

table1_path = os.path.join(TAB_DIR, "01_descriptive_statistics.csv")
table1.to_csv(table1_path)
print(f"Saved Table 1 (Descriptive Statistics): {table1_path}")
display(table1)
