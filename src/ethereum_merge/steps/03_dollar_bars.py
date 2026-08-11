"""Notebook section: build dollar bars and visualize sampling density."""

import os

import pandas as pd
from IPython.display import display
from RiskLabAI.controller.data_structure_controller import Controller

dollar_bar_controller = Controller()


def prepare_tick_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the datetime/price/volume frame expected by RiskLabAI bar builders."""
    tick_df = df.reset_index()
    tick_df.columns = tick_df.columns.str.lower()

    # RiskLabAI expects canonical column names regardless of upstream casing.
    tick_df = tick_df.rename(
        columns={
            "datetime": "date_time",
            "index": "date_time",
            "close": "price",
            "volume": "volume",
        }
    )

    required = ["date_time", "price", "volume"]
    missing = [column for column in required if column not in tick_df.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing}. Available: {tick_df.columns.tolist()}"
        )

    return tick_df[required].sort_values("date_time").dropna()


def build_dollar_bars(
    spot_df: pd.DataFrame,
    asset_label: str,
    quantile: float = 0.80,
) -> tuple[pd.DataFrame, float]:
    """Construct dollar bars for one asset using a turnover-quantile threshold."""
    tick_df = prepare_tick_frame(spot_df)

    # Threshold is a high-turnover minute proxy, not a fixed dollar amount.
    turnover = tick_df["price"] * tick_df["volume"]
    threshold = float(turnover.quantile(quantile))

    bars = dollar_bar_controller.handle_input_command(
        method_name="dollar_standard_bars",
        method_arguments={"threshold": threshold},
        input_data=tick_df,
    ).copy()

    # Normalize column names across RiskLabAI versions.
    bars = bars.rename(
        columns={column: column.lower().replace(" ", "_") for column in bars.columns}
    )
    if "date_time" in bars.columns:
        bars["datetime"] = pd.to_datetime(bars["date_time"])

    bars["asset"] = asset_label
    return bars, threshold


def plot_dollar_bar_sampling(
    raw_df: pd.DataFrame,
    dollar_df: pd.DataFrame,
    asset_name: str,
    threshold: float,
    start_date: str = "2022-09-01",
    end_date: str = "2022-09-30",
) -> None:
    """Compare 1-minute time bars with dollar-bar sample points over one window."""
    mask_raw = (raw_df.index >= start_date) & (raw_df.index <= end_date)
    mask_dollar = (dollar_df["datetime"] >= start_date) & (
        dollar_df["datetime"] <= end_date
    )

    subset_raw = raw_df.loc[mask_raw]
    subset_dollar = dollar_df.loc[mask_dollar]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        subset_raw.index,
        subset_raw["close"],
        color="gray",
        alpha=0.4,
        label="1-Min Time Bars",
        linewidth=1,
    )
    ax.scatter(
        subset_dollar["datetime"],
        subset_dollar["close"],
        color="#E63946",
        s=1,
        alpha=0.9,
        label=f"Dollar Bar Samples (Thresh={threshold:,.0f})",
    )

    ax.set_title(
        f"{asset_name}: Dollar Bar Sampling Density (Merge Period)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, f"04_{asset_name}_dollar_bar_sampling.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved Figure: {path}")
    plt.show()


print("Generating Dollar Bars (this may take a moment)...")
btc_dollar_bars, btc_threshold = build_dollar_bars(btc_spot_hf, "BTC")
eth_dollar_bars, eth_threshold = build_dollar_bars(eth_spot_hf, "ETH")
print(
    f"Generated {len(btc_dollar_bars):,} BTC bars and {len(eth_dollar_bars):,} ETH bars."
)
display(eth_dollar_bars.head())

# Focus on the merge window where activity spikes are most visible.
plot_dollar_bar_sampling(
    eth_spot_hf,
    eth_dollar_bars,
    "ETH",
    threshold=eth_threshold,
    start_date="2022-09-10",
    end_date="2022-09-20",
)
