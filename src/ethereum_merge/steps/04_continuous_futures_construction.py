"""Notebook section: stitch quarterly contracts into backward-adjusted series."""

import os
from datetime import timedelta


def find_roll_point(current_data, next_data, expiry_date, roll_days=14):
    """Pick the roll timestamp in the overlap window closest to the ideal pre-expiry date."""
    overlap_times = set(current_data["datetime"]) & set(next_data["datetime"])
    if not overlap_times:
        return None

    ideal_roll = expiry_date - timedelta(days=roll_days)
    overlap_sorted = sorted(overlap_times)
    candidates = [timestamp for timestamp in overlap_sorted if timestamp <= ideal_roll]
    return max(candidates) if candidates else min(overlap_sorted)


def create_continuous_futures(all_contracts_df, roll_days_before_expiry=14):
    """Build a backward-adjusted continuous futures series from quarterly contracts."""
    if all_contracts_df.empty:
        return pd.DataFrame()

    df = all_contracts_df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry"] + "-01") + pd.offsets.MonthEnd(0)
    contracts = sorted(df["expiry"].unique())

    # Segment each contract up to its roll point against the next expiry.
    segments = {}
    for index, contract in enumerate(contracts):
        data = df[df["expiry"] == contract].sort_values("datetime")
        if data.empty:
            continue

        if index < len(contracts) - 1:
            next_contract_data = df[df["expiry"] == contracts[index + 1]].sort_values(
                "datetime"
            )
            roll_time = find_roll_point(
                data,
                next_contract_data,
                data["expiry_date"].iloc[0],
                roll_days_before_expiry,
            )
            segments[contract] = (
                data[data["datetime"] <= roll_time].copy() if roll_time else data.copy()
            )
        else:
            segments[contract] = data.copy()

    # Backward-adjust older segments so prices are comparable across rolls.
    adjusted = {}
    for index in range(len(contracts) - 1, -1, -1):
        contract = contracts[index]
        if contract not in segments:
            continue

        segment = segments[contract].copy()
        if index == len(contracts) - 1:
            segment["adjusted_price"] = segment["close"]
            segment["adjustment"] = 0
        else:
            next_contract = contracts[index + 1]
            roll_time = segment["datetime"].max()

            current_close_slice = segment.loc[segment["datetime"] == roll_time, "close"]
            if next_contract in adjusted:
                next_segment = adjusted[next_contract]
                # Backward adjustment is the calendar spread at the roll instant,
                # so both legs must be priced at the same timestamp. Only when the
                # next contract has no quote at that timestamp do we fall back to
                # its first later quote.
                next_close_slice = next_segment.loc[
                    next_segment["datetime"] == roll_time,
                    "adjusted_price",
                ]
                if next_close_slice.empty:
                    next_close_slice = next_segment.loc[
                        next_segment["datetime"] > roll_time,
                        "adjusted_price",
                    ]
            else:
                next_close_slice = pd.Series(dtype=float)

            if not current_close_slice.empty and not next_close_slice.empty:
                adjustment = next_close_slice.iloc[0] - current_close_slice.iloc[0]
            else:
                # Large data gaps should not crash the stitch; leave price unadjusted.
                adjustment = 0
                print(
                    f"Warning: Gap detected at roll for {contract}. Adjustment set to 0."
                )

            segment["adjusted_price"] = segment["close"] + adjustment
            segment["adjustment"] = adjustment

        segment["contract"] = contract
        adjusted[contract] = segment

    # Drop overlapping timestamps so each datetime maps to one active contract.
    pieces = []
    previous_end = None
    for contract in contracts:
        if contract not in adjusted:
            continue

        segment = adjusted[contract]
        if previous_end is not None:
            segment = segment[segment["datetime"] > previous_end]

        if segment.empty:
            continue

        pieces.append(
            segment[["datetime", "close", "adjusted_price", "adjustment", "contract"]]
        )
        previous_end = segment["datetime"].max()

    if not pieces:
        return pd.DataFrame()

    result = (
        pd.concat(pieces, ignore_index=True)
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    return result.rename(columns={"close": "raw_price"})


def contract_roll_dates(plot_df: pd.DataFrame) -> np.ndarray:
    """Return timestamps where the active quarterly contract changes."""
    contract_changed = plot_df["contract"] != plot_df["contract"].shift(1)
    return plot_df.loc[contract_changed, "datetime"].values


def plot_continuous_futures_panel(
    ax,
    plot_df: pd.DataFrame,
    title: str,
    ylabel: str,
    roll_dates: np.ndarray,
    palette: str,
) -> None:
    """Plot one asset's adjusted futures with contract colors and roll markers."""
    colors = sns.color_palette(palette, len(plot_df["contract"].unique()))
    for index, contract in enumerate(plot_df["contract"].unique()):
        subset = plot_df[plot_df["contract"] == contract]
        ax.plot(
            subset["datetime"],
            subset["adjusted_price"],
            linewidth=2.5,
            alpha=0.9,
            label=contract,
            color=colors[index],
        )

    for index, roll_date in enumerate(roll_dates):
        label = "Contract Roll Points" if index == 0 else None
        ax.axvline(
            roll_date, color="red", linestyle=":", linewidth=2.5, alpha=0.5, label=label
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.legend(
        loc="upper left", fontsize=10, ncol=4, frameon=True, shadow=True, fancybox=True
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
    ax.tick_params(axis="both", which="major", labelsize=11)


def plot_adjustment_panel(
    ax, plot_df: pd.DataFrame, title: str, color: str, roll_dates: np.ndarray
) -> None:
    """Plot cumulative backward adjustment amounts for one asset."""
    ax.step(
        plot_df["datetime"],
        plot_df["adjustment"],
        where="post",
        linewidth=3,
        color=color,
        alpha=0.8,
    )
    ax.fill_between(
        plot_df["datetime"],
        0,
        plot_df["adjustment"],
        step="post",
        alpha=0.3,
        color=color,
    )

    for roll_date in roll_dates:
        ax.axvline(roll_date, color="black", linestyle=":", linewidth=2, alpha=0.4)

    ax.axhline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.6)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax.set_ylabel("Adjustment (USD)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
    ax.tick_params(axis="both", which="major", labelsize=11)


print("Creating continuous futures series...")
btc_continuous = create_continuous_futures(btc_all_contracts)
eth_continuous = create_continuous_futures(eth_all_contracts)

print("\nContinuous futures created")
print(
    f"  BTC: {len(btc_continuous):,} obs | "
    f"{btc_continuous['datetime'].min().date()} to {btc_continuous['datetime'].max().date()}"
)
print(
    f"  ETH: {len(eth_continuous):,} obs | "
    f"{eth_continuous['datetime'].min().date()} to {eth_continuous['datetime'].max().date()}"
)

plot_start = pd.Timestamp("2021-06-01")
plot_end = pd.Timestamp("2025-09-30")
btc_plot = btc_continuous[
    (btc_continuous["datetime"] >= plot_start)
    & (btc_continuous["datetime"] <= plot_end)
]
eth_plot = eth_continuous[
    (eth_continuous["datetime"] >= plot_start)
    & (eth_continuous["datetime"] <= plot_end)
]

fig = plt.figure(figsize=(28, 18))
grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.25, wspace=0.25)

btc_roll_dates = contract_roll_dates(btc_plot)
eth_roll_dates = contract_roll_dates(eth_plot)

plot_continuous_futures_panel(
    fig.add_subplot(grid[0, :]),
    btc_plot,
    "BTC: Backward-Adjusted Continuous Futures (Color-Coded by Contract)",
    "Futures Price (USD)",
    btc_roll_dates,
    "husl",
)
plot_continuous_futures_panel(
    fig.add_subplot(grid[1, :]),
    eth_plot,
    "ETH: Backward-Adjusted Continuous Futures (Color-Coded by Contract)",
    "Futures Price (USD)",
    eth_roll_dates,
    "husl",
)
plot_adjustment_panel(
    fig.add_subplot(grid[2, 0]),
    btc_plot,
    "BTC: Cumulative Backward Adjustment Amount",
    "#E63946",
    btc_roll_dates,
)
plot_adjustment_panel(
    fig.add_subplot(grid[2, 1]),
    eth_plot,
    "ETH: Cumulative Backward Adjustment Amount",
    "#627EEA",
    eth_roll_dates,
)

plt.tight_layout()
save_path = os.path.join(FIG_DIR, "01_continuous_futures_construction.png")
plt.savefig(save_path, bbox_inches="tight")
print(f"Saved Figure: {save_path}")
plt.show()
