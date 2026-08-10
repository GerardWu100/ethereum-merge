"""Notebook section: structural break diagnostics and DiD regression."""

import os

import statsmodels.formula.api as smf
from IPython.display import display
from RiskLabAI.features.structural_breaks import get_bsadf_statistic


def basis_log_series(df: pd.DataFrame, freq: str = "1D") -> pd.Series:
    """Return log(1 + basis) sampled at the requested frequency."""
    resampled_basis = (
        df.set_index("datetime")["basis"]
        .resample(freq)
        .mean()
        .replace([-np.inf, np.inf], np.nan)
        .dropna()
    )
    return np.log1p(resampled_basis)


def compute_bsadf_path(
    log_series: pd.Series, min_window: int = 120, step: int = 5
) -> pd.Series:
    """Compute BSADF statistics on an expanding window grid."""
    stats = []
    timestamps = []

    for end in range(min_window, len(log_series) + 1, step):
        window_series = log_series.iloc[:end]
        result = get_bsadf_statistic(
            log_price=window_series,
            min_sample_length=min_window,
            constant="ct",
            lags=1,
        )
        stats.append(result["bsadf"])
        timestamps.append(window_series.index[-1])

    return pd.Series(stats, index=pd.DatetimeIndex(timestamps), name="bsadf")


print("Running SADF Structural Break Tests...")
eth_log_basis = basis_log_series(eth_merged)
btc_log_basis = basis_log_series(btc_merged)

min_window = 180
step = 5
eth_bsadf = compute_bsadf_path(eth_log_basis, min_window=min_window, step=step)
btc_bsadf = compute_bsadf_path(btc_log_basis, min_window=min_window, step=step)

# Pre-merge 95th percentile acts as a visual critical-value reference line.
# Named apart from the dollar-bar thresholds of step 03, which share this namespace.
eth_bsadf_threshold = eth_bsadf[eth_bsadf.index < MERGE_DATE].quantile(0.95)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    eth_bsadf.index,
    eth_bsadf.values,
    label="ETH basis BSADF",
    color="#627EEA",
    linewidth=2.2,
)
ax.plot(
    btc_bsadf.index,
    btc_bsadf.values,
    label="BTC basis BSADF",
    color="#F7931A",
    linewidth=2.0,
    alpha=0.8,
)
ax.axvline(
    MERGE_DATE,
    color="black",
    linestyle="--",
    linewidth=2.5,
    label="Merge (Sep 15, 2022)",
)
ax.axhline(
    eth_bsadf_threshold,
    color="#627EEA",
    linestyle=":",
    linewidth=2,
    label="ETH 95% Critical Value",
)
ax.set_title(
    "SADF / BSADF Structural Break Diagnostics", fontsize=16, fontweight="bold"
)
ax.legend(loc="upper left", fontsize=11)

save_path = os.path.join(FIG_DIR, "02_sadf_structural_break.png")
plt.savefig(save_path, bbox_inches="tight")
print(f"Saved Figure: {save_path}")
plt.show()

# Difference-in-differences model: triple interaction captures merge-specific basis effect.
formula = (
    "forward_vol_24h ~ basis + PostMerge + Ethereum + "
    "basis:PostMerge + basis:Ethereum + PostMerge:Ethereum + "
    "basis:PostMerge:Ethereum"
)

print("Running DiD Regression...")
model = smf.ols(formula=formula, data=pooled_data_clean).fit(
    cov_type="HAC", cov_kwds={"maxlags": 24}
)

results_summary = pd.DataFrame(
    {
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-Statistic": model.tvalues,
        "p-Value": model.pvalues,
    }
).round(4)
results_summary["Sig"] = results_summary["p-Value"].apply(
    lambda p_value: (
        "***"
        if p_value < 0.01
        else ("**" if p_value < 0.05 else ("*" if p_value < 0.1 else ""))
    )
)

table2_path = os.path.join(TAB_DIR, "02_did_regression_results.csv")
results_summary.to_csv(table2_path)
with open(os.path.join(TAB_DIR, "02_did_regression_full.txt"), "w") as file_handle:
    file_handle.write(model.summary().as_text())

print(f"Saved Table 2 (Regression Results): {table2_path}")
display(results_summary)
