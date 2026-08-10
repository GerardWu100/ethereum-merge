"""Notebook section: meta-labeling and clustered feature importance."""

import os

import numpy as np
import pandas as pd
from IPython.display import display
from RiskLabAI.cluster.clustering import cluster_k_means_top
from RiskLabAI.data.labeling import labeling as rl_labeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    log_loss,
)


def ewm_volatility(series: pd.Series, span: int = 48) -> pd.Series:
    """Estimate dynamic barrier width from exponentially weighted return volatility."""
    returns = np.log(series).diff().dropna()
    return returns.ewm(span=span).std().dropna()


def build_meta_dataset(
    asset_df: pd.DataFrame,
    asset_label: str,
    pooled_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build one asset's meta-labeling dataset using CUSUM events and triple barriers."""
    asset_df = asset_df.sort_values("datetime").drop_duplicates(
        subset="datetime", keep="last"
    )

    price_series = asset_df.set_index("datetime")["raw_price"]
    basis_series = asset_df.set_index("datetime")["basis"].dropna()
    if basis_series.empty:
        return pd.DataFrame()

    # CUSUM filter finds basis shocks large enough to treat as tradeable events.
    cusum_threshold = basis_series.std() * 0.5
    raw_events = rl_labeling.symmetric_cusum_filter(
        basis_series, threshold=cusum_threshold
    )

    upper, lower = basis_series.quantile(0.80), basis_series.quantile(0.20)
    side_series = pd.Series(index=basis_series.index, dtype=float)
    side_series[basis_series >= upper] = 1.0
    side_series[basis_series <= lower] = -1.0
    side_series = side_series.dropna()

    event_index = raw_events.intersection(side_series.index)
    if len(event_index) == 0:
        return pd.DataFrame()

    ewm_vol = ewm_volatility(price_series)
    vertical_times = rl_labeling.vertical_barrier(
        price_series, event_index, number_days=3
    )
    events = rl_labeling.meta_events(
        close=price_series,
        time_events=event_index,
        ptsl=[1.5, 1.5],
        target=ewm_vol,
        return_min=ewm_vol.quantile(0.3),
        vertical_barrier_times=vertical_times,
        side=side_series,
        num_threads=1,
    )

    meta_labels = rl_labeling.meta_labeling(events, price_series).dropna(
        subset=["Label"]
    )
    if meta_labels.empty:
        return pd.DataFrame()

    meta_labels = meta_labels.reset_index().rename(columns={"index": "datetime"})
    meta_labels["asset"] = asset_label

    pooled_reset = pooled_features.reset_index()
    meta_joined = meta_labels.merge(
        pooled_reset[["datetime", "asset"] + feature_cols],
        on=["datetime", "asset"],
        how="inner",
    )
    return meta_joined.set_index(["datetime", "asset"]).dropna(subset=feature_cols)


def plot_cluster_heatmap(
    feature_frame: pd.DataFrame, cluster_dict: dict[str, list[str]]
) -> None:
    """Plot the feature correlation matrix sorted by k-means cluster blocks."""
    correlation = feature_frame.corr()
    sort_order = []
    for features in cluster_dict.values():
        sort_order.extend(features)

    correlation_sorted = correlation.loc[sort_order, sort_order]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_sorted,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Feature Correlation Matrix (Sorted by Clusters)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    path = os.path.join(FIG_DIR, "05_feature_cluster_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved Figure: {path}")
    plt.show()


def save_classification_performance(
    model, feature_frame, labels, output_dir: str
) -> None:
    """Persist confusion matrix figure and sklearn classification report."""
    predictions = model.predict(feature_frame)

    confusion = confusion_matrix(labels, predictions, labels=model.classes_)
    display_obj = ConfusionMatrixDisplay(
        confusion_matrix=confusion, display_labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    display_obj.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Meta-Labeling Confusion Matrix", fontsize=14, fontweight="bold")

    confusion_path = os.path.join(output_dir, "06_meta_labeling_confusion_matrix.png")
    plt.savefig(confusion_path, bbox_inches="tight")
    print(f"Saved Figure: {confusion_path}")
    plt.show()

    report_path = os.path.join(TABLES_DIR, "04_meta_labeling_report.txt")
    with open(report_path, "w") as file_handle:
        file_handle.write(classification_report(labels, predictions))
    print(f"Saved Report: {report_path}")


# DiD interaction terms become model features for meta-labeling and importance tests.
pooled_features = pooled_data_clean.copy()
pooled_features["basis_PostMerge"] = (
    pooled_features["basis"] * pooled_features["PostMerge"]
)
pooled_features["basis_Ethereum"] = (
    pooled_features["basis"] * pooled_features["Ethereum"]
)
pooled_features["PostMerge_Ethereum"] = (
    pooled_features["PostMerge"] * pooled_features["Ethereum"]
)
pooled_features["basis_PostMerge_Ethereum"] = (
    pooled_features["basis"]
    * pooled_features["PostMerge"]
    * pooled_features["Ethereum"]
)
pooled_features = pooled_features.set_index(["datetime", "asset"])

feature_cols = [
    "basis",
    "PostMerge",
    "Ethereum",
    "basis_PostMerge",
    "basis_Ethereum",
    "PostMerge_Ethereum",
    "basis_PostMerge_Ethereum",
]

print("Building Meta-Labeling Dataset...")
meta_eth = build_meta_dataset(eth_merged.copy(), "ETH", pooled_features, feature_cols)
meta_btc = build_meta_dataset(btc_merged.copy(), "BTC", pooled_features, feature_cols)
meta_dataset = pd.concat([meta_eth, meta_btc]).sort_index().reset_index()
meta_dataset["Label"] = meta_dataset["Label"].astype(int)
print(f"Meta-Labeled Dataset: {len(meta_dataset)} events")

print("\nCalculating Clustered Feature Importance (MDA)...")
feature_frame = meta_dataset[feature_cols]
labels = meta_dataset["Label"]

correlation_matrix = feature_frame.corr().fillna(0)
_, clusters_dict, _ = cluster_k_means_top(
    correlation_matrix,
    max_clusters=4,
    iterations=20,
    random_state=42,
)
cluster_config = {
    f"Cluster_{index}": members for index, members in clusters_dict.items()
}

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight="balanced_subsample",
    random_state=42,
)
model.fit(feature_frame, labels)
baseline_loss = log_loss(
    labels, model.predict_proba(feature_frame), labels=model.classes_
)

importances = {}
# Seeded generator so repeated runs on the same data give the same importances,
# matching the fixed random_state used by the clustering and the forest above.
shuffle_generator = np.random.default_rng(42)
for cluster_name, features in cluster_config.items():
    shuffled_frame = feature_frame.copy()
    shuffled_values = shuffled_frame[features].to_numpy(copy=True)
    # Rows are permuted together, so the cluster is broken as a block while the
    # correlation between its own columns is preserved.
    shuffle_generator.shuffle(shuffled_values)
    shuffled_frame[features] = shuffled_values

    shuffled_loss = log_loss(
        labels, model.predict_proba(shuffled_frame), labels=model.classes_
    )
    importances[cluster_name] = shuffled_loss - baseline_loss

importance_frame = (
    pd.Series(importances)
    .sort_values(ascending=False)
    .to_frame("Mean Decrease Accuracy")
)
print("\n--- Feature Cluster Importance ---")
print(
    "High positive values indicate the cluster contains significant predictive signal."
)
display(importance_frame)
print(f"\nCluster Composition: {cluster_config}")

importance_table = importance_frame.rename(
    columns={"Mean Decrease Accuracy": "Importance (MDA)"}
)
importance_table.index.name = "Feature Cluster"
table3_path = os.path.join(TAB_DIR, "03_feature_importance.csv")
importance_table.to_csv(table3_path)
print(f"Saved Table 3 (Feature Importance): {table3_path}")
display(importance_table)

plot_cluster_heatmap(feature_frame, cluster_config)
save_classification_performance(model, feature_frame, labels, FIG_DIR)
