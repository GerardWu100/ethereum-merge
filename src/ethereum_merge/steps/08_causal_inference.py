"""Notebook section: visualize parallel-trends assumption for causal inference."""

import os


def plot_parallel_trends(btc_data, eth_data, merge_date, window_days=90, save_path=None):
    """Plot normalized pre/post volatility paths to inspect parallel trends."""
    start_date = merge_date - pd.Timedelta(days=window_days)
    end_date = merge_date + pd.Timedelta(days=window_days)

    # Daily averages make the pre/post divergence easier to read than hourly noise.
    btc_trend = btc_data.set_index('datetime')['forward_vol_24h'].resample('D').mean()
    eth_trend = eth_data.set_index('datetime')['forward_vol_24h'].resample('D').mean()

    trend_frame = pd.concat([btc_trend, eth_trend], axis=1).dropna()
    trend_frame.columns = ['BTC_Vol', 'ETH_Vol']
    trend_frame = trend_frame[(trend_frame.index >= start_date) & (trend_frame.index <= end_date)]

    # Re-base both series to 100 at the window start for shape comparison.
    trend_frame['BTC_Indexed'] = (trend_frame['BTC_Vol'] / trend_frame['BTC_Vol'].iloc[0]) * 100
    trend_frame['ETH_Indexed'] = (trend_frame['ETH_Vol'] / trend_frame['ETH_Vol'].iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(14, 7))

    pre_merge = trend_frame[trend_frame.index < merge_date]
    post_merge = trend_frame[trend_frame.index >= merge_date]

    ax.plot(pre_merge.index, pre_merge['BTC_Indexed'], color='#F7931A', linestyle='-', linewidth=2, label='BTC (Control)')
    ax.plot(pre_merge.index, pre_merge['ETH_Indexed'], color='#627EEA', linestyle='-', linewidth=2, label='ETH (Treatment)')
    ax.plot(post_merge.index, post_merge['BTC_Indexed'], color='#F7931A', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(post_merge.index, post_merge['ETH_Indexed'], color='#627EEA', linestyle='--', linewidth=2, alpha=0.7)

    ax.axvline(merge_date, color='black', linestyle=':', linewidth=2, label='The Merge (Intervention)')
    ax.axvspan(start_date, merge_date, color='green', alpha=0.05, label='Pre-Treatment (Parallel Trends)')
    ax.axvspan(merge_date, end_date, color='red', alpha=0.05, label='Post-Treatment')

    ax.set_title('Causal Inference: Parallel Trends Visualization', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized Volatility (Start = 100)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved Figure: {save_path}")

    plt.show()


print("Visualizing Causal Inference assumptions...")
plot_parallel_trends(
    btc_merged,
    eth_merged,
    MERGE_DATE,
    save_path=os.path.join(FIG_DIR, '03_causal_parallel_trends.png'),
)
