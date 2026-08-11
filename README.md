# Ethereum Merge and Futures Basis Predictability

Studies whether the ETH futures basis (the gap between futures and spot price)
behaved differently around the September 2022 Ethereum Merge, the switch from
proof-of-work to proof-of-stake. This is a notebook-to-project conversion: the
original research notebook is preserved unchanged in
`docs/reference/ethereum-merge.ipynb`, and the analysis now runs as an
ordered Python pipeline under `src/`.

## What it does

- Loads BTC and ETH spot and quarterly-futures hourly OHLC data from local
  parquet files (Binance).
- Builds dollar bars and stitches quarterly futures contracts into a
  backward-adjusted continuous series.
- Computes the futures basis and forward-volatility features.
- Tests for structural breaks in log(1 + basis) with the BSADF (backward
  supremum augmented Dickey-Fuller) statistic, from the RiskLabAI package.
- Fits a difference-in-differences (DiD) regression around the Merge date
  (2022-09-15) and checks the parallel-trends assumption behind it.
- Runs meta-labeling and clustered feature importance on the engineered
  features.
- Writes figures and tables to `outputs/`.

## Requirements

- Python 3.13
- `uv`
- No external services or API keys are required; all inputs ship as local
  parquet files under `data/processed/`.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run ethereum-merge                 # run the full pipeline (2021-06-01 to 2025-08-31)
uv run ethereum-merge --smoke         # run over the smoke-test window from config.toml
uv run ethereum-merge --print-keys    # print the final execution context keys
uv sync --group dev && uv run pytest  # run the smoke test
```

`scripts/run_pipeline.py` is a thin wrapper around the same CLI, for example
`uv run python scripts/run_pipeline.py --smoke`.

If you need to rebuild the parquet inputs, restore the original source files
under `data/raw/` and rerun `uv run python scripts/convert_raw_to_parquet.py`.

## Configuration

`config.toml` sets the smoke-test date window
(`pipeline.smoke.start` / `pipeline.smoke.end`). The `--smoke` CLI flag reads
these values at runtime.

## Layout

- `config.toml`: smoke-test date window and other tunable pipeline settings.
- `data/raw/`: place restored source files here before rebuilding parquet inputs.
- `data/processed/`: local parquet inputs used by the pipeline.
- `docs/reference/`: original notebook, its Markdown copy-out, and the
  section-to-script map.
- `notebooks/demo.ipynb`: thin notebook wrapper that calls the package pipeline.
- `scripts/`: CLI wrapper and the one-off raw-to-parquet conversion helper.
- `src/ethereum_merge/`: package with config, pipeline runner, and CLI.
- `src/ethereum_merge/steps/`: notebook-derived Python section scripts,
  executed in order.
- `tests/`: smoke test for the pipeline.

## Output

- `outputs/figures/`: continuous futures construction, structural break
  diagnostics, causal parallel-trends check, dollar bar sampling, feature
  cluster heatmap, meta-labeling confusion matrix.
- `outputs/tables/`: descriptive statistics, DiD regression results, feature
  importance, meta-labeling report.

## License

All rights reserved. See [LICENSE](LICENSE).
