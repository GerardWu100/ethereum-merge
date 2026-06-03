# Ethereum Merge and Futures Basis Predictability

Splits the Merge notebook into section scripts and relocates the crypto inputs into local parquet assets for reproducible execution.

The original notebook is preserved unchanged in `docs/reference/ethereum-merge.ipynb`. The new execution notebook in `notebooks/demo.ipynb` only calls the Python backend under `src/`.

## Layout

- `config.toml`: smoke-test date window and other tunable pipeline settings
- `docs/reference/`: original notebook, pasted notebook content, and split map
- `docs/reference/notebook_reference.md`: full notebook content copied into Markdown
- `src/ethereum_merge/`: package with config, pipeline runner, and CLI
- `src/ethereum_merge/steps/`: notebook-derived Python section scripts
- `scripts/run_pipeline.py`: thin CLI wrapper
- `scripts/convert_raw_to_parquet.py`: one-off helper used to rebuild the parquet files from the original source inputs
- `notebooks/demo.ipynb`: thin notebook wrapper
- `data/raw/`: place restored source files before running the conversion script
- `data/processed/`: local parquet inputs used by the pipeline
- `outputs/`: generated figures and tables
- `tests/`: smoke verification for the pipeline
- `docs/reference/notebook_split.md`: section-to-script map

## Run

```bash
uv sync
uv run ethereum-merge
uv run ethereum-merge --smoke
uv run python scripts/run_pipeline.py --smoke
uv sync --group dev && uv run pytest
```

If you ever need to rebuild the parquet files, restore the original crypto source files under `data/raw/` and rerun `uv run python scripts/convert_raw_to_parquet.py`.
