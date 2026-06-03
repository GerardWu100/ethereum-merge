# Project Overview

## File Tree

```text
ethereum-merge/
├── GUIDE_OVERVIEW.md
├── GUIDE_ROOT.md
├── README.md
├── config.toml
├── docs/
│   └── reference/
│       ├── GUIDE_reference.md
│       ├── ethereum-merge.ipynb
│       ├── notebook_reference.md
│       └── notebook_split.md
├── notebooks/
│   ├── GUIDE_notebooks.md
│   └── demo.ipynb
├── scripts/
│   ├── GUIDE_scripts.md
│   ├── convert_raw_to_parquet.py
│   └── run_pipeline.py
├── src/
│   ├── GUIDE_src.md
│   └── ethereum_merge/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── pipeline.py
│       └── steps/
├── tests/
│   └── test_smoke_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── logs/
└── outputs/
    ├── figures/
    └── tables/
```

## Purpose

Splits the Merge notebook into section scripts and relocates the crypto inputs into local parquet assets for reproducible execution.

## Flow

1. `uv run ethereum-merge`, `scripts/run_pipeline.py`, or `notebooks/demo.ipynb` calls the package pipeline.
2. `scripts/convert_raw_to_parquet.py` documents the one-off conversion used to rebuild the runtime parquet files from the original source files.
3. The pipeline builds a shared execution context with project paths and optional smoke-test overrides from `config.toml`.
4. The step scripts in `src/ethereum_merge/steps/` execute in notebook order.
5. Outputs are written under `outputs/`, while `docs/reference/` holds the original notebook, the notebook copy-out, and the split map, and the runtime parquet data lives under `data/processed/`.

## Main Assumptions

- The generated step scripts should stay close to the notebook code instead of being deeply refactored.
- Notebook state is preserved through one shared execution context.
- The bundled runtime data in `data/processed/` is local to this project copy and does not mutate the original `one-time-projects` files.
