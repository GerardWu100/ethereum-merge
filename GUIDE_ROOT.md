# GUIDE_ROOT.md

## Part 1: Conceptual Explanation

This repository is a notebook-to-project conversion. The notebook reference materials live under `docs/reference/`, and the executable workflow is split into ordered Python step scripts under `src/ethereum_merge/steps/`. The root folder keeps the project thin: `scripts/run_pipeline.py` and the `ethereum-merge` console entrypoint execute the notebook-derived pipeline, `config.toml` holds tunable settings, `pyproject.toml` defines the Python 3.13 environment, `notebooks/` holds the new thin execution notebook, `docs/` records both the section split and a direct notebook copy-out, and `data/` plus `outputs/` hold local inputs and generated artifacts. Because this notebook depended on local crypto source files, `scripts/convert_raw_to_parquet.py` records the one-off raw-to-parquet conversion process, and `data/processed/` stores the parquet files used by the scripts.

The execution model intentionally mirrors notebook semantics. Each step script is executed in order inside one shared namespace, so variables, functions, and imported modules persist across sections just as they did in the original notebook. This keeps the code close to the source notebook while moving the reusable logic out of the new notebook wrapper.

## Part 2: Code Reference

- `config.toml`: Smoke-test date window and other pipeline settings read by the CLI.
- `scripts/run_pipeline.py`: Thin wrapper around the package CLI.
- `pyproject.toml` `[project.scripts]`: Installs the `ethereum-merge` console command.
- `src/ethereum_merge/cli.py`: Command-line entrypoint. Supports `--smoke` using values from `config.toml`.
- `src/ethereum_merge/config.py`: Defines project paths and builds the shared execution context.
- `src/ethereum_merge/pipeline.py`: Runs each generated step script in notebook order.
- `src/ethereum_merge/steps/`: Contains the notebook-derived Python scripts, one file per major notebook section.
- `scripts/convert_raw_to_parquet.py`: One-off helper that rebuilds the filtered parquet inputs when the original source files are restored locally.
- `notebooks/demo.ipynb`: Thin notebook that only calls the backend pipeline.
- `docs/reference/ethereum-merge.ipynb`: Unchanged copy of the original source notebook.
- `docs/reference/notebook_reference.md`: Markdown copy-out of the source notebook in notebook order.
- `docs/reference/notebook_split.md`: Maps notebook sections to generated script files.
- `data/raw/`: Expected location for restored source files before conversion.
- `data/processed/`: Parquet copies used by the converted pipeline.
- `tests/test_smoke_pipeline.py`: End-to-end smoke verification.

## Part 3: Short Journal

- 2026-04-16: Split the original notebook into ordered step scripts while preserving the raw notebook under `docs/reference/`.
- 2026-04-16: Added a Markdown notebook copy-out so the project keeps a readable version of the original notebook text locally.
- 2026-04-16: Replaced the earlier pickle-based processed data layer with an explicit raw-to-parquet conversion step.
- 2026-05-20: Aligned layout with the standard project structure: CLI in package, thin `scripts/` wrapper, `config.toml`, and `tests/`.
- 2026-08-10: Added `CODE_EXPLAINED.html`, a single-page walkthrough of the pipeline structure, formulas, and sample values. Fixed the defects it uncovered: missing `numpy` and `statsmodels.formula.api` imports that stopped any run, a roll adjustment that compared contract prices one bar apart, an unseeded shuffle in the feature-importance loop, an `eth_threshold` name collision between steps 03 and 06, and a stale RiskLabAI commit in `uv.lock`.
