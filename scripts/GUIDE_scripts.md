# GUIDE_scripts.md

## Part 1: Conceptual Explanation

The `scripts/` folder holds thin command-line wrappers and one-off data preparation utilities. Reusable pipeline logic lives in `src/ethereum_merge/`; these files parse inputs or delegate to the package.

## Part 2: Code Reference

- `run_pipeline.py`: Thin wrapper that calls `ethereum_merge.cli.main`.
- `convert_raw_to_parquet.py`: Reads the original crypto files when they are restored locally under `data/raw/`, applies the same date-window filtering used by the project, and writes the runtime parquet files consumed by `src/ethereum_merge/steps/`.

## Part 3: Short Journal

- 2026-04-16: Added the raw-to-parquet conversion script to document how the bundled parquet inputs were produced.
- 2026-05-20: Moved the pipeline CLI wrapper from the repo root into `scripts/`.
