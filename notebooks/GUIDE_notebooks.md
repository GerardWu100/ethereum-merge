# GUIDE_notebooks.md

## Part 1: Conceptual Explanation

The `notebooks/` folder contains a thin execution notebook that delegates to the Python package under `src/`. The original source notebook is preserved unchanged under `docs/reference/`.

## Part 2: Code Reference

- `demo.ipynb`: Thin execution notebook that calls `ethereum_merge.pipeline.run_pipeline()`.

## Part 3: Short Journal

- 2026-04-16: Added a wrapper notebook so exploratory execution stays in Jupyter without duplicating backend logic.
