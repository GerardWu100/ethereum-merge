# GUIDE_src.md

## Part 1: Conceptual Explanation

The `src/ethereum_merge` package is a thin execution layer around the original notebook logic. `config.py` defines where inputs and outputs live, `pipeline.py` executes the notebook-derived step files in order, `cli.py` exposes the command-line entrypoint, and `steps/` contains the copied notebook code grouped by major heading. This design preserves the original linear workflow while removing backend logic from the new notebook wrapper.

The `steps/` folder is intentionally procedural. Each file corresponds to a notebook section and expects the shared context created by the pipeline. That context includes filesystem paths, runtime overrides, and any variables created by earlier sections. The result is notebook parity without keeping the operational logic inside `.ipynb` files.

## Part 2: Code Reference

- `config.py`: Path configuration, `config.toml` loading, and context assembly for step execution.
- `pipeline.py`: Sequentially executes all files in `steps/` with one shared namespace.
- `cli.py`: Command-line entrypoint (`ethereum-merge` console script).
- `steps/*.py`: Notebook-derived code files ordered by notebook section.

## Part 3: Short Journal

- 2026-04-16: Kept notebook semantics by executing ordered step scripts inside one shared context instead of rewriting the workflow into new abstractions.
- 2026-05-20: Moved CLI into the package and added `config.toml` loading for smoke-test overrides.
