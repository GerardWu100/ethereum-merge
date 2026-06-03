"""Command-line entrypoint for the notebook-derived pipeline."""

from __future__ import annotations

import argparse
from pprint import pprint
from typing import Any

from .config import load_smoke_overrides
from .pipeline import run_pipeline


def build_smoke_overrides() -> dict[str, Any]:
    """Return smoke-test overrides loaded from config.toml."""
    overrides = load_smoke_overrides()
    overrides['SMOKE_TEST_MODE'] = True
    return overrides


def main() -> None:
    """Run the project pipeline from the command line."""
    parser = argparse.ArgumentParser(description='Run the notebook-derived pipeline.')
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run with smaller verification settings when supported.',
    )
    parser.add_argument(
        '--print-keys',
        action='store_true',
        help='Print the final context keys after execution.',
    )
    args = parser.parse_args()

    overrides = build_smoke_overrides() if args.smoke else {}
    context = run_pipeline(context_overrides=overrides)

    if args.print_keys:
        pprint(sorted(context.keys()))


if __name__ == '__main__':
    main()
