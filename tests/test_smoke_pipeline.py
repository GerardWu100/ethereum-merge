"""Smoke test that the notebook-derived pipeline runs end-to-end."""

from ethereum_merge.cli import build_smoke_overrides
from ethereum_merge.pipeline import run_pipeline


def test_smoke_pipeline_runs() -> None:
    """Run the pipeline on the configured smoke date window."""
    context = run_pipeline(context_overrides=build_smoke_overrides())
    assert isinstance(context, dict)
    assert context.get('SMOKE_TEST_MODE') is True
