"""Execute notebook-derived step scripts in original notebook order."""

from __future__ import annotations

from typing import Any

from .config import ProjectConfig, build_execution_context


def run_pipeline(context_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run each generated step script with a shared notebook-style context.

    Returns the final execution namespace so callers can inspect variables
    produced by the last step (for example with ``--print-keys``).
    """
    config = ProjectConfig()
    context = build_execution_context(config=config, context_overrides=context_overrides)

    # Step filenames are numbered so lexical sort preserves notebook order.
    for step_path in sorted(config.steps_dir.glob('*.py')):
        context['__file__'] = str(step_path)
        source = step_path.read_text()
        exec(compile(source, str(step_path), 'exec'), context)

    return context
