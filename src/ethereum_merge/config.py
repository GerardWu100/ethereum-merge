"""Project paths and execution context for the notebook-derived pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def load_smoke_overrides() -> dict[str, Any]:
    """Load smoke-test date bounds from ``config.toml`` at the project root."""
    config_path = project_root() / 'config.toml'
    if not config_path.exists():
        return {}

    with config_path.open('rb') as config_file:
        settings = tomllib.load(config_file)

    smoke_settings = settings.get('pipeline', {}).get('smoke', {})
    overrides: dict[str, Any] = {}
    if 'start' in smoke_settings:
        overrides['smoke_start'] = smoke_settings['start']
    if 'end' in smoke_settings:
        overrides['smoke_end'] = smoke_settings['end']
    return overrides


@dataclass(slots=True)
class ProjectConfig:
    """Filesystem locations used by the generated pipeline."""

    project_root: Path = field(default_factory=project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / 'data'

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / 'processed'

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / 'outputs'

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / 'figures'

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / 'tables'

    @property
    def steps_dir(self) -> Path:
        return Path(__file__).resolve().parent / 'steps'


def build_execution_context(
    config: ProjectConfig,
    context_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared global namespace used when executing step scripts.

    Step scripts are executed with ``exec`` and expect notebook-style globals
    such as ``PROCESSED_DATA_DIR``, ``FIGURES_DIR``, and ``SMOKE_TEST_MODE``.
    """
    # Ensure output locations exist before any step writes figures or tables.
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    overrides = dict(context_overrides or {})
    smoke_test_mode = bool(overrides.get('SMOKE_TEST_MODE', False))

    context: dict[str, Any] = {
        '__name__': '__main__',
        'PROJECT_ROOT': config.project_root,
        'DATA_DIR': config.data_dir,
        'PROCESSED_DATA_DIR': config.processed_data_dir,
        'OUTPUTS_DIR': config.outputs_dir,
        'FIGURES_DIR': config.figures_dir,
        'TABLES_DIR': config.tables_dir,
        'OVERRIDES': overrides,
        'SMOKE_TEST_MODE': smoke_test_mode,
    }
    # Caller overrides win over defaults (for example smoke date bounds).
    context.update(overrides)
    return context
