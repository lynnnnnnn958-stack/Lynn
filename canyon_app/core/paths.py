"""Filesystem paths for Canyon v9.

Keep path logic in one place so pages and runners do not guess where files live.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT
VAULT_ROOT = PROJECT_ROOT / "canyon_output_vault"


def project_file(name: str) -> Path:
    """Return an absolute path under the project root."""
    return PROJECT_ROOT / name


def output_file(name: str) -> Path:
    """Return an absolute output path.

    Current Canyon outputs are still written to the project root. This wrapper
    makes it easy to move outputs into a dedicated folder later.
    """
    return OUTPUT_ROOT / name

