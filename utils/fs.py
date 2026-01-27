"""Filesystem helpers used across modules."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create *path* if needed and return it.

    Keeping this helper centralised makes it easier to ensure temp and output
    directories exist before long-running tasks start.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path
