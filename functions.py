"""Shared filesystem and reporting helpers for workflow tasks."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create directory and parents, then return its path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(
    rows: Iterable[Mapping[str, object]],
    path: Path,
    *,
    fieldnames: Sequence[str],
) -> Path:
    """Write CSV atomically so interrupted runs do not leave partial reports."""
    ensure_dir(path.parent)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
