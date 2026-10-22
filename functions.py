"""Shared helpers for workflow tasks.

Contents:
- ensure_dir: create output directory tree
- configure_geospatial_environment: configure and validate Conda GDAL/PROJ data paths
- write_csv: write CSV atomically
- should_skip, skip_task: skip-if-existing logic and log message
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def configure_geospatial_environment() -> Path | None:
    """Point geospatial libraries at the active Conda environment's data files."""
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None

    proj_dir = Path(prefix) / "share" / "proj"
    proj_database = proj_dir / "proj.db"
    if not proj_database.is_file():
        raise FileNotFoundError(f"PROJ database missing: {proj_database}")

    os.environ["PROJ_LIB"] = str(proj_dir)
    os.environ["PROJ_DATA"] = str(proj_dir)
    gdal_dir = Path(prefix) / "share" / "gdal"
    if gdal_dir.is_dir():
        os.environ["GDAL_DATA"] = str(gdal_dir)

    from pyproj import CRS, datadir

    datadir.set_data_dir(str(proj_dir))
    try:
        CRS.from_epsg(4326)
    except Exception as exc:
        raise RuntimeError(f"PROJ database failed validation: {proj_database}") from exc

    print(f"[env] PROJ data: {proj_dir}")
    if gdal_dir.is_dir():
        print(f"[env] GDAL data: {gdal_dir}")
    return proj_dir


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


def should_skip(overwrite_existing: bool, paths: Iterable[Path]) -> bool:
    """Skip task when every nonempty output already exists and overwrite is off."""
    targets = list(paths)
    return bool(targets) and not overwrite_existing and all(
        path.exists() and path.stat().st_size > 0 for path in targets
    )


def skip_task(name: str, *paths: Path) -> None:
    """Log existing outputs when shared skip-if-present logic skips a task."""
    targets = ", ".join(str(path) for path in paths)
    print(f"[skip] {name}: existing targets: {targets}")
