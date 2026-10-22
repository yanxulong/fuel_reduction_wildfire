"""Shared helpers for workflow tasks.

Contents:
- ensure_dir: create output directory tree
- configure_geospatial_environment: configure and validate Conda GDAL/PROJ data paths
- inspect_compute_capacity: report CPU/RAM and select bounded geospatial worker count
- find_fire_point_source: resolve the first available MTBS Fire Occurrence points path
- write_csv: write CSV atomically
- should_skip, skip_task: skip-if-existing logic and log message
"""

from __future__ import annotations

import multiprocessing
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


def inspect_compute_capacity(
    *, memory_per_worker_gib: float = 4.0, max_workers: int = 8
) -> dict[str, float | int | None]:
    """Inspect node capacity and pick safe parallelism for shard-level geospatial work."""
    cpu_count = os.cpu_count() or multiprocessing.cpu_count() or 1
    total_gib = None
    available_gib = None
    try:
        import psutil

        memory = psutil.virtual_memory()
        total_gib = memory.total / 1024**3
        available_gib = memory.available / 1024**3
    except ImportError:
        pass
    if available_gib is None:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_gib = pages * page_size / 1024**3
            total_pages = os.sysconf("SC_PHYS_PAGES")
            total_gib = total_pages * page_size / 1024**3
        except (AttributeError, OSError, ValueError):
            pass
    if available_gib is None:
        worker_count = 1
    else:
        memory_workers = max(1, int(available_gib // memory_per_worker_gib))
        worker_count = min(max_workers, max(1, cpu_count - 1), memory_workers)
    capacity_text = (
        f"[capacity] cpus={cpu_count}; total_ram_gib={total_gib:.1f}"
        if total_gib is not None
        else f"[capacity] cpus={cpu_count}; total_ram_gib=unknown"
    )
    print(capacity_text, end="")
    print(
        f"; available_ram_gib={available_gib:.1f}; geospatial_workers={worker_count}"
        if available_gib is not None else f"; available_ram_gib=unknown; geospatial_workers={worker_count}"
    )
    return {
        "cpus": cpu_count,
        "total_ram_gib": total_gib,
        "available_ram_gib": available_gib,
        "workers": worker_count,
    }


def find_fire_point_source(candidates: Iterable[Path]) -> Path:
    """Return first nonempty MTBS Fire Occurrence point source from configured paths."""
    paths = tuple(Path(path) for path in candidates)
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No nonempty MTBS Fire Occurrence point shapefile found; checked: {paths}")


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
