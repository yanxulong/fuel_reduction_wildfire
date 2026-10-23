"""Shared helpers for workflow tasks.

Contents:
- ensure_dir: create output directory tree
- configure_geospatial_environment: configure and validate Conda GDAL/PROJ data paths
- inspect_compute_capacity: report allocated CPU/RAM and select bounded geospatial worker count
- find_fire_point_source: resolve the first available MTBS Fire Occurrence points path
- find_spatial_source: resolve canonical or unique recursive spatial input, optionally requiring matching DBF
- normalize_identifier: normalize event and activity IDs for cross-layer joins
- build_treatment_time_windows: build configurable calendar, cumulative, and lag-band windows
- treatment_window_membership: classify exact or year-only treatment dates against a window
- angular_treatment_metrics: measure WUI-facing angular treatment coverage and thickness
- write_csv: write CSV atomically
- should_skip, skip_task: skip-if-existing logic and log message
"""

from __future__ import annotations

import multiprocessing
import csv
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def configure_geospatial_environment() -> Path | None:
    """Point geospatial libraries at the active Conda environment's data files."""
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix

    proj_dir = Path(prefix) / "share" / "proj"
    proj_database = proj_dir / "proj.db"
    if not proj_database.is_file():
        if os.environ.get("CONDA_PREFIX"):
            raise FileNotFoundError(f"PROJ database missing: {proj_database}")
        return None

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
    """Inspect job-limited CPU/RAM and pick safe shard-level parallelism."""
    detected_cpus = os.cpu_count() or multiprocessing.cpu_count() or 1
    try:
        detected_cpus = min(detected_cpus, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    try:
        cpu_count = min(detected_cpus, int(slurm_cpus)) if slurm_cpus else detected_cpus
    except (TypeError, ValueError):
        cpu_count = detected_cpus
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
    # Cgroup limits reflect Slurm allocations more closely than whole-node RAM.
    memory_limit_bytes = None
    memory_current_bytes = None
    for limit_file, current_file in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    ):
        try:
            raw_limit = limit_file.read_text().strip()
            limit_value = int(raw_limit)
            if limit_value > 0 and limit_value < 2**60:
                memory_limit_bytes = limit_value
                memory_current_bytes = int(current_file.read_text().strip())
                break
        except (OSError, ValueError):
            continue
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if memory_limit_bytes is None and slurm_mem:
        try:
            memory_limit_bytes = int(slurm_mem) * 1024**2
        except ValueError:
            pass
    if memory_limit_bytes is not None:
        cgroup_total_gib = memory_limit_bytes / 1024**3
        cgroup_available_gib = max(0.0, (memory_limit_bytes - (memory_current_bytes or 0)) / 1024**3)
        total_gib = min(total_gib, cgroup_total_gib) if total_gib is not None else cgroup_total_gib
        available_gib = min(available_gib, cgroup_available_gib) if available_gib is not None else cgroup_available_gib
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


def build_treatment_time_windows(
    *, rolling_years: Iterable[int] = range(1, 11), lag_bands: Iterable[tuple[str, float, float]] = (("lag_1_2y", 1, 2), ("lag_2_4y", 2, 4))
) -> list[dict[str, object]]:
    """Return named temporal definitions; lag bands use age [min, max)."""
    windows: list[dict[str, object]] = [
        {"name": "previous_calendar_year", "kind": "previous_calendar_year"}
    ]
    seen = {"previous_calendar_year"}
    for years in rolling_years:
        numeric_years = float(years)
        if not math.isfinite(numeric_years) or not numeric_years.is_integer() or numeric_years < 1:
            raise ValueError(f"Rolling treatment windows must be positive; got {years}")
        years = int(numeric_years)
        name = f"rolling_{years}y"
        if name in seen:
            raise ValueError(f"Duplicate treatment time window: {name}")
        windows.append({"name": name, "kind": "rolling", "max_years": years})
        seen.add(name)

    normalized_bands = []
    for name, minimum, maximum in lag_bands:
        minimum = float(minimum)
        maximum = float(maximum)
        if (
            not name
            or not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or not minimum.is_integer()
            or not maximum.is_integer()
            or minimum < 0
            or maximum <= minimum
        ):
            raise ValueError(f"Invalid treatment lag band: {(name, minimum, maximum)}")
        normalized_bands.append((str(name), minimum, maximum))
    ordered_bands = sorted(normalized_bands, key=lambda row: (row[1], row[2], row[0]))
    for previous, current in zip(ordered_bands, ordered_bands[1:]):
        if current[1] < previous[2]:
            raise ValueError(f"Treatment lag bands overlap: {previous} and {current}")
    for name, minimum, maximum in ordered_bands:
        if name in seen:
            raise ValueError(f"Duplicate treatment time window: {name}")
        windows.append({"name": name, "kind": "lag_band", "min_years": minimum, "max_years": maximum})
        seen.add(name)
    return windows


def treatment_window_membership(
    completion_date: object,
    completion_year: object,
    fire_date: object,
    fire_year: object,
    window: Mapping[str, object],
    *,
    date_precision: str = "exact",
) -> bool | None:
    """Classify treatment timing; return None when year-only evidence is ambiguous."""
    import pandas as pd

    kind = str(window["kind"])
    completion = pd.to_datetime(completion_date, errors="coerce")
    fire = pd.to_datetime(fire_date, errors="coerce")
    if pd.notna(completion) and pd.notna(fire) and date_precision == "exact":
        if completion >= fire:
            return False
        if kind == "previous_calendar_year":
            start = pd.Timestamp(year=fire.year - 1, month=1, day=1)
            end = pd.Timestamp(year=fire.year, month=1, day=1)
        elif kind == "rolling":
            start = fire - pd.DateOffset(years=int(window["max_years"]))
            end = fire
        elif kind == "lag_band":
            start = fire - pd.DateOffset(years=int(window["max_years"]))
            end = fire - pd.DateOffset(years=int(window["min_years"]))
        else:
            raise ValueError(f"Unknown treatment window kind: {kind}")
        # Rolling is [F-n years, F); lag age bands are [min, max), in elapsed age.
        if kind == "lag_band":
            return bool(start < completion <= end)
        return bool(start <= completion < end)

    if pd.isna(completion_year) or pd.isna(fire_year):
        return None
    try:
        treatment_year = int(completion_year)
        event_year = int(fire_year)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (1900 <= treatment_year <= 2100 and 1900 <= event_year <= 2100):
        return None
    if kind == "previous_calendar_year":
        start = pd.Timestamp(year=event_year - 1, month=1, day=1)
        end = pd.Timestamp(year=event_year, month=1, day=1)
    elif kind == "rolling":
        fire_anchor = fire if pd.notna(fire) else pd.Timestamp(year=event_year, month=7, day=1)
        start = fire_anchor - pd.DateOffset(years=int(window["max_years"]))
        end = fire_anchor
    elif kind == "lag_band":
        fire_anchor = fire if pd.notna(fire) else pd.Timestamp(year=event_year, month=7, day=1)
        start = fire_anchor - pd.DateOffset(years=int(window["max_years"]))
        end = fire_anchor - pd.DateOffset(years=int(window["min_years"]))
    else:
        raise ValueError(f"Unknown treatment window kind: {kind}")

    if date_precision == "calendar_year":
        year_start = pd.Timestamp(year=treatment_year, month=1, day=1)
        year_end = pd.Timestamp(year=treatment_year + 1, month=1, day=1)
    elif date_precision == "fiscal_year":
        year_start = pd.Timestamp(year=treatment_year - 1, month=10, day=1)
        year_end = pd.Timestamp(year=treatment_year, month=10, day=1)
    else:
        # Award year does not establish completion timing.
        return None

    if kind == "lag_band":
        wholly_inside = year_start > start and year_end <= end
        disjoint = year_end <= start or year_start > end
    else:
        wholly_inside = year_start >= start and year_end <= end
        disjoint = year_end <= start or year_start >= end
    if wholly_inside:
        return True
    if disjoint:
        return False
    return None


def normalize_identifier(value: object) -> str | None:
    """Normalize a tabular event/activity identifier without changing its identity."""
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text.upper() or None


def find_spatial_source(
    candidates: Iterable[Path],
    search_roots: Iterable[Path],
    pattern: str,
    *,
    require_attribute_table: bool = False,
) -> Path:
    """Return first canonical source, optionally requiring a nonempty matching DBF."""
    checked = tuple(Path(path) for path in candidates)
    roots = tuple(Path(path) for path in search_roots)

    def has_attribute_table(path: Path) -> bool:
        if not require_attribute_table:
            return True
        expected_stem = path.stem.casefold()
        try:
            return any(
                sibling.is_file()
                and sibling.suffix.casefold() == ".dbf"
                and sibling.stem.casefold() == expected_stem
                and sibling.stat().st_size > 0
                for sibling in path.parent.iterdir()
            )
        except OSError:
            return False

    def usable(path: Path) -> bool:
        return (
            path.is_file()
            and path.stat().st_size > 0
            and has_attribute_table(path)
        )

    for path in checked:
        if usable(path):
            return path
    all_discovered = sorted(
        {
            path.resolve()
            for root in roots
            if root.is_dir()
            for path in root.rglob(pattern)
            if path.is_file() and path.stat().st_size > 0
        },
        key=str,
    )
    discovered = [path for path in all_discovered if has_attribute_table(path)]
    if len(discovered) == 1:
        return discovered[0]
    if len(discovered) > 1:
        rank = lambda path: (
            int("archive" in {part.lower() for part in path.parts}),
            int(any(part.lower().startswith("snapshot") for part in path.parts)),
        )
        best = min(rank(path) for path in discovered)
        preferred = [path for path in discovered if rank(path) == best]
        if len(preferred) == 1:
            return preferred[0]
        raise FileExistsError(
            f"Multiple {pattern} sources have equal priority; configure a canonical path:\n  "
            + "\n  ".join(str(path) for path in preferred)
        )
    raise FileNotFoundError(
        (
            f"No {pattern} spatial source with a matching nonempty .dbf attribute table found."
            if require_attribute_table
            else f"No {pattern} spatial source found."
        )
        + " Checked exact paths:\n  "
        + "\n  ".join(map(str, checked))
        + "\nSearch roots:\n  "
        + "\n  ".join(map(str, roots))
        + (
            "\nShapefiles found without a matching nonempty .dbf:\n  "
            + "\n  ".join(str(path) for path in all_discovered if not has_attribute_table(path))
            if require_attribute_table and all_discovered
            else ""
        )
    )


def angular_treatment_metrics(
    origin: object,
    wui_polygon: object,
    treatment_geometries: Sequence[object],
    *,
    thickness_step_degrees: float = 0.25,
    compute_thickness_when_full: bool = True,
) -> dict[str, float | int | bool | None]:
    """Measure treatment angular coverage before first WUI contact and ray thickness.

    Coverage status is evaluated between polygon-vertex/intersection bearing events.
    Thickness distribution uses deterministic angular quadrature within those intervals.
    All coordinates must share a projected CRS with meter units.
    """
    import numpy as np
    import shapely
    from shapely.geometry import LineString

    if not math.isfinite(thickness_step_degrees) or thickness_step_degrees <= 0:
        raise ValueError("thickness_step_degrees must be a positive finite number")

    empty = {
        "angular_total_degrees": 0.0,
        "angular_covered_degrees": 0.0,
        "angular_coverage_fraction": None,
        "thickness_mean_m": None,
        "thickness_p10_m": None,
        "thickness_min_m": None,
        "thickness_max_m": None,
        "contiguous_thickness_mean_m": None,
        "angular_geometry_status": "ok",
    }
    if origin is None or wui_polygon is None or shapely.is_empty(origin) or shapely.is_empty(wui_polygon):
        empty["angular_geometry_status"] = "missing_geometry"
        return empty
    if wui_polygon.covers(origin):
        empty["angular_geometry_status"] = "fire_point_inside_or_on_wui"
        return empty

    valid_treatments = [
        geometry for geometry in treatment_geometries
        if geometry is not None and not shapely.is_empty(geometry)
    ]
    treatment_union = shapely.union_all(valid_treatments) if valid_treatments else shapely.GeometryCollection()
    wui_boundary = wui_polygon.boundary
    event_geometries = [wui_boundary]
    if not shapely.is_empty(treatment_union):
        event_geometries.append(treatment_union.boundary)
        boundary_intersections = treatment_union.boundary.intersection(wui_boundary)
        if not shapely.is_empty(boundary_intersections):
            event_geometries.append(boundary_intersections)

    x0, y0 = float(origin.x), float(origin.y)
    coordinate_arrays = [shapely.get_coordinates(geometry) for geometry in event_geometries]
    coordinate_arrays = [coords for coords in coordinate_arrays if len(coords)]
    if not coordinate_arrays:
        empty["angular_geometry_status"] = "no_boundary_coordinates"
        return empty
    coordinates = np.concatenate(coordinate_arrays, axis=0)
    angles = np.mod(np.arctan2(coordinates[:, 1] - y0, coordinates[:, 0] - x0), 2 * math.pi)
    angles = np.unique(np.round(angles, 12))
    if len(angles) < 2:
        angles = np.asarray([0.0, math.pi, 2 * math.pi])
    angles.sort()
    max_radius = float(np.max(np.hypot(coordinates[:, 0] - x0, coordinates[:, 1] - y0))) + 2.0
    if not math.isfinite(max_radius) or max_radius <= 2.0:
        empty["angular_geometry_status"] = "invalid_radius"
        return empty

    def ray_wui_distance(theta: float) -> float | None:
        ux, uy = math.cos(theta), math.sin(theta)
        ray = LineString([(x0, y0), (x0 + ux * max_radius, y0 + uy * max_radius)])
        crossing = ray.intersection(wui_boundary)
        crossing_coordinates = shapely.get_coordinates(crossing)
        if not len(crossing_coordinates):
            return None
        distances = (crossing_coordinates[:, 0] - x0) * ux + (crossing_coordinates[:, 1] - y0) * uy
        forward = distances[distances > 1e-7]
        return float(forward.min()) if len(forward) else None

    def ray_treatment_lengths(theta: float, wui_distance: float) -> tuple[float, float]:
        if shapely.is_empty(treatment_union) or wui_distance <= 1e-7:
            return 0.0, 0.0
        ux, uy = math.cos(theta), math.sin(theta)
        end_distance = max(0.0, wui_distance - 1e-6)
        segment = LineString([(x0, y0), (x0 + ux * end_distance, y0 + uy * end_distance)])
        crossed = treatment_union.intersection(segment)
        total_length = float(crossed.length)
        if crossed.geom_type == "LineString":
            longest = total_length
        elif hasattr(crossed, "geoms"):
            longest = max((float(part.length) for part in crossed.geoms if part.geom_type == "LineString"), default=0.0)
        else:
            longest = 0.0
        return total_length, longest

    visible_degrees = 0.0
    covered_degrees = 0.0
    visible_intervals: list[tuple[float, float]] = []
    for index, start in enumerate(angles):
        end = angles[index + 1] if index + 1 < len(angles) else float(angles[0] + 2 * math.pi)
        width = float(end - start)
        if width <= 1e-11:
            continue
        midpoint = float(start + width / 2)
        midpoint_wui_distance = ray_wui_distance(midpoint)
        if midpoint_wui_distance is None:
            continue
        visible_degrees += math.degrees(width)
        visible_intervals.append((float(start), width))
        midpoint_length, _ = ray_treatment_lengths(midpoint, midpoint_wui_distance)
        is_covered = midpoint_length > 1e-7
        if is_covered:
            covered_degrees += math.degrees(width)

    if visible_degrees <= 1e-10:
        empty["angular_geometry_status"] = "no_visible_wui_angle"
        return empty
    coverage_fraction = min(1.0, max(0.0, covered_degrees / visible_degrees))
    thickness_samples: list[tuple[float, float, float]] = []
    if compute_thickness_when_full and coverage_fraction >= 1.0 - 1e-10:
        for start, width in visible_intervals:
            sample_count = max(1, int(math.ceil(math.degrees(width) / thickness_step_degrees)))
            subwidth = width / sample_count
            for sample_index in range(sample_count):
                theta = float(start + (sample_index + 0.5) * subwidth)
                radial_wui_distance = ray_wui_distance(theta)
                if radial_wui_distance is None:
                    continue
                total_length, longest = ray_treatment_lengths(theta, radial_wui_distance)
                thickness_samples.append((total_length, longest, math.degrees(subwidth)))
    sample_values = [row[0] for row in thickness_samples if row[0] > 1e-7]
    sample_longest = [row[1] for row in thickness_samples if row[0] > 1e-7]
    sample_weights = [row[2] for row in thickness_samples if row[0] > 1e-7]
    thickness_mean = float(np.average(sample_values, weights=sample_weights)) if sample_values else None
    contiguous_mean = float(np.average(sample_longest, weights=sample_weights)) if sample_longest else None
    weighted_p10 = None
    if sample_values:
        order = np.argsort(sample_values)
        sorted_values = np.asarray(sample_values, dtype=np.float64)[order]
        sorted_weights = np.asarray(sample_weights, dtype=np.float64)[order]
        cumulative = np.cumsum(sorted_weights) / sorted_weights.sum()
        weighted_p10 = float(np.interp(0.10, cumulative, sorted_values))
    return {
        "angular_total_degrees": visible_degrees,
        "angular_covered_degrees": covered_degrees,
        "angular_coverage_fraction": coverage_fraction,
        "thickness_mean_m": thickness_mean,
        "thickness_p10_m": weighted_p10,
        "thickness_min_m": float(min(sample_values)) if sample_values else None,
        "thickness_max_m": float(max(sample_values)) if sample_values else None,
        "contiguous_thickness_mean_m": contiguous_mean,
        "angular_geometry_status": "ok",
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
