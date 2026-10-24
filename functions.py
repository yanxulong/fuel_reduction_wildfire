"""Shared helpers for workflow tasks.

Contents:
- ensure_dir: create output directory tree
- configure_geospatial_environment: configure and validate Conda GDAL/PROJ data paths
- inspect_compute_capacity: report allocated CPU/RAM and select bounded geospatial worker count
- discover_landfire_tiles, validate_landfire_tiles, build_landfire_vrt: find and validate tiled LANDFIRE rasters without mosaicking data
- zonal_landfire_terrain_stats, sample_landfire_terrain_points: summarize elevation, aspect, and slope at polygon and point levels
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


def discover_landfire_tiles(layer_dir: Path) -> tuple[Path, ...]:
    """Find nonempty GeoTIFF tiles for one LANDFIRE layer."""
    directory = Path(layer_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"LANDFIRE layer directory is missing: {directory}")
    tiles = tuple(
        path.resolve()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"} and path.stat().st_size > 0
    )
    if not tiles:
        raise FileNotFoundError(f"No nonempty GeoTIFF tiles found under: {directory}")
    return tiles


def validate_landfire_tiles(layer_name: str, tiles: Iterable[Path]) -> dict[str, object]:
    """Validate tile CRS, band count, pixel grid, and NoData consistency."""
    import math

    import rasterio
    from pyproj import CRS

    paths = tuple(Path(path) for path in tiles)
    if not paths:
        raise ValueError(f"No tiles supplied for LANDFIRE {layer_name}")
    base = None
    base_crs = None
    base_nodata = None
    for path in paths:
        with rasterio.open(path) as source:
            if source.count != 1:
                raise ValueError(f"Expected one band in LANDFIRE {layer_name} tile: {path}; bands={source.count}")
            if source.crs is None:
                raise ValueError(f"LANDFIRE {layer_name} tile has no CRS: {path}")
            current_crs = CRS.from_user_input(source.crs)
            if base is None:
                base = source.transform
                base_crs = current_crs
                base_nodata = source.nodata
            else:
                if current_crs != base_crs:
                    raise ValueError(f"LANDFIRE {layer_name} tile CRS mismatch: {path}")
                transform = source.transform
                if not (
                    math.isclose(transform.a, base.a, rel_tol=0, abs_tol=1e-9)
                    and math.isclose(transform.e, base.e, rel_tol=0, abs_tol=1e-9)
                    and math.isclose(transform.b, base.b, rel_tol=0, abs_tol=1e-12)
                    and math.isclose(transform.d, base.d, rel_tol=0, abs_tol=1e-12)
                ):
                    raise ValueError(f"LANDFIRE {layer_name} tile resolution or rotation mismatch: {path}")
                nodata_equal = source.nodata == base_nodata
                if source.nodata is not None and base_nodata is not None:
                    nodata_equal = nodata_equal or (
                        math.isnan(source.nodata)
                        and math.isnan(base_nodata)
                    )
                if not nodata_equal:
                    raise ValueError(f"LANDFIRE {layer_name} tile NoData mismatch: {path}")
            transform = source.transform
            col_offset = (transform.c - base.c) / base.a
            row_offset = (transform.f - base.f) / base.e
            if not (
                math.isclose(col_offset, round(col_offset), rel_tol=0, abs_tol=1e-5)
                and math.isclose(row_offset, round(row_offset), rel_tol=0, abs_tol=1e-5)
            ):
                raise ValueError(f"LANDFIRE {layer_name} tile is misaligned to pixel grid: {path}")
    return {
        "layer": layer_name,
        "tile_count": len(paths),
        "crs": base_crs,
        "resolution": (base.a, base.e),
        "nodata": base_nodata,
    }


def build_landfire_vrt(tiles: Iterable[Path], output_path: Path) -> Path:
    """Build a lightweight GDAL virtual raster; source pixel data stays in tiles."""
    import math
    import xml.etree.ElementTree as ET

    import rasterio
    from pyproj import CRS

    paths = tuple(Path(path).resolve() for path in tiles)
    if not paths:
        raise ValueError("Cannot build a LANDFIRE VRT without input tiles")
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with rasterio.open(paths[0]) as first:
        first_transform = first.transform
        if first_transform.b != 0 or first_transform.d != 0 or first_transform.a <= 0 or first_transform.e >= 0:
            raise ValueError(f"LANDFIRE VRT requires north-up, unrotated tiles: {paths[0]}")
        x_resolution = first_transform.a
        y_resolution = -first_transform.e
        dtype = first.dtypes[0]
        crs_definition = CRS.from_user_input(first.crs) if first.crs else None
        crs_wkt = crs_definition.to_wkt() if crs_definition else None
        nodata = first.nodata
        band_count = first.count
        source_metadata = []
        left = float("inf")
        right = float("-inf")
        bottom = float("inf")
        top = float("-inf")
        for path in paths:
            with rasterio.open(path) as source:
                if source.count != band_count or source.dtypes[0] != dtype or source.crs is None:
                    raise ValueError(f"LANDFIRE tile band, dtype, or CRS mismatch: {path}")
                transform = source.transform
                if (
                    not math.isclose(transform.a, x_resolution, rel_tol=0, abs_tol=1e-9)
                    or not math.isclose(transform.e, -y_resolution, rel_tol=0, abs_tol=1e-9)
                    or transform.b != 0
                    or transform.d != 0
                ):
                    raise ValueError(f"LANDFIRE tile grid mismatch while building VRT: {path}")
                nodata_equal = source.nodata == nodata
                if source.nodata is not None and nodata is not None:
                    nodata_equal = nodata_equal or (math.isnan(source.nodata) and math.isnan(nodata))
                if CRS.from_user_input(source.crs) != crs_definition or not nodata_equal:
                    raise ValueError(f"LANDFIRE tile CRS or NoData mismatch while building VRT: {path}")
                tile_left = transform.c
                tile_top = transform.f
                tile_right = tile_left + source.width * x_resolution
                tile_bottom = tile_top - source.height * y_resolution
                left = min(left, tile_left)
                right = max(right, tile_right)
                bottom = min(bottom, tile_bottom)
                top = max(top, tile_top)
                block_height, block_width = source.block_shapes[0]
                source_metadata.append((path, source.width, source.height, transform, block_width, block_height))

    gdal_types = {
        "uint8": "Byte",
        "uint16": "UInt16",
        "int16": "Int16",
        "uint32": "UInt32",
        "int32": "Int32",
        "uint64": "UInt64",
        "int64": "Int64",
        "float32": "Float32",
        "float64": "Float64",
    }
    if dtype not in gdal_types:
        raise ValueError(f"Unsupported LANDFIRE raster dtype for VRT: {dtype}")
    width = round((right - left) / x_resolution)
    height = round((top - bottom) / y_resolution)
    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    if crs_wkt:
        ET.SubElement(root, "SRS").text = crs_wkt
    ET.SubElement(root, "GeoTransform").text = (
        f"{left:.15g}, {x_resolution:.15g}, 0, {top:.15g}, 0, {-y_resolution:.15g}"
    )
    band = ET.SubElement(root, "VRTRasterBand", dataType=gdal_types[dtype], band="1")
    if nodata is not None:
        ET.SubElement(band, "NoDataValue").text = str(nodata)
    for path, tile_width, tile_height, transform, block_width, block_height in source_metadata:
        col_offset = round((transform.c - left) / x_resolution)
        row_offset = round((top - transform.f) / y_resolution)
        if (
            not math.isclose((transform.c - left) / x_resolution, col_offset, rel_tol=0, abs_tol=1e-5)
            or not math.isclose((top - transform.f) / y_resolution, row_offset, rel_tol=0, abs_tol=1e-5)
        ):
            raise ValueError(f"LANDFIRE tile is not aligned to VRT pixel grid: {path}")
        source = ET.SubElement(band, "SimpleSource")
        ET.SubElement(source, "SourceFilename", relativeToVRT="0").text = str(path)
        ET.SubElement(source, "SourceBand").text = "1"
        ET.SubElement(
            source,
            "SourceProperties",
            RasterXSize=str(tile_width),
            RasterYSize=str(tile_height),
            DataType=gdal_types[dtype],
            BlockXSize=str(block_width),
            BlockYSize=str(block_height),
        )
        ET.SubElement(source, "SrcRect", xOff="0", yOff="0", xSize=str(tile_width), ySize=str(tile_height))
        ET.SubElement(
            source,
            "DstRect",
            xOff=str(col_offset),
            yOff=str(row_offset),
            xSize=str(tile_width),
            ySize=str(tile_height),
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output_path, encoding="UTF-8", xml_declaration=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"GDAL produced an empty LANDFIRE VRT: {output_path}")
    with rasterio.open(output_path) as check:
        if check.width != width or check.height != height or check.count != 1:
            raise RuntimeError(f"Generated LANDFIRE VRT failed metadata validation: {output_path}")
    return output_path


def _landfire_grid(paths: Mapping[str, Path]):
    """Open LANDFIRE VRTs and verify shared grid before zonal rasterization."""
    import rasterio

    datasets = {name: rasterio.open(path) for name, path in paths.items()}
    first_name = next(iter(datasets))
    first = datasets[first_name]
    for name, source in datasets.items():
        if source.crs is None or source.crs != first.crs:
            for opened in datasets.values():
                opened.close()
            raise ValueError(f"LANDFIRE VRT CRS mismatch: {name}={source.crs}; {first_name}={first.crs}")
        if source.transform != first.transform or source.width != first.width or source.height != first.height:
            for opened in datasets.values():
                opened.close()
            raise ValueError(f"LANDFIRE VRT grids differ: {name} does not align with {first_name}")
        if source.count != 1:
            for opened in datasets.values():
                opened.close()
            raise ValueError(f"LANDFIRE VRT must be single-band: {name} has {source.count} bands")
    return datasets


def zonal_landfire_terrain_stats(
    geometries: Sequence[object],
    vrt_paths: Mapping[str, Path],
    *,
    block_size: int = 2048,
) -> dict[str, object]:
    """Compute center-in-pixel zonal means by rasterizing many small polygons per block.

    WUI single-part geometries should not overlap. Fire areas can overlap, so call this
    with one dissolved event geometry at a time. Aspect code -1 is treated as flat;
    directional components use valid aspect values from 0 through 359 degrees.
    """
    import math

    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window, from_bounds

    if block_size < 1:
        raise ValueError("block_size must be positive")
    required = {"elev", "asp", "slpd"}
    if set(vrt_paths) != required:
        raise ValueError(f"LANDFIRE VRT mapping must contain {sorted(required)}; got {sorted(vrt_paths)}")
    count = len(geometries)
    elev_sum = np.zeros(count, dtype=np.float64)
    elev_count = np.zeros(count, dtype=np.int64)
    slope_sum = np.zeros(count, dtype=np.float64)
    slope_count = np.zeros(count, dtype=np.int64)
    aspect_north_sum = np.zeros(count, dtype=np.float64)
    aspect_east_sum = np.zeros(count, dtype=np.float64)
    aspect_count = np.zeros(count, dtype=np.int64)
    aspect_valid_count = np.zeros(count, dtype=np.int64)
    aspect_flat_count = np.zeros(count, dtype=np.int64)

    datasets = _landfire_grid(vrt_paths)
    try:
        base = datasets["elev"]
        transform = base.transform
        chunk_features: dict[tuple[int, int], list[int]] = {}
        for index, geometry in enumerate(geometries):
            if geometry is None or geometry.is_empty:
                continue
            bounds_window = from_bounds(*geometry.bounds, transform=transform)
            row_start = max(0, math.floor(bounds_window.row_off))
            row_stop = min(base.height, math.ceil(bounds_window.row_off + bounds_window.height))
            col_start = max(0, math.floor(bounds_window.col_off))
            col_stop = min(base.width, math.ceil(bounds_window.col_off + bounds_window.width))
            if row_start >= row_stop or col_start >= col_stop:
                continue
            first_block_row = (row_start // block_size) * block_size
            first_block_col = (col_start // block_size) * block_size
            for block_row in range(first_block_row, row_stop, block_size):
                for block_col in range(first_block_col, col_stop, block_size):
                    chunk_features.setdefault((block_row, block_col), []).append(index)

        for (block_row, block_col), feature_indices in chunk_features.items():
            height = min(block_size, base.height - block_row)
            width = min(block_size, base.width - block_col)
            window = Window(block_col, block_row, width, height)
            block_transform = base.window_transform(window)
            shapes = ((geometries[index], index + 1) for index in feature_indices)
            labels = rasterize(
                shapes,
                out_shape=(height, width),
                transform=block_transform,
                fill=0,
                dtype="int32",
                all_touched=False,
            )
            assigned = labels > 0
            if not assigned.any():
                continue
            for layer, source in datasets.items():
                raster = source.read(1, window=window, masked=True)
                values = np.asarray(raster.data)
                valid = assigned & ~np.ma.getmaskarray(raster) & np.isfinite(values)
                if layer == "elev" or layer == "slpd":
                    ids = labels[valid]
                    weights = values[valid].astype(np.float64, copy=False)
                    layer_counts = np.bincount(ids, minlength=count + 1)
                    layer_sums = np.bincount(ids, weights=weights, minlength=count + 1)
                    if layer == "elev":
                        elev_count += layer_counts[1:]
                        elev_sum += layer_sums[1:]
                    else:
                        slope_count += layer_counts[1:]
                        slope_sum += layer_sums[1:]
                else:
                    selected = values[valid].astype(np.float64, copy=False)
                    selected_ids = labels[valid]
                    supported = (selected == -1) | ((selected >= 0) & (selected <= 359))
                    if supported.any():
                        aspect_count += np.bincount(selected_ids[supported], minlength=count + 1)[1:]
                    flat = selected == -1
                    if flat.any():
                        aspect_flat_count += np.bincount(selected_ids[flat], minlength=count + 1)[1:]
                    directional = (selected >= 0) & (selected <= 359)
                    if directional.any():
                        theta = np.deg2rad(selected[directional])
                        ids = selected_ids[directional]
                        north = np.bincount(ids, weights=np.cos(theta), minlength=count + 1)
                        east = np.bincount(ids, weights=np.sin(theta), minlength=count + 1)
                        aspect_north_sum += north[1:]
                        aspect_east_sum += east[1:]
                        aspect_valid_count += np.bincount(ids, minlength=count + 1)[1:]
    finally:
        for source in datasets.values():
            source.close()

    def mean(values, counts):
        return np.divide(values, counts, out=np.full(count, np.nan), where=counts > 0)

    northness = mean(aspect_north_sum, aspect_valid_count)
    eastness = mean(aspect_east_sum, aspect_valid_count)
    circular_degrees = np.mod(np.degrees(np.arctan2(eastness, northness)), 360.0)
    no_direction = aspect_valid_count == 0
    circular_degrees[no_direction] = np.nan
    return {
        "elev_mean_m": mean(elev_sum, elev_count),
        "elev_valid_pixels": elev_count,
        "slpd_mean_degrees": mean(slope_sum, slope_count),
        "slpd_valid_pixels": slope_count,
        "asp_mean_northness": northness,
        "asp_mean_eastness": eastness,
        "asp_circular_mean_degrees": circular_degrees,
        "asp_valid_pixels": aspect_count,
        "asp_directional_pixels": aspect_valid_count,
        "asp_flat_pixels": aspect_flat_count,
    }


def sample_landfire_terrain_points(
    coordinates: Sequence[tuple[float, float]],
    vrt_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Sample one raster cell per point; return NaN for masked or out-of-extent cells."""
    import numpy as np

    if not coordinates:
        return {layer: np.empty(0, dtype=np.float64) for layer in vrt_paths}
    datasets = _landfire_grid(vrt_paths)
    try:
        results = {}
        for layer, source in datasets.items():
            sampled = np.ma.vstack(list(source.sample(coordinates, indexes=1, masked=True)))
            values = np.asarray(sampled[:, 0].astype(np.float64).filled(np.nan), dtype=np.float64)
            results[layer] = values
    finally:
        for source in datasets.values():
            source.close()
    return results


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
