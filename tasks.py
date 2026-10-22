"""WUI dataset tasks.

Contents:
- build_wui_singlepart_dataset: combine 2000, 2010, and 2020 WUI blocks as single-part records
- build_fire_wui_centroid_paths: link MTBS fire centroids to WUI single-part edges within a configurable radius
- build_fire_wui_centroid_path_distance_samples: build path datasets for 1 to 10 miles
"""

from __future__ import annotations

import functions


def build_wui_singlepart_dataset(project):
    """Build source-preserving single-part WUI GeoParquet shards for three vintages."""
    functions.configure_geospatial_environment()

    import os
    import shutil
    import tempfile
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyogrio
    import shapely
    from pyproj import CRS
    from tqdm import tqdm

    years = (2000, 2010, 2020)
    chunk_size = 100_000
    input_paths = {year: project.paths[f"wui_by_year_{year}"] for year in years}
    output_dir = project.paths["wui_singlepart_dataset"]
    summary_path = project.paths["wui_singlepart_summary"]
    overwrite = bool(project.config.get("overwrite_existing", False))

    if functions.should_skip(overwrite, [output_dir, summary_path]):
        functions.skip_task("build_wui_singlepart_dataset", output_dir, summary_path)
        return {"status": "skipped", "dataset": output_dir, "summary": summary_path}

    missing = [path for path in input_paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty SILVIS WUI inputs: {missing}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory exists without a complete summary: {output_dir}. "
            "Set overwrite_existing=True to rebuild it."
        )

    source_fields = []
    field_kinds = {}
    source_crs = None
    metadata_by_year = {}
    reserved_fields = {
        "year",
        "wui_source_row_number",
        "wui_single_part_number",
        "wui_single_part_id",
    }
    for year, path in input_paths.items():
        metadata = pyogrio.read_info(path)
        fields = list(metadata["fields"])
        if "BLK20" not in fields:
            raise ValueError(f"Input lacks required BLK20 field: {path}")
        collisions = sorted(reserved_fields.intersection(fields))
        if collisions:
            raise ValueError(f"Input fields conflict with generated metadata fields {collisions}: {path}")
        current_crs = metadata["crs"]
        if current_crs is None:
            raise ValueError(f"Input has no CRS: {path}")
        if source_crs is None:
            source_crs = CRS.from_user_input(current_crs)
        elif CRS.from_user_input(current_crs) != source_crs:
            raise ValueError(f"WUI input CRS differs across vintages: {path}")
        metadata_by_year[year] = metadata
        for field in fields:
            if field not in source_fields:
                source_fields.append(field)
        for field, dtype in zip(fields, metadata["dtypes"]):
            kind = np.dtype(dtype).kind
            field_kinds.setdefault(field, set()).add(kind)

    output_fields = [
        *source_fields,
        "year",
        "wui_source_row_number",
        "wui_single_part_number",
        "wui_single_part_id",
        "geometry",
    ]
    functions.ensure_dir(output_dir.parent)
    functions.ensure_dir(summary_path.parent)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".wui_singlepart.", dir=output_dir.parent))
    temporary_summary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    summary_rows = []

    try:
        for year, path in input_paths.items():
            metadata = metadata_by_year[year]
            geometry_column = metadata["geometry_name"] or "wkb_geometry"
            crs_text = str(metadata["crs"])
            try:
                feature_count = int(metadata.get("features"))
            except (TypeError, ValueError):
                feature_count = -1
            total_batches = (
                (feature_count + chunk_size - 1) // chunk_size if feature_count >= 0 else None
            )
            year_dir = temporary_dir / f"year_{year}"
            year_dir.mkdir(parents=True, exist_ok=True)
            source_rows = 0
            single_part_rows = 0
            dropped_nonpolygon_components = 0
            part_file_number = 0
            row_number_offset = 0

            tqdm.write(f"[progress] WUI {year}: reading {path}")

            with pyogrio.open_arrow(
                path,
                use_pyarrow=True,
                batch_size=chunk_size,
            ) as (_, reader):
                with tqdm(
                    total=total_batches,
                    desc=f"WUI {year}",
                    unit="batch",
                    dynamic_ncols=True,
                ) as progress:
                    def advance_progress():
                        progress.set_postfix(
                            rows=f"{source_rows:,}",
                            parts=f"{single_part_rows:,}",
                            shards=part_file_number,
                            refresh=False,
                        )
                        progress.update(1)

                    for batch in reader:
                        attributes = batch.to_pandas()
                        if geometry_column not in attributes:
                            raise ValueError(f"Could not find geometry column {geometry_column!r} in {path}")
                        wkb = attributes.pop(geometry_column)
                        batch_rows = len(attributes)
                        if batch_rows == 0:
                            advance_progress()
                            continue
                        source_rows += batch_rows
                        row_numbers = np.arange(
                            row_number_offset + 1,
                            row_number_offset + batch_rows + 1,
                            dtype=np.int64,
                        )
                        row_number_offset += batch_rows
                        attributes["wui_source_row_number"] = row_numbers
                        geometries = shapely.from_wkb(wkb.to_numpy())
                        frame = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs_text)
                        invalid = ~frame.geometry.is_valid
                        if invalid.any():
                            frame.loc[invalid, "geometry"] = shapely.make_valid(
                                frame.loc[invalid, "geometry"].to_numpy()
                            )
                        frame = frame.loc[~frame.geometry.isna() & ~frame.geometry.is_empty].copy()
                        if frame.empty:
                            advance_progress()
                            continue

                        parts = frame.explode(index_parts=False).reset_index(drop=True)
                        polygon_mask = parts.geom_type.eq("Polygon")
                        dropped_nonpolygon_components += int((~polygon_mask).sum())
                        parts = parts.loc[polygon_mask].copy()
                        if parts.empty:
                            advance_progress()
                            continue
                        parts["wui_single_part_number"] = (
                            parts.groupby("wui_source_row_number", sort=False, observed=True).cumcount() + 1
                        ).astype("int32")

                        for field in source_fields:
                            if field not in parts:
                                parts[field] = pd.NA
                            kinds = field_kinds.get(field, set())
                            if kinds.intersection({"O", "S", "U"}):
                                parts[field] = parts[field].astype("string")
                            elif "f" in kinds:
                                parts[field] = pd.to_numeric(parts[field], errors="coerce").astype("Float64")
                            elif kinds.intersection({"i", "u", "b"}):
                                parts[field] = pd.to_numeric(parts[field], errors="coerce").astype("Int64")
                        parts["year"] = year
                        parts["wui_single_part_id"] = (
                            str(year)
                            + "_"
                            + parts["wui_source_row_number"].astype("string")
                            + "_"
                            + parts["wui_single_part_number"].astype("string")
                        )
                        parts = parts[output_fields]
                        out_part = year_dir / f"part-{part_file_number:06d}.parquet"
                        tqdm.write(
                            f"[progress] WUI {year}: writing shard {part_file_number + 1} "
                            f"({len(parts):,} single parts)",
                        )
                        parts.to_parquet(out_part, index=False, compression="zstd")
                        part_file_number += 1
                        single_part_rows += len(parts)
                        advance_progress()

            if source_rows == 0 or single_part_rows == 0:
                raise RuntimeError(f"No WUI polygon records processed for vintage {year}: {path}")
            tqdm.write(
                f"[progress] WUI {year}: complete; source rows={source_rows:,}; "
                f"single parts={single_part_rows:,}; shards={part_file_number}"
            )
            summary_rows.append(
                {
                    "year": year,
                    "source_path": str(path),
                    "source_rows": source_rows,
                    "single_part_rows": single_part_rows,
                    "extra_nonpolygon_components_dropped": dropped_nonpolygon_components,
                    "part_files": part_file_number,
                    "source_fields": ",".join(source_fields),
                    "crs": crs_text,
                }
            )

        functions.write_csv(
            summary_rows,
            temporary_summary,
            fieldnames=(
                "year",
                "source_path",
                "source_rows",
                "single_part_rows",
                "extra_nonpolygon_components_dropped",
                "part_files",
                "source_fields",
                "crs",
            ),
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
        temporary_summary.replace(summary_path)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        temporary_summary.unlink(missing_ok=True)
        raise

    print(f"[output] Single-part WUI dataset: {output_dir}")
    print(f"[output] Vintage summary: {summary_path}")
    return {"status": "built", "dataset": output_dir, "summary": summary_path}


def build_fire_wui_centroid_paths(project):
    """Write every fire-centroid/WUI-edge pair within configured miles as a line feature."""
    functions.configure_geospatial_environment()

    import os
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyogrio
    import shapely
    from tqdm import tqdm

    distance_miles = float(project.config.get("fire_wui_distance_miles", 3.0))
    if not np.isfinite(distance_miles) or distance_miles <= 0:
        raise ValueError(f"fire_wui_distance_miles must be positive and finite; got {distance_miles}")
    distance_m = distance_miles * 1609.344
    distance_label = f"{distance_miles:g}mi"
    source_points = functions.find_fire_point_source(project.paths["mtbs_fod_points_candidates"])
    singlepart_root = project.paths["wui_singlepart_dataset"]
    output_dir = project.paths["fire_wui_path_dataset"] / f"distance_{distance_label}"
    summary_path = project.paths["fire_wui_path_summary"].with_name(
        f"fire_wui_centroid_paths_{distance_label}_summary.csv"
    )
    overwrite = bool(project.config.get("overwrite_existing", False))

    if functions.should_skip(overwrite, [output_dir, summary_path]):
        functions.skip_task("build_fire_wui_centroid_paths", output_dir, summary_path)
        return {"status": "skipped", "dataset": output_dir, "summary": summary_path}
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory exists without a complete summary: {output_dir}. "
            "Set overwrite_existing=True to rebuild it."
        )
    if not singlepart_root.is_dir():
        raise FileNotFoundError(
            f"Missing single-part WUI dataset: {singlepart_root}. "
            "Run build_wui_singlepart_dataset first."
        )

    point_info = pyogrio.read_info(source_points)
    point_fields = list(point_info["fields"])
    year_field = next(
        (field for field in point_fields if field.casefold() in {"year", "fire_year"}),
        None,
    )
    ignition_date_field = next(
        (field for field in point_fields if field.casefold() in {"ig_date", "ignition_date"}),
        None,
    )
    if year_field is None and ignition_date_field is None:
        raise ValueError(
            f"Fire point source lacks YEAR/FIRE_YEAR or IG_DATE/IGNITION_DATE field: {source_points}. "
            f"Available fields: {point_fields}"
        )
    point_crs = point_info.get("crs")
    if point_crs is None:
        raise ValueError(f"Fire point source has no CRS: {source_points}")
    tqdm.write(f"[progress] Fire points: reading {source_points}")
    year_source_field = year_field or ignition_date_field
    tqdm.write(f"[progress] Fire year source field: {year_source_field}")
    fires = gpd.read_file(source_points, use_arrow=True)
    if fires.empty:
        raise RuntimeError(f"Fire point source contains no records: {source_points}")
    fires["_fire_source_row_number"] = np.arange(1, len(fires) + 1, dtype=np.int64)
    if year_field is not None:
        fire_years = pd.to_numeric(fires[year_field], errors="coerce")
    else:
        fire_years = pd.to_datetime(fires[ignition_date_field], errors="coerce").dt.year
    fires["_fire_year_numeric"] = fire_years.astype("Int64")
    no_year_count = int(fires["_fire_year_numeric"].isna().sum())
    fires = fires.loc[fires["_fire_year_numeric"].notna() & fires.geometry.notna() & ~fires.geometry.is_empty].copy()
    fires["_fire_year_numeric"] = fires["_fire_year_numeric"].astype("int32")
    fires = fires.to_crs(5070)
    fires = fires.rename(columns={"_fire_year_numeric": "_fire_year"})
    fires["_fire_wui_vintage"] = np.select(
        [fires["_fire_year"].between(2000, 2009), fires["_fire_year"].between(2010, 2019), fires["_fire_year"].ge(2020)],
        [2000, 2010, 2020],
        default=0,
    ).astype("int16")
    excluded_outside_period = int((fires["_fire_wui_vintage"] == 0).sum())
    fires = fires.loc[fires["_fire_wui_vintage"] > 0].copy()
    if fires.empty:
        raise RuntimeError("No MTBS fire points fall in 2000 or later, required for WUI vintage matching")
    compute = functions.inspect_compute_capacity(memory_per_worker_gib=8.0, max_workers=32)
    workers = int(compute["workers"])

    source_fields = [field for field in point_fields if field in fires.columns]
    fire_renames = {field: f"fire_{field}" for field in source_fields}
    fire_attributes = fires.drop(
        columns=["geometry", "_fire_source_row_number", "_fire_year", "_fire_wui_vintage"]
    ).rename(columns=fire_renames)
    fires = fires.reset_index(drop=True)
    fire_attributes = fire_attributes.reset_index(drop=True)
    fire_attributes["fire_source_row_number"] = fires["_fire_source_row_number"].to_numpy()
    fire_attributes["fire_year"] = fires["_fire_year"].to_numpy(dtype="int32")
    fire_attributes["wui_vintage_year"] = fires["_fire_wui_vintage"].to_numpy(dtype="int16")
    fire_geometries = fires.geometry.to_numpy()
    fire_attributes["fire_centroid_x"] = shapely.get_x(fire_geometries)
    fire_attributes["fire_centroid_y"] = shapely.get_y(fire_geometries)

    functions.ensure_dir(output_dir.parent)
    functions.ensure_dir(summary_path.parent)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".fire_wui_paths.", dir=output_dir.parent))
    temporary_summary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    summary_rows = []
    matched_fire_rows = set()
    total_paths = 0

    def process_shard(task):
        wui_year, input_path, output_path = task
        wui = gpd.read_parquet(input_path)
        required_wui = {"wui_single_part_id", "wui_source_row_number", "wui_single_part_number", "year"}
        missing_wui = sorted(required_wui.difference(wui.columns))
        if missing_wui:
            raise ValueError(f"WUI single-part shard lacks required fields {missing_wui}: {input_path}")
        if not wui.empty and not wui["year"].eq(wui_year).all():
            raise ValueError(f"WUI vintage mismatch in shard expected={wui_year}: {input_path}")
        wui = wui.to_crs(5070).reset_index(drop=True)
        boundaries = shapely.boundary(wui.geometry.to_numpy())
        tree = shapely.STRtree(boundaries)
        fire_idx = np.flatnonzero(fires["_fire_wui_vintage"].to_numpy() == wui_year)
        fire_idx = fire_idx[np.argsort(fires.iloc[fire_idx]["_fire_year"].to_numpy(), kind="stable")]
        path_count = 0
        shard_matched_fire_rows = set()
        for start in range(0, len(fire_idx), 2000):
            fire_chunk_idx = fire_idx[start : start + 2000]
            if len(fire_chunk_idx) == 0:
                continue
            candidate_pairs = tree.query(
                fire_geometries[fire_chunk_idx],
                predicate="dwithin",
                distance=distance_m,
            )
            if candidate_pairs.size == 0:
                continue
            local_fire_idx, wui_idx = candidate_pairs
            source_fire_idx = fire_chunk_idx[local_fire_idx]
            line_geometries = shapely.shortest_line(
                fire_geometries[source_fire_idx],
                boundaries[wui_idx],
            )
            distances_m = shapely.length(line_geometries)
            keep = distances_m <= distance_m + 1e-7
            if not keep.any():
                continue
            source_fire_idx = source_fire_idx[keep]
            wui_idx = wui_idx[keep]
            line_geometries = line_geometries[keep]
            distances_m = distances_m[keep]
            fire_rows = fire_attributes.iloc[source_fire_idx].reset_index(drop=True)
            wui_rows = wui.drop(columns=wui.geometry.name).iloc[wui_idx].reset_index(drop=True)
            wui_column_names = {
                "wui_single_part_id": "wui_single_part_id",
                "wui_source_row_number": "wui_source_row_number",
                "wui_single_part_number": "wui_single_part_number",
                "year": "wui_vintage_year",
            }
            wui_rows = wui_rows.rename(
                columns={
                    column: wui_column_names.get(column, f"wui_source_{column}")
                    for column in wui_rows.columns
                }
            )
            edge_coords = shapely.get_point(line_geometries, -1)
            result = pd.concat([fire_rows, wui_rows], axis=1)
            result["fire_to_wui_edge_distance_m"] = distances_m
            result["fire_to_wui_edge_distance_miles"] = distances_m / 1609.344
            result["distance_limit_miles"] = distance_miles
            result["wui_edge_x"] = shapely.get_x(edge_coords)
            result["wui_edge_y"] = shapely.get_y(edge_coords)
            result["path_id"] = (
                result["fire_source_row_number"].astype("string")
                + "_"
                + result["wui_single_part_id"].astype("string")
            )
            path_gdf = gpd.GeoDataFrame(result, geometry=line_geometries, crs=5070)
            path_gdf.to_parquet(output_path, index=False, compression="zstd")
            path_count += len(path_gdf)
            shard_matched_fire_rows.update(source_fire_idx.tolist())
        return {
            "wui_year": wui_year,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "wui_single_parts": len(wui),
            "path_rows": path_count,
            "matched_fire_source_rows": shard_matched_fire_rows,
        }

    try:
        tasks_to_run = []
        for wui_year in (2000, 2010, 2020):
            year_dir = singlepart_root / f"year_{wui_year}"
            shard_paths = sorted(year_dir.glob("part-*.parquet"))
            if not shard_paths:
                raise FileNotFoundError(f"No single-part WUI parquet shards found: {year_dir}")
            year_output = temporary_dir / f"wui_vintage_{wui_year}"
            year_output.mkdir(parents=True, exist_ok=True)
            for shard_index, input_path in enumerate(shard_paths):
                output_path = year_output / f"paths_part-{shard_index:06d}.parquet"
                tasks_to_run.append((wui_year, input_path, output_path))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_shard, task) for task in tasks_to_run]
            with tqdm(total=len(futures), desc=f"Fire–WUI {distance_label}", unit="shard", dynamic_ncols=True) as progress:
                for future in as_completed(futures):
                    result = future.result()
                    total_paths += result["path_rows"]
                    matched_fire_rows.update(result["matched_fire_source_rows"])
                    summary_rows.append({key: value for key, value in result.items() if key != "matched_fire_source_rows"})
                    tqdm.write(
                        f"[progress] WUI {result['wui_year']}: paths={result['path_rows']:,}; "
                        f"source parts={result['wui_single_parts']:,}"
                    )
                    progress.update(1)

        if total_paths == 0:
            raise RuntimeError(f"No fire-centroid/WUI-edge pairs found within {distance_miles:g} miles")
        summary_rows.sort(key=lambda row: (row["wui_year"], row["input_path"]))
        for row in summary_rows:
            row["distance_miles"] = distance_miles
            row["distance_m"] = distance_m
            row["crs_epsg"] = 5070
            row["point_source"] = str(source_points)
        summary_rows.append(
            {
                "wui_year": "ALL",
                "input_path": "",
                "output_path": str(temporary_dir),
                "wui_single_parts": "",
                "path_rows": total_paths,
                "distance_miles": distance_miles,
                "distance_m": distance_m,
                "crs_epsg": 5070,
                "point_source": str(source_points),
                "fire_points_matched": len(matched_fire_rows),
                "fire_points_without_year_or_geometry": no_year_count,
                "fire_points_outside_vintage_period": excluded_outside_period,
            }
        )
        functions.write_csv(
            summary_rows,
            temporary_summary,
            fieldnames=(
                "wui_year",
                "input_path",
                "output_path",
                "wui_single_parts",
                "path_rows",
                "distance_miles",
                "distance_m",
                "crs_epsg",
                "point_source",
                "fire_points_matched",
                "fire_points_without_year_or_geometry",
                "fire_points_outside_vintage_period",
            ),
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
        temporary_summary.replace(summary_path)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        temporary_summary.unlink(missing_ok=True)
        raise

    print(f"[output] Fire–WUI path dataset: {output_dir}")
    print(f"[output] Path summary: {summary_path}")
    print(f"[result] paths={total_paths:,}; matched fire-point records={len(matched_fire_rows):,}; distance={distance_miles:g} miles")
    return {
        "status": "built",
        "dataset": output_dir,
        "summary": summary_path,
        "path_rows": total_paths,
        "matched_fire_points": len(matched_fire_rows),
        "distance_miles": distance_miles,
    }


def build_fire_wui_centroid_path_distance_samples(project):
    """Build resumable fire-to-WUI path datasets for distances from 1 to 10 miles."""
    distances_miles = tuple(round(1.0 + 0.5 * index, 1) for index in range(19))
    configured_distance = project.config.get("fire_wui_distance_miles", 3.0)
    results = []
    try:
        for index, distance_miles in enumerate(distances_miles, start=1):
            print(
                f"[progress] Distance sample {index}/{len(distances_miles)}: "
                f"{distance_miles:g} miles"
            )
            project.config["fire_wui_distance_miles"] = distance_miles
            results.append(build_fire_wui_centroid_paths(project))
    finally:
        project.config["fire_wui_distance_miles"] = configured_distance

    built = sum(result["status"] == "built" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    print(f"[result] distance samples built={built}; skipped={skipped}")
    return {
        "status": "built" if built else "skipped",
        "distances_miles": distances_miles,
        "built": built,
        "skipped": skipped,
        "samples": results,
    }
