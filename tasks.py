"""WUI dataset tasks.

Contents:
- build_wui_singlepart_dataset: combine 2000, 2010, and 2020 WUI blocks as single-part records
- build_fire_wui_geometry_characteristics: create stand-alone MTBS point, fire-perimeter, and WUI polygon characteristic tables
- build_fire_wui_centroid_paths: link MTBS fire centroids to WUI single-part edges within a configurable radius
- build_fire_wui_centroid_path_distance_samples: build path datasets for 1 to 10 miles
- fire_filteration: run fire-centroid-to-WUI-edge filtering at 1 to 10 miles in 0.5-mile steps
- build_wui_annual_mtbs_outcomes: calculate annual burn, area, and severity for WUI single parts
- prepare_facts_treatments: retain completed FACTS prevention polygons and treatment-date provenance
- build_fire_wui_treatment_definitions: create treatment windows, path/angular exposure, thickness, and fire-perimeter controls
- finalize_fire_wui_treatment_definitions: validate treatment array manifests and atomically assemble worker outputs
- landfire_controls: summarize LF2020 terrain rasters for WUI single parts, MTBS fire points, and MTBS fire areas
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


def build_fire_wui_geometry_characteristics(project):
    """Write descriptive characteristics for MTBS points/perimeters and WUI single-part polygons."""
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

    analysis_start_year = 2000
    analysis_end_year = 2025
    singlepart_root = project.paths["wui_singlepart_dataset"]
    point_path = functions.find_fire_point_source(project.paths["mtbs_fod_points_candidates"])
    perimeter_path = functions.find_spatial_source(
        project.paths["mtbs_perimeter_candidates"],
        project.paths.get(
            "mtbs_search_roots",
            (project.paths["raw"] / "MTBS", project.paths["data_archive"] / "raw" / "MTBS"),
        ),
        "mtbs_perims_DD.shp",
        require_attribute_table=True,
    )
    output_dir = project.paths["fire_wui_geometry_characteristics_dataset"]
    summary_path = project.paths["fire_wui_geometry_characteristics_summary"]
    overwrite = bool(project.config.get("overwrite_existing", False))

    if not singlepart_root.is_dir():
        raise FileNotFoundError(
            f"Missing WUI single-part dataset: {singlepart_root}. Run build_wui_singlepart_dataset first."
        )
    wui_inputs = []
    for vintage in (2000, 2010, 2020):
        shard_paths = sorted((singlepart_root / f"year_{vintage}").glob("part-*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(f"No WUI single-part Parquet shards found for vintage {vintage}")
        for shard_path in shard_paths:
            wui_inputs.append((vintage, shard_path))

    point_output_rel = Path("fire_points/mtbs_fod_points_2000_2025.parquet")
    perimeter_output_rel = Path("fire_perimeters/mtbs_perimeter_events_2000_2025.parquet")
    wui_output_rels = [
        Path("wui_polygons") / f"year_{vintage}" / shard_path.name
        for vintage, shard_path in wui_inputs
    ]
    expected_outputs = [point_output_rel, perimeter_output_rel, *wui_output_rels]
    complete = (
        summary_path.is_file()
        and summary_path.stat().st_size > 0
        and all(
            (output_dir / relative).is_file() and (output_dir / relative).stat().st_size > 0
            for relative in expected_outputs
        )
    )
    if complete and not overwrite:
        functions.skip_task(
            "build_fire_wui_geometry_characteristics",
            output_dir,
            summary_path,
        )
        return {"status": "skipped", "dataset": output_dir, "summary": summary_path}

    compute = functions.inspect_compute_capacity(memory_per_worker_gib=6.0, max_workers=8)
    workers = int(compute["workers"])
    print(
        f"[input] MTBS Fire Occurrence points: {point_path}\n"
        f"[input] MTBS perimeters: {perimeter_path}\n"
        f"[input] WUI single-part shards: {len(wui_inputs):,}\n"
        f"[capacity] CPUs={compute['cpus']}; total_ram_gib={compute['total_ram_gib']:.1f}; "
        f"available_ram_gib={compute['available_ram_gib']:.1f}; workers={workers}",
        flush=True,
    )

    functions.ensure_dir(output_dir.parent)
    functions.ensure_dir(summary_path.parent)
    stage_root = Path(tempfile.mkdtemp(prefix=".fire_wui_geometry_characteristics.", dir=output_dir.parent))
    staged_dataset = stage_root / "dataset"
    staged_summary = stage_root / "summary.csv"
    functions.ensure_dir(staged_dataset)
    summary_rows = []

    try:
        point_info = pyogrio.read_info(point_path)
        point_fields = list(point_info.get("fields", []))
        point_field_map = {field.casefold(): field for field in point_fields}
        point_id_field = next(
            (point_field_map[key] for key in ("event_id", "fire_id") if key in point_field_map),
            None,
        )
        point_year_field = next(
            (point_field_map[key] for key in ("year", "fire_year") if key in point_field_map),
            None,
        )
        point_date_field = next(
            (point_field_map[key] for key in ("ig_date", "ignition_date") if key in point_field_map),
            None,
        )
        if point_id_field is None:
            raise ValueError(f"MTBS Fire Occurrence point layer lacks EVENT_ID/FIRE_ID: {point_fields}")
        if point_year_field is None and point_date_field is None:
            raise ValueError(f"MTBS point layer lacks YEAR/FIRE_YEAR or IG_DATE/IGNITION_DATE: {point_fields}")

        print("[progress] Reading MTBS Fire Occurrence points", flush=True)
        points = gpd.read_file(point_path, use_arrow=True)
        if points.crs is None:
            raise ValueError(f"MTBS Fire Occurrence points have no CRS: {point_path}")
        point_source_crs = str(points.crs)
        source_point_count = len(points)
        points["fire_source_row_number"] = np.arange(1, source_point_count + 1, dtype=np.int64)
        if point_year_field is not None:
            point_years = pd.to_numeric(points[point_year_field], errors="coerce")
        else:
            point_years = pd.to_datetime(points[point_date_field], errors="coerce").dt.year
        points["fire_year"] = point_years
        points["fire_event_id_normalized"] = points[point_id_field].map(functions.normalize_identifier)
        point_geometries = points.geometry.to_numpy()
        point_geometry_ok = np.asarray(
            [geometry is not None and not shapely.is_empty(geometry) for geometry in point_geometries],
            dtype=bool,
        )
        in_year_window = points["fire_year"].between(analysis_start_year, analysis_end_year).fillna(False).to_numpy()
        keep_points = point_geometry_ok & in_year_window
        excluded_point_count = int((~keep_points).sum())
        points = points.loc[keep_points].copy().reset_index(drop=True)
        if points.empty:
            raise ValueError(f"No valid MTBS Fire Occurrence points fall in {analysis_start_year}-{analysis_end_year}")
        points["fire_year"] = points["fire_year"].astype("int16")
        points = points.to_crs(5070)
        point_geometries = points.geometry.to_numpy()
        points["fire_anchor_type"] = "mtbs_centroid"
        points["fire_point_interpretation"] = "MTBS Fire Occurrence perimeter centroid; not verified ignition location"
        points["fire_point_x_m"] = shapely.get_x(point_geometries)
        points["fire_point_y_m"] = shapely.get_y(point_geometries)
        lonlat_geometries = points.to_crs(4326).geometry.to_numpy()
        points["fire_point_longitude"] = shapely.get_x(lonlat_geometries)
        points["fire_point_latitude"] = shapely.get_y(lonlat_geometries)
        point_event_counts = (
            points.dropna(subset=["fire_event_id_normalized"])
            .groupby("fire_event_id_normalized", sort=False)
            .size()
            .to_dict()
        )
        point_event_years = (
            points.dropna(subset=["fire_event_id_normalized"])
            .groupby("fire_event_id_normalized", sort=False)["fire_year"]
            .first()
            .to_dict()
        )
        point_event_ids = set(point_event_counts)
        point_output = staged_dataset / point_output_rel
        functions.ensure_dir(point_output.parent)
        points.drop(columns=[points.geometry.name]).to_parquet(
            point_output,
            index=False,
            compression="zstd",
        )
        summary_rows.append({
            "layer": "fire_points",
            "vintage_year": "all",
            "source_path": str(point_path),
            "output_path": str(output_dir / point_output_rel),
            "source_crs": point_source_crs,
            "metric_crs": "EPSG:5070",
            "source_rows": source_point_count,
            "output_rows": len(points),
            "excluded_rows": excluded_point_count,
            "invalid_geometry_count": int((~point_geometry_ok).sum()),
            "analysis_year_start": analysis_start_year,
            "analysis_year_end": analysis_end_year,
            "notes": "Points labeled MTBS centroids; source does not establish true ignition locations.",
        })

        perimeter_info = pyogrio.read_info(perimeter_path)
        perimeter_fields = list(perimeter_info.get("fields", []))
        perimeter_field_map = {field.casefold(): field for field in perimeter_fields}
        perimeter_id_field = next(
            (perimeter_field_map[key] for key in ("event_id", "fire_id") if key in perimeter_field_map),
            None,
        )
        perimeter_year_field = next(
            (perimeter_field_map[key] for key in ("year", "fire_year") if key in perimeter_field_map),
            None,
        )
        if perimeter_id_field is None:
            raise ValueError(f"MTBS perimeter layer lacks EVENT_ID/FIRE_ID: {perimeter_fields}")
        perimeter_columns = [perimeter_id_field]
        if perimeter_year_field is not None and perimeter_year_field not in perimeter_columns:
            perimeter_columns.append(perimeter_year_field)
        print(
            f"[progress] Reading MTBS perimeter features; event field={perimeter_id_field}; "
            f"year field={perimeter_year_field or 'matched from point records'}",
            flush=True,
        )
        perimeters = gpd.read_file(perimeter_path, columns=perimeter_columns, use_arrow=True)
        if perimeters.crs is None:
            raise ValueError(f"MTBS perimeter layer has no CRS: {perimeter_path}")
        perimeter_source_crs = str(perimeters.crs)
        perimeters = perimeters.to_crs(5070)
        source_perimeter_count = len(perimeters)
        perimeters["fire_event_id_normalized"] = perimeters[perimeter_id_field].map(functions.normalize_identifier)
        if perimeter_year_field is not None:
            perimeters["fire_year"] = pd.to_numeric(perimeters[perimeter_year_field], errors="coerce")
        else:
            perimeters["fire_year"] = np.nan
        perimeters["fire_year"] = perimeters["fire_year"].fillna(
            perimeters["fire_event_id_normalized"].map(point_event_years)
        )
        perimeters = perimeters.loc[
            perimeters["fire_event_id_normalized"].notna()
            & perimeters["fire_year"].between(analysis_start_year, analysis_end_year).fillna(False)
            & perimeters.geometry.notna()
            & ~perimeters.geometry.is_empty
        ].copy()
        unmatched_event_count = int(
            (~perimeters["fire_event_id_normalized"].isin(point_event_ids)).groupby(
                perimeters["fire_event_id_normalized"]
            ).any().sum()
        )
        perimeter_records = []
        perimeter_groups = perimeters.groupby("fire_event_id_normalized", sort=False)
        with tqdm(
            total=perimeter_groups.ngroups,
            desc="Dissolving MTBS fire perimeters",
            unit="event",
            dynamic_ncols=True,
        ) as progress:
            for fire_id, group in perimeter_groups:
                geometries = group.geometry.to_numpy()
                valid = np.asarray(shapely.is_valid(geometries), dtype=bool)
                repaired_count = int((~valid).sum())
                if repaired_count:
                    geometries = geometries.copy()
                    geometries[~valid] = shapely.make_valid(geometries[~valid])
                dissolved = shapely.union_all(geometries)
                if dissolved is None or shapely.is_empty(dissolved):
                    progress.update(1)
                    continue
                geometry_metrics = functions.polygon_geometry_characteristics(
                    [dissolved],
                    prefix="fire_perimeter",
                )
                year_values = pd.to_numeric(group["fire_year"], errors="coerce").dropna()
                perimeter_records.append({
                    "fire_event_id_normalized": fire_id,
                    "fire_event_id_source": str(group[perimeter_id_field].iloc[0]),
                    "fire_year": int(year_values.iloc[0]) if len(year_values) else None,
                    "fire_point_record_count": int(point_event_counts.get(fire_id, 0)),
                    "perimeter_source_feature_count": len(group),
                    "perimeter_source_invalid_geometry_count": repaired_count,
                    **{key: value[0] for key, value in geometry_metrics.items()},
                })
                progress.update(1)
        perimeter_frame = pd.DataFrame(perimeter_records)
        perimeter_output = staged_dataset / perimeter_output_rel
        functions.ensure_dir(perimeter_output.parent)
        perimeter_frame.to_parquet(perimeter_output, index=False, compression="zstd")
        summary_rows.append({
            "layer": "fire_perimeters",
            "vintage_year": "all",
            "source_path": str(perimeter_path),
            "output_path": str(output_dir / perimeter_output_rel),
            "source_crs": perimeter_source_crs,
            "metric_crs": "EPSG:5070",
            "source_rows": source_perimeter_count,
            "output_rows": len(perimeter_frame),
            "excluded_rows": source_perimeter_count - len(perimeters),
            "invalid_geometry_count": int(perimeter_frame.get(
                "perimeter_source_invalid_geometry_count", pd.Series(dtype="int64")
            ).sum()),
            "analysis_year_start": analysis_start_year,
            "analysis_year_end": analysis_end_year,
            "notes": f"Dissolved by normalized event ID; events without a matching point: {unmatched_event_count}.",
        })
        del perimeters

        def process_wui_shard(task):
            vintage, input_path = task
            frame = gpd.read_parquet(input_path)
            if frame.crs is None:
                raise ValueError(f"WUI single-part shard has no CRS: {input_path}")
            source_crs = str(frame.crs)
            required = {"wui_single_part_id", "wui_source_row_number", "wui_single_part_number", "year"}
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"WUI shard lacks required identifiers {missing}: {input_path}")
            if not frame.empty and not frame["year"].eq(vintage).all():
                raise ValueError(f"WUI vintage mismatch in shard expected={vintage}: {input_path}")
            frame = frame.to_crs(5070)
            metrics = functions.polygon_geometry_characteristics(
                frame.geometry.to_numpy(),
                prefix="wui",
            )
            for column, values in metrics.items():
                frame[column] = values
            frame["wui_vintage_year"] = np.int16(vintage)
            frame = frame.rename(columns={"year": "wui_source_year"})
            output_relative = Path("wui_polygons") / f"year_{vintage}" / input_path.name
            shard_output = staged_dataset / output_relative
            functions.ensure_dir(shard_output.parent)
            frame.drop(columns=[frame.geometry.name]).to_parquet(
                shard_output,
                index=False,
                compression="zstd",
            )
            invalid_count = int((~np.asarray(metrics["wui_geometry_is_valid"], dtype=bool)).sum())
            return {
                "layer": "wui_polygons",
                "vintage_year": vintage,
                "source_path": str(input_path),
                "output_path": str(output_dir / output_relative),
                "source_crs": source_crs,
                "metric_crs": "EPSG:5070",
                "source_rows": len(frame),
                "output_rows": len(frame),
                "excluded_rows": 0,
                "invalid_geometry_count": invalid_count,
                "analysis_year_start": analysis_start_year,
                "analysis_year_end": analysis_end_year,
                "notes": "Source attributes and IDs retained; geometry summarized and omitted from compact output.",
            }

        print("[progress] Summarizing WUI polygon shards", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_wui_shard, item) for item in wui_inputs]
            with tqdm(total=len(futures), desc="WUI characteristics", unit="shard", dynamic_ncols=True) as progress:
                for future in as_completed(futures):
                    summary_rows.append(future.result())
                    progress.update(1)

        summary_fields = (
            "layer",
            "vintage_year",
            "source_path",
            "output_path",
            "source_crs",
            "metric_crs",
            "source_rows",
            "output_rows",
            "excluded_rows",
            "invalid_geometry_count",
            "analysis_year_start",
            "analysis_year_end",
            "notes",
        )
        functions.write_csv(summary_rows, staged_summary, fieldnames=summary_fields)

        backup_dataset = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.backup")
        backup_summary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.backup")
        if backup_dataset.exists():
            shutil.rmtree(backup_dataset)
        backup_dataset_exists = output_dir.exists()
        backup_summary_exists = summary_path.exists()
        if backup_dataset_exists:
            output_dir.replace(backup_dataset)
        if backup_summary_exists:
            summary_path.replace(backup_summary)
        try:
            staged_dataset.replace(output_dir)
            staged_summary.replace(summary_path)
        except BaseException:
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            if summary_path.exists():
                summary_path.unlink(missing_ok=True)
            if backup_dataset_exists and backup_dataset.exists():
                backup_dataset.replace(output_dir)
            if backup_summary_exists and backup_summary.exists():
                backup_summary.replace(summary_path)
            raise
        if backup_dataset.exists():
            shutil.rmtree(backup_dataset)
        if backup_summary.exists():
            backup_summary.unlink()
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    shutil.rmtree(stage_root, ignore_errors=True)

    layer_totals = pd.DataFrame(summary_rows).groupby("layer", dropna=False)["output_rows"].sum().to_dict()
    print(f"[output] Geometry characteristics: {output_dir}")
    print(f"[output] Summary: {summary_path}")
    print(f"[result] rows={layer_totals}", flush=True)
    return {"status": "built", "dataset": output_dir, "summary": summary_path, "rows": layer_totals}


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

    if not singlepart_root.is_dir():
        raise FileNotFoundError(
            f"Missing single-part WUI dataset: {singlepart_root}. "
            "Run build_wui_singlepart_dataset first."
        )

    expected_shards = []
    for wui_year in (2000, 2010, 2020):
        year_dir = singlepart_root / f"year_{wui_year}"
        shard_paths = sorted(year_dir.glob("part-*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(f"No single-part WUI parquet shards found: {year_dir}")
        for shard_index, input_path in enumerate(shard_paths):
            output_path = output_dir / f"wui_vintage_{wui_year}" / f"paths_part-{shard_index:06d}"
            expected_shards.append((wui_year, input_path, output_path))

    def existing_path_dataset_status():
        missing = []
        if not output_dir.is_dir():
            missing.append("path dataset directory")
        if not summary_path.is_file() or summary_path.stat().st_size == 0:
            missing.append("path summary")
            summary = None
        else:
            try:
                summary = pd.read_csv(summary_path)
            except Exception as exc:
                missing.append(f"unreadable path summary ({exc})")
                summary = None

        if summary is not None:
            required = {"wui_year", "input_path", "path_rows"}
            if not required.issubset(summary.columns):
                missing.append("path summary lacks shard index fields")
                summary = None

        for wui_year, input_path, shard_output in expected_shards:
            shard_key = f"{wui_year}:{input_path.name}"
            if not shard_output.is_dir():
                missing.append(shard_key)
                continue
            if summary is None:
                continue
            years = pd.to_numeric(summary["wui_year"], errors="coerce")
            input_names = summary["input_path"].fillna("").map(lambda value: Path(str(value)).name)
            rows = summary.loc[(years == wui_year) & (input_names == input_path.name), "path_rows"]
            if len(rows) != 1:
                missing.append(f"{shard_key}: summary row")
                continue
            path_rows = pd.to_numeric(rows, errors="coerce").iloc[0]
            parquet_parts = [
                path for path in shard_output.glob("*.parquet")
                if path.is_file() and path.stat().st_size > 0
            ]
            if pd.isna(path_rows) or (path_rows > 0 and not parquet_parts):
                missing.append(f"{shard_key}: parquet data")
        return not missing, missing

    complete, missing_outputs = existing_path_dataset_status()
    if complete and not overwrite:
        functions.skip_task("build_fire_wui_centroid_paths", output_dir, summary_path)
        return {"status": "skipped", "dataset": output_dir, "summary": summary_path}
    if not overwrite and (output_dir.exists() or summary_path.exists()):
        sample = "; ".join(missing_outputs[:8])
        remaining = len(missing_outputs) - min(len(missing_outputs), 8)
        suffix = f"; plus {remaining} more" if remaining else ""
        print(f"[rebuild] incomplete {distance_label} path outputs: {sample}{suffix}", flush=True)

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
        [
            fires["_fire_year"].between(2000, 2009),
            fires["_fire_year"].between(2010, 2019),
            fires["_fire_year"].between(2020, 2025),
        ],
        [2000, 2010, 2020],
        default=0,
    ).astype("int16")
    excluded_outside_period = int((fires["_fire_wui_vintage"] == 0).sum())
    fires = fires.loc[fires["_fire_wui_vintage"] > 0].copy()
    if fires.empty:
        raise RuntimeError("No MTBS fire points fall in 2000-2025, required for WUI vintage matching")
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
        part_file_number = 0
        shard_matched_fire_rows = set()
        output_path.mkdir(parents=True, exist_ok=True)
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
                "year": "wui_source_year",
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
            chunk_path = output_path / f"part-{part_file_number:06d}.parquet"
            path_gdf.to_parquet(chunk_path, index=False, compression="zstd")
            part_file_number += 1
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
        for wui_year, input_path, final_output_path in expected_shards:
            output_path = temporary_dir / f"wui_vintage_{wui_year}" / final_output_path.name
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


def fire_filteration(project):
    """Build fire-WUI edge-distance samples from 1 to 10 miles by 0.5 miles."""
    return build_fire_wui_centroid_path_distance_samples(project)


def build_wui_annual_mtbs_outcomes(project):
    """Build annual WUI single-part outcomes from MTBS burn boundaries and severity mosaics."""
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
    import rasterio
    import shapely
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds
    from rasterio.errors import WindowError
    from shapely.geometry import mapping
    from tqdm import tqdm

    years_by_vintage = {
        2000: tuple(range(2000, 2010)),
        2010: tuple(range(2010, 2020)),
        2020: tuple(range(2020, 2026)),
    }
    singlepart_root = project.paths["wui_singlepart_dataset"]
    output_dir = project.paths["wui_annual_mtbs_outcomes"]
    summary_path = project.paths["wui_annual_mtbs_outcomes_summary"]
    perimeter_candidates = project.paths["mtbs_perimeter_candidates"]
    perimeter_path = next(
        (Path(path) for path in perimeter_candidates if Path(path).is_file() and Path(path).stat().st_size > 0),
        None,
    )
    if perimeter_path is None:
        search_roots = project.paths.get(
            "mtbs_search_roots",
            (project.paths["raw"] / "MTBS", project.paths["data_archive"] / "raw" / "MTBS"),
        )
        discovered = sorted(
            {
                path.resolve()
                for root in search_roots
                if root.is_dir()
                for path in root.rglob("mtbs_perims_DD.shp")
                if path.is_file() and path.stat().st_size > 0
            },
            key=str,
        )
        if discovered:
            rank = lambda path: (
                int("archive" in {part.lower() for part in path.parts}),
                int(any(part.lower().startswith("snapshot") for part in path.parts)),
            )
            best_rank = min(rank(path) for path in discovered)
            preferred = [path for path in discovered if rank(path) == best_rank]
        else:
            preferred = []
        if len(preferred) == 1:
            perimeter_path = preferred[0]
        elif len(preferred) > 1:
            raise FileExistsError(
                "Multiple MTBS perimeter shapefiles found; select one canonical copy by placing it at "
                f"{perimeter_candidates[0]} or removing duplicate search results:\n  "
                + "\n  ".join(str(path) for path in preferred)
            )
    if perimeter_path is None:
        search_roots = project.paths.get(
            "mtbs_search_roots",
            (project.paths["raw"] / "MTBS", project.paths["data_archive"] / "raw" / "MTBS"),
        )
        raise FileNotFoundError(
            "No MTBS burned-area boundary shapefile found. Checked exact paths:\n  "
            + "\n  ".join(str(path) for path in perimeter_candidates)
            + "\nRecursive search roots:\n  "
            + "\n  ".join(str(path) for path in search_roots)
        )
    print(f"[input] MTBS burned-area boundaries: {perimeter_path}")

    overwrite = bool(project.config.get("overwrite_existing", False))
    if functions.should_skip(overwrite, [output_dir, summary_path]):
        functions.skip_task("build_wui_annual_mtbs_outcomes", output_dir, summary_path)
        return {"status": "skipped", "dataset": output_dir, "summary": summary_path}
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Outcome directory exists without complete summary: {output_dir}. "
            "Set overwrite_existing=True to rebuild it."
        )
    if not singlepart_root.is_dir():
        raise FileNotFoundError(
            f"Missing WUI single-part dataset: {singlepart_root}. Run build_wui_singlepart_dataset first."
        )

    raster_roots = tuple(
        Path(path)
        for path in project.paths.get(
            "mtbs_severity_raster_roots",
            (project.paths["mtbs_severity_raster_root"],),
        )
    )
    raster_paths = {}
    for year in (year for years in years_by_vintage.values() for year in years):
        filename = f"mtbs_CONUS_{year}.tif"
        candidates = tuple(root / str(year) / filename for root in raster_roots)
        raster_paths[year] = next(
            (path for path in candidates if path.is_file() and path.stat().st_size > 0),
            candidates[0],
        )
    missing_rasters = [str(path) for path in raster_paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing_rasters:
        raise FileNotFoundError(
            "Missing annual MTBS CONUS burn-severity mosaics for 2000-2025:\n  "
            + "\n  ".join(missing_rasters)
        )

    perimeter_info = pyogrio.read_info(perimeter_path)
    perimeter_year_field = next(
        (field for field in perimeter_info["fields"] if field.casefold() in {"year", "fire_year"}),
        None,
    )
    perimeter_ignition_date_field = next(
        (field for field in perimeter_info["fields"] if field.casefold() in {"ig_date", "ignition_date"}),
        None,
    )
    if perimeter_year_field is None and perimeter_ignition_date_field is None:
        raise ValueError(
            f"MTBS perimeter layer lacks YEAR/FIRE_YEAR or IG_DATE/IGNITION_DATE: {perimeter_path}; "
            f"fields={list(perimeter_info['fields'])}"
        )
    year_source_field = perimeter_year_field or perimeter_ignition_date_field
    year_source_label = "year field" if perimeter_year_field else "ignition-date field"
    tqdm.write(f"[progress] MTBS fire year source: {year_source_field} ({year_source_label})")
    tqdm.write(f"[progress] MTBS burned-area boundaries: reading {perimeter_path}")
    perimeters = gpd.read_file(perimeter_path, columns=[year_source_field], use_arrow=True)
    if perimeter_year_field:
        perimeters["_outcome_year"] = pd.to_numeric(perimeters[perimeter_year_field], errors="coerce")
    else:
        date_values = perimeters[perimeter_ignition_date_field].astype("string").str.strip()
        compact_dates = date_values.str.fullmatch(r"\d{8}", na=False)
        parsed_dates = pd.to_datetime(date_values.mask(compact_dates), errors="coerce")
        if compact_dates.any():
            parsed_dates.loc[compact_dates] = pd.to_datetime(
                date_values.loc[compact_dates], format="%Y%m%d", errors="coerce"
            )
        perimeters["_outcome_year"] = parsed_dates.dt.year

    geometry_types = set(perimeters.geometry.geom_type.dropna().unique())
    unsupported_geometry_types = geometry_types.difference({"Polygon", "MultiPolygon"})
    if not geometry_types or unsupported_geometry_types:
        raise ValueError(
            f"MTBS burned-area boundaries must contain polygon geometries; "
            f"found {sorted(geometry_types)} in {perimeter_path}"
        )
    invalid_year_count = int(perimeters["_outcome_year"].isna().sum())
    if invalid_year_count:
        tqdm.write(f"[warning] Dropping {invalid_year_count:,} perimeters with unparseable fire year")
    perimeters = perimeters.loc[perimeters["_outcome_year"].between(2000, 2025)].copy()
    if perimeters.empty:
        raise ValueError(
            f"No MTBS perimeter records with valid fire years 2000-2025 in {perimeter_path}"
        )
    tqdm.write(
        f"[progress] Retained {len(perimeters):,} polygon records; "
        f"fire years {int(perimeters['_outcome_year'].min())}-"
        f"{int(perimeters['_outcome_year'].max())}"
    )
    perimeters = perimeters.to_crs(5070)
    perimeter_parts_by_year = {}
    perimeter_trees_by_year = {}
    for year in range(2000, 2026):
        annual_geometries = perimeters.loc[perimeters["_outcome_year"].eq(year)].geometry.to_numpy()
        if len(annual_geometries):
            valid_mask = shapely.is_valid(annual_geometries)
            invalid_count = int((~valid_mask).sum())
            if invalid_count:
                tqdm.write(
                    f"[progress] MTBS {year}: repairing {invalid_count:,} invalid "
                    "perimeter geometries with make_valid"
                )
                annual_geometries = annual_geometries.copy()
                annual_geometries[~valid_mask] = shapely.make_valid(annual_geometries[~valid_mask])
                still_invalid = int((~shapely.is_valid(annual_geometries)).sum())
                if still_invalid:
                    raise ValueError(
                        f"MTBS {year}: {still_invalid:,} perimeter geometries remain invalid after make_valid"
                    )

            annual_parts = np.asarray(shapely.get_parts(annual_geometries), dtype=object)
            polygonal_mask = np.fromiter(
                (geometry.geom_type in {"Polygon", "MultiPolygon"} for geometry in annual_parts),
                dtype=bool,
                count=len(annual_parts),
            )
            discarded_parts = int((~polygonal_mask).sum())
            if discarded_parts:
                tqdm.write(
                    f"[warning] MTBS {year}: discarding {discarded_parts:,} non-polygonal "
                    "make_valid remnants"
                )
            annual_parts = annual_parts[polygonal_mask]
            if not len(annual_parts):
                raise ValueError(f"MTBS {year}: no polygonal perimeter geometry remains after repair")

            dissolved = shapely.union_all(annual_parts)
            parts = np.asarray(shapely.get_parts(dissolved), dtype=object)
            parts = parts[~shapely.is_empty(parts)]
        else:
            parts = np.empty(0, dtype=object)
        perimeter_parts_by_year[year] = parts
        perimeter_trees_by_year[year] = shapely.STRtree(parts) if len(parts) else None

    for year, path in raster_paths.items():
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_epsg() != 5070:
                raise ValueError(f"Expected EPSG:5070 MTBS severity raster; got {src.crs}: {path}")
            if src.count != 1:
                raise ValueError(f"Expected single-band MTBS severity raster; found {src.count}: {path}")
    compute = functions.inspect_compute_capacity(memory_per_worker_gib=8.0, max_workers=32)
    workers = int(compute["workers"])
    functions.ensure_dir(output_dir.parent)
    functions.ensure_dir(summary_path.parent)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".wui_mtbs_outcomes.", dir=output_dir.parent))
    temporary_summary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")

    def severity_counts(src, geometry):
        bounds_window = from_bounds(*geometry.bounds, transform=src.transform)
        bounds_window = bounds_window.round_offsets().round_lengths()
        raster_window = Window(0, 0, src.width, src.height)
        try:
            window = bounds_window.intersection(raster_window)
        except WindowError:
            return np.zeros(7, dtype=np.int64)
        if window.width <= 0 or window.height <= 0:
            return np.zeros(7, dtype=np.int64)
        counts = np.zeros(7, dtype=np.int64)
        chunk_rows = 512
        chunk_columns = 2048
        row_end = int(window.row_off + window.height)
        column_end = int(window.col_off + window.width)
        for row_start in range(int(window.row_off), row_end, chunk_rows):
            height = min(chunk_rows, row_end - row_start)
            for column_start in range(int(window.col_off), column_end, chunk_columns):
                width = min(chunk_columns, column_end - column_start)
                chunk_window = Window(column_start, row_start, width, height)
                values = src.read(1, window=chunk_window, boundless=False)
                inside = geometry_mask(
                    [mapping(geometry)],
                    out_shape=values.shape,
                    transform=src.window_transform(chunk_window),
                    invert=True,
                    all_touched=False,
                )
                selected = values[inside]
                if selected.size:
                    counts += np.bincount(selected.astype(np.int64), minlength=7)[:7]
        return counts

    def process_wui_shard(task):
        wui_vintage, shard_index, input_path = task
        wui = gpd.read_parquet(input_path)
        required = {"wui_single_part_id", "wui_source_row_number", "wui_single_part_number", "year"}
        missing = sorted(required.difference(wui.columns))
        if missing:
            raise ValueError(f"WUI shard lacks required fields {missing}: {input_path}")
        if not wui.empty and not wui["year"].eq(wui_vintage).all():
            raise ValueError(f"WUI vintage mismatch, expected {wui_vintage}: {input_path}")
        if wui.crs is None:
            raise ValueError(f"WUI shard has no CRS: {input_path}")
        wui = wui.to_crs(5070).reset_index(drop=True)
        wui_geometries = wui.geometry.to_numpy()
        shard_results = []
        for year in years_by_vintage[wui_vintage]:
            burned_area_m2 = np.zeros(len(wui), dtype=np.float64)
            tree = perimeter_trees_by_year[year]
            parts = perimeter_parts_by_year[year]
            if tree is not None and len(wui):
                pairs = tree.query(wui_geometries, predicate="intersects")
                if pairs.size:
                    wui_idx, perimeter_idx = pairs
                    intersections = shapely.intersection(
                        wui_geometries[wui_idx], parts[perimeter_idx]
                    )
                    intersection_areas = shapely.area(intersections)
                    positive = intersection_areas > 0
                    if positive.any():
                        burned_area_m2 = np.bincount(
                            wui_idx[positive],
                            weights=intersection_areas[positive],
                            minlength=len(wui),
                        )
            any_burn = burned_area_m2 > 0
            class_counts = np.zeros((len(wui), 7), dtype=np.int64)
            severity_path = raster_paths[year]
            with rasterio.open(severity_path) as src:
                for wui_idx in np.flatnonzero(any_burn):
                    class_counts[wui_idx] = severity_counts(src, wui_geometries[wui_idx])

            valid_severity = class_counts[:, 1:5].sum(axis=1)
            weighted_classes = (class_counts[:, 1:5] * np.arange(1, 5)).sum(axis=1)
            mean_severity = np.divide(
                weighted_classes,
                valid_severity,
                out=np.full(len(wui), np.nan, dtype=np.float64),
                where=valid_severity > 0,
            )
            high_share = np.divide(
                class_counts[:, 4],
                valid_severity,
                out=np.full(len(wui), np.nan, dtype=np.float64),
                where=valid_severity > 0,
            )
            rows = wui.drop(columns=wui.geometry.name).copy()
            rows = rows.rename(columns={"year": "wui_vintage_year"})
            rows["year"] = year
            rows["any_burn"] = any_burn.astype("int8")
            rows["burn_area_acres"] = burned_area_m2 / 4046.8564224
            rows["log1p_burn_area_acres"] = np.log1p(rows["burn_area_acres"])
            rows["mean_mtbs_severity_class_1_4"] = mean_severity
            rows["high_severity_share"] = high_share
            rows["severity_valid_pixels"] = valid_severity
            rows["severity_class_1_pixels"] = class_counts[:, 1]
            rows["severity_class_2_pixels"] = class_counts[:, 2]
            rows["severity_class_3_pixels"] = class_counts[:, 3]
            rows["severity_class_4_pixels"] = class_counts[:, 4]
            rows["increased_greenness_pixels"] = class_counts[:, 5]
            rows["nonprocessing_pixels"] = class_counts[:, 6]
            rows["severity_valid_acres"] = valid_severity * (900.0 / 4046.8564224)
            target = temporary_dir / f"year_{year}" / f"part-{shard_index:06d}.parquet"
            functions.ensure_dir(target.parent)
            rows.to_parquet(target, index=False, compression="zstd")
            shard_results.append({
                "wui_vintage_year": wui_vintage,
                "year": year,
                "input_path": str(input_path),
                "output_path": str(target),
                "wui_single_parts": len(wui),
                "outcome_rows": len(rows),
                "burned_wui_parts": int(any_burn.sum()),
                "burn_area_acres": float(rows["burn_area_acres"].sum()),
                "severity_valid_pixels": int(valid_severity.sum()),
            })
        return shard_results

    try:
        tasks_to_run = []
        for wui_vintage in years_by_vintage:
            shard_paths = sorted((singlepart_root / f"year_{wui_vintage}").glob("part-*.parquet"))
            if not shard_paths:
                raise FileNotFoundError(
                    f"No WUI single-part parquet shards found: {singlepart_root / f'year_{wui_vintage}'}"
                )
            tasks_to_run.extend(
                (wui_vintage, shard_index, input_path)
                for shard_index, input_path in enumerate(shard_paths)
            )
        summary_rows = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_wui_shard, task) for task in tasks_to_run]
            expected_year_shards = sum(len(years_by_vintage[task[0]]) for task in tasks_to_run)
            with tqdm(total=expected_year_shards, desc="WUI–MTBS outcomes", unit="year-shard", dynamic_ncols=True) as progress:
                for future in as_completed(futures):
                    shard_results = future.result()
                    for result in shard_results:
                        summary_rows.append(result)
                        tqdm.write(
                            f"[progress] WUI vintage {result['wui_vintage_year']} / fire year {result['year']}: "
                            f"rows={result['outcome_rows']:,}; burned_parts={result['burned_wui_parts']:,}; "
                            f"burn_acres={result['burn_area_acres']:,.1f}"
                        )
                        progress.update(1)

        summary_rows.sort(key=lambda row: (row["year"], row["wui_vintage_year"], row["input_path"]))
        functions.write_csv(
            summary_rows,
            temporary_summary,
            fieldnames=(
                "wui_vintage_year",
                "year",
                "input_path",
                "output_path",
                "wui_single_parts",
                "outcome_rows",
                "burned_wui_parts",
                "burn_area_acres",
                "severity_valid_pixels",
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

    total_rows = sum(row["outcome_rows"] for row in summary_rows)
    total_burned = sum(row["burned_wui_parts"] for row in summary_rows)
    print(f"[output] Annual WUI–MTBS outcomes: {output_dir}")
    print(f"[output] Outcome summary: {summary_path}")
    print(f"[result] polygon-year rows={total_rows:,}; burned polygon-years={total_burned:,}")
    return {
        "status": "built",
        "dataset": output_dir,
        "summary": summary_path,
        "outcome_rows": total_rows,
        "burned_polygon_years": total_burned,
    }


def prepare_facts_treatments(project):
    """Contents: resolve FACTS source; retain completed prevention polygons and timing provenance."""
    functions.configure_geospatial_environment()

    import os
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyogrio
    import shapely
    from tqdm import tqdm

    source = functions.find_spatial_source(
        project.paths["facts_treatment_candidates"],
        project.paths.get(
            "facts_search_roots",
            (project.paths["raw"] / "FACTS", project.paths["data_archive"] / "raw" / "FACTS"),
        ),
        "Actv_HazFuelTrt_PL.shp",
        require_attribute_table=True,
    )
    destination = project.paths["facts_treatments_eligible"]
    overwrite = bool(project.config.get("overwrite_existing", False))
    if functions.should_skip(overwrite, [destination]):
        functions.skip_task("prepare_facts_treatments", destination)
        return {"status": "skipped", "dataset": destination}
    info = pyogrio.read_info(source)
    fields = list(info.get("fields", []))
    print(f"[input] FACTS: {source}", flush=True)
    print(f"[check] FACTS fields={len(fields)}", flush=True)
    if not fields:
        raise ValueError(
            f"FACTS shapefile has no readable attribute table: {source}. "
            "The selected .dbf may be missing, empty, corrupt, or from a different download; treatment status, acres, and dates are required."
        )
    field_by_lower = {field.casefold(): field for field in fields}

    def find_field(*names):
        return next((field_by_lower[name.casefold()] for name in names if name.casefold() in field_by_lower), None)

    stage_field = find_field("STAGE_VALU", "STAGE_VALUE", "STATUS")
    acres_field = find_field("GIS_ACRES", "TREAT_ACRES", "ACRES")
    category_field = find_field("CAT_NM", "CATEGORY_NAME", "ACTIVITY_CATEGORY")
    treatment_id_field = find_field("SUID", "FACTS_ID", "ACTIVITY_ID", "ACTIVITY_IDENTIFIER")
    completion_field = find_field("DATE_COMPL", "COMPLETION_DATE", "COMPL_DATE", "DATE_COMPLETE")
    fiscal_year_field = find_field("FISCAL_Y_1", "FY_COMPLETED", "FY_COMPL", "COMPLETION_YEAR")
    award_year_field = find_field("FY_AWARDED", "AWARD_YEAR")
    required = {"stage": stage_field, "mapped acres": acres_field, "category": category_field}
    missing = [label for label, field in required.items() if field is None]
    if missing:
        raise ValueError(
            f"FACTS source lacks required fields {missing}: {source}. Available fields: {fields}"
        )
    if completion_field is None and fiscal_year_field is None and award_year_field is None:
        raise ValueError(f"FACTS source has no completion or year field: {source}. Fields: {fields}")

    compute = functions.inspect_compute_capacity(memory_per_worker_gib=6.0, max_workers=4)
    geometry_batch_size = 25_000 * int(compute["workers"])
    print(f"[input] FACTS: {source}; features={info.get('features')}; fields={len(fields)}")
    print(f"[strategy] Arrow-backed read; vectorized geometry audit in batches of {geometry_batch_size:,}", flush=True)
    treatments = gpd.read_file(source, use_arrow=True)
    if treatments.crs is None:
        raise ValueError(f"FACTS treatment layer has no CRS: {source}")
    stage = treatments[stage_field].astype("string").str.strip().str.casefold()
    category = treatments[category_field].astype("string").str.strip().str.casefold()
    acres = pd.to_numeric(treatments[acres_field], errors="coerce")
    keep = (
        stage.eq("accomplished")
        & acres.gt(0)
        & ~category.eq("wildfire non-treatment")
    ).fillna(False)
    before_count = len(treatments)
    treatments = treatments.loc[keep].copy().reset_index(drop=True)
    treatments["GIS_ACRES_clean"] = acres.loc[keep].to_numpy(dtype=np.float64)
    treatments["treatment_source_row_number"] = np.flatnonzero(keep.to_numpy()).astype("int64") + 1

    if completion_field is not None:
        raw_date = treatments[completion_field].astype("string").str.strip()
        year_only = raw_date.str.fullmatch(r"\d{4}", na=False)
        parsed_date = pd.to_datetime(raw_date.mask(year_only), errors="coerce")
        treatments["treatment_completed_date"] = parsed_date
        treatments["treatment_completed_year"] = parsed_date.dt.year.astype("Int64")
        treatments["treatment_date_precision"] = pd.Series("unknown", index=treatments.index, dtype="string")
        treatments.loc[parsed_date.notna(), "treatment_date_precision"] = "exact"
        year_only_values = pd.to_numeric(raw_date.where(year_only), errors="coerce")
        year_only_mask = year_only_values.notna()
        treatments.loc[year_only_mask, "treatment_completed_year"] = year_only_values.loc[year_only_mask].astype("Int64")
        treatments.loc[year_only_mask, "treatment_date_precision"] = "calendar_year"
        treatments["treatment_date_source"] = completion_field
    else:
        treatments["treatment_completed_date"] = pd.NaT
        treatments["treatment_completed_year"] = pd.Series(pd.NA, index=treatments.index, dtype="Int64")
        treatments["treatment_date_precision"] = pd.Series("unknown", index=treatments.index, dtype="string")
        treatments["treatment_date_source"] = ""

    if fiscal_year_field is not None:
        fiscal_years = pd.to_numeric(treatments[fiscal_year_field], errors="coerce").astype("Int64")
        use_fiscal = treatments["treatment_completed_year"].isna() & fiscal_years.notna()
        treatments.loc[use_fiscal, "treatment_completed_year"] = fiscal_years.loc[use_fiscal]
        treatments.loc[use_fiscal, "treatment_date_precision"] = "fiscal_year"
        treatments.loc[use_fiscal, "treatment_date_source"] = fiscal_year_field
    if award_year_field is not None:
        award_years = pd.to_numeric(treatments[award_year_field], errors="coerce").astype("Int64")
        use_award = treatments["treatment_completed_year"].isna() & award_years.notna()
        treatments.loc[use_award, "treatment_completed_year"] = award_years.loc[use_award]
        treatments.loc[use_award, "treatment_date_precision"] = "award_year_proxy"
        treatments.loc[use_award, "treatment_date_source"] = award_year_field

    fallback_ids = treatments["treatment_source_row_number"].astype("string")
    if treatment_id_field is not None:
        source_ids = treatments[treatment_id_field].astype("string").str.strip()
        has_source_id = source_ids.notna() & source_ids.ne("")
        treatments["treatment_record_id"] = source_ids.where(has_source_id, fallback_ids)
        treatments["treatment_record_id_source"] = treatment_id_field
        treatments.loc[~has_source_id, "treatment_record_id_source"] = "source_row_number"
    else:
        treatments["treatment_record_id"] = fallback_ids
        treatments["treatment_record_id_source"] = "source_row_number"
    treatments["treatment_category_clean"] = category.loc[keep].reset_index(drop=True)
    treatments = treatments.to_crs(5070)
    geometries = treatments.geometry.to_numpy()
    invalid_count = 0
    missing_count = 0
    for start in tqdm(
        range(0, len(geometries), geometry_batch_size),
        desc="FACTS geometry audit",
        unit="batch",
        dynamic_ncols=True,
    ):
        stop = min(start + geometry_batch_size, len(geometries))
        chunk = geometries[start:stop]
        missing = shapely.is_missing(chunk)
        missing_count += int(missing.sum())
        valid = shapely.is_valid(chunk)
        invalid = (~valid) & (~missing)
        invalid_count += int(invalid.sum())
        if invalid.any():
            chunk = chunk.copy()
            chunk[invalid] = shapely.make_valid(chunk[invalid])
            geometries[start:stop] = chunk
    if invalid_count:
        print(f"[progress] FACTS: repaired {invalid_count:,} invalid treatment geometries", flush=True)
    if missing_count:
        print(f"[progress] FACTS: dropping {missing_count:,} missing treatment geometries", flush=True)
    missing = shapely.is_missing(geometries)
    polygonal = np.fromiter(
        (
            not is_missing
            and geometry.geom_type in {"Polygon", "MultiPolygon"}
            and not geometry.is_empty
            for geometry, is_missing in zip(geometries, missing)
        ),
        dtype=bool,
        count=len(geometries),
    )
    treatments = treatments.loc[polygonal].copy().reset_index(drop=True)
    treatments.geometry = geometries[polygonal]
    if treatments.empty:
        raise ValueError(f"No positive-acre accomplished FACTS treatment polygons remain after filtering: {source}")

    functions.ensure_dir(destination.parent)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.parquet")
    try:
        treatments.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    unknown_dates = int(treatments["treatment_date_precision"].isin(["unknown", "award_year_proxy"]).sum())
    print(
        f"[result] FACTS rows={before_count:,}; eligible polygons={len(treatments):,}; "
        f"unresolved completion timing={unknown_dates:,}; geometry_batch_size={geometry_batch_size:,}"
    )
    print(f"[output] Eligible treatment polygons: {destination}")
    return {
        "status": "built",
        "dataset": destination,
        "source_rows": before_count,
        "eligible_rows": len(treatments),
        "unresolved_date_rows": unknown_dates,
    }


def build_fire_wui_treatment_definitions_rolling5y_mixed_fires(project):
    """Contents: link fire perimeters; measure five-year path, angular coverage, and full-coverage thickness."""
    functions.configure_geospatial_environment()

    import hashlib
    import math
    import json
    import os
    import re
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds
    import pyogrio
    import shapely
    from shapely.geometry import Point
    from tqdm import tqdm

    treatment_path = project.paths["facts_treatments_eligible"]
    if not treatment_path.is_file() or treatment_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing eligible FACTS treatment layer: {treatment_path}. Run prepare_facts_treatments first."
        )
    path_distance_miles = float(project.config.get("fire_wui_distance_miles", 3.0))
    if not math.isfinite(path_distance_miles) or path_distance_miles <= 0:
        raise ValueError(f"fire_wui_distance_miles must be positive and finite; got {path_distance_miles}")
    path_distance_label = f"{path_distance_miles:g}mi"
    path_root = project.paths["fire_wui_path_dataset"] / f"distance_{path_distance_label}"
    singlepart_root = project.paths["wui_singlepart_dataset"]
    output_dir = project.paths["fire_wui_treatment_dataset"]
    summary_path = project.paths["fire_wui_treatment_summary"]
    catalog_path = project.paths["treatment_definition_catalog"]
    sample_summary_path = project.paths["fire_wui_treatment_sample_summary"]
    overwrite = bool(project.config.get("overwrite_existing", False))
    raw_worker_id = project.config.get("treatment_array_worker_id")
    raw_worker_count = project.config.get("treatment_array_worker_count")
    array_run_id = project.config.get("treatment_array_run_id")
    array_mode = raw_worker_id is not None or raw_worker_count is not None
    worker_id = None
    worker_count = None
    worker_run_root = None
    worker_output_dir = None
    if array_mode:
        if raw_worker_id is None or raw_worker_count is None or not array_run_id:
            raise ValueError("Array mode requires worker ID, worker count, and run ID")
        worker_id = int(raw_worker_id)
        worker_count = int(raw_worker_count)
        array_run_id = str(array_run_id)
        if worker_count < 1 or not 0 <= worker_id < worker_count:
            raise ValueError(f"Invalid array worker {worker_id}/{worker_count}")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", array_run_id):
            raise ValueError(f"Unsafe treatment array run ID: {array_run_id!r}")
        worker_run_root = project.paths["fire_wui_treatment_runs"] / array_run_id
        worker_output_dir = worker_run_root / f"worker_{worker_id:03d}"
        if worker_output_dir.exists():
            manifest_path = worker_output_dir / "manifest.json"
            if manifest_path.is_file():
                with manifest_path.open(encoding="utf-8") as stream:
                    completed_manifest = json.load(stream)
                if (
                    completed_manifest.get("status") == "complete"
                    and completed_manifest.get("run_id") == array_run_id
                    and int(completed_manifest.get("worker_id", -1)) == worker_id
                    and int(completed_manifest.get("worker_count", -1)) == worker_count
                    and completed_manifest.get("treatment_config_signature") == treatment_config_signature
                    and math.isclose(
                        float(completed_manifest.get("path_search_distance_miles", -1)),
                        path_distance_miles,
                    )
                ):
                    print(f"[skip] completed array worker already present: {worker_output_dir}", flush=True)
                    return {
                        "status": "worker_skipped",
                        "worker_id": worker_id,
                        "worker_count": worker_count,
                        "run_id": array_run_id,
                        "manifest": manifest_path,
                    }
            raise FileExistsError(
                f"Unverified worker output already exists: {worker_output_dir}; use a fresh --array-run-id."
            )
    elif functions.should_skip(overwrite, [output_dir, summary_path, catalog_path]):
        functions.skip_task("build_fire_wui_treatment_definitions", output_dir, summary_path, catalog_path)
        return {"status": "skipped", "dataset": output_dir, "summary": summary_path}
    if not path_root.is_dir() or not singlepart_root.is_dir():
        raise FileNotFoundError(
            f"Missing {path_distance_label} fire-WUI paths or WUI single-part inputs: "
            f"{path_root}; {singlepart_root}"
        )

    windows = functions.build_treatment_time_windows(
        rolling_years=project.config.get("treatment_rolling_years", range(1, 11)),
        lag_bands=project.config.get("treatment_lag_bands", (("lag_1_2y", 1, 2), ("lag_2_4y", 2, 4))),
    )
    if not bool(project.config.get("treatment_include_previous_calendar_year", True)):
        windows = [window for window in windows if window["kind"] != "previous_calendar_year"]
    if not windows:
        raise ValueError("Treatment task requires at least one configured timing window")
    require_mixed_fire_paths = bool(project.config.get("treatment_require_mixed_fire_paths", True))
    if not require_mixed_fire_paths:
        raise ValueError("This focused treatment dataset requires mixed treated/control paths within each fire")
    if require_mixed_fire_paths and [window["name"] for window in windows] != ["rolling_5y"]:
        raise ValueError(
            "Mixed-fire treatment sample requires exactly one window: rolling_5y; "
            f"got {[window['name'] for window in windows]}"
        )
    compute_angular = bool(project.config.get("treatment_compute_angular_exposure", True))
    treatment_buffer_miles = project.config.get("treatment_wui_buffer_miles", 1.5)
    perimeter_buffer_miles = float(project.config.get("fire_perimeter_threat_buffer_miles", 1.5))
    pair_batch_size = max(1, int(project.config.get("treatment_pair_batch_size", 256)))
    if treatment_buffer_miles is not None and float(treatment_buffer_miles) <= 0:
        raise ValueError("treatment_wui_buffer_miles must be positive or None")
    if perimeter_buffer_miles < 0:
        raise ValueError("fire_perimeter_threat_buffer_miles cannot be negative")

    compute = functions.inspect_compute_capacity(memory_per_worker_gib=16.0, max_workers=24)
    workers = int(compute["workers"])
    print(f"[input] Fire-WUI paths: {path_root}; max distance={path_distance_miles:g} miles", flush=True)
    print(f"[input] Eligible FACTS: {treatment_path}", flush=True)
    print(
        "[sample] windows=" + ",".join(window["name"] for window in windows)
        + f"; path-only={not compute_angular}; mixed-fire filter="
        + str(require_mixed_fire_paths)
        + "; angular output=continuous percent; thickness when coverage is full",
        flush=True,
    )
    treatments = gpd.read_parquet(treatment_path)
    if treatments.crs is None:
        raise ValueError(f"Eligible FACTS parquet has no CRS: {treatment_path}")
    treatments = treatments.to_crs(5070).reset_index(drop=True)
    if treatments.empty:
        raise ValueError("Eligible FACTS treatment parquet contains no rows")
    treatment_geometries = treatments.geometry.to_numpy()
    treatment_tree = shapely.STRtree(treatment_geometries)

    perimeter_path = functions.find_spatial_source(
        project.paths["mtbs_perimeter_candidates"],
        project.paths.get(
            "mtbs_search_roots",
            (project.paths["raw"] / "MTBS", project.paths["data_archive"] / "raw" / "MTBS"),
        ),
        "mtbs_perims_DD.shp",
        require_attribute_table=True,
    )
    perimeter_info = pyogrio.read_info(perimeter_path)
    perimeter_fields = list(perimeter_info.get("fields", []))
    perimeter_field_map = {field.casefold(): field for field in perimeter_fields}
    perimeter_id_field = next(
        (perimeter_field_map[key] for key in ("event_id", "fire_id") if key in perimeter_field_map),
        None,
    )
    if perimeter_id_field is None:
        raise ValueError(f"MTBS perimeters lack EVENT_ID/FIRE_ID for fire-level matching: {perimeter_path}; fields={perimeter_fields}")
    print(f"[input] MTBS perimeters: {perimeter_path}; join={perimeter_id_field}", flush=True)
    perimeters = gpd.read_file(perimeter_path, columns=[perimeter_id_field], use_arrow=True)
    if perimeters.crs is None:
        raise ValueError(f"MTBS perimeters have no CRS: {perimeter_path}")
    perimeters = perimeters.to_crs(5070)
    perimeters["_fire_id"] = perimeters[perimeter_id_field].map(functions.normalize_identifier)
    perimeter_by_fire = {}
    for fire_id, group in perimeters.dropna(subset=["_fire_id"]).groupby("_fire_id", sort=False):
        geometries = group.geometry.to_numpy()
        valid = shapely.is_valid(geometries)
        if (~valid).any():
            geometries = geometries.copy()
            geometries[~valid] = shapely.make_valid(geometries[~valid])
        perimeter_by_fire[fire_id] = shapely.union_all(geometries)
    del perimeters

    source_files = []
    for vintage in (2000, 2010, 2020):
        wui_dir = singlepart_root / f"year_{vintage}"
        path_dir = path_root / f"wui_vintage_{vintage}"
        wui_paths = sorted(wui_dir.glob("part-*.parquet"))
        if not wui_paths:
            raise FileNotFoundError(f"No WUI single-part shards found: {wui_dir}")
        for wui_path in wui_paths:
            shard = wui_path.stem.replace("part-", "")
            path_path = path_dir / f"paths_part-{shard}"
            if not path_path.is_dir():
                raise FileNotFoundError(
                    f"Missing matching {path_distance_miles:g}-mile path shard for WUI input "
                    f"{wui_path}: {path_path}"
                )
            source_files.append((vintage, wui_path, path_path))
    if not source_files:
        raise FileNotFoundError(
            f"No matching {path_distance_miles:g}-mile path/WUI shard pairs found under {path_root}"
        )

    all_source_files = source_files
    all_source_shards = [f"{vintage}/{wui_path.stem}" for vintage, wui_path, _ in all_source_files]
    if array_mode:
        source_files = [
            source for index, source in enumerate(all_source_files)
            if index % worker_count == worker_id
        ]
        if not source_files:
            raise ValueError(f"Array worker {worker_id} received no WUI shards")
    assigned_source_shards = [f"{vintage}/{wui_path.stem}" for vintage, wui_path, _ in source_files]
    if array_mode:
        print(
            f"[array] run={array_run_id}; worker={worker_id + 1}/{worker_count}; "
            f"shards={len(source_files)}/{len(all_source_files)}; output={worker_output_dir}",
            flush=True,
        )

    functions.ensure_dir(output_dir.parent)
    functions.ensure_dir(summary_path.parent)
    if array_mode:
        functions.ensure_dir(worker_run_root)
        temporary_stage_root = Path(tempfile.mkdtemp(
            prefix=f".worker_{worker_id:03d}.", dir=worker_run_root
        ))
        temporary_dir = temporary_stage_root / "dataset"
        functions.ensure_dir(temporary_dir)
        temporary_summary = temporary_stage_root / "summary.csv"
        temporary_catalog = temporary_stage_root / "catalog.csv"
        temporary_manifest = temporary_stage_root / "manifest.json"
        temporary_sample_summary = None
    else:
        temporary_stage_root = None
        temporary_dir = Path(tempfile.mkdtemp(prefix=".fire_wui_treatment.", dir=output_dir.parent))
        temporary_summary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        temporary_catalog = catalog_path.with_name(f".{catalog_path.name}.{os.getpid()}.tmp")
        temporary_sample_summary = sample_summary_path.with_name(
            f".{sample_summary_path.name}.{os.getpid()}.tmp"
        )
    summary_rows = []
    total_pairs = 0
    total_exposure_rows = 0
    perimeter_matched = 0
    perimeter_buffer_m = perimeter_buffer_miles * 1609.344
    def process_pair(pair_row, wui_by_id):
        pair = pair_row.to_dict() if hasattr(pair_row, "to_dict") else dict(pair_row)
        path_geometry = pair.pop("geometry", None)
        wui_id = functions.normalize_identifier(pair.get("wui_single_part_id"))
        wui = wui_by_id.get(wui_id)
        if path_geometry is None or wui is None:
            raise ValueError(
                f"Fire-WUI pair is missing path or WUI geometry: path_id={pair.get('path_id')}; "
                f"wui_single_part_id={pair.get('wui_single_part_id')}"
            )
        coordinates = list(path_geometry.coords)
        if len(coordinates) < 2:
            raise ValueError(f"Fire-WUI pair has invalid path geometry: {pair.get('path_id')}")
        origin = Point(coordinates[0])
        fire_id_value = pair.get("fire_event_id")
        fire_id = functions.normalize_identifier(fire_id_value)
        perimeter = perimeter_by_fire.get(fire_id)
        perimeter_matched_here = perimeter is not None
        wui_area = float(wui.area)
        perimeter_distance = float(perimeter.distance(wui)) if perimeter_matched_here else np.nan
        overlap_area = float(shapely.area(shapely.intersection(perimeter, wui))) if perimeter_matched_here else np.nan
        burn_fraction = min(1.0, max(0.0, overlap_area / wui_area)) if perimeter_matched_here and wui_area > 0 else np.nan
        perimeter_risk = bool(perimeter_distance <= perimeter_buffer_m) if perimeter_matched_here else None
        any_burn = bool(overlap_area > 0.0) if perimeter_matched_here else None
        perimeter_control = (
            int(not any_burn)
            if perimeter_matched_here and perimeter_risk
            else None
        )

        edge_x = float(pair.get("wui_edge_x", coordinates[-1][0]))
        edge_y = float(pair.get("wui_edge_y", coordinates[-1][1]))
        fire_x, fire_y = float(origin.x), float(origin.y)
        angle_degrees = math.degrees(math.atan2(edge_y - fire_y, edge_x - fire_x)) % 360.0
        direction_bin_24 = int(math.floor(angle_degrees / 15.0)) % 24
        distance_m = float(pair.get("fire_to_wui_edge_distance_m", path_geometry.length))
        distance_bin_500m = int(math.floor(distance_m / 500.0))
        search_geometry = path_geometry
        candidate_idx = np.asarray(treatment_tree.query(search_geometry, predicate="intersects"), dtype=np.int64)
        base = {
            **pair,
            "path_id": pair.get("path_id"),
            "path_search_distance_miles": path_distance_miles,
            "fire_event_id_normalized": fire_id,
            "fire_anchor_type": "mtbs_centroid",
            "fire_direction_degrees": angle_degrees,
            "fire_direction_bin_24": direction_bin_24,
            "distance_bin_500m": distance_bin_500m,
            "fire_perimeter_match_status": "matched" if perimeter_matched_here else "unmatched",
            "fire_perimeter_distance_m": perimeter_distance,
            "fire_perimeter_distance_miles": perimeter_distance / 1609.344 if perimeter_matched_here else np.nan,
            "fire_perimeter_threat_buffer_miles": perimeter_buffer_miles,
            "within_fire_perimeter_threat_buffer": perimeter_risk,
            "fire_perimeter_risk_set_eligible": int(bool(perimeter_risk)) if perimeter_matched_here else None,
            "fire_perimeter_control": perimeter_control,
            "any_burn_by_fire": int(any_burn) if perimeter_matched_here else None,
            "burned_area_in_wui_m2": overlap_area,
            "burned_wui_fraction": burn_fraction,
            "wui_area_m2": wui_area,
            "candidate_treatment_polygon_count": int(len(candidate_idx)),
        }

        candidate_records = treatments.iloc[candidate_idx]
        candidate_geometries = treatment_geometries[candidate_idx]
        completion_date = pair.get("fire_ig_date")
        if completion_date is None or pd.isna(completion_date):
            completion_date = pair.get("fire_ignition_date")
        fire_year = pair.get("fire_year")
        if fire_year is None or pd.isna(fire_year):
            raise ValueError(f"Fire-WUI pair lacks fire year: {pair.get('path_id')}")
        if fire_id is None:
            raise ValueError(f"Fire-WUI pair lacks MTBS event ID: {pair.get('path_id')}")
        candidate_count = len(candidate_records)
        candidate_dates = candidate_records.get(
            "treatment_completed_date", pd.Series([None] * candidate_count)
        ).tolist()
        candidate_years = candidate_records.get(
            "treatment_completed_year", pd.Series([None] * candidate_count)
        ).tolist()
        candidate_precisions = candidate_records.get(
            "treatment_date_precision", pd.Series(["unknown"] * candidate_count)
        ).tolist()
        memberships = functions.treatment_window_membership_matrix(
            candidate_dates,
            candidate_years,
            candidate_precisions,
            completion_date,
            fire_year,
            windows,
        )
        candidate_ids = [
            functions.normalize_identifier(value)
            for value in candidate_records.get("treatment_record_id", pd.Series([None] * candidate_count)).tolist()
        ]
        candidate_categories = candidate_records.get(
            "treatment_category_clean", pd.Series([None] * candidate_count)
        ).tolist()
        candidate_path_hits = (
            np.asarray(shapely.intersects(candidate_geometries, path_geometry), dtype=bool)
            if candidate_count
            else np.empty(0, dtype=bool)
        )
        union_cache = {}
        path_summary_cache = {}
        angular_cache = {}

        def cached_union(local_indices):
            key = tuple(local_indices)
            if key not in union_cache:
                selected_geometries = [candidate_geometries[index] for index in key]
                union_cache[key] = (
                    shapely.union_all(selected_geometries)
                    if selected_geometries
                    else shapely.GeometryCollection()
                )
            return key, union_cache[key]

        def cached_path_summary(key, treatment_union):
            if key not in path_summary_cache:
                intersects = bool(
                    not shapely.is_empty(treatment_union)
                    and path_geometry.intersects(treatment_union)
                )
                length = (
                    float(path_geometry.intersection(treatment_union).length)
                    if not shapely.is_empty(treatment_union)
                    else 0.0
                )
                path_summary_cache[key] = (intersects, length)
            return path_summary_cache[key]

        def cached_angular(key, treatment_union):
            cache_key = key
            if cache_key not in angular_cache:
                angular_cache[cache_key] = functions.angular_treatment_metrics(
                    origin,
                    wui,
                    [treatment_union] if not shapely.is_empty(treatment_union) else [],
                    thickness_step_degrees=0.25,
                    compute_thickness_when_full=True,
                )
            return angular_cache[cache_key]

        output_rows = []
        for window_idx, window in enumerate(windows):
            membership = memberships[:, window_idx]
            known_local_idx = np.flatnonzero(np.equal(membership, True)).tolist()
            uncertain_local_idx = np.flatnonzero(np.equal(membership, None)).tolist()
            path_local_idx = [index for index in known_local_idx if candidate_path_hits[index]]
            union_local_idx = path_local_idx
            union_key, treatment_union = cached_union(union_local_idx)
            known_path_intersection, path_length = cached_path_summary(union_key, treatment_union)
            uncertain_path_hit = bool(candidate_path_hits[uncertain_local_idx].any())
            possible_coverage = None
            if compute_angular and uncertain_local_idx:
                uncertain_path_idx = [
                    index for index in uncertain_local_idx if candidate_path_hits[index]
                ]
                possible_local_idx = sorted((*path_local_idx, *uncertain_path_idx))
                possible_key, possible_union = cached_union(possible_local_idx)
                possible_angular = cached_angular(possible_key, possible_union)
                possible_coverage = possible_angular["angular_coverage_fraction"]
            if known_path_intersection:
                nearest_status = "treated"
            elif uncertain_path_hit:
                nearest_status = "unknown"
            else:
                nearest_status = "control"
            if compute_angular:
                angular = cached_angular(union_key, treatment_union)
                coverage = angular["angular_coverage_fraction"]
                if not uncertain_local_idx:
                    possible_coverage = coverage
            else:
                angular = {
                    "angular_total_degrees": np.nan,
                    "angular_covered_degrees": np.nan,
                    "angular_coverage_fraction": np.nan,
                    "thickness_mean_m": np.nan,
                    "thickness_p10_m": np.nan,
                    "thickness_min_m": np.nan,
                    "thickness_max_m": np.nan,
                    "contiguous_thickness_mean_m": np.nan,
                    "angular_geometry_status": "not_computed_for_path_only_run",
                }
                coverage = None
            date_complete = len(uncertain_local_idx) == 0
            known_treatment_ids = [candidate_ids[index] for index in known_local_idx]
            uncertain_treatment_ids = [candidate_ids[index] for index in uncertain_local_idx]
            path_treatment_ids = [
                candidate_ids[index]
                for index in known_local_idx
                if candidate_path_hits[index]
            ]
            category_names = sorted({
                str(candidate_categories[index])
                for index in known_local_idx
                if pd.notna(candidate_categories[index])
            })
            result = {
                **base,
                "time_window": window["name"],
                "time_window_kind": window["kind"],
                "treatment_count_in_window": len(known_local_idx),
                "treatment_date_uncertain_count": len(uncertain_local_idx),
                "treatment_date_classification_complete": date_complete,
                "treatment_record_ids_in_window": json.dumps(known_treatment_ids),
                "undated_or_ambiguous_treatment_record_ids": json.dumps(uncertain_treatment_ids),
                "treatment_categories_in_window": json.dumps(category_names),
                "nearest_path_treatment_record_ids": json.dumps(path_treatment_ids),
                "nearest_path_treatment_status": nearest_status,
                "nearest_path_treatment_intersects": (
                    1 if known_path_intersection else (None if uncertain_path_hit else 0)
                ),
                "nearest_path_treated_length_m": path_length,
                "angular_total_degrees": angular["angular_total_degrees"],
                "angular_covered_degrees": angular["angular_covered_degrees"],
                "angular_coverage_fraction": coverage,
                "angular_coverage_percent": (
                    100.0 * coverage if coverage is not None and np.isfinite(coverage) else np.nan
                ),
                "angular_coverage_fraction_if_undated_candidates_eligible": possible_coverage,
                "angular_coverage_percent_if_undated_candidates_eligible": (
                    100.0 * possible_coverage
                    if possible_coverage is not None and np.isfinite(possible_coverage)
                    else np.nan
                ),
                "thickness_mean_m": angular["thickness_mean_m"],
                "thickness_p10_m": angular["thickness_p10_m"],
                "thickness_min_m": angular["thickness_min_m"],
                "thickness_max_m": angular["thickness_max_m"],
                "contiguous_thickness_mean_m": angular["contiguous_thickness_mean_m"],
                "angular_geometry_status": angular["angular_geometry_status"],
                "crs_epsg": 5070,
            }
            result["nearest_path_treated"] = (
                1 if nearest_status == "treated" else (0 if nearest_status == "control" else None)
            )
            result["nearest_path_control"] = (
                1 if nearest_status == "control" else (0 if nearest_status == "treated" else None)
            )
            output_rows.append(result)
        if len(output_rows) != len(windows):
            raise RuntimeError(f"Incomplete treatment-window output for path_id={pair.get('path_id')}")
        return output_rows, {"status": "ok", "perimeter_matched": perimeter_matched_here}

    try:
        with tqdm(total=len(source_files), desc="FACTS fire-WUI treatment", unit="shard", dynamic_ncols=True) as progress:
            for vintage, wui_path, path_path in source_files:
                wui_frame = gpd.read_parquet(wui_path)
                if wui_frame.crs is None:
                    raise ValueError(f"WUI shard has no CRS: {wui_path}")
                required_wui = {"wui_single_part_id"}
                missing_wui = sorted(required_wui.difference(wui_frame.columns))
                if missing_wui:
                    raise ValueError(f"WUI shard lacks fields {missing_wui}: {wui_path}")
                wui_frame = wui_frame.to_crs(5070)
                wui_by_id = {
                    functions.normalize_identifier(identifier): geometry
                    for identifier, geometry in zip(wui_frame["wui_single_part_id"], wui_frame.geometry)
                }
                shard_out_dir = temporary_dir / f"wui_vintage_{vintage}" / wui_path.stem
                functions.ensure_dir(shard_out_dir)
                path_part_files = sorted(path_path.glob("part-*.parquet"))
                chunk_number = 0
                shard_pair_count = 0
                shard_rows = 0
                shard_perimeter_matches = 0
                if not path_part_files:
                    summary_rows.append({
                        "wui_vintage_year": vintage,
                        "path_search_distance_miles": path_distance_miles,
                        "wui_input": str(wui_path),
                        "path_input": str(path_path),
                        "fire_wui_pairs": 0,
                        "treatment_window_rows": 0,
                        "perimeter_matched_pairs": 0,
                        "perimeter_match_share": np.nan,
                        "output_dir": str(output_dir / f"wui_vintage_{vintage}" / wui_path.stem),
                    })
                    tqdm.write(f"[progress] WUI {vintage} {wui_path.name}: no fire-WUI path pairs")
                    progress.update(1)
                    continue
                path_dataset = ds.dataset([str(path) for path in path_part_files], format="parquet")
                pair_progress = tqdm(
                    desc=f"Treatment pairs {vintage} {wui_path.stem}",
                    unit="pair",
                    dynamic_ncols=True,
                    leave=False,
                )
                try:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        for batch in path_dataset.to_batches(batch_size=pair_batch_size):
                            pair_frame = batch.to_pandas()
                            if "geometry" in pair_frame.columns:
                                pair_frame["geometry"] = shapely.from_wkb(pair_frame["geometry"].to_numpy())
                            else:
                                raise ValueError(f"Path shard lacks geometry column: {path_path}")
                            futures = {
                                executor.submit(process_pair, row, wui_by_id): index
                                for index, row in enumerate(pair_frame.to_dict(orient="records"))
                            }
                            ordered_results = [None] * len(futures)
                            for future in as_completed(futures):
                                ordered_results[futures[future]] = future.result()
                                pair_progress.update(1)
                            pair_results = ordered_results
                            rows = []
                            for pair_rows, status in pair_results:
                                shard_pair_count += 1
                                shard_perimeter_matches += int(status["perimeter_matched"])
                                rows.extend(pair_rows)
                            if rows:
                                output_path = shard_out_dir / f"part-{chunk_number:06d}.parquet"
                                pd.DataFrame(rows).to_parquet(output_path, index=False, compression="zstd")
                                chunk_number += 1
                                shard_rows += len(rows)
                            total_pairs += len(pair_results)
                            total_exposure_rows += len(rows)
                finally:
                    pair_progress.close()
                perimeter_matched += shard_perimeter_matches
                summary_rows.append({
                    "wui_vintage_year": vintage,
                    "path_search_distance_miles": path_distance_miles,
                    "wui_input": str(wui_path),
                    "path_input": str(path_path),
                    "fire_wui_pairs": shard_pair_count,
                    "treatment_window_rows": shard_rows,
                    "perimeter_matched_pairs": shard_perimeter_matches,
                    "perimeter_match_share": shard_perimeter_matches / shard_pair_count if shard_pair_count else np.nan,
                    "output_dir": str(output_dir / f"wui_vintage_{vintage}" / wui_path.stem),
                })
                tqdm.write(
                    f"[progress] WUI {vintage} {wui_path.name}: pairs={shard_pair_count:,}; "
                    f"window rows={shard_rows:,}; perimeter matches={shard_perimeter_matches:,}"
                )
                progress.update(1)

        catalog_rows = []
        treatment_scope = "centroid_to_wui_edge_path_intersection"
        for window in windows:
            if window["kind"] == "previous_calendar_year":
                lower_rule = "January 1 of previous calendar year, inclusive"
                upper_rule = "January 1 of fire year, exclusive"
            elif window["kind"] == "rolling":
                lower_rule = f"fire date minus {window['max_years']} calendar years, inclusive"
                upper_rule = "fire date, exclusive"
            else:
                lower_rule = f"fire date minus {window['max_years']} calendar years, exclusive"
                upper_rule = f"fire date minus {window['min_years']} calendar years, inclusive"
            catalog_rows.append({
                "time_window": window["name"],
                "time_window_kind": window["kind"],
                "path_search_distance_miles": path_distance_miles,
                "treatment_config_signature": treatment_config_signature,
                "spatial_definition": "nearest_path_intersection_with_continuous_angular_coverage",
                "time_lower_bound": lower_rule,
                "time_upper_bound": upper_rule,
                "completion_date_precision_rule": "use DATE_COMPL when exact; use FISCAL_Y_1 as fiscal-year interval; FY_AWARDED or missing year remains timing-unknown",
                "treatment_spatial_scope": treatment_scope,
                "treatment_wui_buffer_miles": treatment_buffer_miles,
                "fire_perimeter_threat_buffer_miles": perimeter_buffer_miles,
                "comparison_rule": "path intersection=treated; no intersection=control; undated intersecting treatment=unknown",
                "angular_treatment_rule": (
                    "continuous percent of visible WUI-boundary angles shielded by rolling-five-year treatment; "
                    "candidate treatment polygons must intersect nearest centroid-to-WUI-edge path"
                ),
                "thickness_rule": (
                    "calculate treatment thickness only when angular coverage is complete; "
                    "0.25-degree weighted angular quadrature; report total and longest contiguous radial thickness"
                ),
                "fire_perimeter_control_rule": "within threat buffer and no overlap; unmatched perimeter is unknown",
                "mixed_fire_rule": (
                    "retain fire only if at least one treated and one control fire-WUI path exist"
                    if bool(project.config.get("treatment_require_mixed_fire_paths", True))
                    else "no fire-level mixed-exposure restriction"
                ),
                "path_sample_rule": (
                    "retain treated/control paths; exclude unknown timing paths"
                    if bool(project.config.get("treatment_keep_only_classified_paths", True))
                    else "retain all path statuses for qualifying fires"
                ),
                "angular_metrics_computed": compute_angular,
            })
        if not array_mode:
            sample_metrics = functions.filter_mixed_fire_path_dataset(
                temporary_dir,
                required_window="rolling_5y",
                keep_only_classified_paths=bool(
                    project.config.get("treatment_keep_only_classified_paths", True)
                ),
            )
            functions.write_csv(
                [{
                    "run_id": "single_process",
                    "time_window": "rolling_5y",
                    "sample_rule": "fire has at least one treated and one control centroid-to-WUI path",
                    **sample_metrics,
                }],
                temporary_sample_summary,
                fieldnames=(
                    "run_id", "time_window", "sample_rule", "input_fire_wui_paths",
                    "input_unique_fires", "input_treated_paths", "input_control_paths",
                    "input_unknown_paths", "qualifying_fires", "retained_fire_wui_paths",
                    "retained_treated_paths", "retained_control_paths", "excluded_fire_wui_paths",
                ),
            )

        functions.write_csv(
            summary_rows,
            temporary_summary,
            fieldnames=("wui_vintage_year", "path_search_distance_miles", "wui_input", "path_input", "fire_wui_pairs", "treatment_window_rows", "perimeter_matched_pairs", "perimeter_match_share", "output_dir"),
        )
        functions.write_csv(
            catalog_rows,
            temporary_catalog,
            fieldnames=("time_window", "time_window_kind", "path_search_distance_miles", "spatial_definition", "time_lower_bound", "time_upper_bound", "completion_date_precision_rule", "treatment_spatial_scope", "treatment_wui_buffer_miles", "fire_perimeter_threat_buffer_miles", "comparison_rule", "angular_treatment_rule", "thickness_rule", "fire_perimeter_control_rule", "mixed_fire_rule", "path_sample_rule", "angular_metrics_computed"),
        )
        if array_mode:
            manifest = {
                "status": "complete",
                "run_id": array_run_id,
                "worker_id": worker_id,
                "worker_count": worker_count,
                "path_search_distance_miles": path_distance_miles,
                "expected_shard_count": len(all_source_shards),
                "expected_source_shards": all_source_shards,
                "assigned_source_shards": assigned_source_shards,
                "fire_wui_pairs": total_pairs,
                "treatment_window_rows": total_exposure_rows,
                "perimeter_matched_pairs": perimeter_matched,
                "dataset_file_count": sum(
                    1 for path in temporary_dir.rglob("*.parquet") if path.is_file()
                ),
                "summary_file": "summary.csv",
                "catalog_file": "catalog.csv",
                "dataset_dir": "dataset",
            }
            with temporary_manifest.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
                stream.write("\n")
            temporary_stage_root.replace(worker_output_dir)
            print(f"[output] worker dataset: {worker_output_dir / 'dataset'}", flush=True)
            print(f"[output] worker manifest: {worker_output_dir / 'manifest.json'}", flush=True)
            return {
                "status": "worker_complete",
                "worker_id": worker_id,
                "worker_count": worker_count,
                "run_id": array_run_id,
                "manifest": worker_output_dir / "manifest.json",
                "pairs": total_pairs,
                "exposure_rows": total_exposure_rows,
            }
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
        temporary_summary.replace(summary_path)
        temporary_catalog.replace(catalog_path)
        if not array_mode:
            temporary_sample_summary.replace(sample_summary_path)
    except BaseException:
        if temporary_stage_root is not None:
            shutil.rmtree(temporary_stage_root, ignore_errors=True)
        else:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        temporary_summary.unlink(missing_ok=True)
        temporary_catalog.unlink(missing_ok=True)
        if temporary_sample_summary is not None:
            temporary_sample_summary.unlink(missing_ok=True)
        raise

    print(f"[output] Fire-WUI treatment definitions: {output_dir}")
    print(f"[output] Treatment definition catalog: {catalog_path}")
    if not array_mode:
        print(f"[output] Mixed-fire sample summary: {sample_summary_path}")
    print(
        f"[result] pairs={total_pairs:,}; exposure rows={total_exposure_rows:,}; "
        f"perimeter matches={perimeter_matched:,}/{total_pairs:,}; windows={len(windows)}; angular measure=continuous percent"
    )
    if not array_mode:
        print(
            f"[sample] qualifying fires={sample_metrics['qualifying_fires']:,}; "
            f"retained paths={sample_metrics['retained_fire_wui_paths']:,}; "
            f"treated={sample_metrics['retained_treated_paths']:,}; "
            f"controls={sample_metrics['retained_control_paths']:,}",
            flush=True,
        )
    return {
        "status": "built",
        "dataset": output_dir,
        "summary": summary_path,
        "catalog": catalog_path,
        "sample_summary": sample_summary_path if not array_mode else None,
        "pairs": total_pairs,
        "exposure_rows": total_exposure_rows,
        "perimeter_matches": perimeter_matched,
    }


def finalize_fire_wui_treatment_definitions(project):
    """Contents: verify every array manifest; assemble unique shard outputs and publish final tables."""
    functions.configure_geospatial_environment()

    import hashlib
    import json
    import os
    import re
    import shutil
    import tempfile
    from pathlib import Path

    import pandas as pd
    from tqdm import tqdm

    run_id = project.config.get("treatment_array_run_id")
    worker_count = project.config.get("treatment_array_worker_count")
    if not run_id or worker_count is None:
        raise ValueError("Finalizer requires --array-run-id and --array-worker-count")
    run_id = str(run_id)
    worker_count = int(worker_count)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError(f"Unsafe treatment array run ID: {run_id!r}")
    if worker_count < 1:
        raise ValueError(f"Array worker count must be positive; got {worker_count}")
    functions.inspect_compute_capacity(memory_per_worker_gib=2.0, max_workers=1)

    output_dir = project.paths["fire_wui_treatment_dataset"]
    summary_path = project.paths["fire_wui_treatment_summary"]
    catalog_path = project.paths["treatment_definition_catalog"]
    overwrite = bool(project.config.get("overwrite_existing", False))
    if functions.should_skip(overwrite, [output_dir, summary_path, catalog_path]):
        functions.skip_task("finalize_fire_wui_treatment_definitions", output_dir, summary_path, catalog_path)
        return {"status": "skipped", "dataset": output_dir, "summary": summary_path}

    run_root = project.paths["fire_wui_treatment_runs"] / run_id
    worker_dirs = [run_root / f"worker_{worker_id:03d}" for worker_id in range(worker_count)]
    manifests = []
    for worker_id, worker_dir in enumerate(worker_dirs):
        manifest_path = worker_dir / "manifest.json"
        if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing successful worker manifest: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if (
            manifest.get("status") != "complete"
            or manifest.get("run_id") != run_id
            or int(manifest.get("worker_id", -1)) != worker_id
            or int(manifest.get("worker_count", -1)) != worker_count
        ):
            raise ValueError(f"Worker manifest identity/status mismatch: {manifest_path}")
        manifests.append(manifest)

    treatment_signatures = {
        manifest.get("treatment_config_signature") for manifest in manifests
    }
    if len(treatment_signatures) != 1 or None in treatment_signatures:
        raise ValueError("Treatment workers used different or unrecorded treatment configurations")

    expected_shards = manifests[0].get("expected_source_shards")
    if not isinstance(expected_shards, list) or not expected_shards:
        raise ValueError(f"Worker manifest has no expected source roster: {worker_dirs[0] / 'manifest.json'}")
    if len(expected_shards) != int(manifests[0].get("expected_shard_count", -1)):
        raise ValueError("Expected WUI shard roster count does not match manifest count")
    assigned_shards = []
    summary_rows = []
    catalog_rows = None
    catalog_digest = None
    total_pairs = 0
    total_rows = 0
    perimeter_matches = 0
    worker_datasets = []
    for worker_id, (worker_dir, manifest) in enumerate(zip(worker_dirs, manifests)):
        if manifest.get("expected_source_shards") != expected_shards:
            raise ValueError(f"Workers disagree on expected WUI shard roster: {worker_dir}")
        if int(manifest.get("expected_shard_count", -1)) != len(expected_shards):
            raise ValueError(f"Worker expected-shard count mismatch: {worker_dir}")
        expected_for_worker = [
            shard for index, shard in enumerate(expected_shards)
            if index % worker_count == worker_id
        ]
        assigned = manifest.get("assigned_source_shards")
        if assigned != expected_for_worker:
            raise ValueError(f"Worker shard assignment is incomplete or inconsistent: {worker_dir}")
        assigned_shards.extend(assigned)
        dataset_dir = worker_dir / str(manifest.get("dataset_dir", "dataset"))
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Worker dataset directory missing: {dataset_dir}")
        worker_parquets = [path for path in dataset_dir.rglob("*.parquet") if path.is_file()]
        if len(worker_parquets) != int(manifest.get("dataset_file_count", -1)):
            raise ValueError(f"Worker Parquet file count mismatch: {dataset_dir}")
        for shard_key in assigned:
            vintage, shard_name = shard_key.split("/", 1)
            expected_dir = dataset_dir / f"wui_vintage_{vintage}" / shard_name
            if not expected_dir.is_dir():
                raise FileNotFoundError(f"Worker output shard missing: {expected_dir}")
        worker_datasets.append(dataset_dir)

        worker_summary = worker_dir / str(manifest.get("summary_file", "summary.csv"))
        worker_catalog = worker_dir / str(manifest.get("catalog_file", "catalog.csv"))
        if not worker_summary.is_file() or not worker_catalog.is_file():
            raise FileNotFoundError(f"Worker summary/catalog missing: {worker_dir}")
        local_summary_rows = pd.read_csv(worker_summary).to_dict(orient="records")
        if len(local_summary_rows) != len(assigned):
            raise ValueError(f"Worker summary row count mismatch: {worker_summary}")
        summary_rows.extend(local_summary_rows)
        local_catalog_rows = pd.read_csv(worker_catalog).to_dict(orient="records")
        local_digest = hashlib.sha256(worker_catalog.read_bytes()).hexdigest()
        if catalog_digest is None:
            catalog_digest = local_digest
            catalog_rows = local_catalog_rows
        elif local_digest != catalog_digest:
            raise ValueError(f"Workers produced different treatment catalogs: {worker_catalog}")
        total_pairs += int(manifest.get("fire_wui_pairs", 0))
        total_rows += int(manifest.get("treatment_window_rows", 0))
        perimeter_matches += int(manifest.get("perimeter_matched_pairs", 0))

    if len(assigned_shards) != len(set(assigned_shards)) or set(assigned_shards) != set(expected_shards):
        raise ValueError("Worker manifests do not cover each expected WUI shard exactly once")
    summary_pairs = sum(int(row["fire_wui_pairs"]) for row in summary_rows)
    summary_exposures = sum(int(row["treatment_window_rows"]) for row in summary_rows)
    summary_matches = sum(int(row["perimeter_matched_pairs"]) for row in summary_rows)
    if (summary_pairs, summary_exposures, summary_matches) != (total_pairs, total_rows, perimeter_matches):
        raise ValueError("Worker summary totals do not match manifest totals")
    summary_rows.sort(key=lambda row: (int(row["wui_vintage_year"]), str(row["wui_input"])))

    functions.ensure_dir(output_dir.parent)
    functions.ensure_dir(summary_path.parent)
    temporary_root = Path(tempfile.mkdtemp(prefix=".fire_wui_treatment_finalize.", dir=output_dir.parent))
    temporary_dir = temporary_root / "dataset"
    temporary_dir.mkdir()
    temporary_summary = temporary_root / "summary.csv"
    temporary_catalog = temporary_root / "catalog.csv"
    temporary_sample_summary = temporary_root / "sample_summary.csv"
    try:
        shard_files = [
            (path, dataset_dir)
            for dataset_dir in worker_datasets
            for path in dataset_dir.rglob("*")
            if path.is_file()
        ]
        seen_relpaths = set()
        with tqdm(total=len(shard_files), desc="Assemble treatment shards", unit="file", dynamic_ncols=True) as progress:
            for source, dataset_dir in shard_files:
                relative = source.relative_to(dataset_dir)
                if relative in seen_relpaths:
                    raise ValueError(f"Duplicate worker output path: {relative}")
                seen_relpaths.add(relative)
                destination = temporary_dir / relative
                functions.ensure_dir(destination.parent)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
                progress.update(1)

        sample_metrics = functions.filter_mixed_fire_path_dataset(
            temporary_dir,
            required_window="rolling_5y",
            keep_only_classified_paths=bool(
                project.config.get("treatment_keep_only_classified_paths", True)
            ),
        )
        if sample_metrics["input_fire_wui_paths"] != total_rows:
            raise ValueError(
                "Worker exposure-row total differs from assembled path rows: "
                f"{total_rows:,} != {sample_metrics['input_fire_wui_paths']:,}"
            )
        functions.write_csv(
            [{
                "run_id": run_id,
                "time_window": "rolling_5y",
                "sample_rule": "fire has at least one treated and one control centroid-to-WUI path",
                **sample_metrics,
            }],
            temporary_sample_summary,
            fieldnames=(
                "run_id", "time_window", "sample_rule", "input_fire_wui_paths",
                "input_unique_fires", "input_treated_paths", "input_control_paths",
                "input_unknown_paths", "qualifying_fires", "retained_fire_wui_paths",
                "retained_treated_paths", "retained_control_paths", "excluded_fire_wui_paths",
            ),
        )

        functions.write_csv(
            summary_rows,
            temporary_summary,
            fieldnames=("wui_vintage_year", "path_search_distance_miles", "wui_input", "path_input", "fire_wui_pairs", "treatment_window_rows", "perimeter_matched_pairs", "perimeter_match_share", "output_dir"),
        )
        functions.write_csv(
            catalog_rows or [],
            temporary_catalog,
            fieldnames=("time_window", "time_window_kind", "path_search_distance_miles", "spatial_definition", "time_lower_bound", "time_upper_bound", "completion_date_precision_rule", "treatment_spatial_scope", "treatment_wui_buffer_miles", "fire_perimeter_threat_buffer_miles", "comparison_rule", "angular_treatment_rule", "thickness_rule", "fire_perimeter_control_rule", "mixed_fire_rule", "path_sample_rule", "angular_metrics_computed"),
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
        temporary_summary.replace(summary_path)
        temporary_catalog.replace(catalog_path)
        temporary_sample_summary.replace(sample_summary_path)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    shutil.rmtree(temporary_root, ignore_errors=True)

    print(f"[output] Fire-WUI treatment definitions: {output_dir}")
    print(f"[output] Treatment definition catalog: {catalog_path}")
    print(f"[output] Mixed-fire sample summary: {sample_summary_path}")
    print(f"[provenance] Worker manifests retained: {run_root}")
    print(
        f"[result] pairs={total_pairs:,}; exposure rows={total_rows:,}; "
        f"perimeter matches={perimeter_matches:,}/{total_pairs:,}; "
        f"qualifying fires={sample_metrics['qualifying_fires']:,}; "
        f"retained paths={sample_metrics['retained_fire_wui_paths']:,}; "
        f"workers={worker_count}; shards={len(expected_shards)}"
    )
    return {
        "status": "finalized",
        "dataset": output_dir,
        "summary": summary_path,
        "catalog": catalog_path,
        "sample_summary": sample_summary_path,
        "run_manifest_root": run_root,
        "workers": worker_count,
        "shards": len(expected_shards),
        "pairs": total_pairs,
        "exposure_rows": total_rows,
    }


def build_fire_wui_treatment_definitions_rolling5y_mixed_fires_1p5mi(project):
    """Run shared rolling-5y mixed-fire treatment task on 1.5-mile path sample."""
    return functions.run_treatment_definition_variant(
        project,
        build_fire_wui_treatment_definitions_rolling5y_mixed_fires,
        variant="1p5mi",
        distance_miles=1.5,
        task_name="build_fire_wui_treatment_definitions_rolling5y_mixed_fires_1p5mi",
        finalizer_task_name="finalize_fire_wui_treatment_definitions_1p5mi",
    )


def finalize_fire_wui_treatment_definitions_1p5mi(project):
    """Finalize rolling-5y mixed-fire treatment outputs isolated for 1.5-mile paths."""
    return functions.run_treatment_definition_variant(
        project,
        finalize_fire_wui_treatment_definitions,
        variant="1p5mi",
        distance_miles=1.5,
        task_name="build_fire_wui_treatment_definitions_rolling5y_mixed_fires_1p5mi",
        finalizer_task_name="finalize_fire_wui_treatment_definitions_1p5mi",
    )


def landfire_controls(project):
    """Create LF2020 terrain controls for WUI units, MTBS point records, and fire areas."""
    functions.configure_geospatial_environment()

    import os
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    import shapely
    from pyproj import CRS
    from tqdm import tqdm

    years = (2000, 2025)
    output_dir = Path(project.paths["landfire_control_dataset"])
    summary_path = Path(project.paths["landfire_controls_summary"])
    layer_dirs = {
        "elev": Path(project.paths["landfire_elev_tiles"]),
        "asp": Path(project.paths["landfire_asp_tiles"]),
        "slpd": Path(project.paths["landfire_slpd_tiles"]),
    }
    layer_tiles = {
        name: functions.discover_landfire_tiles(directory)
        for name, directory in layer_dirs.items()
    }
    tile_metadata = {
        name: functions.validate_landfire_tiles(name, tiles)
        for name, tiles in layer_tiles.items()
    }
    tile_counts = {name: int(metadata["tile_count"]) for name, metadata in tile_metadata.items()}
    print("[input] LANDFIRE LF2020 tile counts: " + ", ".join(
        f"{name}={count}" for name, count in tile_counts.items()
    ), flush=True)

    singlepart_root = Path(project.paths["wui_singlepart_dataset"])
    if not singlepart_root.is_dir():
        raise FileNotFoundError(
            f"Missing WUI single-part dataset: {singlepart_root}. Run build_wui_singlepart_dataset first."
        )
    wui_inputs = []
    for vintage in (2000, 2010, 2020):
        year_dir = singlepart_root / f"year_{vintage}"
        shards = sorted(year_dir.glob("part-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No WUI single-part parquet shards found: {year_dir}")
        wui_inputs.extend((vintage, shard) for shard in shards)

    fire_point_path = functions.find_fire_point_source(project.paths["mtbs_fod_points_candidates"])
    perimeter_path = functions.find_spatial_source(
        project.paths["mtbs_perimeter_candidates"],
        project.paths.get("mtbs_search_roots", ()),
        "mtbs_perims_DD.shp",
        require_attribute_table=True,
    )
    print(f"[input] MTBS fire points: {fire_point_path}", flush=True)
    print(f"[input] MTBS fire areas: {perimeter_path}", flush=True)

    wui_output = output_dir / "wui_singlepart"
    point_output = output_dir / "fire_points" / "fire_points.parquet"
    area_output = output_dir / "fire_areas" / "fire_areas.parquet"
    summary_fields = (
        "level",
        "wui_vintage_year",
        "analysis_year_start",
        "analysis_year_end",
        "source_records",
        "complete_records",
        "partial_records",
        "no_data_records",
        "raster_vintage",
        "elev_tile_count",
        "asp_tile_count",
        "slpd_tile_count",
        "elev_tile_dir",
        "asp_tile_dir",
        "slpd_tile_dir",
        "input_path",
        "output_path",
    )

    def product_is_complete() -> bool:
        expected_wui = [
            wui_output / f"year_{vintage}" / shard.name
            for vintage, shard in wui_inputs
        ]
        expected = [*expected_wui, point_output, area_output]
        if not summary_path.is_file() or summary_path.stat().st_size == 0:
            return False
        if any(not path.is_file() or path.stat().st_size == 0 for path in expected):
            return False
        try:
            summary = pd.read_csv(summary_path)
        except Exception:
            return False
        if not {"level", "output_path"}.issubset(summary.columns):
            return False
        if len(summary.loc[summary["level"].eq("wui_singlepart")]) != len(wui_inputs):
            return False
        if len(summary.loc[summary["level"].eq("fire_points")]) != 1:
            return False
        if len(summary.loc[summary["level"].eq("fire_areas")]) != 1:
            return False
        summary_outputs = set(summary["output_path"].dropna().astype(str))
        return all(str(path) in summary_outputs for path in expected)

    overwrite = bool(project.config.get("overwrite_existing", False))
    if product_is_complete() and not overwrite:
        functions.skip_task("landfire_controls", output_dir, summary_path)
        return {"status": "skipped", "dataset": output_dir, "summary": summary_path}
    if output_dir.exists() and not overwrite:
        print("[rebuild] LANDFIRE controls are incomplete; rebuilding full control set.", flush=True)

    functions.ensure_dir(output_dir.parent)
    functions.ensure_dir(summary_path.parent)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".landfire_controls.", dir=output_dir.parent))
    temporary_summary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    intermediate_dir = Path(project.paths["intermediate"])
    functions.ensure_dir(intermediate_dir)
    summary_rows = []

    try:
        with tempfile.TemporaryDirectory(prefix="landfire_vrt.", dir=intermediate_dir) as vrt_dir:
            vrt_paths = {
                layer: functions.build_landfire_vrt(
                    layer_tiles[layer], Path(vrt_dir) / f"LF2020_{layer}.vrt"
                )
                for layer in ("elev", "asp", "slpd")
            }
            with rasterio.open(vrt_paths["elev"]) as reference:
                raster_crs = reference.crs
                if raster_crs is None:
                    raise ValueError(f"LANDFIRE Elev VRT has no CRS: {vrt_paths['elev']}")
                crs_definition = CRS.from_user_input(raster_crs)
                if (
                    not crs_definition.is_projected
                    or not crs_definition.axis_info
                    or any(abs(axis.unit_conversion_factor - 1.0) > 1e-9 for axis in crs_definition.axis_info[:2])
                ):
                    raise ValueError(
                        f"LANDFIRE raster CRS must use projected meter units for fire-area measurements; "
                        f"got {crs_definition}"
                    )
                raster_bounds = reference.bounds
                print(
                    f"[raster] CRS={raster_crs}; resolution={reference.res}; "
                    f"bounds={tuple(round(value, 2) for value in raster_bounds)}",
                    flush=True,
                )
            for layer, path in vrt_paths.items():
                with rasterio.open(path) as source:
                    if source.crs != raster_crs:
                        raise ValueError(f"LANDFIRE {layer} does not match Elev CRS: {source.crs}")

            capacity = functions.inspect_compute_capacity(memory_per_worker_gib=8.0, max_workers=8)
            workers = int(capacity["workers"])

            def process_wui_shard(item):
                vintage, input_path = item
                wui = gpd.read_parquet(input_path)
                required = {"wui_single_part_id", "wui_source_row_number", "wui_single_part_number", "year"}
                missing = sorted(required.difference(wui.columns))
                if missing:
                    raise ValueError(f"WUI shard lacks required fields {missing}: {input_path}")
                if wui.crs is None:
                    raise ValueError(f"WUI shard has no CRS: {input_path}")
                if not wui.empty and not wui["year"].eq(vintage).all():
                    raise ValueError(f"WUI vintage mismatch, expected {vintage}: {input_path}")
                wui = wui.to_crs(raster_crs).reset_index(drop=True)
                stats = functions.zonal_landfire_terrain_stats(wui.geometry.to_list(), vrt_paths)
                rows = pd.DataFrame({
                    "wui_single_part_id": wui["wui_single_part_id"],
                    "wui_source_row_number": wui["wui_source_row_number"],
                    "wui_single_part_number": wui["wui_single_part_number"],
                    "wui_vintage_year": np.full(len(wui), vintage, dtype=np.int16),
                    "landfire_raster_vintage": np.full(len(wui), 2020, dtype=np.int16),
                })
                for name, values in stats.items():
                    rows[f"lf2020_{name}"] = values
                rows["lf2020_status"] = np.select(
                    [
                        (rows["lf2020_elev_valid_pixels"] > 0)
                        & (rows["lf2020_slpd_valid_pixels"] > 0)
                        & (rows["lf2020_asp_valid_pixels"] > 0),
                        (rows["lf2020_elev_valid_pixels"] > 0)
                        | (rows["lf2020_slpd_valid_pixels"] > 0)
                        | (rows["lf2020_asp_valid_pixels"] > 0),
                    ],
                    ["complete", "partial"],
                    default="no_data",
                )
                out_path = temporary_dir / "wui_singlepart" / f"year_{vintage}" / input_path.name
                functions.ensure_dir(out_path.parent)
                rows.to_parquet(out_path, index=False, compression="zstd")
                status_counts = rows["lf2020_status"].value_counts()
                return {
                    "level": "wui_singlepart",
                    "wui_vintage_year": vintage,
                    "source_records": len(rows),
                    "complete_records": int(status_counts.get("complete", 0)),
                    "partial_records": int(status_counts.get("partial", 0)),
                    "no_data_records": int(status_counts.get("no_data", 0)),
                    "input_path": str(input_path),
                    "output_path": str(wui_output / f"year_{vintage}" / input_path.name),
                }

            tqdm.write(f"[progress] WUI controls: {len(wui_inputs):,} shards; workers={workers}")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_wui_shard, item) for item in wui_inputs]
                with tqdm(total=len(futures), desc="LANDFIRE WUI controls", unit="shard", dynamic_ncols=True) as progress:
                    for future in as_completed(futures):
                        summary_rows.append(future.result())
                        progress.update(1)

            point_info = __import__("pyogrio").read_info(fire_point_path)
            point_fields = list(point_info["fields"])
            point_map = {field.casefold(): field for field in point_fields}
            point_id_field = point_map.get("event_id") or point_map.get("fire_id")
            point_year_field = point_map.get("year") or point_map.get("fire_year")
            point_date_field = point_map.get("ig_date") or point_map.get("ignition_date")
            if point_id_field is None or (point_year_field is None and point_date_field is None):
                raise ValueError(
                    f"MTBS fire points need EVENT_ID/FIRE_ID and YEAR/FIRE_YEAR or IG_DATE; "
                    f"fields={point_fields}"
                )
            selected_point_fields = [point_id_field]
            if point_year_field:
                selected_point_fields.append(point_year_field)
            elif point_date_field:
                selected_point_fields.append(point_date_field)
            tqdm.write("[progress] Reading MTBS fire-point records")
            points = gpd.read_file(fire_point_path, columns=selected_point_fields, use_arrow=True)
            points["fire_source_row_number"] = np.arange(1, len(points) + 1, dtype=np.int64)
            if point_year_field:
                point_years = pd.to_numeric(points[point_year_field], errors="coerce")
            else:
                raw_dates = points[point_date_field].astype("string").str.strip()
                compact = raw_dates.str.fullmatch(r"\d{8}", na=False)
                parsed = pd.to_datetime(raw_dates.mask(compact), errors="coerce")
                if compact.any():
                    parsed.loc[compact] = pd.to_datetime(raw_dates.loc[compact], format="%Y%m%d", errors="coerce")
                point_years = parsed.dt.year
            points["fire_year"] = pd.to_numeric(point_years, errors="coerce")
            points["fire_event_id_normalized"] = points[point_id_field].map(functions.normalize_identifier)
            points = points.loc[
                points["fire_year"].between(*years)
                & points["fire_event_id_normalized"].notna()
                & points.geometry.notna()
                & ~points.geometry.is_empty
            ].copy()
            if points.geometry.geom_type.ne("Point").any():
                raise ValueError(f"MTBS fire-point layer contains non-Point geometries: {fire_point_path}")
            if points.empty:
                raise ValueError(f"No MTBS fire points in {years[0]}-{years[1]}: {fire_point_path}")
            points = points.to_crs(raster_crs)
            coordinates = list(zip(shapely.get_x(points.geometry.to_numpy()), shapely.get_y(points.geometry.to_numpy())))
            point_values = functions.sample_landfire_terrain_points(coordinates, vrt_paths)
            point_rows = pd.DataFrame({
                "fire_source_row_number": points["fire_source_row_number"].to_numpy(dtype=np.int64),
                "fire_event_id": points[point_id_field].astype("string").to_numpy(),
                "fire_event_id_normalized": points["fire_event_id_normalized"].astype("string").to_numpy(),
                "fire_year": points["fire_year"].to_numpy(dtype=np.int16),
                "fire_anchor_type": "mtbs_centroid",
                "landfire_raster_vintage": 2020,
            })
            point_rows["lf2020_elev_m"] = point_values["elev"]
            point_rows["lf2020_slpd_degrees"] = point_values["slpd"]
            point_rows["lf2020_asp_degrees"] = point_values["asp"]
            point_rows["lf2020_asp_is_flat"] = point_rows["lf2020_asp_degrees"].eq(-1).astype("int8")
            aspect_valid = point_rows["lf2020_asp_degrees"].between(0, 359)
            theta = np.deg2rad(point_rows.loc[aspect_valid, "lf2020_asp_degrees"].to_numpy(dtype=float))
            point_rows["lf2020_asp_northness"] = np.nan
            point_rows["lf2020_asp_eastness"] = np.nan
            point_rows.loc[aspect_valid, "lf2020_asp_northness"] = np.cos(theta)
            point_rows.loc[aspect_valid, "lf2020_asp_eastness"] = np.sin(theta)
            valid_layers = point_rows[["lf2020_elev_m", "lf2020_slpd_degrees"]].notna().sum(axis=1)
            valid_layers += point_rows["lf2020_asp_degrees"].notna().astype(int)
            point_rows["lf2020_status"] = np.select(
                [valid_layers.eq(3), valid_layers.gt(0)], ["complete", "partial"], default="no_data"
            )
            functions.ensure_dir((temporary_dir / "fire_points").parent)
            (temporary_dir / "fire_points").mkdir(parents=True, exist_ok=True)
            point_rows.to_parquet(temporary_dir / "fire_points" / "fire_points.parquet", index=False, compression="zstd")
            point_status = point_rows["lf2020_status"].value_counts()
            summary_rows.append({
                "level": "fire_points",
                "wui_vintage_year": None,
                "source_records": len(point_rows),
                "complete_records": int(point_status.get("complete", 0)),
                "partial_records": int(point_status.get("partial", 0)),
                "no_data_records": int(point_status.get("no_data", 0)),
                "input_path": str(fire_point_path),
                "output_path": str(output_dir / "fire_points" / "fire_points.parquet"),
            })
            print(f"[progress] Fire-point controls: {len(point_rows):,} records; years={int(point_rows.fire_year.min())}-{int(point_rows.fire_year.max())}", flush=True)

            perimeter_info = __import__("pyogrio").read_info(perimeter_path)
            perimeter_fields = list(perimeter_info["fields"])
            perimeter_map = {field.casefold(): field for field in perimeter_fields}
            perimeter_id_field = perimeter_map.get("event_id") or perimeter_map.get("fire_id")
            perimeter_year_field = perimeter_map.get("year") or perimeter_map.get("fire_year")
            perimeter_date_field = perimeter_map.get("ig_date") or perimeter_map.get("ignition_date")
            if perimeter_id_field is None or (perimeter_year_field is None and perimeter_date_field is None):
                raise ValueError(
                    f"MTBS perimeters need EVENT_ID/FIRE_ID and YEAR/FIRE_YEAR or IG_DATE; "
                    f"fields={perimeter_fields}"
                )
            selected_perimeter_fields = [perimeter_id_field]
            if perimeter_year_field:
                selected_perimeter_fields.append(perimeter_year_field)
            elif perimeter_date_field:
                selected_perimeter_fields.append(perimeter_date_field)
            tqdm.write("[progress] Reading MTBS burned-area perimeters")
            perimeters = gpd.read_file(perimeter_path, columns=selected_perimeter_fields, use_arrow=True)
            perimeters["fire_event_id_normalized"] = perimeters[perimeter_id_field].map(functions.normalize_identifier)
            if perimeter_year_field:
                perimeter_years = pd.to_numeric(perimeters[perimeter_year_field], errors="coerce")
            else:
                raw_dates = perimeters[perimeter_date_field].astype("string").str.strip()
                compact = raw_dates.str.fullmatch(r"\d{8}", na=False)
                parsed = pd.to_datetime(raw_dates.mask(compact), errors="coerce")
                if compact.any():
                    parsed.loc[compact] = pd.to_datetime(raw_dates.loc[compact], format="%Y%m%d", errors="coerce")
                perimeter_years = parsed.dt.year
            perimeters["fire_year"] = pd.to_numeric(perimeter_years, errors="coerce")
            perimeters = perimeters.loc[
                perimeters["fire_year"].between(*years)
                & perimeters["fire_event_id_normalized"].notna()
                & perimeters.geometry.notna()
                & ~perimeters.geometry.is_empty
            ].copy()
            if perimeters.empty:
                raise ValueError(f"No MTBS perimeters in {years[0]}-{years[1]}: {perimeter_path}")
            perimeters = perimeters.to_crs(raster_crs)
            perimeter_groups = []
            for (fire_id, fire_year), group in perimeters.groupby(
                ["fire_event_id_normalized", "fire_year"], sort=False
            ):
                geometries = group.geometry.to_numpy()
                valid = shapely.is_valid(geometries)
                if (~valid).any():
                    geometries = geometries.copy()
                    geometries[~valid] = shapely.make_valid(geometries[~valid])
                polygon = shapely.union_all(geometries)
                if polygon.geom_type not in {"Polygon", "MultiPolygon"}:
                    polygon = shapely.union_all(
                        [part for part in shapely.get_parts(polygon) if part.geom_type in {"Polygon", "MultiPolygon"}]
                    )
                if polygon.is_empty:
                    continue
                perimeter_groups.append((str(fire_id), int(fire_year), polygon))
            if not perimeter_groups:
                raise ValueError(f"No polygonal MTBS fire areas remain after grouping: {perimeter_path}")

            area_rows = []
            with tqdm(total=len(perimeter_groups), desc="LANDFIRE fire-area controls", unit="fire", dynamic_ncols=True) as progress:
                for fire_id, fire_year, geometry in perimeter_groups:
                    stats = functions.zonal_landfire_terrain_stats([geometry], vrt_paths)
                    row = {
                        "fire_event_id_normalized": fire_id,
                        "fire_year": fire_year,
                        "landfire_raster_vintage": 2020,
                        "fire_area_m2": float(geometry.area),
                    }
                    row.update({f"lf2020_{name}": values[0] for name, values in stats.items()})
                    layer_count = sum(
                        int(row.get(f"lf2020_{field}", 0) > 0)
                        for field in ("elev_valid_pixels", "slpd_valid_pixels", "asp_valid_pixels")
                    )
                    row["lf2020_status"] = "complete" if layer_count == 3 else ("partial" if layer_count else "no_data")
                    area_rows.append(row)
                    progress.update(1)
            area_frame = pd.DataFrame(area_rows)
            if area_frame.duplicated(["fire_event_id_normalized", "fire_year"]).any():
                raise ValueError("MTBS fire-area output has duplicate event-year keys")
            (temporary_dir / "fire_areas").mkdir(parents=True, exist_ok=True)
            area_frame.to_parquet(temporary_dir / "fire_areas" / "fire_areas.parquet", index=False, compression="zstd")
            area_status = area_frame["lf2020_status"].value_counts()
            summary_rows.append({
                "level": "fire_areas",
                "wui_vintage_year": None,
                "source_records": len(area_frame),
                "complete_records": int(area_status.get("complete", 0)),
                "partial_records": int(area_status.get("partial", 0)),
                "no_data_records": int(area_status.get("no_data", 0)),
                "input_path": str(perimeter_path),
                "output_path": str(output_dir / "fire_areas" / "fire_areas.parquet"),
            })

        backup_dir = None
        for row in summary_rows:
            row.update({
                "analysis_year_start": years[0],
                "analysis_year_end": years[1],
                "raster_vintage": 2020,
                "elev_tile_count": tile_counts["elev"],
                "asp_tile_count": tile_counts["asp"],
                "slpd_tile_count": tile_counts["slpd"],
                "elev_tile_dir": str(layer_dirs["elev"]),
                "asp_tile_dir": str(layer_dirs["asp"]),
                "slpd_tile_dir": str(layer_dirs["slpd"]),
            })
        summary_rows.sort(key=lambda row: (row["level"], row["wui_vintage_year"] or 0, row["input_path"]))
        functions.write_csv(summary_rows, temporary_summary, fieldnames=summary_fields)
        if output_dir.exists():
            backup_dir = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.old")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            output_dir.replace(backup_dir)
            temporary_dir.replace(output_dir)
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            temporary_dir.replace(output_dir)
        temporary_summary.replace(summary_path)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        temporary_summary.unlink(missing_ok=True)
        if "backup_dir" in locals() and backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise

    wui_records = sum(row["source_records"] for row in summary_rows if row["level"] == "wui_singlepart")
    print(f"[output] LANDFIRE controls: {output_dir}", flush=True)
    print(f"[output] Summary: {summary_path}", flush=True)
    print(
        f"[result] WUI rows={wui_records:,}; fire-point rows={len(point_rows):,}; "
        f"fire-area rows={len(area_frame):,}",
        flush=True,
    )
    return {
        "status": "built",
        "dataset": output_dir,
        "summary": summary_path,
        "wui_records": wui_records,
        "fire_point_records": len(point_rows),
        "fire_area_records": len(area_frame),
    }
