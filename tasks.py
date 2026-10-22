"""WUI dataset tasks.

Contents:
- build_wui_singlepart_dataset: combine 2000, 2010, and 2020 WUI blocks as single-part records
"""

from __future__ import annotations

import functions


def build_wui_singlepart_dataset(project):
    """Build source-preserving single-part WUI GeoParquet shards for three vintages."""
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

    years = (2000, 2010, 2020)
    fire_periods = {2000: (2000, 2009), 2010: (2010, 2019), 2020: (2020, 2024)}
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
        "fire_year_start",
        "fire_year_end",
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
        "fire_year_start",
        "fire_year_end",
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
            year_dir = temporary_dir / f"year_{year}"
            year_dir.mkdir(parents=True, exist_ok=True)
            source_rows = 0
            single_part_rows = 0
            dropped_nonpolygon_components = 0
            part_file_number = 0
            row_number_offset = 0

            with pyogrio.open_arrow(
                path,
                use_pyarrow=True,
                batch_size=chunk_size,
            ) as (_, reader):
                for batch in reader:
                    attributes = batch.to_pandas()
                    if geometry_column not in attributes:
                        raise ValueError(f"Could not find geometry column {geometry_column!r} in {path}")
                    wkb = attributes.pop(geometry_column)
                    batch_rows = len(attributes)
                    if batch_rows == 0:
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
                        continue

                    parts = frame.explode(index_parts=False).reset_index(drop=True)
                    polygon_mask = parts.geom_type.eq("Polygon")
                    dropped_nonpolygon_components += int((~polygon_mask).sum())
                    parts = parts.loc[polygon_mask].copy()
                    if parts.empty:
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
                    parts["fire_year_start"], parts["fire_year_end"] = fire_periods[year]
                    parts["wui_single_part_id"] = (
                        str(year)
                        + "_"
                        + parts["wui_source_row_number"].astype("string")
                        + "_"
                        + parts["wui_single_part_number"].astype("string")
                    )
                    parts = parts[output_fields]
                    out_part = year_dir / f"part-{part_file_number:06d}.parquet"
                    parts.to_parquet(out_part, index=False, compression="zstd")
                    part_file_number += 1
                    single_part_rows += len(parts)

            if source_rows == 0 or single_part_rows == 0:
                raise RuntimeError(f"No WUI polygon records processed for vintage {year}: {path}")
            summary_rows.append(
                {
                    "year": year,
                    "fire_year_start": fire_periods[year][0],
                    "fire_year_end": fire_periods[year][1],
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
                "fire_year_start",
                "fire_year_end",
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
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        temporary_summary.unlink(missing_ok=True)
        raise

    print(f"[output] Single-part WUI dataset: {output_dir}")
    print(f"[output] Vintage summary: {summary_path}")
    return {"status": "built", "dataset": output_dir, "summary": summary_path}
