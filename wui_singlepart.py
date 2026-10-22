"""Single-part WUI fire-facing panel builder ported from archived analysis."""

from __future__ import annotations

from pathlib import Path

import functions

CONUS_EXTENT = (-125.0, -66.5, 24.0, 50.0)


YEAR_MIN = 2000


YEAR_MAX = 2024


def _skip_task(name: str, *paths: Path) -> None:
    targets = ", ".join(str(path) for path in paths)
    print(f"[skip] {name}: existing targets: {targets}")


def _sql_list(values) -> str:
    clean = [str(value).replace("'", "''") for value in values]
    return "(" + ",".join(f"'{value}'" for value in clean) + ")"


def _wui_vintage_for_fire_year(year: pd.Series) -> pd.Series:
    import pandas as pd
    year = pd.to_numeric(year, errors="coerce")
    return pd.Series(
        pd.NA,
        index=year.index,
        dtype="Int64",
    ).mask(year.between(2000, 2009), 2000).mask(year.between(2010, 2019), 2010).mask(year.ge(2020), 2020)


def _sector_polygon(cx, cy, bearing_degrees, radius_m, half_angle_degrees, n_arc_points=17):
    import pandas as pd
    import math

    from shapely.geometry import Polygon

    if not all(pd.notna(value) for value in [cx, cy, bearing_degrees]) or radius_m <= 0:
        return Polygon()
    angles = [bearing_degrees - half_angle_degrees + (2.0 * half_angle_degrees * step / (n_arc_points - 1)) for step in range(n_arc_points)]
    coords = [(float(cx), float(cy))]
    for angle in angles:
        radians = math.radians(float(angle))
        coords.append((float(cx) + radius_m * math.cos(radians), float(cy) + radius_m * math.sin(radians)))
    coords.append((float(cx), float(cy)))
    return Polygon(coords)


def _load_completed_facts_treatment_polygons_for_links(path: Path, states: list[str]) -> gpd.GeoDataFrame:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyogrio
    import shapely

    columns = [
        "SUID",
        "ACTIVITY_C",
        "ACTIVITY",
        "STATE_ABBR",
        "DATE_COMPL",
        "FY_AWARDED",
        "FISCAL_Y_1",
        "STAGE_VALU",
        "CAT_NM",
        "METHOD",
        "TREATMENT1",
        "FACTS_ID",
        "UK",
        "GIS_ACRES",
    ]
    xmin, xmax, ymin, ymax = CONUS_EXTENT
    where = "STAGE_VALU = 'Accomplished'"
    if states:
        where = f"{where} AND STATE_ABBR IN {_sql_list(states)}"
    gdf = pyogrio.read_dataframe(
        path,
        columns=columns,
        bbox=(xmin, ymin, xmax, ymax),
        where=where,
        fid_as_index=True,
        use_arrow=True,
    )
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)
    gdf["facts_fid"] = pd.to_numeric(pd.Series(gdf.index, index=gdf.index), errors="coerce").astype("Int64")
    gdf["treat_id"] = gdf["facts_fid"].astype("string")
    gdf["FY_AWARDED"] = pd.to_numeric(gdf["FY_AWARDED"], errors="coerce")
    gdf["FISCAL_Y_1"] = pd.to_numeric(gdf["FISCAL_Y_1"], errors="coerce")
    gdf["GIS_ACRES"] = pd.to_numeric(gdf["GIS_ACRES"], errors="coerce")
    gdf["treatment_completed_date"] = pd.to_datetime(gdf["DATE_COMPL"], errors="coerce")

    completed_year = gdf["treatment_completed_date"].dt.year.astype("float64")
    completed_year_source = pd.Series(pd.NA, index=gdf.index, dtype="string")
    completed_year_source.loc[completed_year.notna()] = "DATE_COMPL"
    fiscal_year = gdf["FISCAL_Y_1"].where(gdf["FISCAL_Y_1"].between(1900, 2100))
    use_fiscal = completed_year.isna() & fiscal_year.notna()
    completed_year.loc[use_fiscal] = fiscal_year.loc[use_fiscal]
    completed_year_source.loc[use_fiscal] = "FISCAL_Y_1"
    award_year = gdf["FY_AWARDED"].where(gdf["FY_AWARDED"].between(1900, 2100))
    use_award = completed_year.isna() & award_year.notna()
    completed_year.loc[use_award] = award_year.loc[use_award]
    completed_year_source.loc[use_award] = "FY_AWARDED"

    category = gdf["CAT_NM"].astype("string").str.strip().str.casefold().fillna("")
    text = (
        gdf["CAT_NM"].astype("string").fillna("")
        + " "
        + gdf["ACTIVITY"].astype("string").fillna("")
        + " "
        + gdf["METHOD"].astype("string").fillna("")
        + " "
        + gdf["TREATMENT1"].astype("string").fillna("")
    ).str.casefold()
    treat_is_prescribed_fire = category.eq("fire") | text.str.contains("prescribed|\\bburn\\b|underburn|broadcast burn|pile burn", regex=True, na=False)
    treat_is_mechanical = category.eq("mechanical") | text.str.contains("mechanical|mastication|thin|chipping|mowing|cutting", regex=True, na=False)
    treat_is_both = treat_is_prescribed_fire & treat_is_mechanical

    keep = (
        completed_year.between(YEAR_MIN - 10, YEAR_MAX)
        & gdf["GIS_ACRES"].gt(0)
        & ~category.eq("wildfire non-treatment")
        & ~gdf.geometry.isna()
        & ~gdf.geometry.is_empty
    )
    gdf = gdf.loc[keep].copy()
    gdf["treatment_completed_year"] = completed_year.loc[keep].astype("int64").to_numpy()
    gdf["treatment_completed_year_source"] = completed_year_source.loc[keep].astype("string").to_numpy()
    gdf["treatment_acres"] = gdf["GIS_ACRES"].astype(float)
    gdf["treat_is_prescribed_fire"] = treat_is_prescribed_fire.loc[keep].astype(bool).to_numpy()
    gdf["treat_is_mechanical"] = treat_is_mechanical.loc[keep].astype(bool).to_numpy()
    gdf["treat_is_both_polygon"] = treat_is_both.loc[keep].astype(bool).to_numpy()
    gdf["treatment_type"] = np.select(
        [
            gdf["treat_is_both_polygon"],
            gdf["treat_is_prescribed_fire"],
            gdf["treat_is_mechanical"],
        ],
        ["both", "prescribed_fire", "mechanical"],
        default="other",
    )
    if gdf.crs != "EPSG:5070":
        gdf = gdf.to_crs("EPSG:5070")
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = shapely.make_valid(gdf.loc[invalid, "geometry"].to_numpy())
    return gdf[
        [
            "treat_id",
            "facts_fid",
            "SUID",
            "FACTS_ID",
            "UK",
            "ACTIVITY_C",
            "ACTIVITY",
            "STATE_ABBR",
            "DATE_COMPL",
            "treatment_completed_date",
            "FY_AWARDED",
            "FISCAL_Y_1",
            "treatment_completed_year",
            "treatment_completed_year_source",
            "CAT_NM",
            "METHOD",
            "TREATMENT1",
            "GIS_ACRES",
            "treatment_acres",
            "treat_is_prescribed_fire",
            "treat_is_mechanical",
            "treat_is_both_polygon",
            "treatment_type",
            "geometry",
        ]
    ].reset_index(drop=True)


def _load_direct_mtbs_wildfires(p, states: set[str]) -> gpd.GeoDataFrame:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyogrio
    import shapely

    columns = [
        "FIRE_ID",
        "FIRE_NAME",
        "FIRE_TYPE",
        "YEAR",
        "STARTMONTH",
        "ACRES",
        "Event_ID",
        "Ig_Date",
    ]
    fires = pyogrio.read_dataframe(
        p.paths["mtbs_perimeters"],
        columns=columns,
        use_arrow=True,
    )
    fires = gpd.GeoDataFrame(fires, geometry="geometry", crs=fires.crs)
    fires["fire_id"] = fires["FIRE_ID"].astype("string")
    fires["fire_state"] = fires["fire_id"].str[:2]
    fires["fire_year"] = pd.to_numeric(fires["YEAR"], errors="coerce")
    fires["ignition_date"] = pd.to_datetime(fires["Ig_Date"], errors="coerce", utc=True).dt.tz_convert(None)
    keep = (
        fires["FIRE_TYPE"].eq("Wildfire")
        & fires["fire_year"].between(YEAR_MIN, YEAR_MAX)
        & fires["fire_state"].isin(states)
        & fires["ignition_date"].notna()
        & ~fires.geometry.isna()
        & ~fires.geometry.is_empty
    )
    fires = fires.loc[keep].copy()
    if fires.empty:
        raise RuntimeError("No direct MTBS CONUS wildfire perimeters are available for 2000-2024.")
    if fires.crs != "EPSG:5070":
        fires = fires.to_crs("EPSG:5070")
    invalid = ~fires.geometry.is_valid
    if invalid.any():
        fires.loc[invalid, "geometry"] = shapely.make_valid(fires.loc[invalid, "geometry"].to_numpy())

    points = pyogrio.read_dataframe(
        p.paths["mtbs_fod_points"],
        columns=["FIRE_ID", "Event_ID", "Ig_Date", "FIRE_TYPE", "YEAR"],
        use_arrow=True,
    )
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=points.crs)
    points["fire_id"] = points["FIRE_ID"].astype("string")
    points = points.loc[points["fire_id"].isin(fires["fire_id"])].copy()
    if not points.empty:
        if points.crs != "EPSG:5070":
            points = points.to_crs("EPSG:5070")
        points["origin_x"] = points.geometry.x
        points["origin_y"] = points.geometry.y
        points = points.drop_duplicates("fire_id")
        fires = fires.merge(points[["fire_id", "origin_x", "origin_y"]], on="fire_id", how="left")
    else:
        fires["origin_x"] = np.nan
        fires["origin_y"] = np.nan

    missing_origin = ~(np.isfinite(fires["origin_x"]) & np.isfinite(fires["origin_y"]))
    if missing_origin.any():
        centroid = fires.loc[missing_origin, "geometry"].centroid
        fires.loc[missing_origin, "origin_x"] = centroid.x
        fires.loc[missing_origin, "origin_y"] = centroid.y

    fires["fire_year"] = fires["fire_year"].astype(int)
    fires["mtbs_acres"] = pd.to_numeric(fires["ACRES"], errors="coerce")
    fires["fire_idx"] = np.arange(len(fires), dtype=np.int32)
    return fires[
        [
            "fire_idx",
            "fire_id",
            "FIRE_NAME",
            "FIRE_TYPE",
            "fire_year",
            "STARTMONTH",
            "mtbs_acres",
            "Event_ID",
            "ignition_date",
            "fire_state",
            "origin_x",
            "origin_y",
            "geometry",
        ]
    ].rename(columns={"FIRE_NAME": "fire_name"}).reset_index(drop=True)


def _add_pre5_fireface_radius_variants_from_existing_panel(
    p,
    *,
    panel_path: Path,
    links_path: Path,
    radius_panel_paths: dict[str, Path],
    summary_path: Path,
    by_year_type_path: Path,
    shield_radius_specs: list[tuple[float, str]],
    shield_half_angle_degrees: float,
    chunk_size: int,
) -> None:
    """Augment an existing fire-WUI panel when raw WUI geometry is unavailable."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    if not panel_path.exists():
        raise FileNotFoundError(f"Missing existing fire-facing panel for radius augmentation: {panel_path}")

    panel = pd.read_parquet(panel_path)
    states = sorted(panel["fire_state"].dropna().astype(str).unique().tolist())
    treatments = _load_completed_facts_treatment_polygons_for_links(p.paths["treatment_polygons"], states)
    treatment_cols = [
        "treat_id",
        "facts_fid",
        "SUID",
        "FACTS_ID",
        "UK",
        "ACTIVITY_C",
        "ACTIVITY",
        "STATE_ABBR",
        "DATE_COMPL",
        "treatment_completed_date",
        "FY_AWARDED",
        "FISCAL_Y_1",
        "treatment_completed_year",
        "treatment_completed_year_source",
        "CAT_NM",
        "METHOD",
        "TREATMENT1",
        "GIS_ACRES",
        "treatment_acres",
        "treat_is_prescribed_fire",
        "treat_is_mechanical",
        "treat_is_both_polygon",
        "treatment_type",
        "geometry",
    ]
    sector_base_cols = [
        "panel_row_id",
        "fire_id",
        "fire_name",
        "fire_year",
        "fire_month",
        "fire_season",
        "fire_state",
        "ignition_date",
        "wui_id",
        "wui_part_number",
        "wui_part_id",
        "direction",
        "distance_bin_2km",
    ]
    link_base_cols = [
        *sector_base_cols,
        "sector_label",
        "sector_orientation",
        "sector_radius_m",
        "sector_radius_label",
        "sector_half_angle_degrees",
        "sector_bearing_degrees",
        *[col for col in treatment_cols if col != "geometry"],
    ]
    links = pd.read_parquet(links_path) if links_path.exists() else pd.DataFrame(columns=link_base_cols)

    def _build_sector_links(radius_m: float, radius_label: str) -> pd.DataFrame:
        link_frames = []
        sector_label = f"fireface_{radius_label}"
        for start in range(0, len(panel), chunk_size):
            stop = min(start + chunk_size, len(panel))
            part = panel.iloc[start:stop].copy()
            bearing = part["bearing_wui_to_ignition_degrees"].to_numpy(dtype=float)
            sector_geometry = [
                _sector_polygon(row.wui_part_x, row.wui_part_y, sector_bearing, radius_m, shield_half_angle_degrees)
                for row, sector_bearing in zip(part.itertuples(index=False), bearing)
            ]
            sector_frame = part[sector_base_cols].copy()
            sector_frame["sector_label"] = sector_label
            sector_frame["sector_orientation"] = "fireface"
            sector_frame["sector_radius_m"] = radius_m
            sector_frame["sector_radius_label"] = radius_label
            sector_frame["sector_half_angle_degrees"] = shield_half_angle_degrees
            sector_frame["sector_bearing_degrees"] = bearing
            sectors = gpd.GeoDataFrame(sector_frame, geometry=sector_geometry, crs="EPSG:5070")
            sectors = sectors.loc[~sectors.geometry.is_empty].copy()
            if sectors.empty:
                continue
            joined = gpd.sjoin(
                sectors,
                treatments[treatment_cols],
                how="inner",
                predicate="intersects",
            ).drop(columns=["index_right"], errors="ignore")
            if joined.empty:
                continue
            joined = joined.drop_duplicates(["sector_label", "panel_row_id", "treat_id"])
            link_frames.append(pd.DataFrame(joined.drop(columns="geometry")))
        return pd.concat(link_frames, ignore_index=True) if link_frames else pd.DataFrame(columns=link_base_cols)

    existing_sector_labels = set(links["sector_label"].astype(str).unique()) if "sector_label" in links else set()
    new_link_frames = [links]
    for radius_m, radius_label in shield_radius_specs:
        sector_label = f"fireface_{radius_label}"
        if sector_label not in existing_sector_labels:
            new_link_frames.append(_build_sector_links(radius_m, radius_label))
    links = pd.concat(new_link_frames, ignore_index=True)
    if not links.empty:
        links = links.drop_duplicates(["sector_label", "panel_row_id", "treat_id"]).reset_index(drop=True)
        links["treatment_completed_date"] = pd.to_datetime(links["treatment_completed_date"], errors="coerce")
        links["ignition_date"] = pd.to_datetime(links["ignition_date"], errors="coerce")
        links["treatment_completed_year"] = pd.to_numeric(links["treatment_completed_year"], errors="coerce")
        links["fire_year"] = pd.to_numeric(links["fire_year"], errors="coerce")
        links["completed_before_ignition"] = False
        has_exact_date = links["treatment_completed_date"].notna() & links["ignition_date"].notna()
        links.loc[has_exact_date, "completed_before_ignition"] = (
            links.loc[has_exact_date, "treatment_completed_date"] < links.loc[has_exact_date, "ignition_date"]
        )
        year_only = ~has_exact_date & links["treatment_completed_year"].notna() & links["fire_year"].notna()
        links.loc[year_only, "completed_before_ignition"] = links.loc[year_only, "treatment_completed_year"] < links.loc[year_only, "fire_year"]
        links["days_before_ignition"] = np.nan
        links.loc[has_exact_date, "days_before_ignition"] = (
            links.loc[has_exact_date, "ignition_date"] - links.loc[has_exact_date, "treatment_completed_date"]
        ).dt.days
        links["years_before_fire"] = links["fire_year"] - links["treatment_completed_year"]
        links["completed_before_fire_year"] = links["years_before_fire"].ge(1).astype(int)
        links["prior_5yr"] = links["years_before_fire"].between(1, 5).astype(int)
        links["prior_10yr"] = links["years_before_fire"].between(1, 10).astype(int)
        links["current_or_post_fire_year"] = links["years_before_fire"].le(0).astype(int)

    def _add_pre5_aggregates(radius_label: str) -> None:
        link_df = links.loc[links["sector_label"].astype(str).eq(f"fireface_{radius_label}")].copy()
        n = len(panel)
        count = np.zeros(n, dtype="int32")
        acres = np.zeros(n, dtype=float)
        prescribed_acres = np.zeros(n, dtype=float)
        mechanical_acres = np.zeros(n, dtype=float)
        prescribed_any = np.zeros(n, dtype="int8")
        mechanical_any = np.zeros(n, dtype="int8")
        other_any = np.zeros(n, dtype="int8")
        most_recent_year = np.full(n, np.nan, dtype=float)
        most_recent_date = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns]")
        if not link_df.empty:
            timed = link_df.loc[link_df["prior_5yr"].astype(float).gt(0)].copy()
            if not timed.empty:
                timed["prescribed_fire_treatment_acres"] = timed["treatment_acres"].astype(float) * timed["treat_is_prescribed_fire"].astype(float)
                timed["mechanical_treatment_acres"] = timed["treatment_acres"].astype(float) * timed["treat_is_mechanical"].astype(float)
                timed["treat_is_other"] = ~(timed["treat_is_prescribed_fire"].astype(bool) | timed["treat_is_mechanical"].astype(bool))
                grouped = timed.groupby("panel_row_id", observed=True).agg(
                    treat_count=("treat_id", "nunique"),
                    treated_acres=("treatment_acres", "sum"),
                    prescribed_fire_acres=("prescribed_fire_treatment_acres", "sum"),
                    mechanical_acres=("mechanical_treatment_acres", "sum"),
                    prescribed_fire_any=("treat_is_prescribed_fire", "max"),
                    mechanical_any=("treat_is_mechanical", "max"),
                    other_any=("treat_is_other", "max"),
                    most_recent_treatment_year=("treatment_completed_year", "max"),
                    most_recent_treatment_date=("treatment_completed_date", "max"),
                )
                row_id = grouped.index.to_numpy(dtype="int64")
                count[row_id] = grouped["treat_count"].to_numpy(dtype="int32")
                acres[row_id] = grouped["treated_acres"].to_numpy(dtype=float)
                prescribed_acres[row_id] = grouped["prescribed_fire_acres"].to_numpy(dtype=float)
                mechanical_acres[row_id] = grouped["mechanical_acres"].to_numpy(dtype=float)
                prescribed_any[row_id] = grouped["prescribed_fire_any"].astype(bool).to_numpy(dtype="int8")
                mechanical_any[row_id] = grouped["mechanical_any"].astype(bool).to_numpy(dtype="int8")
                other_any[row_id] = grouped["other_any"].astype(bool).to_numpy(dtype="int8")
                most_recent_year[row_id] = grouped["most_recent_treatment_year"].to_numpy(dtype=float)
                most_recent_date.iloc[row_id] = pd.to_datetime(grouped["most_recent_treatment_date"], errors="coerce").to_numpy()
        suffix = f"pre5_{radius_label}"
        any_treat = count > 0
        both_type = prescribed_any.astype(bool) & mechanical_any.astype(bool)
        mechanical_only = mechanical_any.astype(bool) & ~prescribed_any.astype(bool)
        prescribed_only = prescribed_any.astype(bool) & ~mechanical_any.astype(bool)
        exposure_type = np.select(
            [both_type, prescribed_only, mechanical_only, any_treat],
            ["both", "prescribed_fire", "mechanical", "other"],
            default="none",
        )
        panel[f"wui_fireface_treat_count_{suffix}"] = count
        panel[f"wui_fireface_treated_acres_{suffix}"] = acres
        panel[f"wui_fireface_prescribed_fire_acres_{suffix}"] = prescribed_acres
        panel[f"wui_fireface_mechanical_acres_{suffix}"] = mechanical_acres
        panel[f"wui_fireface_any_treat_{suffix}"] = any_treat.astype(float)
        panel[f"log1p_wui_fireface_treated_acres_{suffix}"] = np.log1p(acres)
        panel[f"wui_fireface_treated_acres_100_{suffix}"] = acres / 100.0
        panel[f"wui_fireface_prescribed_fire_any_{suffix}"] = prescribed_any.astype(float)
        panel[f"wui_fireface_mechanical_any_{suffix}"] = mechanical_any.astype(float)
        panel[f"wui_fireface_other_any_{suffix}"] = other_any.astype(float)
        panel[f"wui_fireface_prescribed_fire_only_{suffix}"] = prescribed_only.astype(float)
        panel[f"wui_fireface_mechanical_only_{suffix}"] = mechanical_only.astype(float)
        panel[f"wui_fireface_both_type_{suffix}"] = both_type.astype(float)
        panel[f"wui_fireface_exposure_type_{suffix}"] = exposure_type
        panel[f"wui_fireface_most_recent_treatment_year_{suffix}"] = most_recent_year
        panel[f"wui_fireface_most_recent_treatment_date_{suffix}"] = most_recent_date.to_numpy()

    for _, radius_label in shield_radius_specs:
        _add_pre5_aggregates(radius_label)

    functions.ensure_dir(panel_path.parent)
    panel.to_parquet(panel_path, index=False)
    links.to_parquet(links_path, index=False)

    base_radius_cols = [
        col
        for col in panel.columns
        if not col.startswith("wui_fireface_") and not col.startswith("log1p_wui_fireface_")
    ]
    for radius_m, radius_label in shield_radius_specs:
        if radius_label not in radius_panel_paths:
            continue
        suffix = f"_{radius_label}"
        radius_cols = [
            col
            for col in panel.columns
            if (
                (col.startswith("wui_fireface_") or col.startswith("log1p_wui_fireface_"))
                and col.endswith(suffix)
            )
        ]
        radius_panel = panel[list(dict.fromkeys([*base_radius_cols, *radius_cols]))].copy()
        radius_panel["shield_radius_km"] = radius_m / 1000.0
        radius_panel["shield_radius_label"] = radius_label
        for col in radius_cols:
            radius_panel[col.removesuffix(suffix)] = radius_panel[col]
        radius_panel.to_parquet(radius_panel_paths[radius_label], index=False)

    update_rows = [
        {"metric": "fire_facing_sector_radii", "value": ",".join(str(radius_m / 1000.0) for radius_m, _ in shield_radius_specs), "unit": "km from WUI representative point"},
        *[
            {
                "metric": f"fireface_{radius_label}_treatment_link_rows",
                "value": int((links["sector_label"].astype(str) == f"fireface_{radius_label}").sum()),
                "unit": "sector-treatment links",
            }
            for _, radius_label in shield_radius_specs
        ],
        *[
            {
                "metric": f"treated_rows_pre5_{radius_label}",
                "value": int(panel[f"wui_fireface_any_treat_pre5_{radius_label}"].sum()),
                "unit": "fire-WUI single-part rows",
            }
            for _, radius_label in shield_radius_specs
        ],
        *[
            {
                "metric": f"panel_output_{radius_label}",
                "value": str(radius_panel_paths[radius_label]),
                "unit": "path",
            }
            for radius_label in radius_panel_paths
        ],
    ]
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(columns=["metric", "value", "unit"])
    update = pd.DataFrame(update_rows)
    summary = summary.loc[~summary["metric"].astype(str).isin(update["metric"].astype(str))].copy()
    updated_summary = pd.concat([summary, update], ignore_index=True)
    functions.write_csv(
        updated_summary.to_dict("records"),
        summary_path,
        fieldnames=updated_summary.columns,
    )

    if by_year_type_path.exists():
        by_year_type = pd.read_csv(by_year_type_path)
    else:
        by_year_type = pd.DataFrame()
    by_year_frames = []
    for radius_label in ["0p5km", "1km"]:
        sector_links = links.loc[links["sector_label"].astype(str).eq(f"fireface_{radius_label}")].copy()
        timed = sector_links.loc[sector_links["prior_5yr"].astype(float).gt(0)].copy()
        if timed.empty:
            continue
        by_year_frames.append(
            timed.groupby(["fire_year", "treatment_type"], as_index=False, observed=True)
            .agg(
                treatment_link_rows=("treat_id", "size"),
                treated_fire_wui_part_rows=("panel_row_id", "nunique"),
                treatment_polygons=("treat_id", "nunique"),
                treated_acres=("treatment_acres", "sum"),
                fires=("fire_id", "nunique"),
                wui_single_parts=("wui_part_id", "nunique"),
            )
            .assign(timing_window="prior_5yr", sector_label=f"fireface_{radius_label}")
        )
    if by_year_frames:
        new_by_year = pd.concat(by_year_frames, ignore_index=True)
        if not by_year_type.empty and "sector_label" in by_year_type:
            by_year_type = by_year_type.loc[
                ~by_year_type["sector_label"].astype(str).isin(["fireface_0p5km", "fireface_1km"])
            ].copy()
        updated_by_year_type = pd.concat([by_year_type, new_by_year], ignore_index=True)
        functions.write_csv(
            updated_by_year_type.to_dict("records"),
            by_year_type_path,
            fieldnames=updated_by_year_type.columns,
        )


def build_wui_singlepart_fireface_regression_data(p):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyogrio
    import shapely
    out_panel = p.paths["processed"] / "wui_singlepart_fireface_regression_panel.parquet"
    out_links = p.paths["processed"] / "wui_singlepart_fireface_treatment_links.parquet"
    radius_panel_paths = {
        "0p5km": p.paths["processed"] / "wui_singlepart_fireface_regression_panel_0p5km.parquet",
        "1km": p.paths["processed"] / "wui_singlepart_fireface_regression_panel_1km.parquet",
        "5km": p.paths["processed"] / "wui_singlepart_fireface_regression_panel_5km.parquet",
    }
    summary_path = p.paths["tables"] / "wui_singlepart_fireface_regression_summary.csv"
    by_year_type_path = p.paths["tables"] / "wui_singlepart_fireface_treatment_by_year_type.csv"
    if functions.should_skip(
        p.config["overwrite_existing"],
        [out_panel, out_links, *radius_panel_paths.values(), summary_path, by_year_type_path],
    ):
        _skip_task(
            "build_wui_singlepart_fireface_regression_data",
            out_panel,
            out_links,
            *radius_panel_paths.values(),
            summary_path,
            by_year_type_path,
        )
        return

    acre_m2 = 4046.8564224
    risk_buffer_m = 5_000.0
    shield_radius_m = 2_500.0
    robustness_shield_radius_m = 5_000.0
    shield_radius_specs = [
        (500.0, "0p5km"),
        (1_000.0, "1km"),
        (shield_radius_m, "2p5km"),
        (robustness_shield_radius_m, "5km"),
    ]
    shield_half_angle_degrees = 45.0
    albers = "EPSG:5070"
    chunk_size = 25_000
    conus_states = {
        "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "ID", "IL",
        "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
        "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR",
        "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
        "WY",
    }
    wui_classes = [
        "Low_Dens_Interface",
        "Med_Dens_Interface",
        "High_Dens_Interface",
        "Low_Dens_Intermix",
        "Med_Dens_Intermix",
        "High_Dens_Intermix",
    ]

    fires = _load_direct_mtbs_wildfires(p, conus_states)
    fires["fire_month"] = pd.to_numeric(fires["STARTMONTH"], errors="coerce").astype("Int64")
    fires["fire_season"] = np.select(
        [
            fires["fire_month"].isin([12, 1, 2]),
            fires["fire_month"].isin([3, 4, 5]),
            fires["fire_month"].isin([6, 7, 8]),
            fires["fire_month"].isin([9, 10, 11]),
        ],
        ["winter", "spring", "summer", "fall"],
        default="unknown",
    )
    fires["mtbs_size_class"] = pd.cut(
        pd.to_numeric(fires["mtbs_acres"], errors="coerce"),
        bins=[0, 1_000, 5_000, 10_000, 50_000, np.inf],
        labels=["0-1k acres", "1k-5k acres", "5k-10k acres", "10k-50k acres", "50k+ acres"],
        include_lowest=True,
    ).astype("string")
    states = sorted(set(fires["fire_state"].dropna().astype(str)) & conus_states)
    if not states:
        states = sorted(conus_states)

    wui_path = p.paths["wui_geodatabase"]
    if not wui_path.exists():
        if out_panel.exists():
            _add_pre5_fireface_radius_variants_from_existing_panel(
                p,
                panel_path=out_panel,
                links_path=out_links,
                radius_panel_paths=radius_panel_paths,
                summary_path=summary_path,
                by_year_type_path=by_year_type_path,
                shield_radius_specs=shield_radius_specs,
                shield_half_angle_degrees=shield_half_angle_degrees,
                chunk_size=chunk_size,
            )
            return
        raise FileNotFoundError(f"Missing CONUS WUI geodatabase and existing fire-facing panel: {wui_path}")
    wui_layer = "CONUS_WUI_block_1990_2020_change_v4"
    wui_where = (
        f"STATE IN {_sql_list(states)} AND ("
        f"WUICLASS2000 IN {_sql_list(wui_classes)} OR "
        f"WUICLASS2010 IN {_sql_list(wui_classes)} OR "
        f"WUICLASS2020 IN {_sql_list(wui_classes)})"
    )
    wui_columns = [
        "BLK20",
        "STATE",
        "HU2000",
        "HU2010",
        "HU2020",
        "HUDEN2000",
        "HUDEN2010",
        "HUDEN2020",
        "WUICLASS2000",
        "WUICLASS2010",
        "WUICLASS2020",
    ]
    wui = pyogrio.read_dataframe(
        wui_path,
        layer=wui_layer,
        columns=wui_columns,
        where=wui_where,
        fid_as_index=True,
        use_arrow=True,
    )
    wui = gpd.GeoDataFrame(wui, geometry="geometry", crs=wui.crs)
    if wui.crs != albers:
        wui = wui.to_crs(albers)
    wui["wui_id"] = pd.to_numeric(pd.Series(wui.index, index=wui.index), errors="coerce").astype("int64")
    wui["wui_parent_geom_type"] = wui.geom_type.astype("string")
    invalid_wui = ~wui.geometry.is_valid
    if invalid_wui.any():
        wui.loc[invalid_wui, "geometry"] = shapely.make_valid(wui.loc[invalid_wui, "geometry"].to_numpy())

    wui_parts = wui.explode(index_parts=False).reset_index(drop=True)
    wui_parts = wui_parts.loc[
        ~wui_parts.geometry.isna()
        & ~wui_parts.geometry.is_empty
        & wui_parts.geom_type.eq("Polygon")
    ].copy()
    wui_parts["wui_part_number"] = wui_parts.groupby("wui_id", observed=True).cumcount() + 1
    wui_parts["wui_part_id"] = wui_parts["wui_id"].astype(str) + "-" + wui_parts["wui_part_number"].astype(str)
    wui_parts["wui_part_geom_type"] = wui_parts.geom_type.astype("string")
    wui_parts["wui_part_area_m2"] = wui_parts.geometry.area
    representative = wui_parts.geometry.representative_point()
    wui_parts["wui_part_x"] = representative.x
    wui_parts["wui_part_y"] = representative.y
    wui_parent_count = int(wui["wui_id"].nunique())
    wui_part_count = int(len(wui_parts))

    fire_buffers = fires[
        [
            "fire_idx",
            "fire_id",
            "fire_name",
            "FIRE_TYPE",
            "fire_year",
            "fire_month",
            "fire_season",
            "mtbs_acres",
            "mtbs_size_class",
            "Event_ID",
            "ignition_date",
            "fire_state",
            "origin_x",
            "origin_y",
            "geometry",
        ]
    ].copy()
    fire_buffers["geometry"] = fire_buffers.geometry.buffer(risk_buffer_m)
    risk = gpd.sjoin(
        wui_parts[
            [
                "wui_id",
                "wui_part_number",
                "wui_part_id",
                "wui_parent_geom_type",
                "wui_part_geom_type",
                "BLK20",
                "STATE",
                "HU2000",
                "HU2010",
                "HU2020",
                "HUDEN2000",
                "HUDEN2010",
                "HUDEN2020",
                "WUICLASS2000",
                "WUICLASS2010",
                "WUICLASS2020",
                "wui_part_area_m2",
                "wui_part_x",
                "wui_part_y",
                "geometry",
            ]
        ],
        fire_buffers,
        how="inner",
        predicate="intersects",
    ).drop(columns=["index_right"])
    if risk.empty:
        raise RuntimeError("No direct-MTBS fire-WUI single-part risk-set rows were created.")

    risk["wui_vintage"] = _wui_vintage_for_fire_year(risk["fire_year"])
    risk["WUICLASS"] = pd.NA
    for vintage in [2000, 2010, 2020]:
        mask = risk["wui_vintage"].eq(vintage)
        risk.loc[mask, "WUICLASS"] = risk.loc[mask, f"WUICLASS{vintage}"]
    risk = risk.loc[risk["WUICLASS"].isin(wui_classes)].copy()
    if risk.empty:
        raise RuntimeError("No fire-WUI single-part rows remain after applying the fire-year WUI vintage.")
    risk = risk.drop_duplicates(["fire_id", "wui_part_id"]).reset_index(drop=True)

    buffer_by_fire_idx = fire_buffers.set_index("fire_idx")["geometry"]
    rows_outside_5km_buffer = 0
    for start in range(0, len(risk), chunk_size):
        stop = min(start + chunk_size, len(risk))
        part = risk.iloc[start:stop]
        in_buffer = shapely.intersects(
            part.geometry.to_numpy(),
            buffer_by_fire_idx.loc[part["fire_idx"].astype(int)].to_numpy(),
        )
        rows_outside_5km_buffer += int((~in_buffer).sum())
    if rows_outside_5km_buffer:
        raise RuntimeError(f"Risk-set validation failed: {rows_outside_5km_buffer:,} rows are outside the 5 km fire buffer.")

    risk["wui_is_intermix"] = risk["WUICLASS"].astype("string").str.contains("Intermix", na=False).astype(int)
    risk["wui_density_rank"] = (
        risk["WUICLASS"]
        .astype("string")
        .str.extract(r"^(Low|Med|High)_Dens", expand=False)
        .map({"Low": 1, "Med": 2, "High": 3})
        .fillna(0)
        .astype(float)
    )
    risk["wui_housing_units"] = np.select(
        [risk["wui_vintage"].eq(2000), risk["wui_vintage"].eq(2010), risk["wui_vintage"].eq(2020)],
        [
            pd.to_numeric(risk["HU2000"], errors="coerce"),
            pd.to_numeric(risk["HU2010"], errors="coerce"),
            pd.to_numeric(risk["HU2020"], errors="coerce"),
        ],
        default=np.nan,
    )
    risk["wui_housing_density"] = np.select(
        [risk["wui_vintage"].eq(2000), risk["wui_vintage"].eq(2010), risk["wui_vintage"].eq(2020)],
        [
            pd.to_numeric(risk["HUDEN2000"], errors="coerce"),
            pd.to_numeric(risk["HUDEN2010"], errors="coerce"),
            pd.to_numeric(risk["HUDEN2020"], errors="coerce"),
        ],
        default=np.nan,
    )
    risk["wui_housing_units"] = pd.to_numeric(risk["wui_housing_units"], errors="coerce").fillna(0).clip(lower=0)
    risk["wui_housing_density"] = pd.to_numeric(risk["wui_housing_density"], errors="coerce").fillna(0).clip(lower=0)
    risk["log_wui_part_area_m2"] = np.log1p(risk["wui_part_area_m2"].clip(lower=0))
    risk["log_wui_housing_units"] = np.log1p(risk["wui_housing_units"])
    risk["log_wui_housing_density"] = np.log1p(risk["wui_housing_density"])

    dx_wui_to_ignition = risk["origin_x"].to_numpy() - risk["wui_part_x"].to_numpy()
    dy_wui_to_ignition = risk["origin_y"].to_numpy() - risk["wui_part_y"].to_numpy()
    bearing_wui_to_ignition = (np.degrees(np.arctan2(dy_wui_to_ignition, dx_wui_to_ignition)) + 360.0) % 360.0
    bearing_ignition_to_wui = (bearing_wui_to_ignition + 180.0) % 360.0
    distance_m = np.sqrt(dx_wui_to_ignition**2 + dy_wui_to_ignition**2)
    risk["bearing_wui_to_ignition_degrees"] = bearing_wui_to_ignition
    risk["bearing_ignition_to_wui_degrees"] = bearing_ignition_to_wui
    risk["direction"] = (np.floor(((bearing_ignition_to_wui + 22.5) % 360.0) / 45.0).astype(int) + 1).astype(int)
    risk["distance_to_ignition_km"] = distance_m / 1000.0
    risk["distance_to_ignition_km_sq"] = risk["distance_to_ignition_km"] ** 2
    risk["log_distance_to_ignition_km"] = np.log1p(risk["distance_to_ignition_km"])
    risk["distance_bin_2km"] = np.floor(risk["distance_to_ignition_km"].clip(lower=0) / 2.0).astype("int16")
    risk["fire_direction_FE"] = risk["fire_id"].astype(str) + "-" + risk["direction"].astype(str)
    risk["fire_direction_distance_2km_FE"] = (
        risk["fire_id"].astype(str)
        + "-"
        + risk["direction"].astype(str)
        + "-"
        + risk["distance_bin_2km"].astype(str)
    )

    fire_geoms = fires.set_index("fire_idx")["geometry"]
    burned = np.zeros(len(risk), dtype=bool)
    burn_share = np.zeros(len(risk), dtype=float)
    severity_sample_x = np.full(len(risk), np.nan, dtype=float)
    severity_sample_y = np.full(len(risk), np.nan, dtype=float)
    wui_area = risk["wui_part_area_m2"].to_numpy()
    for start in range(0, len(risk), chunk_size):
        stop = min(start + chunk_size, len(risk))
        part = risk.iloc[start:stop]
        wui_geoms = part.geometry.to_numpy()
        perims = fire_geoms.loc[part["fire_idx"].astype(int)].to_numpy()
        part_burned = shapely.intersects(wui_geoms, perims)
        burned[start:stop] = part_burned
        if part_burned.any():
            local = np.where(part_burned)[0]
            intersections = shapely.intersection(wui_geoms[local], perims[local])
            area = shapely.area(intersections)
            denom = wui_area[start:stop][local]
            burn_share[start + local] = np.divide(area, denom, out=np.zeros_like(area, dtype=float), where=denom > 0)
            positive_area = area > 0
            if positive_area.any():
                points = shapely.point_on_surface(intersections[positive_area])
                severity_sample_x[start + local[positive_area]] = shapely.get_x(points)
                severity_sample_y[start + local[positive_area]] = shapely.get_y(points)

    panel_columns = [
        "wui_id",
        "wui_part_number",
        "wui_part_id",
        "wui_parent_geom_type",
        "wui_part_geom_type",
        "BLK20",
        "STATE",
        "wui_vintage",
        "WUICLASS",
        "wui_part_area_m2",
        "wui_part_x",
        "wui_part_y",
        "wui_is_intermix",
        "wui_density_rank",
        "wui_housing_units",
        "wui_housing_density",
        "log_wui_part_area_m2",
        "log_wui_housing_units",
        "log_wui_housing_density",
        "fire_id",
        "fire_idx",
        "fire_name",
        "FIRE_TYPE",
        "fire_year",
        "fire_month",
        "fire_season",
        "mtbs_acres",
        "mtbs_size_class",
        "Event_ID",
        "ignition_date",
        "fire_state",
        "origin_x",
        "origin_y",
        "bearing_wui_to_ignition_degrees",
        "bearing_ignition_to_wui_degrees",
        "direction",
        "distance_to_ignition_km",
        "distance_to_ignition_km_sq",
        "log_distance_to_ignition_km",
        "distance_bin_2km",
        "fire_direction_FE",
        "fire_direction_distance_2km_FE",
    ]
    panel = pd.DataFrame(risk.drop(columns="geometry"))[panel_columns].reset_index(drop=True)
    panel["panel_row_id"] = np.arange(len(panel), dtype="int64")
    panel["mtbs_wui_any_burn"] = burned.astype(float)
    panel["mtbs_wui_burn_share"] = np.clip(burn_share, 0, 1)
    panel["mtbs_wui_burn_acres"] = panel["mtbs_wui_burn_share"].clip(lower=0) * panel["wui_part_area_m2"].clip(lower=0) / acre_m2
    panel["log_mtbs_wui_burn_acres"] = np.log1p(panel["mtbs_wui_burn_acres"])
    panel["mtbs_burn_severity_sample_value"] = np.nan
    panel["mtbs_wui_moderate_high_severity"] = np.nan
    panel["mtbs_wui_high_severity"] = np.nan
    valid_severity_years = []
    invalid_severity_years = []
    try:
        import rasterio
        from pyproj import Transformer

        verification_path = p.paths["tables"] / "mtbs_conus_burn_severity_mosaics_2000_2010_verification.csv"
        invalid_verified_years = set()
        if verification_path.exists():
            verification = pd.read_csv(verification_path)
            if {"year", "valid"}.issubset(verification.columns):
                invalid_verified_years = set(
                    pd.to_numeric(verification.loc[~verification["valid"].astype(bool), "year"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
        for year in sorted(panel["fire_year"].dropna().astype(int).unique()):
            tif_path = p.paths["mtbs_bsmosaics"] / str(year) / f"mtbs_CONUS_{year}.tif"
            year_mask = panel["fire_year"].eq(year)
            if year in invalid_verified_years or not tif_path.exists():
                invalid_severity_years.append(year)
                continue
            panel.loc[year_mask, "mtbs_wui_moderate_high_severity"] = 0.0
            panel.loc[year_mask, "mtbs_wui_high_severity"] = 0.0
            sample_mask = year_mask & np.isfinite(severity_sample_x) & np.isfinite(severity_sample_y)
            if not sample_mask.any():
                valid_severity_years.append(year)
                continue
            with rasterio.open(tif_path) as src:
                transformer = Transformer.from_crs(albers, src.crs, always_xy=True)
                xs, ys = transformer.transform(severity_sample_x[sample_mask.to_numpy()], severity_sample_y[sample_mask.to_numpy()])
                values = np.array([sample[0] for sample in src.sample(list(zip(xs, ys)))], dtype=float)
            idx = panel.index[sample_mask]
            valid_values = np.isin(values, [1, 2, 3, 4, 5])
            panel.loc[idx[valid_values], "mtbs_burn_severity_sample_value"] = values[valid_values]
            panel.loc[idx[valid_values], "mtbs_wui_moderate_high_severity"] = np.isin(values[valid_values], [3, 4]).astype(float)
            panel.loc[idx[valid_values], "mtbs_wui_high_severity"] = (values[valid_values] == 4).astype(float)
            if (~valid_values).any():
                panel.loc[idx[~valid_values], ["mtbs_wui_moderate_high_severity", "mtbs_wui_high_severity"]] = np.nan
            valid_severity_years.append(year)
    except Exception as exc:
        print(f"[warning] MTBS burn-severity sampling failed: {exc}")
        valid_severity_years = []
        invalid_severity_years = sorted(panel["fire_year"].dropna().astype(int).unique().tolist())

    treatments = _load_completed_facts_treatment_polygons_for_links(p.paths["treatment_polygons"], states)
    if treatments.empty:
        raise RuntimeError("No completed FACTS fuel-reduction treatment polygons remain after cleaning.")
    treatment_cols = [
        "treat_id",
        "facts_fid",
        "SUID",
        "FACTS_ID",
        "UK",
        "ACTIVITY_C",
        "ACTIVITY",
        "STATE_ABBR",
        "DATE_COMPL",
        "treatment_completed_date",
        "FY_AWARDED",
        "FISCAL_Y_1",
        "treatment_completed_year",
        "treatment_completed_year_source",
        "CAT_NM",
        "METHOD",
        "TREATMENT1",
        "GIS_ACRES",
        "treatment_acres",
        "treat_is_prescribed_fire",
        "treat_is_mechanical",
        "treat_is_both_polygon",
        "treatment_type",
        "geometry",
    ]
    sector_base_cols = [
        "panel_row_id",
        "fire_id",
        "fire_name",
        "fire_year",
        "fire_month",
        "fire_season",
        "fire_state",
        "ignition_date",
        "wui_id",
        "wui_part_number",
        "wui_part_id",
        "direction",
        "distance_bin_2km",
    ]
    link_base_cols = [
        *sector_base_cols,
        "sector_label",
        "sector_orientation",
        "sector_radius_m",
        "sector_radius_label",
        "sector_half_angle_degrees",
        "sector_bearing_degrees",
        *[col for col in treatment_cols if col != "geometry"],
    ]

    def _build_sector_links(radius_m: float, radius_label: str, orientation: str) -> pd.DataFrame:
        link_frames = []
        is_opposite = orientation == "opposite"
        sector_label = f"{orientation}_{radius_label}"
        for start in range(0, len(panel), chunk_size):
            stop = min(start + chunk_size, len(panel))
            part = panel.iloc[start:stop].copy()
            bearing = part["bearing_wui_to_ignition_degrees"].to_numpy(dtype=float)
            if is_opposite:
                bearing = (bearing + 180.0) % 360.0
            sector_geometry = [
                _sector_polygon(row.wui_part_x, row.wui_part_y, sector_bearing, radius_m, shield_half_angle_degrees)
                for row, sector_bearing in zip(part.itertuples(index=False), bearing)
            ]
            sector_frame = part[sector_base_cols].copy()
            sector_frame["sector_label"] = sector_label
            sector_frame["sector_orientation"] = orientation
            sector_frame["sector_radius_m"] = radius_m
            sector_frame["sector_radius_label"] = radius_label
            sector_frame["sector_half_angle_degrees"] = shield_half_angle_degrees
            sector_frame["sector_bearing_degrees"] = bearing
            sectors = gpd.GeoDataFrame(sector_frame, geometry=sector_geometry, crs=albers)
            sectors = sectors.loc[~sectors.geometry.is_empty].copy()
            if sectors.empty:
                continue
            joined = gpd.sjoin(
                sectors,
                treatments[treatment_cols],
                how="inner",
                predicate="intersects",
            ).drop(columns=["index_right"], errors="ignore")
            if joined.empty:
                continue
            joined = joined.drop_duplicates(["sector_label", "panel_row_id", "treat_id"])
            link_frames.append(pd.DataFrame(joined.drop(columns="geometry")))
        if link_frames:
            return pd.concat(link_frames, ignore_index=True)
        return pd.DataFrame(columns=link_base_cols)

    fireface_link_frames = [
        _build_sector_links(radius_m, radius_label, "fireface")
        for radius_m, radius_label in shield_radius_specs
    ]
    links = pd.concat(
        [
            *fireface_link_frames,
            _build_sector_links(shield_radius_m, "2p5km", "opposite"),
        ],
        ignore_index=True,
    )

    if not links.empty:
        links["treatment_completed_date"] = pd.to_datetime(links["treatment_completed_date"], errors="coerce")
        links["ignition_date"] = pd.to_datetime(links["ignition_date"], errors="coerce")
        links["treatment_completed_year"] = pd.to_numeric(links["treatment_completed_year"], errors="coerce")
        links["fire_year"] = pd.to_numeric(links["fire_year"], errors="coerce")
        links["completed_before_ignition"] = False
        has_exact_date = links["treatment_completed_date"].notna() & links["ignition_date"].notna()
        links.loc[has_exact_date, "completed_before_ignition"] = (
            links.loc[has_exact_date, "treatment_completed_date"] < links.loc[has_exact_date, "ignition_date"]
        )
        year_only = ~has_exact_date & links["treatment_completed_year"].notna() & links["fire_year"].notna()
        links.loc[year_only, "completed_before_ignition"] = links.loc[year_only, "treatment_completed_year"] < links.loc[year_only, "fire_year"]
        links["days_before_ignition"] = np.nan
        links.loc[has_exact_date, "days_before_ignition"] = (
            links.loc[has_exact_date, "ignition_date"] - links.loc[has_exact_date, "treatment_completed_date"]
        ).dt.days
        links["years_before_fire"] = links["fire_year"] - links["treatment_completed_year"]
        links["completed_before_fire_year"] = links["years_before_fire"].ge(1).astype(int)
        links["prior_5yr"] = links["years_before_fire"].between(1, 5).astype(int)
        links["prior_10yr"] = links["years_before_fire"].between(1, 10).astype(int)
        links["current_or_post_fire_year"] = links["years_before_fire"].le(0).astype(int)
    else:
        links = pd.DataFrame(
            columns=[
                "panel_row_id",
                "fire_id",
                "fire_name",
                "fire_year",
                "fire_month",
                "fire_season",
                "fire_state",
                "ignition_date",
                "wui_id",
                "wui_part_number",
                "wui_part_id",
                "direction",
                "distance_bin_2km",
                "sector_label",
                "sector_orientation",
                "sector_radius_m",
                "sector_radius_label",
                "sector_half_angle_degrees",
                "sector_bearing_degrees",
                *[col for col in treatment_cols if col != "geometry"],
                "completed_before_ignition",
                "days_before_ignition",
                "years_before_fire",
                "completed_before_fire_year",
                "prior_5yr",
                "prior_10yr",
                "current_or_post_fire_year",
            ]
        )

    links_fireface_by_radius = {
        radius_label: links.loc[links["sector_label"].eq(f"fireface_{radius_label}")].copy()
        if not links.empty
        else links.copy()
        for _, radius_label in shield_radius_specs
    }
    links_fireface_2p5 = links_fireface_by_radius["2p5km"]
    links_fireface_5km = links_fireface_by_radius["5km"]
    links_opposite_2p5 = links.loc[links["sector_label"].eq("opposite_2p5km")].copy() if not links.empty else links.copy()

    def _add_treatment_aggregates(
        panel_df: pd.DataFrame,
        link_df: pd.DataFrame,
        flag_col: str,
        suffix: str,
        base_prefix: str = "wui_fireface",
    ) -> pd.DataFrame:
        n = len(panel_df)
        count = np.zeros(n, dtype="int32")
        acres = np.zeros(n, dtype=float)
        prescribed_acres = np.zeros(n, dtype=float)
        mechanical_acres = np.zeros(n, dtype=float)
        prescribed_any = np.zeros(n, dtype="int8")
        mechanical_any = np.zeros(n, dtype="int8")
        other_any = np.zeros(n, dtype="int8")
        most_recent_year = np.full(n, np.nan, dtype=float)
        most_recent_date = pd.Series(pd.NaT, index=panel_df.index, dtype="datetime64[ns]")
        if not link_df.empty:
            timed = link_df.loc[link_df[flag_col].astype(float).gt(0)].copy()
            if not timed.empty:
                timed["prescribed_fire_treatment_acres"] = timed["treatment_acres"].astype(float) * timed["treat_is_prescribed_fire"].astype(float)
                timed["mechanical_treatment_acres"] = timed["treatment_acres"].astype(float) * timed["treat_is_mechanical"].astype(float)
                timed["treat_is_other"] = ~(timed["treat_is_prescribed_fire"].astype(bool) | timed["treat_is_mechanical"].astype(bool))
                grouped = timed.groupby("panel_row_id", observed=True).agg(
                    treat_count=("treat_id", "nunique"),
                    treated_acres=("treatment_acres", "sum"),
                    prescribed_fire_acres=("prescribed_fire_treatment_acres", "sum"),
                    mechanical_acres=("mechanical_treatment_acres", "sum"),
                    prescribed_fire_any=("treat_is_prescribed_fire", "max"),
                    mechanical_any=("treat_is_mechanical", "max"),
                    other_any=("treat_is_other", "max"),
                    most_recent_treatment_year=("treatment_completed_year", "max"),
                    most_recent_treatment_date=("treatment_completed_date", "max"),
                )
                row_id = grouped.index.to_numpy(dtype="int64")
                count[row_id] = grouped["treat_count"].to_numpy(dtype="int32")
                acres[row_id] = grouped["treated_acres"].to_numpy(dtype=float)
                prescribed_acres[row_id] = grouped["prescribed_fire_acres"].to_numpy(dtype=float)
                mechanical_acres[row_id] = grouped["mechanical_acres"].to_numpy(dtype=float)
                prescribed_any[row_id] = grouped["prescribed_fire_any"].astype(bool).to_numpy(dtype="int8")
                mechanical_any[row_id] = grouped["mechanical_any"].astype(bool).to_numpy(dtype="int8")
                other_any[row_id] = grouped["other_any"].astype(bool).to_numpy(dtype="int8")
                most_recent_year[row_id] = grouped["most_recent_treatment_year"].to_numpy(dtype=float)
                most_recent_date.iloc[row_id] = pd.to_datetime(grouped["most_recent_treatment_date"], errors="coerce").to_numpy()
        any_treat = count > 0
        both_type = prescribed_any.astype(bool) & mechanical_any.astype(bool)
        mechanical_only = mechanical_any.astype(bool) & ~prescribed_any.astype(bool)
        prescribed_only = prescribed_any.astype(bool) & ~mechanical_any.astype(bool)
        exposure_type = np.select(
            [both_type, prescribed_only, mechanical_only, any_treat],
            ["both", "prescribed_fire", "mechanical", "other"],
            default="none",
        )
        panel_df[f"{base_prefix}_treat_count_{suffix}"] = count
        panel_df[f"{base_prefix}_treated_acres_{suffix}"] = acres
        panel_df[f"{base_prefix}_prescribed_fire_acres_{suffix}"] = prescribed_acres
        panel_df[f"{base_prefix}_mechanical_acres_{suffix}"] = mechanical_acres
        panel_df[f"{base_prefix}_any_treat_{suffix}"] = any_treat.astype(float)
        panel_df[f"log1p_{base_prefix}_treated_acres_{suffix}"] = np.log1p(acres)
        panel_df[f"{base_prefix}_treated_acres_100_{suffix}"] = acres / 100.0
        panel_df[f"{base_prefix}_prescribed_fire_any_{suffix}"] = prescribed_any.astype(float)
        panel_df[f"{base_prefix}_mechanical_any_{suffix}"] = mechanical_any.astype(float)
        panel_df[f"{base_prefix}_other_any_{suffix}"] = other_any.astype(float)
        panel_df[f"{base_prefix}_prescribed_fire_only_{suffix}"] = prescribed_only.astype(float)
        panel_df[f"{base_prefix}_mechanical_only_{suffix}"] = mechanical_only.astype(float)
        panel_df[f"{base_prefix}_both_type_{suffix}"] = both_type.astype(float)
        panel_df[f"{base_prefix}_exposure_type_{suffix}"] = exposure_type
        panel_df[f"{base_prefix}_most_recent_treatment_year_{suffix}"] = most_recent_year
        panel_df[f"{base_prefix}_most_recent_treatment_date_{suffix}"] = most_recent_date.to_numpy()
        return panel_df

    panel = _add_treatment_aggregates(panel, links_fireface_2p5, "completed_before_fire_year", "any_prior_2p5km")
    for _, radius_label in shield_radius_specs:
        panel = _add_treatment_aggregates(
            panel,
            links_fireface_by_radius[radius_label],
            "prior_5yr",
            f"pre5_{radius_label}",
        )
    panel = _add_treatment_aggregates(panel, links_fireface_2p5, "prior_10yr", "pre10_2p5km")
    panel = _add_treatment_aggregates(panel, links_opposite_2p5, "prior_5yr", "pre5_2p5km", base_prefix="wui_opposite")

    current_or_post_count = np.zeros(len(panel), dtype="int32")
    if not links_fireface_2p5.empty:
        current_post = links_fireface_2p5.loc[links_fireface_2p5["current_or_post_fire_year"].astype(float).gt(0)]
        if not current_post.empty:
            current_post_grouped = current_post.groupby("panel_row_id", observed=True)["treat_id"].nunique()
            current_or_post_count[current_post_grouped.index.to_numpy(dtype="int64")] = current_post_grouped.to_numpy(dtype="int32")
    panel["wui_fireface_current_or_post_treat_count_2p5km"] = current_or_post_count

    expected_pre5_count = np.zeros(len(panel), dtype="int32")
    expected_pre5_acres = np.zeros(len(panel), dtype=float)
    if not links_fireface_2p5.empty:
        pre5 = links_fireface_2p5.loc[links_fireface_2p5["prior_5yr"].astype(float).gt(0)]
        if not pre5.empty:
            expected = pre5.groupby("panel_row_id", observed=True).agg(
                count=("treat_id", "nunique"),
                acres=("treatment_acres", "sum"),
            )
            expected_pre5_count[expected.index.to_numpy(dtype="int64")] = expected["count"].to_numpy(dtype="int32")
            expected_pre5_acres[expected.index.to_numpy(dtype="int64")] = expected["acres"].to_numpy(dtype=float)
    aggregate_count_mismatches = int((expected_pre5_count != panel["wui_fireface_treat_count_pre5_2p5km"].to_numpy()).sum())
    aggregate_acres_mismatches = int((np.abs(expected_pre5_acres - panel["wui_fireface_treated_acres_pre5_2p5km"].to_numpy()) > 1e-8).sum())

    if not links.empty:
        by_year_frames = []
        for sector_label, sector_links in links.groupby("sector_label", observed=True):
            for flag_col, timing_window in [
                ("completed_before_fire_year", "any_prior"),
                ("prior_5yr", "prior_5yr"),
                ("prior_10yr", "prior_10yr"),
            ]:
                timed = sector_links.loc[sector_links[flag_col].astype(float).gt(0)].copy()
                if timed.empty:
                    continue
                by_year_frames.append(
                    timed.groupby(["fire_year", "treatment_type"], as_index=False, observed=True)
                    .agg(
                        treatment_link_rows=("treat_id", "size"),
                        treated_fire_wui_part_rows=("panel_row_id", "nunique"),
                        treatment_polygons=("treat_id", "nunique"),
                        treated_acres=("treatment_acres", "sum"),
                        fires=("fire_id", "nunique"),
                        wui_single_parts=("wui_part_id", "nunique"),
                    )
                    .assign(timing_window=timing_window, sector_label=sector_label)
                )
        by_year_type = pd.concat(by_year_frames, ignore_index=True) if by_year_frames else pd.DataFrame()
    else:
        by_year_type = pd.DataFrame()
    if by_year_type.empty:
        by_year_type = pd.DataFrame(
            columns=[
                "fire_year",
                "treatment_type",
                "treatment_link_rows",
                "treated_fire_wui_part_rows",
                "treatment_polygons",
                "treated_acres",
                "fires",
                "wui_single_parts",
                "timing_window",
                "sector_label",
            ]
        )

    vintage_mismatch = ~(
        (panel["fire_year"].between(2000, 2009) & panel["wui_vintage"].eq(2000))
        | (panel["fire_year"].between(2010, 2019) & panel["wui_vintage"].eq(2010))
        | (panel["fire_year"].between(2020, 2024) & panel["wui_vintage"].eq(2020))
    )
    non_polygon_parts = int((panel["wui_part_geom_type"].astype(str) != "Polygon").sum())
    burn_share_invalid = int((panel["mtbs_wui_burn_share"].lt(-1e-12) | panel["mtbs_wui_burn_share"].gt(1 + 1e-12)).sum())
    type_logic_failures = int(
        (
            panel["wui_fireface_both_type_pre5_2p5km"].astype(bool)
            & (
                ~panel["wui_fireface_prescribed_fire_any_pre5_2p5km"].astype(bool)
                | ~panel["wui_fireface_mechanical_any_pre5_2p5km"].astype(bool)
            )
        ).sum()
    )
    greenwood_fire_id = "MN4755309164820210815"
    greenwood = panel.loc[panel["fire_id"].eq(greenwood_fire_id)]
    summary = pd.DataFrame(
        [
            {"metric": "source_fire_data", "value": "Direct MTBS wildfire perimeters and ignition points", "unit": "text"},
            {"metric": "source_mtbs_perimeters", "value": str(p.paths["mtbs_perimeters"]), "unit": "path"},
            {"metric": "source_mtbs_ignition_points", "value": str(p.paths["mtbs_fod_points"]), "unit": "path"},
            {"metric": "source_wui_data", "value": str(wui_path), "unit": "path"},
            {"metric": "source_treatment_data", "value": str(p.paths["treatment_polygons"]), "unit": "path"},
            {"metric": "direct_mtbs_wildfire_perimeters", "value": int(fires["fire_id"].nunique()), "unit": "fires"},
            {"metric": "wui_parent_polygons_loaded", "value": wui_parent_count, "unit": "parent polygons"},
            {"metric": "wui_single_polygon_parts_loaded", "value": wui_part_count, "unit": "single polygon parts"},
            {"metric": "fire_wui_singlepart_risk_rows", "value": int(len(panel)), "unit": "fire-WUI single-part rows"},
            {"metric": "fires_in_panel", "value": int(panel["fire_id"].nunique()), "unit": "fires"},
            {"metric": "wui_single_parts_in_panel", "value": int(panel["wui_part_id"].nunique()), "unit": "single WUI polygon parts"},
            {"metric": "risk_set_buffer_radius", "value": risk_buffer_m / 1000.0, "unit": "km from MTBS fire perimeter"},
            {"metric": "fire_facing_sector_radius", "value": shield_radius_m / 1000.0, "unit": "km from WUI representative point"},
            {"metric": "robustness_fire_facing_sector_radius", "value": robustness_shield_radius_m / 1000.0, "unit": "km from WUI representative point"},
            {"metric": "fire_facing_sector_radii", "value": ",".join(str(radius_m / 1000.0) for radius_m, _ in shield_radius_specs), "unit": "km from WUI representative point"},
            {"metric": "fire_facing_sector_half_angle", "value": shield_half_angle_degrees, "unit": "degrees"},
            {"metric": "wui_vintage_rule", "value": "2000 WUI for 2000-2009 fires; 2010 WUI for 2010-2019 fires; 2020 WUI for 2020-2024 fires", "unit": "text"},
            {"metric": "wui_vintage_rule_mismatches", "value": int(vintage_mismatch.sum()), "unit": "rows"},
            {"metric": "non_polygon_wui_part_rows", "value": non_polygon_parts, "unit": "rows"},
            {"metric": "rows_outside_5km_mtbs_fire_buffer", "value": rows_outside_5km_buffer, "unit": "rows"},
            {"metric": "burn_share_invalid_rows", "value": burn_share_invalid, "unit": "rows"},
            {"metric": "treatment_link_rows", "value": int(len(links)), "unit": "sector-treatment links across primary and robustness sectors"},
            *[
                {
                    "metric": f"fireface_{radius_label}_treatment_link_rows",
                    "value": int(len(links_fireface_by_radius[radius_label])),
                    "unit": "sector-treatment links",
                }
                for _, radius_label in shield_radius_specs
            ],
            {"metric": "primary_2p5km_treatment_link_rows", "value": int(len(links_fireface_2p5)), "unit": "sector-treatment links"},
            {"metric": "robustness_5km_treatment_link_rows", "value": int(len(links_fireface_5km)), "unit": "sector-treatment links"},
            {"metric": "opposite_2p5km_treatment_link_rows", "value": int(len(links_opposite_2p5)), "unit": "sector-treatment links"},
            {"metric": "treatment_polygons_linked", "value": int(links["treat_id"].nunique()) if not links.empty else 0, "unit": "FACTS polygons"},
            *[
                {
                    "metric": f"treated_rows_pre5_{radius_label}",
                    "value": int(panel[f"wui_fireface_any_treat_pre5_{radius_label}"].sum()),
                    "unit": "fire-WUI single-part rows",
                }
                for _, radius_label in shield_radius_specs
            ],
            {"metric": "treated_rows_pre5", "value": int(panel["wui_fireface_any_treat_pre5_2p5km"].sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "treated_rows_pre5_5km", "value": int(panel["wui_fireface_any_treat_pre5_5km"].sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "opposite_rows_pre5_2p5km", "value": int(panel["wui_opposite_any_treat_pre5_2p5km"].sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "treated_rows_pre10", "value": int(panel["wui_fireface_any_treat_pre10_2p5km"].sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "treated_rows_any_prior", "value": int(panel["wui_fireface_any_treat_any_prior_2p5km"].sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "mechanical_rows_pre5", "value": int(panel["wui_fireface_mechanical_any_pre5_2p5km"].sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "prescribed_fire_rows_pre5", "value": int(panel["wui_fireface_prescribed_fire_any_pre5_2p5km"].sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "both_type_rows_pre5", "value": int(panel["wui_fireface_both_type_pre5_2p5km"].sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "current_or_post_fire_year_linked_rows", "value": int((panel["wui_fireface_current_or_post_treat_count_2p5km"] > 0).sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "mtbs_burn_severity_valid_years", "value": ",".join(str(year) for year in valid_severity_years), "unit": "years"},
            {"metric": "mtbs_burn_severity_invalid_or_missing_years", "value": ",".join(str(year) for year in invalid_severity_years), "unit": "years"},
            {"metric": "mtbs_burn_severity_observed_rows", "value": int(panel["mtbs_burn_severity_sample_value"].notna().sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "mtbs_burn_severity_regression_rows", "value": int(panel["mtbs_wui_moderate_high_severity"].notna().sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "mtbs_moderate_high_severity_rows", "value": int(panel["mtbs_wui_moderate_high_severity"].fillna(0).sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "mtbs_high_severity_rows", "value": int(panel["mtbs_wui_high_severity"].fillna(0).sum()), "unit": "fire-WUI single-part rows"},
            {"metric": "aggregate_count_mismatches_pre5", "value": aggregate_count_mismatches, "unit": "rows"},
            {"metric": "aggregate_acres_mismatches_pre5", "value": aggregate_acres_mismatches, "unit": "rows"},
            {"metric": "treatment_type_logic_failures_pre5", "value": type_logic_failures, "unit": "rows"},
            {"metric": "greenwood_fire_rows", "value": int(len(greenwood)), "unit": "fire-WUI single-part rows"},
            {"metric": "greenwood_fire_treated_rows_pre5", "value": int(greenwood["wui_fireface_any_treat_pre5_2p5km"].sum()) if not greenwood.empty else 0, "unit": "fire-WUI single-part rows"},
            {"metric": "greenwood_fire_burned_rows", "value": int(greenwood["mtbs_wui_any_burn"].sum()) if not greenwood.empty else 0, "unit": "fire-WUI single-part rows"},
            {"metric": "panel_output", "value": str(out_panel), "unit": "path"},
            *[
                {
                    "metric": f"panel_output_{radius_label}",
                    "value": str(radius_panel_paths[radius_label]),
                    "unit": "path",
                }
                for radius_label in radius_panel_paths
            ],
            {"metric": "treatment_links_output", "value": str(out_links), "unit": "path"},
        ]
    )

    functions.ensure_dir(out_panel.parent)
    panel.to_parquet(out_panel, index=False)
    base_radius_cols = [
        col
        for col in panel.columns
        if not col.startswith("wui_fireface_") and not col.startswith("log1p_wui_fireface_")
    ]
    for radius_m, radius_label in shield_radius_specs:
        if radius_label not in radius_panel_paths:
            continue
        suffix = f"_{radius_label}"
        radius_cols = [
            col
            for col in panel.columns
            if (
                (col.startswith("wui_fireface_") or col.startswith("log1p_wui_fireface_"))
                and col.endswith(suffix)
            )
        ]
        radius_panel = panel[list(dict.fromkeys([*base_radius_cols, *radius_cols]))].copy()
        radius_panel["shield_radius_km"] = radius_m / 1000.0
        radius_panel["shield_radius_label"] = radius_label
        for col in radius_cols:
            radius_panel[col.removesuffix(suffix)] = radius_panel[col]
        radius_panel.to_parquet(radius_panel_paths[radius_label], index=False)
    links.to_parquet(out_links, index=False)
    functions.write_csv(summary.to_dict("records"), summary_path, fieldnames=summary.columns)
    functions.write_csv(by_year_type.to_dict("records"), by_year_type_path, fieldnames=by_year_type.columns)
