"""
Build the Forest x Year panel for the analysis.

Steps:
1. Load cleaned FRT records and aggregate to (FORESTORGC, year) sums.
2. Load NF boundaries and WFBZ disasters; spatial-join disasters -> forests.
3. Aggregate disasters to (FORESTORGC, year) indicator.
4. Build the balanced panel: 112 forests x 25 years.
5. Save panel.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
DATA = ROOT / "data"
PROC = DATA / "processed"
PROC.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2000, 2025))  # 2000-2024

# ----- 1. FRT aggregation -----
print("=" * 60)
print("Step 1: Aggregate FRT to Forest-Year")
print("=" * 60)
frt = pd.read_csv(PROC / "frt_clean.csv", low_memory=False)
print(f"FRT records: {len(frt):,}")
print(f"FS_UNIT_ID dtype: {frt['FS_UNIT_ID'].dtype}")

# Standardize forest ID format: zero-pad to 4 chars
frt["FORESTORGC"] = frt["FS_UNIT_ID"].astype(str).str.zfill(4)
frt["FY_AWARDED"] = frt["FY_AWARDED"].astype(int)

# Filter to non-Wildfire treatments (Mechanical, Fire, Other are intentional treatments)
frt = frt[frt["CAT_NM"] != "Wildfire Non-Treatment"].copy()
print(f"After excluding Wildfire-Non-Treatment: {len(frt):,}")

# Annual aggregation by forest-year
annual = frt.groupby(["FORESTORGC", "FY_AWARDED"]).agg(
    n_treatments=("SUID", "count"),
    acres_treated=("GIS_ACRES", "sum"),
    acres_mechanical=("GIS_ACRES", lambda x: frt.loc[x.index].query("CAT_NM == 'Mechanical'")["GIS_ACRES"].sum()),
    acres_fire=("GIS_ACRES", lambda x: frt.loc[x.index].query("CAT_NM == 'Fire'")["GIS_ACRES"].sum()),
    acres_wui=("GIS_ACRES", lambda x: frt.loc[x.index].query("ISWUI == 'Y'")["GIS_ACRES"].sum()),
    avg_cost_per_acre=("COST_PER_U", "mean"),
).reset_index().rename(columns={"FY_AWARDED": "year"})

print(f"\nAnnual aggregation rows: {len(annual):,}")
print(f"Forests with any treatment: {annual['FORESTORGC'].nunique()}")
print(f"\nTop 10 forest-years by acres:")
print(annual.nlargest(10, 'acres_treated')[['FORESTORGC', 'year', 'acres_treated']])

# ----- 2. NF boundaries -----
print("\n" + "=" * 60)
print("Step 2: Load National Forest boundaries")
print("=" * 60)
nf = gpd.read_file(DATA / "S_USA.BdyAdm_LSRS_AdministrativeForest.shp")
nf["FORESTORGC"] = nf["FORESTORGC"].astype(str).str.zfill(4)
print(f"NF polygons: {len(nf)}")
print(f"NF CRS: {nf.crs}")

# Some forests have multiple polygons in the file; dissolve to one row per FORESTORGC
nf_dissolved = nf.dissolve(by="FORESTORGC", aggfunc={"FORESTNAME": "first", "REGION": "first", "GIS_ACRES": "sum"}).reset_index()
print(f"After dissolve: {len(nf_dissolved)} unique forests")

# Compare FRT forests to NF boundary forests
frt_forests = set(annual["FORESTORGC"].unique())
nf_forests = set(nf_dissolved["FORESTORGC"].unique())
print(f"\nFRT forests:    {len(frt_forests)}")
print(f"NF forests:     {len(nf_forests)}")
print(f"Intersection:   {len(frt_forests & nf_forests)}")
print(f"FRT only:       {sorted(frt_forests - nf_forests)[:10]}")
print(f"NF only:        {sorted(nf_forests - frt_forests)[:10]}")

# ----- 3. Disasters -----
print("\n" + "=" * 60)
print("Step 3: Load and process WFBZ disasters")
print("=" * 60)
wfbz = gpd.read_file(ROOT / "wildfire" / "wfbz_disasters_2000-2025.geojson")
print(f"Disasters total: {len(wfbz)}")
wfbz = wfbz[(wfbz["wildfire_year"] >= 2000) & (wfbz["wildfire_year"] <= 2024)].copy()
wfbz["year"] = wfbz["wildfire_year"].astype(int)
print(f"Disasters 2000-2024: {len(wfbz)}")

# Clean geometries: keep only valid polygons / multipolygons
wfbz = wfbz[wfbz.geometry.notnull()].copy()
# Keep polygon-like; for points/collections, buffer or skip
wfbz["geom_type"] = wfbz.geometry.geom_type
print(f"Geometry types:\n{wfbz['geom_type'].value_counts()}")

# Restrict to polygon/multipolygon for spatial join (drops 137 points + 508 GeometryCollections)
poly_mask = wfbz["geom_type"].isin(["Polygon", "MultiPolygon"])
wfbz_poly = wfbz[poly_mask].copy()
print(f"Polygon disasters used for spatial join: {len(wfbz_poly)}")

# Also keep the geometry collections after extracting polygons
gc_mask = wfbz["geom_type"] == "GeometryCollection"
if gc_mask.sum() > 0:
    print(f"Extracting polygon parts from {gc_mask.sum()} GeometryCollections...")
    from shapely.geometry import MultiPolygon, Polygon
    def extract_poly(g):
        if g is None: return None
        polys = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        if not polys: return None
        if len(polys) == 1: return polys[0]
        # Build a MultiPolygon from polygons; if multipolygons present, flatten
        flat = []
        for p in polys:
            if p.geom_type == "Polygon":
                flat.append(p)
            else:
                flat.extend(list(p.geoms))
        return MultiPolygon(flat) if flat else None
    gc_subset = wfbz[gc_mask].copy()
    gc_subset["geometry"] = gc_subset["geometry"].apply(extract_poly)
    gc_subset = gc_subset[gc_subset.geometry.notnull()]
    wfbz_poly = pd.concat([wfbz_poly, gc_subset], ignore_index=True)
    wfbz_poly = gpd.GeoDataFrame(wfbz_poly, geometry="geometry", crs=wfbz.crs)
    print(f"After GeometryCollection extraction: {len(wfbz_poly)}")

# ----- 4. Spatial join: disasters -> forests -----
print("\n" + "=" * 60)
print("Step 4: Spatial join disasters to forests")
print("=" * 60)
# Ensure CRS match
nf_dissolved = nf_dissolved.to_crs(wfbz_poly.crs)

# Use intersects predicate
joined = gpd.sjoin(wfbz_poly[["wildfire_id", "year", "wildfire_struct_destroyed",
                              "wildfire_civil_fatalities", "wildfire_max_civil_fatalities",
                              "wildfire_fema_dec", "wildfire_disaster_criteria_met",
                              "wildfire_cost", "wildfire_area", "geometry"]],
                  nf_dissolved[["FORESTORGC", "FORESTNAME", "geometry"]],
                  how="inner", predicate="intersects")
print(f"Disaster-forest intersections: {len(joined)}")
print(f"Forests with at least one disaster: {joined['FORESTORGC'].nunique()}")

# Aggregate to forest-year disaster panel
disaster_year = joined.groupby(["FORESTORGC", "year"]).agg(
    n_disasters=("wildfire_id", "nunique"),
    n_struct_destroyed=("wildfire_struct_destroyed", "sum"),
    n_fatalities=("wildfire_max_civil_fatalities", "sum"),
    any_fema_dec=("wildfire_fema_dec", "max"),
    total_disaster_cost=("wildfire_cost", "sum"),
    total_burn_area=("wildfire_area", "sum"),
).reset_index()
print(f"\nDisaster panel rows: {len(disaster_year):,}")
print(f"Top 10 forest-years by struct destroyed:")
print(disaster_year.nlargest(10, "n_struct_destroyed")[["FORESTORGC", "year", "n_disasters", "n_struct_destroyed"]])

# ----- 5. Build balanced panel -----
print("\n" + "=" * 60)
print("Step 5: Build balanced Forest-Year panel")
print("=" * 60)

# Use the union of forests appearing in either FRT or NF boundary file
all_forests = sorted(nf_forests | frt_forests)
print(f"Total unique forests in panel: {len(all_forests)}")

# Build skeleton
panel = pd.MultiIndex.from_product([all_forests, YEARS], names=["FORESTORGC", "year"]).to_frame(index=False)

# Merge in FRT annual flows
panel = panel.merge(annual, on=["FORESTORGC", "year"], how="left")
for col in ["n_treatments", "acres_treated", "acres_mechanical", "acres_fire", "acres_wui"]:
    panel[col] = panel[col].fillna(0)
panel["avg_cost_per_acre"] = panel["avg_cost_per_acre"].fillna(np.nan)

# Compute cumulative acres treated within forest, by year (the stock)
panel = panel.sort_values(["FORESTORGC", "year"])
panel["cum_acres"] = panel.groupby("FORESTORGC")["acres_treated"].cumsum()
panel["cum_acres_mech"] = panel.groupby("FORESTORGC")["acres_mechanical"].cumsum()
panel["cum_acres_fire"] = panel.groupby("FORESTORGC")["acres_fire"].cumsum()
panel["cum_acres_wui"] = panel.groupby("FORESTORGC")["acres_wui"].cumsum()

# Treatment indicator: any cumulative treatment so far
panel["FRT_ever"] = (panel["cum_acres"] > 0).astype(int)

# Year of first treatment (for event study and CS)
ever = panel[panel["FRT_ever"] == 1].groupby("FORESTORGC")["year"].min().rename("first_treat_year")
panel = panel.merge(ever, on="FORESTORGC", how="left")
panel["event_time"] = panel["year"] - panel["first_treat_year"]

# Merge in disaster outcomes
panel = panel.merge(disaster_year, on=["FORESTORGC", "year"], how="left")
for col in ["n_disasters", "n_struct_destroyed", "n_fatalities", "total_disaster_cost", "total_burn_area"]:
    panel[col] = panel[col].fillna(0)
panel["any_fema_dec"] = panel["any_fema_dec"].fillna(False)
panel["disaster"] = (panel["n_disasters"] > 0).astype(int)

# Merge in NF metadata (forest size, name, region)
panel = panel.merge(nf_dissolved[["FORESTORGC", "FORESTNAME", "REGION", "GIS_ACRES"]].rename(
    columns={"GIS_ACRES": "forest_acres"}), on="FORESTORGC", how="left")

print(f"\nFinal panel shape: {panel.shape}")
print(f"Forests x Years: {panel['FORESTORGC'].nunique()} x {panel['year'].nunique()}")
print(f"Disaster mean: {panel['disaster'].mean():.4f}")
print(f"FRT_ever mean: {panel['FRT_ever'].mean():.4f}")
print(f"Forests ever treated: {panel.groupby('FORESTORGC')['FRT_ever'].max().sum()}")
print(f"Forests with any disaster: {panel.groupby('FORESTORGC')['disaster'].max().sum()}")

# Save
out = PROC / "panel_forest_year.csv"
panel.to_csv(out, index=False)
print(f"\nWrote: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
print("\nFirst few rows:")
print(panel.head())
