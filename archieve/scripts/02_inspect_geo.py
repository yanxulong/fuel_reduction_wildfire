"""Inspect WFBZ disasters and NF boundaries to understand structure."""
import geopandas as gpd
from pathlib import Path

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")

# WFBZ disasters
print("=" * 60)
print("WFBZ DISASTERS")
print("=" * 60)
wfbz = gpd.read_file(ROOT / "wildfire" / "wfbz_disasters_2000-2025.geojson")
print(f"Shape: {wfbz.shape}")
print(f"CRS: {wfbz.crs}")
print(f"Columns: {list(wfbz.columns)}")
print(f"\nDtypes:\n{wfbz.dtypes}")
print(f"\nFirst record:")
for c in wfbz.columns:
    if c != "geometry":
        print(f"  {c}: {wfbz.iloc[0][c]}")
print(f"\nYear distribution:")
print(wfbz['wildfire_year'].value_counts().sort_index().head(30))
print(f"\nDisaster criteria distribution:")
if 'wildfire_disaster_criteria_met' in wfbz.columns:
    print(wfbz['wildfire_disaster_criteria_met'].value_counts(dropna=False))
print(f"\nGeometry types:")
print(wfbz.geom_type.value_counts())

# NF boundaries
print("\n" + "=" * 60)
print("NATIONAL FOREST BOUNDARIES")
print("=" * 60)
nf = gpd.read_file(ROOT / "data" / "S_USA.BdyAdm_LSRS_AdministrativeForest.shp")
print(f"Shape: {nf.shape}")
print(f"CRS: {nf.crs}")
print(f"Columns: {list(nf.columns)}")
print(f"\nDtypes:\n{nf.dtypes}")
print(f"\nFirst 5 records:")
for c in nf.columns:
    if c != "geometry":
        print(f"  {c}: {list(nf[c].head().values)}")
print(f"\nGeometry types:")
print(nf.geom_type.value_counts())
print(f"\nUnique forests: {len(nf)}")
