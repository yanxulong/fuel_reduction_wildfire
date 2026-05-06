"""
Extract FACTS treatment centroid coordinates from the local DBF + SHP files.
We use the SHP file directly via geopandas to get geometry, but only read
records that meet our filter criteria. Output: parquet of (SUID, FY_AWARDED, lon, lat, acres, cat_nm, state).

Full-US version: keeps records from ALL 50 states (no Western bounding-box filter).
"""
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
SHP = ROOT / "data" / "Actv_HazFuelTrt_PL.dbf"  # DBF triggers shp/shx alongside via fiona

OUT = ROOT / "data" / "processed" / "frt_centroids_us.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Check shp file exists
shp_path = ROOT / "data" / "Actv_HazFuelTrt_PL.shp"
print(f"Shp exists: {shp_path.exists()}")
if not shp_path.exists():
    # Need to copy from Box
    print("Need to copy shp/shx from Box folder...")
    import shutil
    src = Path("/Users/long/Library/CloudStorage/Box-Box/External Bil_Biocarbon for Fuel Reduction/UMN/Yanxu Long/data/raw_data/Actv_HazFuelTrt_PL")
    for ext in ['.shp', '.shx', '.prj', '.cpg']:
        s = src / f"Actv_HazFuelTrt_PL{ext}"
        d = ROOT / "data" / f"Actv_HazFuelTrt_PL{ext}"
        if s.exists() and not d.exists():
            print(f"  copying {s.name}...")
            shutil.copy(str(s), str(d))
    print("done copying.")

print(f"\nReading shapefile in chunks (corrupt last record requires sequential read)...")

# Use fiona to read sequentially with error handling
import fiona
from shapely.geometry import shape, mapping
from shapely.ops import transform
records = []
n_skipped = 0

with fiona.open(str(shp_path)) as src:
    print(f"Total features in source: {len(src)}")
    print(f"Schema: {src.schema['properties'].keys()}")
    print(f"CRS: {src.crs}")

    for i, feat in enumerate(src):
        if i % 50000 == 0:
            print(f"  read {i:,} / kept {len(records):,}")
        try:
            props = feat['properties']
            if props.get('STAGE_VALU') != 'Accomplished':
                continue
            fy = props.get('FY_AWARDED')
            if fy is None or fy < 2000 or fy > 2024:
                continue
            acres = props.get('GIS_ACRES')
            if acres is None or acres <= 0:
                continue
            state = props.get('STATE_ABBR')
            if state is None:
                continue
            geom = feat.get('geometry')
            if geom is None:
                continue
            shp_geom = shape(geom)
            if not shp_geom.is_valid or shp_geom.is_empty:
                continue
            cent = shp_geom.centroid
            records.append({
                'SUID': props.get('SUID'),
                'FACTS_ID': props.get('FACTS_ID'),
                'ACTIVITY': props.get('ACTIVITY'),
                'CAT_NM': props.get('CAT_NM'),
                'FY_AWARDED': int(fy),
                'GIS_ACRES': float(acres),
                'COST_PER_U': props.get('COST_PER_U'),
                'STATE_ABBR': state,
                'FS_UNIT_ID': props.get('FS_UNIT_ID'),
                'lon': cent.x,
                'lat': cent.y,
            })
        except Exception as e:
            n_skipped += 1
            if n_skipped < 5:
                print(f"  skipped feature {i}: {e}")

print(f"\nDone. Read {i+1:,} features, kept {len(records):,}, skipped {n_skipped:,}")
gdf = pd.DataFrame(records)
print(f"CRS: EPSG:4269 (NAD83 geographic)")

# Already filtered above
out_df = gdf[['SUID','FACTS_ID','ACTIVITY','CAT_NM','FY_AWARDED','GIS_ACRES','COST_PER_U','STATE_ABBR','FS_UNIT_ID','lon','lat']].copy()
out_df.to_parquet(OUT)
print(f"\nWrote: {OUT}")
print(f"Centroids range: lon [{out_df.lon.min():.2f}, {out_df.lon.max():.2f}]  lat [{out_df.lat.min():.2f}, {out_df.lat.max():.2f}]")
print(f"\nBy state:")
print(out_df.groupby('STATE_ABBR').size().sort_values(ascending=False))
