"""
Extract WUI Intermix and Interface polygons for Western states from the national gpkg.
Output: parquet of (wui_id, STATE, WUICLASS, lon, lat, area_km2) — centroids only.
We keep all Intermix + Interface (the WUI proper). Pure NoVeg/Veg are excluded.
"""
import sqlite3
from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
GPKG = "/Volumes/SSD/WUI_2020.gpkg"
OUT = ROOT / "data" / "processed" / "wui_centroids_west.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

WEST = ('CA','OR','WA','ID','MT','WY','NV','UT','AZ','CO','NM')
WUI_CLASSES = ('Low_Dens_Intermix','Med_Dens_Intermix','High_Dens_Intermix',
               'Low_Dens_Interface','Med_Dens_Interface','High_Dens_Interface')

# Filter to Intermix only (Intermix is the canonical Radeloff WUI; Interface is also valuable
# but Intermix is the primary class with structures embedded in vegetation)
# We will keep BOTH classes and tag them.

# Read in chunks per state via SQL where clause
all_dfs = []
for state in WEST:
    print(f"\nReading {state}...")
    where = f"STATE = '{state}' AND WUICLASS IN {WUI_CLASSES}"
    gdf = gpd.read_file(GPKG, layer="WUI_2020", where=where)
    print(f"  loaded {len(gdf):,} polygons; columns: {list(gdf.columns)}")
    if len(gdf) == 0:
        continue

    # Reproject to a meters CRS to compute area, then back to NAD83 for centroids
    gdf_m = gdf.to_crs(5070)
    gdf['area_km2'] = gdf_m.geometry.area / 1e6

    # Centroid in NAD83 geographic
    cent = gdf.geometry.centroid
    gdf['lon'] = cent.x
    gdf['lat'] = cent.y

    # Use index as the row id (geopandas drops 'fid' to index by default)
    gdf = gdf.reset_index().rename(columns={'index': 'row_idx'})
    gdf['wui_id'] = state + '_' + gdf['row_idx'].astype(str)
    out = gdf[['wui_id','STATE','WUICLASS','area_km2','lon','lat']].copy()
    all_dfs.append(out)

print(f"\nConcatenating {len(all_dfs)} state files...")
df = pd.concat(all_dfs, ignore_index=True)
print(f"Total Western WUI polygons: {len(df):,}")

# Mark intermix vs interface
df['is_intermix'] = df['WUICLASS'].str.contains('Intermix').astype(int)
df['density'] = df['WUICLASS'].str.split('_').str[0]  # Low/Med/High

print(df.groupby(['STATE','is_intermix']).size().unstack(fill_value=0))

df.to_parquet(OUT)
print(f"\nWrote: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
