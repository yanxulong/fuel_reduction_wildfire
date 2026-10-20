"""
Build the WUI polygon-year panel:
  1. Read FRT centroids and WUI centroids
  2. Use KDTree to find all WUI polygons within 10 km of each treatment centroid
  3. Aggregate treatments per (wui_id, year): cumulative acres, FRT_it indicator
  4. Read WFBZ disasters; spatial-join to WUI polygons (point-in-polygon via centroid is too lossy
     for disasters which are large; instead, find disasters whose burn perimeter intersects each
     WUI centroid's surrounding area, but a simpler proxy is centroid-in-perimeter buffered by ~5km)
  5. Output panel: (wui_id, year, FRT, cum_acres, disaster, n_disasters)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
OUT_PANEL = ROOT / "data" / "processed" / "panel_wui_year.parquet"
OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load WUI centroids and FRT centroids
# -----------------------------------------------------------------------------
print("Loading WUI centroids (national)...")
wui = pd.read_parquet(ROOT / "data" / "processed" / "wui_centroids_us.parquet")
print(f"  WUI: {len(wui):,}")

print("Loading FRT centroids (national, Wear-cleaned: human fuel reduction only)...")
frt = pd.read_parquet(ROOT / "data" / "processed" / "frt_centroids_us_clean.parquet")
print(f"  FRT: {len(frt):,}")

# -----------------------------------------------------------------------------
# 2. Project to meters for accurate KDTree distance calculations
#    Use NAD83 Conus Albers (EPSG:5070) - meters
# -----------------------------------------------------------------------------
print("\nPreparing coordinates...")
wui = wui.dropna(subset=['lon','lat']).copy()
frt = frt.dropna(subset=['lon','lat']).copy()
print(f"  WUI after NaN drop: {len(wui):,}")
print(f"  FRT after NaN drop: {len(frt):,}")

# Script 18 already wrote WUI centroid lon/lat in EPSG:5070 meters.
print("  WUI centroids already in EPSG:5070 meters.")
wui['x_m'] = wui['lon'].values
wui['y_m'] = wui['lat'].values

# FRT centroids are in EPSG:4269 (degrees); reproject to EPSG:5070
print("  Reprojecting FRT EPSG:4269 -> EPSG:5070...")
frt_pts = gpd.GeoSeries(gpd.points_from_xy(frt.lon, frt.lat), crs=4269).to_crs(5070)
frt['x_m'] = frt_pts.x.values
frt['y_m'] = frt_pts.y.values

# Sanity
wui = wui[np.isfinite(wui['x_m']) & np.isfinite(wui['y_m'])].reset_index(drop=True)
frt = frt[np.isfinite(frt['x_m']) & np.isfinite(frt['y_m'])].reset_index(drop=True)
print(f"  WUI x: [{wui.x_m.min():.0f}, {wui.x_m.max():.0f}]  y: [{wui.y_m.min():.0f}, {wui.y_m.max():.0f}]")
print(f"  FRT x: [{frt.x_m.min():.0f}, {frt.x_m.max():.0f}]  y: [{frt.y_m.min():.0f}, {frt.y_m.max():.0f}]")

# -----------------------------------------------------------------------------
# 3. KDTree query: for each FRT, find all WUI within 10 km
# -----------------------------------------------------------------------------
print("\nBuilding KDTree on WUI centroids...")
tree = cKDTree(wui[['x_m','y_m']].values)
BUFFER_M = 10_000  # 10 km

print(f"Querying KDTree for each FRT (buffer = {BUFFER_M/1000:.0f} km)...")
# Returns list of arrays of WUI indices within distance for each FRT
matches = tree.query_ball_point(frt[['x_m','y_m']].values, r=BUFFER_M)

# Build long format: (frt_idx, wui_idx)
print("Expanding matches to long format...")
frt_idx_list = []
wui_idx_list = []
for i, m in enumerate(matches):
    if m:
        frt_idx_list.extend([i] * len(m))
        wui_idx_list.extend(m)

print(f"  {len(frt_idx_list):,} (FRT, WUI) match pairs")

# -----------------------------------------------------------------------------
# 4. Build aggregated FRT panel: per (wui_id, year)
# -----------------------------------------------------------------------------
print("\nBuilding (wui_id, year) treatment aggregates...")
matches_df = pd.DataFrame({
    'wui_idx': wui_idx_list,
    'frt_idx': frt_idx_list,
})
matches_df['wui_id'] = wui.iloc[matches_df['wui_idx']]['wui_id'].values
matches_df['year']   = frt.iloc[matches_df['frt_idx']]['FY_AWARDED'].values
matches_df['acres']  = frt.iloc[matches_df['frt_idx']]['GIS_ACRES'].values
matches_df['cat_nm'] = frt.iloc[matches_df['frt_idx']]['CAT_NM'].values
matches_df['cost_per_u'] = frt.iloc[matches_df['frt_idx']]['COST_PER_U'].values

# Aggregate
agg = matches_df.groupby(['wui_id','year']).agg(
    n_treatments=('frt_idx','count'),
    acres_treated=('acres','sum'),
    avg_cost_per_acre=('cost_per_u','mean'),
).reset_index()
agg['acres_mech'] = matches_df[matches_df['cat_nm']=='Mechanical'].groupby(['wui_id','year'])['acres'].sum().reindex(
    pd.MultiIndex.from_frame(agg[['wui_id','year']])).fillna(0).values
agg['acres_fire'] = matches_df[matches_df['cat_nm']=='Fire'].groupby(['wui_id','year'])['acres'].sum().reindex(
    pd.MultiIndex.from_frame(agg[['wui_id','year']])).fillna(0).values
print(f"Aggregated: {len(agg):,} treated (wui_id, year) cells")

# -----------------------------------------------------------------------------
# 5. Load disasters and find their wui_id intersections
# -----------------------------------------------------------------------------
print("\nLoading WFBZ disasters...")
disasters = gpd.read_file(str(ROOT / "wildfire" / "wfbz_disasters_2000-2025.geojson"))
print(f"  loaded {len(disasters):,} disaster events")
print(f"  CRS: {disasters.crs}")

# National analysis: keep all disasters (no Western bbox filter).
print(f"  using all {len(disasters):,} national disaster events")

# Project to meters
print("  projecting disasters to EPSG:5070...")
disasters = disasters.to_crs(5070)

# For each disaster, find WUI centroids that fall inside the disaster perimeter.
# Use spatial index on disasters.
print("  spatial-joining disasters to WUI centroids...")
wui_gdf = gpd.GeoDataFrame(
    wui[['wui_id','x_m','y_m','STATE','WUICLASS']].copy(),
    geometry=gpd.points_from_xy(wui['x_m'], wui['y_m']),
    crs=5070,
)
joined = gpd.sjoin(wui_gdf, disasters[['wildfire_id','wildfire_year','wildfire_max_civil_fatalities',
                                         'wildfire_struct_destroyed','wildfire_fema_dec','geometry']],
                    predicate='within', how='inner')
print(f"  joined rows: {len(joined):,}")

# Build disaster aggregates: per (wui_id, year)
joined['year'] = pd.to_numeric(joined['wildfire_year'], errors='coerce').astype('Int64')
joined = joined.dropna(subset=['year'])
joined['n_struct_destroyed'] = pd.to_numeric(joined['wildfire_struct_destroyed'], errors='coerce').fillna(0)
joined['n_fatalities'] = pd.to_numeric(joined['wildfire_max_civil_fatalities'], errors='coerce').fillna(0)
joined['fema_dec'] = joined['wildfire_fema_dec'].fillna(False).astype(bool)

dis_agg = joined.groupby(['wui_id','year']).agg(
    n_disasters=('wildfire_id','nunique'),
    n_struct_destroyed=('n_struct_destroyed','sum'),
    n_fatalities=('n_fatalities','sum'),
    any_fema_dec=('fema_dec','max'),
).reset_index()
print(f"Disaster aggregates: {len(dis_agg):,} (wui_id, year) cells")

# -----------------------------------------------------------------------------
# 6. Build balanced panel: every (wui_id) x year in 2000..2024
# -----------------------------------------------------------------------------
print("\nBuilding balanced panel...")
years = list(range(2000, 2025))
# Restrict to wui_ids that ever appear in either FRT or disaster join, or sample 50% of others
# To keep size tractable, restrict to WUI polygons within 10 km of *any* treatment OR any disaster
treated_wui = set(agg['wui_id'].unique())
disaster_wui = set(dis_agg['wui_id'].unique())
relevant = treated_wui | disaster_wui
# Add a random sample of untreated/non-disaster WUI as controls
all_wui = set(wui['wui_id'])
untreated_pool = list(all_wui - relevant)
np.random.seed(42)
control_n = min(len(untreated_pool), max(50_000, len(relevant)))
controls = set(np.random.choice(untreated_pool, size=control_n, replace=False))
panel_wui_ids = list(relevant | controls)
print(f"  panel WUI polygons: {len(panel_wui_ids):,}")
print(f"    treated: {len(treated_wui):,}")
print(f"    disaster (no treatment): {len(disaster_wui - treated_wui):,}")
print(f"    pure controls: {len(controls):,}")

# Build cartesian
panel = pd.MultiIndex.from_product(
    [panel_wui_ids, years], names=['wui_id','year']
).to_frame(index=False)

# Merge treatments
panel = panel.merge(agg, on=['wui_id','year'], how='left')
for c in ['n_treatments','acres_treated','acres_mech','acres_fire']:
    panel[c] = panel[c].fillna(0)

# Sort and compute cumulative
panel = panel.sort_values(['wui_id','year'])
panel['cum_acres'] = panel.groupby('wui_id')['acres_treated'].cumsum()
panel['cum_acres_mech'] = panel.groupby('wui_id')['acres_mech'].cumsum()
panel['cum_acres_fire'] = panel.groupby('wui_id')['acres_fire'].cumsum()

# Merge disasters
panel = panel.merge(dis_agg, on=['wui_id','year'], how='left')
for c in ['n_disasters','n_struct_destroyed','n_fatalities']:
    panel[c] = panel[c].fillna(0)
panel['any_fema_dec'] = panel['any_fema_dec'].fillna(False).astype(bool)
panel['disaster'] = (panel['n_disasters'] > 0).astype(int)

# Merge WUI metadata
wui_meta = wui[['wui_id','STATE','WUICLASS','area_km2','is_intermix','density']].copy()
panel = panel.merge(wui_meta, on='wui_id', how='left')

# Treatment indicators
panel['FRT_it'] = (panel['cum_acres'] > 0).astype(int)
# First treat year per polygon
first_treat = panel[panel['acres_treated'] > 0].groupby('wui_id')['year'].min()
panel['first_treat_year'] = panel['wui_id'].map(first_treat)
panel['event_time'] = panel['year'] - panel['first_treat_year']

print(f"\nPanel: {panel.shape}")
print(f"Disaster rate: {panel['disaster'].mean():.4f}")
print(f"FRT_it rate: {panel['FRT_it'].mean():.4f}")
print(f"Treated polygon-years: {panel['FRT_it'].sum():,}")
print(f"Disaster polygon-years: {panel['disaster'].sum():,}")

# Save
panel.to_parquet(OUT_PANEL)
print(f"\nWrote: {OUT_PANEL}  ({OUT_PANEL.stat().st_size / 1e6:.1f} MB)")
