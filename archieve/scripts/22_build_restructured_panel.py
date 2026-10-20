"""
Restructured panel construction for AER quality.

Changes vs. script 13:
  1. Tighter buffer: 5 km (was 10 km), closer to documented stand-level effect range
  2. Decay-weighted treatment stock: T_it = sum_{s<=t} acres_s * exp(-(t-s)/5)
     Half-life ~3.5 years; treatments older than ~10 years contribute < 14%
  3. Baseline covariates for matching: state, density class, intermix dummy,
     polygon area, polygon-mean tmean and ppt over the 2000-2002 pre-period
  4. NO random untreated controls (those are added by the matching script later)
  5. Two treatment definitions kept side-by-side:
       - FRT_it          binary "any treatment within 5 km up to year t"
       - log_T_stock_it  decay-weighted log stock
  6. Pre-period FACTS aggregates (2000-2002) computed per polygon and per state,
     for use as the Bartik instrument's initial-share weights

Output: data/processed/panel_wui_year_v2.parquet
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
OUT = ROOT / "data" / "processed" / "panel_wui_year_v2.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

BUFFER_M = 5_000        # 5 km, half the previous buffer
DECAY_YEARS = 5         # exponential decay parameter (half-life ~3.5 yr)
PRE_YEARS = (2000, 2001, 2002)
YEARS = list(range(2000, 2025))

t0 = time.time()
print("=" * 72)
print("RESTRUCTURED PANEL CONSTRUCTION")
print(f"  buffer = {BUFFER_M/1000:.0f} km")
print(f"  decay  = exp(-(t-s)/{DECAY_YEARS})")
print(f"  pre-period for baselines = {PRE_YEARS}")
print("=" * 72)

# -----------------------------------------------------------------------------
# 1. Load WUI and FRT
# -----------------------------------------------------------------------------
print("\n[1] Loading WUI centroids ...")
wui = pd.read_parquet(ROOT / "data" / "processed" / "wui_centroids_us.parquet")
wui = wui.dropna(subset=["lon", "lat"]).reset_index(drop=True)
print(f"    WUI: {len(wui):,}")

print("\n[2] Loading FRT centroids (Wear-cleaned) ...")
frt = pd.read_parquet(ROOT / "data" / "processed" / "frt_centroids_us_clean.parquet")
frt = frt.dropna(subset=["lon", "lat"]).reset_index(drop=True)
print(f"    FRT: {len(frt):,}")

# WUI already in EPSG:5070 m, FRT in EPSG:4269 deg
wui["x_m"] = wui["lon"].values
wui["y_m"] = wui["lat"].values
print("    Reprojecting FRT 4269 -> 5070 ...")
frt_pts = gpd.GeoSeries(gpd.points_from_xy(frt.lon, frt.lat), crs=4269).to_crs(5070)
frt["x_m"] = frt_pts.x.values
frt["y_m"] = frt_pts.y.values
frt = frt[np.isfinite(frt["x_m"]) & np.isfinite(frt["y_m"])].reset_index(drop=True)

# -----------------------------------------------------------------------------
# 2. KDTree spatial join with 5 km buffer
# -----------------------------------------------------------------------------
print(f"\n[3] KDTree query, buffer = {BUFFER_M/1000:.0f} km ...")
tree = cKDTree(wui[["x_m", "y_m"]].values)
matches = tree.query_ball_point(frt[["x_m", "y_m"]].values, r=BUFFER_M)

frt_idx_list, wui_idx_list = [], []
for i, m in enumerate(matches):
    if m:
        frt_idx_list.extend([i] * len(m))
        wui_idx_list.extend(m)
print(f"    {len(frt_idx_list):,} (FRT, WUI) match pairs")

m_df = pd.DataFrame({"wui_idx": wui_idx_list, "frt_idx": frt_idx_list})
m_df["wui_id"] = wui.iloc[m_df["wui_idx"]]["wui_id"].values
m_df["year"] = frt.iloc[m_df["frt_idx"]]["FY_AWARDED"].values
m_df["acres"] = frt.iloc[m_df["frt_idx"]]["GIS_ACRES"].values
m_df["cat_nm"] = frt.iloc[m_df["frt_idx"]]["CAT_NM"].values
m_df["cost_per_u"] = frt.iloc[m_df["frt_idx"]]["COST_PER_U"].values
m_df = m_df.dropna(subset=["year"])
m_df["year"] = m_df["year"].astype(int)

# -----------------------------------------------------------------------------
# 3. Aggregate annual treatment per (wui_id, year)
# -----------------------------------------------------------------------------
print("\n[4] Aggregating treatments per (wui_id, year) ...")
agg = m_df.groupby(["wui_id", "year"]).agg(
    n_treatments=("frt_idx", "count"),
    acres_treated=("acres", "sum"),
    avg_cost_per_acre=("cost_per_u", "mean"),
).reset_index()
agg["acres_mech"] = (
    m_df[m_df["cat_nm"] == "Mechanical"]
    .groupby(["wui_id", "year"])["acres"]
    .sum()
    .reindex(pd.MultiIndex.from_frame(agg[["wui_id", "year"]]))
    .fillna(0)
    .values
)
agg["acres_fire"] = (
    m_df[m_df["cat_nm"] == "Fire"]
    .groupby(["wui_id", "year"])["acres"]
    .sum()
    .reindex(pd.MultiIndex.from_frame(agg[["wui_id", "year"]]))
    .fillna(0)
    .values
)
print(f"    {len(agg):,} treated polygon-years")

# -----------------------------------------------------------------------------
# 4. Disasters via spatial intersection
# -----------------------------------------------------------------------------
print("\n[5] Loading WFBZ disasters ...")
dis = gpd.read_file(str(ROOT / "wildfire" / "wfbz_disasters_2000-2025.geojson"))
print(f"    {len(dis):,} disasters")
dis = dis.to_crs(5070)

print("    spatial join: WUI centroids in disaster perimeters ...")
wui_gdf = gpd.GeoDataFrame(
    wui[["wui_id", "x_m", "y_m", "STATE", "WUICLASS"]],
    geometry=gpd.points_from_xy(wui["x_m"], wui["y_m"]),
    crs=5070,
)
joined = gpd.sjoin(
    wui_gdf,
    dis[["wildfire_id", "wildfire_year", "wildfire_max_civil_fatalities",
         "wildfire_struct_destroyed", "wildfire_fema_dec", "geometry"]],
    predicate="within", how="inner",
)
joined["year"] = pd.to_numeric(joined["wildfire_year"], errors="coerce").astype("Int64")
joined = joined.dropna(subset=["year"])
joined["n_struct_destroyed"] = pd.to_numeric(joined["wildfire_struct_destroyed"], errors="coerce").fillna(0)
joined["n_fatalities"] = pd.to_numeric(joined["wildfire_max_civil_fatalities"], errors="coerce").fillna(0)
joined["fema_dec"] = joined["wildfire_fema_dec"].fillna(False).astype(bool)

dis_agg = joined.groupby(["wui_id", "year"]).agg(
    n_disasters=("wildfire_id", "nunique"),
    n_struct_destroyed=("n_struct_destroyed", "sum"),
    n_fatalities=("n_fatalities", "sum"),
    any_fema_dec=("fema_dec", "max"),
).reset_index()
print(f"    {len(dis_agg):,} disaster polygon-years")

# -----------------------------------------------------------------------------
# 5. Build a relevance set (NO random untreated)
#    Keep every polygon that is either treated or in a disaster, plus everyone
#    within 25 km of one of those polygons (the "vicinity" pool from which
#    matching draws controls)
# -----------------------------------------------------------------------------
print("\n[6] Building relevance set ...")
treated_ids = set(agg["wui_id"].unique())
disaster_ids = set(dis_agg["wui_id"].unique())
core = treated_ids | disaster_ids
print(f"    treated: {len(treated_ids):,}")
print(f"    in disaster: {len(disaster_ids):,}")
print(f"    union: {len(core):,}")

# Vicinity expansion: every WUI polygon within 25 km of a core polygon is
# eligible to be a matched control. This avoids dropping the entire pool of
# untreated polygons, while keeping the panel manageable.
core_idx = wui[wui["wui_id"].isin(core)].index.values
core_xy = wui.loc[core_idx, ["x_m", "y_m"]].values
print("    expanding vicinity by 25 km ...")
all_xy = wui[["x_m", "y_m"]].values
tree_all = cKDTree(all_xy)
nearby_idx_lists = tree_all.query_ball_point(core_xy, r=25_000)
nearby = set()
for lst in nearby_idx_lists:
    nearby.update(lst)
vicinity_ids = set(wui.iloc[list(nearby)]["wui_id"].values)
panel_ids = sorted(core | vicinity_ids)
print(f"    vicinity total: {len(panel_ids):,}")

# -----------------------------------------------------------------------------
# 6. Build the balanced panel (wui_id × year)
# -----------------------------------------------------------------------------
print("\n[7] Building balanced panel ...")
panel = pd.MultiIndex.from_product(
    [panel_ids, YEARS], names=["wui_id", "year"]
).to_frame(index=False)

panel = panel.merge(agg, on=["wui_id", "year"], how="left")
for c in ["n_treatments", "acres_treated", "acres_mech", "acres_fire"]:
    panel[c] = panel[c].fillna(0)

# Sort then build cumulative + decay stock
panel = panel.sort_values(["wui_id", "year"]).reset_index(drop=True)

print("    cumulative acres ...")
panel["cum_acres"] = panel.groupby("wui_id")["acres_treated"].cumsum()
panel["cum_acres_mech"] = panel.groupby("wui_id")["acres_mech"].cumsum()
panel["cum_acres_fire"] = panel.groupby("wui_id")["acres_fire"].cumsum()

# Decay-weighted stock: T_t = sum_{s<=t} a_s * exp(-(t-s)/tau)
#                            = exp(-t/tau) * cumsum(a_s * exp(s/tau))
# Fully vectorized within each polygon group, no Python loops.
print(f"    decay-weighted stock (decay = {DECAY_YEARS} yr) ...")
year0 = panel["year"].min()
yidx = (panel["year"].values - year0).astype(np.float64)
exp_pos = np.exp(yidx / DECAY_YEARS)         # exp(s/tau)
exp_neg = np.exp(-yidx / DECAY_YEARS)        # exp(-t/tau)
panel["_b"] = panel["acres_treated"].values * exp_pos
panel["_cumb"] = panel.groupby("wui_id")["_b"].cumsum()
panel["T_stock"] = panel["_cumb"].values * exp_neg
panel = panel.drop(columns=["_b", "_cumb"])
panel["log_T_stock"] = np.log1p(panel["T_stock"])
print(f"    T_stock summary: mean={panel['T_stock'].mean():.1f} "
      f"max={panel['T_stock'].max():.0f} "
      f"frac>0={(panel['T_stock']>0).mean():.3f}")

# Disasters
panel = panel.merge(dis_agg, on=["wui_id", "year"], how="left")
for c in ["n_disasters", "n_struct_destroyed", "n_fatalities"]:
    panel[c] = panel[c].fillna(0)
panel["any_fema_dec"] = panel["any_fema_dec"].fillna(False).astype(bool)
panel["disaster"] = (panel["n_disasters"] > 0).astype(int)

# WUI metadata
wui_meta = wui[["wui_id", "STATE", "WUICLASS", "area_km2", "is_intermix", "density"]]
panel = panel.merge(wui_meta, on="wui_id", how="left")

# Treatment indicators
panel["FRT_it"] = (panel["cum_acres"] > 0).astype(int)
first_treat = panel[panel["acres_treated"] > 0].groupby("wui_id")["year"].min()
panel["first_treat_year"] = panel["wui_id"].map(first_treat)
panel["event_time"] = panel["year"] - panel["first_treat_year"]

print(f"    panel: {panel.shape}")
print(f"    polygons: {panel['wui_id'].nunique():,}")
print(f"    treated polygon-years: {panel['FRT_it'].sum():,}")
print(f"    disaster polygon-years: {panel['disaster'].sum():,}")
print(f"    mean disaster rate: {panel['disaster'].mean():.5f}")

# -----------------------------------------------------------------------------
# 7. Merge climate
# -----------------------------------------------------------------------------
print("\n[8] Merging PRISM climate ...")
clim = pd.read_parquet(ROOT / "data" / "processed" / "wui_climate_panel.parquet")
panel = panel.merge(clim, on=["wui_id", "year"], how="left")
# Polygon-mean fill for any missing
panel["tmean_c"] = panel.groupby("wui_id")["tmean_c"].transform(lambda s: s.fillna(s.mean()))
panel["ppt_mm"] = panel.groupby("wui_id")["ppt_mm"].transform(lambda s: s.fillna(s.mean()))
print(f"    tmean NaN: {panel['tmean_c'].isna().sum():,}")
print(f"    ppt   NaN: {panel['ppt_mm'].isna().sum():,}")

# -----------------------------------------------------------------------------
# 8. Pre-period baselines per polygon (for matching)
# -----------------------------------------------------------------------------
print("\n[9] Computing 2000-2002 pre-period baselines per polygon ...")
pre = panel[panel["year"].isin(PRE_YEARS)].groupby("wui_id").agg(
    pre_tmean=("tmean_c", "mean"),
    pre_ppt=("ppt_mm", "mean"),
    pre_acres=("acres_treated", "sum"),
    pre_disaster=("disaster", "max"),
).reset_index()
panel = panel.merge(pre, on="wui_id", how="left")
print(f"    baselines for {len(pre):,} polygons")

# -----------------------------------------------------------------------------
# 9. Pre-period state shares for Bartik instrument
# -----------------------------------------------------------------------------
print("\n[10] Computing state-level pre-period FACTS shares ...")
state_pre = panel[panel["year"].isin(PRE_YEARS)].groupby("STATE").agg(
    state_pre_acres=("acres_treated", "sum"),
).reset_index()
total_pre = state_pre["state_pre_acres"].sum()
state_pre["state_pre_share"] = state_pre["state_pre_acres"] / total_pre
panel = panel.merge(state_pre[["STATE", "state_pre_share"]], on="STATE", how="left")
print("    top 10 state shares:")
print(state_pre.sort_values("state_pre_share", ascending=False).head(10).to_string(index=False))

# -----------------------------------------------------------------------------
# 10. National FACTS budget time series
# -----------------------------------------------------------------------------
print("\n[11] Computing national FACTS budget by year ...")
nat = panel.groupby("year").agg(
    nat_acres=("acres_treated", "sum"),
).reset_index()
panel = panel.merge(nat, on="year", how="left")
print(nat.to_string(index=False))

# Bartik shift-share variable: state_pre_share * nat_acres
panel["bartik_acres"] = panel["state_pre_share"] * panel["nat_acres"]
panel["log_bartik"] = np.log1p(panel["bartik_acres"])

# Save
panel.to_parquet(OUT)
elapsed = time.time() - t0
print(f"\nWrote: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
print(f"Elapsed: {elapsed/60:.1f} min")
