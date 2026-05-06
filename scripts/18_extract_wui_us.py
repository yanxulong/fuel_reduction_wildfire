"""
Extract WUI Intermix and Interface block centroids for ALL 50 US states + DC
from the per-state SILVIS lab v4 shapefile zip files on Google Drive.

Iterates one state zip at a time:
  1. Reads zip via fiona /vsizip/ virtual filesystem (no extraction needed).
  2. Filters to the six WUI density-class combinations (Intermix + Interface).
  3. Computes centroid lon/lat in EPSG:5070 meters and area in km2.
  4. Appends to a national dataframe and writes wui_centroids_us.parquet.

Output: data/processed/wui_centroids_us.parquet
"""
import os
import sys
import time
from pathlib import Path
import pandas as pd
import geopandas as gpd
import fiona
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
GD_DIR = Path(
    "/Users/long/Library/CloudStorage/GoogleDrive-yxlong@umn.edu/"
    "Shared drives/NatCapTEEMs/Files/base_data/submissions/WUI"
)
OUT = ROOT / "data" / "processed" / "wui_centroids_us.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

STATES = [
    "AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","IA","ID","IL","IN","KS",
    "KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ",
    "NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT",
    "WA","WI","WV","WY",
]

# DBF column-name truncation: WUICLASS2020 → WUICLASS_2 (10-char limit).
COL_WUICLASS = "WUICLASS_2"
COL_HUDEN = "HUDEN2020"
COL_POP = "POP2020"

KEEP_CLASSES = {
    "Low_Dens_Intermix",  "Med_Dens_Intermix",  "High_Dens_Intermix",
    "Low_Dens_Interface", "Med_Dens_Interface", "High_Dens_Interface",
}


def process_state(state: str) -> pd.DataFrame:
    zip_path = GD_DIR / f"{state}_wui_block_1990_2020_change_v4_shp.zip"
    if not zip_path.exists():
        print(f"  [{state}] zip missing: {zip_path}")
        return None
    # /vsizip/ vsi-prefixed path lets fiona read the .shp inside the zip directly.
    base = f"{state}_wui_block_1990_2020_change_v4"
    vsi = f"/vsizip/{zip_path}/{base}/{base}.shp"

    t0 = time.time()
    try:
        gdf = gpd.read_file(vsi)
    except Exception as e:
        print(f"  [{state}] read error: {e}")
        return None
    n_total = len(gdf)
    if n_total == 0:
        print(f"  [{state}] empty shapefile")
        return None

    # Filter to the six Intermix + Interface classes.
    if COL_WUICLASS not in gdf.columns:
        print(f"  [{state}] missing column {COL_WUICLASS}; cols: {list(gdf.columns)[:8]}...")
        return None
    gdf = gdf[gdf[COL_WUICLASS].isin(KEEP_CLASSES)].copy()
    n_kept = len(gdf)
    if n_kept == 0:
        print(f"  [{state}] no Intermix/Interface blocks ({n_total:,} total)")
        return None

    # Reproject to EPSG:5070 meters if needed (the source is already NAD83 Albers).
    if gdf.crs is None or gdf.crs.to_epsg() != 5070:
        gdf = gdf.to_crs(5070)

    gdf["area_km2"] = gdf.geometry.area / 1e6
    cent = gdf.geometry.centroid
    out = pd.DataFrame({
        "wui_id": [f"{state}_{i}" for i in range(n_kept)],
        "STATE": state,
        "WUICLASS": gdf[COL_WUICLASS].values,
        "area_km2": gdf["area_km2"].values,
        "lon": cent.x.values,   # x-meters in EPSG:5070
        "lat": cent.y.values,   # y-meters in EPSG:5070
        "HUDEN2020": gdf[COL_HUDEN].values if COL_HUDEN in gdf.columns else 0,
        "POP2020": gdf[COL_POP].values if COL_POP in gdf.columns else 0,
    })
    out["is_intermix"] = out["WUICLASS"].str.contains("Intermix").astype(int)
    out["density"] = out["WUICLASS"].str.split("_").str[0]  # Low/Med/High

    dt = time.time() - t0
    print(f"  [{state}] {n_total:>8,} blocks → {n_kept:>7,} WUI    "
          f"({dt:5.1f}s)")
    return out


def main():
    print(f"Extracting WUI Intermix + Interface blocks from {len(STATES)} states...")
    print(f"Source dir: {GD_DIR}")
    print()
    parts = []
    t_start = time.time()
    for i, st in enumerate(STATES, 1):
        print(f"[{i:>2}/{len(STATES)}] {st} ...")
        try:
            df = process_state(st)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  [{st}] FAILED: {e}")
            df = None
        if df is not None and len(df) > 0:
            parts.append(df)
        # Periodic checkpoint
        if i % 5 == 0 and parts:
            cum = pd.concat(parts, ignore_index=True)
            cum.to_parquet(OUT)
            print(f"  -- checkpoint: {len(cum):,} blocks → {OUT.name} "
                  f"({OUT.stat().st_size/1e6:.1f} MB, "
                  f"elapsed {(time.time()-t_start)/60:.1f} min)")

    if not parts:
        print("ERROR: nothing extracted")
        sys.exit(1)

    df = pd.concat(parts, ignore_index=True)
    print(f"\nTotal national WUI Intermix+Interface blocks: {len(df):,}")
    print("By state:")
    print(df.groupby("STATE").size().sort_values(ascending=False).to_string())
    print()
    print("By density:")
    print(df.groupby("density").size().to_string())
    print()
    print("By is_intermix:")
    print(df.groupby("is_intermix").size().to_string())

    df.to_parquet(OUT)
    print(f"\nWrote: {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
