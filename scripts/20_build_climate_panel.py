"""
Download PRISM 4km annual temperature and precipitation (2000-2024),
sample at WUI centroids, build a (wui_id, year, tmean_c, ppt_mm) panel,
and merge into the existing panel_wui_year.parquet.

Data source: PRISM Climate Group, Oregon State University
  https://services.nacse.org/prism/data/get/us/4km/{var}/{year}

Variables:
  tmean = annual mean of monthly mean temperatures, degrees C
  ppt   = annual total precipitation, mm

Output:
  data/processed/wui_climate_panel.parquet   (wui_id, year, tmean_c, ppt_mm)
  data/processed/panel_wui_year.parquet      (overwritten with climate columns)
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
RAW = ROOT / "data" / "raw" / "prism"
RAW.mkdir(parents=True, exist_ok=True)

OUT_CLIM = ROOT / "data" / "processed" / "wui_climate_panel.parquet"
OUT_PANEL = ROOT / "data" / "processed" / "panel_wui_year.parquet"

YEARS = list(range(2000, 2025))
VARS = ["tmean", "ppt"]

HEADERS = {"User-Agent": "fuel-reduction-paper-research/1.0 (yxlong@umn.edu)"}


# -----------------------------------------------------------------------------
# 1. Download PRISM rasters (one zip per var-year)
# -----------------------------------------------------------------------------
def download(var: str, year: int) -> Path:
    out_zip = RAW / var / f"prism_{var}_{year}.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists() and out_zip.stat().st_size > 100_000:
        return out_zip
    url = f"https://services.nacse.org/prism/data/get/us/4km/{var}/{year}"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            if len(r.content) < 100_000:
                # Throttle response, back off
                print(f"  [{var} {year}] tiny response ({len(r.content)} B), retry ...")
                time.sleep(30)
                continue
            out_zip.write_bytes(r.content)
            return out_zip
        except Exception as e:
            print(f"  [{var} {year}] attempt {attempt+1} failed: {e}")
            time.sleep(10)
    raise RuntimeError(f"failed to download {var} {year}")


print("=" * 72)
print("1. Downloading PRISM 4km annual rasters")
print("=" * 72)
for var in VARS:
    for year in YEARS:
        out = download(var, year)
        print(f"  {var:6s} {year}  {out.stat().st_size/1e6:5.1f} MB  {out.name}")
        time.sleep(1)  # polite pause between requests
print()


# -----------------------------------------------------------------------------
# 2. Load WUI centroids and reproject to EPSG:4269 (PRISM CRS)
# -----------------------------------------------------------------------------
print("=" * 72)
print("2. Loading WUI centroids")
print("=" * 72)
wui = pd.read_parquet(ROOT / "data" / "processed" / "wui_centroids_us.parquet")
print(f"  loaded {len(wui):,} centroids")

# wui lon/lat are in EPSG:5070 meters (see scripts 18 & 13)
print("  reprojecting 5070 meters -> 4269 lon/lat ...")
tf = Transformer.from_crs(5070, 4269, always_xy=True)
lon_deg, lat_deg = tf.transform(wui["lon"].values, wui["lat"].values)
wui_pts = pd.DataFrame({
    "wui_id": wui["wui_id"].values,
    "lon_deg": lon_deg,
    "lat_deg": lat_deg,
})
wui_pts = wui_pts.dropna(subset=["lon_deg", "lat_deg"]).reset_index(drop=True)
print(f"  kept {len(wui_pts):,} valid centroids")
print(f"  lon range: [{wui_pts.lon_deg.min():.3f}, {wui_pts.lon_deg.max():.3f}]")
print(f"  lat range: [{wui_pts.lat_deg.min():.3f}, {wui_pts.lat_deg.max():.3f}]")
print()

coords = list(zip(wui_pts["lon_deg"].values, wui_pts["lat_deg"].values))


# -----------------------------------------------------------------------------
# 3. Sample rasters at centroids, one var-year at a time
# -----------------------------------------------------------------------------
def sample_raster(zip_path: Path) -> np.ndarray:
    # Find the .tif inside the zip
    with zipfile.ZipFile(zip_path) as z:
        tif_name = next(n for n in z.namelist() if n.endswith(".tif"))
    vsi = f"/vsizip/{zip_path}/{tif_name}"
    with rasterio.open(vsi) as ds:
        vals = np.fromiter(
            (v[0] for v in ds.sample(coords)),
            dtype="float32",
            count=len(coords),
        )
    # Replace nodata with NaN
    vals = np.where(vals < -9000, np.nan, vals)
    return vals


print("=" * 72)
print("3. Sampling PRISM rasters at WUI centroids")
print("=" * 72)
records = []
for year in YEARS:
    t0 = time.time()
    tmean_vals = sample_raster(RAW / "tmean" / f"prism_tmean_{year}.zip")
    ppt_vals = sample_raster(RAW / "ppt" / f"prism_ppt_{year}.zip")
    yr_df = pd.DataFrame({
        "wui_id": wui_pts["wui_id"].values,
        "year": year,
        "tmean_c": tmean_vals,
        "ppt_mm": ppt_vals,
    })
    records.append(yr_df)
    dt = time.time() - t0
    n_ok = int(np.isfinite(tmean_vals).sum())
    print(f"  [{year}] tmean mean={np.nanmean(tmean_vals):5.2f}C  "
          f"ppt mean={np.nanmean(ppt_vals):6.0f}mm  "
          f"n_valid={n_ok:,}  ({dt:.1f}s)")

climate = pd.concat(records, ignore_index=True)
climate.to_parquet(OUT_CLIM)
print(f"\nWrote: {OUT_CLIM}  ({OUT_CLIM.stat().st_size/1e6:.1f} MB)")
print(f"Climate panel: {len(climate):,} rows")
print(f"  tmean range: [{climate['tmean_c'].min():.2f}, {climate['tmean_c'].max():.2f}] C")
print(f"  ppt   range: [{climate['ppt_mm'].min():.0f}, {climate['ppt_mm'].max():.0f}] mm")
print(f"  tmean NaN: {climate['tmean_c'].isna().sum():,}")
print(f"  ppt   NaN: {climate['ppt_mm'].isna().sum():,}")
print()


# -----------------------------------------------------------------------------
# 4. Merge into the panel
# -----------------------------------------------------------------------------
print("=" * 72)
print("4. Merging climate into panel_wui_year")
print("=" * 72)
panel = pd.read_parquet(OUT_PANEL)
print(f"  panel shape before: {panel.shape}")
print(f"  panel columns: {list(panel.columns)}")
before_cols = set(panel.columns)
panel = panel.merge(climate, on=["wui_id", "year"], how="left")
new_cols = [c for c in panel.columns if c not in before_cols]
print(f"  new columns: {new_cols}")
print(f"  panel shape after:  {panel.shape}")
print(f"  tmean NaN in panel: {panel['tmean_c'].isna().sum():,} "
      f"({panel['tmean_c'].isna().mean()*100:.2f}%)")
print(f"  ppt   NaN in panel: {panel['ppt_mm'].isna().sum():,} "
      f"({panel['ppt_mm'].isna().mean()*100:.2f}%)")

panel.to_parquet(OUT_PANEL)
print(f"\nOverwrote: {OUT_PANEL}  ({OUT_PANEL.stat().st_size/1e6:.1f} MB)")
