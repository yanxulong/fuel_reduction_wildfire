# Ad hoc research tasks

Use this folder for one-off data intake and maintenance tasks that do not belong in the repeatable analysis pipeline. Keep task scripts reviewable and rerunnable. Store downloaded data under the project `data/` tree, never in Git.

## SILVIS WUI Change 1990–2020, Version 4

`download_silvis_wui.sh` downloads the conterminous-U.S. File Geodatabase archive linked from the [official SILVIS WUI Change page](https://silvis.forest.wisc.edu/data/wui-change/). Product covers WUI for 1990, 2000, 2010, and 2020. It saves the archive and a verification record under the project’s `data/raw/silvis_wui/` tree. The download resumes from a partial `.part` file and checks ZIP integrity before promoting it to the final filename.

Run from the MSI checkout after syncing code:

```bash
cd /users/8/yxlong/projects/fuel_reduction_wildfire/code
bash ad_hoc/download_silvis_wui.sh
```

Check available project storage before starting. Do not commit the archive or generated metadata.

## Split SILVIS WUI by mapped year

`split_silvis_wui_by_year.sh` exports the extracted File Geodatabase into four single-file FlatGeobuf files under `data/processed/wui_by_year/`: 1990, 2000, 2010, and 2020. Each output retains every source block, including non-WUI blocks (`WUIFLAG` 0), so non-WUI comparison areas remain available. It keeps block/state/FIPS identifiers, water and protected-area fields, plus housing, density, vegetation, WUI flag, and WUI class fields for the selected year. FlatGeobuf is available in the MSI GDAL build and supports spatial indexing. The SILVIS source describes the four mapped years and WUI classes on its [WUI Change page](https://silvis.forest.wisc.edu/data/wui-change/).

Run from the MSI checkout in an interactive compute allocation after syncing code:

```bash
cd /users/8/yxlong/projects/fuel_reduction_wildfire/code
module load gdal/3.12.1-gcc-11.3.0-netcdf-4.9.3
bash ad_hoc/split_silvis_wui_by_year.sh
```

The source has about 9.16 million block features; the script checks output counts and refuses to overwrite existing FlatGeobuf files. Check project quota before running. Keep generated data and logs out of Git.

## Stage FACTS, PRISM, TIGER, and MTBS on MSI

`stage_core_datasets_msi.sh` downloads public source archives directly to MSI under `data/raw/`, validates ZIP integrity, records source URLs and SHA-256 hashes, and extracts the FACTS, TIGER, and MTBS shapefiles. PRISM covers annual 4-km `tmean` and `ppt` for 2000–2024 (50 ZIPs). TIGER uses 2023 county boundaries to match the project's documented 2023 boundary vintage. MTBS scope here is national burned-area perimeter polygons, not annual severity rasters.

Run from the MSI clone:

```bash
cd /users/8/yxlong/projects/fuel_reduction_wildfire/code
bash ad_hoc/stage_core_datasets_msi.sh
```

The script keeps date-stamped FACTS and MTBS snapshots. Set `DATA_SNAPSHOT_DATE=YYYYMMDD` to resume or rerun a snapshot on a later day. Keep all downloads, extractions, and manifests out of Git.
