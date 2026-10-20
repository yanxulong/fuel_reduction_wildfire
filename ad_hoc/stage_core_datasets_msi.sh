#!/usr/bin/env bash
# Stage current public source data on MSI. Run from the MSI clone under code/.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RAW_ROOT="$PROJECT_ROOT/data/raw"
RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_DATE="${DATA_SNAPSHOT_DATE:-${RUN_ID%%T*}}"
MANIFEST="$RAW_ROOT/dataset_download_manifest_${RUN_ID}.tsv"

if [[ ! "$RUN_DATE" =~ ^[0-9]{8}$ ]]; then
    printf '[error] DATA_SNAPSHOT_DATE must use YYYYMMDD format; got: %s\n' "$RUN_DATE" >&2
    exit 2
fi

FACTS_URL="https://data.fs.usda.gov/geodata/edw/edw_resources/shp/Actv_HazFuelTrt_PL.zip"
MTBS_URL="https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
TIGER_URL="https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
PRISM_BASE="https://services.nacse.org/prism/data/public/4km"

mkdir -p "$RAW_ROOT"
printf 'dataset\tpath\tsource_url\tbytes\tsha256\trecorded_utc\n' > "$MANIFEST"
printf '[start] project: %s\n' "$PROJECT_ROOT"
printf '[start] raw root: %s\n' "$RAW_ROOT"
printf '[start] snapshot date: %s\n' "$RUN_DATE"
printf '[start] manifest: %s\n' "$MANIFEST"
df -h "$RAW_ROOT"
if command -v quota >/dev/null 2>&1; then
    quota -s || true
fi

record_file() {
    local dataset="$1" path="$2" url="$3" bytes sha
    bytes="$(wc -c < "$path" | tr -d ' ')"
    sha="$(sha256sum "$path" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "${path#"$PROJECT_ROOT"/}" "$url" "$bytes" "$sha" \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$MANIFEST"
}

download_zip() {
    local dataset="$1" url="$2" dest="$3" part="${3}.part" bad
    mkdir -p "$(dirname -- "$dest")"

    if [[ -s "$dest" ]] && unzip -tq "$dest" >/dev/null 2>&1; then
        printf '[skip] %s: valid archive already exists (%s bytes)\n' \
            "$dataset" "$(wc -c < "$dest" | tr -d ' ')"
        record_file "$dataset" "$dest" "$url"
        return
    fi

    if [[ -e "$dest" ]]; then
        bad="${dest}.invalid.${RUN_ID}"
        mv -- "$dest" "$bad"
        printf '[warn] moved invalid archive to %s\n' "$bad"
    fi

    if [[ -s "$part" ]] && unzip -tq "$part" >/dev/null 2>&1; then
        mv -- "$part" "$dest"
        printf '[resume] %s: completed archive recovered from partial file\n' "$dataset"
    else
        printf '[download] %s\n  url: %s\n  to:  %s\n' "$dataset" "$url" "$part"
        if ! curl --fail --location --retry 5 \
            --retry-delay 5 --continue-at - --progress-bar \
            --output "$part" "$url"; then
            printf '[error] %s download failed; partial file retained for resume\n' "$dataset" >&2
            return 1
        fi
        if ! unzip -tq "$part" >/dev/null; then
            bad="${part}.invalid.${RUN_ID}"
            mv -- "$part" "$bad"
            printf '[error] %s response was not a valid ZIP; saved at %s\n' \
                "$dataset" "$bad" >&2
            return 1
        fi
        mv -- "$part" "$dest"
    fi

    record_file "$dataset" "$dest" "$url"
    printf '[verified] %s: %s bytes; ZIP integrity passed\n' \
        "$dataset" "$(wc -c < "$dest" | tr -d ' ')"
}

printf '\n[stage 1/4] FACTS treatment polygons\n'
FACTS_DIR="$RAW_ROOT/FACTS/snapshot_${RUN_DATE}"
FACTS_ZIP="$FACTS_DIR/Actv_HazFuelTrt_PL.zip"
FACTS_EXTRACT="$FACTS_DIR/Actv_HazFuelTrt_PL"
FACTS_SHP="$FACTS_EXTRACT/Actv_HazFuelTrt_PL.shp"
if [[ -s "$FACTS_SHP" && \
      -s "$FACTS_EXTRACT/Actv_HazFuelTrt_PL.shx" && \
      -s "$FACTS_EXTRACT/Actv_HazFuelTrt_PL.dbf" && \
      -s "$FACTS_EXTRACT/Actv_HazFuelTrt_PL.prj" ]]; then
    printf '[skip] complete FACTS shapefile bundle already staged: %s\n' "$FACTS_EXTRACT"
    for ext in shp shx dbf prj cpg; do
        path="$FACTS_EXTRACT/Actv_HazFuelTrt_PL.$ext"
        if [[ -s "$path" ]]; then
            record_file "FACTS_HazFuelTrt_PL_${ext}" "$path" "local_upload_or_existing_bundle"
        fi
    done
else
    download_zip "FACTS_HazFuelTrt_PL" "$FACTS_URL" "$FACTS_ZIP"
    mkdir -p "$FACTS_EXTRACT"
    unzip -oq "$FACTS_ZIP" -d "$FACTS_EXTRACT"
fi
FACTS_SHP="$(find "$FACTS_EXTRACT" -type f -name 'Actv_HazFuelTrt_PL.shp' -print -quit)"
if [[ -z "$FACTS_SHP" ]]; then
    printf '[error] expected Actv_HazFuelTrt_PL.shp missing after extraction\n' >&2
    exit 1
fi
printf '[verified] FACTS shapefile: %s\n' "$FACTS_SHP"

printf '\n[stage 2/4] PRISM annual temperature and precipitation, 2000-2024\n'
PRISM_DIR="$RAW_ROOT/prism"
total=50
done_count=0
for variable in tmean ppt; do
    for year in $(seq 2000 2024); do
        dest="$PRISM_DIR/$variable/prism_${variable}_${year}.zip"
        url="$PRISM_BASE/$variable/$year"
        download_zip "PRISM_${variable}_${year}" "$url" "$dest"
        done_count=$((done_count + 1))
        printf '[progress] PRISM %d/%d archives complete\n' "$done_count" "$total"
    done
done

printf '\n[stage 3/4] TIGER/Line 2023 national county boundaries\n'
TIGER_DIR="$RAW_ROOT/tiger/2023"
TIGER_ZIP="$TIGER_DIR/tl_2023_us_county.zip"
download_zip "TIGER_2023_US_COUNTY" "$TIGER_URL" "$TIGER_ZIP"
TIGER_EXTRACT="$TIGER_DIR/tl_2023_us_county"
mkdir -p "$TIGER_EXTRACT"
unzip -oq "$TIGER_ZIP" -d "$TIGER_EXTRACT"
if [[ ! -s "$TIGER_EXTRACT/tl_2023_us_county.shp" ]]; then
    printf '[error] expected tl_2023_us_county.shp missing after extraction\n' >&2
    exit 1
fi
printf '[verified] TIGER county shapefile: %s\n' "$TIGER_EXTRACT/tl_2023_us_county.shp"

printf '\n[stage 4/4] MTBS national burned-area perimeter polygons\n'
MTBS_DIR="$RAW_ROOT/MTBS/snapshot_${RUN_DATE}"
MTBS_ZIP="$MTBS_DIR/mtbs_perimeter_data.zip"
download_zip "MTBS_PERIMETERS" "$MTBS_URL" "$MTBS_ZIP"
MTBS_EXTRACT="$MTBS_DIR/mtbs_perimeter_data"
mkdir -p "$MTBS_EXTRACT"
unzip -oq "$MTBS_ZIP" -d "$MTBS_EXTRACT"
MTBS_SHP="$(find "$MTBS_EXTRACT" -type f -iname '*perim*.shp' -print -quit)"
if [[ -z "$MTBS_SHP" ]]; then
    printf '[error] no MTBS perimeter shapefile found after extraction\n' >&2
    exit 1
fi
printf '[verified] MTBS perimeter shapefile: %s\n' "$MTBS_SHP"

printf '\n[complete] All requested dataset archives passed ZIP validation.\n'
printf '[output] FACTS: %s\n' "$FACTS_DIR"
printf '[output] PRISM: %s\n' "$PRISM_DIR"
printf '[output] TIGER: %s\n' "$TIGER_DIR"
printf '[output] MTBS: %s\n' "$MTBS_DIR"
printf '[output] manifest: %s\n' "$MANIFEST"
printf '[summary] %s\n' "$(du -sh "$FACTS_DIR" "$PRISM_DIR" "$TIGER_DIR" "$MTBS_DIR" | tr '\n' ';')"
