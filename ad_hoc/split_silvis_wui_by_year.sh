#!/usr/bin/env bash
set -euo pipefail

for command_name in ogr2ogr ogrinfo awk mktemp tee; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "${command_name}" >&2
        if [[ "${command_name}" == ogr2ogr || "${command_name}" == ogrinfo ]]; then
            printf 'On MSI, load GDAL first: module load gdal/3.12.1-gcc-11.3.0-netcdf-4.9.3\n' >&2
        fi
        exit 1
    fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd -- "${CODE_DIR}/.." && pwd)"
SOURCE_GDB="${PROJECT_DIR}/data/raw/silvis_wui/CONUS_wui_block_1990_2020_change_v4_fgdb/CONUS_WUI_block_1990_2020_change_v4_web.gdb"
SOURCE_LAYER="CONUS_WUI_block_1990_2020_change_v4"
OUTPUT_DIR="${PROJECT_DIR}/data/processed/wui_by_year"
LOG_PATH="${OUTPUT_DIR}/split_silvis_wui_by_year.log"
YEARS=(1990 2000 2010 2020)

if [[ ! -d "${SOURCE_GDB}" ]]; then
    printf 'Source File Geodatabase not found: %s\n' "${SOURCE_GDB}" >&2
    exit 1
fi

if ! ogrinfo --format FlatGeobuf >/dev/null 2>&1; then
    printf 'GDAL FlatGeobuf driver unavailable. Check loaded GDAL module.\n' >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

for year in "${YEARS[@]}"; do
    output_path="${OUTPUT_DIR}/silvis_wui_${year}.fgb"
    if [[ -e "${output_path}" ]]; then
        printf 'Output already exists; refusing to overwrite: %s\n' "${output_path}" >&2
        exit 1
    fi
done

SOURCE_FEATURE_COUNT="$(
    ogrinfo -ro -so "${SOURCE_GDB}" "${SOURCE_LAYER}" |
        awk -F': ' '/Feature Count:/ { print $2; exit }'
)"
if [[ ! "${SOURCE_FEATURE_COUNT}" =~ ^[0-9]+$ ]]; then
    printf 'Could not read source feature count for layer %s.\n' "${SOURCE_LAYER}" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d "${OUTPUT_DIR}/.split_silvis_wui.XXXXXX")"
trap 'rm -rf -- "${WORK_DIR}"' EXIT

printf 'Source layer: %s (%s features)\n' "${SOURCE_LAYER}" "${SOURCE_FEATURE_COUNT}"
printf 'All records retained, including non-WUI blocks; output format: FlatGeobuf.\n'

for year in "${YEARS[@]}"; do
    case "${year}" in
        1990) year_fields=(HU1990 HUDEN1990 VEG1992PC WUIFLAG1990 WUICLASS1990) ;;
        2000) year_fields=(HU2000 HUDEN2000 VEG2001PC WUIFLAG2000 WUICLASS2000) ;;
        2010) year_fields=(HU2010 HUDEN2010 VEG2011PC WUIFLAG2010 WUICLASS2010) ;;
        2020) year_fields=(HU2020 HUDEN2020 VEG2019PC WUIFLAG2020 WUICLASS2020) ;;
        *)
            printf 'Unsupported mapped year: %s\n' "${year}" >&2
            exit 1
            ;;
    esac

    selected_fields=(BLK20 STATE fips PUBFLAG WATER20 AWATER20PC "${year_fields[@]}")
    select_list="$(IFS=,; printf '%s' "${selected_fields[*]}")"
    temporary_output="${WORK_DIR}/silvis_wui_${year}.fgb"
    temporary_layer="wui_${year}"

    printf 'Exporting %s to FlatGeobuf...\n' "${year}"
    ogr2ogr \
        -f FlatGeobuf "${temporary_output}" \
        "${SOURCE_GDB}" "${SOURCE_LAYER}" \
        -select "${select_list}" \
        -nln "${temporary_layer}" \
        -lco SPATIAL_INDEX=YES \
        -progress

    if [[ ! -s "${temporary_output}" ]]; then
        printf 'Export produced an empty file for %s.\n' "${year}" >&2
        exit 1
    fi

    output_feature_count="$(
        ogrinfo -ro -so "${temporary_output}" "${temporary_layer}" |
            awk -F': ' '/Feature Count:/ { print $2; exit }'
    )"
    if [[ "${output_feature_count}" != "${SOURCE_FEATURE_COUNT}" ]]; then
        printf 'Feature-count mismatch for %s: source=%s output=%s\n' \
            "${year}" "${SOURCE_FEATURE_COUNT}" "${output_feature_count}" >&2
        exit 1
    fi

    printf 'Validated %s: %s features.\n' "${year}" "${output_feature_count}"
done

for year in "${YEARS[@]}"; do
    mv -- \
            "${WORK_DIR}/silvis_wui_${year}.fgb" \
            "${OUTPUT_DIR}/silvis_wui_${year}.fgb"
    printf 'Created: %s\n' "${OUTPUT_DIR}/silvis_wui_${year}.fgb"
done

printf 'All four year files created successfully.\n'
