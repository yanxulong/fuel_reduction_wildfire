#!/usr/bin/env bash
set -euo pipefail

SOURCE_URL="https://geoserver.silvis.forest.wisc.edu/geodata/wui_change_2020_v4/zip/fgdb/CONUS_wui_block_1990_2020_change_v4_fgdb.zip"
ARCHIVE_NAME="CONUS_wui_block_1990_2020_change_v4_fgdb.zip"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd -- "${CODE_DIR}/.." && pwd)"
DEST_DIR="${PROJECT_DIR}/data/raw/silvis_wui"
ARCHIVE_PATH="${DEST_DIR}/${ARCHIVE_NAME}"
PART_PATH="${ARCHIVE_PATH}.part"
METADATA_PATH="${DEST_DIR}/${ARCHIVE_NAME}.metadata.txt"

for command_name in curl unzip sha256sum stat; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "${command_name}" >&2
        exit 1
    fi
done

mkdir -p "${DEST_DIR}"

if [[ -f "${ARCHIVE_PATH}" ]] && unzip -tq "${ARCHIVE_PATH}" >/dev/null 2>&1; then
    printf 'Verified archive already exists: %s\n' "${ARCHIVE_PATH}"
else
    printf 'Downloading or resuming SILVIS WUI archive to: %s\n' "${PART_PATH}"
    curl --fail --location --retry 5 --retry-delay 10 --retry-connrefused \
        --continue-at - --output "${PART_PATH}" "${SOURCE_URL}"
    printf 'Checking ZIP integrity...\n'
    unzip -tq "${PART_PATH}"
    mv -f -- "${PART_PATH}" "${ARCHIVE_PATH}"
fi

ARCHIVE_BYTES="$(stat -c '%s' "${ARCHIVE_PATH}")"
ARCHIVE_SHA256="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
VERIFIED_AT_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

cat > "${METADATA_PATH}" <<EOF
source_url=${SOURCE_URL}
product=SILVIS WUI Change 1990-2020 Version 4
extent=Conterminous United States
mapped_years=1990,2000,2010,2020
archive=${ARCHIVE_NAME}
archive_bytes=${ARCHIVE_BYTES}
sha256=${ARCHIVE_SHA256}
verified_at_utc=${VERIFIED_AT_UTC}
EOF

printf 'Verified archive: %s (%s bytes)\n' "${ARCHIVE_PATH}" "${ARCHIVE_BYTES}"
printf 'SHA-256: %s\n' "${ARCHIVE_SHA256}"
printf 'Metadata: %s\n' "${METADATA_PATH}"
