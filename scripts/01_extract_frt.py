"""
Extract FACTS Hazardous Fuel Treatment records from the local DBF.
Output: parquet/csv with one row per accomplished treatment, columns relevant to the analysis.
"""
import os
from pathlib import Path
from dbfread import DBF
import pandas as pd
import numpy as np

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
DBF_PATH = ROOT / "data" / "Actv_HazFuelTrt_PL.dbf"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fields to keep (must match DBF column names exactly)
KEEP = [
    "SUID",          # subunit ID
    "FACTS_ID",      # activity unit ID
    "ACTIVITY",      # activity description
    "CAT_NM",        # category: Mechanical or Fire
    "METHOD",        # method
    "EQUIPMENT",     # equipment
    "STAGE_VALU",    # Accomplished/Planned/Active
    "FY_AWARDED",    # fiscal year awarded
    "DATE_COMPL",    # date completed
    "GIS_ACRES",     # treatment area
    "COST_PER_U",    # unit cost
    "STATE_ABBR",    # state
    "FS_UNIT_ID",    # forest unit ID (key for Forest-Year aggregation)
    "FS_UNIT_NA",    # forest name
    "ISWUI",         # Y/N flag for WUI
    "CWPP",          # CWPP status
    "FUND_CODE",     # funding source
    "KEYPOINT",      # NFP key point
]

print(f"Reading DBF: {DBF_PATH}")
print(f"File size: {DBF_PATH.stat().st_size / 1e9:.2f} GB")

# DBF iteration is memory-efficient
table = DBF(str(DBF_PATH), load=False, encoding="utf-8")
print(f"Total records: {len(table):,}")
print(f"Available fields: {len(table.field_names)}")

# Find the actual field names that match our KEEP list (DBF may truncate)
available = set(table.field_names)
keep_actual = [f for f in KEEP if f in available]
missing = [f for f in KEEP if f not in available]
print(f"Keeping {len(keep_actual)} fields. Missing: {missing}")

# Stream through records, keep only Accomplished and FY_AWARDED in [2000, 2024]
rows = []
n_total = 0
n_kept = 0
for rec in table:
    n_total += 1
    if n_total % 100000 == 0:
        print(f"  processed {n_total:,} / kept {n_kept:,}")
    stage = rec.get("STAGE_VALU", "")
    fy = rec.get("FY_AWARDED")
    acres = rec.get("GIS_ACRES")
    if stage != "Accomplished":
        continue
    if fy is None or fy < 2000 or fy > 2024:
        continue
    if acres is None or acres <= 0:
        continue
    rows.append({k: rec.get(k) for k in keep_actual})
    n_kept += 1

print(f"\nDone. Total: {n_total:,}  Kept: {n_kept:,}  ({100*n_kept/n_total:.1f}%)")

df = pd.DataFrame(rows)
print(f"\nDataFrame shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Type cleanup
df["FY_AWARDED"] = pd.to_numeric(df["FY_AWARDED"], errors="coerce").astype("Int64")
df["GIS_ACRES"]  = pd.to_numeric(df["GIS_ACRES"], errors="coerce")
df["COST_PER_U"] = pd.to_numeric(df["COST_PER_U"], errors="coerce")

print("\nSummary:")
print(f"  Year range: {df['FY_AWARDED'].min()} - {df['FY_AWARDED'].max()}")
print(f"  Total acres: {df['GIS_ACRES'].sum():,.0f}")
print(f"  Unique forests (FS_UNIT_ID): {df['FS_UNIT_ID'].nunique()}")
print(f"  States: {df['STATE_ABBR'].nunique()}")
print(f"  CAT_NM distribution:\n{df['CAT_NM'].value_counts(dropna=False).head(10)}")
print(f"  ISWUI distribution:\n{df['ISWUI'].value_counts(dropna=False)}")

# Save
out_csv = OUT_DIR / "frt_clean.csv"
df.to_csv(out_csv, index=False)
print(f"\nWrote: {out_csv}  ({out_csv.stat().st_size / 1e6:.1f} MB)")
