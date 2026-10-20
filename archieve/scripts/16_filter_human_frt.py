"""
Filter the FRT centroid file to keep only human, intentional, preventive
fuel-reduction treatments, following Wear, Wibbenmeyer & Zhu (2025).

Drops two classes of records:
  (1) Wildfire-related activity codes (these are reactive, not preventive).
  (2) Activity codes unrelated to fuels management (regeneration cuts, site
      preparation, salvage, sanitation, etc.) per their footnote 2.

In addition we drop CAT_NM == 'Wildfire Non-Treatment' as a belt-and-braces
catch for the wildfire-product records.

Outputs:
  data/processed/frt_centroids_west_clean.parquet
"""
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
SRC = ROOT / "data" / "processed" / "frt_centroids_us.parquet"
DST = ROOT / "data" / "processed" / "frt_centroids_us_clean.parquet"

# -----------------------------------------------------------------------------
# Wear et al. (2025) exclusion lists
# -----------------------------------------------------------------------------
DROP_WILDFIRE = {
    "Wildfire - Natural Ignition",
    "Wildfire - Human Ignition",
    "Wildfire - Fuels Benefit",
    "Wildland Fire Use",
    "Planned Treatment Burned in Wildfire",
}

# Wear footnote 2: activities unrelated to intentional fuels management.
# We keep their list verbatim where the FACTS spelling matches, plus the
# additional silvicultural cuts and site-prep variants that fall under the
# same logic.
DROP_NONFUELS = {
    # Wear's explicit list
    "Landing Treated - Area Mitigated",
    "Stand Clearcut (w/ leave trees) (EA/RH/FH)",
    "Stand Clearcut (EA/RH/FH)",
    "Fuel Break",
    "Tree Release and Weed",
    "Salvage Cut (intermediate treatment, not regeneration)",
    "Prune",
    "Sanitation Cut",
    "Improvement Cut",
    "Wildlife Habitat Precommercial thinning",
    "Slashing - Pre-Site Preparation",
    "Single Tree Selection Method (UA/RH/FH)",
    "Site Preparation for Planting - Manual",
    "Site Preparation for Planting - Chemical",
    # Other clearly-non-fuels activities consistent with their spirit
    "Site Preparation for Planting - Mechanical",
    "Site Preparation for Planting - Burning",
    "Site Preparation for Planting - Other",
    "Site Preparation for Natural Regeneration - Manual",
    "Site Preparation for Natural Regeneration - Mechanical",
    "Site Preparation for Natural Regeneration - Burning",
    "Site Preparation for Natural Regeneration - Chemical",
    "Site Preparation for Natural Regeneration - Other",
    "Patch Clearcut (EA/RH/FH)",
    "Patch Clearcut (w/ leave trees) (EA/RH/FH)",
    "Two-aged Stand Clearcut (w/res) (2A/RH/FH)",
    "Two-aged Patch Clearcut (w/res) (2A/RH/FH)",
    "Two-aged Shelterwood Establishment Cut (w/res) (2A/RH/NFH)",
    "Two-aged Shelterwood Establishment and Removal Cut (w/ res) (2A/RH/FH)",
    "Two-aged Shelterwood Final Removal Cut (w/res) (2A/NRH/FH)",
    "Two-aged Seed-tree Seed and Removal Cut (w/res) (2A/RH/FH)",
    "Two-aged Preparatory Cut (w/res) (2A/NRH/NFH)",
    "Shelterwood Establishment Cut (with or without leave trees) (EA/RH/NFH)",
    "Shelterwood Removal Cut (EA/NRH/FH)",
    "Shelterwood Removal Cut (w/ leave trees) (EA/NRH/FH)",
    "Shelterwood Preparatory Cut (EA/NRH/NFH)",
    "Shelterwood Staged Removal Cut (EA/NRH/NFH)",
    "Seed-tree Seed Cut (with and without leave trees) (EA/RH/NFH)",
    "Seed-tree Removal Cut (w/ leave trees) (EA/NRH/FH)",
    "Seed-tree Preparatory Cut (EA/NRH/NFH)",
    "Seed-tree Final Cut (EA/NRH/FH)",
    "Coppice Cut (w/leave trees) (EA/RH/FH)",
    "Coppice Cut (EA/RH/FH)",
    "Liberation Cut",
    "Overstory Removal Cut (from advanced regeneration) (EA/RH/FH)",
    "Harvest Without Restocking",
    "Permanent Land Clearing",
    "Special Products Removal",
    "Recreation Removal of hazard trees and snags - Area",
    "Right of Way Maintenance",
    "Road Maintenance - Vegetation Reduction",
    "Wildlife Habitat Regeneration cut",
    "Wildlife Habitat Intermediate cut",
    "Anadromous Fish Thinning for Fish Habitat Improvement",
    "Visual Resource Slash treatment",
    "Precommercial thinning for visual",
    "Re-vegetation treatments - herbicides",
    "Re-vegetation treatments - vegetation removal",
    "Invasives - Pesticide Application",
    "Invasives - Mechanical /Physical",
    "Invasives - Cultural /Fire",
    "Invasives - Biocontrol, Livestock",
    "Range Cover Manipulation",
    "Range Control Vegetation",
    "Range Piling Slash",
    "Range NEPA/AMP Implementation",
    "Natural regeneration - prescribed fire",
}

DROP_ALL = DROP_WILDFIRE | DROP_NONFUELS

print(f"Reading {SRC} ...")
df = pd.read_parquet(SRC)
n_in = len(df)
print(f"  in: {n_in:,} records")

# Stage 1: drop CAT_NM-level wildfire products
mask_cat = df["CAT_NM"].isin({"Wildfire Non-Treatment"})
n_cat = int(mask_cat.sum())

# Stage 2: drop activity-level exclusions
mask_act_wf  = df["ACTIVITY"].isin(DROP_WILDFIRE)
mask_act_nf  = df["ACTIVITY"].isin(DROP_NONFUELS)
n_act_wf = int(mask_act_wf.sum())
n_act_nf = int(mask_act_nf.sum())

drop_mask = mask_cat | mask_act_wf | mask_act_nf
df_clean = df.loc[~drop_mask].copy()
n_out = len(df_clean)

print()
print("Drop counts:")
print(f"  CAT_NM == 'Wildfire Non-Treatment'           : {n_cat:>9,}")
print(f"  ACTIVITY in wildfire list                    : {n_act_wf:>9,}")
print(f"  ACTIVITY in non-fuels list                   : {n_act_nf:>9,}")
print(f"  Union of all three (overlap removed)         : {int(drop_mask.sum()):>9,}")
print()
print(f"Records out: {n_out:,}  ({n_out/n_in*100:.1f}% of input)")
print(f"Acres in : {df['GIS_ACRES'].sum():>15,.0f}")
print(f"Acres out: {df_clean['GIS_ACRES'].sum():>15,.0f}  "
      f"({df_clean['GIS_ACRES'].sum()/df['GIS_ACRES'].sum()*100:.1f}% of input)")
print()
print("By CAT_NM after cleaning:")
print(df_clean.groupby("CAT_NM").size().sort_values(ascending=False).to_string())
print()
print("Top 20 surviving activities:")
top = df_clean["ACTIVITY"].value_counts().head(20)
for a, c in top.items():
    print(f"  {c:>7,}  {a}")

df_clean.to_parquet(DST)
print(f"\nWrote: {DST}  ({DST.stat().st_size/1e6:.1f} MB)")
