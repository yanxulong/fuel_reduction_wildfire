"""
Two follow-up diagnostics motivated by 26_diag_did_vs_longdiff.py:

  D1. Ashenfelter test: do polygons get treated the year after a disaster?
      Compare the FACTS-treatment hazard rate in year t conditional on a
      disaster in year t-1 vs no disaster in t-1. If the post-disaster
      hazard is materially higher, that explains the k=-1 spike.

  D2. Density-class baseline variance: is the matched DiD heterogeneity
      gradient (Low +, Med -, High --) explainable by baseline-rate
      mean reversion alone? Check var(disaster) by density class on the
      pre-period of treated polygons. If High has higher baseline variance,
      mean reversion mechanically gives a larger negative coefficient there.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
m = pd.read_parquet(ROOT / "data" / "processed" / "panel_wui_year_v2.parquet")
m = m[m["matched_sample"] == 1].copy()

ever = m.groupby("wui_id")["FRT_it"].max()
treated_ids = set(ever[ever == 1].index)
m["is_treated"] = m["wui_id"].isin(treated_ids)
treated = m[m["is_treated"]].copy().sort_values(["wui_id", "year"])

# ============================================================
# D1. Ashenfelter test
# ============================================================
print("=" * 72)
print("D1. ASHENFELTER TEST: P(first treatment in t | disaster in t-1) vs no")
print("=" * 72)

# year of first treatment per treated polygon
g = treated.loc[treated["FRT_it"] == 1].groupby("wui_id")["year"].min()
treated["g"] = treated["wui_id"].map(g)
# build "got treated this year" indicator (transition from FRT=0 to FRT=1)
treated["new_treated"] = (treated["year"] == treated["g"]).astype(int)
# previous-year disaster
treated["disaster_lag1"] = treated.groupby("wui_id")["disaster"].shift(1)

# in-sample years where polygon was eligible (year < g + 1, ie pre-treatment plus the year of treatment)
elig = treated[(treated["year"] <= treated["g"]) & treated["disaster_lag1"].notna()].copy()
print(f"  eligible polygon-years for hazard model: {len(elig):,}")
hazard = elig.groupby("disaster_lag1")["new_treated"].agg(["mean", "count"])
print(hazard.to_string())
ratio = hazard.loc[1, "mean"] / hazard.loc[0, "mean"] if 0 in hazard.index and 1 in hazard.index else float("nan")
print(f"\n  hazard ratio: P(treat_t | disaster_{{t-1}}=1) / P(treat_t | disaster_{{t-1}}=0) = {ratio:.2f}")
print("    (>>1 means treatment is concentrated in years following disaster years)")

# also lag-2
treated["disaster_lag2"] = treated.groupby("wui_id")["disaster"].shift(2)
elig2 = treated[(treated["year"] <= treated["g"]) & treated["disaster_lag2"].notna()].copy()
hazard2 = elig2.groupby("disaster_lag2")["new_treated"].agg(["mean", "count"])
print("\n  Lag-2 (placebo)")
print(hazard2.to_string())

# ============================================================
# D2. Density-class baseline variance
# ============================================================
print("\n" + "=" * 72)
print("D2. BASELINE VARIANCE BY DENSITY CLASS (treated polygons only, pre-period)")
print("=" * 72)

PRE = (2000, 2002)
treated_pre = treated[(treated["year"] >= PRE[0]) & (treated["year"] <= PRE[1])]

# detect the density column
candidates = [c for c in m.columns if "dens" in c.lower() or "wui_class" in c.lower() or c.lower() in ("density_class", "class")]
print(f"  candidate density columns: {candidates}")

# infer from class dummies
for c in m.columns:
    if c.lower() in ("low", "medium", "high", "low_density", "med_density", "high_density"):
        print(f"   {c}: nonzero share = {m[c].mean():.4f}")

# best guess based on script 22-23: typically there is a 'class' or 'wui_class' column
# we'll compute baseline disaster mean and variance by all candidate density-like columns
for col in candidates + ["wui_class", "density"]:
    if col in m.columns:
        print(f"\n  using {col}")
        agg = treated_pre.groupby(col)["disaster"].agg(["mean", "var", "count"])
        print(agg.to_string())

# event-study coefficients are reported by Low/Med/High in the paper. Just compute
# the overall pre-period disaster RATE for treated polygons grouped by something we know
# exists: the SILVIS density class dummies are usually named like 'IF_HD' etc
print("\n  raw class column inspection:")
for col in m.columns:
    if m[col].dtype == "O" and m[col].nunique() < 20:
        print(f"    {col:20s}: {m[col].unique()[:8]}")
