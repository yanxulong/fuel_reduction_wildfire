"""
Additional diagnostics: treatment intensity normalized by forest size,
West-only subsample, longer lags, and propensity score visualization.
The Forest-Year aggregation may be too coarse to detect WUI-polygon mechanisms.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from linearmodels.panel import PanelOLS

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PANEL = ROOT / "data" / "processed" / "panel_forest_year.csv"
OUT = ROOT / "data" / "processed" / "diagnostics_results.txt"

df = pd.read_csv(PANEL)
df["FRT_it"] = (df["cum_acres"] > 0).astype(int)
df["treat_intensity"] = df["cum_acres"] / df["forest_acres"]
df["log_treat_intensity"] = np.log1p(df["treat_intensity"])
df["high_intensity"] = (df["treat_intensity"] > df["treat_intensity"].quantile(0.75)).astype(int)
df["west"] = df["REGION"].between(1, 6).astype(int)

# Compute lagged variables
df = df.sort_values(["FORESTORGC", "year"])
for lag in [1, 2, 3, 5]:
    df[f"FRT_lag{lag}"] = df.groupby("FORESTORGC")["FRT_it"].shift(lag).fillna(0).astype(int)
    df[f"log_cum_lag{lag}"] = np.log1p(
        df.groupby("FORESTORGC")["cum_acres"].shift(lag).fillna(0)
    )
    df[f"intensity_lag{lag}"] = (
        df.groupby("FORESTORGC")["treat_intensity"].shift(lag).fillna(0)
    )

# Cumulative treated share by year t
df["cum_treated_share"] = df["cum_acres"] / df["forest_acres"]
df["cum_treated_share_3yr"] = df.groupby("FORESTORGC")["cum_treated_share"].shift(3).fillna(0)

df_idx = df.set_index(["FORESTORGC", "year"])

lines = []
def log(s=""):
    print(s)
    lines.append(s)

def fit(formula, data, name, key, **kwargs):
    try:
        mod = PanelOLS.from_formula(formula, data=data, drop_absorbed=True, **kwargs)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        coef = res.params[key]
        se = res.std_errors[key]
        p = res.pvalues[key]
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        log(f"{name:50s}  beta={coef:+.6f} se={se:.6f} p={p:.4f}{sig:3s} N={int(res.nobs)}")
        return res
    except Exception as e:
        log(f"{name}: FAILED - {e}")
        return None

log("=" * 80)
log("DIAGNOSTIC 1: TREATMENT INTENSITY (normalized by forest size)")
log("=" * 80)
log(f"Treatment intensity (cum_acres/forest_acres) summary:")
log(str(df["treat_intensity"].describe()))
log("")

# Q: when normalized, is there an effect?
fit("disaster ~ treat_intensity + EntityEffects + TimeEffects", df_idx,
    "intensity_TWFE", "treat_intensity")
fit("disaster ~ log_treat_intensity + EntityEffects + TimeEffects", df_idx,
    "log_intensity_TWFE", "log_treat_intensity")
fit("disaster ~ high_intensity + EntityEffects + TimeEffects", df_idx,
    "high_intensity_TWFE", "high_intensity")

log("")
log("=" * 80)
log("DIAGNOSTIC 2: LONGER LAGS (treatments take time to show effects)")
log("=" * 80)
for lag in [1, 2, 3, 5]:
    fit(f"disaster ~ FRT_lag{lag} + EntityEffects + TimeEffects", df_idx,
        f"FRT_lag{lag}_TWFE", f"FRT_lag{lag}")
    fit(f"disaster ~ log_cum_lag{lag} + EntityEffects + TimeEffects", df_idx,
        f"log_cum_lag{lag}_TWFE", f"log_cum_lag{lag}")

log("")
log("=" * 80)
log("DIAGNOSTIC 3: WESTERN FORESTS ONLY (USFS Regions 1-6)")
log("=" * 80)
df_west = df[df["west"] == 1].copy()
log(f"Western sample: {len(df_west)} forest-years, {df_west['FORESTORGC'].nunique()} forests")
log(f"Disaster rate: {df_west['disaster'].mean():.4f}")
df_west_idx = df_west.set_index(["FORESTORGC", "year"])
fit("disaster ~ FRT_it + EntityEffects + TimeEffects", df_west_idx,
    "WEST_FRT_it", "FRT_it")
fit("disaster ~ log_treat_intensity + EntityEffects + TimeEffects", df_west_idx,
    "WEST_log_intensity", "log_treat_intensity")
fit("disaster ~ FRT_lag3 + EntityEffects + TimeEffects", df_west_idx,
    "WEST_FRT_lag3", "FRT_lag3")
fit("disaster ~ log_cum_lag3 + EntityEffects + TimeEffects", df_west_idx,
    "WEST_log_cum_lag3", "log_cum_lag3")

log("")
log("=" * 80)
log("DIAGNOSTIC 4: HIGH-RISK FORESTS (top quartile of disaster rate)")
log("=" * 80)
forest_disaster_rate = df.groupby("FORESTORGC")["disaster"].mean()
high_risk = forest_disaster_rate[forest_disaster_rate > forest_disaster_rate.quantile(0.75)].index
df_hr = df[df["FORESTORGC"].isin(high_risk)].copy()
log(f"High-risk sample: {len(df_hr)} forest-years, {df_hr['FORESTORGC'].nunique()} forests")
df_hr_idx = df_hr.set_index(["FORESTORGC", "year"])
fit("disaster ~ FRT_it + EntityEffects + TimeEffects", df_hr_idx,
    "HIGH_RISK_FRT_it", "FRT_it")
fit("disaster ~ log_cum_lag3 + EntityEffects + TimeEffects", df_hr_idx,
    "HIGH_RISK_log_cum_lag3", "log_cum_lag3")

log("")
log("=" * 80)
log("DIAGNOSTIC 5: STRUCTURE LOSS COUNT (continuous outcome)")
log("=" * 80)
df["log_struct_destroyed"] = np.log1p(df["n_struct_destroyed"])
df["log_burn_area"] = np.log1p(df["total_burn_area"])
df_idx2 = df.set_index(["FORESTORGC", "year"])
fit("log_struct_destroyed ~ FRT_it + EntityEffects + TimeEffects", df_idx2,
    "STRUCT_FRT_it", "FRT_it")
fit("log_struct_destroyed ~ log_cum_lag3 + EntityEffects + TimeEffects", df_idx2,
    "STRUCT_log_cum_lag3", "log_cum_lag3")
fit("log_burn_area ~ FRT_it + EntityEffects + TimeEffects", df_idx2,
    "BURN_FRT_it", "FRT_it")
fit("log_burn_area ~ log_cum_lag3 + EntityEffects + TimeEffects", df_idx2,
    "BURN_log_cum_lag3", "log_cum_lag3")

log("")
log("=" * 80)
log("DIAGNOSTIC 6: CONTEXT - what fraction of forest is being treated?")
log("=" * 80)
treated_yrs = df[df["FRT_it"] == 1]
log(f"Cumulative treated share quantiles (by year, treated obs):")
log(str(treated_yrs["cum_treated_share"].describe()))
log("")
log("Hypothesis: 1% treatment of forest area is too small to move the disaster needle.")
log("The paper's WUI polygon design (10km buffer) requires polygon-level data we don't have.")

# Save
OUT.write_text("\n".join(lines))
print(f"\nWrote: {OUT}")
