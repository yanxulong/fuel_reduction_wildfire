"""
MAIN ANALYSIS for the JEEM paper.

Improvements over the naive TWFE:
  1. Use LAGGED treatment (treatment in year t-1, t-2 to predict disaster in year t)
     -- treatments cannot affect contemporaneous fires through fuel load
  2. Treatment intensity as share of forest area treated (not log)
  3. Match the unit of analysis to the question: forest-year, with the
     stock of cumulative protection as the key regressor
  4. Multiple outcomes: binary disaster, log structures destroyed, log burn area
  5. Fixed effects: forest, year, and region-by-year (to absorb regional weather shocks)
  6. Cluster-robust SEs at forest level
  7. Restrict to forests with non-trivial activity (drop tiny / no-activity forests)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy import stats

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PROC = ROOT / "data" / "processed"
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

panel = pd.read_csv(PROC / "panel_forest_year.csv", low_memory=False)
print(f"Panel: {panel.shape}")

# ---- Variable construction ----
panel = panel.sort_values(["FORESTORGC", "year"]).reset_index(drop=True)

# Lagged stock variables
for k in (1, 2, 3):
    panel[f"cum_acres_lag{k}"] = panel.groupby("FORESTORGC")["cum_acres"].shift(k)
    panel[f"cum_acres_wui_lag{k}"] = panel.groupby("FORESTORGC")["cum_acres_wui"].shift(k)

# Intensity = cumulative acres / forest size  (share of forest treated as stock)
panel["intensity"]      = panel["cum_acres"]      / panel["forest_acres"]
panel["intensity_lag1"] = panel["cum_acres_lag1"] / panel["forest_acres"]
panel["intensity_lag2"] = panel["cum_acres_lag2"] / panel["forest_acres"]
panel["intensity_lag3"] = panel["cum_acres_lag3"] / panel["forest_acres"]

# Log forms (with the standard log(1+x) transformation)
panel["log_intensity_lag1"] = np.log1p(panel["intensity_lag1"])
panel["log_intensity_lag2"] = np.log1p(panel["intensity_lag2"])
panel["log_acres_lag1"]     = np.log1p(panel["cum_acres_lag1"])
panel["log_acres_lag2"]     = np.log1p(panel["cum_acres_lag2"])
panel["log_acres_wui_lag1"] = np.log1p(panel["cum_acres_wui_lag1"])

# Continuous outcomes
panel["log_struct_dest"] = np.log1p(panel["n_struct_destroyed"])
panel["log_burn_area"]   = np.log1p(panel["total_burn_area"])

# FEMA outcome
panel["fema_dec"] = (panel["any_fema_dec"] == True).astype(int)

# Region groups
WEST = ["01", "02", "03", "04", "05", "06"]
panel["west"] = panel["REGION"].isin(WEST).astype(int)

# Restrict to forests with at least 100 acres of cumulative treatment by 2024
forest_total = panel.groupby("FORESTORGC")["cum_acres"].max()
keep = forest_total[forest_total >= 100].index
panel_use = panel[panel["FORESTORGC"].isin(keep)].copy()
print(f"Forests after activity filter: {panel_use['FORESTORGC'].nunique()}")
print(f"Forest-year obs: {len(panel_use)}")


def twfe(df, y, xs, fe_state_year=False, cluster_col="FORESTORGC"):
    """
    Run TWFE with optional Region-by-Year fixed effects, clustered SEs.
    Returns dict of results.
    """
    cols = [y, cluster_col, "year", "REGION"] + xs
    sub = df[cols].dropna().copy()
    if len(sub) < 50:
        return None

    Y = sub[y].astype(float).values
    X_main = sub[xs].astype(float).reset_index(drop=True)

    ent_d = pd.get_dummies(sub[cluster_col], prefix="ent", drop_first=True).astype(float).reset_index(drop=True)
    if fe_state_year:
        sub["region_year"] = sub["REGION"].astype(str) + "_" + sub["year"].astype(str)
        ry_d = pd.get_dummies(sub["region_year"], prefix="ry", drop_first=True).astype(float).reset_index(drop=True)
        time_d = ry_d
    else:
        time_d = pd.get_dummies(sub["year"], prefix="yr", drop_first=True).astype(float).reset_index(drop=True)

    X = pd.concat([X_main, ent_d, time_d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    Xa = X.values.astype(float)

    model = sm.OLS(Y, Xa)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": sub[cluster_col].values})

    out = {"n_obs": int(res.nobs), "y_mean": float(Y.mean()), "r2": float(res.rsquared)}
    cols_list = list(X.columns)
    for x in xs:
        i = cols_list.index(x)
        out[f"{x}_coef"]    = float(res.params[i])
        out[f"{x}_se"]      = float(res.bse[i])
        out[f"{x}_t"]       = float(res.tvalues[i])
        out[f"{x}_p"]       = float(res.pvalues[i])
        ci = res.conf_int()[i]
        out[f"{x}_ci_low"]  = float(ci[0])
        out[f"{x}_ci_high"] = float(ci[1])
    return out


results = {}

print("\n" + "=" * 70)
print("MAIN ANALYSIS WITH LAGGED TREATMENT")
print("=" * 70)

# ===== 1. Effect on disaster probability with lagged log acres =====
print("\n[1] disaster ~ log(cum_acres_lag1) + Forest FE + Year FE")
r = twfe(panel_use, "disaster", ["log_acres_lag1"])
print(f"   coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}  N = {r['n_obs']}")
results["disaster_log_lag1"] = r

print("\n[1b] disaster ~ log(cum_acres_lag1) + Forest FE + Region x Year FE")
r = twfe(panel_use, "disaster", ["log_acres_lag1"], fe_state_year=True)
print(f"   coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}")
results["disaster_log_lag1_ry"] = r

# ===== 2. Treatment intensity (share of forest) =====
print("\n[2] disaster ~ intensity_lag1 (share of forest treated)")
r = twfe(panel_use, "disaster", ["intensity_lag1"])
print(f"   coef = {r['intensity_lag1_coef']:+.5f}  SE = {r['intensity_lag1_se']:.5f}  p = {r['intensity_lag1_p']:.4f}")
print(f"   Mean intensity (treated): {panel_use[panel_use['intensity_lag1']>0]['intensity_lag1'].mean():.4f}")
results["disaster_intensity_lag1"] = r

print("\n[2b] disaster ~ log_intensity_lag1")
r = twfe(panel_use, "disaster", ["log_intensity_lag1"])
print(f"   coef = {r['log_intensity_lag1_coef']:+.5f}  SE = {r['log_intensity_lag1_se']:.5f}  p = {r['log_intensity_lag1_p']:.4f}")
results["disaster_log_intensity_lag1"] = r

# ===== 3. Long lags (effects accumulate over time) =====
print("\n[3] disaster ~ log(cum_acres_lag2) (2-year lag)")
r = twfe(panel_use, "disaster", ["log_acres_lag2"])
print(f"   coef = {r['log_acres_lag2_coef']:+.5f}  SE = {r['log_acres_lag2_se']:.5f}  p = {r['log_acres_lag2_p']:.4f}")
results["disaster_log_lag2"] = r

# ===== 4. WUI-targeted treatments specifically =====
print("\n[4] disaster ~ log(cum_acres_wui_lag1)")
r = twfe(panel_use, "disaster", ["log_acres_wui_lag1"])
print(f"   coef = {r['log_acres_wui_lag1_coef']:+.5f}  SE = {r['log_acres_wui_lag1_se']:.5f}  p = {r['log_acres_wui_lag1_p']:.4f}")
results["disaster_wui_lag1"] = r

print("\n[4b] disaster ~ log(cum_acres_wui_lag1) + Region x Year FE")
r = twfe(panel_use, "disaster", ["log_acres_wui_lag1"], fe_state_year=True)
print(f"   coef = {r['log_acres_wui_lag1_coef']:+.5f}  SE = {r['log_acres_wui_lag1_se']:.5f}  p = {r['log_acres_wui_lag1_p']:.4f}")
results["disaster_wui_lag1_ry"] = r

# ===== 5. Continuous outcomes =====
print("\n[5] log(struct_destroyed+1) ~ log(cum_acres_lag1)")
r = twfe(panel_use, "log_struct_dest", ["log_acres_lag1"])
print(f"   coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}")
results["struct_log_lag1"] = r

print("\n[5b] log(struct_destroyed+1) ~ log(cum_acres_lag1) + Region x Year FE")
r = twfe(panel_use, "log_struct_dest", ["log_acres_lag1"], fe_state_year=True)
print(f"   coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}")
results["struct_log_lag1_ry"] = r

print("\n[5c] log(burn_area+1) ~ log(cum_acres_lag1) + Region x Year FE")
r = twfe(panel_use, "log_burn_area", ["log_acres_lag1"], fe_state_year=True)
print(f"   coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}")
results["burn_area_log_lag1_ry"] = r

# ===== 6. FEMA outcome =====
print("\n[6] fema_dec ~ log(cum_acres_lag1)")
r = twfe(panel_use, "fema_dec", ["log_acres_lag1"])
print(f"   coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}")
print(f"   Mean: {r['y_mean']:.4f}")
results["fema_log_lag1"] = r

# ===== 7. Treatment type decomposition with lags =====
print("\n[7] disaster ~ log(cum_mech_lag1) + log(cum_fire_lag1)")
panel_use["log_mech_lag1"] = np.log1p(panel_use.groupby("FORESTORGC")["cum_acres_mech"].shift(1))
panel_use["log_fire_lag1"] = np.log1p(panel_use.groupby("FORESTORGC")["cum_acres_fire"].shift(1))
r = twfe(panel_use, "disaster", ["log_mech_lag1", "log_fire_lag1"])
print(f"   mech: {r['log_mech_lag1_coef']:+.5f}  ({r['log_mech_lag1_se']:.5f})  p={r['log_mech_lag1_p']:.4f}")
print(f"   fire: {r['log_fire_lag1_coef']:+.5f}  ({r['log_fire_lag1_se']:.5f})  p={r['log_fire_lag1_p']:.4f}")
results["disaster_by_type_lag1"] = r

# ===== 8. Restrict to Western forests (highest fire risk) =====
print("\n[8] WESTERN FORESTS ONLY: disaster ~ log(cum_acres_lag1)")
west_sub = panel_use[panel_use["west"]==1]
print(f"   Forests: {west_sub['FORESTORGC'].nunique()}, Obs: {len(west_sub)}")
r = twfe(west_sub, "disaster", ["log_acres_lag1"])
if r is not None:
    print(f"   coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}")
    results["west_disaster_lag1"] = r
else:
    print("   (insufficient observations)")

print("\n[8b] WESTERN: log(struct+1) ~ log(cum_acres_lag1) + Region x Year FE")
r = twfe(panel_use[panel_use["west"]==1], "log_struct_dest", ["log_acres_lag1"], fe_state_year=True)
print(f"   coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}")
results["west_struct_lag1_ry"] = r

print("\n[8c] WESTERN: log(struct+1) ~ log(cum_acres_wui_lag1) + Region x Year FE")
r = twfe(panel_use[panel_use["west"]==1], "log_struct_dest", ["log_acres_wui_lag1"], fe_state_year=True)
print(f"   coef = {r['log_acres_wui_lag1_coef']:+.5f}  SE = {r['log_acres_wui_lag1_se']:.5f}  p = {r['log_acres_wui_lag1_p']:.4f}")
results["west_struct_wui_lag1_ry"] = r

# ===== 9. Heterogeneity by forest size (within western) =====
print("\n[9] Western, by forest size tercile")
west = panel_use[panel_use["west"]==1].copy()
west["sz"] = pd.qcut(west["forest_acres"], q=3, labels=["small","med","large"])
for s in ["small", "med", "large"]:
    sub = west[west["sz"]==s]
    if len(sub) < 30: continue
    r = twfe(sub, "disaster", ["log_acres_lag1"])
    if r:
        print(f"   {s}: coef = {r['log_acres_lag1_coef']:+.5f}  SE = {r['log_acres_lag1_se']:.5f}  p = {r['log_acres_lag1_p']:.4f}  N = {r['n_obs']}")
        results[f"west_size_{s}"] = r

# ===== 10. Distributed lag (k=1..3) =====
print("\n[10] Distributed lag: disaster ~ log_lag1 + log_lag2 + log_lag3")
r = twfe(panel_use, "disaster",
         ["log_acres_lag1", "log_acres_lag2"])
print(f"   lag1: {r['log_acres_lag1_coef']:+.5f}  ({r['log_acres_lag1_se']:.5f})  p={r['log_acres_lag1_p']:.4f}")
print(f"   lag2: {r['log_acres_lag2_coef']:+.5f}  ({r['log_acres_lag2_se']:.5f})  p={r['log_acres_lag2_p']:.4f}")
results["distributed_lag"] = r

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
mean_disaster = panel_use["disaster"].mean()
print(f"Forest-year obs: {len(panel_use):,}")
print(f"Forests: {panel_use['FORESTORGC'].nunique()}")
print(f"Years: {panel_use['year'].min()}-{panel_use['year'].max()}")
print(f"Disaster mean: {mean_disaster:.4f}")
print(f"Mean cum_acres at end (2024): {panel_use[panel_use['year']==2024]['cum_acres'].mean():,.0f}")
print(f"Median cum_acres at end (2024): {panel_use[panel_use['year']==2024]['cum_acres'].median():,.0f}")

# Save
with open(OUT / "main_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {OUT / 'main_analysis_results.json'}")
