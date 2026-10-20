"""
Main regressions for the JEEM paper using manual within-transformation
with statsmodels and clustered standard errors.

Outcome: forest-year wildfire disaster indicator.
Treatments: extensive (FRT_active) and intensive (log cumulative acres).
Identification: TWFE (forest + year).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy import stats

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

panel = pd.read_csv(PROC / "panel_forest_year.csv", low_memory=False)
print(f"Panel: {panel.shape}")

# ----- Build variables -----
panel["FRT_active"] = ((panel["first_treat_year"].notna()) &
                       (panel["year"] >= panel["first_treat_year"])).astype(int)
panel["log_acres"]  = np.log1p(panel["cum_acres"])
panel["log_mech"]   = np.log1p(panel["cum_acres_mech"])
panel["log_fire"]   = np.log1p(panel["cum_acres_fire"])
panel["log_wui"]    = np.log1p(panel["cum_acres_wui"])
panel = panel.sort_values(["FORESTORGC", "year"])
panel["disaster_lag1"] = panel.groupby("FORESTORGC")["disaster"].shift(1)
panel["log_struct_dest"] = np.log1p(panel["n_struct_destroyed"])

print(f"FRT_active mean: {panel['FRT_active'].mean():.4f}")
print(f"Disaster mean:   {panel['disaster'].mean():.4f}")
print(f"Forests treated: {panel.groupby('FORESTORGC')['FRT_active'].max().sum()} / {panel['FORESTORGC'].nunique()}")


def twfe_clustered(df, y_col, x_cols, cluster_col="FORESTORGC"):
    """
    Two-way fixed effects via dummy variables, with cluster-robust SEs.
    """
    sub = df[[y_col, cluster_col, "year"] + x_cols].dropna().copy()
    if len(sub) == 0:
        return None
    Y = sub[y_col].values.astype(float)

    # Build dummy matrix for entity and time fixed effects
    # Drop one entity dummy and one time dummy to avoid collinearity
    entity_dummies = pd.get_dummies(sub[cluster_col], prefix="ent", drop_first=True).astype(float)
    time_dummies   = pd.get_dummies(sub["year"], prefix="yr", drop_first=True).astype(float)
    X_main = sub[x_cols].astype(float).reset_index(drop=True)

    X = pd.concat([X_main, entity_dummies.reset_index(drop=True), time_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    X_arr = X.values.astype(float)

    # Fit OLS
    model = sm.OLS(Y, X_arr)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": sub[cluster_col].values})

    out = {
        "n_obs": int(res.nobs),
        "y_mean": float(Y.mean()),
        "r2": float(res.rsquared),
    }
    for x in x_cols:
        idx = list(X.columns).index(x)
        out[f"{x}_coef"]   = float(res.params[idx])
        out[f"{x}_se"]     = float(res.bse[idx])
        out[f"{x}_t"]      = float(res.tvalues[idx])
        out[f"{x}_p"]      = float(res.pvalues[idx])
        ci = res.conf_int()[idx]
        out[f"{x}_ci_low"] = float(ci[0])
        out[f"{x}_ci_high"]= float(ci[1])
    return out, res, list(X.columns)


results = {}

print("\n" + "=" * 70)
print("MAIN RESULTS")
print("=" * 70)

# 1. Extensive margin
print("\n[1] Extensive margin: disaster ~ FRT_active + forest FE + year FE")
r, _, _ = twfe_clustered(panel, "disaster", ["FRT_active"])
print(f"   coef = {r['FRT_active_coef']:+.5f}  SE = {r['FRT_active_se']:.5f}  p = {r['FRT_active_p']:.4f}  N = {r['n_obs']}")
print(f"   Mean of dep var = {r['y_mean']:.4f}  Implied effect = {100*r['FRT_active_coef']/r['y_mean']:+.1f}%")
results["extensive_twfe"] = r

# 2. Intensive margin: log cumulative acres
print("\n[2] Intensive margin: disaster ~ log(1 + cum_acres) + forest FE + year FE")
r, _, _ = twfe_clustered(panel, "disaster", ["log_acres"])
print(f"   coef = {r['log_acres_coef']:+.5f}  SE = {r['log_acres_se']:.5f}  p = {r['log_acres_p']:.4f}  N = {r['n_obs']}")
print(f"   Mean of dep var = {r['y_mean']:.4f}")
results["intensive_twfe"] = r

# 3. Joint specification: mech + fire
print("\n[3] Joint by treatment type: log(mech), log(fire)")
r, _, _ = twfe_clustered(panel, "disaster", ["log_mech", "log_fire"])
print(f"   log_mech: coef = {r['log_mech_coef']:+.5f}  SE = {r['log_mech_se']:.5f}  p = {r['log_mech_p']:.4f}")
print(f"   log_fire: coef = {r['log_fire_coef']:+.5f}  SE = {r['log_fire_se']:.5f}  p = {r['log_fire_p']:.4f}")
results["intensive_joint"] = r

# 4. WUI-targeted treatments only
print("\n[4] WUI-targeted treatment intensity: log(cum_acres_wui)")
r, _, _ = twfe_clustered(panel, "disaster", ["log_wui"])
print(f"   coef = {r['log_wui_coef']:+.5f}  SE = {r['log_wui_se']:.5f}  p = {r['log_wui_p']:.4f}")
results["intensive_wui"] = r

# 5. Continuous outcome
print("\n[5] Continuous outcome: log(struct_destroyed + 1) ~ log_acres")
r, _, _ = twfe_clustered(panel, "log_struct_dest", ["log_acres"])
print(f"   coef = {r['log_acres_coef']:+.5f}  SE = {r['log_acres_se']:.5f}  p = {r['log_acres_p']:.4f}")
results["continuous_struct"] = r

# 6. With lagged disaster control
print("\n[6] With lagged disaster control")
r, _, _ = twfe_clustered(panel, "disaster", ["log_acres", "disaster_lag1"])
print(f"   log_acres:    coef = {r['log_acres_coef']:+.5f}  SE = {r['log_acres_se']:.5f}  p = {r['log_acres_p']:.4f}")
print(f"   disaster_lag: coef = {r['disaster_lag1_coef']:+.5f}  SE = {r['disaster_lag1_se']:.5f}  p = {r['disaster_lag1_p']:.4f}")
results["intensive_with_lag"] = r

# 7. Event study (TWFE)
print("\n[7] Event study")
# Bin event time at +/- 5..10
panel["et"] = panel["event_time"].copy()
panel.loc[panel["first_treat_year"].isna(), "et"] = np.nan
# Clip to a reasonable window
panel.loc[panel["et"] < -5, "et"] = -5  # bin negative tail (we will treat -5 as "5 or more years before")
panel.loc[panel["et"] > 10, "et"] = 10  # bin positive tail (10 or more)

# Build event-time dummies for k in -5..10, omitting -1
ev_levels = [k for k in range(-5, 11) if k != -1]
for k in ev_levels:
    name = f"ev_{'m' if k<0 else 'p'}{abs(k)}"
    panel[name] = (panel["et"] == k).astype(int)
ev_cols = [f"ev_{'m' if k<0 else 'p'}{abs(k)}" for k in ev_levels]

r_ev, res_ev, X_cols_ev = twfe_clustered(panel, "disaster", ev_cols)

event_study = []
for k, c in zip(ev_levels, ev_cols):
    event_study.append({
        "k": k,
        "coef":   r_ev[f"{c}_coef"],
        "se":     r_ev[f"{c}_se"],
        "p":      r_ev[f"{c}_p"],
        "ci_low": r_ev[f"{c}_ci_low"],
        "ci_high":r_ev[f"{c}_ci_high"],
    })
event_study = sorted(event_study, key=lambda d: d["k"])
print("    k    coef        SE       p")
for e in event_study:
    print(f"    {e['k']:+3d}  {e['coef']:+.5f}  {e['se']:.5f}  {e['p']:.4f}")
results["event_study"] = event_study

# Pre-trends joint test
pre_cols = [c for c in ev_cols if c.startswith("ev_m")]
if pre_cols:
    full_X_idx = {c: X_cols_ev.index(c) for c in pre_cols}
    R = np.zeros((len(pre_cols), len(X_cols_ev)))
    for i, c in enumerate(pre_cols):
        R[i, full_X_idx[c]] = 1
    b = np.asarray(res_ev.params)
    V = np.asarray(res_ev.cov_params())
    Rb = R @ b
    var_Rb = R @ V @ R.T
    try:
        chi2 = float(Rb @ np.linalg.solve(var_Rb, Rb))
        F = chi2 / len(pre_cols)
        df1 = len(pre_cols)
        df2 = int(res_ev.nobs - res_ev.df_model - 1)
        p_F = 1 - stats.f.cdf(F, df1, df2)
        print(f"\n    Pre-trends joint F-test: F({df1},{df2}) = {F:.3f}, p = {p_F:.4f}")
        results["pre_trends_F"] = F
        results["pre_trends_p"] = float(p_F)
    except Exception as e:
        print(f"   Pre-trends test failed: {e}")

# 8. Heterogeneity by forest size
print("\n[8] Heterogeneity by forest size (terciles)")
panel["size_t"] = pd.qcut(panel["forest_acres"], q=3, labels=["small", "med", "large"])
panel["int_small"] = (panel["size_t"] == "small").astype(int) * panel["log_acres"]
panel["int_med"]   = (panel["size_t"] == "med").astype(int)   * panel["log_acres"]
panel["int_large"] = (panel["size_t"] == "large").astype(int) * panel["log_acres"]
r, _, _ = twfe_clustered(panel, "disaster", ["int_small", "int_med", "int_large"])
print(f"   small:  {r['int_small_coef']:+.5f}  ({r['int_small_se']:.5f})  p={r['int_small_p']:.4f}")
print(f"   medium: {r['int_med_coef']:+.5f}  ({r['int_med_se']:.5f})  p={r['int_med_p']:.4f}")
print(f"   large:  {r['int_large_coef']:+.5f}  ({r['int_large_se']:.5f})  p={r['int_large_p']:.4f}")
results["heterogeneity_size"] = r

# 9. Heterogeneity by region
print("\n[9] Heterogeneity by Region (West vs East/Other)")
panel["west"] = panel["REGION"].isin(["01", "02", "03", "04", "05", "06"]).astype(int)
panel["int_west"] = panel["west"] * panel["log_acres"]
panel["int_east"] = (1 - panel["west"]) * panel["log_acres"]
r, _, _ = twfe_clustered(panel, "disaster", ["int_west", "int_east"])
print(f"   west:   {r['int_west_coef']:+.5f}  ({r['int_west_se']:.5f})  p={r['int_west_p']:.4f}")
print(f"   east:   {r['int_east_coef']:+.5f}  ({r['int_east_se']:.5f})  p={r['int_east_p']:.4f}")
results["heterogeneity_region"] = r

# 10. Western forests only
print("\n[10] Sample restriction: Western forests only")
west_panel = panel[panel["west"] == 1].copy()
r, _, _ = twfe_clustered(west_panel, "disaster", ["log_acres"])
print(f"   coef = {r['log_acres_coef']:+.5f}  SE = {r['log_acres_se']:.5f}  p = {r['log_acres_p']:.4f}  N = {r['n_obs']}")
results["west_only"] = r

# 11. Robustness: alternative outcome variants
print("\n[11] Robustness: any FEMA declaration outcome")
panel["fema_dec"] = (panel["any_fema_dec"] == True).astype(int)
r, _, _ = twfe_clustered(panel, "fema_dec", ["log_acres"])
print(f"   coef = {r['log_acres_coef']:+.5f}  SE = {r['log_acres_se']:.5f}  p = {r['log_acres_p']:.4f}")
print(f"   Mean of fema_dec: {r['y_mean']:.4f}")
results["fema_outcome"] = r

# Summary statistics for the paper
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
sumstats = {
    "n_forests": int(panel["FORESTORGC"].nunique()),
    "n_years":  int(panel["year"].nunique()),
    "n_obs":    int(len(panel)),
    "n_treated_forests": int(panel.groupby("FORESTORGC")["FRT_active"].max().sum()),
    "n_disaster_forests": int(panel.groupby("FORESTORGC")["disaster"].max().sum()),
    "disaster_mean": float(panel["disaster"].mean()),
    "disaster_total": int(panel["disaster"].sum()),
    "n_disasters": int(panel["n_disasters"].sum()),
    "n_struct_destroyed": int(panel["n_struct_destroyed"].sum()),
    "n_fatalities": float(panel["n_fatalities"].sum()),
    "total_acres_treated_M": float(panel["acres_treated"].sum() / 1e6),
    "mean_acres_treated_per_forest_year": float(panel["acres_treated"].mean()),
    "median_cum_acres_at_2024": float(panel[panel["year"]==2024]["cum_acres"].median()),
    "mean_cum_acres_at_2024_M": float(panel[panel["year"]==2024]["cum_acres"].mean() / 1e6),
}
for k, v in sumstats.items():
    print(f"   {k}: {v}")
results["summary_stats"] = sumstats

# Save
with open(OUT / "regression_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults written to: {OUT / 'regression_results.json'}")
