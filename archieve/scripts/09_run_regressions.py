"""
Main regression analysis for the JEEM paper.
Reads data/processed/panel_forest_year.csv and runs:
  1) Summary statistics
  2) TWFE extensive margin (3 specifications)
  3) TWFE intensive margin (log acres + IHS + by treatment type)
  4) Event study
  5) Heterogeneity by forest size and region
  6) Robustness: alternative outcome definitions
  7) Cost-benefit analysis
Outputs all results to data/processed/regression_results.txt and saves coefficients to JSON
for downstream paper updating.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PANEL = ROOT / "data" / "processed" / "panel_forest_year.csv"
OUT_DIR = ROOT / "data" / "processed"
RESULTS_TXT = OUT_DIR / "regression_results.txt"
RESULTS_JSON = OUT_DIR / "regression_coefs.json"

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df = pd.read_csv(PANEL)
print(f"Loaded panel: {df.shape}")

# Numeric cleaning
df["FRT_it"] = (df["cum_acres"] > 0).astype(int)
df["log_cum_acres"] = np.log1p(df["cum_acres"])
df["ihs_cum_acres"] = np.arcsinh(df["cum_acres"])
df["log_cum_acres_mech"] = np.log1p(df["cum_acres_mech"])
df["log_cum_acres_fire"] = np.log1p(df["cum_acres_fire"])
df["disaster_struct"] = (df["n_struct_destroyed"] > 0).astype(int)
df["disaster_fatal"] = (df["n_fatalities"] > 0).astype(int)
df["log_forest_acres"] = np.log(df["forest_acres"].replace(0, np.nan)).fillna(0)

# Set panel index for linearmodels
df = df.set_index(["FORESTORGC", "year"]).sort_index()

print(f"Forests: {df.index.get_level_values(0).nunique()}")
print(f"Years: {df.index.get_level_values(1).min()} to {df.index.get_level_values(1).max()}")
print(f"Disaster rate (any): {df['disaster'].mean():.4f}")
print(f"Disaster rate (struct): {df['disaster_struct'].mean():.4f}")
print(f"Treated polygon-years: {df['FRT_it'].sum()} / {len(df)}")

# -------------------------------------------------------------------
# Helper to format results
# -------------------------------------------------------------------
results_lines = []
coefs = {}


def log(s=""):
    print(s)
    results_lines.append(s)


def format_panel_result(name, res, key_var):
    coef = res.params[key_var]
    se = res.std_errors[key_var]
    t = coef / se
    pval = res.pvalues[key_var]
    nobs = int(res.nobs)
    r2 = float(res.rsquared)
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
    log(f"{name:40s}  beta={coef:+.6f} se={se:.6f} t={t:+.2f} p={pval:.4f}{sig:3s} N={nobs}  R2={r2:.4f}")
    coefs[name] = {"coef": float(coef), "se": float(se), "t": float(t), "pval": float(pval),
                   "nobs": nobs, "r2": r2, "sig": sig.strip()}


# -------------------------------------------------------------------
# 1. Summary statistics
# -------------------------------------------------------------------
log("=" * 80)
log("1. SUMMARY STATISTICS")
log("=" * 80)
df_flat = df.reset_index()
ever_treated = df_flat.groupby("FORESTORGC")["FRT_it"].max() == 1
log(f"Total forests: {len(ever_treated)}")
log(f"Ever-treated forests: {int(ever_treated.sum())}")
log(f"Never-treated forests: {int((~ever_treated).sum())}")
log(f"Polygon-year observations: {len(df_flat)}")
log("")

stats_cols = ["disaster", "FRT_it", "cum_acres", "n_disasters",
              "n_struct_destroyed", "n_fatalities", "forest_acres"]
log("                     mean        sd       min       max")
for c in stats_cols:
    s = df_flat[c]
    log(f"{c:20s}  {s.mean():10.3f}  {s.std():10.3f}  {s.min():10.0f}  {s.max():12.0f}")

# Treated vs untreated comparison
log("")
log("Disaster rate by treatment status:")
for status, sub in df_flat.groupby(df_flat["cum_acres"] > 0):
    label = "Treated" if status else "Untreated"
    log(f"  {label:10s}: P(disaster) = {sub['disaster'].mean():.4f}  N = {len(sub)}")

# -------------------------------------------------------------------
# 2. EXTENSIVE MARGIN: TWFE
# -------------------------------------------------------------------
log("\n" + "=" * 80)
log("2. EXTENSIVE MARGIN: TWO-WAY FIXED EFFECTS")
log("=" * 80)

# Spec 1: forest FE only
mod1 = PanelOLS.from_formula(
    "disaster ~ FRT_it + EntityEffects",
    data=df,
)
res1 = mod1.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("ext_m1_FE_only", res1, "FRT_it")

# Spec 2: forest + year FE
mod2 = PanelOLS.from_formula(
    "disaster ~ FRT_it + EntityEffects + TimeEffects",
    data=df,
)
res2 = mod2.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("ext_m2_TWFE", res2, "FRT_it")

# Spec 3: TWFE + log forest size as time-varying control (drop absorbed)
mod3 = PanelOLS.from_formula(
    "disaster ~ FRT_it + log_forest_acres + EntityEffects + TimeEffects",
    data=df,
    drop_absorbed=True,
)
res3 = mod3.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("ext_m3_TWFE_ctrl", res3, "FRT_it")

# Spec 5: lagged treatment (protective channel: treatment in t-1 to t-3 reduces disaster in t)
df_lag = df.reset_index().sort_values(["FORESTORGC", "year"]).copy()
df_lag["FRT_lag1"] = df_lag.groupby("FORESTORGC")["FRT_it"].shift(1).fillna(0).astype(int)
df_lag["FRT_lag3"] = (
    (df_lag.groupby("FORESTORGC")["FRT_it"].shift(1).fillna(0)
     + df_lag.groupby("FORESTORGC")["FRT_it"].shift(2).fillna(0)
     + df_lag.groupby("FORESTORGC")["FRT_it"].shift(3).fillna(0)) > 0
).astype(int)
df_lag["log_cum_acres_lag1"] = np.log1p(
    df_lag.groupby("FORESTORGC")["cum_acres"].shift(1).fillna(0)
)
df_lag = df_lag.set_index(["FORESTORGC", "year"])

mod_lag1 = PanelOLS.from_formula(
    "disaster ~ FRT_lag1 + EntityEffects + TimeEffects",
    data=df_lag,
)
res_lag1 = mod_lag1.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("ext_m5_FRT_lag1", res_lag1, "FRT_lag1")

mod_lag3 = PanelOLS.from_formula(
    "disaster ~ FRT_lag3 + EntityEffects + TimeEffects",
    data=df_lag,
)
res_lag3 = mod_lag3.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("ext_m6_FRT_lag1to3", res_lag3, "FRT_lag3")

mod_loglag = PanelOLS.from_formula(
    "disaster ~ log_cum_acres_lag1 + EntityEffects + TimeEffects",
    data=df_lag,
)
res_loglag = mod_loglag.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("ext_m7_log_cum_lag1", res_loglag, "log_cum_acres_lag1")

# Spec 4: TWFE with region-by-year FE (state-by-year analog)
df_ry = df.reset_index().copy()
df_ry["region_year"] = df_ry["REGION"].astype(str) + "_" + df_ry["year"].astype(str)
df_ry = df_ry.set_index(["FORESTORGC", "year"])
mod4 = PanelOLS.from_formula(
    "disaster ~ FRT_it + C(region_year) + EntityEffects",
    data=df_ry,
    drop_absorbed=True,
)
res4 = mod4.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("ext_m4_RegionYearFE", res4, "FRT_it")

# Mean of disaster
mean_y = df["disaster"].mean()
log(f"Mean disaster rate: {mean_y:.4f}")
coefs["mean_disaster"] = float(mean_y)
implied_pct = res2.params["FRT_it"] / mean_y * 100
log(f"Implied % effect (TWFE): {implied_pct:.1f}%")
coefs["implied_pct_extensive"] = float(implied_pct)

# -------------------------------------------------------------------
# 3. INTENSIVE MARGIN
# -------------------------------------------------------------------
log("\n" + "=" * 80)
log("3. INTENSIVE MARGIN")
log("=" * 80)

# 3a. log(cum_acres + 1)
mod_log = PanelOLS.from_formula(
    "disaster ~ log_cum_acres + EntityEffects + TimeEffects",
    data=df,
)
res_log = mod_log.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("int_log_acres", res_log, "log_cum_acres")

# 3b. IHS(cum_acres)
mod_ihs = PanelOLS.from_formula(
    "disaster ~ ihs_cum_acres + EntityEffects + TimeEffects",
    data=df,
)
res_ihs = mod_ihs.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("int_ihs_acres", res_ihs, "ihs_cum_acres")

# 3c. By treatment type
mod_type = PanelOLS.from_formula(
    "disaster ~ log_cum_acres_mech + log_cum_acres_fire + EntityEffects + TimeEffects",
    data=df,
)
res_type = mod_type.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("int_log_acres_mech", res_type, "log_cum_acres_mech")
format_panel_result("int_log_acres_fire", res_type, "log_cum_acres_fire")

# -------------------------------------------------------------------
# 4. EVENT STUDY
# -------------------------------------------------------------------
log("\n" + "=" * 80)
log("4. EVENT STUDY (binned event time)")
log("=" * 80)

df_es = df.reset_index().copy()
# Bin event time: pre <=-5, -4..-2, -1 (omitted), 0, 1..3, 4..6, 7+
def bin_et(et):
    if pd.isna(et):
        return "never"
    et = int(et)
    if et <= -5: return "pre5plus"
    if et <= -2: return "preMid"
    if et == -1: return "OMIT"
    if et == 0: return "post0"
    if et <= 3: return "post1_3"
    if et <= 6: return "post4_6"
    return "post7plus"

df_es["et_bin"] = df_es["event_time"].apply(bin_et)
# Build dummies
for b in ["pre5plus", "preMid", "post0", "post1_3", "post4_6", "post7plus"]:
    df_es[f"D_{b}"] = (df_es["et_bin"] == b).astype(int)

df_es = df_es.set_index(["FORESTORGC", "year"])
es_terms = ["D_pre5plus", "D_preMid", "D_post0", "D_post1_3", "D_post4_6", "D_post7plus"]
formula = "disaster ~ " + " + ".join(es_terms) + " + EntityEffects + TimeEffects"
mod_es = PanelOLS.from_formula(formula, data=df_es)
res_es = mod_es.fit(cov_type="clustered", cluster_entity=True)

es_results = {}
for term in es_terms:
    if term in res_es.params.index:
        c = res_es.params[term]
        s = res_es.std_errors[term]
        p = res_es.pvalues[term]
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        log(f"  {term:15s}  {c:+.6f}  ({s:.6f})  p={p:.4f}{sig}")
        es_results[term] = {"coef": float(c), "se": float(s), "pval": float(p), "sig": sig}
coefs["event_study"] = es_results

# Joint test of pre-trends
pre_terms = [t for t in ["D_pre5plus", "D_preMid"] if t in res_es.params.index]
if len(pre_terms) >= 2:
    try:
        joint_test = res_es.wald_test(formula=" = ".join(pre_terms) + " = 0")
        log(f"  Joint pre-trend F-test p-value: {float(joint_test.pval):.4f}")
        coefs["pretrend_pval"] = float(joint_test.pval)
    except Exception as e:
        log(f"  Joint pre-trend test failed: {e}")

# -------------------------------------------------------------------
# 5. HETEROGENEITY
# -------------------------------------------------------------------
log("\n" + "=" * 80)
log("5. HETEROGENEITY")
log("=" * 80)

# 5a. By forest size tercile
df_het = df.reset_index().copy()
forest_size = df_het.groupby("FORESTORGC")["forest_acres"].first()
size_tercile = pd.qcut(forest_size, 3, labels=["small", "medium", "large"])
df_het["size_tercile"] = df_het["FORESTORGC"].map(size_tercile)
for s in ["small", "medium", "large"]:
    df_het[f"FRT_{s}"] = df_het["FRT_it"] * (df_het["size_tercile"] == s).astype(int)
df_het_idx = df_het.set_index(["FORESTORGC", "year"])

mod_het = PanelOLS.from_formula(
    "disaster ~ FRT_small + FRT_medium + FRT_large + EntityEffects + TimeEffects",
    data=df_het_idx,
)
res_het = mod_het.fit(cov_type="clustered", cluster_entity=True)
for k in ["FRT_small", "FRT_medium", "FRT_large"]:
    format_panel_result(f"het_size_{k}", res_het, k)

# 5b. By region (West vs other; West = USFS Regions 1-6)
df_het["west"] = df_het["REGION"].between(1, 6).astype(int)
df_het["FRT_west"] = df_het["FRT_it"] * df_het["west"]
df_het["FRT_other"] = df_het["FRT_it"] * (1 - df_het["west"])
df_het_idx2 = df_het.set_index(["FORESTORGC", "year"])
mod_west = PanelOLS.from_formula(
    "disaster ~ FRT_west + FRT_other + EntityEffects + TimeEffects",
    data=df_het_idx2,
)
res_west = mod_west.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("het_west_FRT_west", res_west, "FRT_west")
format_panel_result("het_west_FRT_other", res_west, "FRT_other")

# -------------------------------------------------------------------
# 6. ROBUSTNESS: alternative outcomes
# -------------------------------------------------------------------
log("\n" + "=" * 80)
log("6. ROBUSTNESS: ALTERNATIVE OUTCOME DEFINITIONS")
log("=" * 80)

for outcome in ["disaster_struct", "disaster_fatal", "any_fema_dec"]:
    df_alt = df.copy()
    df_alt[outcome] = df_alt[outcome].astype(int)
    mod_alt = PanelOLS.from_formula(
        f"{outcome} ~ FRT_it + EntityEffects + TimeEffects",
        data=df_alt,
    )
    res_alt = mod_alt.fit(cov_type="clustered", cluster_entity=True)
    format_panel_result(f"robust_{outcome}", res_alt, "FRT_it")

# Robustness: log treatments per polygon-year (count)
df["log_n_treat"] = np.log1p(df["n_treatments"])
mod_n = PanelOLS.from_formula(
    "disaster ~ log_n_treat + EntityEffects + TimeEffects",
    data=df,
)
res_n = mod_n.fit(cov_type="clustered", cluster_entity=True)
format_panel_result("robust_log_n_treatments", res_n, "log_n_treat")

# -------------------------------------------------------------------
# 7. PLACEBO TEST: future treatment
# -------------------------------------------------------------------
log("\n" + "=" * 80)
log("7. PLACEBO TEST: future treatment (5-year lead)")
log("=" * 80)

df_pl = df.reset_index().sort_values(["FORESTORGC", "year"]).copy()
df_pl["FRT_lead5"] = df_pl.groupby("FORESTORGC")["FRT_it"].shift(-5).fillna(0).astype(int)
# Restrict to pre-period only
df_pre = df_pl[df_pl["FRT_it"] == 0].set_index(["FORESTORGC", "year"])
try:
    mod_pl = PanelOLS.from_formula(
        "disaster ~ FRT_lead5 + EntityEffects + TimeEffects",
        data=df_pre,
    )
    res_pl = mod_pl.fit(cov_type="clustered", cluster_entity=True)
    format_panel_result("placebo_lead5", res_pl, "FRT_lead5")
except Exception as e:
    log(f"  Placebo failed: {e}")

# -------------------------------------------------------------------
# 8. COST-BENEFIT
# -------------------------------------------------------------------
log("\n" + "=" * 80)
log("8. COST-BENEFIT ANALYSIS")
log("=" * 80)

# Per-disaster average components from the panel
disasters = df.reset_index()
disasters_with = disasters[disasters["disaster"] == 1]
log(f"Disaster events (forest-years): {len(disasters_with)}")

avg_struct = disasters_with["n_struct_destroyed"].mean()
avg_fatal = disasters_with["n_fatalities"].mean()
log(f"Avg structures destroyed per disaster: {avg_struct:.1f}")
log(f"Avg fatalities per disaster: {avg_fatal:.2f}")

# Use FRT extensive margin coefficient
beta1 = abs(res2.params["FRT_it"])
log(f"|Beta1| (extensive margin TWFE): {beta1:.6f}")

# CBA parameters
VSL = 11.6e6  # USD
struct_value = 350e3  # USD per structure (median US WUI)
suppression_per_disaster = 5e6  # USD avg federal suppression cost per disaster (conservative)

# Avg loss per disaster
loss_per_disaster = (VSL * avg_fatal +
                      struct_value * avg_struct +
                      suppression_per_disaster)
log(f"Avg total loss per disaster: ${loss_per_disaster/1e6:.1f}M")
log(f"  Fatality:    ${VSL * avg_fatal / 1e6:.1f}M")
log(f"  Structures:  ${struct_value * avg_struct / 1e6:.1f}M")
log(f"  Suppression: ${suppression_per_disaster / 1e6:.1f}M")

# Annual avoided damage per treated forest
annual_benefit_per_forest = beta1 * loss_per_disaster
log(f"Avoided damage per treated forest-year: ${annual_benefit_per_forest:,.0f}")

# Treatment cost: avg per acre from FACTS, times avg acres treated per treated forest-year
treated_year_data = df.reset_index()
treated_year_data = treated_year_data[treated_year_data["acres_treated"] > 0]
avg_acres_per_year = treated_year_data["acres_treated"].mean()
avg_cost_per_acre = treated_year_data["avg_cost_per_acre"].mean()
log(f"Avg acres treated per treated forest-year: {avg_acres_per_year:,.0f}")
log(f"Avg cost per acre: ${avg_cost_per_acre:,.0f}")
annual_cost_per_forest = avg_acres_per_year * avg_cost_per_acre
log(f"Annual treatment cost per treated forest: ${annual_cost_per_forest:,.0f}")

bcr = annual_benefit_per_forest / annual_cost_per_forest
log(f"\nBenefit-Cost Ratio: {bcr:.2f}")

coefs["cba"] = {
    "beta1_abs": float(beta1),
    "avg_loss_per_disaster": float(loss_per_disaster),
    "avg_fatalities": float(avg_fatal),
    "avg_structures": float(avg_struct),
    "VSL": VSL,
    "struct_value": struct_value,
    "suppression_per_disaster": suppression_per_disaster,
    "annual_benefit_per_forest": float(annual_benefit_per_forest),
    "annual_cost_per_forest": float(annual_cost_per_forest),
    "avg_acres_per_year": float(avg_acres_per_year),
    "avg_cost_per_acre": float(avg_cost_per_acre),
    "BCR": float(bcr),
}

# Sensitivity: BCR across VSL and discount rates
log("\nBCR sensitivity:")
for VSL_alt in [3e6, 7e6, 11.6e6, 15e6]:
    loss_alt = (VSL_alt * avg_fatal + struct_value * avg_struct + suppression_per_disaster)
    bcr_alt = beta1 * loss_alt / annual_cost_per_forest
    log(f"  VSL=${VSL_alt/1e6:.1f}M -> BCR={bcr_alt:.2f}")
    coefs[f"bcr_VSL_{int(VSL_alt/1e6)}"] = float(bcr_alt)

# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------
RESULTS_TXT.write_text("\n".join(results_lines))
print(f"\nWrote results to: {RESULTS_TXT}")

with open(RESULTS_JSON, "w") as f:
    json.dump(coefs, f, indent=2, default=str)
print(f"Wrote coefficients to: {RESULTS_JSON}")

log("\nDONE.")
