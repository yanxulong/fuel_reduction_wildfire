"""
WUI polygon-year regressions, the paper's preferred spatial scale.

Panel: 280k WUI polygons x 25 years = 7.0M polygon-year observations.
Treatment = any FACTS fuel reduction treatment within 10 km of the polygon centroid.
Outcome: binary disaster indicator + continuous structure damage.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from linearmodels.panel import PanelOLS

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PANEL_PATH = ROOT / "data" / "processed" / "panel_wui_year.parquet"
OUT_TXT = ROOT / "data" / "processed" / "wui_regression_results.txt"
OUT_JSON = ROOT / "data" / "processed" / "wui_regression_coefs.json"

print("Loading panel...")
df = pd.read_parquet(PANEL_PATH)
print(f"  shape: {df.shape}")

# Create derived variables
df["log_cum_acres"] = np.log1p(df["cum_acres"])
df["ihs_cum_acres"] = np.arcsinh(df["cum_acres"])
df["log_cum_mech"]  = np.log1p(df["cum_acres_mech"])
df["log_cum_fire"]  = np.log1p(df["cum_acres_fire"])
df["disaster_struct"] = (df["n_struct_destroyed"] > 0).astype(int)
df["disaster_fatal"]  = (df["n_fatalities"] > 0).astype(int)
df["log_struct_destroyed"] = np.log1p(df["n_struct_destroyed"])

# Integer index for efficient panel
# Convert wui_id string to int code
df["wui_code"] = pd.Categorical(df["wui_id"]).codes
print(f"  unique wui_ids: {df['wui_code'].nunique():,}")

# Sort and build lags
df = df.sort_values(["wui_code","year"]).reset_index(drop=True)
df["FRT_lag1"] = df.groupby("wui_code")["FRT_it"].shift(1).fillna(0).astype(int)
df["FRT_lag3"] = (
    (df.groupby("wui_code")["FRT_it"].shift(1).fillna(0) +
     df.groupby("wui_code")["FRT_it"].shift(2).fillna(0) +
     df.groupby("wui_code")["FRT_it"].shift(3).fillna(0)) > 0
).astype(int)
df["log_cum_lag3"] = np.log1p(
    df.groupby("wui_code")["cum_acres"].shift(3).fillna(0)
)

# Set MultiIndex for linearmodels
df_idx = df.set_index(["wui_code","year"])

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
lines = []
coefs = {}
def log(s=""):
    print(s)
    lines.append(s)

def fit(formula, data, name, key, **kwargs):
    try:
        mod = PanelOLS.from_formula(formula, data=data, drop_absorbed=True, **kwargs)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        coef = float(res.params[key])
        se   = float(res.std_errors[key])
        p    = float(res.pvalues[key])
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        log(f"{name:45s}  beta={coef:+.6f}  se={se:.6f}  p={p:.4f}{sig:3s}  N={int(res.nobs):,}")
        coefs[name] = {"coef": coef, "se": se, "pval": p, "nobs": int(res.nobs), "sig": sig.strip()}
        return res
    except Exception as e:
        log(f"{name}: FAILED - {e}")
        return None

log("=" * 80)
log("WUI POLYGON-YEAR REGRESSIONS")
log("=" * 80)
log(f"Panel: {df.shape[0]:,} polygon-years, {df['wui_code'].nunique():,} unique polygons")
log(f"Years: {df['year'].min()}-{df['year'].max()}")
log(f"Mean disaster rate: {df['disaster'].mean():.5f}")
log(f"Mean FRT_it: {df['FRT_it'].mean():.4f}")
log(f"Treated polygon-years: {int(df['FRT_it'].sum()):,}")
log(f"Disaster polygon-years: {int(df['disaster'].sum()):,}")
log("")
log("Disaster rate by treatment status:")
log(f"  Untreated FRT_it=0: {df[df['FRT_it']==0]['disaster'].mean():.5f} (N={int((df['FRT_it']==0).sum()):,})")
log(f"  Treated   FRT_it=1: {df[df['FRT_it']==1]['disaster'].mean():.5f} (N={int((df['FRT_it']==1).sum()):,})")

# -----------------------------------------------------------------------------
# 1. EXTENSIVE MARGIN: TWFE
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("1. EXTENSIVE MARGIN: TWO-WAY FIXED EFFECTS (binary disaster)")
log("=" * 80)

fit("disaster ~ FRT_it + EntityEffects", df_idx, "m1_FE_only", "FRT_it")
fit("disaster ~ FRT_it + EntityEffects + TimeEffects", df_idx, "m2_TWFE", "FRT_it")
fit("disaster ~ FRT_lag1 + EntityEffects + TimeEffects", df_idx, "m3_FRT_lag1", "FRT_lag1")
fit("disaster ~ FRT_lag3 + EntityEffects + TimeEffects", df_idx, "m4_FRT_lag3", "FRT_lag3")

log("")
log(f"Mean disaster: {df['disaster'].mean():.5f}")
coefs["mean_disaster"] = float(df['disaster'].mean())

# -----------------------------------------------------------------------------
# 2. INTENSIVE MARGIN
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("2. INTENSIVE MARGIN")
log("=" * 80)

fit("disaster ~ log_cum_acres + EntityEffects + TimeEffects", df_idx,
    "int1_log_cum", "log_cum_acres")
fit("disaster ~ ihs_cum_acres + EntityEffects + TimeEffects", df_idx,
    "int2_ihs", "ihs_cum_acres")
fit("disaster ~ log_cum_lag3 + EntityEffects + TimeEffects", df_idx,
    "int3_log_cum_lag3", "log_cum_lag3")
fit("disaster ~ log_cum_mech + log_cum_fire + EntityEffects + TimeEffects", df_idx,
    "int4_mech", "log_cum_mech")
fit("disaster ~ log_cum_mech + log_cum_fire + EntityEffects + TimeEffects", df_idx,
    "int4_fire", "log_cum_fire")

# -----------------------------------------------------------------------------
# 3. SEVERITY OUTCOME (continuous structures destroyed)
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("3. SEVERITY MARGIN: log(1 + structures destroyed)")
log("=" * 80)

fit("log_struct_destroyed ~ FRT_it + EntityEffects + TimeEffects", df_idx,
    "sev1_FRT", "FRT_it")
fit("log_struct_destroyed ~ log_cum_acres + EntityEffects + TimeEffects", df_idx,
    "sev2_log_cum", "log_cum_acres")
fit("log_struct_destroyed ~ log_cum_lag3 + EntityEffects + TimeEffects", df_idx,
    "sev3_log_cum_lag3", "log_cum_lag3")
fit("log_struct_destroyed ~ FRT_lag3 + EntityEffects + TimeEffects", df_idx,
    "sev4_FRT_lag3", "FRT_lag3")

# -----------------------------------------------------------------------------
# 4. INTERMIX ONLY (Radeloff's canonical WUI class)
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("4. INTERMIX ONLY")
log("=" * 80)

df_im = df[df["is_intermix"] == 1].copy()
df_im_idx = df_im.set_index(["wui_code","year"])
log(f"Intermix sample: {len(df_im):,} polygon-years")
log(f"Mean disaster (Intermix): {df_im['disaster'].mean():.5f}")

fit("disaster ~ FRT_it + EntityEffects + TimeEffects", df_im_idx,
    "im1_FRT_TWFE", "FRT_it")
fit("disaster ~ log_cum_lag3 + EntityEffects + TimeEffects", df_im_idx,
    "im2_log_cum_lag3", "log_cum_lag3")
fit("log_struct_destroyed ~ log_cum_lag3 + EntityEffects + TimeEffects", df_im_idx,
    "im3_sev_log_cum_lag3", "log_cum_lag3")

# -----------------------------------------------------------------------------
# 5. EVENT STUDY (binary disaster outcome)
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("5. EVENT STUDY (binned event time)")
log("=" * 80)

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

df["et_bin"] = df["event_time"].apply(bin_et)
for b in ["pre5plus","preMid","post0","post1_3","post4_6","post7plus"]:
    df[f"D_{b}"] = (df["et_bin"] == b).astype(int)

df_es_idx = df.set_index(["wui_code","year"])
es_terms = ["D_pre5plus","D_preMid","D_post0","D_post1_3","D_post4_6","D_post7plus"]
formula = "disaster ~ " + " + ".join(es_terms) + " + EntityEffects + TimeEffects"
mod_es = PanelOLS.from_formula(formula, data=df_es_idx, drop_absorbed=True)
res_es = mod_es.fit(cov_type="clustered", cluster_entity=True)

es_results = {}
for t in es_terms:
    if t in res_es.params.index:
        c = float(res_es.params[t])
        s = float(res_es.std_errors[t])
        p = float(res_es.pvalues[t])
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        log(f"  {t:15s}  {c:+.6f}  ({s:.6f})  p={p:.4f}{sig}")
        es_results[t] = {"coef": c, "se": s, "pval": p, "sig": sig.strip()}
coefs["event_study"] = es_results

# Joint test of pre-trends
try:
    pre_terms = [t for t in ["D_pre5plus","D_preMid"] if t in res_es.params.index]
    if len(pre_terms) >= 2:
        joint_test = res_es.wald_test(formula=" = ".join(pre_terms) + " = 0")
        jp = float(joint_test.pval)
        log(f"  Joint pre-trend p: {jp:.4f}")
        coefs["pretrend_pval"] = jp
except Exception as e:
    log(f"  Joint pre-trend test failed: {e}")

# -----------------------------------------------------------------------------
# 6. HETEROGENEITY: by density
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("6. HETEROGENEITY: by WUI density class")
log("=" * 80)

for d in ["Low","Med","High"]:
    df[f"FRT_{d}"] = df["FRT_it"] * (df["density"] == d).astype(int)

df_het_idx = df.set_index(["wui_code","year"])
mod_het = PanelOLS.from_formula(
    "disaster ~ FRT_Low + FRT_Med + FRT_High + EntityEffects + TimeEffects",
    data=df_het_idx, drop_absorbed=True,
)
res_het = mod_het.fit(cov_type="clustered", cluster_entity=True)
for k in ["FRT_Low","FRT_Med","FRT_High"]:
    if k in res_het.params.index:
        c = float(res_het.params[k]); s = float(res_het.std_errors[k]); p = float(res_het.pvalues[k])
        sig = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else ""))
        log(f"  {k:20s}  {c:+.6f}  ({s:.6f})  p={p:.4f}{sig}")
        coefs[f"het_{k}"] = {"coef": c, "se": s, "pval": p, "sig": sig.strip()}

# -----------------------------------------------------------------------------
# 7. PLACEBO: future treatment lead
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("7. PLACEBO: 5-year future treatment")
log("=" * 80)

df_pl = df.sort_values(["wui_code","year"]).copy()
df_pl["FRT_lead5"] = df_pl.groupby("wui_code")["FRT_it"].shift(-5).fillna(0).astype(int)
# Use pre-treatment window only
df_pre = df_pl[df_pl["FRT_it"] == 0].copy()
df_pre_idx = df_pre.set_index(["wui_code","year"])
fit("disaster ~ FRT_lead5 + EntityEffects + TimeEffects", df_pre_idx,
    "placebo_lead5", "FRT_lead5")

# -----------------------------------------------------------------------------
# 8. COST-BENEFIT
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("8. COST-BENEFIT ANALYSIS")
log("=" * 80)

# Use the TWFE extensive margin coefficient
beta1 = coefs["m2_TWFE"]["coef"]
log(f"Beta1 (extensive margin TWFE): {beta1:+.6f}")
log(f"|Beta1|: {abs(beta1):.6f}")

# Per-disaster loss components from disaster polygon-years in the panel
dis = df[df["disaster"] == 1]
log(f"Disaster polygon-years in panel: {len(dis):,}")
avg_fatal = float(dis["n_fatalities"].mean())
avg_struct = float(dis["n_struct_destroyed"].mean())
log(f"Avg fatalities per disaster polygon-year: {avg_fatal:.3f}")
log(f"Avg structures destroyed per disaster polygon-year: {avg_struct:.1f}")

VSL = 11.6e6
struct_value = 350_000
suppression_per_disaster = 5e6
loss_per_disaster = VSL*avg_fatal + struct_value*avg_struct + suppression_per_disaster
log(f"Avg total loss per disaster polygon-year: ${loss_per_disaster/1e6:.2f}M")

# Avoided damage per treated polygon-year
annual_benefit = abs(beta1) * loss_per_disaster
log(f"Avoided damage per treated polygon-year (binary spec): ${annual_benefit:,.0f}")

# Per-polygon treatment cost: treated polygons receive on average X acres per year
treated = df[df["acres_treated"] > 0]
avg_acres_per_polygon_year = float(treated["acres_treated"].mean())
avg_cost_per_acre = float(treated["avg_cost_per_acre"].mean(skipna=True))
log(f"Avg acres treated per treated polygon-year: {avg_acres_per_polygon_year:,.0f}")
log(f"Avg cost per acre: ${avg_cost_per_acre:,.0f}")
annual_cost = avg_acres_per_polygon_year * avg_cost_per_acre
log(f"Annual cost per treated polygon: ${annual_cost:,.0f}")

if beta1 < 0 and annual_cost > 0:
    bcr = annual_benefit / annual_cost
    log(f"\n  BCR (binary spec): {bcr:.2f}")
    coefs["bcr_extensive"] = bcr

# VSL sensitivity
log("\nVSL sensitivity on binary-spec BCR:")
for VSL_alt in [3e6, 7e6, 11.6e6, 15e6]:
    loss_alt = VSL_alt*avg_fatal + struct_value*avg_struct + suppression_per_disaster
    bcr_alt = abs(beta1) * loss_alt / annual_cost if annual_cost > 0 else float('nan')
    log(f"  VSL=${VSL_alt/1e6:5.1f}M -> BCR={bcr_alt:.2f}")

coefs["cba"] = {
    "beta1": beta1,
    "avg_fatal_per_disaster": avg_fatal,
    "avg_struct_per_disaster": avg_struct,
    "loss_per_disaster": loss_per_disaster,
    "annual_benefit": annual_benefit,
    "annual_cost": annual_cost,
    "avg_acres_per_polygon_year": avg_acres_per_polygon_year,
    "avg_cost_per_acre": avg_cost_per_acre,
    "VSL": VSL,
    "struct_value": struct_value,
    "suppression_per_disaster": suppression_per_disaster,
}

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
OUT_TXT.write_text("\n".join(lines))
with open(OUT_JSON, "w") as f:
    json.dump(coefs, f, indent=2, default=str)
print(f"\nWrote: {OUT_TXT}")
print(f"Wrote: {OUT_JSON}")
