"""
Robustness checks targeting the two identification threats from script 24:
  1. Pre-trend failure (matched event study, joint p < 0.0001)
  2. Significant 5-year lead placebo

Strategies:
  R1. DONUT EVENT STUDY: drop t in [-3, -1] (the immediate pre-period
      that contains any anticipation/Ashenfelter spike), refit event study
      using t = -5 as the omitted reference instead of t = -1.

  R2. DONUT TWFE: same data restriction, refit the static TWFE
      coefficient on FRT_it, log_T_stock, and FRT_lag3.

  R3. LONG-DIFFERENCE: collapse the panel to two periods per polygon:
        period A = 2000-2002 (pre-HFRA, untreated by definition)
        period B = 2018-2024 (mature post-HFRA window)
      Run a polygon-level first-difference regression of
        Delta(disaster_rate) on Delta(FRT_ever)
      with state fixed effects and pre-period climate controls.

  R4. NEVER-vs-EVENTUALLY: drop "always-treated" polygons (treated in
      every year of the post-HFRA window) and re-estimate the static
      matched-DiD spec on the cleaner sample.

Outputs:
  data/processed/wui_regs_v2_robust.txt
  data/processed/wui_regs_v2_robust_coefs.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from linearmodels.panel import PanelOLS

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PANEL_PATH = ROOT / "data" / "processed" / "panel_wui_year_v2.parquet"
OUT_TXT = ROOT / "data" / "processed" / "wui_regs_v2_robust.txt"
OUT_JSON = ROOT / "data" / "processed" / "wui_regs_v2_robust_coefs.json"

t0 = time.time()
print("Loading panel v2 ...")
df = pd.read_parquet(PANEL_PATH)
print(f"  panel: {df.shape}")

# Build derived
df["log_acres_treated"] = np.log1p(df["acres_treated"])
df["log_T_stock"] = np.log1p(df["T_stock"])
df["log_struct_destroyed"] = np.log1p(df["n_struct_destroyed"])
df["log_ppt"] = np.log1p(df["ppt_mm"])

df["tmean_c"] = df.groupby("wui_id")["tmean_c"].transform(lambda s: s.fillna(s.mean()))
df["log_ppt"] = df.groupby("wui_id")["log_ppt"].transform(lambda s: s.fillna(s.mean()))
df = df.dropna(subset=["tmean_c", "log_ppt"]).copy()

df["wui_code"] = pd.Categorical(df["wui_id"]).codes
df = df.sort_values(["wui_code", "year"]).reset_index(drop=True)

df["FRT_lag3"] = (
    (df.groupby("wui_code")["FRT_it"].shift(1).fillna(0) +
     df.groupby("wui_code")["FRT_it"].shift(2).fillna(0) +
     df.groupby("wui_code")["FRT_it"].shift(3).fillna(0)) > 0
).astype(int)

mdf = df[df["matched_sample"] == 1].copy()
print(f"  matched panel: {mdf.shape}")

lines = []
coefs = {}


def log(s=""):
    print(s)
    lines.append(s)


def fit(formula, data, name, keys):
    try:
        mod = PanelOLS.from_formula(formula, data=data, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        log(f"--- {name} ---")
        log(f"  N={int(res.nobs):,}  within-R2={res.rsquared_within:.4f}")
        row = {}
        for k in keys:
            if k in res.params.index:
                c = float(res.params[k])
                s = float(res.std_errors[k])
                p = float(res.pvalues[k])
                sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                log(f"  {k:22s}  {c:+.6f}  ({s:.6f})  p={p:.4f}{sig}")
                row[k] = {"coef": c, "se": s, "pval": p, "sig": sig.strip()}
        row["nobs"] = int(res.nobs)
        coefs[name] = row
        return res
    except Exception as e:
        log(f"  FAILED: {e}")
        return None


log("=" * 80)
log("ROBUSTNESS: DONUT, LONG-DIFFERENCE, NEVER-VS-EVENTUALLY")
log("=" * 80)

# -----------------------------------------------------------------------------
# R1. DONUT EVENT STUDY: drop t in [-3,-1], reference t=-5
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("R1. DONUT EVENT STUDY (drop t in [-3,-1], reference t=-5)")
log("=" * 80)

mdf_d = mdf.copy()
# Drop polygon-years where event_time is in {-3,-2,-1}
keep = ~mdf_d["event_time"].isin([-3, -2, -1])
mdf_d = mdf_d[keep].copy()
log(f"  After dropping t in [-3,-1]: {len(mdf_d):,} obs (was {len(mdf):,})")


def bin_donut(et):
    if pd.isna(et):
        return "never"
    et = int(et)
    if et <= -5:
        return "OMIT"
    if et == -4:
        return "pre4"
    if et == 0:
        return "post0"
    if et <= 3:
        return "post1_3"
    if et <= 6:
        return "post4_6"
    return "post7plus"


mdf_d["et_bin"] = mdf_d["event_time"].apply(bin_donut)
for b in ["pre4", "post0", "post1_3", "post4_6", "post7plus"]:
    mdf_d[f"D_{b}"] = (mdf_d["et_bin"] == b).astype(int)

es_terms = ["D_pre4", "D_post0", "D_post1_3", "D_post4_6", "D_post7plus"]
mdf_d_idx = mdf_d.set_index(["wui_code", "year"])
formula = "disaster ~ " + " + ".join(es_terms) + " + tmean_c + log_ppt + EntityEffects + TimeEffects"
try:
    mod = PanelOLS.from_formula(formula, data=mdf_d_idx, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    es_results = {}
    for t in es_terms + ["tmean_c", "log_ppt"]:
        if t in res.params.index:
            c = float(res.params[t])
            s = float(res.std_errors[t])
            p = float(res.pvalues[t])
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            log(f"  {t:15s}  {c:+.6f}  ({s:.6f})  p={p:.4f}{sig}")
            es_results[t] = {"coef": c, "se": s, "pval": p, "sig": sig.strip()}
    coefs["R1_donut_event_study"] = es_results
    if "D_pre4" in res.params.index:
        try:
            jtest = res.wald_test(formula="D_pre4 = 0")
            log(f"  Pre-period (t=-4) coefficient p: {float(jtest.pval):.4f}")
            coefs["R1_donut_event_study"]["pretrend_pval"] = float(jtest.pval)
        except Exception as e:
            log(f"  pretrend test failed: {e}")
except Exception as e:
    log(f"  Donut event study FAILED: {e}")

# -----------------------------------------------------------------------------
# R2. DONUT TWFE static specs
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("R2. DONUT TWFE STATIC SPECS (drop t in [-3,-1])")
log("=" * 80)

mdf_d_idx = mdf_d.set_index(["wui_code", "year"])
fit("disaster ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_d_idx, "R2a_FRT_donut", ["FRT_it", "tmean_c", "log_ppt"])
fit("disaster ~ log_T_stock + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_d_idx, "R2b_Tstock_donut", ["log_T_stock", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_d_idx, "R2c_sevFRT_donut", ["FRT_it", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ FRT_lag3 + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_d_idx, "R2d_sevFRTlag3_donut", ["FRT_lag3", "tmean_c", "log_ppt"])

# Heterogeneity by density on donut sample
for d in ["Low", "Med", "High"]:
    mdf_d[f"FRT_{d}"] = mdf_d["FRT_it"] * (mdf_d["density"] == d).astype(int)
mdf_d_idx2 = mdf_d.set_index(["wui_code", "year"])
fit("disaster ~ FRT_Low + FRT_Med + FRT_High + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_d_idx2, "R2e_density_donut",
    ["FRT_Low", "FRT_Med", "FRT_High", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# R3. LONG-DIFFERENCE: pre 2000-2002 vs post 2018-2024
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("R3. LONG-DIFFERENCE: pre 2000-2002 vs post 2018-2024")
log("=" * 80)

PRE = (2000, 2002)
POST = (2018, 2024)

# Use the matched sample, collapse to polygon means by period
mdf["period"] = np.where(
    mdf["year"].between(PRE[0], PRE[1]), "pre",
    np.where(mdf["year"].between(POST[0], POST[1]), "post", "drop")
)
mdf2 = mdf[mdf["period"] != "drop"].copy()
log(f"  pre-period rows: {(mdf2['period'] == 'pre').sum():,}")
log(f"  post-period rows: {(mdf2['period'] == 'post').sum():,}")

agg = mdf2.groupby(["wui_id", "period"]).agg(
    disaster_rate=("disaster", "mean"),
    log_struct=("log_struct_destroyed", "mean"),
    FRT_ever=("FRT_it", "max"),
    log_T_stock=("log_T_stock", "mean"),
    tmean_c=("tmean_c", "mean"),
    log_ppt=("log_ppt", "mean"),
    STATE=("STATE", "first"),
    density=("density", "first"),
    is_intermix=("is_intermix", "first"),
).reset_index()

# Pivot to wide
wide = agg.pivot(index="wui_id", columns="period",
                 values=["disaster_rate", "log_struct", "FRT_ever",
                         "log_T_stock", "tmean_c", "log_ppt"]).reset_index()
wide.columns = ["wui_id"] + [f"{a}_{b}" for a, b in wide.columns[1:]]
# Bring back state, density
meta = mdf.groupby("wui_id").agg(
    STATE=("STATE", "first"),
    density=("density", "first"),
    is_intermix=("is_intermix", "first"),
).reset_index()
wide = wide.merge(meta, on="wui_id", how="left")
wide = wide.dropna(subset=["disaster_rate_post", "disaster_rate_pre",
                            "FRT_ever_post", "tmean_c_post", "log_ppt_post"])
log(f"  long-difference cross-section: {len(wide):,} polygons")

# Construct deltas
wide["d_disaster"] = wide["disaster_rate_post"] - wide["disaster_rate_pre"]
wide["d_log_struct"] = wide["log_struct_post"] - wide["log_struct_pre"]
wide["d_FRT_ever"] = wide["FRT_ever_post"] - wide["FRT_ever_pre"]
wide["d_log_T_stock"] = wide["log_T_stock_post"] - wide["log_T_stock_pre"]
wide["d_tmean"] = wide["tmean_c_post"] - wide["tmean_c_pre"]
wide["d_log_ppt"] = wide["log_ppt_post"] - wide["log_ppt_pre"]

# State dummies
state_d = pd.get_dummies(wide["STATE"], prefix="ST", drop_first=True).astype(float)
wide_x = pd.concat([wide, state_d], axis=1)

# Long-difference regression: Delta(y) ~ Delta(treat) + Delta(climate) + state FE
import statsmodels.api as sm

state_cols = list(state_d.columns)


def ld_fit(yvar, xvar, name):
    try:
        Y = wide_x[yvar].astype(float)
        X = wide_x[[xvar, "d_tmean", "d_log_ppt"] + state_cols].astype(float)
        X = sm.add_constant(X)
        ok = Y.notna() & X.notna().all(axis=1)
        Y, X = Y[ok], X[ok]
        m = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": wide_x.loc[ok, "STATE"]})
        c = float(m.params[xvar])
        s = float(m.bse[xvar])
        p = float(m.pvalues[xvar])
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        log(f"--- {name} ---  N={int(m.nobs):,}")
        log(f"  {xvar:15s}  {c:+.6f}  ({s:.6f})  p={p:.4f}{sig}")
        log(f"  d_tmean        {float(m.params['d_tmean']):+.6f}  ({float(m.bse['d_tmean']):.6f})")
        log(f"  d_log_ppt      {float(m.params['d_log_ppt']):+.6f}  ({float(m.bse['d_log_ppt']):.6f})")
        coefs[name] = {
            xvar: {"coef": c, "se": s, "pval": p, "sig": sig.strip()},
            "nobs": int(m.nobs),
        }
        return m
    except Exception as e:
        log(f"  {name} FAILED: {e}")
        return None


ld_fit("d_disaster", "d_FRT_ever", "R3a_LD_disaster_FRT")
ld_fit("d_disaster", "d_log_T_stock", "R3b_LD_disaster_Tstock")
ld_fit("d_log_struct", "d_FRT_ever", "R3c_LD_severity_FRT")
ld_fit("d_log_struct", "d_log_T_stock", "R3d_LD_severity_Tstock")

# Density-stratified long differences
for dens in ["Low", "Med", "High"]:
    sub = wide_x[wide_x["density"] == dens].copy()
    if len(sub) < 100:
        log(f"  R3_LD_density_{dens}: too few obs ({len(sub)})")
        continue
    Y = sub["d_disaster"].astype(float)
    X = sub[["d_FRT_ever", "d_tmean", "d_log_ppt"] + state_cols].astype(float)
    X = sm.add_constant(X)
    ok = Y.notna() & X.notna().all(axis=1)
    Y, X = Y[ok], X[ok]
    try:
        m = sm.OLS(Y, X).fit(cov_type="cluster",
                              cov_kwds={"groups": sub.loc[ok, "STATE"]})
        c = float(m.params["d_FRT_ever"])
        s = float(m.bse["d_FRT_ever"])
        p = float(m.pvalues["d_FRT_ever"])
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        log(f"  R3_LD_density_{dens:4s}  N={int(m.nobs):>6,}  d_FRT_ever {c:+.6f} ({s:.6f}) p={p:.4f}{sig}")
        coefs[f"R3_LD_density_{dens}"] = {
            "d_FRT_ever": {"coef": c, "se": s, "pval": p, "sig": sig.strip()},
            "nobs": int(m.nobs),
        }
    except Exception as e:
        log(f"  R3_LD_density_{dens} FAILED: {e}")

# -----------------------------------------------------------------------------
# R4. NEVER-VS-EVENTUALLY: drop always-treated, restrict to clean comparison
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("R4. NEVER-VS-EVENTUALLY (drop always-treated polygons)")
log("=" * 80)

# Define "always treated" as polygons with first_treat_year <= 2003
always_t = mdf[mdf["first_treat_year"] <= 2003]["wui_id"].unique()
log(f"  always-treated (first_treat_year <= 2003): {len(always_t):,}")
mdf_clean = mdf[~mdf["wui_id"].isin(always_t)].copy()
log(f"  clean sample: {len(mdf_clean):,} obs, {mdf_clean['wui_id'].nunique():,} polygons")

mdf_c_idx = mdf_clean.set_index(["wui_code", "year"])
fit("disaster ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_c_idx, "R4a_FRT_clean", ["FRT_it", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_c_idx, "R4b_sevFRT_clean", ["FRT_it", "tmean_c", "log_ppt"])

for d in ["Low", "Med", "High"]:
    mdf_clean[f"FRT_{d}"] = mdf_clean["FRT_it"] * (mdf_clean["density"] == d).astype(int)
mdf_c2_idx = mdf_clean.set_index(["wui_code", "year"])
fit("disaster ~ FRT_Low + FRT_Med + FRT_High + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_c2_idx, "R4c_density_clean",
    ["FRT_Low", "FRT_Med", "FRT_High", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
OUT_TXT.write_text("\n".join(lines))
with open(OUT_JSON, "w") as f:
    json.dump(coefs, f, indent=2, default=str)
print(f"\nWrote: {OUT_TXT}")
print(f"Wrote: {OUT_JSON}")
print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
