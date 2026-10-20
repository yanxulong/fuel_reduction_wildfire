"""
Re-estimate fuel reduction effects on the restructured panel v2.

Specifications (all use polygon + year FE, climate controls, polygon-clustered SE):

  A. MATCHED-DID  (panel restricted to PSM-matched sample)
     A1: disaster ~ FRT_it
     A2: disaster ~ FRT_it + tmean + log_ppt
     A3: disaster ~ log_T_stock + tmean + log_ppt        (decay-weighted intensity)
     A4: disaster ~ log_cum_acres + tmean + log_ppt      (cumulative intensity)

  B. SEVERITY ON MATCHED
     B1: log_struct_destroyed ~ FRT_it + tmean + log_ppt
     B2: log_struct_destroyed ~ log_T_stock + tmean + log_ppt
     B3: log_struct_destroyed ~ log_cum_acres + tmean + log_ppt

  C. BARTIK INSTRUMENT (full panel; instrument is state share x national budget)
     First stage : log_acres_treated ~ log_bartik + tmean + log_ppt + FE
     Reduced form: disaster          ~ log_bartik + tmean + log_ppt + FE
     2SLS Wald   : beta_iv = beta_RF / beta_FS  (delta-method SE)

  D. EVENT STUDY on matched sample using binned event time

  E. BALANCE check on the matched sample

Outputs:
  data/processed/wui_regs_v2_results.txt
  data/processed/wui_regs_v2_coefs.json
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
MATCH_PATH = ROOT / "data" / "processed" / "psm_matches.parquet"
OUT_TXT = ROOT / "data" / "processed" / "wui_regs_v2_results.txt"
OUT_JSON = ROOT / "data" / "processed" / "wui_regs_v2_coefs.json"

t0 = time.time()
print("Loading panel v2 ...")
df = pd.read_parquet(PANEL_PATH)
print(f"  panel: {df.shape}")
print(f"  matched_sample = 1: {df['matched_sample'].sum():,}")

# Derived variables
df["log_acres_treated"] = np.log1p(df["acres_treated"])
df["log_cum_acres"] = np.log1p(df["cum_acres"])
df["log_T_stock"] = np.log1p(df["T_stock"])
df["log_struct_destroyed"] = np.log1p(df["n_struct_destroyed"])
df["log_ppt"] = np.log1p(df["ppt_mm"])
df["log_bartik"] = np.log1p(df["bartik_acres"])

# Fill NaN climate via polygon mean
df["tmean_c"] = df.groupby("wui_id")["tmean_c"].transform(lambda s: s.fillna(s.mean()))
df["log_ppt"] = df.groupby("wui_id")["log_ppt"].transform(lambda s: s.fillna(s.mean()))
n0 = len(df)
df = df.dropna(subset=["tmean_c", "log_ppt"]).copy()
print(f"  dropped {n0-len(df):,} rows missing climate")

# Integer panel index
df["wui_code"] = pd.Categorical(df["wui_id"]).codes
print(f"  unique polygons (full): {df['wui_code'].nunique():,}")

# Sort for lags
df = df.sort_values(["wui_code", "year"]).reset_index(drop=True)

# Lag-3 binary indicator (used in severity spec)
df["FRT_lag3"] = (
    (df.groupby("wui_code")["FRT_it"].shift(1).fillna(0) +
     df.groupby("wui_code")["FRT_it"].shift(2).fillna(0) +
     df.groupby("wui_code")["FRT_it"].shift(3).fillna(0)) > 0
).astype(int)

# Matched subset (PSM treated + their controls)
mdf = df[df["matched_sample"] == 1].copy()
print(f"  matched panel: {mdf.shape}")
print(f"  matched polygons: {mdf['wui_id'].nunique():,}")

mdf_idx = mdf.set_index(["wui_code", "year"])
df_idx = df.set_index(["wui_code", "year"])

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
log("RESTRUCTURED ESTIMATION: MATCHED-DID + BARTIK IV ON PANEL v2")
log("=" * 80)
log(f"Full panel:    {df.shape[0]:,} obs, {df['wui_code'].nunique():,} polygons")
log(f"Matched panel: {mdf.shape[0]:,} obs, {mdf['wui_id'].nunique():,} polygons")
log(f"Years: {df['year'].min()}-{df['year'].max()}")
log(f"Mean disaster (full):    {df['disaster'].mean():.5f}")
log(f"Mean disaster (matched): {mdf['disaster'].mean():.5f}")
log(f"Mean FRT_it (full):      {df['FRT_it'].mean():.4f}")
log(f"Mean FRT_it (matched):   {mdf['FRT_it'].mean():.4f}")

# -----------------------------------------------------------------------------
# A. MATCHED-DID, BINARY DISASTER
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("A. MATCHED-DID, BINARY DISASTER OUTCOME")
log("=" * 80)

fit("disaster ~ FRT_it + EntityEffects + TimeEffects",
    mdf_idx, "A1_FRT_naive", ["FRT_it"])
fit("disaster ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "A2_FRT_clim", ["FRT_it", "tmean_c", "log_ppt"])
fit("disaster ~ log_T_stock + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "A3_Tstock", ["log_T_stock", "tmean_c", "log_ppt"])
fit("disaster ~ log_cum_acres + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "A4_logcum", ["log_cum_acres", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# B. SEVERITY MARGIN, MATCHED SAMPLE
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("B. MATCHED-DID, SEVERITY OUTCOME log(1 + structures destroyed)")
log("=" * 80)

fit("log_struct_destroyed ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "B1_FRT_clim", ["FRT_it", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ log_T_stock + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "B2_Tstock", ["log_T_stock", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ log_cum_acres + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "B3_logcum", ["log_cum_acres", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ FRT_lag3 + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "B4_FRT_lag3", ["FRT_lag3", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# C. BARTIK SHIFT-SHARE INSTRUMENT (full panel for first stage power)
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("C. BARTIK SHIFT-SHARE IV")
log("    Instrument: log(1 + state_pre_share * nat_acres_t)")
log("    Endogenous: log(1 + acres_treated_it)")
log("=" * 80)

# First stage on full panel
fs_full = fit(
    "log_acres_treated ~ log_bartik + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "C1_first_stage_FULL", ["log_bartik", "tmean_c", "log_ppt"]
)

# Reduced form on full panel
rf_full = fit(
    "disaster ~ log_bartik + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "C2_reduced_form_FULL", ["log_bartik", "tmean_c", "log_ppt"]
)

# 2SLS Wald (full panel)
if fs_full is not None and rf_full is not None:
    b_fs = float(fs_full.params["log_bartik"])
    s_fs = float(fs_full.std_errors["log_bartik"])
    b_rf = float(rf_full.params["log_bartik"])
    s_rf = float(rf_full.std_errors["log_bartik"])
    if abs(b_fs) > 1e-12:
        b_iv = b_rf / b_fs
        # Delta-method SE: var(rf/fs) ~= (1/fs^2)*var(rf) + (rf^2/fs^4)*var(fs)
        var_iv = (s_rf**2) / (b_fs**2) + (b_rf**2) * (s_fs**2) / (b_fs**4)
        s_iv = float(np.sqrt(var_iv))
        # First-stage F (squared t)
        F_fs = (b_fs / s_fs) ** 2
        # Wald p-value (z-stat)
        from scipy.stats import norm
        z_iv = b_iv / s_iv
        p_iv = float(2 * (1 - norm.cdf(abs(z_iv))))
        log(f"\n  --- C3_2SLS_Wald_FULL ---")
        log(f"    First-stage F:     {F_fs:.2f}")
        log(f"    Reduced-form beta: {b_rf:+.6f}  ({s_rf:.6f})")
        log(f"    First-stage beta:  {b_fs:+.6f}  ({s_fs:.6f})")
        log(f"    2SLS coefficient:  {b_iv:+.6f}  ({s_iv:.6f})  p={p_iv:.4f}")
        coefs["C3_2SLS_Wald_FULL"] = {
            "first_stage_beta": b_fs, "first_stage_se": s_fs, "first_stage_F": F_fs,
            "reduced_form_beta": b_rf, "reduced_form_se": s_rf,
            "iv_coef": b_iv, "iv_se": s_iv, "iv_pval": p_iv,
        }

# Same on matched sample
fs_m = fit(
    "log_acres_treated ~ log_bartik + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "C4_first_stage_MATCHED", ["log_bartik", "tmean_c", "log_ppt"]
)
rf_m = fit(
    "disaster ~ log_bartik + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_idx, "C5_reduced_form_MATCHED", ["log_bartik", "tmean_c", "log_ppt"]
)
if fs_m is not None and rf_m is not None:
    b_fs = float(fs_m.params["log_bartik"])
    s_fs = float(fs_m.std_errors["log_bartik"])
    b_rf = float(rf_m.params["log_bartik"])
    s_rf = float(rf_m.std_errors["log_bartik"])
    if abs(b_fs) > 1e-12:
        b_iv = b_rf / b_fs
        var_iv = (s_rf**2) / (b_fs**2) + (b_rf**2) * (s_fs**2) / (b_fs**4)
        s_iv = float(np.sqrt(var_iv))
        F_fs = (b_fs / s_fs) ** 2
        from scipy.stats import norm
        z_iv = b_iv / s_iv
        p_iv = float(2 * (1 - norm.cdf(abs(z_iv))))
        log(f"\n  --- C6_2SLS_Wald_MATCHED ---")
        log(f"    First-stage F:     {F_fs:.2f}")
        log(f"    Reduced-form beta: {b_rf:+.6f}  ({s_rf:.6f})")
        log(f"    First-stage beta:  {b_fs:+.6f}  ({s_fs:.6f})")
        log(f"    2SLS coefficient:  {b_iv:+.6f}  ({s_iv:.6f})  p={p_iv:.4f}")
        coefs["C6_2SLS_Wald_MATCHED"] = {
            "first_stage_beta": b_fs, "first_stage_se": s_fs, "first_stage_F": F_fs,
            "reduced_form_beta": b_rf, "reduced_form_se": s_rf,
            "iv_coef": b_iv, "iv_se": s_iv, "iv_pval": p_iv,
        }

# -----------------------------------------------------------------------------
# D. EVENT STUDY ON MATCHED SAMPLE
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("D. EVENT STUDY ON MATCHED SAMPLE (binned event time)")
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


mdf["et_bin"] = mdf["event_time"].apply(bin_et)
for b in ["pre5plus", "preMid", "post0", "post1_3", "post4_6", "post7plus"]:
    mdf[f"D_{b}"] = (mdf["et_bin"] == b).astype(int)
es_terms = ["D_pre5plus", "D_preMid", "D_post0", "D_post1_3", "D_post4_6", "D_post7plus"]

mdf_es_idx = mdf.set_index(["wui_code", "year"])
formula = "disaster ~ " + " + ".join(es_terms) + " + tmean_c + log_ppt + EntityEffects + TimeEffects"
try:
    mod_es = PanelOLS.from_formula(formula, data=mdf_es_idx, drop_absorbed=True)
    res_es = mod_es.fit(cov_type="clustered", cluster_entity=True)
    es_results = {}
    for t in es_terms + ["tmean_c", "log_ppt"]:
        if t in res_es.params.index:
            c = float(res_es.params[t])
            s = float(res_es.std_errors[t])
            p = float(res_es.pvalues[t])
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            log(f"  {t:15s}  {c:+.6f}  ({s:.6f})  p={p:.4f}{sig}")
            es_results[t] = {"coef": c, "se": s, "pval": p, "sig": sig.strip()}
    coefs["event_study_matched"] = es_results
    # Joint pre-trend test
    pre_terms = [t for t in ["D_pre5plus", "D_preMid"] if t in res_es.params.index]
    if len(pre_terms) >= 2:
        joint_test = res_es.wald_test(formula=" = ".join(pre_terms) + " = 0")
        jp = float(joint_test.pval)
        log(f"  Joint pre-trend p: {jp:.4f}")
        coefs["event_study_matched"]["pretrend_pval"] = jp
except Exception as e:
    log(f"  Event study FAILED: {e}")

# -----------------------------------------------------------------------------
# E. HETEROGENEITY BY WUI DENSITY ON MATCHED SAMPLE
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("E. HETEROGENEITY BY WUI DENSITY (matched sample)")
log("=" * 80)
for d in ["Low", "Med", "High"]:
    mdf[f"FRT_{d}"] = mdf["FRT_it"] * (mdf["density"] == d).astype(int)
mdf_h_idx = mdf.set_index(["wui_code", "year"])
fit("disaster ~ FRT_Low + FRT_Med + FRT_High + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_h_idx, "E1_density",
    ["FRT_Low", "FRT_Med", "FRT_High", "tmean_c", "log_ppt"])

# Intermix only
mdf_im = mdf[mdf["is_intermix"] == 1].copy()
mdf_im_idx = mdf_im.set_index(["wui_code", "year"])
log(f"\nIntermix subset: {len(mdf_im):,} polygon-years, {mdf_im['wui_id'].nunique():,} polygons")
fit("disaster ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_im_idx, "E2_intermix_FRT",
    ["FRT_it", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ log_T_stock + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_im_idx, "E3_intermix_sev_Tstock",
    ["log_T_stock", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# F. PLACEBO: 5-year FUTURE TREATMENT (matched sample)
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("F. PLACEBO: 5-year FUTURE TREATMENT (matched sample)")
log("=" * 80)
mdf_pl = mdf.sort_values(["wui_code", "year"]).copy()
mdf_pl["FRT_lead5"] = mdf_pl.groupby("wui_code")["FRT_it"].shift(-5).fillna(0).astype(int)
mdf_pre = mdf_pl[mdf_pl["FRT_it"] == 0].copy()
mdf_pre_idx = mdf_pre.set_index(["wui_code", "year"])
fit("disaster ~ FRT_lead5 + tmean_c + log_ppt + EntityEffects + TimeEffects",
    mdf_pre_idx, "F1_placebo_lead5", ["FRT_lead5", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# G. COST-BENEFIT (using matched-sample severity coefficient)
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("G. COST-BENEFIT ANALYSIS (using matched-sample TWFE)")
log("=" * 80)

if "A2_FRT_clim" in coefs and "FRT_it" in coefs["A2_FRT_clim"]:
    beta1 = coefs["A2_FRT_clim"]["FRT_it"]["coef"]
else:
    beta1 = float("nan")

log(f"Beta1 (matched A2_FRT_clim): {beta1:+.6f}")

# Per-disaster loss components
dis = mdf[mdf["disaster"] == 1]
log(f"Disaster polygon-years (matched): {len(dis):,}")
avg_fatal = float(dis["n_fatalities"].mean()) if len(dis) > 0 else 0.0
avg_struct = float(dis["n_struct_destroyed"].mean()) if len(dis) > 0 else 0.0
log(f"Avg fatalities per disaster: {avg_fatal:.3f}")
log(f"Avg structures destroyed: {avg_struct:.1f}")

VSL = 11.6e6
struct_value = 350_000
suppression_per_disaster = 5e6
loss_per_disaster = VSL * avg_fatal + struct_value * avg_struct + suppression_per_disaster
log(f"Avg total loss per disaster polygon-year: ${loss_per_disaster/1e6:.2f}M")

annual_benefit = abs(beta1) * loss_per_disaster
log(f"Avoided damage per treated polygon-year: ${annual_benefit:,.0f}")

treated = mdf[mdf["acres_treated"] > 0]
avg_acres_per_polygon_year = float(treated["acres_treated"].mean()) if len(treated) > 0 else 0.0
avg_cost_per_acre = float(treated["avg_cost_per_acre"].mean(skipna=True)) if len(treated) > 0 else 0.0
log(f"Avg acres treated per treated polygon-year: {avg_acres_per_polygon_year:,.0f}")
log(f"Avg cost per acre: ${avg_cost_per_acre:,.0f}")
annual_cost = avg_acres_per_polygon_year * avg_cost_per_acre
log(f"Annual cost per treated polygon: ${annual_cost:,.0f}")

if not np.isnan(beta1) and beta1 < 0 and annual_cost > 0:
    bcr = annual_benefit / annual_cost
    log(f"\n  BCR (matched binary spec): {bcr:.2f}")
    coefs["bcr_matched"] = bcr

# -----------------------------------------------------------------------------
# Meta + save
# -----------------------------------------------------------------------------
coefs["meta"] = {
    "n_obs_full": int(len(df)),
    "n_obs_matched": int(len(mdf)),
    "n_polygons_full": int(df["wui_code"].nunique()),
    "n_polygons_matched": int(mdf["wui_id"].nunique()),
    "mean_disaster_full": float(df["disaster"].mean()),
    "mean_disaster_matched": float(mdf["disaster"].mean()),
    "mean_FRT_full": float(df["FRT_it"].mean()),
    "mean_FRT_matched": float(mdf["FRT_it"].mean()),
    "mean_tmean_c": float(df["tmean_c"].mean()),
    "mean_log_ppt": float(df["log_ppt"].mean()),
}

OUT_TXT.write_text("\n".join(lines))
with open(OUT_JSON, "w") as f:
    json.dump(coefs, f, indent=2, default=str)
print(f"\nWrote: {OUT_TXT}")
print(f"Wrote: {OUT_JSON}")
print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
