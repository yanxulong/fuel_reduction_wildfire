"""
Re-run the WUI polygon-year panel regressions adding time-varying
temperature (tmean_c) and precipitation (ppt_mm) controls from PRISM 4 km.

All specifications retain polygon and year fixed effects. The climate
controls capture within-polygon time variation in fire-weather that the
year fixed effect cannot absorb (because year FE only remove the national
mean for each year).

Outputs:
  data/processed/wui_regression_results_climate.txt
  data/processed/wui_regression_coefs_climate.json
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
OUT_TXT = ROOT / "data" / "processed" / "wui_regression_results_climate.txt"
OUT_JSON = ROOT / "data" / "processed" / "wui_regression_coefs_climate.json"

print("Loading panel (with climate)...")
df = pd.read_parquet(PANEL_PATH)
print(f"  shape: {df.shape}")
assert "tmean_c" in df.columns, "climate columns missing; run 20 first"
print(f"  tmean NaN: {df['tmean_c'].isna().sum():,}")
print(f"  ppt   NaN: {df['ppt_mm'].isna().sum():,}")

# Fill NaNs using polygon mean (handles a few WUI blocks that fall outside
# PRISM ocean/coastal mask). Drop any remaining NaN rows.
df["tmean_c"] = df.groupby("wui_id")["tmean_c"].transform(lambda s: s.fillna(s.mean()))
df["ppt_mm"]  = df.groupby("wui_id")["ppt_mm"].transform(lambda s: s.fillna(s.mean()))
n0 = len(df)
df = df.dropna(subset=["tmean_c", "ppt_mm"]).copy()
print(f"  dropped {n0-len(df):,} rows with missing climate")

# Derived variables (same as script 14)
df["log_cum_acres"] = np.log1p(df["cum_acres"])
df["ihs_cum_acres"] = np.arcsinh(df["cum_acres"])
df["log_cum_mech"]  = np.log1p(df["cum_acres_mech"])
df["log_cum_fire"]  = np.log1p(df["cum_acres_fire"])
df["disaster_struct"] = (df["n_struct_destroyed"] > 0).astype(int)
df["log_struct_destroyed"] = np.log1p(df["n_struct_destroyed"])

# Integer code for panel index
df["wui_code"] = pd.Categorical(df["wui_id"]).codes
print(f"  unique wui_ids: {df['wui_code'].nunique():,}")

# Lags
df = df.sort_values(["wui_code", "year"]).reset_index(drop=True)
df["FRT_lag3"] = (
    (df.groupby("wui_code")["FRT_it"].shift(1).fillna(0) +
     df.groupby("wui_code")["FRT_it"].shift(2).fillna(0) +
     df.groupby("wui_code")["FRT_it"].shift(3).fillna(0)) > 0
).astype(int)
df["log_cum_lag3"] = np.log1p(
    df.groupby("wui_code")["cum_acres"].shift(3).fillna(0)
)

# Precipitation often enters nonlinearly; include log(1+ppt) for robustness.
df["log_ppt"] = np.log1p(df["ppt_mm"])

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
log("WUI POLYGON-YEAR REGRESSIONS WITH CLIMATE CONTROLS")
log("=" * 80)
log(f"Panel: {df.shape[0]:,} polygon-years, {df['wui_code'].nunique():,} polygons")
log(f"Years: {df['year'].min()}-{df['year'].max()}")
log(f"Mean tmean: {df['tmean_c'].mean():.2f} C")
log(f"Mean ppt:   {df['ppt_mm'].mean():.0f} mm")
log(f"Mean disaster rate: {df['disaster'].mean():.5f}")
log(f"Mean FRT_it:        {df['FRT_it'].mean():.4f}")
log("")

# -----------------------------------------------------------------------------
# 1. MAIN TABLE equivalents with climate
# -----------------------------------------------------------------------------
log("=" * 80)
log("1. EXTENSIVE + INTENSIVE MARGIN (binary disaster outcome)")
log("=" * 80)

fit("disaster ~ FRT_it + tmean_c + log_ppt + EntityEffects",
    df_idx, "m1_FE_only", ["FRT_it", "tmean_c", "log_ppt"])
fit("disaster ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "m2_TWFE", ["FRT_it", "tmean_c", "log_ppt"])
fit("disaster ~ log_cum_acres + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "m3_log_cum", ["log_cum_acres", "tmean_c", "log_ppt"])
fit("disaster ~ log_cum_lag3 + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "m4_log_cum_lag3", ["log_cum_lag3", "tmean_c", "log_ppt"])
fit("disaster ~ log_cum_mech + log_cum_fire + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "m5_mech_fire", ["log_cum_mech", "log_cum_fire", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# 2. SEVERITY MARGIN
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("2. SEVERITY MARGIN: log(1 + structures destroyed)")
log("=" * 80)

fit("log_struct_destroyed ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "sev1_FRT", ["FRT_it", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ log_cum_acres + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "sev2_log_cum", ["log_cum_acres", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ log_cum_lag3 + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "sev3_log_cum_lag3", ["log_cum_lag3", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ FRT_lag3 + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_idx, "sev4_FRT_lag3", ["FRT_lag3", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# 3. HETEROGENEITY: density interactions
# -----------------------------------------------------------------------------
log("\n" + "=" * 80)
log("3. HETEROGENEITY: by WUI density class (occurrence outcome)")
log("=" * 80)

for d in ["Low", "Med", "High"]:
    df[f"FRT_{d}"] = df["FRT_it"] * (df["density"] == d).astype(int)
df_het_idx = df.set_index(["wui_code", "year"])

fit("disaster ~ FRT_Low + FRT_Med + FRT_High + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_het_idx, "het_density",
    ["FRT_Low", "FRT_Med", "FRT_High", "tmean_c", "log_ppt"])

# Intermix-only interaction
df_im = df[df["is_intermix"] == 1].copy()
df_im_idx = df_im.set_index(["wui_code", "year"])
log(f"\nIntermix sample: {len(df_im):,} polygon-years")

fit("disaster ~ FRT_it + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_im_idx, "im_FRT_TWFE", ["FRT_it", "tmean_c", "log_ppt"])
fit("log_struct_destroyed ~ log_cum_lag3 + tmean_c + log_ppt + EntityEffects + TimeEffects",
    df_im_idx, "im_sev_log_cum_lag3", ["log_cum_lag3", "tmean_c", "log_ppt"])

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
coefs["meta"] = {
    "n_obs": int(len(df)),
    "n_polygons": int(df["wui_code"].nunique()),
    "mean_tmean_c": float(df["tmean_c"].mean()),
    "mean_ppt_mm": float(df["ppt_mm"].mean()),
    "mean_disaster": float(df["disaster"].mean()),
    "mean_FRT_it": float(df["FRT_it"].mean()),
}

OUT_TXT.write_text("\n".join(lines))
with open(OUT_JSON, "w") as f:
    json.dump(coefs, f, indent=2, default=str)
print(f"\nWrote: {OUT_TXT}")
print(f"Wrote: {OUT_JSON}")
