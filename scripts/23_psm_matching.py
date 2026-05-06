"""
Propensity score matching to build a credible control sample for the
fuel reduction project.

Treated units:    polygons with any FACTS treatment within 5 km buffer between
                  2003 and 2024 (post-HFRA window)
Control units:    polygons with NO treatment in the entire 2000-2024 panel,
                  drawn from the 25-km vicinity around treated/disaster polygons

Matching strategy:
  1. Compute baseline pre-period (2000-2002) covariates per polygon:
       state (one-hot), density class, intermix dummy, log area, pre_tmean,
       pre_ppt, pre_disaster, pre_acres (set to 0 for never-treated by construction)
  2. Estimate propensity score via logit on the pre-period covariates
  3. 1:K nearest-neighbor matching on the propensity score, with caliper
     0.2 * sd(logit(p)), without replacement
  4. Coarsened exact matching on (state) before nearest-neighbor to ensure
     every match is within the same state (state-level fixed effects later
     would absorb cross-state shocks anyway)
  5. Save the matched (treated_id, control_id, ps_treated, ps_control) table
     and a flag column on the panel: matched_sample = 1

Output:
  data/processed/psm_matches.parquet         (treated_id, control_id, ps_t, ps_c)
  data/processed/panel_wui_year_v2.parquet   (overwritten with `matched_sample` col)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PANEL_PATH = ROOT / "data" / "processed" / "panel_wui_year_v2.parquet"
OUT_MATCH = ROOT / "data" / "processed" / "psm_matches.parquet"

K_NEAREST = 5            # 1:5 matching
CALIPER_SD = 0.2         # caliper as fraction of sd(logit p)
MIN_TREAT_YEAR = 2003    # post-HFRA

t0 = time.time()
print("=" * 72)
print("PROPENSITY SCORE MATCHING")
print(f"  K = {K_NEAREST}, caliper = {CALIPER_SD} * sd(logit p)")
print(f"  treated def: any FRT within 5 km, year >= {MIN_TREAT_YEAR}")
print("=" * 72)

# -----------------------------------------------------------------------------
# 1. Load panel and reduce to one row per polygon
# -----------------------------------------------------------------------------
print("\n[1] Loading panel ...")
df = pd.read_parquet(PANEL_PATH)
print(f"    {df.shape[0]:,} rows, {df['wui_id'].nunique():,} polygons")

# Treated = any treatment 2003-2024 (some polygons were treated only in
# 2000-2002 pre-period; we exclude them to keep the post-HFRA shock clean)
post = df[df["year"] >= MIN_TREAT_YEAR]
ever_treat_post = post.groupby("wui_id")["FRT_it"].max().rename("treated")

# Pre-period treatments (used to flag and exclude already-treated controls)
pre = df[df["year"] < MIN_TREAT_YEAR]
ever_treat_pre = pre.groupby("wui_id")["FRT_it"].max().rename("pre_treated")

cross = (
    df.groupby("wui_id")
    .first()[
        ["STATE", "WUICLASS", "area_km2", "is_intermix", "density",
         "pre_tmean", "pre_ppt", "pre_disaster", "pre_acres"]
    ]
    .join(ever_treat_post)
    .join(ever_treat_pre)
    .reset_index()
)
cross = cross.dropna(subset=["pre_tmean", "pre_ppt", "STATE", "density"])
cross["log_area"] = np.log1p(cross["area_km2"])
cross["density_low"] = (cross["density"] == "Low").astype(int)
cross["density_med"] = (cross["density"] == "Med").astype(int)
cross["density_high"] = (cross["density"] == "High").astype(int)

# Treated subsample = treated post-2003 AND not pre-treated
treated_mask = (cross["treated"] == 1) & (cross["pre_treated"] == 0)
control_mask = (cross["treated"] == 0) & (cross["pre_treated"] == 0)
print(f"    treated candidates: {treated_mask.sum():,}")
print(f"    control candidates: {control_mask.sum():,}")

# Drop polygons with neither treated nor control status (i.e., pre_treated = 1)
elig = cross[treated_mask | control_mask].reset_index(drop=True)
print(f"    eligible total: {len(elig):,}")

# -----------------------------------------------------------------------------
# 2. Coarsened exact matching on STATE, then PSM within each state
# -----------------------------------------------------------------------------
print("\n[2] State-stratified PSM ...")
features = [
    "log_area", "is_intermix", "density_low", "density_med",
    "density_high", "pre_tmean", "pre_ppt", "pre_disaster"
]

all_matches = []
states = sorted(elig["STATE"].unique())

for st in states:
    sub = elig[elig["STATE"] == st]
    n_t = int((sub["treated"] == 1).sum())
    n_c = int((sub["treated"] == 0).sum())
    if n_t == 0 or n_c == 0:
        continue

    X = sub[features].values
    y = sub["treated"].values
    if y.std() == 0:
        continue
    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Logit propensity
    try:
        logit = LogisticRegression(
            solver="lbfgs", max_iter=2000, n_jobs=1, C=1.0
        ).fit(Xs, y)
    except Exception as e:
        print(f"    {st}: logit failed ({e}); skipping")
        continue
    p = logit.predict_proba(Xs)[:, 1]
    # Logit transform of propensity for caliper
    eps = 1e-6
    p_clip = np.clip(p, eps, 1 - eps)
    lp = np.log(p_clip / (1 - p_clip))
    sd_lp = float(lp.std())
    cal = CALIPER_SD * sd_lp if sd_lp > 0 else np.inf

    # Treated and control indices in this state
    t_idx = np.where(sub["treated"].values == 1)[0]
    c_idx = np.where(sub["treated"].values == 0)[0]

    # Nearest neighbors on logit propensity
    lp_t = lp[t_idx].reshape(-1, 1)
    lp_c = lp[c_idx].reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=min(K_NEAREST, len(c_idx)))
    nn.fit(lp_c)
    dist, ind = nn.kneighbors(lp_t)

    # Apply caliper: drop matches with distance > caliper
    rows = []
    used_c = set()
    for ti, (dist_row, ind_row) in enumerate(zip(dist, ind)):
        for d, k in zip(dist_row, ind_row):
            if d > cal:
                continue
            ci = int(c_idx[k])
            if ci in used_c:
                continue   # without replacement
            used_c.add(ci)
            rows.append({
                "STATE": st,
                "treated_wui_id": sub["wui_id"].iloc[t_idx[ti]],
                "control_wui_id": sub["wui_id"].iloc[ci],
                "ps_treated": float(p[t_idx[ti]]),
                "ps_control": float(p[ci]),
                "logit_dist": float(d),
            })
    if rows:
        all_matches.append(pd.DataFrame(rows))
    print(f"    {st}: T={n_t:>6,}  C={n_c:>6,}  matched_pairs={len(rows):>6,}  sd_lp={sd_lp:.3f}  caliper={cal:.3f}")

matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
print(f"\n    total matched pairs: {len(matches):,}")
print(f"    unique treated:      {matches['treated_wui_id'].nunique():,}")
print(f"    unique controls:     {matches['control_wui_id'].nunique():,}")

# -----------------------------------------------------------------------------
# 3. Save matches and update panel
# -----------------------------------------------------------------------------
matches.to_parquet(OUT_MATCH)
print(f"\nWrote: {OUT_MATCH}")

# Update panel with matched_sample flag
print("\n[3] Updating panel with matched_sample flag ...")
in_match = set(matches["treated_wui_id"]) | set(matches["control_wui_id"])
df["matched_sample"] = df["wui_id"].isin(in_match).astype(int)
df.to_parquet(PANEL_PATH)
print(f"    matched_sample = 1 rows: {df['matched_sample'].sum():,}")
print(f"    matched_sample polygons: {df[df['matched_sample']==1]['wui_id'].nunique():,}")

# -----------------------------------------------------------------------------
# 4. Quick balance check
# -----------------------------------------------------------------------------
print("\n[4] Balance check on matched sample (means) ...")
m_t = elig[elig["wui_id"].isin(matches["treated_wui_id"])]
m_c = elig[elig["wui_id"].isin(matches["control_wui_id"])]
print(f"  {'feature':25s} {'treated':>10s} {'control':>10s} {'std diff':>10s}")
for f in features + ["area_km2"]:
    if f not in elig.columns:
        continue
    mt = float(m_t[f].mean())
    mc = float(m_c[f].mean())
    sd = float(elig[f].std())
    sdiff = (mt - mc) / sd if sd > 0 else 0.0
    print(f"  {f:25s} {mt:>10.3f} {mc:>10.3f} {sdiff:>+10.3f}")

print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")
