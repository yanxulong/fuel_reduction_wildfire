"""
Diagnostic: why do matched DiD and long-difference give opposite signs on
the same matched panel?

Three possible explanations:
  H1. delta(FRT) has near-zero variance in long-diff (treated polygons
      already FRT=0 in 2000-02 and ~1 in 2018-24, controls 0 in both,
      so delta(FRT) is mechanically a level cross-section).
  H2. Matched DiD loads on year-FE-residualized differential trends that
      long-diff averages away.
  H3. Treatment effects are concentrated in years 1-3 post-treatment
      and decay, so the two designs identify different estimands.

What we report:
  - distribution of FRT_it within polygon over the pre and post windows
    (separately for matched treated and matched controls)
  - var(delta FRT) for matched treated only and overall
  - share of treated polygons with FRT_pre > 0 (already-treated by 2002)
    and share with FRT_post < 1 (not yet treated by 2018)
  - same diagnostics for log_T_stock and log_cum_acres (continuous)
  - within-polygon time pattern of the disaster outcome around event time
    on matched sample, broken out by treated vs matched-control
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PANEL_PATH = ROOT / "data" / "processed" / "panel_wui_year_v2.parquet"

t0 = time.time()
print("Loading panel v2 ...")
df = pd.read_parquet(PANEL_PATH)
print(f"  {df.shape[0]:,} rows, {df['wui_id'].nunique():,} polygons")

m = df[df["matched_sample"] == 1].copy()
print(f"  matched: {m.shape[0]:,} rows, {m['wui_id'].nunique():,} polygons")

# Identify treated vs matched-control polygons by their EVER-treated status
ever_treated = m.groupby("wui_id")["FRT_it"].max()
treated_ids = set(ever_treated[ever_treated == 1].index)
control_ids = set(ever_treated[ever_treated == 0].index)
print(f"  treated polygons:  {len(treated_ids):,}")
print(f"  control polygons:  {len(control_ids):,}")

m["is_treated"] = m["wui_id"].isin(treated_ids)

PRE = (2000, 2002)
POST = (2018, 2024)

# H1 diagnostic: distribution of FRT in pre/post windows
def block_stats(sub, window, label):
    yr_lo, yr_hi = window
    s = sub[(sub["year"] >= yr_lo) & (sub["year"] <= yr_hi)]
    g = s.groupby("wui_id")["FRT_it"].mean()
    return pd.Series({
        f"{label}_n_polys": len(g),
        f"{label}_share_FRT_eq_0": (g == 0).mean(),
        f"{label}_share_FRT_eq_1": (g == 1).mean(),
        f"{label}_share_FRT_in_(0,1)": ((g > 0) & (g < 1)).mean(),
        f"{label}_mean_FRT": g.mean(),
    })

print("\n=== H1: distribution of FRT_it in pre and post windows ===")
print("\n  TREATED polygons (matched sample):")
treated = m[m["is_treated"]]
print(block_stats(treated, PRE,  "treated_pre").to_string())
print()
print(block_stats(treated, POST, "treated_post").to_string())

print("\n  CONTROL polygons (matched sample):")
ctrl = m[~m["is_treated"]]
print(block_stats(ctrl, PRE,  "ctrl_pre").to_string())
print()
print(block_stats(ctrl, POST, "ctrl_post").to_string())

# Long-difference DELTA_FRT distribution
print("\n=== H1: distribution of Delta(FRT) used in long-difference ===")
def long_diff_var(sub, var, treated_label):
    pre = sub[(sub["year"] >= PRE[0])  & (sub["year"] <= PRE[1])].groupby("wui_id")[var].mean()
    post = sub[(sub["year"] >= POST[0]) & (sub["year"] <= POST[1])].groupby("wui_id")[var].mean()
    common = pre.index.intersection(post.index)
    delta = (post.loc[common] - pre.loc[common])
    print(f"\n  {var:18s}  ({treated_label})")
    print(f"    n polygons        : {len(delta):,}")
    print(f"    mean(delta)       : {delta.mean():+.6f}")
    print(f"    sd(delta)         : {delta.std():.6f}")
    print(f"    var(delta)        : {delta.var():.8f}")
    print(f"    share delta == 0  : {(delta == 0).mean():.4f}")
    print(f"    share delta > 0   : {(delta > 0).mean():.4f}")
    print(f"    share delta == 1  : {(delta == 1).mean():.4f}  (binary FRT only)")

print("\n  Pooled matched sample:")
long_diff_var(m, "FRT_it", "all")

print("\n  Treated only:")
long_diff_var(treated, "FRT_it", "treated")

print("\n  Controls only:")
long_diff_var(ctrl, "FRT_it", "controls")

# Continuous treatment in long-diff
print("\n=== H1 continuous: distribution of Delta(log T_stock) and Delta(log cum acres) ===")
m["log_T_stock"] = np.log1p(m["T_stock"])
m["log_cum_acres"] = np.log1p(m["cum_acres"])
treated2 = m[m["is_treated"]]
ctrl2 = m[~m["is_treated"]]
for col in ["log_T_stock", "log_cum_acres"]:
    long_diff_var(m,        col, "all")
    long_diff_var(treated2, col, "treated")
    long_diff_var(ctrl2,    col, "controls")

# H3: pre/post pattern by event time
print("\n=== H3: mean disaster rate by event time, treated polygons only ===")
first_treat = treated.loc[treated["FRT_it"] == 1].groupby("wui_id")["year"].min()
treated = treated.copy()
treated["g"] = treated["wui_id"].map(first_treat)
treated["k"] = treated["year"] - treated["g"]
by_k = treated.groupby("k")["disaster"].agg(["mean", "count"]).reset_index()
print(by_k[(by_k["k"] >= -10) & (by_k["k"] <= 15)].to_string(index=False))

# Mean disaster rate by year for treated vs controls  raw
print("\n=== Annual disaster rate, treated vs matched controls (raw, no FE) ===")
yr = m.groupby(["year", "is_treated"])["disaster"].mean().unstack()
yr.columns = ["control", "treated"]
yr["diff"] = yr["treated"] - yr["control"]
print(yr.to_string())

print(f"\ndone in {time.time() - t0:.1f}s")
