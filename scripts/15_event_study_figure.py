"""
Year-by-year event study on the WUI polygon-year panel.
Estimates coefficients mu_k for k in [-5, 10] with k = -1 as the omitted reference.
Endpoint bins absorb |k| > 5 (pre) and k > 10 (post). Outcome: binary disaster.

Outputs:
  figures/event_study.pdf
  figures/event_study.png
  data/processed/event_study_yearly.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linearmodels.panel import PanelOLS

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
PANEL = ROOT / "data" / "processed" / "panel_wui_year.parquet"
FIG_DIR = ROOT / "figures"
OUT_CSV = ROOT / "data" / "processed" / "event_study_yearly.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading panel...")
df = pd.read_parquet(PANEL)
print(f"  shape: {df.shape}")

# Construct event time dummies. Range we want to plot: k in [-5, 10] with k = -1 omitted.
K_PRE_MIN = -5
K_POST_MAX = 10
OMIT = -1

df["wui_code"] = pd.Categorical(df["wui_id"]).codes
df = df.sort_values(["wui_code", "year"]).reset_index(drop=True)

# Bin event times outside the plotted window
def bin_et(et):
    if pd.isna(et):
        return np.nan
    et = int(et)
    if et < K_PRE_MIN:
        return -999  # pre endpoint bin
    if et > K_POST_MAX:
        return 999   # post endpoint bin
    return et

df["k"] = df["event_time"].apply(bin_et)

def kname(k):
    """Formula-safe variable name for event time k."""
    return f"D_n{abs(k)}" if k < 0 else f"D_p{k}"

# Build dummies for plotted ks (skip the reference)
plot_ks = [k for k in range(K_PRE_MIN, K_POST_MAX + 1) if k != OMIT]
for k in plot_ks:
    df[kname(k)] = (df["k"] == k).astype(int)
df["D_pre_far"] = (df["k"] == -999).astype(int)
df["D_post_far"] = (df["k"] == 999).astype(int)

# Always-control polygons (never treated) take all dummies = 0; they identify the
# baseline movement in the year fixed effects.

regressors = [kname(k) for k in plot_ks] + ["D_pre_far", "D_post_far"]
formula = "disaster ~ " + " + ".join(regressors) + " + EntityEffects + TimeEffects"

df_idx = df.set_index(["wui_code", "year"])
print("Fitting TWFE event study...")
mod = PanelOLS.from_formula(formula, data=df_idx, drop_absorbed=True)
res = mod.fit(cov_type="clustered", cluster_entity=True)

print("Done. Extracting coefficients...")
rows = []
for k in plot_ks:
    name = kname(k)
    if name in res.params.index:
        b = float(res.params[name])
        se = float(res.std_errors[name])
        rows.append({"k": k, "coef": b, "se": se,
                     "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
                     "pval": float(res.pvalues[name])})
# Add k = -1 reference (omitted)
rows.append({"k": -1, "coef": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "pval": np.nan})
es = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
es.to_csv(OUT_CSV, index=False)
print(es.to_string(index=False))

# Joint pre-trend test on plotted pre-period leads
pre_terms = [kname(k) for k in plot_ks if k < 0]
try:
    formula_pre = " = ".join(pre_terms) + " = 0"
    wt = res.wald_test(formula=formula_pre)
    pre_p = float(wt.pval)
    print(f"Joint pre-trend test (k in [-5,-2]): p = {pre_p:.4f}")
except Exception as e:
    pre_p = np.nan
    print(f"Pre-trend test failed: {e}")

# ---------------- Plot ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.0, 4.4))

pre = es[es["k"] < 0]
post = es[es["k"] >= 0]
ref = es[es["k"] == -1]

# Shade pre-treatment region
ax.axvspan(es["k"].min() - 0.5, -0.5, color="#f0f0f0", alpha=0.7, zorder=0)
ax.axhline(0, color="black", linewidth=0.6, linestyle="-", zorder=1)
ax.axvline(-0.5, color="gray", linewidth=0.7, linestyle="--", zorder=1)

# Pre coefficients
ax.errorbar(pre["k"], pre["coef"], yerr=1.96 * pre["se"],
            fmt="o", color="#888888", ecolor="#888888", elinewidth=1.0,
            capsize=2.5, markersize=4.5, label="Pre (leads)", zorder=3)

# Reference period (k = -1)
ax.scatter(ref["k"], ref["coef"], marker="D", color="white",
           edgecolor="black", s=45, zorder=4, label="Reference (k = -1)")

# Post coefficients
ax.errorbar(post["k"], post["coef"], yerr=1.96 * post["se"],
            fmt="o", color="#1f4e79", ecolor="#1f4e79", elinewidth=1.0,
            capsize=2.5, markersize=4.5, label="Post (lags)", zorder=3)

# Connecting line
ax.plot(es["k"], es["coef"], color="#1f4e79", linewidth=0.8, alpha=0.45, zorder=2)

ax.set_xlabel(r"Event time $k$ (years from first treatment)")
ax.set_ylabel(r"Estimated $\hat{\mu}_k$  (effect on Pr(disaster))")
ax.set_xticks(list(range(K_PRE_MIN, K_POST_MAX + 1)))
ax.set_xlim(K_PRE_MIN - 0.5, K_POST_MAX + 0.5)

ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax.legend(loc="upper left", frameon=False, ncol=1)

# Annotate pre-trend test
if not np.isnan(pre_p):
    ax.text(0.99, 0.03, f"Joint pre-trend $p$ = {pre_p:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", linewidth=0.5))

plt.tight_layout()
out_pdf = FIG_DIR / "event_study.pdf"
out_png = FIG_DIR / "event_study.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"\nWrote: {out_pdf}")
print(f"Wrote: {out_png}")
print(f"Wrote: {OUT_CSV}")
