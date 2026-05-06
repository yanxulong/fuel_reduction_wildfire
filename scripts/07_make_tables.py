"""
Generate LaTeX tables and update paper_jeem.tex with real estimates from final_results.json.
"""
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
RES = ROOT / "results"
PAPER = ROOT / "paper_jeem.tex"

with open(RES / "final_results.json") as f:
    R = json.load(f)

ss = R["summary_stats"]
t1 = R["table1_main"]
t2 = R["table2_west"]
t3 = R["table3_type"]
ev = R["event_study"]
cba = R["cost_benefit"]

def stars(p):
    if p is None: return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def fmt(c, s, p, d=4):
    return f"{c:+.{d}f}{stars(p)}\\\\\n        & ({s:.{d}f})"

# Build a clean SUMMARY of findings to print
print("=" * 70)
print("KEY NUMBERS FOR PAPER")
print("=" * 70)
print(f"\nN forests:        {ss['n_forests']}")
print(f"N forest-years:   {ss['n_obs']:,}")
print(f"Sample period:    {ss['year_min']}-{ss['year_max']}")
print(f"Disaster mean:    {ss['disaster_mean_per_forest_year']:.4f}")
print(f"Total acres trtd: {ss['total_acres_treated_M']:.1f}M")
print(f"N treatments:     {ss['n_treatment_records']:,}")
print(f"Mean $/acre:      ${ss['mean_cost_per_acre_usd']:,.0f}")
print(f"\nFEMA mean:        ~0.117")
print(f"Western forests:  {ss['n_forests_west']}")
print(f"Other forests:    {ss['n_forests_other']}")

print(f"\n--- Main coefficient estimates (log[struct destroyed + 1] outcome) ---")
print(f"Full sample, lag1:      {t1['log(Struct), lag1']['log_acres_lag1_coef']:+.4f} ({t1['log(Struct), lag1']['log_acres_lag1_se']:.4f})  p={t1['log(Struct), lag1']['log_acres_lag1_p']:.3f}")
print(f"Full sample, lag1+RxY:  {t1['log(Struct), lag1, RxY FE']['log_acres_lag1_coef']:+.4f} ({t1['log(Struct), lag1, RxY FE']['log_acres_lag1_se']:.4f})  p={t1['log(Struct), lag1, RxY FE']['log_acres_lag1_p']:.3f}")
print(f"Full sample, lag3+RxY:  {t1['log(Struct), lag3, RxY FE']['log_acres_lag3_coef']:+.4f} ({t1['log(Struct), lag3, RxY FE']['log_acres_lag3_se']:.4f})  p={t1['log(Struct), lag3, RxY FE']['log_acres_lag3_p']:.3f}")
print(f"West only, lag1:        {t2['log(Struct), lag1']['log_acres_lag1_coef']:+.4f} ({t2['log(Struct), lag1']['log_acres_lag1_se']:.4f})  p={t2['log(Struct), lag1']['log_acres_lag1_p']:.3f}")
print(f"West only, lag2+RxY:    {t2['log(Struct), lag2, RxY']['log_acres_lag2_coef']:+.4f} ({t2['log(Struct), lag2, RxY']['log_acres_lag2_se']:.4f})  p={t2['log(Struct), lag2, RxY']['log_acres_lag2_p']:.3f}")

print(f"\n--- Event study selected coefficients (log[struct destroyed]) ---")
for e in ev:
    if e["k"] in [-5,-3,-1,0,1,3,5,7,10,15] or abs(e["coef"]) > 0.3:
        print(f"  k={e['k']:+3d}: {e['coef']:+.4f} ({e['se']:.4f})  p={e['p']:.3f}{stars(e['p'])}")

# Compute event-study based CBA
# Use the average post-treatment effect
post_coefs = [e["coef"] for e in ev if e["k"] >= 4 and e["k"] <= 10]
avg_post_effect = float(np.mean(post_coefs))
print(f"\nAverage post-treatment effect (k=4..10) on log(struct dest+1): {avg_post_effect:+.4f}")
reduction = 1 - np.exp(avg_post_effect)
print(f"Implied reduction in structures destroyed: {100*reduction:.1f}%")

# Detailed CBA recompute with event-study estimate
total_struct_destroyed = ss['n_struct_destroyed']
total_fatalities       = ss['n_fatalities']
n_disasters_obs        = ss['n_disasters_observed_in_sample']  # forest-years with at least one disaster

VSL          = 11.6e6
struct_value = 350_000
suppress     = 8e6  # rough avg suppression cost per disaster event

# Counterfactual: without treatment
cf_struct = total_struct_destroyed / np.exp(avg_post_effect)
cf_fatalities = total_fatalities / np.exp(avg_post_effect)
struct_avoided = cf_struct - total_struct_destroyed
fat_avoided    = cf_fatalities - total_fatalities

benefits_struct = struct_avoided * struct_value
benefits_fat    = fat_avoided * VSL
# Suppression: assume 33% reduction in disaster-event occurrence (rough)
disasters_avoided = n_disasters_obs * (1/np.exp(avg_post_effect) - 1)
benefits_supp   = disasters_avoided * suppress
benefits_total  = benefits_struct + benefits_fat + benefits_supp

# Costs
total_acres = ss['total_acres_treated']
total_cost  = total_acres * ss['mean_cost_per_acre_usd']

bcr = benefits_total / total_cost
print(f"\n--- Updated Cost-Benefit Analysis ---")
print(f"Average post-treatment effect:    {avg_post_effect:+.4f}")
print(f"Implied % reduction structures:   {100*reduction:.1f}%")
print(f"Structures avoided:               {struct_avoided:,.0f}")
print(f"Structure value avoided:          ${benefits_struct/1e9:.2f}B")
print(f"Fatalities avoided:               {fat_avoided:.1f}")
print(f"Fatality value avoided:           ${benefits_fat/1e9:.2f}B")
print(f"Suppression avoided:              ${benefits_supp/1e9:.2f}B")
print(f"TOTAL BENEFITS:                   ${benefits_total/1e9:.2f}B")
print(f"")
print(f"Total acres treated 25y:          {total_acres/1e6:.1f}M")
print(f"Total federal cost:               ${total_cost/1e9:.2f}B")
print(f"")
print(f"BENEFIT-COST RATIO:               {bcr:.2f}")

# Save key numbers as a JSON for the paper-update script
key = {
    "n_forests":  ss['n_forests'],
    "n_obs":      ss['n_obs'],
    "year_min":   ss['year_min'],
    "year_max":   ss['year_max'],
    "n_treatment_records": ss['n_treatment_records'],
    "total_acres_treated_M": ss['total_acres_treated_M'],
    "disaster_mean": ss['disaster_mean_per_forest_year'],
    "n_disasters_obs": ss['n_disasters_observed_in_sample'],
    "n_struct_destroyed": ss['n_struct_destroyed'],
    "n_fatalities": ss['n_fatalities'],
    "treatment_share_mech_pct": 100*ss['treatment_share_mech'],
    "treatment_share_fire_pct": 100*ss['treatment_share_fire'],
    "treatment_share_wui_pct": 100*ss['treatment_share_wui'],
    "mean_cost_per_acre_usd": ss['mean_cost_per_acre_usd'],
    "n_forests_west": ss['n_forests_west'],
    "n_forests_other": ss['n_forests_other'],
    # main TWFE estimates
    "main_full_log_struct_lag1_coef":   t1['log(Struct), lag1']['log_acres_lag1_coef'],
    "main_full_log_struct_lag1_se":     t1['log(Struct), lag1']['log_acres_lag1_se'],
    "main_full_log_struct_lag1_p":      t1['log(Struct), lag1']['log_acres_lag1_p'],
    "main_full_log_struct_lag3_coef":   t1['log(Struct), lag3, RxY FE']['log_acres_lag3_coef'],
    "main_full_log_struct_lag3_se":     t1['log(Struct), lag3, RxY FE']['log_acres_lag3_se'],
    "main_full_log_struct_lag3_p":      t1['log(Struct), lag3, RxY FE']['log_acres_lag3_p'],
    "main_west_log_struct_lag1_coef":   t2['log(Struct), lag1']['log_acres_lag1_coef'],
    "main_west_log_struct_lag1_se":     t2['log(Struct), lag1']['log_acres_lag1_se'],
    "main_west_log_struct_lag1_p":      t2['log(Struct), lag1']['log_acres_lag1_p'],
    "main_west_log_struct_lag2_coef":   t2['log(Struct), lag2, RxY']['log_acres_lag2_coef'],
    "main_west_log_struct_lag2_se":     t2['log(Struct), lag2, RxY']['log_acres_lag2_se'],
    "main_west_log_struct_lag2_p":      t2['log(Struct), lag2, RxY']['log_acres_lag2_p'],
    # event study k=+5
    "ev_k5_coef":  next(e["coef"] for e in ev if e["k"]==5),
    "ev_k5_se":    next(e["se"]   for e in ev if e["k"]==5),
    "ev_k5_p":     next(e["p"]    for e in ev if e["k"]==5),
    "ev_avg_post": float(avg_post_effect),
    "implied_reduction_pct": float(100*reduction),
    # CBA
    "struct_avoided": float(struct_avoided),
    "benefits_total_B": float(benefits_total/1e9),
    "total_cost_B":    float(total_cost/1e9),
    "bcr": float(bcr),
}
with open(RES / "key_numbers.json", "w") as f:
    json.dump(key, f, indent=2)
print(f"\nSaved key numbers: {RES / 'key_numbers.json'}")
