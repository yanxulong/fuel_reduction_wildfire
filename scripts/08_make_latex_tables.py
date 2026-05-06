"""
Generate LaTeX tables (saved to tables/) for inclusion in paper_final.tex.
"""
import json
from pathlib import Path

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
RES = ROOT / "results"
TBL = ROOT / "tables"
TBL.mkdir(parents=True, exist_ok=True)

with open(RES / "final_results.json") as f:
    R = json.load(f)
with open(RES / "key_numbers.json") as f:
    K = json.load(f)

ss = R["summary_stats"]
t1 = R["table1_main"]
t2 = R["table2_west"]
t3 = R["table3_type"]
ev = R["event_study"]

def stars(p):
    if p is None: return ""
    if p < 0.01: return "$^{***}$"
    if p < 0.05: return "$^{**}$"
    if p < 0.10: return "$^{*}$"
    return ""

def cell(c, s, p, d=4):
    return f"{c:+.{d}f}{stars(p)} \\\\\n        & ({s:.{d}f})"

# ============================================================================
# TABLE 1: SUMMARY STATISTICS
# ============================================================================
mech_pct = K['treatment_share_mech_pct']
fire_pct = K['treatment_share_fire_pct']
wui_pct = K['treatment_share_wui_pct']
table_summary = r"""\begin{table}[H]
\centering
\small
\begin{threeparttable}
\caption{Summary Statistics: National Forest Panel, 2000--2024}
\label{tab:summary}
\begin{tabular}{lr}
\toprule
Variable & Value \\
\midrule
\multicolumn{2}{l}{\textit{Panel A: Sample structure}} \\
\quad National Forests in panel  & """ + f"{ss['n_forests']:,}" + r""" \\
\quad Years                       & """ + f"{ss['n_years']}" + r""" \\
\quad Forest-year observations    & """ + f"{ss['n_obs']:,}" + r""" \\
\quad Western forests (Regions 1--6) & """ + f"{ss['n_forests_west']}" + r""" \\
\quad Other forests (Regions 7--10)  & """ + f"{ss['n_forests_other']}" + r""" \\
\addlinespace
\multicolumn{2}{l}{\textit{Panel B: Treatment activity (FACTS)}} \\
\quad Accomplished treatment records          & """ + f"{ss['n_treatment_records']:,}" + r""" \\
\quad Total cumulative acres treated (millions) & """ + f"{ss['total_acres_treated_M']:.1f}" + r""" \\
\quad Mechanical share (\%)        & """ + f"{mech_pct:.1f}" + r""" \\
\quad Prescribed fire share (\%)   & """ + f"{fire_pct:.1f}" + r""" \\
\quad WUI-flagged share (\%)       & """ + f"{wui_pct:.1f}" + r""" \\
\quad Mean cost per acre (\$, FACTS-weighted) & """ + f"{ss['mean_cost_per_acre_usd']:,.0f}" + r""" \\
\addlinespace
\multicolumn{2}{l}{\textit{Panel C: Disaster outcomes (WFBZ)}} \\
\quad Disaster events (forest-year level)         & """ + f"{ss['n_disasters_observed_in_sample']:,}" + r""" \\
\quad Total fire-disaster intersections           & """ + f"{ss['n_disaster_events']:,}" + r""" \\
\quad Mean disaster rate (forest-year)            & """ + f"{ss['disaster_mean_per_forest_year']:.4f}" + r""" \\
\quad Cumulative structures destroyed             & """ + f"{ss['n_struct_destroyed']:,}" + r""" \\
\quad Cumulative civilian fatalities              & """ + f"{int(ss['n_fatalities']):,}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} The unit of observation is the National Forest by year. Treatment data are from the US Forest Service Activity Tracking System (FACTS), restricted to records with \texttt{STAGE\_VALU = ``Accomplished''} and \texttt{FY\_AWARDED} between 2000 and 2024. Disaster data are from the Wildfire Burn Zone Disasters database \citep{wilner2025}, classified using fatality, structural destruction, or FMAG declaration criteria.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
(TBL / "table_summary.tex").write_text(table_summary)
print("Wrote table_summary.tex")

# ============================================================================
# TABLE 2 (label tab:main): MAIN RESULTS
# ============================================================================
def get(t, key, suffix):
    """Get a coef/se/p triplet."""
    if key not in t:
        return None, None, None
    r = t[key]
    return r[f"log_acres_lag{suffix}_coef"], r[f"log_acres_lag{suffix}_se"], r[f"log_acres_lag{suffix}_p"]

def fcell(c, s, p):
    if c is None: return "---"
    return f"{c:+.4f}{stars(p)}"

def secell(s):
    if s is None: return ""
    return f"({s:.4f})"

# Pull from t1
m1 = t1.get('Disaster, lag1');           c1,s1,p1 = m1['log_acres_lag1_coef'],m1['log_acres_lag1_se'],m1['log_acres_lag1_p']
m2 = t1.get('Disaster, lag1, RxY FE');   c2,s2,p2 = m2['log_acres_lag1_coef'],m2['log_acres_lag1_se'],m2['log_acres_lag1_p']
m3 = t1.get('log(Struct), lag1');        c3,s3,p3 = m3['log_acres_lag1_coef'],m3['log_acres_lag1_se'],m3['log_acres_lag1_p']
m4 = t1.get('log(Struct), lag1, RxY FE');c4,s4,p4 = m4['log_acres_lag1_coef'],m4['log_acres_lag1_se'],m4['log_acres_lag1_p']
m5 = t1.get('log(Struct), lag2, RxY FE');c5,s5,p5 = m5['log_acres_lag2_coef'],m5['log_acres_lag2_se'],m5['log_acres_lag2_p']
m6 = t1.get('log(Struct), lag3, RxY FE');c6,s6,p6 = m6['log_acres_lag3_coef'],m6['log_acres_lag3_se'],m6['log_acres_lag3_p']
m7 = t1.get('log(Burn area), lag1, RxY FE'); c7,s7,p7 = m7['log_acres_lag1_coef'],m7['log_acres_lag1_se'],m7['log_acres_lag1_p']
m8 = t1.get('FEMA, lag1, RxY FE');       c8,s8,p8 = m8['log_acres_lag1_coef'],m8['log_acres_lag1_se'],m8['log_acres_lag1_p']

table_main = r"""\begin{table}[H]
\centering
\small
\begin{threeparttable}
\caption{Main Results: Effect of Cumulative Fuel Reduction Treatments on Forest-Year Disaster Outcomes}
\label{tab:main}
\begin{tabular}{lcccccccc}
\toprule
& (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) \\
& Disaster & Disaster & log(Struct) & log(Struct) & log(Struct) & log(Struct) & log(Burn) & FEMA \\
& lag1 & lag1 & lag1 & lag1 & lag2 & lag3 & lag1 & lag1 \\
\midrule
$\ln(1+A_{i,t-k})$ &
""" + f"{fcell(c1,s1,p1)}" + r""" &
""" + f"{fcell(c2,s2,p2)}" + r""" &
""" + f"{fcell(c3,s3,p3)}" + r""" &
""" + f"{fcell(c4,s4,p4)}" + r""" &
""" + f"{fcell(c5,s5,p5)}" + r""" &
""" + f"{fcell(c6,s6,p6)}" + r""" &
""" + f"{fcell(c7,s7,p7)}" + r""" &
""" + f"{fcell(c8,s8,p8)}" + r""" \\
&
""" + f"{secell(s1)}" + r""" &
""" + f"{secell(s2)}" + r""" &
""" + f"{secell(s3)}" + r""" &
""" + f"{secell(s4)}" + r""" &
""" + f"{secell(s5)}" + r""" &
""" + f"{secell(s6)}" + r""" &
""" + f"{secell(s7)}" + r""" &
""" + f"{secell(s8)}" + r""" \\
\midrule
Forest FE       & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes \\
Year FE         & Yes & No  & Yes & No  & No  & No  & No  & No  \\
Region $\times$ Year FE & No  & Yes & No  & Yes & Yes & Yes & Yes & Yes \\
Mean of $Y_{it}$ & 0.266 & 0.266 & 0.722 & 0.722 & 0.722 & 0.722 & 0.622 & 0.117 \\
Observations    & 2{,}640 & 2{,}640 & 2{,}640 & 2{,}640 & 2{,}530 & 2{,}420 & 2{,}640 & 2{,}640 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} The dependent variable is indicated in the column header: ``Disaster'' is a binary indicator equal to one if any wildfire burn zone disaster intersected the forest in year $t$; ``log(Struct)'' is $\ln(1 + \text{structures destroyed in year } t)$; ``log(Burn)'' is $\ln(1 + \text{burn area in km}^2)$; ``FEMA'' is a binary indicator for any FMAG declaration in year $t$. The independent variable is the natural log of cumulative acres treated within the forest through year $t-k$, where $k$ is indicated in the column header. All specifications include forest fixed effects. Year and Region-by-Year fixed effects are included as indicated. The Region-by-Year specification absorbs all annual shocks at the USFS administrative region level. Standard errors clustered at the forest level are in parentheses. * $p<0.10$, ** $p<0.05$, *** $p<0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
(TBL / "table_main.tex").write_text(table_main)
print("Wrote table_main.tex")

# ============================================================================
# TABLE 3 (label tab:west): WESTERN SUBSAMPLE
# ============================================================================
w1 = t2.get('Disaster, lag1');               wc1,ws1,wp1 = w1['log_acres_lag1_coef'],w1['log_acres_lag1_se'],w1['log_acres_lag1_p']
w2 = t2.get('Disaster, lag1, RxY');          wc2,ws2,wp2 = w2['log_acres_lag1_coef'],w2['log_acres_lag1_se'],w2['log_acres_lag1_p']
w3 = t2.get('log(Struct), lag1');            wc3,ws3,wp3 = w3['log_acres_lag1_coef'],w3['log_acres_lag1_se'],w3['log_acres_lag1_p']
w4 = t2.get('log(Struct), lag1, RxY');       wc4,ws4,wp4 = w4['log_acres_lag1_coef'],w4['log_acres_lag1_se'],w4['log_acres_lag1_p']
w5 = t2.get('log(Struct), lag2, RxY');       wc5,ws5,wp5 = w5['log_acres_lag2_coef'],w5['log_acres_lag2_se'],w5['log_acres_lag2_p']
w6 = t2.get('FEMA, lag1, RxY');              wc6,ws6,wp6 = w6['log_acres_lag1_coef'],w6['log_acres_lag1_se'],w6['log_acres_lag1_p']

table_west = r"""\begin{table}[H]
\centering
\small
\begin{threeparttable}
\caption{Western Forests Subsample}
\label{tab:west}
\begin{tabular}{lcccccc}
\toprule
& (1) & (2) & (3) & (4) & (5) & (6) \\
& Disaster & Disaster & log(Struct) & log(Struct) & log(Struct) & FEMA \\
& lag1 & lag1 & lag1 & lag1 & lag2 & lag1 \\
\midrule
$\ln(1+A_{i,t-k})$ &
""" + f"{fcell(wc1,ws1,wp1)}" + r""" &
""" + f"{fcell(wc2,ws2,wp2)}" + r""" &
""" + f"{fcell(wc3,ws3,wp3)}" + r""" &
""" + f"{fcell(wc4,ws4,wp4)}" + r""" &
""" + f"{fcell(wc5,ws5,wp5)}" + r""" &
""" + f"{fcell(wc6,ws6,wp6)}" + r""" \\
&
""" + f"{secell(ws1)}" + r""" &
""" + f"{secell(ws2)}" + r""" &
""" + f"{secell(ws3)}" + r""" &
""" + f"{secell(ws4)}" + r""" &
""" + f"{secell(ws5)}" + r""" &
""" + f"{secell(ws6)}" + r""" \\
\midrule
Forest FE       & Yes & Yes & Yes & Yes & Yes & Yes \\
Year FE         & Yes & No  & Yes & No  & No  & No  \\
Region $\times$ Year FE & No & Yes & No & Yes & Yes & Yes \\
Mean of $Y_{it}$ & 0.310 & 0.310 & 0.870 & 0.870 & 0.870 & 0.139 \\
Observations    & 1{,}896 & 1{,}896 & 1{,}896 & 1{,}896 & 1{,}817 & 1{,}896 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Sample restricted to the 79 National Forests located in USFS Regions 1--6 (Northern, Rocky Mountain, Southwestern, Intermountain, Pacific Southwest, Pacific Northwest). Mean disaster rate in the Western subsample (0.310) is higher than in the full sample (0.266). All other notes follow Table \ref{tab:main}.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
(TBL / "table_west.tex").write_text(table_west)
print("Wrote table_west.tex")

# ============================================================================
# TABLE 4 (label tab:type): TREATMENT TYPE DECOMPOSITION
# ============================================================================
def gtype(t, key):
    if key not in t:
        return None
    r = t[key]
    return {
        "mc": r['log_mech_lag1_coef'], "ms": r['log_mech_lag1_se'], "mp": r['log_mech_lag1_p'],
        "fc": r['log_fire_lag1_coef'], "fs": r['log_fire_lag1_se'], "fp": r['log_fire_lag1_p'],
        "n":  r['n_obs'],
    }

ty1 = gtype(t3, 'Disaster, mech+fire lag1, RxY')
ty2 = gtype(t3, 'logStruct, mech+fire lag1, RxY')
ty3 = gtype(t3, 'West Disaster, mech+fire lag1, RxY')
ty4 = gtype(t3, 'West logStruct, mech+fire lag1, RxY')

def cell_mech(d):
    if d is None: return "---", "---"
    return f"{d['mc']:+.4f}{stars(d['mp'])}", f"({d['ms']:.4f})"
def cell_fire(d):
    if d is None: return "---", "---"
    return f"{d['fc']:+.4f}{stars(d['fp'])}", f"({d['fs']:.4f})"

m1c, m1s = cell_mech(ty1); f1c, f1s = cell_fire(ty1)
m2c, m2s = cell_mech(ty2); f2c, f2s = cell_fire(ty2)
m3c, m3s = cell_mech(ty3); f3c, f3s = cell_fire(ty3)
m4c, m4s = cell_mech(ty4); f4c, f4s = cell_fire(ty4)

table_type = r"""\begin{table}[H]
\centering
\small
\begin{threeparttable}
\caption{Treatment Type Decomposition: Mechanical vs. Prescribed Fire}
\label{tab:type}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Full sample} & \multicolumn{2}{c}{Western forests} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& Disaster & log(Struct) & Disaster & log(Struct) \\
& (1) & (2) & (3) & (4) \\
\midrule
$\ln(1+A^{\text{mech}}_{i,t-1})$ &
""" + m1c + " & " + m2c + " & " + m3c + " & " + m4c + r""" \\
&
""" + m1s + " & " + m2s + " & " + m3s + " & " + m4s + r""" \\
$\ln(1+A^{\text{fire}}_{i,t-1})$ &
""" + f1c + " & " + f2c + " & " + f3c + " & " + f4c + r""" \\
&
""" + f1s + " & " + f2s + " & " + f3s + " & " + f4s + r""" \\
\midrule
Forest FE         & Yes & Yes & Yes & Yes \\
Region $\times$ Year FE & Yes & Yes & Yes & Yes \\
Observations      & """ + f"{ty1['n']:,}" + " & " + f"{ty2['n']:,}" + " & " + f"{ty3['n']:,}" + " & " + f"{ty4['n']:,}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} $A^{\text{mech}}_{i,t-1}$ is cumulative acres treated by mechanical methods through year $t-1$ (FACTS \texttt{CAT\_NM = ``Mechanical''}). $A^{\text{fire}}_{i,t-1}$ is cumulative acres treated by prescribed fire (FACTS \texttt{CAT\_NM = ``Fire''}). Standard errors clustered at the forest level. * $p<0.10$, ** $p<0.05$, *** $p<0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
(TBL / "table_type.tex").write_text(table_type)
print("Wrote table_type.tex")

# ============================================================================
# TABLE 5 (label tab:welfare): WELFARE / CBA
# ============================================================================
table_welfare = r"""\begin{table}[H]
\centering
\small
\begin{threeparttable}
\caption{Welfare Analysis: Benefits and Costs of Federal Fuel Reduction Treatments, 2000--2024}
\label{tab:welfare}
\begin{tabular}{lr}
\toprule
Component & Value \\
\midrule
\multicolumn{2}{l}{\textit{Panel A: Treatment-effect parameter}} \\
\quad Average post-treatment effect on $\ln(1+\text{struct})$ ($k=4..10$) & """ + f"{K['ev_avg_post']:+.4f}" + r""" \\
\quad Implied counterfactual ratio                                          & """ + f"{1/0.732:.3f}" + r""" \\
\quad Implied reduction in structures destroyed                             & """ + f"{K['implied_reduction_pct']:.1f}\\%" + r""" \\
\addlinespace
\multicolumn{2}{l}{\textit{Panel B: Avoided social damages (benefits)}} \\
\quad Observed structures destroyed (2000--2024)                           & """ + f"{ss['n_struct_destroyed']:,}" + r""" \\
\quad Counterfactual structures destroyed                                  & """ + f"{ss['n_struct_destroyed']+int(K['struct_avoided']):,}" + r""" \\
\quad Structures avoided                                                   & """ + f"{int(K['struct_avoided']):,}" + r""" \\
\quad Avoided property damage (\$B), at \$350K/structure                   & """ + r"7.81 \\" + r"""
\quad Observed civilian fatalities                                         & """ + f"{int(ss['n_fatalities'])}" + r""" \\
\quad Avoided fatalities                                                   & """ + r"60.7 \\" + r"""
\quad Avoided fatality damages (\$B), at VSL = \$11.6M                     & """ + r"0.70 \\" + r"""
\quad Avoided suppression cost (\$B), at \$8M/event                        & """ + r"2.14 \\" + r"""
\quad \textbf{Total benefits (\$B)}                                        & \textbf{""" + f"{K['benefits_total_B']:.2f}" + r"""} \\
\addlinespace
\multicolumn{2}{l}{\textit{Panel C: Treatment costs}} \\
\quad Total accomplished treatment activities                              & """ + f"{ss['n_treatment_records']:,}" + r""" \\
\quad Total cumulative acres treated                                       & """ + f"{ss['total_acres_treated_M']:.1f}M" + r""" \\
\quad Acres-weighted average cost per acre (\$, FACTS)                     & """ + f"{ss['mean_cost_per_acre_usd']:,.0f}" + r""" \\
\quad \textbf{Total federal cost (\$B)}                                    & \textbf{""" + f"{K['total_cost_B']:.2f}" + r"""} \\
\addlinespace
\multicolumn{2}{l}{\textit{Panel D: Welfare summary}} \\
\quad Net present value (\$B, point estimate)                              & """ + f"{K['benefits_total_B'] - K['total_cost_B']:+.2f}" + r""" \\
\quad \textbf{Benefit-cost ratio}                                          & \textbf{""" + f"{K['bcr']:.2f}" + r"""} \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Treatment effect derived from the average event-study coefficient over post-treatment years $k=4..10$ in the log(structures destroyed + 1) specification with forest and Region-by-Year fixed effects. The implied reduction is $1 - \exp(-0.312) = 0.268$. Benefit valuations follow standard environmental economics conventions: VSL of \$11.6 million from US Department of Transportation 2024 guidance; structure value of \$350{,}000 from approximate national median home value; suppression cost of \$8 million per averted disaster event (conservative). Treatment costs computed from FACTS \texttt{COST\_PER\_U} weighted by treated acres. The aggregate benefit-cost ratio of 0.52 should be interpreted in light of the substantial limitations discussed in Section \ref{sec:welfare}: cost-side overstatement (failure to net out timber revenue and ecological co-benefits), benefit-side understatement (omission of spillover damages to adjacent state and private lands), and aggregation across heterogeneous forests with varying treatment effectiveness.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
(TBL / "table_welfare.tex").write_text(table_welfare)
print("Wrote table_welfare.tex")

print("\nAll tables generated. Files in:", TBL)
print("Run: pdflatex paper_final.tex")
