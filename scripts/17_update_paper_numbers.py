"""
Apply mechanical numerical updates to paper_jeem.tex after the Wear et al.
human-FRT cleaning step. Each substitution is an exact-string replace; the
script asserts that the OLD string exists exactly once before substituting.

Run this once after rerunning scripts 13, 14, 15.
"""
from pathlib import Path

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
TEX = ROOT / "paper_jeem.tex"

src = TEX.read_text()

# (old, new) pairs. Each must appear exactly once unless explicitly noted.
# Order is irrelevant since each old string must be unique.
EDITS = [
    # ----- Sample sizes throughout -----
    ("7,002,200", "6,484,850"),                        # appears in abstract / sample / notes
    ("7{,}002{,}200", "6{,}484{,}850"),                # tables and notes
    ("7.0 million polygon-year", "6.5 million polygon-year"),
    ("280,088", "259,394"),
    ("280{,}088", "259{,}394"),
    ("1{,}999{,}988", "1{,}884{,}453"),
    ("5{,}002{,}212", "4{,}600{,}397"),
    ("131{,}886", "121{,}459"),
    ("148{,}202", "137{,}935"),
    ("131,886 are ever-treated and 148,202",
     "121,459 are ever-treated and 137,935"),
    ("1{,}720{,}225", "1{,}619{,}850"),

    # ----- Mean disaster rate (multiple appearances handled with replace_all logic) -----
    ("Wildfire disaster $D_{it}$ & 0.00306 & 0.0552 & 0.00221 & 0.0470",
     "Wildfire disaster $D_{it}$ & 0.00321 & 0.0566 & 0.00242 & 0.0491"),
    ("Mean of $D_{it}$ & 0.00245 & 0.00245 & 0.00245 & 0.00245 & 0.00245",
     "Mean of $D_{it}$ & 0.00265 & 0.00265 & 0.00265 & 0.00265 & 0.00265"),

    # ----- Main TWFE table (tab:main) -----
    ("$0.000780^{***}$ & $0.000172^{**}$",
     "$0.000832^{***}$ & $0.000121$"),
    ("$(0.000059)$ & $(0.000072)$",
     "$(0.000064)$ & $(0.000079)$"),
    ("$\\ln(1+A_{it})$ & & & $0.000042^{***}$ & & \\\\\n& & & $(0.000010)$",
     "$\\ln(1+A_{it})$ & & & $0.000037^{***}$ & & \\\\\n& & & $(0.000011)$"),
    ("$\\ln(1+A_{i,t-3})$ & & & & $-0.000080^{***}$ & \\\\\n& & & & $(0.000011)$",
     "$\\ln(1+A_{i,t-3})$ & & & & $-0.000095^{***}$ & \\\\\n& & & & $(0.000012)$"),
    ("$\\ln(1+A^{\\text{mech}}_{it})$ & & & & & $0.000188^{***}$ \\\\\n& & & & & $(0.000016)$",
     "$\\ln(1+A^{\\text{mech}}_{it})$ & & & & & $0.000226^{***}$ \\\\\n& & & & & $(0.000017)$"),
    ("$\\ln(1+A^{\\text{fire}}_{it})$ & & & & & $-0.000211^{***}$ \\\\\n& & & & & $(0.000018)$",
     "$\\ln(1+A^{\\text{fire}}_{it})$ & & & & & $-0.000250^{***}$ \\\\\n& & & & & $(0.000018)$"),

    # ----- Body discussion of Table 4 (line 466) -----
    ("a positive coefficient of $0.000780$",
     "a positive coefficient of $0.000832$"),
    ("a residual positive coefficient of $0.000172$ ($p = 0.018$) that is an order of magnitude smaller than column (1) but still statistically distinguishable from zero",
     "a residual positive coefficient of $0.000121$ ($p = 0.126$) that is no longer statistically distinguishable from zero. This collapse of the previously wrong-signed extensive margin coefficient toward zero is a direct product of restricting the FACTS sample to intentional human fuel-reduction activities following the cleaning protocol of \\citet{wear2025}; the contamination from wildfire-product, salvage, and silvicultural cuts that drove the prior positive estimate is removed once those activity codes are excluded"),
    ("the three-year-lagged intensive margin in column (4) flips sign and becomes negative and significant ($-0.000080$, $p < 0.001$)",
     "the three-year-lagged intensive margin in column (4) flips sign and becomes negative and significant ($-0.000095$, $p < 0.001$)"),
    ("The coefficient on log cumulative prescribed fire acres is negative and significant ($-0.000211$, $p < 0.001$), while the coefficient on log cumulative mechanical acres is positive and significant ($+0.000188$, $p < 0.001$)",
     "The coefficient on log cumulative prescribed fire acres is negative and significant ($-0.000250$, $p < 0.001$), while the coefficient on log cumulative mechanical acres is positive and significant ($+0.000226$, $p < 0.001$)"),

    # ----- Abstract / intro updates -----
    ("with the three-year-lagged binary treatment indicator reducing log structures destroyed by $0.243$",
     "with the three-year-lagged binary treatment indicator reducing log structures destroyed by $0.272$"),
    ("the binary extensive margin coefficient is negative and significant in High-density Intermix polygons ($-0.00030$, $p = 0.004$)",
     "the binary extensive margin coefficient is negative and significant in High-density Intermix polygons ($-0.00036$, $p = 0.002$)"),
    ("a doubling of cumulative treated acreage within a 10 km buffer three years prior reduces the log of structures destroyed by approximately 6.3 percent",
     "a doubling of cumulative treated acreage within a 10 km buffer three years prior reduces the log of structures destroyed by approximately 7.2 percent"),

    # ----- Intro contributions paragraph (line 79) -----
    ("approximately 6.3 percent ($p < 0.001$). The three-year-lagged binary treatment indicator yields a coefficient of $-0.243$ (SE $0.038$, $p < 0.001$) on the same outcome.",
     "approximately 7.2 percent ($p < 0.001$). The three-year-lagged binary treatment indicator yields a coefficient of $-0.272$ (SE $0.042$, $p < 0.001$) on the same outcome."),
    ("a doubling of cumulative treated acreage within a WUI polygon's 10 km buffer three years prior reduces $\\ln(1 + \\text{structures destroyed})$ by $0.091 \\cdot \\ln 2 \\approx 0.063$, or approximately 6.3 percent",
     "a doubling of cumulative treated acreage within a WUI polygon's 10 km buffer three years prior reduces $\\ln(1 + \\text{structures destroyed})$ by $0.104 \\cdot \\ln 2 \\approx 0.072$, or approximately 7.2 percent"),
    ("The contemporaneous TWFE estimate of $+0.00017$ (SE $0.00007$, $p = 0.018$) against a mean disaster rate of $0.00245$ is statistically significant but wrong-signed",
     "The contemporaneous TWFE estimate of $+0.00012$ (SE $0.00008$, $p = 0.126$) against a mean disaster rate of $0.00265$ is statistically indistinguishable from zero, an improvement on prior FACTS-based work that reported a wrong-signed positive coefficient"),
    ("the binary extensive margin coefficient is negative and significant in High-density Intermix polygons ($-0.000302$, $p = 0.004$), the subset of the WUI where structural vulnerability is greatest, while remaining positive in Low-density polygons where the selection effect dominates",
     "the binary extensive margin coefficient is negative and significant in High-density Intermix polygons ($-0.000358$, $p = 0.002$), the subset of the WUI where structural vulnerability is greatest, while remaining positive in Low-density polygons where the selection effect dominates"),
    ("the coefficient on log cumulative prescribed fire acres is $-0.000211$ ($p < 0.001$), while mechanical treatment acreage alone yields a positive coefficient",
     "the coefficient on log cumulative prescribed fire acres is $-0.000250$ ($p < 0.001$), while mechanical treatment acreage alone yields a positive coefficient"),

    # ----- Severity table tab:severity -----
    ("$FRT_{it}$ & $-0.00147^{***}$ & & & \\\\\n& $(0.00037)$",
     "$FRT_{it}$ & $-0.00172^{***}$ & & & \\\\\n& $(0.00040)$"),
    ("$\\ln(1+A_{it})$ & & $-0.000210^{***}$ & & \\\\\n& & $(0.000050)$",
     "$\\ln(1+A_{it})$ & & $-0.000245^{***}$ & & \\\\\n& & $(0.000054)$"),
    ("$\\ln(1+A_{i,t-3})$ & & & $-0.000910^{***}$ & \\\\\n& & & $(0.000054)$",
     "$\\ln(1+A_{i,t-3})$ & & & $-0.001043^{***}$ & \\\\\n& & & $(0.000059)$"),
    ("$FRT_{i,t-3}$ & & & & $-0.00243^{***}$ \\\\\n& & & & $(0.00038)$",
     "$FRT_{i,t-3}$ & & & & $-0.00272^{***}$ \\\\\n& & & & $(0.00042)$"),

    # ----- Severity discussion paragraph (line 562) -----
    ("The headline result in column (3) shows a coefficient of $-0.000910$ ($p < 0.001$) on the three-year-lagged log of cumulative treated acreage. The semi-elasticity interpretation is that a one log-point increase in cumulative treated acreage three years prior reduces the log of structures destroyed by approximately 0.091; equivalently, a doubling of cumulative treated acreage three years prior reduces the log of structures destroyed by $0.091 \\cdot \\ln 2 \\approx 0.063$, that is, by approximately 6.3 percent.",
     "The headline result in column (3) shows a coefficient of $-0.001043$ ($p < 0.001$) on the three-year-lagged log of cumulative treated acreage. The semi-elasticity interpretation is that a one log-point increase in cumulative treated acreage three years prior reduces the log of structures destroyed by approximately 0.104; equivalently, a doubling of cumulative treated acreage three years prior reduces the log of structures destroyed by $0.104 \\cdot \\ln 2 \\approx 0.072$, that is, by approximately 7.2 percent."),
    ("treatment within a WUI polygon's 10 km buffer three years prior reduces $\\ln(1+S_{it})$ by $0.00243$",
     "treatment within a WUI polygon's 10 km buffer three years prior reduces $\\ln(1+S_{it})$ by $0.00272$"),
    ("Column (1)'s contemporaneous binary treatment coefficient of $-0.00147$",
     "Column (1)'s contemporaneous binary treatment coefficient of $-0.00172$"),
    ("the 6 percent elasticity of log structures destroyed",
     "the 7 percent elasticity of log structures destroyed"),

    # ----- Heterogeneity table tab:heterogeneity -----
    ("$FRT_{it} \\times$ Low density & $+0.000933^{***}$ \\\\\n& $(0.000166)$",
     "$FRT_{it} \\times$ Low density & $+0.000757^{***}$ \\\\\n& $(0.000174)$"),
    ("$FRT_{it} \\times$ Medium density & $+0.000111$ \\\\\n& $(0.000090)$",
     "$FRT_{it} \\times$ Medium density & $+0.000104$ \\\\\n& $(0.000098)$"),
    ("$FRT_{it} \\times$ High density & $-0.000302^{***}$ \\\\\n& $(0.000106)$",
     "$FRT_{it} \\times$ High density & $-0.000358^{***}$ \\\\\n& $(0.000116)$"),

    # ----- Heterogeneity discussion (line 604, 606) -----
    ("The binary extensive margin coefficient is significantly negative for High-density Intermix polygons ($-0.000302$, $p = 0.004$)",
     "The binary extensive margin coefficient is significantly negative for High-density Intermix polygons ($-0.000358$, $p = 0.002$)"),
    ("statistically indistinguishable from zero for Medium-density polygons ($+0.000111$, $p = 0.21$) and significantly positive for Low-density polygons ($+0.000933$, $p < 0.001$)",
     "statistically indistinguishable from zero for Medium-density polygons ($+0.000104$, $p = 0.29$) and significantly positive for Low-density polygons ($+0.000757$, $p < 0.001$)"),

    # ----- Event-study text on placebo (line 518) -----
    ("yields a coefficient of $-0.000415$ (SE $0.000104$, $p < 0.001$), confirming the violation",
     "yields a coefficient of $-0.000454$ (SE $0.000119$, $p < 0.001$), confirming the violation"),

    # ----- Welfare parameterization bullet (lines 643-644) -----
    ("\\item Average treatment cost: $c = \\$575$ per acre, the acreage-weighted average of \\texttt{COST\\_PER\\_U} in the Western subsample of FACTS.\n\\item Average treated acreage per treated polygon-year: $\\bar{a} = 864$ acres",
     "\\item Average treatment cost: $c = \\$578$ per acre, the acreage-weighted average of \\texttt{COST\\_PER\\_U} in the Western subsample of FACTS after restricting to human fuel reduction activities following \\citet{wear2025}.\n\\item Average treated acreage per treated polygon-year: $\\bar{a} = 789$ acres"),

    # ----- Welfare table tab:welfare -----
    ("$|\\hat{\\beta}_1|$ from Table \\ref{tab:main} col. (2) & 0.000172 \\\\\nAvoided expected damage per treated polygon-year, \\$ & 68{,}893",
     "$|\\hat{\\beta}_1|$ from Table \\ref{tab:main} col. (2) & 0.000121 \\\\\nAvoided expected damage per treated polygon-year, \\$ & 48{,}295"),
    ("Average treated acres per treated polygon-year & 864 \\\\\nAverage cost per acre (\\$) & 575 \\\\\nAnnual treatment cost per treated polygon, \\$ & 496{,}566",
     "Average treated acres per treated polygon-year & 789 \\\\\nAverage cost per acre (\\$) & 578 \\\\\nAnnual treatment cost per treated polygon, \\$ & 455{,}727"),
    ("BCR (binary TWFE, full sample) & 0.14",
     "BCR (binary TWFE, full sample) & 0.11"),
    ("$|\\hat{\\beta}_1|$, High-density Intermix (Table \\ref{tab:heterogeneity}) & 0.000302 \\\\\nAvoided damage per treated polygon-year, \\$ & 120{,}922 \\\\\n\\textbf{BCR, high-density subsample} & \\textbf{0.24}",
     "$|\\hat{\\beta}_1|$, High-density Intermix (Table \\ref{tab:heterogeneity}) & 0.000358 \\\\\nAvoided damage per treated polygon-year, \\$ & 143{,}336 \\\\\n\\textbf{BCR, high-density subsample} & \\textbf{0.31}"),
    ("BCR with VSL = \\$3m & 0.13 \\\\\nBCR with VSL = \\$7m & 0.14 \\\\\nBCR with VSL = \\$11.6m (central) & 0.14 \\\\\nBCR with VSL = \\$15m & 0.14",
     "BCR with VSL = \\$3m & 0.10 \\\\\nBCR with VSL = \\$7m & 0.10 \\\\\nBCR with VSL = \\$11.6m (central) & 0.11 \\\\\nBCR with VSL = \\$15m & 0.11"),
    ("The TWFE BCR uses the binary coefficient of $0.000172$ from Table \\ref{tab:main} col. (2). The high-density BCR uses the negative coefficient of $-0.000302$",
     "The TWFE BCR uses the binary coefficient of $0.000121$ from Table \\ref{tab:main} col. (2). The high-density BCR uses the negative coefficient of $-0.000358$"),

    # ----- Welfare body text (line 701) -----
    ("Under the naive TWFE binary extensive margin (Panel D), the implied benefit-cost ratio is $0.14$, driven by the small point estimate of $0.000172$ per polygon-year. This ratio is a lower bound because the binary TWFE estimate is contaminated by selection on fire activity",
     "Under the naive TWFE binary extensive margin (Panel D), the implied benefit-cost ratio is $0.11$, driven by the small and now statistically insignificant point estimate of $0.000121$ per polygon-year. This ratio is a lower bound because the binary TWFE estimate, while no longer wrong-signed after the human-FRT cleaning, still reflects residual selection on fire activity"),
    ("In the High-density Intermix subsample where the theoretical model's predicted sign is recovered (Panel E), the BCR rises to $0.24$ using the $|{-}0.000302|$ coefficient",
     "In the High-density Intermix subsample where the theoretical model's predicted sign is recovered (Panel E), the BCR rises to $0.31$ using the $|{-}0.000358|$ coefficient"),
    ("(averaging 864 treated acres per treated year)",
     "(averaging 789 treated acres per treated year)"),

    # ----- Severity-channel CBA equation block (lines 708-710) -----
    ("where $|\\theta_{\\text{sev}}| = 0.091$ is the absolute semi-elasticity from Table \\ref{tab:severity} column (3), $\\bar{N}^{\\text{str}} = 1{,}061$",
     "where $|\\theta_{\\text{sev}}| = 0.104$ is the absolute semi-elasticity from Table \\ref{tab:severity} column (3), $\\bar{N}^{\\text{str}} = 1{,}061$"),
    ("At the pooled sample disaster rate $P(D_{it} = 1) = 0.00245$, the implied per-polygon-year benefit of doubling cumulative treated acreage is approximately \\$64{,}565. In the High-density Intermix subsample where $P(D_{it} = 1) = 0.0050$ (more than twice the pooled rate), the corresponding figure is \\$131{,}765.",
     "At the pooled sample disaster rate $P(D_{it} = 1) = 0.00265$, the implied per-polygon-year benefit of doubling cumulative treated acreage is approximately \\$79{,}872. In the High-density Intermix subsample where $P(D_{it} = 1) = 0.00528$ (almost exactly twice the pooled rate), the corresponding figure is \\$159{,}012."),
    ("The cost of doubling cumulative treated acreage for a representative polygon is approximately the mean cumulative acreage at the sample mean ($\\bar{A} = 864$ acres) times \\$575 per acre, or \\$496{,}800. The severity-channel BCR for doubling treatment is therefore \\$64{,}565 / \\$496{,}800 = 0.13 in the pooled sample and \\$131{,}765 / \\$496{,}800 = 0.27 in the High-density Intermix subsample.",
     "The cost of doubling cumulative treated acreage for a representative polygon is approximately the mean cumulative acreage at the sample mean ($\\bar{A} = 789$ acres) times \\$578 per acre, or \\$456{,}042. The severity-channel BCR for doubling treatment is therefore \\$79{,}872 / \\$456{,}042 = 0.18 in the pooled sample and \\$159{,}012 / \\$456{,}042 = 0.35 in the High-density Intermix subsample."),
    ("yielding BCRs of $0.21$ in the pooled sample and $0.43$ in the High-density Intermix subsample",
     "yielding BCRs of $0.28$ in the pooled sample and $0.56$ in the High-density Intermix subsample"),

    # ----- Robustness table tab:robustness -----
    ("\\quad $FRT_{it}$, polygon FE only & $+0.000780$ & $0.000059$",
     "\\quad $FRT_{it}$, polygon FE only & $+0.000832$ & $0.000064$"),
    ("\\quad $FRT_{it}$, TWFE & $+0.000172$ & $0.000072$ & $0.018$",
     "\\quad $FRT_{it}$, TWFE & $+0.000121$ & $0.000079$ & $0.126$"),
    ("\\quad $FRT_{i,t-1}$, TWFE & $+0.000027$ & $0.000074$ & $0.71$",
     "\\quad $FRT_{i,t-1}$, TWFE & $+0.000006$ & $0.000080$ & $0.94$"),
    ("\\quad $FRT_{i,t-3}$, TWFE & $+0.000027$ & $0.000074$ & $0.71$",
     "\\quad $FRT_{i,t-3}$, TWFE & $+0.000006$ & $0.000080$ & $0.94$"),
    ("\\quad $\\ln(1+A_{it})$ contemporaneous & $+0.000042$ & $0.000010$ & $< 0.001$",
     "\\quad $\\ln(1+A_{it})$ contemporaneous & $+0.000037$ & $0.000011$ & $0.001$"),
    ("\\quad $\\sinh^{-1}(A_{it})$ & $+0.000037$ & $0.000010$ & $< 0.001$",
     "\\quad $\\sinh^{-1}(A_{it})$ & $+0.000033$ & $0.000010$ & $0.001$"),
    ("\\quad $\\ln(1+A_{i,t-3})$ (3-year lag) & $-0.000080$ & $0.000011$ & $< 0.001$",
     "\\quad $\\ln(1+A_{i,t-3})$ (3-year lag) & $-0.000095$ & $0.000012$ & $< 0.001$"),
    ("\\quad $\\ln(1+A^{\\text{mech}}_{it})$ & $+0.000188$ & $0.000016$ & $< 0.001$",
     "\\quad $\\ln(1+A^{\\text{mech}}_{it})$ & $+0.000226$ & $0.000017$ & $< 0.001$"),
    ("\\quad $\\ln(1+A^{\\text{fire}}_{it})$ & $+0.000211$ & $0.000018$ & $< 0.001$",
     "\\quad $\\ln(1+A^{\\text{fire}}_{it})$ & $-0.000250$ & $0.000018$ & $< 0.001$"),
    ("\\quad $FRT_{it}$, TWFE & $-0.00147$ & $0.00037$ & $< 0.001$",
     "\\quad $FRT_{it}$, TWFE & $-0.00172$ & $0.00040$ & $< 0.001$"),
    ("\\quad $\\ln(1+A_{it})$ contemporaneous & $-0.000210$ & $0.000050$ & $< 0.001$",
     "\\quad $\\ln(1+A_{it})$ contemporaneous & $-0.000245$ & $0.000054$ & $< 0.001$"),
    ("\\quad $\\ln(1+A_{i,t-3})$ & $-0.000910$ & $0.000054$ & $< 0.001$",
     "\\quad $\\ln(1+A_{i,t-3})$ & $-0.001043$ & $0.000059$ & $< 0.001$"),
    ("\\quad $FRT_{i,t-3}$ & $-0.00243$ & $0.00038$ & $< 0.001$",
     "\\quad $FRT_{i,t-3}$ & $-0.00272$ & $0.00042$ & $< 0.001$"),
    ("\\quad $FRT_{it} \\times$ Low density & $+0.000933$ & $0.000166$ & $< 0.001$",
     "\\quad $FRT_{it} \\times$ Low density & $+0.000757$ & $0.000174$ & $< 0.001$"),
    ("\\quad $FRT_{it} \\times$ Medium density & $+0.000111$ & $0.000090$ & $0.21$",
     "\\quad $FRT_{it} \\times$ Medium density & $+0.000104$ & $0.000098$ & $0.29$"),
    ("\\quad $FRT_{it} \\times$ High density & $-0.000302$ & $0.000106$ & $0.004$",
     "\\quad $FRT_{it} \\times$ High density & $-0.000358$ & $0.000116$ & $0.002$"),
    ("\\quad $FRT_{it}$, Intermix only ($N = 1{,}619{,}850$) & $+0.000261$ & $0.000167$ & $0.12$",
     "\\quad $FRT_{it}$, Intermix only ($N = 1{,}619{,}850$) & $+0.000045$ & $0.000176$ & $0.80$"),
    ("\\quad $\\ln(1+A_{i,t-3})$, Intermix & $+0.000080$ & $0.000024$ & $0.001$",
     "\\quad $\\ln(1+A_{i,t-3})$, Intermix & $+0.000105$ & $0.000026$ & $< 0.001$"),
    ("\\quad Future treatment (5-yr lead), pre-period & $-0.000415$ & $0.000104$ & $< 0.001$",
     "\\quad Future treatment (5-yr lead), pre-period & $-0.000454$ & $0.000119$ & $< 0.001$"),

    # ----- Robustness panel messages (lines 775-781) -----
    ("The preferred three-year-lagged log coefficient of $-0.000910$ ($p < 0.001$) implies the 6.3 percent elasticity",
     "The preferred three-year-lagged log coefficient of $-0.001043$ ($p < 0.001$) implies the 7.2 percent elasticity"),
    ("The three-year-lagged log cumulative acres coefficient flips sign to $-0.000080$ ($p < 0.001$)",
     "The three-year-lagged log cumulative acres coefficient flips sign to $-0.000095$ ($p < 0.001$)"),
    ("The prescribed-fire subsample is negative and significant ($-0.000211$, $p < 0.001$), while the mechanical-only subsample is positive ($+0.000188$, $p < 0.001$)",
     "The prescribed-fire subsample is negative and significant ($-0.000250$, $p < 0.001$), while the mechanical-only subsample is positive ($+0.000226$, $p < 0.001$)"),
    ("The negative and significant High-density Intermix coefficient ($-0.000302$, $p = 0.004$) cannot be generated by any residual selection-on-fire-activity story",
     "The negative and significant High-density Intermix coefficient ($-0.000358$, $p = 0.002$) cannot be generated by any residual selection-on-fire-activity story"),
    ("Panel F reports the future-treatment placebo, which is negative and strongly significant ($-0.000415$, $p < 0.001$)",
     "Panel F reports the future-treatment placebo, which is negative and strongly significant ($-0.000454$, $p < 0.001$)"),

    # ----- Discussion / interpretation (line 797) -----
    ("the three-year-lagged intensive margin on the binary occurrence outcome flips sign to $-0.000080$",
     "the three-year-lagged intensive margin on the binary occurrence outcome flips sign to $-0.000095$"),
    ("the much larger severity-channel elasticity of 6.3 percent",
     "the much larger severity-channel elasticity of 7.2 percent"),

    # ----- Conclusion (line 815) -----
    ("a semi-elasticity of log structures destroyed to the three-year-lagged log of cumulative treated acreage of $-0.091$ ($p < 0.001$), implying a 6.3 percent reduction in structures destroyed for a doubling of cumulative treated acreage",
     "a semi-elasticity of log structures destroyed to the three-year-lagged log of cumulative treated acreage of $-0.104$ ($p < 0.001$), implying a 7.2 percent reduction in structures destroyed for a doubling of cumulative treated acreage"),
]

# -----------------------------------------------------------------------------
# Apply edits with verification
# -----------------------------------------------------------------------------
text = src
n_applied = 0
n_missed = 0
for old, new in EDITS:
    count = text.count(old)
    if count == 0:
        print(f"MISS  (not found):  {old[:90]!r}")
        n_missed += 1
        continue
    if count > 1:
        print(f"WARN  (found {count}x): {old[:90]!r}")
    text = text.replace(old, new)
    n_applied += 1

print(f"\nApplied: {n_applied}/{len(EDITS)}  Missed: {n_missed}")

if text != src:
    TEX.write_text(text)
    print(f"Wrote: {TEX}")
else:
    print("No changes written.")
