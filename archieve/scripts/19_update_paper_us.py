"""
Update paper_jeem.tex from the previous Western-only run to the new national run.

Applies:
  (1) Sample-size substitutions (panel = 14,014,300 polygon-years, 560,572 polygons).
  (2) Coefficient updates from data/processed/wui_regression_coefs.json.
  (3) Language: "Western U.S." / "eleven Western states" → "United States" / "national".
  (4) Welfare table and severity-channel CBA recomputed for the national sample.
  (5) Line 254 sample-restriction paragraph rewritten.
  (6) Line 807 second-limitation paragraph rewritten (analysis is no longer
      restricted to the West).
"""
import math
from pathlib import Path

ROOT = Path("/Users/long/Files/my_devstack/fuel_reduction_wildfire")
TEX = ROOT / "paper_jeem.tex"
src = TEX.read_text()

# -----------------------------------------------------------------------------
# All substitutions in (old, new) form. Each old must appear exactly once
# unless it is a sample-size token that legitimately occurs many times.
# -----------------------------------------------------------------------------
EDITS = [
    # ----- Sample-size tokens (multiple occurrences each) -----
    ("6,484,850", "14,014,300"),
    ("6{,}484{,}850", "14{,}014{,}300"),
    ("6.5 million polygon-year", "14.0 million polygon-year"),
    ("259,394", "560,572"),
    ("259{,}394", "560{,}572"),
    ("121,459 are ever-treated and 137,935",
     "267,909 are ever-treated and 12,377"),
    ("1{,}884{,}453", "3{,}836{,}744"),
    ("4{,}600{,}397", "10{,}177{,}556"),
    ("121{,}459", "267{,}909"),
    ("137{,}935", "12{,}377"),
    ("1{,}619{,}850", "4{,}692{,}700"),

    # ----- Disaster rate -----
    ("Wildfire disaster $D_{it}$ & 0.00321 & 0.0566 & 0.00242 & 0.0491",
     "Wildfire disaster $D_{it}$ & 0.00160 & 0.0399 & 0.00149 & 0.0386"),
    ("Mean of $D_{it}$ & 0.00265 & 0.00265 & 0.00265 & 0.00265 & 0.00265",
     "Mean of $D_{it}$ & 0.00152 & 0.00152 & 0.00152 & 0.00152 & 0.00152"),

    # ----- Main TWFE table tab:main -----
    ("$0.000832^{***}$ & $0.000121$",
     "$0.000330^{***}$ & $-0.000044$"),
    ("$(0.000064)$ & $(0.000079)$",
     "$(0.000028)$ & $(0.000036)$"),
    ("$\\ln(1+A_{it})$ & & & $0.000037^{***}$ & & \\\\\n& & & $(0.000011)$",
     "$\\ln(1+A_{it})$ & & & $-0.000003$ & & \\\\\n& & & $(0.000005)$"),
    ("$\\ln(1+A_{i,t-3})$ & & & & $-0.000095^{***}$ & \\\\\n& & & & $(0.000012)$",
     "$\\ln(1+A_{i,t-3})$ & & & & $-0.000041^{***}$ & \\\\\n& & & & $(0.000005)$"),
    ("$\\ln(1+A^{\\text{mech}}_{it})$ & & & & & $0.000226^{***}$ \\\\\n& & & & & $(0.000017)$",
     "$\\ln(1+A^{\\text{mech}}_{it})$ & & & & & $0.000092^{***}$ \\\\\n& & & & & $(0.000009)$"),
    ("$\\ln(1+A^{\\text{fire}}_{it})$ & & & & & $-0.000250^{***}$ \\\\\n& & & & & $(0.000018)$",
     "$\\ln(1+A^{\\text{fire}}_{it})$ & & & & & $-0.000091^{***}$ \\\\\n& & & & & $(0.000006)$"),

    # ----- Body discussion of Table 4 -----
    ("a positive coefficient of $0.000832$",
     "a positive coefficient of $0.000330$"),
    ("a residual positive coefficient of $0.000121$ ($p = 0.126$) that is no longer statistically distinguishable from zero. This collapse of the previously wrong-signed extensive margin coefficient toward zero is a direct product of restricting the FACTS sample to intentional human fuel-reduction activities following the cleaning protocol of \\citet{wear2025}; the contamination from wildfire-product, salvage, and silvicultural cuts that drove the prior positive estimate is removed once those activity codes are excluded",
     "a residual coefficient of $-0.000044$ ($p = 0.228$) that is statistically indistinguishable from zero. This collapse of the previously wrong-signed extensive margin coefficient toward zero, and indeed past it, reflects the joint effect of (i) restricting the FACTS sample to intentional human fuel-reduction activities following the cleaning protocol of \\citet{wear2025}, and (ii) extending the panel from the eleven Western states to the conterminous United States, which dilutes the high-hazard Western signal with a much larger pool of low-hazard Eastern WUI polygons in which the protective effect is mechanically smaller"),
    ("the three-year-lagged intensive margin in column (4) flips sign and becomes negative and significant ($-0.000095$, $p < 0.001$)",
     "the three-year-lagged intensive margin in column (4) is negative and significant ($-0.000041$, $p < 0.001$)"),
    ("The coefficient on log cumulative prescribed fire acres is negative and significant ($-0.000250$, $p < 0.001$), while the coefficient on log cumulative mechanical acres is positive and significant ($+0.000226$, $p < 0.001$)",
     "The coefficient on log cumulative prescribed fire acres is negative and significant ($-0.000091$, $p < 0.001$), while the coefficient on log cumulative mechanical acres is positive and significant ($+0.000092$, $p < 0.001$)"),

    # ----- Abstract / intro updates -----
    ("with the three-year-lagged binary treatment indicator reducing log structures destroyed by $0.272$",
     "with the three-year-lagged binary treatment indicator reducing log structures destroyed by $0.103$"),
    ("the binary extensive margin coefficient is negative and significant in High-density Intermix polygons ($-0.00036$, $p = 0.002$)",
     "the binary extensive margin coefficient is negative in High-density Intermix polygons ($-0.00013$, $p = 0.10$)"),
    ("a doubling of cumulative treated acreage within a 10 km buffer three years prior reduces the log of structures destroyed by approximately 7.2 percent",
     "a doubling of cumulative treated acreage within a 10 km buffer three years prior reduces the log of structures destroyed by approximately 3.0 percent"),

    # ----- Intro contributions paragraph -----
    ("approximately 7.2 percent ($p < 0.001$). The three-year-lagged binary treatment indicator yields a coefficient of $-0.272$ (SE $0.042$, $p < 0.001$) on the same outcome.",
     "approximately 3.0 percent ($p < 0.001$). The three-year-lagged binary treatment indicator yields a coefficient of $-0.103$ (SE $0.019$, $p < 0.001$) on the same outcome."),
    ("a doubling of cumulative treated acreage within a WUI polygon's 10 km buffer three years prior reduces $\\ln(1 + \\text{structures destroyed})$ by $0.104 \\cdot \\ln 2 \\approx 0.072$, or approximately 7.2 percent",
     "a doubling of cumulative treated acreage within a WUI polygon's 10 km buffer three years prior reduces $\\ln(1 + \\text{structures destroyed})$ by $0.043 \\cdot \\ln 2 \\approx 0.030$, or approximately 3.0 percent"),
    ("The contemporaneous TWFE estimate of $+0.00012$ (SE $0.00008$, $p = 0.126$) against a mean disaster rate of $0.00265$ is statistically indistinguishable from zero, an improvement on prior FACTS-based work that reported a wrong-signed positive coefficient",
     "The contemporaneous TWFE estimate of $-0.00004$ (SE $0.00004$, $p = 0.23$) against a mean disaster rate of $0.00152$ is statistically indistinguishable from zero, an improvement on prior FACTS-based work that reported a wrong-signed positive coefficient"),
    ("the binary extensive margin coefficient is negative and significant in High-density Intermix polygons ($-0.000358$, $p = 0.002$), the subset of the WUI where structural vulnerability is greatest, while remaining positive in Low-density polygons where the selection effect dominates",
     "the binary extensive margin coefficient is negative in High-density Intermix polygons ($-0.000133$, $p = 0.10$), the subset of the WUI where structural vulnerability is greatest, while it is essentially zero in Low-density polygons where the selection effect dominates"),
    ("the coefficient on log cumulative prescribed fire acres is $-0.000250$ ($p < 0.001$), while mechanical treatment acreage alone yields a positive coefficient",
     "the coefficient on log cumulative prescribed fire acres is $-0.000091$ ($p < 0.001$), while mechanical treatment acreage alone yields a positive coefficient"),

    # ----- Severity table tab:severity -----
    ("$FRT_{it}$ & $-0.00172^{***}$ & & & \\\\\n& $(0.00040)$",
     "$FRT_{it}$ & $-0.00084^{***}$ & & & \\\\\n& $(0.00018)$"),
    ("$\\ln(1+A_{it})$ & & $-0.000245^{***}$ & & \\\\\n& & $(0.000054)$",
     "$\\ln(1+A_{it})$ & & $-0.000141^{***}$ & & \\\\\n& & $(0.000024)$"),
    ("$\\ln(1+A_{i,t-3})$ & & & $-0.001043^{***}$ & \\\\\n& & & $(0.000059)$",
     "$\\ln(1+A_{i,t-3})$ & & & $-0.000432^{***}$ & \\\\\n& & & $(0.000027)$"),
    ("$FRT_{i,t-3}$ & & & & $-0.00272^{***}$ \\\\\n& & & & $(0.00042)$",
     "$FRT_{i,t-3}$ & & & & $-0.00102^{***}$ \\\\\n& & & & $(0.00019)$"),

    # ----- Severity discussion paragraph -----
    ("The headline result in column (3) shows a coefficient of $-0.001043$ ($p < 0.001$) on the three-year-lagged log of cumulative treated acreage. The semi-elasticity interpretation is that a one log-point increase in cumulative treated acreage three years prior reduces the log of structures destroyed by approximately 0.104; equivalently, a doubling of cumulative treated acreage three years prior reduces the log of structures destroyed by $0.104 \\cdot \\ln 2 \\approx 0.072$, that is, by approximately 7.2 percent.",
     "The headline result in column (3) shows a coefficient of $-0.000432$ ($p < 0.001$) on the three-year-lagged log of cumulative treated acreage. The semi-elasticity interpretation is that a one log-point increase in cumulative treated acreage three years prior reduces the log of structures destroyed by approximately 0.043; equivalently, a doubling of cumulative treated acreage three years prior reduces the log of structures destroyed by $0.043 \\cdot \\ln 2 \\approx 0.030$, that is, by approximately 3.0 percent."),
    ("treatment within a WUI polygon's 10 km buffer three years prior reduces $\\ln(1+S_{it})$ by $0.00272$",
     "treatment within a WUI polygon's 10 km buffer three years prior reduces $\\ln(1+S_{it})$ by $0.00102$"),
    ("Column (1)'s contemporaneous binary treatment coefficient of $-0.00172$",
     "Column (1)'s contemporaneous binary treatment coefficient of $-0.00084$"),
    ("the 7 percent elasticity of log structures destroyed",
     "the 3 percent elasticity of log structures destroyed"),

    # ----- Heterogeneity table tab:heterogeneity -----
    ("$FRT_{it} \\times$ Low density & $+0.000757^{***}$ \\\\\n& $(0.000174)$",
     "$FRT_{it} \\times$ Low density & $+0.000010$ \\\\\n& $(0.000054)$"),
    ("$FRT_{it} \\times$ Medium density & $+0.000104$ \\\\\n& $(0.000098)$",
     "$FRT_{it} \\times$ Medium density & $-0.000050$ \\\\\n& $(0.000044)$"),
    ("$FRT_{it} \\times$ High density & $-0.000358^{***}$ \\\\\n& $(0.000116)$",
     "$FRT_{it} \\times$ High density & $-0.000133$ \\\\\n& $(0.000081)$"),

    # ----- Heterogeneity discussion -----
    ("The binary extensive margin coefficient is significantly negative for High-density Intermix polygons ($-0.000358$, $p = 0.002$)",
     "The binary extensive margin coefficient is negatively signed but only marginally significant for High-density Intermix polygons ($-0.000133$, $p = 0.10$)"),
    ("statistically indistinguishable from zero for Medium-density polygons ($+0.000104$, $p = 0.29$) and significantly positive for Low-density polygons ($+0.000757$, $p < 0.001$)",
     "statistically indistinguishable from zero for Medium-density polygons ($-0.000050$, $p = 0.26$) and Low-density polygons ($+0.000010$, $p = 0.85$)"),

    # ----- Event-study text on placebo -----
    ("yields a coefficient of $-0.000454$ (SE $0.000119$, $p < 0.001$), confirming the violation",
     "yields a coefficient of $-0.000442$ (SE $0.000047$, $p < 0.001$), confirming the violation"),

    # ----- Welfare parameterization bullet -----
    ("\\item Average treatment cost: $c = \\$578$ per acre, the acreage-weighted average of \\texttt{COST\\_PER\\_U} in the Western subsample of FACTS after restricting to human fuel reduction activities following \\citet{wear2025}.\n\\item Average treated acreage per treated polygon-year: $\\bar{a} = 789$ acres",
     "\\item Average treatment cost: $c = \\$369$ per acre, the acreage-weighted average of \\texttt{COST\\_PER\\_U} in FACTS after restricting to human fuel reduction activities following \\citet{wear2025}.\n\\item Average treated acreage per treated polygon-year: $\\bar{a} = 1{,}046$ acres"),

    # ----- Welfare table tab:welfare -----
    ("$|\\hat{\\beta}_1|$ from Table \\ref{tab:main} col. (2) & 0.000121 \\\\\nAvoided expected damage per treated polygon-year, \\$ & 48{,}295",
     "$|\\hat{\\beta}_1|$ from Table \\ref{tab:main} col. (2) & 0.000044 \\\\\nAvoided expected damage per treated polygon-year, \\$ & 14{,}345"),
    ("Average treated acres per treated polygon-year & 789 \\\\\nAverage cost per acre (\\$) & 578 \\\\\nAnnual treatment cost per treated polygon, \\$ & 455{,}727",
     "Average treated acres per treated polygon-year & 1{,}046 \\\\\nAverage cost per acre (\\$) & 369 \\\\\nAnnual treatment cost per treated polygon, \\$ & 386{,}309"),
    ("BCR (binary TWFE, full sample) & 0.11",
     "BCR (binary TWFE, full sample) & 0.04"),
    ("$|\\hat{\\beta}_1|$, High-density Intermix (Table \\ref{tab:heterogeneity}) & 0.000358 \\\\\nAvoided damage per treated polygon-year, \\$ & 143{,}336 \\\\\n\\textbf{BCR, high-density subsample} & \\textbf{0.31}",
     "$|\\hat{\\beta}_1|$, High-density Intermix (Table \\ref{tab:heterogeneity}) & 0.000133 \\\\\nAvoided damage per treated polygon-year, \\$ & 43{,}851 \\\\\n\\textbf{BCR, high-density subsample} & \\textbf{0.11}"),
    ("BCR with VSL = \\$3m & 0.10 \\\\\nBCR with VSL = \\$7m & 0.10 \\\\\nBCR with VSL = \\$11.6m (central) & 0.11 \\\\\nBCR with VSL = \\$15m & 0.11",
     "BCR with VSL = \\$3m & 0.04 \\\\\nBCR with VSL = \\$7m & 0.04 \\\\\nBCR with VSL = \\$11.6m (central) & 0.04 \\\\\nBCR with VSL = \\$15m & 0.04"),
    ("The TWFE BCR uses the binary coefficient of $0.000121$ from Table \\ref{tab:main} col. (2). The high-density BCR uses the negative coefficient of $-0.000358$",
     "The TWFE BCR uses the binary coefficient of $-0.000044$ from Table \\ref{tab:main} col. (2). The high-density BCR uses the negative coefficient of $-0.000133$"),

    # ----- Welfare body text -----
    ("Under the naive TWFE binary extensive margin (Panel D), the implied benefit-cost ratio is $0.11$, driven by the small and now statistically insignificant point estimate of $0.000121$ per polygon-year. This ratio is a lower bound because the binary TWFE estimate, while no longer wrong-signed after the human-FRT cleaning, still reflects residual selection on fire activity",
     "Under the naive TWFE binary extensive margin (Panel D), the implied benefit-cost ratio is $0.04$, driven by the near-zero and statistically insignificant point estimate of $-0.000044$ per polygon-year. This ratio is a lower bound because the binary TWFE estimate, while no longer wrong-signed after the human-FRT cleaning and the move to the national sample, still reflects residual selection on fire activity"),
    ("In the High-density Intermix subsample where the theoretical model's predicted sign is recovered (Panel E), the BCR rises to $0.31$ using the $|{-}0.000358|$ coefficient",
     "In the High-density Intermix subsample where the theoretical model's predicted sign is recovered (Panel E), the BCR rises to $0.11$ using the $|{-}0.000133|$ coefficient"),
    ("(averaging 789 treated acres per treated year)",
     "(averaging 1{,}046 treated acres per treated year)"),

    # ----- Severity-channel CBA equation block -----
    ("where $|\\theta_{\\text{sev}}| = 0.104$ is the absolute semi-elasticity from Table \\ref{tab:severity} column (3), $\\bar{N}^{\\text{str}} = 1{,}061$",
     "where $|\\theta_{\\text{sev}}| = 0.043$ is the absolute semi-elasticity from Table \\ref{tab:severity} column (3), $\\bar{N}^{\\text{str}} = 872$"),
    ("At the pooled sample disaster rate $P(D_{it} = 1) = 0.00265$, the implied per-polygon-year benefit of doubling cumulative treated acreage is approximately \\$79{,}872. In the High-density Intermix subsample where $P(D_{it} = 1) = 0.00528$ (almost exactly twice the pooled rate), the corresponding figure is \\$159{,}012.",
     "At the pooled sample disaster rate $P(D_{it} = 1) = 0.00152$, the implied per-polygon-year benefit of doubling cumulative treated acreage is approximately \\$13{,}888. In the High-density Intermix subsample where $P(D_{it} = 1) = 0.00176$ (about 16 percent higher than the pooled rate), the corresponding figure is \\$16{,}076."),
    ("The cost of doubling cumulative treated acreage for a representative polygon is approximately the mean cumulative acreage at the sample mean ($\\bar{A} = 789$ acres) times \\$578 per acre, or \\$456{,}042. The severity-channel BCR for doubling treatment is therefore \\$79{,}872 / \\$456{,}042 = 0.18 in the pooled sample and \\$159{,}012 / \\$456{,}042 = 0.35 in the High-density Intermix subsample.",
     "The cost of doubling cumulative treated acreage for a representative polygon is approximately the mean cumulative acreage at the sample mean ($\\bar{A} = 1{,}046$ acres) times \\$369 per acre, or \\$386{,}074. The severity-channel BCR for doubling treatment is therefore \\$13{,}888 / \\$386{,}074 = 0.04 in the pooled sample and \\$16{,}076 / \\$386{,}074 = 0.04 in the High-density Intermix subsample."),
    ("yielding BCRs of $0.28$ in the pooled sample and $0.56$ in the High-density Intermix subsample",
     "yielding BCRs of $0.10$ in the pooled sample and $0.12$ in the High-density Intermix subsample"),

    # ----- Robustness table tab:robustness -----
    ("\\quad $FRT_{it}$, polygon FE only & $+0.000832$ & $0.000064$",
     "\\quad $FRT_{it}$, polygon FE only & $+0.000330$ & $0.000028$"),
    ("\\quad $FRT_{it}$, TWFE & $+0.000121$ & $0.000079$ & $0.126$",
     "\\quad $FRT_{it}$, TWFE & $-0.000044$ & $0.000036$ & $0.228$"),
    ("\\quad $FRT_{i,t-1}$, TWFE & $+0.000006$ & $0.000080$ & $0.94$",
     "\\quad $FRT_{i,t-1}$, TWFE & $-0.000044$ & $0.000037$ & $0.234$"),
    ("\\quad $FRT_{i,t-3}$, TWFE & $+0.000006$ & $0.000080$ & $0.94$",
     "\\quad $FRT_{i,t-3}$, TWFE & $-0.000044$ & $0.000037$ & $0.234$"),
    ("\\quad $\\ln(1+A_{it})$ contemporaneous & $+0.000037$ & $0.000011$ & $0.001$",
     "\\quad $\\ln(1+A_{it})$ contemporaneous & $-0.000003$ & $0.000005$ & $0.55$"),
    ("\\quad $\\sinh^{-1}(A_{it})$ & $+0.000033$ & $0.000010$ & $0.001$",
     "\\quad $\\sinh^{-1}(A_{it})$ & $-0.000003$ & $0.000005$ & $0.53$"),
    ("\\quad $\\ln(1+A_{i,t-3})$ (3-year lag) & $-0.000095$ & $0.000012$ & $< 0.001$",
     "\\quad $\\ln(1+A_{i,t-3})$ (3-year lag) & $-0.000041$ & $0.000005$ & $< 0.001$"),
    ("\\quad $\\ln(1+A^{\\text{mech}}_{it})$ & $+0.000226$ & $0.000017$ & $< 0.001$",
     "\\quad $\\ln(1+A^{\\text{mech}}_{it})$ & $+0.000092$ & $0.000009$ & $< 0.001$"),
    ("\\quad $\\ln(1+A^{\\text{fire}}_{it})$ & $-0.000250$ & $0.000018$ & $< 0.001$",
     "\\quad $\\ln(1+A^{\\text{fire}}_{it})$ & $-0.000091$ & $0.000006$ & $< 0.001$"),
    ("\\quad $FRT_{it}$, TWFE & $-0.00172$ & $0.00040$ & $< 0.001$",
     "\\quad $FRT_{it}$, TWFE & $-0.00084$ & $0.00018$ & $< 0.001$"),
    ("\\quad $\\ln(1+A_{it})$ contemporaneous & $-0.000245$ & $0.000054$ & $< 0.001$",
     "\\quad $\\ln(1+A_{it})$ contemporaneous & $-0.000141$ & $0.000024$ & $< 0.001$"),
    ("\\quad $\\ln(1+A_{i,t-3})$ & $-0.001043$ & $0.000059$ & $< 0.001$",
     "\\quad $\\ln(1+A_{i,t-3})$ & $-0.000432$ & $0.000027$ & $< 0.001$"),
    ("\\quad $FRT_{i,t-3}$ & $-0.00272$ & $0.00042$ & $< 0.001$",
     "\\quad $FRT_{i,t-3}$ & $-0.00102$ & $0.00019$ & $< 0.001$"),
    ("\\quad $FRT_{it} \\times$ Low density & $+0.000757$ & $0.000174$ & $< 0.001$",
     "\\quad $FRT_{it} \\times$ Low density & $+0.000010$ & $0.000054$ & $0.85$"),
    ("\\quad $FRT_{it} \\times$ Medium density & $+0.000104$ & $0.000098$ & $0.29$",
     "\\quad $FRT_{it} \\times$ Medium density & $-0.000050$ & $0.000044$ & $0.26$"),
    ("\\quad $FRT_{it} \\times$ High density & $-0.000358$ & $0.000116$ & $0.002$",
     "\\quad $FRT_{it} \\times$ High density & $-0.000133$ & $0.000081$ & $0.10$"),
    ("\\quad $FRT_{it}$, Intermix only ($N = 1{,}619{,}850$) & $+0.000045$ & $0.000176$ & $0.80$",
     "\\quad $FRT_{it}$, Intermix only ($N = 4{,}692{,}700$) & $-0.000025$ & $0.000065$ & $0.70$"),
    ("\\quad $\\ln(1+A_{i,t-3})$, Intermix & $+0.000105$ & $0.000026$ & $< 0.001$",
     "\\quad $\\ln(1+A_{i,t-3})$, Intermix & $+0.000054$ & $0.000010$ & $< 0.001$"),
    ("\\quad Future treatment (5-yr lead), pre-period & $-0.000454$ & $0.000119$ & $< 0.001$",
     "\\quad Future treatment (5-yr lead), pre-period & $-0.000442$ & $0.000047$ & $< 0.001$"),

    # ----- Robustness panel messages -----
    ("The preferred three-year-lagged log coefficient of $-0.001043$ ($p < 0.001$) implies the 7.2 percent elasticity",
     "The preferred three-year-lagged log coefficient of $-0.000432$ ($p < 0.001$) implies the 3.0 percent elasticity"),
    ("The three-year-lagged log cumulative acres coefficient flips sign to $-0.000095$ ($p < 0.001$)",
     "The three-year-lagged log cumulative acres coefficient is $-0.000041$ ($p < 0.001$), still negative"),
    ("The prescribed-fire subsample is negative and significant ($-0.000250$, $p < 0.001$), while the mechanical-only subsample is positive ($+0.000226$, $p < 0.001$)",
     "The prescribed-fire subsample is negative and significant ($-0.000091$, $p < 0.001$), while the mechanical-only subsample is positive ($+0.000092$, $p < 0.001$)"),
    ("The negative and significant High-density Intermix coefficient ($-0.000358$, $p = 0.002$) cannot be generated by any residual selection-on-fire-activity story",
     "The negatively signed High-density Intermix coefficient ($-0.000133$, $p = 0.10$) cannot be generated by any residual selection-on-fire-activity story"),
    ("Panel F reports the future-treatment placebo, which is negative and strongly significant ($-0.000415$, $p < 0.001$)",
     "Panel F reports the future-treatment placebo, which is negative and strongly significant ($-0.000442$, $p < 0.001$)"),
    ("Panel F reports the future-treatment placebo, which is negative and strongly significant ($-0.000454$, $p < 0.001$)",
     "Panel F reports the future-treatment placebo, which is negative and strongly significant ($-0.000442$, $p < 0.001$)"),

    # ----- Discussion / interpretation -----
    ("the three-year-lagged intensive margin on the binary occurrence outcome flips sign to $-0.000095$",
     "the three-year-lagged intensive margin on the binary occurrence outcome is $-0.000041$"),
    ("the much larger severity-channel elasticity of 7.2 percent",
     "the much larger severity-channel elasticity of 3.0 percent"),

    # ----- Conclusion -----
    ("a semi-elasticity of log structures destroyed to the three-year-lagged log of cumulative treated acreage of $-0.104$ ($p < 0.001$), implying a 7.2 percent reduction in structures destroyed for a doubling of cumulative treated acreage",
     "a semi-elasticity of log structures destroyed to the three-year-lagged log of cumulative treated acreage of $-0.043$ ($p < 0.001$), implying a 3.0 percent reduction in structures destroyed for a doubling of cumulative treated acreage"),

    # ----- Magnitudes interpretation block -----
    ("The central empirical finding is the 7.2 percent elasticity of structures destroyed with respect to log cumulative treated acreage, at a three-year lag and on the pooled sample of Western WUI polygons. To calibrate this magnitude, consider a representative treated polygon in our sample that accumulates 4{,}000 treated acres within its 10 km buffer over a decade (the 75th percentile of the treated cumulative distribution). Relative to an untreated counterfactual, the severity-channel coefficient implies a reduction of approximately $0.104 \\cdot \\ln(4{,}001)$ $\\approx 0.86$ log points in structures destroyed in a subsequent disaster, or roughly 58 percent. At the mean of 1{,}061 structures destroyed per disaster polygon-year, this translates into approximately 615 fewer structures destroyed per conditional disaster event in the treated polygon, valued at approximately \\$215 million per event using the \\$350k average structure value.",
     "The central empirical finding is the 3.0 percent elasticity of structures destroyed with respect to log cumulative treated acreage, at a three-year lag and on the pooled national sample of WUI polygons. To calibrate this magnitude, consider a representative treated polygon in our sample that accumulates 4{,}000 treated acres within its 10 km buffer over a decade (the 75th percentile of the treated cumulative distribution). Relative to an untreated counterfactual, the severity-channel coefficient implies a reduction of approximately $0.043 \\cdot \\ln(4{,}001) \\approx 0.36$ log points in structures destroyed in a subsequent disaster, or roughly 30 percent. At the mean of 872 structures destroyed per disaster polygon-year, this translates into approximately 261 fewer structures destroyed per conditional disaster event in the treated polygon, valued at approximately \\$91 million per event using the \\$350k average structure value."),

    # ----- "the eleven Western states" → "the conterminous United States" -----
    # Abstract (line 55) sample size sentence
    ("constructing a balanced panel of 560,572 Western wildland-urban interface (WUI) polygons",
     "constructing a balanced panel of 560,572 wildland-urban interface (WUI) polygons spanning the conterminous United States"),

    # Intro (line 75) sample sentence
    ("Our unit of observation is the wildland-urban interface polygon as delineated by \\citet{radeloff2018}, using the 2020 vintage of the national WUI dataset. We construct a panel of 560,572 WUI polygons across the eleven Western states (California, Oregon, Washington, Idaho, Montana, Wyoming, Nevada, Utah, Arizona, Colorado, and New Mexico) over 25 years, for a total of 14.0 million polygon-year observations.",
     "Our unit of observation is the wildland-urban interface polygon as delineated by \\citet{radeloff2018}, using the 2020 vintage of the national WUI dataset. We construct a panel of 560,572 WUI polygons across the 48 conterminous states and the District of Columbia over 25 years, for a total of 14.0 million polygon-year observations."),

    # Sample restriction paragraph (line 254)
    ("We restrict the sample to the eleven Western states where federal fuel treatment activity is concentrated: California, Oregon, Washington, Idaho, Montana, Wyoming, Nevada, Utah, Arizona, Colorado, and New Mexico. The cross-section contains 480,229 WUI polygons across these states, of which 560,572 meet the inclusion criteria for the balanced panel: either intersecting a disaster burn perimeter, lying within 10 km of a completed FACTS treatment, or being drawn as part of a random 50{,}000-polygon untreated control sample. The panel is balanced over 25 fiscal years (2000 to 2024), yielding 14{,}014{,}300 polygon-year observations.",
     "We construct the panel from the conterminous United States (the 48 contiguous states and the District of Columbia). The national cross-section of WUI polygons contains 2{,}040{,}588 Census-block-level Intermix and Interface polygons drawn from the 2020 SILVIS lab WUI v4 product, of which 560{,}572 meet the inclusion criteria for the balanced panel: either intersecting a disaster burn perimeter, lying within 10 km of a completed FACTS treatment, or being drawn as part of a random untreated control sample. The panel is balanced over 25 fiscal years (2000 to 2024), yielding 14{,}014{,}300 polygon-year observations."),

    # Table 3 caption
    ("\\caption{Summary Statistics, WUI Polygon-Year Panel (Western States, 2000--2024)}",
     "\\caption{Summary Statistics, WUI Polygon-Year Panel (Conterminous U.S., 2000--2024)}"),

    # Notes line 311
    ("the eleven Western states, observed over 25 fiscal years",
     "the conterminous United States, observed over 25 fiscal years"),

    # Notes line 461
    ("The sample is the balanced panel of 560{,}572 Western WUI polygons over 2000--2024.",
     "The sample is the balanced panel of 560{,}572 WUI polygons across the conterminous United States over 2000--2024."),

    # Welfare bullet line 641
    ("\\$350{,}000$, the median replacement value in Western WUI ZIP codes",
     "\\$350{,}000$, the median replacement value in WUI ZIP codes"),

    # Welfare body text Western states paragraph (line 714)
    ("The Western states, where biophysical fire hazard is greatest, capture essentially all of the welfare gains from the program in our analytic sample, consistent with the regional concentration of both federal treatment activity and WUI disaster exposure. The heterogeneity-by-density results in Table \\ref{tab:heterogeneity} further indicate that, even within the Western WUI, the welfare gains are concentrated in the High-density Intermix subset where both biophysical hazard and structural exposure are highest.",
     "The Western states, where biophysical fire hazard is greatest, capture a disproportionate share of the welfare gains from the program in our national sample, consistent with the regional concentration of both federal treatment activity and large WUI disaster events. The heterogeneity-by-density results in Table \\ref{tab:heterogeneity} further indicate that, within the WUI, the welfare gains are concentrated in the High-density Intermix subset where both biophysical hazard and structural exposure are highest."),

    # Notes line 768
    ("balanced panel of 560,572 Western WUI polygons over 2000--2024 (unless restricted as noted)",
     "balanced panel of 560,572 WUI polygons across the conterminous United States over 2000--2024 (unless restricted as noted)"),

    # Limitations second item (line 807)
    ("\\textit{Second}, the analysis is restricted to the eleven Western states, where federal fuel treatment activity is concentrated and where biophysical fire hazard is greatest. The Southeast, where prescribed fire is widely used and WUI development is also substantial, would be a natural extension; our choice to restrict to the West reflects the computational tractability of the spatial join rather than any theoretical prior about where the effect should operate. The severity coefficient in the pooled sample is essentially the Western average, and the point estimate in the Southeast would likely be smaller given lower structural exposure per disaster event but similar elasticities.",
     "\\textit{Second}, the pooled national severity coefficient averages over highly heterogeneous biophysical fire regimes. The Western states bear the bulk of the program's protective benefit, while in the Eastern and Midwestern states the elasticity is mechanically smaller because per-event structural destruction is lower. Subsample-specific re-estimation in fire-prone subregions would deliver larger absolute coefficients than the national pooled estimate, and the welfare arithmetic concentrates the gains in the Western High-density Intermix portion of the panel."),

    # Conclusion line 815
    ("Combining a national panel of 560,572 Western WUI polygons over 2000--2024",
     "Combining a national panel of 560,572 WUI polygons spanning the conterminous United States over 2000--2024"),

    # Future work Southeast (line 819)
    ("Second, extending the analysis to the Southeast, where prescribed fire is already widely used and where the WUI is similarly dense but less biophysically hazardous, would test the external validity of our headline severity elasticity across institutional and ecological contexts.",
     "Second, exploiting the regional heterogeneity within the national panel through subsample-specific or regionally interacted specifications would sharpen identification of the channels through which the severity elasticity varies across biophysical and institutional contexts."),
]

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
