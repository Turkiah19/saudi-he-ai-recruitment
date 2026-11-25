# Stage 1 (Chapter 3): Diversity and Inclusion Climate and Expected Procedural Justice in AI-Mediated Recruitment

Replication code for Stage 1 (Chapter 3) of the doctoral thesis *Stakeholder acceptance of AI-mediated recruitment in Saudi higher education* (King's Business School, King's College London).

Stage 1 asks whether employees who perceive their university's diversity and inclusion climate as more positive also expect an AI-mediated recruitment process to be more procedurally just, and whether that association differs by recruitment experience, experience of AI, age and gender.

## Contents

| File | Purpose |
|---|---|
| `01_prepare_data.py` | Reads the raw Qualtrics export, applies the eligibility and data quality rules, recodes all variables, builds the scale composites and writes the clean data with a sample flow and codebook |
| `02_analysis.py` | Runs every Stage 1 analysis and writes all tables (CSV), figures (PNG, 300 dpi) and a JSON file of key estimates |
| `requirements.txt` | Package versions used |

## How to run

The code was run with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
python 01_prepare_data.py --raw path/to/survey_export.csv --out data
python 02_analysis.py --data data --out outputs
```

Both scripts are deterministic. The random seed for parallel analysis and the wild cluster bootstrap is fixed in `02_analysis.py`, so repeated runs reproduce every reported value. The confirmatory factor models are fitted by numerical optimisation and can differ between runs around the twelfth decimal place, far below any reported precision.

## Data

The survey data are not included in this repository. Responses were collected anonymously under King's College London ethics approval MRSP-24/25-49508. The scripts expect the Qualtrics CSV export with its three header rows.

## Data preparation rules

Applied in this order by `01_prepare_data.py`.

1. Remove survey previews generated while testing the instrument.
2. Remove respondents who declined the data quality commitment (Q1).
3. Remove respondents who did not give consent (Q12).
4. Remove respondents screened out by a role quota before the expected procedural justice items.
5. Remove respondents who failed the attention check (Q23_7, which asks respondents to select "Somewhat agree").

Respondents who straight-lined both scales or completed the survey in under half the median time are flagged and retained, and the analysis reports results without them as robustness checks. The regression models use respondents who disclosed both gender and age, and a robustness model retains everyone with an explicit not-disclosed category.

## Measures

**Diversity and inclusion climate.** Twelve items retained from the Diversity Perceptions Scale (Mor Barak, Cherin and Berkman, 1998) after pilot refinement, Q23_1 to Q23_6 and Q23_8 to Q23_13, scored 1 (strongly disagree) to 5 (strongly agree). All twelve are positively worded and none is reverse-scored. The composite is the item mean. The organisational component is the mean of the eight organisational items and the personal component the mean of the four personal items.

**Expected procedural justice.** Seven items adapted from Colquitt (2001) to a prospective AI-mediated recruitment process, Q24_1 to Q24_7, scored 1 (to a very small extent) to 5 (to a very large extent). The composite is the item mean.

**Moderators and controls.** Recruitment experience (Q21, high for recruiter only or both, low for applicant only), overall experience of AI or digital technology (Q22), age band (Q14, with 55 to 64 and 65 or older merged in the models) and gender (Q13). Stakeholder role (Q20), sector (derived from Q15) and university enter the robustness checks.

## Analyses

* A survey-order check testing whether the randomised conjoint profiles each respondent saw predict the attitudinal scales answered afterwards
* Sample flow, sample characteristics by role, role differences in composition and in the constructs
* Item statistics, Cronbach's alpha with Feldt confidence intervals, McDonald's omega, AVE, KMO and Bartlett's test
* Parallel analysis, exploratory factor analysis (maximum likelihood, oblimin) and confirmatory factor analysis (maximum likelihood) with CFI, TLI, RMSEA and its 90% confidence interval, and SRMR
* Discriminant validity (latent correlation and heterotrait-monotrait ratio) and a single common factor check
* Pearson correlations with Fisher confidence intervals
* OLS regression with HC3 standard errors and t-based inference, climate mean-centred
* Moderation by a fully interacted model, joint Wald tests per moderator, Benjamini-Hochberg correction, one-moderator-at-a-time models and conditional climate slopes
* Diagnostics (generalised VIF, Breusch-Pagan, Jarque-Bera, Shapiro-Wilk, RESET, quadratic term, Cook's distance), with residual plots using a bootstrapped lowess band and a simulated normal quantile envelope
* Robustness of the climate estimate across fifteen alternative specifications, including university fixed effects and a wild cluster restricted bootstrap by university with Webb weights

## Notes on estimation

* `semopy` reports the χ² statistic as N times the minimised maximum likelihood discrepancy, the same convention as the default in `lavaan`.
* The generalised VIF follows Fox and Monette (1992), and the adjusted value GVIF^(1/(2df)) is comparable across terms with different degrees of freedom.
* The wild cluster bootstrap imposes the null hypothesis, uses CR1 cluster-robust standard errors and draws Webb six-point weights (Cameron, Gelbach and Miller, 2008; Webb, 2023).

## References

Atkinson, A.C. (1985) *Plots, transformations, and regression: an introduction to graphical methods of diagnostic regression analysis*. Clarendon Press.

Cameron, A.C., Gelbach, J.B. and Miller, D.L. (2008) 'Bootstrap-based improvements for inference with clustered errors', *Review of Economics and Statistics*, 90(3), pp. 414–427.

Colquitt, J.A. (2001) 'On the dimensionality of organizational justice: a construct validation of a measure', *Journal of Applied Psychology*, 86(3), pp. 386–400.

Fox, J. and Monette, G. (1992) 'Generalized collinearity diagnostics', *Journal of the American Statistical Association*, 87(417), pp. 178–183.

Mor Barak, M.E., Cherin, D.A. and Berkman, S. (1998) 'Organizational and personal dimensions in diversity climate: ethnic and gender differences in employee perceptions', *Journal of Applied Behavioral Science*, 34(1), pp. 82–104.

Webb, M.D. (2023) 'Reworking wild bootstrap-based inference for clustered errors', *Canadian Journal of Economics*, 56(3), pp. 839–858.
