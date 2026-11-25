# Stage 3 (Chapter 5): Preference Variation Across Stakeholder Roles, University Sectors, and Expected Procedural Justice

Replication code for Stage 3 (Chapter 5) of the doctoral thesis *Stakeholder acceptance of AI-mediated recruitment in Saudi higher education* (King's Business School, King's College London).

Stage 3 asks whether the preference structure estimated in Stage 2 is shared across the workforce or differs by stakeholder role, university sector and expected procedural justice, the construct examined in Stage 1.

## Contents

| File | Purpose |
|---|---|
| `01_prepare_subgroups.py` | Combines the Stage 2 choice data with the Stage 1 respondent characteristics, defines the three moderators, checks that the two samples are identical and writes the clean data with subgroup composition and a codebook |
| `02_subgroup_analysis.py` | Runs every Stage 3 analysis and writes all tables (CSV), Figures 5.2 to 5.5 (PNG, 300 dpi) and a JSON file of key estimates |
| `requirements.txt` | Package versions used |

## How to run

The code was run with Python 3.13. Stage 3 builds on the outputs of Stages 1 and 2, so those are run first.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
python ../stage1/01_prepare_data.py --raw path/to/survey_export.csv --out ../stage1/data
python ../stage2/01_prepare_conjoint.py --raw path/to/survey_export.csv --stage1 ../stage1/data --out ../stage2/data
python ../stage2/02_conjoint_analysis.py --data ../stage2/data --out ../stage2/outputs
python 01_prepare_subgroups.py --stage1 ../stage1/data --stage2 ../stage2/data --out data
python 02_subgroup_analysis.py --data data --out outputs --stage2-out ../stage2/outputs
```

All scripts are deterministic. The random seed for the bootstrap is fixed in `02_subgroup_analysis.py`, so repeated runs reproduce every reported value.

## Data

The survey data are not included in this repository. Responses were collected anonymously under King's College London ethics approval MRSP-24/25-49508.

## Moderators

| Moderator | Subgroups | Reference |
|---|---|---|
| Stakeholder role (Q20) | Academic staff, professional staff, leaders | Academic staff |
| University sector, from the respondent's university (Q15) | Public (four universities), private (four universities) | Public |
| Expected procedural justice (Stage 1 composite of PJ1 to PJ7) | Continuous, standardised to mean 0 and SD 1. For display, split at the sample median into a lower group (below the median) and a higher group (at or above it) | Lower group |

## Analyses

* Marginal means within each subgroup and differences in marginal means between subgroups, with standard errors clustered by respondent (CR1, t with G − 1 degrees of freedom), Benjamini-Hochberg correction across the twelve differences of each comparison and the minimum detectable difference
* Omnibus Wald tests of all interactions between subgroup membership and the attribute-level indicators in a pooled linear probability model, for each comparison and jointly across the three roles
* Average marginal component effects within each subgroup, for description
* Expected procedural justice as a continuous moderator of every attribute-level effect, and the role comparison re-estimated with this moderator included
* Conditional logit importance shares and the governance share within each subgroup, differences between subgroups, and the governance share at one standard deviation below and above mean expected procedural justice, with respondent cluster bootstrap intervals (2,000 samples, respondents resampled within each subgroup). The full-sample row reports the Stage 2 estimates and intervals.

## References

Benjamini, Y. and Hochberg, Y. (1995) 'Controlling the false discovery rate: a practical and powerful approach to multiple testing', *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), pp. 289–300.

Cameron, A.C. and Miller, D.L. (2015) 'A practitioner's guide to cluster-robust inference', *Journal of Human Resources*, 50(2), pp. 317–372.

Gelman, A. and Stern, H. (2006) 'The difference between "significant" and "not significant" is not itself statistically significant', *The American Statistician*, 60(4), pp. 328–331.

Leeper, T.J., Hobolt, S.B. and Tilley, J. (2020) 'Measuring subgroup preferences in conjoint experiments', *Political Analysis*, 28(2), pp. 207–221.
