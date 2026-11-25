"""
Stage 1 data preparation.

Reads the raw Qualtrics export (CSV, three header rows), applies the
pre-specified eligibility and data quality rules, recodes all Stage 1
variables, builds the scale composites and writes

    data/stage1_clean.xlsx   analytic sample, full eligible sample,
                             sample flow, codebook
    data/stage1_clean.csv    analytic sample only (input to 02_analysis.py)
    data/stage1_eligible.csv all eligible respondents with quality flags

Usage
    python 01_prepare_data.py --raw path/to/survey_export.csv --out data
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Coding scheme
# ---------------------------------------------------------------------------

AGREE = {
    "Strongly disagree": 1,
    "Somewhat disagree": 2,
    "Neither agree nor disagree": 3,
    "Somewhat agree": 4,
    "Strongly agree": 5,
}
EXTENT = {
    "To a very small extent": 1,
    "To a small extent": 2,
    "To a moderate extent": 3,
    "To a large extent": 4,
    "To a very large extent": 5,
}

# Diversity climate block (Q23). Retained items of the Mor Barak, Cherin and Berkman
# (1998) Diversity Perceptions Scale after pilot refinement. All retained
# items are positively worded, so no item is reverse-scored. Q23_7 is an
# attention check ("Please select Somewhat agree") and is not a
# scale item. Original 16-item numbering is kept in the codebook.
CLIMATE_ITEMS = {
    # new name : (raw column, original DPS item number, subscale)
    "OF1": ("Q23_1", 2, "Organisational fairness"),
    "OF2": ("Q23_2", 3, "Organisational fairness"),
    "OF3": ("Q23_3", 4, "Organisational fairness"),
    "OF4": ("Q23_4", 5, "Organisational fairness"),
    "OF5": ("Q23_5", 6, "Organisational fairness"),
    "OI1": ("Q23_6", 7, "Organisational inclusion"),
    "OI2": ("Q23_8", 8, "Organisational inclusion"),
    "OI3": ("Q23_9", 10, "Organisational inclusion"),
    "PV1": ("Q23_10", 11, "Personal diversity value"),
    "PV2": ("Q23_11", 12, "Personal diversity value"),
    "PV3": ("Q23_12", 13, "Personal diversity value"),
    "PC1": ("Q23_13", 14, "Personal comfort"),
}
ATTENTION_ITEM = "Q23_7"
ATTENTION_KEY = 4  # "Somewhat agree"

# Expected procedural justice (Colquitt, 2001, adapted). Q24_1 to Q24_7.
PJ_ITEMS = {f"PJ{i}": f"Q24_{i}" for i in range(1, 8)}

ORG_ITEMS = ["OF1", "OF2", "OF3", "OF4", "OF5", "OI1", "OI2", "OI3"]
PERS_ITEMS = ["PV1", "PV2", "PV3", "PC1"]
ALL_CLIMATE = ORG_ITEMS + PERS_ITEMS
PJ_LIST = list(PJ_ITEMS)

ROLE = {
    "Academic staff": "Academic staff",
    "Professional staff": "Professional staff",
    "Senior management or Head of Department": "Leaders",
}
SECTOR = {
    "Imam Mohammad Ibn Saud Islamic University": "Public",
    "Prince Sattam bin Abdulaziz University": "Public",
    "Shaqra University": "Public",
    "Majmaah University": "Public",
    "Al Maarefa University": "Private",
    "Dar Al Uloom University": "Private",
    "Arab Open University": "Private",
    "Al Yamamah University": "Private",
}
GENDER = {"Female": "Female", "Male": "Male", "Prefer not to say": "Not disclosed"}
AGE = {
    "18–24": "18 to 24",
    "25–34": "25 to 34",
    "35–44": "35 to 44",
    "45–54": "45 to 54",
    "55–64": "55 to 64",
    "65 or older": "65 or older",
    "Prefer not to say": "Not disclosed",
}
AGE_ORDINAL = {"18 to 24": 1, "25 to 34": 2, "35 to 44": 3,
               "45 to 54": 4, "55 to 64": 5, "65 or older": 6}
RECRUIT = {
    "Applicant": "Applicant only",
    "Recruiter": "Recruiter only",
    "Both": "Both",
}
AI_EXP = {"Mostly negative": "Negative", "Neutral": "Neutral",
          "Mostly positive": "Positive"}
ETHNICITY = {
    "Middle Eastern or North African": "Middle Eastern or North African",
    "Asian": "Asian",
    "White or Caucasian": "White",
    "Black or African": "Black or African",
    "Prefer not to say": "Not disclosed",
}
# Colleges and units grouped into broad fields for description
COLLEGE_FIELD = {
    "Medicine": "Health sciences", "Dentistry": "Health sciences", "Pharmacy": "Health sciences",
    "Nursing": "Health sciences", "Applied Medical or Health Sciences": "Health sciences",
    "Engineering": "Science, engineering and computing",
    "Computer and Information Sciences": "Science, engineering and computing",
    "Sciences": "Science, engineering and computing",
    "Architecture and Design": "Science, engineering and computing",
    "Business Administration": "Business and law", "Law": "Business and law",
    "Languages and Translation": "Humanities, social sciences and education",
    "Arabic Language": "Humanities, social sciences and education",
    "Humanities and Social Sciences": "Humanities, social sciences and education",
    "Media and Communication": "Humanities, social sciences and education",
    "Education": "Humanities, social sciences and education",
    "Sharia and Islamic Studies": "Humanities, social sciences and education",
    "Deanship or administrative unit": "Deanship or administrative unit",
}
# Conjoint attributes and level codes (used only for the exposure check)
CBC_ATTR = {"Algorithm Transparency": "transparency", "Privacy / Ethical Assurance": "privacy",
            "Usefulness": "usefulness", "Ease of Use": "ease"}
CBC_LEVEL = {"Low": 1, "Difficult": 1, "Moderate": 2, "High": 3, "Easy": 3}
N_TASKS, N_ALTS = 9, 2

EDUCATION = {
    "High school or lower": "High school or lower",
    "Diploma (below bachelor’s level)": "Diploma",
    "Bachelor’s degree": "Bachelor's",
    "Master’s degree": "Master's",
    "Doctorate": "Doctorate",
}


def english(series: pd.Series) -> pd.Series:
    """Keep the English part of a bilingual Qualtrics label."""
    s = series.fillna("").astype(str).str.split(" / ").str[0]
    # labels such as "Academic staff (e.g. ...)" keep only the head
    s = s.str.replace(r"\s*\(e\.g\..*$", "", regex=True)
    s = s.str.replace(r"\s*\((applied|involved)[^)]*\)$", "", regex=True)
    s = s.str.replace(r"\s*\(not a college\)$", "", regex=True)
    return s.str.strip().replace("", np.nan)


def strict_map(series: pd.Series, mapping: dict, name: str) -> pd.Series:
    """Map labels to codes and stop if any non-empty label is unrecognised."""
    s = english(series)
    unknown = sorted(set(s.dropna()) - set(mapping))
    if unknown:
        raise ValueError(f"{name}: unrecognised values {unknown}")
    return s.map(mapping)


def conjoint_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Mean level of each attribute across the profiles a respondent was shown,
    and whether every level of every attribute appeared equally often. Attribute
    positions were randomised, so the attribute in each position is read from
    the respondent's own display-name columns."""
    rows = []
    for _, r in df.iterrows():
        shown = {k: [] for k in CBC_ATTR.values()}
        for f in range(len(CBC_ATTR)):
            name = str(r[f"feature{f}.DISPLAY_NAME"]).split(" — ")[0].strip()
            if name not in CBC_ATTR:
                continue
            for t in range(1, N_TASKS + 1):
                for o in range(1, N_ALTS + 1):
                    m = re.search(r"alt='([^']*)'", str(r[f"feature{f}.{t}.{o}_CBCONJOINT"]))
                    if m:
                        shown[CBC_ATTR[name]].append(CBC_LEVEL[m.group(1).split(" — ")[0].strip()])
        row = {"ResponseId": r["ResponseId"]}
        balanced = True
        for k, v in shown.items():
            row[f"exp_{k}"] = np.mean(v) if v else np.nan
            counts = pd.Series(v).value_counts()
            balanced &= len(v) == N_TASKS * N_ALTS and len(counts) == 3 and counts.nunique() == 1
        row["conjoint_balanced"] = int(balanced)
        rows.append(row)
    return pd.DataFrame(rows)


def read_qualtrics(path: Path) -> pd.DataFrame:
    """Read a Qualtrics CSV with its three header rows."""
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    header = raw.iloc[0].tolist()
    if raw.iloc[2, 0] != '{"ImportId":"progress"}':
        raise ValueError("Unexpected header layout in the raw export")
    df = raw.iloc[3:].copy()
    df.columns = header
    return df.reset_index(drop=True)


def main(raw_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = read_qualtrics(raw_path)
    n_raw = len(raw)

    if raw["ResponseId"].duplicated().any():
        raise ValueError("Duplicate ResponseId values in the raw export")

    flow = [("Responses in the raw export", n_raw)]

    # 1. Survey previews generated while testing the instrument
    df = raw[raw["DistributionChannel"] != "preview"].copy()
    flow.append(("Removed, survey previews (instrument testing)", n_raw - len(df)))

    # 2. Data quality commitment (Q1)
    n0 = len(df)
    df = df[english(df["Q1"]) == "Yes, I will"]
    flow.append(("Removed, declined the data quality commitment (Q1)", n0 - len(df)))

    # 3. Informed consent (Q12)
    n0 = len(df)
    df = df[english(df["Q12"]) == "Yes, I agree"]
    flow.append(("Removed, consent not given (Q12)", n0 - len(df)))

    # ---------------- recoding ------------------------------------------
    out = pd.DataFrame({"ResponseId": df["ResponseId"].values})
    out["duration_sec"] = pd.to_numeric(df["Duration (in seconds)"]).values
    out["role"] = strict_map(df["Q20"], ROLE, "Q20").values
    univ = english(df["Q15"])
    unknown = sorted(set(univ.dropna()) - set(SECTOR))
    if unknown:
        raise ValueError(f"Q15: unrecognised universities {unknown}")
    out["university"] = univ.values
    out["sector"] = univ.map(SECTOR).values
    out["gender"] = strict_map(df["Q13"], GENDER, "Q13").values
    out["age_band"] = strict_map(df["Q14"], AGE, "Q14").values
    out["age_ordinal"] = out["age_band"].map(AGE_ORDINAL)
    # Five-band version used in the regression models. The two oldest bands
    # are merged because the 65 or older cell is too small to support an
    # interaction estimate.
    out["age_band5"] = out["age_band"].replace(
        {"55 to 64": "55 or older", "65 or older": "55 or older"})
    out["education"] = strict_map(df["Q17"], EDUCATION, "Q17").values
    out["college_unit"] = english(df["Q16"]).values
    unknown = sorted(set(out["college_unit"].dropna()) - set(COLLEGE_FIELD))
    if unknown:
        raise ValueError(f"Q16: unrecognised units {unknown}")
    out["college_field"] = out["college_unit"].map(COLLEGE_FIELD)
    out["ethnicity"] = strict_map(df["Q18"], ETHNICITY, "Q18").values
    out["quota_screen_out"] = (df["Last Seen Flow Element ID"] == "FL_QUOTA_EOS_ID").astype(int).values
    out["recruit_role"] = strict_map(df["Q21"], RECRUIT, "Q21").values
    out["recruit_exp"] = out["recruit_role"].map(
        {"Applicant only": "Low", "Recruiter only": "High", "Both": "High"})
    out["ai_experience"] = strict_map(df["Q22"], AI_EXP, "Q22").values

    for new, (col, _, _) in CLIMATE_ITEMS.items():
        out[new] = strict_map(df[col], AGREE, col).values
    out["ATTN"] = strict_map(df[ATTENTION_ITEM], AGREE, ATTENTION_ITEM).values
    for new, col in PJ_ITEMS.items():
        out[new] = strict_map(df[col], EXTENT, col).values

    # 4. Respondents screened out by a role quota before the expected procedural justice items
    n0 = len(out)
    out = out[out["quota_screen_out"] == 0]
    flow.append(("Removed, screened out by a role quota", n0 - len(out)))
    # Every remaining respondent must have complete scale data
    complete = out[ALL_CLIMATE + PJ_LIST + ["ATTN"]].notna().all(axis=1)
    if not complete.all():
        n0 = len(out)
        out = out[complete]
        flow.append(("Removed, incomplete scale data", n0 - len(out)))
    eligible = out.copy()
    flow.append(("Eligible respondents with complete scale data", len(eligible)))

    # ---------------- conjoint exposure (for the survey-order check) ----
    exposure = conjoint_exposure(df[df["ResponseId"].isin(eligible["ResponseId"])])
    eligible = eligible.merge(exposure, on="ResponseId", how="left", validate="one_to_one")

    # ---------------- quality flags -------------------------------------
    eligible["attention_pass"] = (eligible["ATTN"] == ATTENTION_KEY).astype(int)
    eligible["straightline_climate"] = (
        eligible[ALL_CLIMATE].nunique(axis=1) == 1).astype(int)
    eligible["straightline_pj"] = (eligible[PJ_LIST].nunique(axis=1) == 1).astype(int)
    eligible["straightline_both"] = (
        eligible["straightline_climate"] & eligible["straightline_pj"]).astype(int)
    half_median = 0.5 * eligible["duration_sec"].median()
    eligible["fast_responder"] = (eligible["duration_sec"] < half_median).astype(int)

    # ---------------- composites ----------------------------------------
    eligible["climate_12"] = eligible[ALL_CLIMATE].mean(axis=1)
    eligible["climate_org"] = eligible[ORG_ITEMS].mean(axis=1)
    eligible["climate_pers"] = eligible[PERS_ITEMS].mean(axis=1)
    eligible["pj"] = eligible[PJ_LIST].mean(axis=1)

    # 5. Attention check
    analytic = eligible[eligible["attention_pass"] == 1].copy()
    flow.append(("Removed, failed the attention check (Q23_7)",
                 len(eligible) - len(analytic)))
    flow.append(("Stage 1 analytic sample", len(analytic)))
    n_reg = int(((analytic["gender"] != "Not disclosed")
                 & (analytic["age_band"] != "Not disclosed")).sum())
    flow.append(("Of whom disclosed both gender and age (regression sample)", n_reg))

    analytic["regression_sample"] = (
        (analytic["gender"] != "Not disclosed")
        & (analytic["age_band"] != "Not disclosed")).astype(int)

    # ---------------- sanity checks -------------------------------------
    for c in ALL_CLIMATE + PJ_LIST:
        assert analytic[c].between(1, 5).all(), c
    assert analytic["ResponseId"].is_unique
    assert analytic["sector"].notna().all()

    # ---------------- outputs -------------------------------------------
    flow_df = pd.DataFrame(flow, columns=["Step", "N"])
    codebook = build_codebook()

    analytic.to_csv(out_dir / "stage1_clean.csv", index=False)
    eligible.to_csv(out_dir / "stage1_eligible.csv", index=False)
    with pd.ExcelWriter(out_dir / "stage1_clean.xlsx", engine="openpyxl") as xw:
        analytic.to_excel(xw, sheet_name="Analytic_sample", index=False)
        eligible.to_excel(xw, sheet_name="All_eligible_flags", index=False)
        flow_df.to_excel(xw, sheet_name="Sample_flow", index=False)
        codebook.to_excel(xw, sheet_name="Codebook", index=False)

    format_workbook(out_dir / "stage1_clean.xlsx")
    print(flow_df.to_string(index=False))


def format_workbook(path: Path) -> None:
    """Readable formatting, and composite columns written as AVERAGE formulas
    over the item columns so the derivation is visible in the workbook."""
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)
    composites = {"climate_12": ALL_CLIMATE, "climate_org": ORG_ITEMS,
                  "climate_pers": PERS_ITEMS, "pj": PJ_LIST}
    for ws in wb.worksheets:
        header = [c.value for c in ws[1]]
        col = {h: get_column_letter(i + 1) for i, h in enumerate(header)}
        if ws.title in ("Analytic_sample", "All_eligible_flags"):
            for comp, items in composites.items():
                cl = col[comp]
                refs = ",".join(f"{col[i]}{{r}}" for i in items)
                for r in range(2, ws.max_row + 1):
                    ws[f"{cl}{r}"] = f"=AVERAGE({refs.format(r=r)})"
                    ws[f"{cl}{r}"].number_format = "0.000"
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name="Times New Roman", size=11, bold=(c.row == 1))
        for c in ws[1]:
            c.fill = PatternFill("solid", fgColor="DDE6F2")
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "B2" if ws.title in ("Analytic_sample", "All_eligible_flags") else "A2"
        for i, h in enumerate(header, start=1):
            width = 14
            if ws.title == "Codebook":
                width = {1: 22, 2: 70, 3: 14, 4: 70}.get(i, 14)
            elif ws.title == "Sample_flow":
                width = {1: 62, 2: 10}.get(i, 14)
            elif h in ("ResponseId", "university", "college_unit"):
                width = 26
            ws.column_dimensions[get_column_letter(i)].width = width
        if ws.title == "Codebook":
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.properties.creator = ""
    wb.properties.lastModifiedBy = ""
    wb.save(path)


def build_codebook() -> pd.DataFrame:
    rows = [
        ("ResponseId", "Qualtrics response identifier", "ResponseId", "Text"),
        ("duration_sec", "Time to complete the survey in seconds", "Duration", "Numeric"),
        ("role", "Stakeholder role", "Q20",
         "Academic staff; Leaders (senior management or head of department); Professional staff"),
        ("university", "University of employment", "Q15", "Eight participating universities"),
        ("sector", "University sector derived from university", "Q15",
         "Public (Imam Mohammad Ibn Saud Islamic, Prince Sattam bin Abdulaziz, Shaqra, Majmaah); "
         "Private (Al Maarefa, Dar Al Uloom, Arab Open, Al Yamamah)"),
        ("gender", "Gender", "Q13", "Female; Male; Not disclosed"),
        ("age_band", "Age band", "Q14", "18 to 24 ... 65 or older; Not disclosed"),
        ("age_ordinal", "Age band as ordinal score", "Q14", "1 = 18 to 24 ... 6 = 65 or older; blank if not disclosed"),
        ("age_band5", "Age band with 55 to 64 and 65 or older merged", "Q14", "Used in regression models"),
        ("education", "Highest qualification", "Q17", "High school or lower; Diploma; Bachelor's; Master's; Doctorate"),
        ("college_unit", "College or unit", "Q16", "Descriptive only"),
        ("college_field", "College or unit grouped into broad fields", "Q16",
         "Health sciences; Science, engineering and computing; Business and law; Humanities, social sciences and education; Deanship or administrative unit"),
        ("ethnicity", "Ethnicity", "Q18", "Middle Eastern or North African; Asian; White; Black or African; Not disclosed"),
        ("quota_screen_out", "Screened out by a role quota", "Last Seen Flow Element ID", "1 = yes"),
        ("recruit_role", "Role personally experienced in recruitment", "Q21", "Applicant only; Recruiter only; Both"),
        ("recruit_exp", "Recruitment experience", "Q21", "High = recruiter only or both; Low = applicant only"),
        ("ai_experience", "Overall experience of using AI or digital technology", "Q22",
         "Negative (mostly negative); Neutral; Positive (mostly positive)"),
    ]
    for new, (col, orig, sub) in CLIMATE_ITEMS.items():
        rows.append((new, f"Diversity climate item, {sub} (original DPS item {orig})", col,
                     "1 strongly disagree to 5 strongly agree; positively worded"))
    rows.append(("ATTN", "Attention check (select Somewhat agree)", ATTENTION_ITEM,
                 "1 to 5; pass = 4"))
    for new, col in PJ_ITEMS.items():
        rows.append((new, "Expected procedural justice item (Colquitt, 2001, adapted)", col,
                     "1 to a very small extent to 5 to a very large extent"))
    rows += [
        ("attention_pass", "Passed the attention check", ATTENTION_ITEM, "1 = yes, 0 = no"),
        ("straightline_climate", "Identical answer to all 12 climate items", "Q23", "1 = yes"),
        ("straightline_pj", "Identical answer to all 7 PJ items", "Q24", "1 = yes"),
        ("straightline_both", "Straight-lined both scales", "Q23, Q24", "1 = yes"),
        ("fast_responder", "Duration below half the median of eligible respondents", "Duration", "1 = yes"),
        ("climate_12", "Diversity climate composite, mean of 12 items", "Q23", "1 to 5"),
        ("climate_org", "Organisational climate component, mean of 8 items (OF1 to OF5, OI1 to OI3)", "Q23", "1 to 5"),
        ("climate_pers", "Personal diversity orientation component, mean of 4 items (PV1 to PV3, PC1)", "Q23", "1 to 5"),
        ("pj", "Expected procedural justice composite, mean of 7 items", "Q24", "1 to 5"),
        ("exp_transparency", "Mean transparency level across the 18 profiles shown", "Conjoint", "1 low to 3 high"),
        ("exp_privacy", "Mean privacy and ethical assurance level across the 18 profiles shown", "Conjoint", "1 low to 3 high"),
        ("exp_usefulness", "Mean usefulness level across the 18 profiles shown", "Conjoint", "1 low to 3 high"),
        ("exp_ease", "Mean ease-of-use level across the 18 profiles shown", "Conjoint", "1 difficult to 3 easy"),
        ("conjoint_balanced", "Every level of every attribute shown equally often", "Conjoint", "1 = yes"),
        ("regression_sample", "Disclosed both gender and age", "Q13, Q14", "1 = included in regression models"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Description", "Source item", "Coding"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path, help="Raw Qualtrics CSV export")
    ap.add_argument("--out", default=Path("data"), type=Path, help="Output folder")
    args = ap.parse_args()
    main(args.raw, args.out)
