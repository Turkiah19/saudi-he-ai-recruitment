"""
Stage 3 data preparation. Preference variation across subgroups.

Combines the Stage 2 choice data with the respondent characteristics from the
Stage 1 clean data and defines the three moderators examined in Chapter 5.

    role     Academic staff (reference), Professional staff, Leaders
    sector   Public (reference), Private, from the respondent's university
    pj       Expected procedural justice (Stage 1 composite), entered as a
             standardised continuous moderator (pj_z) and, for display, split
             at the sample median into a lower group (below the median) and a
             higher group (at or above the median)

The analytic sample is the same 317 respondents analysed in Stages 1 and 2.

Outputs
    data/stage3_choices.csv       long format, one row per respondent, task and profile
    data/stage3_respondents.csv   one row per respondent
    data/stage3_clean.xlsx        the above with subgroup composition and codebook

Usage
    python 01_prepare_subgroups.py --stage1 ../stage1/data --stage2 ../stage2/data --out data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ATTRS = ["privacy", "transparency", "usefulness", "ease"]
ROLES = ["Academic staff", "Professional staff", "Leaders"]
SECTORS = ["Public", "Private"]
PJ_GROUPS = ["Lower expected PJ", "Higher expected PJ"]
N_TASKS = 9


def build(stage1: Path, stage2: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    s1 = pd.read_csv(stage1 / "stage1_clean.csv")
    ch = pd.read_csv(stage2 / "stage2_choices.csv")
    r2 = pd.read_csv(stage2 / "stage2_respondents.csv")

    if set(s1["ResponseId"]) != set(ch["ResponseId"]) or len(s1) != r2["ResponseId"].nunique():
        raise ValueError("Stage 1 and Stage 2 analytic samples differ")

    resp = s1[["ResponseId", "role", "sector", "university", "pj"]].copy()
    for col, allowed in (("role", ROLES), ("sector", SECTORS)):
        bad = set(resp[col].dropna()) - set(allowed)
        if bad or resp[col].isna().any():
            raise ValueError(f"Unexpected or missing values in {col}: {bad}")
    if resp["pj"].isna().any():
        raise ValueError("Missing expected procedural justice scores")

    median = float(resp["pj"].median())
    resp["pj_z"] = (resp["pj"] - resp["pj"].mean()) / resp["pj"].std(ddof=1)
    resp["pj_group"] = np.where(resp["pj"] < median, PJ_GROUPS[0], PJ_GROUPS[1])
    resp["role"] = pd.Categorical(resp["role"], ROLES)
    resp["sector"] = pd.Categorical(resp["sector"], SECTORS)
    resp["pj_group"] = pd.Categorical(resp["pj_group"], PJ_GROUPS)

    keep = ["ResponseId", "task", "profile", "chosen"] + ATTRS
    long = ch[keep].merge(resp[["ResponseId", "role", "sector", "pj", "pj_z", "pj_group"]],
                          on="ResponseId", how="left", validate="many_to_one")
    long = long.sort_values(["ResponseId", "task", "profile"]).reset_index(drop=True)

    # Integrity checks
    if len(long) != len(resp) * N_TASKS * 2:
        raise ValueError("Unexpected number of profile rows")
    per_task = long.groupby(["ResponseId", "task"])["chosen"].sum()
    if not (per_task == 1).all():
        raise ValueError("Every task must have exactly one chosen profile")
    if long[["role", "sector", "pj_z", "pj_group"]].isna().any().any():
        raise ValueError("Subgroup variables missing after merge")

    info = {"pj_median": median, "pj_mean": float(resp["pj"].mean()),
            "pj_sd": float(resp["pj"].std(ddof=1))}
    return long, resp, info


def composition(resp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for moderator_name, col, groups in (("Stakeholder role", "role", ROLES),
                              ("University sector", "sector", SECTORS),
                              ("Expected procedural justice", "pj_group", PJ_GROUPS)):
        for g in groups:
            n = int((resp[col] == g).sum())
            rows.append({"Moderator": moderator_name, "Subgroup": g, "Respondents": n,
                         "Tasks": n * N_TASKS, "Profile rows": n * N_TASKS * 2})
    return pd.DataFrame(rows)


def build_codebook(info: dict) -> pd.DataFrame:
    rows = [
        ("ResponseId", "Qualtrics response identifier", "text", "Links to the Stage 1 and Stage 2 data"),
        ("task", "Choice task number", "1 to 9", ""),
        ("profile", "Profile position in the task", "1 = left, 2 = right", ""),
        ("chosen", "Profile chosen", "1 = chosen, 0 = not chosen", "Exactly one profile chosen per task"),
        ("privacy", "Ethical and privacy assurance", "1 = Low, 2 = Moderate, 3 = High", ""),
        ("transparency", "Algorithmic transparency", "1 = Low, 2 = Moderate, 3 = High", ""),
        ("usefulness", "Perceived usefulness", "1 = Low, 2 = Moderate, 3 = High", ""),
        ("ease", "Perceived ease of use", "1 = Difficult, 2 = Moderate, 3 = Easy", ""),
        ("role", "Stakeholder role (Q20)", "; ".join(ROLES), "Academic staff is the reference category"),
        ("sector", "University sector", "; ".join(SECTORS),
         "Derived from the respondent's university (Q15). Public is the reference category"),
        ("university", "University (Q15)", "eight universities", "Respondent file only"),
        ("pj", "Expected procedural justice", "1 to 5",
         "Mean of PJ1 to PJ7 from the Stage 1 clean data"),
        ("pj_z", "Expected procedural justice, standardised", "mean 0, SD 1",
         f"(pj − {info['pj_mean']:.3f}) / {info['pj_sd']:.3f}"),
        ("pj_group", "Expected procedural justice, median split", "; ".join(PJ_GROUPS),
         f"Lower = below the sample median of {info['pj_median']:.3f}, "
         "higher = at or above it. Used for display only"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Description", "Values", "Notes"])


def format_workbook(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)
    for ws in wb.worksheets:
        header = [c.value for c in ws[1]]
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name="Times New Roman", size=11, bold=(c.row == 1))
        for c in ws[1]:
            c.fill = PatternFill("solid", fgColor="DDE6F2")
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        widths = {"Codebook": {1: 16, 2: 46, 3: 40, 4: 60},
                  "Subgroups": {1: 30, 2: 22, 3: 13, 4: 10, 5: 13}}
        for i, h in enumerate(header, start=1):
            w = widths.get(ws.title, {}).get(
                i, 26 if h in ("ResponseId", "university") else 20 if h in ("role", "pj_group") else 12)
            ws.column_dimensions[get_column_letter(i)].width = w
        if ws.title == "Codebook":
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.properties.creator = ""
    wb.properties.lastModifiedBy = ""
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="../stage1/data")
    ap.add_argument("--stage2", default="../stage2/data")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    long, resp, info = build(Path(args.stage1), Path(args.stage2))
    comp = composition(resp)

    long.to_csv(out / "stage3_choices.csv", index=False)
    resp.to_csv(out / "stage3_respondents.csv", index=False)
    xlsx = out / "stage3_clean.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        long.to_excel(w, sheet_name="Choice_data", index=False)
        resp.to_excel(w, sheet_name="Respondents", index=False)
        comp.to_excel(w, sheet_name="Subgroups", index=False)
        build_codebook(info).to_excel(w, sheet_name="Codebook", index=False)
    format_workbook(xlsx)

    print(comp.to_string(index=False))
    print(f"Expected PJ median {info['pj_median']:.3f}, mean {info['pj_mean']:.3f}, "
          f"SD {info['pj_sd']:.3f}")
    print(f"{resp['ResponseId'].nunique()} respondents, {len(long)} profile rows")


if __name__ == "__main__":
    main()
