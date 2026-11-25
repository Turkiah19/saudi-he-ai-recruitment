"""
Stage 2 data preparation. Choice-based conjoint experiment.

Reads the raw Qualtrics export (CSV, three header rows) together with the
eligible sample produced by the Stage 1 preparation script, decodes the nine
conjoint tasks for every eligible respondent and writes the choice data in
long format, one row per respondent, task and profile.

The analytic sample is the Stage 1 analytic sample (eligible respondents who
passed the attention check), so all three stages rest on the same respondents.
Respondents who failed the attention check are kept in the full file with a
flag, for the robustness checks.

Outputs
    data/stage2_choices.csv         analytic sample, long format
    data/stage2_choices_all.csv     all eligible respondents, long format
    data/stage2_respondents.csv     one row per analytic respondent
    data/stage2_clean.xlsx          the above with design checks and codebook

Usage
    python ../stage1/01_prepare_data.py --raw path/to/survey_export.csv --out ../stage1/data
    python 01_prepare_conjoint.py --raw path/to/survey_export.csv --stage1 ../stage1/data --out data
"""

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_TASKS, N_ALTS = 9, 2

# Attribute display names in the survey and the variable names used here
ATTRIBUTES = {
    "Algorithm Transparency": "transparency",
    "Privacy / Ethical Assurance": "privacy",
    "Usefulness": "usefulness",
    "Ease of Use": "ease",
}
ATTR_ORDER = ["transparency", "privacy", "usefulness", "ease"]

# Level labels as displayed (English part), coded 1 to 3 from worst to best
LEVELS = {
    "transparency": {"Low": 1, "Moderate": 2, "High": 3},
    "privacy": {"Low": 1, "Moderate": 2, "High": 3},
    "usefulness": {"Low": 1, "Moderate": 2, "High": 3},
    "ease": {"Difficult": 1, "Moderate": 2, "Easy": 3},
}
LEVEL_LABELS = {a: {v: k for k, v in m.items()} for a, m in LEVELS.items()}

COVARIATES = ["role", "sector", "university", "gender", "age_band", "age_band5",
              "recruit_exp", "ai_experience", "climate_12", "climate_org",
              "climate_pers", "pj", "duration_sec", "fast_responder",
              "straightline_both", "attention_pass"]


def read_qualtrics(path: Path) -> pd.DataFrame:
    """Read a Qualtrics CSV with its three header rows."""
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    header = raw.iloc[0].tolist()
    if raw.iloc[2, 0] != '{"ImportId":"progress"}':
        raise ValueError("Unexpected header layout in the raw export")
    df = raw.iloc[3:].copy()
    df.columns = header
    return df.reset_index(drop=True)


def decode_level(cell: str, attribute: str) -> int:
    """Return the level code from the image alt text of one conjoint cell."""
    m = re.search(r"alt='([^']*)'", cell)
    if m is None:
        raise ValueError(f"No level found in conjoint cell: {cell[:80]}")
    label = m.group(1).split(" — ")[0].strip()
    if label not in LEVELS[attribute]:
        raise ValueError(f"Unrecognised level '{label}' for {attribute}")
    return LEVELS[attribute][label]


def decode_respondent(r: pd.Series) -> list[dict]:
    """Long-format rows for one respondent. Attribute rows were shown in a
    random order, so each attribute is located through the respondent's own
    display-name columns."""
    position = {}
    for f in range(len(ATTRIBUTES)):
        name = r[f"feature{f}.DISPLAY_NAME"].split(" — ")[0].strip()
        if name not in ATTRIBUTES:
            raise ValueError(f"Unrecognised attribute '{name}'")
        position[ATTRIBUTES[name]] = f
    if sorted(position) != sorted(ATTR_ORDER):
        raise ValueError(f"{r['ResponseId']}: attributes missing or repeated")

    rows = []
    for t in range(1, N_TASKS + 1):
        choice = r[f"C{t}"].strip()
        if choice not in ("1", "2"):
            raise ValueError(f"{r['ResponseId']}: no valid choice in task {t}")
        for o in range(1, N_ALTS + 1):
            row = {"ResponseId": r["ResponseId"], "task": t, "profile": o,
                   "chosen": int(choice == str(o))}
            for a in ATTR_ORDER:
                row[a] = decode_level(r[f"feature{position[a]}.{t}.{o}_CBCONJOINT"], a)
            rows.append(row)
    return rows, position


def main(raw_path: Path, stage1_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = read_qualtrics(raw_path)
    eligible = pd.read_csv(stage1_dir / "stage1_eligible.csv")
    analytic_ids = set(pd.read_csv(stage1_dir / "stage1_clean.csv")["ResponseId"])

    sub = raw[raw["ResponseId"].isin(eligible["ResponseId"])]
    if len(sub) != len(eligible):
        raise ValueError("Eligible respondents missing from the raw export")

    long_rows, resp_rows = [], []
    for _, r in sub.iterrows():
        rows, position = decode_respondent(r)
        long_rows += rows
        resp_rows.append({
            "ResponseId": r["ResponseId"],
            "design_version": int(r["vers_CBCONJOINT"]),
            # 1 = top row of the profile table, 4 = bottom row
            **{f"row_{a}": position[a] + 1 for a in ATTR_ORDER},
        })
    long = pd.DataFrame(long_rows)
    resp = pd.DataFrame(resp_rows)

    # Dominance: one profile at least as good on every attribute and better
    # on at least one. Choosing the dominated profile is not irrational under
    # random utility, but a high rate signals inattention.
    wide = long.pivot_table(index=["ResponseId", "task"], columns="profile",
                            values=ATTR_ORDER)
    diff = pd.concat({a: wide[(a, 1)] - wide[(a, 2)] for a in ATTR_ORDER}, axis=1)
    p1_dom = (diff >= 0).all(axis=1) & (diff > 0).any(axis=1)
    p2_dom = (diff <= 0).all(axis=1) & (diff < 0).any(axis=1)
    chosen1 = (long[long["profile"] == 1].set_index(["ResponseId", "task"])["chosen"]
               .reindex(diff.index))
    dom = pd.DataFrame({"dominated_task": (p1_dom | p2_dom).astype(int),
                        "dominant_chosen": np.where(p1_dom, chosen1, np.where(p2_dom, 1 - chosen1, np.nan))})
    long = long.merge(dom.reset_index(), on=["ResponseId", "task"], how="left")
    dsum = dom.groupby(level=0).agg(n_dominated_tasks=("dominated_task", "sum"),
                                    n_dominant_chosen=("dominant_chosen", "sum"))
    resp = resp.merge(dsum.reset_index(), on="ResponseId", how="left")

    # Covariates from Stage 1
    resp = resp.merge(eligible[["ResponseId"] + COVARIATES], on="ResponseId", how="left")
    resp["analytic"] = resp["ResponseId"].isin(analytic_ids).astype(int)
    if resp["analytic"].sum() != len(analytic_ids):
        raise ValueError("Analytic respondents missing from the conjoint data")

    long = long.merge(resp[["ResponseId", "design_version",
                            "analytic"] + [f"row_{a}" for a in ATTR_ORDER]],
                      on="ResponseId", how="left")
    for a in ATTR_ORDER:
        long[f"{a}_label"] = long[a].map(LEVEL_LABELS[a])
        # Row of the profile table in which this attribute was displayed
        long[f"{a}_row"] = long.pop(f"row_{a}")
    long.insert(1, "resp_no", long["ResponseId"].map(
        {rid: i + 1 for i, rid in enumerate(resp["ResponseId"])}))
    resp.insert(1, "resp_no", range(1, len(resp) + 1))
    long = long.sort_values(["resp_no", "task", "profile"]).reset_index(drop=True)

    # ---------------- integrity checks ---------------------------------
    per_task = long.groupby(["ResponseId", "task"])
    assert (per_task.size() == N_ALTS).all(), "Each task must hold two profiles"
    assert (per_task["chosen"].sum() == 1).all(), "Each task must hold one choice"
    assert long.groupby("ResponseId")["task"].nunique().eq(N_TASKS).all()
    identical = (diff == 0).all(axis=1).sum()

    analytic = long[long["analytic"] == 1].reset_index(drop=True)
    resp_a = resp[resp["analytic"] == 1].reset_index(drop=True)
    checks = design_checks(analytic, resp_a, diff.loc[resp_a["ResponseId"]], identical)
    checks = pd.concat([checks, verify_decoding(sub, analytic, stage1_dir)], ignore_index=True)

    flow = pd.DataFrame([
        ("Eligible respondents (Stage 1)", len(resp), len(resp) * N_TASKS, len(resp) * N_TASKS * N_ALTS),
        ("Removed, failed the attention check (Q23_7)", int((resp["analytic"] == 0).sum()),
         int((resp["analytic"] == 0).sum()) * N_TASKS, int((resp["analytic"] == 0).sum()) * N_TASKS * N_ALTS),
        ("Stage 2 analytic sample", len(resp_a), len(resp_a) * N_TASKS, len(analytic)),
    ], columns=["Step", "Respondents", "Choice tasks", "Profiles (rows)"])

    # ---------------- outputs -------------------------------------------
    analytic.to_csv(out_dir / "stage2_choices.csv", index=False)
    long.to_csv(out_dir / "stage2_choices_all.csv", index=False)
    resp_a.to_csv(out_dir / "stage2_respondents.csv", index=False)
    xlsx = out_dir / "stage2_clean.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        analytic.to_excel(xw, sheet_name="Choice_data", index=False)
        resp_a.to_excel(xw, sheet_name="Respondents", index=False)
        long.to_excel(xw, sheet_name="All_eligible_choices", index=False)
        flow.to_excel(xw, sheet_name="Sample_flow", index=False)
        checks.to_excel(xw, sheet_name="Design_checks", index=False)
        build_codebook().to_excel(xw, sheet_name="Codebook", index=False)
    format_workbook(xlsx)
    print(flow.to_string(index=False))
    print(checks.to_string(index=False))


def verify_decoding(raw: pd.DataFrame, long: pd.DataFrame, stage1_dir: Path,
                    n_check: int = 20, seed: int = 12345) -> pd.DataFrame:
    """Two checks on the decoded levels.

    1. The mean level of each attribute across the 18 profiles a respondent saw
       must equal the exposure measures computed in the Stage 1 preparation.
    2. For a random selection of respondents, every attribute cell and every
       choice is read again from the raw export by a separate route (attribute
       located by keyword in its display name, level read from the first word
       of the image label) and compared with the decoded data.
    """
    out = []
    s1 = pd.read_csv(stage1_dir / "stage1_clean.csv").set_index("ResponseId")
    means = long.groupby("ResponseId")[ATTR_ORDER].mean()
    s1 = s1.loc[means.index, [f"exp_{a}" for a in ATTR_ORDER]]
    s1.columns = ATTR_ORDER
    max_gap = float((means - s1).abs().max().max())
    if max_gap > 1e-9:
        raise ValueError("Decoded levels do not match the Stage 1 exposure measures")
    out.append(("Decoded levels against the Stage 1 exposure measures",
                f"identical for all {len(means)} respondents (largest difference {max_gap:.1e})"))

    keywords = {"transparency": "transparency", "privacy": "privacy",
                "usefulness": "usefulness", "ease": "ease"}
    words = {1: ("Low", "Difficult"), 2: ("Moderate",), 3: ("High", "Easy")}
    rng = np.random.default_rng(seed)
    ids = rng.choice(long["ResponseId"].unique(), n_check, replace=False)
    raw = raw.set_index("ResponseId")
    tasks = cells = mismatches = 0
    for rid in ids:
        r = raw.loc[rid]
        pos = {a: f for f in range(len(ATTRIBUTES)) for a, k in keywords.items()
               if k in r[f"feature{f}.DISPLAY_NAME"].lower()}
        sub = long[long["ResponseId"] == rid].set_index(["task", "profile"])
        for t in range(1, N_TASKS + 1):
            tasks += 1
            for o in range(1, N_ALTS + 1):
                row = sub.loc[(t, o)]
                for a, f in pos.items():
                    cells += 1
                    alt = re.search(r"alt='(\w+)", r[f"feature{f}.{t}.{o}_CBCONJOINT"]).group(1)
                    mismatches += alt not in words[int(row[a])]
                mismatches += int(row["chosen"]) != int(r[f"C{t}"].strip() == str(o))
    if mismatches:
        raise ValueError(f"{mismatches} discrepancies against the raw export")
    out.append(("Decoded data against the raw export, random selection of respondents",
                f"{n_check} respondents, {tasks} tasks, {cells} attribute cells and "
                f"{tasks} choices checked, {mismatches} discrepancies"))
    return pd.DataFrame(out, columns=["Check", "Result"])


def design_checks(long: pd.DataFrame, resp: pd.DataFrame, diff: pd.DataFrame,
                  identical: int) -> pd.DataFrame:
    """Checks on the realised design and on response patterns."""
    out = []
    add = lambda check, result: out.append((check, result))
    add("Respondents", f"{len(resp)}")
    add("Choice tasks", f"{len(resp) * N_TASKS} ({N_TASKS} per respondent)")
    add("Profiles (rows)", f"{len(long)} ({N_ALTS} per task)")
    add("Design versions used", f"{resp['design_version'].nunique()} distinct versions "
        f"(version numbers {resp['design_version'].min()} to {resp['design_version'].max()})")
    add("Distinct profiles shown", f"{long[ATTR_ORDER].drop_duplicates().shape[0]} of 81 possible")
    add("Tasks with two identical profiles", f"{identical}")
    for a in ATTR_ORDER:
        vc = long[a].value_counts().sort_index()
        chi = stats.chisquare(vc.values)
        add(f"Level frequencies, {a}",
            ", ".join(f"{LEVEL_LABELS[a][k]} {v}" for k, v in vc.items())
            + f" (chi2({len(vc) - 1}) = {chi.statistic:.2f}, p = {chi.pvalue:.3f})")
    bal = long.groupby("ResponseId").apply(
        lambda g: all(g[a].value_counts().nunique() == 1 and g[a].nunique() == 3
                      for a in ATTR_ORDER), include_groups=False)
    add("Respondents shown every level of every attribute exactly six times",
        f"{int(bal.sum())} of {len(bal)} ({bal.mean():.1%})")
    for a, b in itertools.combinations(ATTR_ORDER, 2):
        tab = pd.crosstab(long[a], long[b])
        chi = stats.chi2_contingency(tab)
        add(f"Independence of {a} and {b} within profiles",
            f"chi2({chi.dof}) = {chi.statistic:.2f}, p = {chi.pvalue:.3f}")
    same = (diff == 0).sum(axis=1)
    add("Attributes with the same level in both profiles of a task",
        ", ".join(f"{k}: {v} tasks" for k, v in same.value_counts().sort_index().items()))
    left = long.loc[long["profile"] == 1, "chosen"]
    bt = stats.binomtest(int(left.sum()), len(left), 0.5)
    add("Share choosing the left profile (Option 1)",
        f"{left.mean():.3f} (binomial p = {bt.pvalue:.3f})")
    d = resp[["n_dominated_tasks", "n_dominant_chosen"]].sum()
    add("Tasks in which one profile dominated the other",
        f"{int(d['n_dominated_tasks'])}, dominant profile chosen in "
        f"{int(d['n_dominant_chosen'])} ({d['n_dominant_chosen'] / d['n_dominated_tasks']:.1%}, "
        f"binomial p = {stats.binomtest(int(d['n_dominant_chosen']), int(d['n_dominated_tasks'])).pvalue:.1e})")
    rows = resp[[f"row_{a}" for a in ATTR_ORDER]].astype(str).agg("".join, axis=1)
    add("Attribute row orders used", f"{rows.nunique()} of 24 possible")
    return pd.DataFrame(out, columns=["Check", "Result"])


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
        widths = {"Codebook": {1: 24, 2: 70, 3: 22, 4: 60},
                  "Design_checks": {1: 62, 2: 90},
                  "Sample_flow": {1: 46, 2: 13, 3: 13, 4: 16}}
        for i, h in enumerate(header, start=1):
            w = widths.get(ws.title, {}).get(i, 26 if h in ("ResponseId", "university") else 13)
            ws.column_dimensions[get_column_letter(i)].width = w
        if ws.title in ("Codebook", "Design_checks"):
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.properties.creator = ""
    wb.properties.lastModifiedBy = ""
    wb.save(path)


def build_codebook() -> pd.DataFrame:
    rows = [
        ("ResponseId", "Qualtrics response identifier", "ResponseId", "Text"),
        ("resp_no", "Respondent number used in this file", "Derived", "1 to N"),
        ("task", "Choice task", "C1 to C9", "1 to 9, in the order shown"),
        ("profile", "Profile within the task", "Conjoint", "1 = Option 1 (left), 2 = Option 2 (right)"),
        ("chosen", "Profile chosen", "C1 to C9", "1 = chosen, 0 = not chosen"),
    ]
    for a, name in zip(ATTR_ORDER, ATTRIBUTES):
        levels = "; ".join(f"{v} = {k}" for k, v in LEVELS[a].items())
        rows.append((a, f"{name} level of the profile", "Conjoint", levels))
        rows.append((f"{a}_label", f"{name} level as displayed", "Conjoint", "Text"))
        rows.append((f"{a}_row", f"Row of the profile table in which {name} was shown",
                     "feature0 to feature3 DISPLAY_NAME", "1 = top row to 4 = bottom row; fixed within respondent"))
    rows += [
        ("dominated_task", "One profile at least as good on every attribute and better on one", "Derived", "1 = yes"),
        ("dominant_chosen", "Dominant profile chosen, in dominated tasks", "Derived", "1 = yes, 0 = no, blank otherwise"),
        ("design_version", "Qualtrics conjoint design version", "vers_CBCONJOINT", "Integer"),
        ("analytic", "In the analytic sample (passed the attention check)", "Stage 1", "1 = yes"),
        ("n_dominated_tasks", "Tasks in which one profile dominated the other", "Derived", "Count"),
        ("n_dominant_chosen", "Dominated tasks in which the dominant profile was chosen", "Derived", "Count"),
        ("role, sector, university, gender, age_band, age_band5, recruit_exp, ai_experience",
         "Respondent characteristics", "Stage 1", "As in the Stage 1 codebook"),
        ("climate_12, climate_org, climate_pers, pj", "Stage 1 composites", "Stage 1", "1 to 5"),
        ("duration_sec, fast_responder, straightline_both, attention_pass",
         "Response quality indicators", "Stage 1", "As in the Stage 1 codebook"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Description", "Source", "Coding"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path, help="Raw Qualtrics CSV export")
    ap.add_argument("--stage1", default=Path("../stage1/data"), type=Path,
                    help="Folder holding the Stage 1 clean data")
    ap.add_argument("--out", default=Path("data"), type=Path, help="Output folder")
    args = ap.parse_args()
    main(args.raw, args.stage1, args.out)
