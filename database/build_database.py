"""
Builds a relational SQLite database from the clean data of Stages 1 to 3.

Reads the outputs of the three preparation scripts, loads them into the
schema in schema.sql, and checks that the database reproduces the clean
files exactly before writing it.

Usage
    python build_database.py --stage1 ../stage1/data --stage2 ../stage2/data \
        --stage3 ../stage3/data --out data/thesis.db
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

ITEMS = (
    [(f"OF{i}", "Diversity and inclusion climate", "Organisational fairness and inclusion") for i in range(1, 6)]
    + [(f"OI{i}", "Diversity and inclusion climate", "Organisational fairness and inclusion") for i in range(1, 4)]
    + [(f"PV{i}", "Diversity and inclusion climate", "Personal diversity orientation") for i in range(1, 4)]
    + [("PC1", "Diversity and inclusion climate", "Personal diversity orientation")]
    + [(f"PJ{i}", "Expected procedural justice", None) for i in range(1, 8)]
    + [("ATTN", "Attention check", None)]
)
ATTRIBUTES = [
    ("privacy", "Ethical and privacy assurance", "Governance", 1),
    ("transparency", "Algorithmic transparency", "Governance", 2),
    ("usefulness", "Perceived usefulness", "Efficiency", 3),
    ("ease", "Perceived ease of use", "Efficiency", 4),
]
LEVELS = {a[0]: {1: "Low", 2: "Moderate", 3: "High"} for a in ATTRIBUTES}
LEVELS["ease"] = {1: "Difficult", 2: "Moderate", 3: "Easy"}
CODES = [a[0] for a in ATTRIBUTES]


def load(stage1: Path, stage2: Path, stage3: Path) -> dict[str, pd.DataFrame]:
    s1 = pd.read_csv(stage1 / "stage1_clean.csv")
    r2 = pd.read_csv(stage2 / "stage2_respondents.csv")
    ch = pd.read_csv(stage2 / "stage2_choices.csv")
    r3 = pd.read_csv(stage3 / "stage3_respondents.csv")

    uni = (s1[["university", "sector"]].drop_duplicates().sort_values("university")
           .reset_index(drop=True))
    uni.insert(0, "university_id", np.arange(1, len(uni) + 1))

    resp = (s1.merge(uni[["university_id", "university"]], on="university")
            .merge(r3[["ResponseId", "pj_z", "pj_group"]], on="ResponseId", validate="one_to_one")
            .merge(r2[["ResponseId", "design_version"]],
                   on="ResponseId", validate="one_to_one"))
    resp = resp.rename(columns={"ResponseId": "response_id"})
    resp = resp[["response_id", "university_id", "role", "gender", "age_band", "education",
                 "college_field", "recruit_exp", "ai_experience", "duration_sec", "fast_responder",
                 "straightline_both", "regression_sample", "climate_12", "climate_org",
                 "climate_pers", "pj", "pj_z", "pj_group", "design_version"]]

    item_codes = [c for c, _, _ in ITEMS]
    items = (s1.rename(columns={"ResponseId": "response_id"})
             .melt(id_vars="response_id", value_vars=item_codes, var_name="item_code",
                   value_name="value"))
    items["value"] = items["value"].astype(int)

    pos = r2.rename(columns={"ResponseId": "response_id"}).melt(
        id_vars="response_id", value_vars=[f"row_{a}" for a in CODES],
        var_name="attribute_code", value_name="row_position")
    pos["attribute_code"] = pos["attribute_code"].str.replace("row_", "", regex=False)

    ch = ch.rename(columns={"ResponseId": "response_id"})
    tasks = (ch[ch["chosen"] == 1][["response_id", "task", "profile"]]
             .rename(columns={"profile": "chosen_profile"}))
    profiles = ch[["response_id", "task", "profile"] + CODES]

    return {"university": uni, "respondent": resp, "item_response": items,
            "attribute_position": pos, "choice_task": tasks, "profile": profiles,
            "_choices": ch, "_stage1": s1}


def build(tables: dict[str, pd.DataFrame], out: Path) -> None:
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript((HERE / "schema.sql").read_text(encoding="utf-8"))

    con.executemany("INSERT INTO scale_item VALUES (?, ?, ?)", ITEMS)
    con.executemany("INSERT INTO attribute VALUES (?, ?, ?, ?)", ATTRIBUTES)
    con.executemany("INSERT INTO attribute_level VALUES (?, ?, ?)",
                    [(a, l, lab) for a in CODES for l, lab in LEVELS[a].items()])
    con.executemany("INSERT INTO university VALUES (?, ?, ?)",
                    tables["university"][["university_id", "university", "sector"]]
                    .itertuples(index=False, name=None))
    for name in ("respondent", "item_response", "attribute_position", "choice_task", "profile"):
        df = tables[name].astype(object).where(tables[name].notna(), None)
        cols = ", ".join(df.columns)
        marks = ", ".join("?" * len(df.columns))
        con.executemany(f"INSERT INTO {name} ({cols}) VALUES ({marks})",
                        df.itertuples(index=False, name=None))
    con.commit()
    verify(con, tables)
    con.execute("VACUUM")
    con.close()


def verify(con: sqlite3.Connection, tables: dict[str, pd.DataFrame]) -> None:
    """The database must reproduce the clean files exactly."""
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise ValueError(f"Foreign key violations: {fk[:5]}")

    counts = {r[0]: r[1] for r in con.execute(
        "SELECT 'respondent', COUNT(*) FROM respondent UNION ALL "
        "SELECT 'choice_task', COUNT(*) FROM choice_task UNION ALL "
        "SELECT 'profile', COUNT(*) FROM profile UNION ALL "
        "SELECT 'item_response', COUNT(*) FROM item_response")}
    expected = {"respondent": 317, "choice_task": 317 * 9, "profile": 317 * 18,
                "item_response": 317 * len(ITEMS)}
    if counts != expected:
        raise ValueError(f"Unexpected row counts {counts}")

    # Choices rebuilt through the view must match the clean choice data
    db = pd.read_sql("SELECT response_id, task, profile, chosen, privacy, transparency, "
                     "usefulness, ease FROM profile_choice", con)
    ref = tables["_choices"][["response_id", "task", "profile", "chosen"] + CODES]
    m = ref.merge(db, on=["response_id", "task", "profile"], suffixes=("", "_db"),
                  validate="one_to_one")
    for c in ["chosen"] + CODES:
        if not (m[c] == m[f"{c}_db"]).all():
            raise ValueError(f"Column {c} differs from the clean choice data")

    # Scale composites recomputed from the item table must match Stage 1
    comp = pd.read_sql(
        "SELECT ir.response_id, "
        "AVG(CASE WHEN si.scale = 'Diversity and inclusion climate' THEN ir.value END) AS climate_12, "
        "AVG(CASE WHEN si.scale = 'Expected procedural justice' THEN ir.value END) AS pj "
        "FROM item_response AS ir JOIN scale_item AS si USING (item_code) "
        "GROUP BY ir.response_id", con).set_index("response_id")
    s1 = tables["_stage1"].set_index("ResponseId").loc[comp.index]
    gap = max((comp["climate_12"] - s1["climate_12"]).abs().max(),
              (comp["pj"] - s1["pj"]).abs().max())
    if gap > 1e-9:
        raise ValueError("Composites recomputed in the database differ from Stage 1")
    print(f"Database verified: {counts}, largest composite difference {gap:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="../stage1/data")
    ap.add_argument("--stage2", default="../stage2/data")
    ap.add_argument("--stage3", default="../stage3/data")
    ap.add_argument("--out", default="data/thesis.db")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(load(Path(args.stage1), Path(args.stage2), Path(args.stage3)), out)
    print(f"Written {out}")


if __name__ == "__main__":
    main()
