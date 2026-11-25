"""
Stage 3 analysis. Preference variation across stakeholder role, university
sector and expected procedural justice.

Reads the outputs of 01_prepare_subgroups.py and writes every table (CSV),
every figure (PNG, 300 dpi) and a JSON file of the key estimates.

    1  Subgroup composition
    2  Average marginal component effects within each subgroup
    3  Marginal means within each subgroup, and differences in marginal means
       between subgroups
    4  Pooled interaction models, one per moderator, with respondent-clustered
       standard errors, Benjamini-Hochberg correction across the interaction
       terms of each moderator and an omnibus test of all interactions
    5  Expected procedural justice as a continuous moderator, and role
       differences adjusted for expected procedural justice
    6  Conditional logit importance shares and the governance share within each
       subgroup, and differences between subgroups, with respondent cluster
       bootstrap intervals (respondents resampled within each subgroup)

Usage
    python 02_subgroup_analysis.py --data data --out outputs --stage2-out ../stage2/outputs
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

SEED = 12345
N_BOOT = 2000

ATTRS = ["privacy", "transparency", "usefulness", "ease"]
ATTR_NAME = {"privacy": "Ethical and privacy assurance",
             "transparency": "Algorithmic transparency",
             "usefulness": "Perceived usefulness",
             "ease": "Perceived ease of use"}
ATTR_SHORT = {"privacy": "Assurance", "transparency": "Transparency",
              "usefulness": "Usefulness", "ease": "Ease of use"}
LEVEL_NAME = {a: {1: "Low", 2: "Moderate", 3: "High"} for a in ATTRS}
LEVEL_NAME["ease"] = {1: "Difficult", 2: "Moderate", 3: "Easy"}
DUMMIES = [f"{a}_{l}" for a in ATTRS for l in (2, 3)]

MODERATORS = [
    ("role", "Stakeholder role", ["Academic staff", "Professional staff", "Leaders"]),
    ("sector", "University sector", ["Public", "Private"]),
    ("pj_group", "Expected procedural justice", ["Lower expected PJ", "Higher expected PJ"]),
]


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------

def add_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for a in ATTRS:
        for l in (2, 3):
            df[f"{a}_{l}"] = (df[a] == l).astype(float)
    return df


def dummy_label(col: str) -> str:
    a, l = col.rsplit("_", 1)
    return f"{ATTR_NAME[a]}: {LEVEL_NAME[a][int(l)].lower()}"


def ols_cluster(y: pd.Series, X: pd.DataFrame, groups: pd.Series):
    """OLS with standard errors clustered by respondent (CR1, t with G - 1 df)."""
    return sm.OLS(y, X).fit(cov_type="cluster",
                            cov_kwds={"groups": pd.factorize(groups)[0]}, use_t=True)


def amce_table(sub: pd.DataFrame) -> pd.DataFrame:
    fit = ols_cluster(sub["chosen"], sm.add_constant(sub[DUMMIES]), sub["ResponseId"])
    ci = fit.conf_int()
    return pd.DataFrame([{"term": c, "label": dummy_label(c), "b": fit.params[c],
                          "se": fit.bse[c], "ci_lo": ci.loc[c, 0], "ci_hi": ci.loc[c, 1],
                          "p": fit.pvalues[c]} for c in DUMMIES])


def mm_table(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in ATTRS:
        for l in (1, 2, 3):
            s = sub[sub[a] == l]
            fit = ols_cluster(s["chosen"], pd.DataFrame({"const": np.ones(len(s))}, index=s.index),
                              s["ResponseId"])
            ci = fit.conf_int().iloc[0]
            rows.append({"attr": a, "attribute": ATTR_NAME[a], "level_code": l,
                         "level": LEVEL_NAME[a][l], "mm": fit.params.iloc[0],
                         "se": fit.bse.iloc[0], "ci_lo": ci.iloc[0], "ci_hi": ci.iloc[1]})
    return pd.DataFrame(rows)


def interaction_model(df: pd.DataFrame, mods: list[str]):
    """Choice on the attribute-level indicators, the moderator columns and
    every indicator-by-moderator product."""
    X = df[DUMMIES].copy()
    inter = []
    for m in mods:
        X[m] = df[m]
        for c in DUMMIES:
            X[f"{c}_x_{m}"] = df[c] * df[m]
            inter.append(f"{c}_x_{m}")
    fit = ols_cluster(df["chosen"], sm.add_constant(X), df["ResponseId"])
    return fit, inter


def joint_test(fit, terms: list[str]) -> dict:
    ft = fit.f_test(" = 0, ".join(terms) + " = 0")
    return {"F": float(ft.fvalue), "df1": int(ft.df_num), "df2": int(ft.df_denom),
            "p": float(ft.pvalue)}


def differences(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """One row per task: indicator differences (Option 1 minus Option 2) and
    whether Option 1 was chosen. A conditional logit with two alternatives is
    a binary logit on these differences without a constant."""
    cols = DUMMIES + (extra or [])
    w = df.sort_values(["ResponseId", "task", "profile"])
    a = w[w["profile"] == 1].set_index(["ResponseId", "task"])
    b = w[w["profile"] == 2].set_index(["ResponseId", "task"])
    d = a[cols] - b[cols]
    d["y"] = a["chosen"]
    return d.reset_index()


def logit_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100).params


def shares(beta: np.ndarray) -> dict:
    """Importance shares from the logit coefficients on the eight indicators
    (effects-coded moderator_name-worth ranges)."""
    b = dict(zip(DUMMIES, beta[:len(DUMMIES)]))
    rng = {}
    for a in ATTRS:
        u = np.array([0.0, b[f"{a}_2"], b[f"{a}_3"]])
        rng[a] = u.max() - u.min()
    tot = sum(rng.values())
    out = {a: rng[a] / tot for a in ATTRS}
    out["governance"] = out["privacy"] + out["transparency"]
    return out


def boot_p(draws: np.ndarray) -> float:
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

NAVY, ORANGE, GREEN = "#1a476f", "#e37e00", "#55752f"
GREY = "#8c8c8c"
INK = "#000000"
GROUP_COL = [NAVY, ORANGE, GREEN]
GROUP_MARK = ["o", "s", "^"]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out",
        "ytick.direction": "out", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": False, "legend.frameon": False, "savefig.dpi": 300,
        "figure.dpi": 100,
    })


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mm_group_plot(mm: pd.DataFrame, groups: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    k = len(groups)
    offsets = np.linspace(-0.22, 0.22, k) if k > 1 else [0.0]
    y, ticks, labels, heads = 0.0, [], [], []
    for a in ATTRS:
        heads.append(y)
        y -= 1
        for l in (1, 2, 3):
            for j, g in enumerate(groups):
                r = mm[(mm["subgroup"] == g) & (mm["attr"] == a) & (mm["level_code"] == l)].iloc[0]
                ax.errorbar(r["mm"], y - offsets[j], xerr=[[r["mm"] - r["ci_lo"]], [r["ci_hi"] - r["mm"]]],
                            fmt=GROUP_MARK[j], color=GROUP_COL[j], ecolor=GROUP_COL[j], ms=4.5,
                            elinewidth=1.0, capsize=2.5, zorder=3,
                            label=g if (a == ATTRS[0] and l == 1) else None)
            ticks.append(y)
            labels.append(LEVEL_NAME[a][l])
            y -= 1
        y -= 0.4
    ax.axvline(0.5, color=GREY, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for h, a in zip(heads, ATTRS):
        ax.text(-0.02, h, ATTR_NAME[a], transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(y + 0.6, 0.7)
    ax.set_xlabel("Marginal mean, probability of being chosen, 95% CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=k)
    save(fig, path)


def governance_plot(gov: pd.DataFrame, full: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    rows = [("Full sample", full["est"], None, None, INK, "D")]
    for (moderator_name, _, groups), col_set in zip(MODERATORS, [GROUP_COL, GROUP_COL, GROUP_COL]):
        for j, g in enumerate(groups):
            r = gov[gov["subgroup"] == g].iloc[0]
            rows.append((g, r["governance"], r["ci_lo"], r["ci_hi"], col_set[j], GROUP_MARK[j]))
    y = 0.0
    ticks, labels = [], []
    breaks = {1, 4, 6}
    for i, (lab, est, lo, hi, col, mk) in enumerate(rows):
        if i in breaks:
            y -= 0.5
        if lo is None:
            ax.plot(est * 100, y, marker=mk, color=col, ms=5, zorder=3)
            ax.text(est * 100, y, f"  {est * 100:.1f}%", va="center", ha="left", fontsize=8)
        else:
            ax.errorbar(est * 100, y, xerr=[[(est - lo) * 100], [(hi - est) * 100]], fmt=mk,
                        color=col, ecolor=col, ms=5, elinewidth=1.1, capsize=3, zorder=3)
            ax.text(hi * 100, y, f"  {est * 100:.1f}%", va="center", ha="left", fontsize=8)
        ticks.append(y)
        labels.append(lab)
        y -= 1
    ax.axvline(50, color=GREY, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(30, 75)
    ax.set_xlabel("Governance share of attribute importance (%), 95% bootstrap CI")
    save(fig, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--stage2-out", default="../stage2/outputs")
    args = ap.parse_args()
    out = Path(args.out)
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    style()
    R = {}

    df = add_dummies(pd.read_csv(Path(args.data) / "stage3_choices.csv"))
    resp = pd.read_csv(Path(args.data) / "stage3_respondents.csv")
    n_resp = df["ResponseId"].nunique()
    tcrit = stats.t.ppf(0.975, n_resp - 1) + stats.t.ppf(0.80, n_resp - 1)

    # ---------------- 1. composition --------------------------------------
    comp = []
    for col, moderator_name, groups in MODERATORS:
        for g in groups:
            n = int((resp[col] == g).sum())
            comp.append({"moderator": moderator_name, "subgroup": g, "respondents": n,
                         "tasks": n * 9, "profiles": n * 18})
    pd.DataFrame(comp).to_csv(tdir / "subgroup_composition.csv", index=False)

    full_amce = amce_table(df)
    R["full_sample_amce"] = dict(zip(full_amce["term"], full_amce["b"]))

    amce_rows, mm_rows, int_rows, mmd_rows, omnibus = [], [], [], [], []
    for col, moderator_name, groups in MODERATORS:
        ref = groups[0]
        # ---------------- 2 and 3. within-subgroup estimates ---------------
        for g in groups:
            sub = df[df[col] == g]
            t = amce_table(sub)
            t.insert(0, "subgroup", g)
            t.insert(0, "moderator", moderator_name)
            amce_rows.append(t)
            m = mm_table(sub)
            m.insert(0, "subgroup", g)
            m.insert(0, "moderator", moderator_name)
            mm_rows.append(m)

        # ---------------- 4. pooled interaction model ----------------------
        mods = []
        tmp = df.copy()
        for g in groups[1:]:
            name = "g_" + g.lower().replace(" ", "_")
            tmp[name] = (tmp[col] == g).astype(float)
            mods.append((g, name))
        fit, inter = interaction_model(tmp, [n for _, n in mods])
        ci = fit.conf_int()
        moderator_rows = []
        for g, name in mods:
            for c in DUMMIES:
                term = f"{c}_x_{name}"
                moderator_rows.append({"moderator": moderator_name, "contrast": f"{g} vs {ref}", "term": c,
                                  "label": dummy_label(c), "diff": fit.params[term],
                                  "se": fit.bse[term], "ci_lo": ci.loc[term, 0],
                                  "ci_hi": ci.loc[term, 1], "p": fit.pvalues[term],
                                  "mde": tcrit * fit.bse[term]})
            jt = joint_test(fit, [f"{c}_x_{name}" for c in DUMMIES])
            omnibus.append({"moderator": moderator_name, "test": f"{g} vs {ref}", **jt})
        moderator_df = pd.DataFrame(moderator_rows)
        moderator_df["p_bh"] = multipletests(moderator_df["p"], method="fdr_bh")[1]
        int_rows.append(moderator_df)
        if len(mods) > 1:
            omnibus.append({"moderator": moderator_name, "test": "All subgroups equal",
                            **joint_test(fit, inter)})

        # differences in marginal means, each level, against the reference
        for a in ATTRS:
            for l in (1, 2, 3):
                s = tmp[tmp[a] == l]
                X = sm.add_constant(s[[n for _, n in mods]])
                f = ols_cluster(s["chosen"], X, s["ResponseId"])
                for g, name in mods:
                    mmd_rows.append({"moderator": moderator_name, "contrast": f"{g} vs {ref}",
                                     "attribute": ATTR_NAME[a], "level": LEVEL_NAME[a][l],
                                     "diff": f.params[name], "se": f.bse[name],
                                     "p": f.pvalues[name]})

    amce_sg = pd.concat(amce_rows, ignore_index=True)
    amce_sg.to_csv(tdir / "amce_by_subgroup.csv", index=False)
    mm_sg = pd.concat(mm_rows, ignore_index=True)
    # Within every subgroup and attribute, the marginal means should rise from
    # the lowest to the highest level
    order_ok = all(g.sort_values("level_code")["mm"].is_monotonic_increasing
                   for _, g in mm_sg.groupby(["subgroup", "attr"]))
    R["levels_ordered_in_every_subgroup"] = bool(order_ok)
    mm_sg.drop(columns=["attr", "level_code"]).to_csv(tdir / "marginal_means_by_subgroup.csv",
                                                     index=False)
    inter_df = pd.concat(int_rows, ignore_index=True)
    inter_df.to_csv(tdir / "interaction_tests.csv", index=False)
    mmd = pd.DataFrame(mmd_rows)
    mmd["mde"] = tcrit * mmd["se"]
    mmd["p_bh"] = np.nan
    for (moderator_name, con), idx in mmd.groupby(["moderator", "contrast"]).groups.items():
        mmd.loc[idx, "p_bh"] = multipletests(mmd.loc[idx, "p"], method="fdr_bh")[1]
    mmd.to_csv(tdir / "marginal_mean_differences.csv", index=False)
    omni = pd.DataFrame(omnibus)
    omni.to_csv(tdir / "omnibus_tests.csv", index=False)

    # ---------------- 5. continuous expected procedural justice ------------
    fit, inter = interaction_model(df, ["pj_z"])
    ci = fit.conf_int()
    cont = pd.DataFrame([{"term": c, "label": dummy_label(c),
                          "coef": fit.params[f"{c}_x_pj_z"], "se": fit.bse[f"{c}_x_pj_z"],
                          "ci_lo": ci.loc[f"{c}_x_pj_z", 0], "ci_hi": ci.loc[f"{c}_x_pj_z", 1],
                          "p": fit.pvalues[f"{c}_x_pj_z"],
                          "mde": tcrit * fit.bse[f"{c}_x_pj_z"]} for c in DUMMIES])
    cont["p_bh"] = multipletests(cont["p"], method="fdr_bh")[1]
    cont.to_csv(tdir / "pj_continuous_interactions.csv", index=False)
    omni_cont = joint_test(fit, inter)
    R["pj_continuous_omnibus"] = omni_cont

    # Role differences adjusted for expected procedural justice, since the
    # roles differ in their mean expectations (Chapter 3)
    tmp = df.copy()
    tmp["g_professional"] = (tmp["role"] == "Professional staff").astype(float)
    tmp["g_leaders"] = (tmp["role"] == "Leaders").astype(float)
    fit_adj, _ = interaction_model(tmp, ["g_professional", "g_leaders", "pj_z"])
    R["role_adjusted_for_pj"] = {
        "transparency_2_professional": {"diff": float(fit_adj.params["transparency_2_x_g_professional"]),
                                        "p": float(fit_adj.pvalues["transparency_2_x_g_professional"])},
        "transparency_3_professional": {"diff": float(fit_adj.params["transparency_3_x_g_professional"]),
                                        "p": float(fit_adj.pvalues["transparency_3_x_g_professional"])},
        "omnibus_professional": joint_test(fit_adj, [f"{c}_x_g_professional" for c in DUMMIES]),
        "omnibus_leaders": joint_test(fit_adj, [f"{c}_x_g_leaders" for c in DUMMIES])}

    # ---------------- 6. importance shares, bootstrap ----------------------
    dfz = df.copy()
    for c in DUMMIES:
        dfz[f"{c}_z"] = dfz[c] * dfz["pj_z"]
    ZC = [f"{c}_z" for c in DUMMIES]
    d = differences(dfz, ZC)
    rmap = resp.set_index("ResponseId")
    for col, _, _ in MODERATORS:
        d[col] = d["ResponseId"].map(rmap[col])
    X8 = d[DUMMIES].to_numpy()
    X16 = d[DUMMIES + ZC].to_numpy()
    y = d["y"].to_numpy()
    rows_by_resp = {rid: np.asarray(ix) for rid, ix in d.groupby("ResponseId").indices.items()}

    def fit_rows(idx, X):
        return logit_beta(X[idx], y[idx])

    def cont_shares(beta):
        main, slope = beta[:8], beta[8:]
        lo, hi = shares(main - slope), shares(main + slope)
        return lo, hi

    units = {"Full sample": np.array(sorted(rows_by_resp))}
    for col, _, groups in MODERATORS:
        for g in groups:
            units[g] = np.array(sorted(resp.loc[resp[col] == g, "ResponseId"]))

    def idx_for(ids):
        return np.concatenate([rows_by_resp[i] for i in ids])

    point = {k: shares(fit_rows(idx_for(ids), X8)) for k, ids in units.items()}
    cl, ch = cont_shares(fit_rows(idx_for(units["Full sample"]), X16))
    point_cont = {"governance_at_minus1": cl["governance"], "governance_at_plus1": ch["governance"],
                  "difference": ch["governance"] - cl["governance"]}

    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in units}
    draws_cont = []
    for _ in range(N_BOOT):
        for k, ids in units.items():
            b = rng.choice(ids, len(ids), replace=True)
            draws[k].append(shares(fit_rows(idx_for(b), X8)))
        b = rng.choice(units["Full sample"], len(units["Full sample"]), replace=True)
        lo, hi = cont_shares(fit_rows(idx_for(b), X16))
        draws_cont.append(hi["governance"] - lo["governance"])

    keys = ATTRS + ["governance"]
    imp_rows = []
    for k in units:
        arr = {q: np.array([dd[q] for dd in draws[k]]) for q in keys}
        moderator_name = next((p for c, p, gs in MODERATORS if k in gs), "")
        row = {"moderator": moderator_name, "subgroup": k, "respondents": len(units[k])}
        for q in keys:
            row[q] = point[k][q]
            row[f"{q}_ci_lo"], row[f"{q}_ci_hi"] = np.percentile(arr[q], [2.5, 97.5])
        row["governance_p_vs_half"] = boot_p(arr["governance"] - 0.5)
        imp_rows.append(row)
    imp = pd.DataFrame(imp_rows)
    # The full-sample row reports the Stage 2 estimates and intervals, so the
    # two chapters show the same values. The full sample is also bootstrapped
    # here as a check that the two bootstraps agree.
    R["full_sample_check"] = imp[imp["subgroup"] == "Full sample"].iloc[0].to_dict()
    s2 = pd.read_csv(Path(args.stage2_out) / "tables" / "importance.csv")
    s2_name = {"Ethical and privacy assurance": "privacy", "Algorithmic transparency": "transparency",
               "Perceived usefulness": "usefulness", "Perceived ease of use": "ease"}
    with open(Path(args.stage2_out) / "key_results.json") as f:
        s2_gov = json.load(f)["governance_share"]
    i0 = imp.index[imp["subgroup"] == "Full sample"][0]
    for _, r in s2.iterrows():
        a = s2_name[r["attribute"]]
        imp.loc[i0, [a, f"{a}_ci_lo", f"{a}_ci_hi"]] = [r["share"], r["ci_lo"], r["ci_hi"]]
    imp.loc[i0, ["governance", "governance_ci_lo", "governance_ci_hi", "governance_p_vs_half"]] = [
        s2_gov["estimate"], s2_gov["ci"][0], s2_gov["ci"][1], s2_gov["p_vs_half"]]
    if abs(imp.loc[i0, "governance"] - R["full_sample_check"]["governance"]) > 1e-9:
        raise ValueError("Full-sample estimates differ between Stage 2 and Stage 3")
    imp.to_csv(tdir / "importance_by_subgroup.csv", index=False)

    gdiff = []
    for col, moderator_name, groups in MODERATORS:
        ref = groups[0]
        for g in groups[1:]:
            for q in keys:
                dd = np.array([a[q] for a in draws[g]]) - np.array([a[q] for a in draws[ref]])
                lo, hi = np.percentile(dd, [2.5, 97.5])
                gdiff.append({"moderator": moderator_name, "contrast": f"{g} vs {ref}",
                              "quantity": "Governance share" if q == "governance" else ATTR_NAME[q],
                              "diff": point[g][q] - point[ref][q], "ci_lo": lo, "ci_hi": hi,
                              "p": boot_p(dd)})
        if len(groups) == 3:
            q = "governance"
            dd = np.array([a[q] for a in draws[groups[2]]]) - np.array([a[q] for a in draws[groups[1]]])
            lo, hi = np.percentile(dd, [2.5, 97.5])
            gdiff.append({"moderator": moderator_name, "contrast": f"{groups[2]} vs {groups[1]}",
                          "quantity": "Governance share",
                          "diff": point[groups[2]][q] - point[groups[1]][q],
                          "ci_lo": lo, "ci_hi": hi, "p": boot_p(dd)})
    gdiff = pd.DataFrame(gdiff)
    gdiff.to_csv(tdir / "importance_differences.csv", index=False)

    dc = np.array(draws_cont)
    lo, hi = np.percentile(dc, [2.5, 97.5])
    point_cont.update({"ci_lo": float(lo), "ci_hi": float(hi), "p": boot_p(dc)})
    R["pj_continuous_governance_share"] = point_cont

    # ---------------- key results -----------------------------------------
    R["n_respondents"] = n_resp
    R["omnibus"] = omni.to_dict(orient="records")
    R["interaction_summary"] = {
        moderator_name: {"min_p": float(g["p"].min()), "min_p_bh": float(g["p_bh"].min()),
               "max_abs_diff": float(g["diff"].abs().max()),
               "se_range": [float(g["se"].min()), float(g["se"].max())],
               "mde_range": [float(g["mde"].min()), float(g["mde"].max())]}
        for moderator_name, g in inter_df.groupby("moderator")}
    R["pj_continuous_summary"] = {"min_p": float(cont["p"].min()),
                                  "min_p_bh": float(cont["p_bh"].min()),
                                  "coef_range": [float(cont["coef"].min()), float(cont["coef"].max())],
                                  "mde_range": [float(cont["mde"].min()), float(cont["mde"].max())]}
    R["mm_difference_summary"] = {
        f"{p} | {c}": {"min_p": float(g["p"].min()), "min_p_bh": float(g["p_bh"].min()),
                       "max_abs_diff": float(g["diff"].abs().max()),
                       "mde_range": [float(g["mde"].min()), float(g["mde"].max())]}
        for (p, c), g in mmd.groupby(["moderator", "contrast"])}
    R["governance_share"] = {r["subgroup"]: {"est": r["governance"], "ci": [r["governance_ci_lo"],
                             r["governance_ci_hi"]], "p_vs_half": r["governance_p_vs_half"]}
                             for _, r in imp.iterrows()}
    with open(out / "key_results.json", "w") as f:
        json.dump(R, f, indent=2, default=float)

    # ---------------- figures ---------------------------------------------
    names = {"role": "Figure_5_2_marginal_means_by_role.png",
             "sector": "Figure_5_3_marginal_means_by_sector.png",
             "pj_group": "Figure_5_4_marginal_means_by_expected_pj.png"}
    for col, moderator_name, groups in MODERATORS:
        mm_group_plot(mm_sg[mm_sg["moderator"] == moderator_name], groups, fdir / names[col])
    full = imp[imp["subgroup"] == "Full sample"].iloc[0]
    governance_plot(imp.rename(columns={"governance_ci_lo": "ci_lo", "governance_ci_hi": "ci_hi"}),
                    {"est": full["governance"], "lo": full["governance_ci_lo"],
                     "hi": full["governance_ci_hi"]},
                    fdir / "Figure_5_5_governance_share_by_subgroup.png")

    print("Omnibus tests")
    print(omni.round(3).to_string(index=False))
    print("Continuous expected PJ", {k: round(v, 3) for k, v in omni_cont.items()})
    print(imp[["subgroup", "governance", "governance_ci_lo", "governance_ci_hi"]].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
