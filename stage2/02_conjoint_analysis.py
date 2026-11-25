"""
Stage 2 analysis. Choice-based conjoint experiment on AI recruitment systems.

Reads the outputs of 01_prepare_conjoint.py and writes every table (CSV),
every figure (PNG, 300 dpi) and a JSON file of the key estimates.

    1  Design diagnostics
    2  Average marginal component effects (linear probability model,
       respondent-clustered standard errors, Benjamini-Hochberg correction),
       with tests of differences between attributes and between steps
    3  Marginal means, with tests against an even split
    4  Conditional logit part-worths, attribute importance shares and the
       governance share, with respondent cluster bootstrap intervals
    5  Head-to-head choice probabilities for complete system configurations
    6  Trade-off ratios between attributes
    7  Robustness checks

Usage
    python 02_conjoint_analysis.py --data data --out outputs
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
GOVERNANCE = ["privacy", "transparency"]
EFFICIENCY = ["usefulness", "ease"]
LEVEL_NAME = {a: {1: "Low", 2: "Moderate", 3: "High"} for a in ATTRS}
LEVEL_NAME["ease"] = {1: "Difficult", 2: "Moderate", 3: "Easy"}
DUMMIES = [f"{a}_{l}" for a in ATTRS for l in (2, 3)]


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


def lpm(df: pd.DataFrame, extra: list[str] | None = None):
    """Linear probability model of choice on attribute-level indicators, with
    standard errors clustered by respondent (CR1, t with G - 1 df)."""
    cols = DUMMIES + (extra or [])
    X = sm.add_constant(df[cols])
    return sm.OLS(df["chosen"], X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["ResponseId"])[0]},
        use_t=True)


def amce_table(fit) -> pd.DataFrame:
    rows = []
    ci = fit.conf_int()
    for c in DUMMIES:
        rows.append({"term": c, "label": dummy_label(c), "attribute": c.rsplit("_", 1)[0],
                     "b": fit.params[c], "se": fit.bse[c], "p": fit.pvalues[c],
                     "ci_lo": ci.loc[c, 0], "ci_hi": ci.loc[c, 1]})
    t = pd.DataFrame(rows)
    t["p_bh"] = multipletests(t["p"], method="fdr_bh")[1]
    return t


def differences(df: pd.DataFrame) -> pd.DataFrame:
    """One row per task: attribute-indicator differences (Option 1 minus
    Option 2) and whether Option 1 was chosen. A conditional logit with two
    alternatives is a binary logit on these differences without a constant."""
    w = df.sort_values(["ResponseId", "task", "profile"])
    a = w[w["profile"] == 1].set_index(["ResponseId", "task"])
    b = w[w["profile"] == 2].set_index(["ResponseId", "task"])
    d = a[DUMMIES] - b[DUMMIES]
    d["y"] = a["chosen"]
    return d.reset_index()


def clogit(d: pd.DataFrame, cluster: bool = True):
    kw = {}
    if cluster:
        kw = {"cov_type": "cluster",
              "cov_kwds": {"groups": pd.factorize(d["ResponseId"])[0]}}
    return sm.Logit(d["y"], d[DUMMIES]).fit(disp=0, **kw)


def partworths(beta: pd.Series) -> dict:
    """Level utilities centred within each attribute (effects coding)."""
    out = {}
    for a in ATTRS:
        u = np.array([0.0, beta[f"{a}_2"], beta[f"{a}_3"]])
        out[a] = u - u.mean()
    return out


def importance(beta: pd.Series) -> dict:
    pw = partworths(beta)
    rng = {a: pw[a].max() - pw[a].min() for a in ATTRS}
    tot = sum(rng.values())
    return {a: rng[a] / tot for a in ATTRS} | {"range": rng, "total": tot}


def utility(beta: pd.Series, levels: dict) -> float:
    return sum(beta[f"{a}_{l}"] for a, l in levels.items() if l > 1)


CONFIGS = {
    "all_high": {a: 3 for a in ATTRS},
    "all_low": {a: 1 for a in ATTRS},
    "all_moderate": {a: 2 for a in ATTRS},
    "gov_high_eff_low": {"privacy": 3, "transparency": 3, "usefulness": 1, "ease": 1},
    "gov_low_eff_high": {"privacy": 1, "transparency": 1, "usefulness": 3, "ease": 3},
}
CONFIG_LABEL = {
    "all_high": "All four attributes high",
    "all_low": "All four attributes low",
    "all_moderate": "All four attributes moderate",
    "gov_high_eff_low": "Governance high, efficiency low",
    "gov_low_eff_high": "Governance low, efficiency high",
}
MATCHUPS = [
    ("gov_high_eff_low", "gov_low_eff_high"),
    ("gov_high_eff_low", "all_moderate"),
    ("gov_low_eff_high", "all_moderate"),
    ("all_high", "all_moderate"),
    ("all_low", "all_moderate"),
]


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def quantities(beta: pd.Series) -> dict:
    """All functions of the logit coefficients reported in the chapter."""
    imp = importance(beta)
    q = {f"imp_{a}": imp[a] for a in ATTRS}
    q["imp_governance"] = imp["privacy"] + imp["transparency"]
    for x, y in MATCHUPS:
        q[f"p_{x}_vs_{y}"] = logistic(utility(beta, CONFIGS[x]) - utility(beta, CONFIGS[y]))
    # Trade-off ratios: utility of moving an attribute from low to high,
    # relative to the same move in perceived usefulness
    for a in ATTRS:
        q[f"ratio_{a}_to_usefulness"] = beta[f"{a}_3"] / beta["usefulness_3"]
    return q


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

NAVY, ORANGE, GREEN, MAROON = "#1a476f", "#e37e00", "#55752f", "#90353b"
GREY, LIGHT = "#8c8c8c", "#c8c8c8"
INK = "#000000"
ATTR_COL = {"privacy": NAVY, "transparency": NAVY, "usefulness": ORANGE, "ease": ORANGE}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out",
        "ytick.direction": "out", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": False, "legend.frameon": False, "savefig.dpi": 300,
        "figure.dpi": 100,
    })


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def level_plot(ax, rows, xlabel, ref, ref_style="--", fmt="{:+.3f}", baseline_text="baseline"):
    """Rows grouped by attribute: (attribute, level label, est, lo, hi, is_baseline)."""
    y, ticks, labels = 0, [], []
    heads = []
    for a in ATTRS:
        heads.append(y)
        y -= 1
        for r in [r for r in rows if r[0] == a]:
            _, lab, est, lo, hi, base = r
            if base:
                ax.plot(est, y, marker="o", mfc="white", mec=ATTR_COL[a], ms=5, mew=1.1, zorder=3)
                ax.text(est, y, f"  {baseline_text}", va="center", ha="left", fontsize=8, color=GREY)
            else:
                ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o", color=ATTR_COL[a],
                            ecolor=ATTR_COL[a], elinewidth=1.1, capsize=3, ms=5, zorder=3)
                ax.text(hi, y, "  " + fmt.format(est), va="center", ha="left", fontsize=8)
            ticks.append(y)
            labels.append(lab)
            y -= 1
        y -= 0.4
    ax.axvline(ref, color=GREY, linewidth=0.8, linestyle=ref_style, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for h, a in zip(heads, ATTRS):
        ax.text(-0.02, h, ATTR_NAME[a], transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(y + 0.6, 0.7)
    ax.set_xlabel(xlabel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: Path, out_dir: Path) -> None:
    tdir, fdir = out_dir / "tables", out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    style()
    rng = np.random.default_rng(SEED)
    R = {}

    df = add_dummies(pd.read_csv(data_dir / "stage2_choices.csv"))
    alldf = add_dummies(pd.read_csv(data_dir / "stage2_choices_all.csv"))
    resp = pd.read_csv(data_dir / "stage2_respondents.csv")
    n_resp, n_tasks = df["ResponseId"].nunique(), len(df) // 2
    R["N"] = {"respondents": n_resp, "tasks": n_tasks, "profiles": len(df)}

    # ---------------- 1. design diagnostics -----------------------------
    freq = []
    for a in ATTRS:
        vc = df[a].value_counts().sort_index()
        chi = stats.chisquare(vc.values)
        for l, n in vc.items():
            freq.append({"attribute": ATTR_NAME[a], "level": LEVEL_NAME[a][l], "n": n,
                         "share": n / len(df), "chi2": chi.statistic, "p": chi.pvalue})
    pd.DataFrame(freq).to_csv(tdir / "level_frequencies.csv", index=False)

    # ---------------- 2. AMCEs ------------------------------------------
    m = lpm(df)
    amce = amce_table(m)
    amce.to_csv(tdir / "amce.csv", index=False)
    R["amce"] = amce.set_index("term")[["b", "se", "p", "p_bh", "ci_lo", "ci_hi"]].to_dict("index")
    R["amce_model"] = {"intercept": m.params["const"], "R2": m.rsquared, "clusters": n_resp}
    # Step from moderate to high within each attribute
    steps = []
    for a in ATTRS:
        L = np.zeros(len(m.params))
        L[list(m.params.index).index(f"{a}_3")] = 1
        L[list(m.params.index).index(f"{a}_2")] = -1
        t = m.t_test(L)
        steps.append({"attribute": ATTR_NAME[a], "step": "moderate to high",
                      "b": float(t.effect[0]), "se": float(t.sd[0][0]), "p": float(t.pvalue),
                      "ci_lo": float(t.conf_int()[0][0]), "ci_hi": float(t.conf_int()[0][1])})
    pd.DataFrame(steps).to_csv(tdir / "amce_upper_step.csv", index=False)
    R["upper_step"] = {s["attribute"]: s for s in steps}
    # Precision achieved: smallest effect detectable with 80% power, two-sided 5%
    tcrit = stats.t.ppf(0.975, n_resp - 1) + stats.t.ppf(0.80, n_resp - 1)
    R["mde_amce"] = {"min_se": float(amce["se"].min()), "max_se": float(amce["se"].max()),
                     "mde_min": float(tcrit * amce["se"].min()),
                     "mde_max": float(tcrit * amce["se"].max())}

    # Do the effects at the highest level differ between attributes?
    names = list(m.params.index)

    def contrast(weights):
        L = np.zeros(len(names))
        for k, w in weights.items():
            L[names.index(k)] = w
        return L

    tops = [f"{a}_3" for a in ATTRS]
    ft = m.f_test(np.vstack([contrast({tops[0]: 1, t: -1}) for t in tops[1:]]))
    R["highest_level_equal"] = {"F": float(ft.fvalue), "df1": int(ft.df_num),
                                "df2": int(ft.df_denom), "p": float(ft.pvalue)}
    pair_rows = []
    for i, a in enumerate(ATTRS):
        for b in ATTRS[i + 1:]:
            t = m.t_test(contrast({f"{a}_3": 1, f"{b}_3": -1}))
            pair_rows.append({"contrast": f"{ATTR_NAME[a]} minus {ATTR_NAME[b]}, highest level",
                              "diff": float(t.effect[0]), "se": float(t.sd[0][0]),
                              "p": float(t.pvalue), "mde": float(tcrit * t.sd[0][0])})
    pd.DataFrame(pair_rows).to_csv(tdir / "amce_highest_level_differences.csv", index=False)
    R["mde_difference"] = {"min": min(r["mde"] for r in pair_rows),
                           "max": max(r["mde"] for r in pair_rows)}
    # Ease of use: second step (moderate to easy) against first step (difficult to moderate)
    t = m.t_test(contrast({"ease_3": 1, "ease_2": -2}))
    R["ease_step_difference"] = {"diff": float(t.effect[0]), "se": float(t.sd[0][0]),
                                 "p": float(t.pvalue)}

    # ---------------- 3. marginal means ----------------------------------
    mm_rows = []
    groups = pd.factorize(df["ResponseId"])[0]
    for a in ATTRS:
        X = pd.get_dummies(df[a]).astype(float)
        fit = sm.OLS(df["chosen"], X).fit(cov_type="cluster", cov_kwds={"groups": groups},
                                          use_t=True)
        ci = fit.conf_int()
        # Omnibus test that all three marginal means are equal
        Rm = np.array([[1, -1, 0], [0, 1, -1]])
        ft = fit.f_test(Rm)
        for l in (1, 2, 3):
            mm_rows.append({"attribute": ATTR_NAME[a], "level": LEVEL_NAME[a][l],
                            "mm": fit.params[l], "se": fit.bse[l],
                            "ci_lo": ci.loc[l, 0], "ci_hi": ci.loc[l, 1],
                            "omnibus_F": float(ft.fvalue), "omnibus_df1": int(ft.df_num),
                            "omnibus_df2": int(ft.df_denom), "omnibus_p": float(ft.pvalue)})
    mm = pd.DataFrame(mm_rows)
    mm.to_csv(tdir / "marginal_means.csv", index=False)
    # Moderate levels against an even split
    R["moderate_vs_half"] = {}
    for a in ATTRS:
        s_ = df[df[a] == 2]
        fit = sm.OLS(s_["chosen"], np.ones(len(s_))).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(s_["ResponseId"])[0]}, use_t=True)
        R["moderate_vs_half"][a] = {"mm": float(fit.params.iloc[0]),
                                    "p": float(fit.t_test("const = 0.5").pvalue)}
    # A difficult system against the lowest level of each other attribute,
    # respondent cluster bootstrap
    def low_gaps(d):
        e = d.loc[d["ease"] == 1, "chosen"].mean()
        return np.array([e - d.loc[d[a] == 1, "chosen"].mean() for a in ATTRS[:3]])

    by_resp = {k: g for k, g in df.groupby("ResponseId")}
    rid = np.array(list(by_resp))
    rng_mm = np.random.default_rng(SEED)
    draws = np.array([low_gaps(pd.concat([by_resp[i] for i in rng_mm.choice(rid, len(rid))]))
                      for _ in range(N_BOOT)])
    obs = low_gaps(df)
    R["difficult_vs_other_low"] = {
        a: {"diff": float(obs[j]),
            "p": float(min(1, 2 * min((draws[:, j] <= 0).mean(), (draws[:, j] >= 0).mean())))}
        for j, a in enumerate(ATTRS[:3])}

    # ---------------- 4. conditional logit and importance ---------------
    d = differences(df)
    cl = clogit(d)
    imp = importance(cl.params)
    pw = partworths(cl.params)
    # Delta-method standard errors of the centred part-worths
    V = cl.cov_params()
    pw_rows = []
    for a in ATTRS:
        cols = [f"{a}_2", f"{a}_3"]
        Va = V.loc[cols, cols].values
        # centred utilities u = A @ [b2, b3], with u1 = -(b2+b3)/3 etc.
        A = np.array([[-1, -1], [2, -1], [-1, 2]]) / 3.0
        cov = A @ Va @ A.T
        for i, l in enumerate((1, 2, 3)):
            z = pw[a][i] / np.sqrt(cov[i, i])
            pw_rows.append({"attribute": ATTR_NAME[a], "level": LEVEL_NAME[a][l],
                            "partworth": pw[a][i], "se": np.sqrt(cov[i, i]), "z": z,
                            "p": 2 * stats.norm.sf(abs(z))})
    pd.DataFrame(pw_rows).to_csv(tdir / "partworths.csv", index=False)
    clt = pd.DataFrame({"b": cl.params, "se": cl.bse, "p": cl.pvalues})
    clt.to_csv(tdir / "clogit_dummy_coefficients.csv")
    R["clogit"] = {"loglik": cl.llf, "n_tasks": int(cl.nobs),
                   "pseudo_R2": 1 - cl.llf / (cl.nobs * np.log(0.5))}

    # Respondent cluster bootstrap for every function of the logit estimates
    point = quantities(cl.params)
    ids = d["ResponseId"].unique()
    by_id = {i: g for i, g in d.groupby("ResponseId")}
    boot = []
    for _ in range(N_BOOT):
        draw = rng.choice(ids, size=len(ids), replace=True)
        bd = pd.concat([by_id[i] for i in draw], ignore_index=True)
        b = sm.Logit(bd["y"], bd[DUMMIES]).fit(disp=0, method="newton", maxiter=100).params
        boot.append(quantities(b))
    boot = pd.DataFrame(boot)
    qtab = pd.DataFrame({"estimate": pd.Series(point),
                         "ci_lo": boot.quantile(0.025), "ci_hi": boot.quantile(0.975),
                         "boot_se": boot.std()})
    qtab.to_csv(tdir / "logit_quantities_bootstrap.csv")
    # Governance share against an even split, bootstrap p value
    gov = boot["imp_governance"]
    p_gov = 2 * min((gov <= 0.5).mean(), (gov >= 0.5).mean())
    R["governance_share"] = {"estimate": point["imp_governance"],
                             "ci": [float(gov.quantile(0.025)), float(gov.quantile(0.975))],
                             "p_vs_half": float(max(p_gov, 1 / N_BOOT))}
    imp_rows = []
    for a in sorted(ATTRS, key=lambda a: -imp[a]):
        imp_rows.append({"attribute": ATTR_NAME[a],
                         "type": "Governance" if a in GOVERNANCE else "Efficiency",
                         "range": imp["range"][a], "share": imp[a],
                         "ci_lo": qtab.loc[f"imp_{a}", "ci_lo"],
                         "ci_hi": qtab.loc[f"imp_{a}", "ci_hi"]})
    imp_tab = pd.DataFrame(imp_rows)
    imp_tab.to_csv(tdir / "importance.csv", index=False)
    # Pairwise differences in importance, bootstrap
    pair = []
    for i, a in enumerate(ATTRS):
        for b_ in ATTRS[i + 1:]:
            diff = boot[f"imp_{a}"] - boot[f"imp_{b_}"]
            est = point[f"imp_{a}"] - point[f"imp_{b_}"]
            p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
            pair.append({"a": ATTR_NAME[a], "b": ATTR_NAME[b_], "diff": est,
                         "ci_lo": diff.quantile(0.025), "ci_hi": diff.quantile(0.975),
                         "p_boot": max(p, 1 / N_BOOT)})
    pd.DataFrame(pair).to_csv(tdir / "importance_differences.csv", index=False)

    # Importance from the AMCE ranges as a model-free comparison
    amce_rng = {a: max(0, m.params[f"{a}_2"], m.params[f"{a}_3"]) -
                min(0, m.params[f"{a}_2"], m.params[f"{a}_3"]) for a in ATTRS}
    tot = sum(amce_rng.values())
    R["importance_lpm"] = {a: amce_rng[a] / tot for a in ATTRS}

    # ---------------- 5. head-to-head configurations --------------------
    h2h = []
    for x, y in MATCHUPS:
        k = f"p_{x}_vs_{y}"
        h2h.append({"profile_A": CONFIG_LABEL[x], "profile_B": CONFIG_LABEL[y],
                    "p_A_chosen": point[k], "ci_lo": qtab.loc[k, "ci_lo"],
                    "ci_hi": qtab.loc[k, "ci_hi"]})
    h2h = pd.DataFrame(h2h)
    h2h.to_csv(tdir / "head_to_head.csv", index=False)
    # Observed check: tasks in which one profile was governance-better and
    # efficiency-worse than the other
    w = df.sort_values(["ResponseId", "task", "profile"])
    a1 = w[w["profile"] == 1].set_index(["ResponseId", "task"])
    a2 = w[w["profile"] == 2].set_index(["ResponseId", "task"])
    g = (a1[GOVERNANCE].sum(axis=1) - a2[GOVERNANCE].sum(axis=1))
    e = (a1[EFFICIENCY].sum(axis=1) - a2[EFFICIENCY].sum(axis=1))
    conflict = ((g > 0) & (e < 0)) | ((g < 0) & (e > 0))
    gov_side_chosen = np.where(g > 0, a1["chosen"], 1 - a1["chosen"])[conflict.values]
    R["observed_conflict_tasks"] = {"n": int(conflict.sum()),
                                    "share_governance_side": float(gov_side_chosen.mean())}
    eq = conflict & (g.abs() == e.abs())
    gv = np.where(g > 0, a1["chosen"], 1 - a1["chosen"])[eq.values]
    bt = stats.binomtest(int(gv.sum()), len(gv), 0.5)
    R["observed_balanced_conflict"] = {"n": int(eq.sum()), "share_governance_side": float(gv.mean()),
                                       "p": float(bt.pvalue)}

    # ---------------- 6. trade-off ratios --------------------------------
    trade = []
    for a in ATTRS:
        k = f"ratio_{a}_to_usefulness"
        trade.append({"attribute": ATTR_NAME[a], "ratio_to_usefulness": point[k],
                      "ci_lo": qtab.loc[k, "ci_lo"], "ci_hi": qtab.loc[k, "ci_hi"]})
    pd.DataFrame(trade).to_csv(tdir / "tradeoff_ratios.csv", index=False)

    # ---------------- 7. robustness --------------------------------------
    rob = {"Full analytic sample": df}
    rob["Tasks 1 to 4"] = df[df["task"] <= 4]
    rob["Tasks 5 to 9"] = df[df["task"] >= 5]
    rob["Excluding task 1"] = df[df["task"] > 1]
    fast = set(resp.loc[resp["fast_responder"] == 1, "ResponseId"])
    rob["Excluding fast responders"] = df[~df["ResponseId"].isin(fast)]
    rob["Including attention check failures"] = alldf
    rob_rows = []
    for name, sub in rob.items():
        f = lpm(sub)
        for c in DUMMIES:
            rob_rows.append({"specification": name, "term": c, "label": dummy_label(c),
                             "b": f.params[c], "se": f.bse[c], "p": f.pvalues[c],
                             "n_resp": sub["ResponseId"].nunique(), "n_profiles": len(sub)})
    rob_tab = pd.DataFrame(rob_rows)
    rob_tab.to_csv(tdir / "robustness_amce.csv", index=False)
    rob_imp = []
    for name, sub in rob.items():
        q = importance(clogit(differences(sub), cluster=False).params)
        rob_imp.append({"specification": name, **{ATTR_NAME[a]: q[a] for a in ATTRS},
                        "Governance share": q["privacy"] + q["transparency"]})
    pd.DataFrame(rob_imp).to_csv(tdir / "robustness_importance.csv", index=False)

    # Interaction tests: does the AMCE vector change with task half, profile
    # side or the row in which an attribute was displayed?
    tests = []
    def joint(sub, mod_cols, label):
        X = sub[DUMMIES].copy()
        inter = []
        for mc in mod_cols:
            X[mc] = sub[mc]
            for c in DUMMIES:
                X[f"{c}_x_{mc}"] = sub[c] * sub[mc]
                inter.append(f"{c}_x_{mc}")
        fit = sm.OLS(sub["chosen"], sm.add_constant(X)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["ResponseId"])[0]},
            use_t=True)
        ft = fit.f_test(" = 0, ".join(inter) + " = 0")
        tests.append({"test": label, "F": float(ft.fvalue), "df1": int(ft.df_num),
                      "df2": int(ft.df_denom), "p": float(ft.pvalue)})
    tmp = df.copy()
    tmp["late"] = (tmp["task"] >= 5).astype(float)
    joint(tmp, ["late"], "AMCEs differ between tasks 1 to 4 and 5 to 9")
    tmp["task_c"] = tmp["task"] - 5.0
    joint(tmp, ["task_c"], "AMCEs change linearly with task number")
    tmp["right"] = (tmp["profile"] == 2).astype(float)
    joint(tmp, ["right"], "AMCEs differ between the left and right profile")
    # Attribute shown in the top half of the profile table. Exactly two
    # attributes occupy the top two rows, so the four top indicators sum to 2
    # and one of them is omitted to keep the design matrix of full rank.
    X = tmp[DUMMIES].copy()
    inter = []
    for a in ATTRS:
        top = (tmp[f"{a}_row"] <= 2).astype(float)
        if a != ATTRS[-1]:
            X[f"{a}_top"] = top
        for l in (2, 3):
            X[f"{a}_{l}_x_top"] = tmp[f"{a}_{l}"] * top
            inter.append(f"{a}_{l}_x_top")
    fit = sm.OLS(tmp["chosen"], sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(tmp["ResponseId"])[0]}, use_t=True)
    ft = fit.f_test(" = 0, ".join(inter) + " = 0")
    tests.append({"test": "AMCEs differ when the attribute is shown in the top two rows",
                  "F": float(ft.fvalue), "df1": int(ft.df_num), "df2": int(ft.df_denom),
                  "p": float(ft.pvalue)})
    # Position effects in detail: AMCEs when an attribute is shown in the top
    # two and in the bottom two rows, whether the shift is common to all
    # attributes, and AMCEs reweighted to an equal split of positions
    idx = list(fit.params.index)

    def vec(weights):
        v = np.zeros(len(idx))
        for k, w in weights.items():
            v[idx.index(k)] = w
        return v

    pos_rows = []
    for c in DUMMIES:
        bottom, shift = fit.params[c], fit.params[f"{c}_x_top"]
        pos_rows.append({"term": c, "label": dummy_label(c), "amce_bottom": bottom,
                         "amce_top": bottom + shift, "shift": shift,
                         "amce_balanced": bottom + shift / 2,
                         "share_shown_top": float((tmp[f"{c.rsplit('_', 1)[0]}_row"] <= 2).mean())})
    pos = pd.DataFrame(pos_rows)
    pos.to_csv(tdir / "position_effects.csv", index=False)
    t = fit.t_test(vec({f"{a}_3_x_top": 0.25 for a in ATTRS}))
    eq = fit.f_test(np.vstack([vec({f"{ATTRS[0]}_{l}_x_top": 1, f"{a}_{l}_x_top": -1})
                               for a in ATTRS[1:] for l in (2, 3)]))
    bal = pos.set_index("term")["amce_balanced"]
    rng_bal = {a: max(0, bal[f"{a}_2"], bal[f"{a}_3"]) for a in ATTRS}
    R["position"] = {
        "mean_shift_highest": float(t.effect[0]), "mean_shift_p": float(t.pvalue),
        "equal_shift_F": float(eq.fvalue), "equal_shift_df1": int(eq.df_num),
        "equal_shift_p": float(eq.pvalue),
        "governance_share_balanced_lpm": (rng_bal["privacy"] + rng_bal["transparency"])
        / sum(rng_bal.values())}
    tests = pd.DataFrame(tests)
    tests.to_csv(tdir / "robustness_interaction_tests.csv", index=False)
    left = df.loc[df["profile"] == 1, "chosen"]
    R["left_share"] = {"share": float(left.mean()),
                       "p": float(stats.binomtest(int(left.sum()), len(left)).pvalue)}
    dom = df[df["dominated_task"] == 1].drop_duplicates(["ResponseId", "task"])
    R["dominance"] = {"tasks": len(dom), "dominant_chosen": float(dom["dominant_chosen"].mean())}

    # ---------------- figures --------------------------------------------
    # Figure 4.4 average marginal component effects
    rows = []
    for a in ATTRS:
        rows.append((a, LEVEL_NAME[a][1], 0.0, 0.0, 0.0, True))
        for l in (2, 3):
            r = amce.set_index("term").loc[f"{a}_{l}"]
            rows.append((a, LEVEL_NAME[a][l], r["b"], r["ci_lo"], r["ci_hi"], False))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    level_plot(ax, rows, "Change in the probability of being chosen (AMCE), 95% CI", 0.0)
    ax.set_xlim(-0.05, amce["ci_hi"].max() + 0.06)
    save(fig, fdir / "Figure_4_4_amce.png")

    # Figure 4.6 marginal means
    rows = [(a, r["level"], r["mm"], r["ci_lo"], r["ci_hi"], False)
            for a in ATTRS for _, r in mm[mm["attribute"] == ATTR_NAME[a]].iterrows()]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    level_plot(ax, rows, "Marginal mean, probability of being chosen, 95% CI", 0.5,
               fmt="{:.3f}")
    ax.set_xlim(mm["ci_lo"].min() - 0.03, mm["ci_hi"].max() + 0.07)
    save(fig, fdir / "Figure_4_6_marginal_means.png")

    # Figure 4.5 importance shares
    it = imp_tab.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    cols = [NAVY if t == "Governance" else ORANGE for t in it["type"]]
    ax.barh(range(len(it)), it["share"] * 100, color=cols, height=0.6)
    ax.errorbar(it["share"] * 100, range(len(it)),
                xerr=[(it["share"] - it["ci_lo"]) * 100, (it["ci_hi"] - it["share"]) * 100],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for i, r in it.iterrows():
        ax.text(r["ci_hi"] * 100 + 0.8, i, f"{r['share'] * 100:.1f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(it)))
    ax.set_yticklabels(it["attribute"])
    ax.set_xlabel("Importance share (%), 95% bootstrap CI")
    ax.set_xlim(0, (it["ci_hi"].max() * 100) + 9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NAVY, label="Governance"), Patch(color=ORANGE, label="Efficiency")],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    save(fig, fdir / "Figure_4_5_importance.png")

    # Figure 4.7 predicted choice between complete systems
    hh = h2h.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.errorbar(hh["p_A_chosen"], range(len(hh)),
                xerr=[hh["p_A_chosen"] - hh["ci_lo"], hh["ci_hi"] - hh["p_A_chosen"]],
                fmt="o", color=NAVY, ecolor=NAVY, elinewidth=1.1, capsize=3, ms=5)
    for i, r in hh.iterrows():
        ax.text(r["ci_hi"] + 0.015, i, f"{r['p_A_chosen']:.2f}", va="center", fontsize=9)
    ax.axvline(0.5, color=GREY, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(hh)))
    ax.set_yticklabels([f"{a}\nvs {b.lower()}" for a, b in zip(hh["profile_A"], hh["profile_B"])],
                       fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Predicted probability that the first system is chosen, 95% bootstrap CI")
    save(fig, fdir / "Figure_4_7_head_to_head.png")

    with open(out_dir / "key_results.json", "w") as fh:
        json.dump(R, fh, indent=2, default=float)
    print(json.dumps(R, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=Path("data"), type=Path)
    ap.add_argument("--out", default=Path("outputs"), type=Path)
    args = ap.parse_args()
    main(args.data, args.out)
