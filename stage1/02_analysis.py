"""
Stage 1 analysis. Diversity and inclusion climate and expected procedural
justice in AI-mediated recruitment.

Reads the outputs of 01_prepare_data.py and writes every table (CSV), every
figure (PNG, 300 dpi) and a JSON file of the key estimates to the output
folder.

    1  Sample flow and sample characteristics
    2  Item descriptives
    3  Reliability (alpha with Feldt CI, omega, CR, AVE)
    4  Dimensionality (KMO, Bartlett, parallel analysis, CFA)
    5  Discriminant validity (latent correlation, HTMT) and common method check
    6  Correlations
    7  Main-effects regression (OLS, HC3)
    8  Moderation (fully interacted model, Wald tests, BH, conditional slopes)
    9  Diagnostics
    10 Robustness of the climate estimate, incl. wild cluster bootstrap
    11 Decomposition into organisational and personal components
    12 Figures

Usage
    python 02_analysis.py --data data --out outputs
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import semopy
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="semopy")

SEED = 12345
N_PARALLEL = 1000
N_BOOT = 9999

ORG = ["OF1", "OF2", "OF3", "OF4", "OF5", "OI1", "OI2", "OI3"]
PERS = ["PV1", "PV2", "PV3", "PC1"]
CLIM = ORG + PERS
PJ = [f"PJ{i}" for i in range(1, 8)]
PJ_A = ["PJ1", "PJ2", "PJ4", "PJ6"]   # second-factor split found in Section 4
PJ_B = ["PJ3", "PJ5", "PJ7"]

ROLES = ["Academic staff", "Professional staff", "Leaders"]
AGE6 = ["18 to 24", "25 to 34", "35 to 44", "45 to 54", "55 to 64", "65 or older"]
AGE5 = ["18 to 24", "25 to 34", "35 to 44", "45 to 54", "55 or older"]

ITEM_TEXT = {
    "OF1": "Managers hire and promote objectively regardless of race, sex, religion or age",
    "OF2": "Managers give feedback and evaluate fairly regardless of background",
    "OF3": "Managers make layoff decisions fairly regardless of background",
    "OF4": "Managers interpret HR policies fairly for all employees",
    "OF5": "Managers give assignments based on skills and abilities",
    "OI1": "Managers encourage employee network support groups",
    "OI2": "A mentoring programme prepares minority and female employees for promotion",
    "OI3": "The university spends enough on diversity awareness and training",
    "PV1": "Knowing the cultural norms of diverse groups would make me more effective",
    "PV2": "Diverse viewpoints add value",
    "PV3": "Diversity is a strategic business issue",
    "PC1": "I feel at ease with people from other backgrounds",
    "PJ1": "Able to express views and feelings during the procedures",
    "PJ2": "Influence over the outcome of the procedures",
    "PJ3": "Procedures applied consistently",
    "PJ4": "Procedures free of bias",
    "PJ5": "Procedures based on accurate information",
    "PJ6": "Able to appeal the outcome",
    "PJ7": "Procedures uphold ethical and moral standards",
}

# Model terms and their reporting labels
FORMULA_MAIN = ("pj ~ climate_c + C(recruit_exp, Treatment('High'))"
                " + C(ai_experience, Treatment('Negative'))"
                " + C(age_band5, Treatment('18 to 24'))"
                " + C(gender, Treatment('Female'))")
MODS = {
    "recruit": "C(recruit_exp, Treatment('High'))",
    "ai": "C(ai_experience, Treatment('Negative'))",
    "age": "C(age_band5, Treatment('18 to 24'))",
    "gender": "C(gender, Treatment('Female'))",
}
FORMULA_INT = FORMULA_MAIN + "".join(f" + climate_c:{m}" for m in MODS.values())


def label(term: str) -> str:
    rep = {
        "Intercept": "Intercept",
        "climate_c": "D&I climate (centred)",
        "C(recruit_exp, Treatment('High'))[T.Low]": "Recruitment experience (low)",
        "C(ai_experience, Treatment('Negative'))[T.Neutral]": "AI experience (neutral)",
        "C(ai_experience, Treatment('Negative'))[T.Positive]": "AI experience (positive)",
        "C(gender, Treatment('Female'))[T.Male]": "Gender (male)",
        "C(gender, Treatment('Female'))[T.Not disclosed]": "Gender (not disclosed)",
    }
    out = term
    for k, v in rep.items():
        out = out.replace(k, v)
    out = out.replace("C(age_band5, Treatment('18 to 24'))[T.", "Age ")
    out = out.replace("C(age_band5, Treatment('18 to 24'))", "Age")
    out = out.replace("]", "")
    out = out.replace("D&I climate (centred):", "Climate × ")
    out = out.replace("Climate × Recruitment experience (low)", "Climate × recruitment experience (low)")
    out = out.replace("Climate × AI", "Climate × AI")
    out = out.replace("Climate × Age", "Climate × age")
    out = out.replace("Climate × Gender", "Climate × gender")
    return out


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def cronbach_alpha(X: pd.DataFrame) -> float:
    X = np.asarray(X, float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))


def feldt_ci(alpha: float, n: int, k: int, level: float = 0.95):
    """Feldt (1965) confidence interval for Cronbach's alpha."""
    df1, df2 = n - 1, (n - 1) * (k - 1)
    g = 1 - level
    lo = 1 - (1 - alpha) * stats.f.ppf(1 - g / 2, df1, df2)
    hi = 1 - (1 - alpha) * stats.f.ppf(g / 2, df1, df2)
    return lo, hi


def item_rest(X: pd.DataFrame) -> pd.Series:
    tot = X.sum(axis=1)
    return pd.Series({c: np.corrcoef(X[c], tot - X[c])[0, 1] for c in X})


def alpha_if_deleted(X: pd.DataFrame) -> pd.Series:
    return pd.Series({c: cronbach_alpha(X.drop(columns=c)) for c in X})


def kmo(X: pd.DataFrame) -> float:
    R = np.corrcoef(np.asarray(X, float), rowvar=False)
    Ri = np.linalg.inv(R)
    d = np.sqrt(np.outer(np.diag(Ri), np.diag(Ri)))
    P = -Ri / d
    np.fill_diagonal(R, 0)
    np.fill_diagonal(P, 0)
    return (R ** 2).sum() / ((R ** 2).sum() + (P ** 2).sum())


def bartlett(X: pd.DataFrame):
    n, p = X.shape
    R = np.corrcoef(np.asarray(X, float), rowvar=False)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    df = p * (p - 1) / 2
    return chi2, df, stats.chi2.sf(chi2, df)


def parallel_analysis(X: pd.DataFrame, n_iter: int, rng) -> pd.DataFrame:
    """Horn's parallel analysis on principal component eigenvalues."""
    n, p = X.shape
    obs = np.sort(np.linalg.eigvalsh(np.corrcoef(np.asarray(X, float), rowvar=False)))[::-1]
    sim = np.empty((n_iter, p))
    for i in range(n_iter):
        Z = rng.standard_normal((n, p))
        sim[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    return pd.DataFrame({
        "component": np.arange(1, p + 1),
        "observed": obs,
        "random_mean": sim.mean(axis=0),
        "random_p95": np.percentile(sim, 95, axis=0),
    })


def fisher_ci(r: float, n: int, level: float = 0.95):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    q = stats.norm.ppf(1 - (1 - level) / 2)
    return np.tanh(z - q * se), np.tanh(z + q * se)


def pearson(x, y):
    r, p = stats.pearsonr(x, y)
    lo, hi = fisher_ci(r, len(x))
    return r, lo, hi, p


# ---------------------------------------------------------------------------
# CFA helpers
# ---------------------------------------------------------------------------

def rmsea_ci(chi2: float, df: int, n: int, level: float = 0.90):
    """Confidence interval for RMSEA by inverting the noncentral chi-square."""
    def find(target):
        if stats.chi2.cdf(chi2, df) < target:
            return 0.0
        lo, hi = 0.0, max(10 * chi2, 100.0)
        for _ in range(200):
            mid = (lo + hi) / 2
            if stats.ncx2.cdf(chi2, df, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    g = (1 - level) / 2
    lam_lo, lam_hi = find(1 - g), find(g)
    f = lambda lam: np.sqrt(lam / (df * (n - 1)))
    return f(lam_lo), f(lam_hi)


def fit_cfa(desc: str, data: pd.DataFrame, cols):
    m = semopy.Model(desc)
    m.fit(data[cols], obj="MLW")
    st = semopy.calc_stats(m).T["Value"]
    obs = m.vars["observed"]
    S = np.cov(data[obs].values, rowvar=False, ddof=1)
    Sig = m.calc_sigma()[0]
    sd = np.sqrt(np.diag(S))
    resid = (S - Sig) / np.outer(sd, sd)
    idx = np.tril_indices(len(obs))
    srmr = np.sqrt(np.mean(resid[idx] ** 2))
    chi2, df = float(st["chi2"]), int(st["DoF"])
    lo, hi = rmsea_ci(chi2, df, len(data))
    fit = {"chi2": chi2, "df": df, "p": float(st["chi2 p-value"]),
           "CFI": float(st["CFI"]), "TLI": float(st["TLI"]),
           "RMSEA": float(st["RMSEA"]), "RMSEA_lo": lo, "RMSEA_hi": hi,
           "SRMR": srmr, "AIC": float(st["AIC"]), "BIC": float(st["BIC"])}
    est = m.inspect(std_est=True)
    load = est[(est["op"] == "~") & (est["lval"].isin(obs))][["lval", "rval", "Est. Std", "p-value"]]
    load.columns = ["item", "factor", "std_loading", "p"]
    cov = est[(est["op"] == "~~") & (~est["lval"].isin(obs)) & (est["lval"] != est["rval"])]
    fcor = {f"{r.lval}~~{r.rval}": float(r["Est. Std"]) for _, r in cov.iterrows()}
    return fit, load.reset_index(drop=True), fcor


def omega_cr_ave(loadings: np.ndarray):
    lam = np.asarray(loadings, float)
    theta = 1 - lam ** 2
    omega = lam.sum() ** 2 / (lam.sum() ** 2 + theta.sum())
    ave = np.mean(lam ** 2)
    return omega, ave


def htmt(data: pd.DataFrame, a_items, b_items) -> float:
    R = data[a_items + b_items].corr().abs()
    het = R.loc[a_items, b_items].values.mean()
    def mono(items):
        sub = R.loc[items, items].values
        return sub[np.triu_indices(len(items), 1)].mean()
    return het / np.sqrt(mono(a_items) * mono(b_items))


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def ols_hc3(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(cov_type="HC3", use_t=True)


def coef_table(res) -> pd.DataFrame:
    ci = res.conf_int()
    t = pd.DataFrame({
        "term": res.params.index,
        "b": res.params.values,
        "se": res.bse.values,
        "t": res.tvalues.values,
        "p": res.pvalues.values,
        "ci_lo": ci[0].values,
        "ci_hi": ci[1].values,
    })
    t.insert(1, "label", t["term"].map(label))
    return t


def wald(res, terms):
    """Joint Wald F test (HC3 covariance) that all listed terms are zero."""
    names = list(res.params.index)
    R = np.zeros((len(terms), len(names)))
    for i, t in enumerate(terms):
        R[i, names.index(t)] = 1
    w = res.wald_test(R, use_f=True, scalar=True)
    return float(w.statistic), int(w.df_num), int(w.df_denom), float(w.pvalue)


def lincom(res, weights: dict):
    names = list(res.params.index)
    L = np.zeros(len(names))
    for k, v in weights.items():
        L[names.index(k)] = v
    est = float(L @ res.params.values)
    se = float(np.sqrt(L @ res.cov_params().values @ L))
    t = est / se
    df = res.df_resid
    p = 2 * stats.t.sf(abs(t), df)
    q = stats.t.ppf(0.975, df)
    return est, se, p, est - q * se, est + q * se


def gvif(data: pd.DataFrame, formula_rhs: str) -> pd.DataFrame:
    """Generalised VIF (Fox and Monette, 1992) for each model term."""
    import patsy
    X = patsy.dmatrix(formula_rhs, data, return_type="dataframe")
    info = X.design_info
    Xn = X.drop(columns="Intercept")
    R = np.corrcoef(Xn.values, rowvar=False)
    cols = list(Xn.columns)
    rows = []
    for term, sl in info.term_name_slices.items():
        if term == "Intercept":
            continue
        tc = list(X.columns[sl])
        idx = [cols.index(c) for c in tc]
        rest = [i for i in range(len(cols)) if i not in idx]
        g = (np.linalg.det(R[np.ix_(idx, idx)]) * np.linalg.det(R[np.ix_(rest, rest)])
             / np.linalg.det(R))
        dfk = len(idx)
        rows.append({"term": term, "df": dfk, "GVIF": g, "GVIF_adj": g ** (1 / (2 * dfk))})
    return pd.DataFrame(rows)


def cr1_fit(X, y, g):
    """OLS with CR1 cluster-robust covariance. Returns coefficients and SEs."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    n, k = X.shape
    G = len(np.unique(g))
    meat = np.zeros((k, k))
    for c in np.unique(g):
        s = X[g == c].T @ u[g == c]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V))


def wild_cluster_bootstrap(formula, data, cluster, term, n_boot, rng):
    """Wild cluster restricted bootstrap-t with Webb six-point weights
    (Cameron, Gelbach and Miller, 2008; Webb, 2023). Tests H0 coefficient = 0."""
    import patsy
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    j = list(X.columns).index(term)
    yv, Xv = y.values.ravel(), X.values
    g = pd.factorize(data.loc[X.index, cluster])[0]
    b, se = cr1_fit(Xv, yv, g)
    t_obs = b[j] / se[j]
    Xr = np.delete(Xv, j, axis=1)
    br = np.linalg.lstsq(Xr, yv, rcond=None)[0]
    fit_r, u_r = Xr @ br, yv - Xr @ br
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    G = g.max() + 1
    t_star = np.empty(n_boot)
    for i in range(n_boot):
        w = rng.choice(webb, size=G)
        ys = fit_r + u_r * w[g]
        bs, ses = cr1_fit(Xv, ys, g)
        t_star[i] = bs[j] / ses[j]
    p = np.mean(np.abs(t_star) >= abs(t_obs))
    return {"b": b[j], "se_cr1": se[j], "t_cr1": t_obs, "p_wcr": p, "G": int(G)}


def min_detectable_f2(n, k_total, df_num=1, alpha=0.05, power=0.80):
    """Smallest Cohen's f2 detectable for a df_num test in a model with
    k_total predictors (excluding intercept)."""
    df_den = n - k_total - 1
    fcrit = stats.f.ppf(1 - alpha, df_num, df_den)
    lo, hi = 1e-6, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        pw = stats.ncf.sf(fcrit, df_num, df_den, mid * n)
        lo, hi = (mid, hi) if pw < power else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Colour palette, with grey for neutral or undisclosed categories.
NAVY, ORANGE, GREEN, MAROON = "#1a476f", "#e37e00", "#55752f", "#90353b"
GREY, LIGHT = "#8c8c8c", "#c8c8c8"
ROLE_COL = {"Academic staff": NAVY, "Professional staff": ORANGE, "Leaders": GREEN}
RAMP = ["#c6d4e1", "#9db4cb", "#7393b3", "#4b739b", "#2c5a85", NAVY]
INK, INK2, GRID = "#000000", "#404040", "#d9d9d9"


def no_zero(x: float) -> str:
    """Format a correlation to two decimals without the leading zero."""
    return f"{x:.2f}".replace("0.", ".", 1)


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


def text_on(hex_color: str) -> str:
    """Dark or white label text, whichever reads better on the fill."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return INK if lum > 0.30 else "white"


def stacked_pct(ax, tab: pd.DataFrame, colors, legend_title):
    """Horizontal 100% stacked bars, one bar per row of tab (counts)."""
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    left = np.zeros(len(pct))
    for j, c in enumerate(pct.columns):
        v = pct[c].values
        ax.barh(range(len(pct)), v, left=left, color=colors[j], edgecolor="white",
                linewidth=1.0, height=0.6, label=c)
        for i, (x0, w) in enumerate(zip(left, v)):
            if w >= 6:
                ax.text(x0 + w / 2, i, f"{w:.0f}%", ha="center", va="center",
                        fontsize=8.5, color=text_on(colors[j]))
        left += v
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels([f"{r} (n = {int(n)})" for r, n in zip(pct.index, tab.sum(axis=1))])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of respondents")
    ax.legend(title=legend_title, ncol=min(len(pct.columns), 4), loc="upper center",
              bbox_to_anchor=(0.5, -0.22), fontsize=8.5, title_fontsize=9)


def grouped_bars(ax, tab: pd.DataFrame, legend_title):
    """Grouped vertical bars of within-row percentages, groups = columns."""
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    n_r, n_c = pct.shape
    w = 0.8 / n_r
    for i, r in enumerate(pct.index):
        x = np.arange(n_c) + (i - (n_r - 1) / 2) * w
        bars = ax.bar(x, pct.loc[r].values, width=w * 0.92, color=ROLE_COL.get(r, RAMP[i]),
                      label=f"{r} (n = {int(tab.loc[r].sum())})")
        for b, v in zip(bars, pct.loc[r].values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(np.arange(n_c))
    ax.set_xticklabels(pct.columns)
    ax.set_ylabel("Percentage within role")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, pct.values.max() * 1.15)
    ax.legend(title=legend_title, fontsize=8.5, title_fontsize=9, ncol=n_r,
              loc="upper center", bbox_to_anchor=(0.5, -0.24))


def dist_by_role(ax, data, var, xlabel, rng):
    """Box plot (median, interquartile range, 1.5 x IQR whiskers) drawn as an
    outline in the role colour, with every respondent shown as a point."""
    for i, r in enumerate(ROLES):
        v = data.loc[data["role"] == r, var].values
        jit = rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(i + jit, v, s=11, facecolors="none", edgecolors=ROLE_COL[r], alpha=0.55,
                   linewidths=0.6, zorder=1)
        ax.boxplot(v, positions=[i], widths=0.5, orientation="vertical", patch_artist=True,
                   showfliers=False,
                   boxprops={"facecolor": "none", "edgecolor": ROLE_COL[r], "linewidth": 1.6},
                   medianprops={"color": ROLE_COL[r], "linewidth": 2.2},
                   whiskerprops={"color": ROLE_COL[r], "linewidth": 1.3},
                   capprops={"color": ROLE_COL[r], "linewidth": 1.3}, zorder=2)
        ax.scatter([i], [v.mean()], marker="D", s=28, color="white", edgecolor=INK,
                   linewidth=1, zorder=3)
    ax.set_xticks(range(len(ROLES)))
    ax.set_xticklabels([f"{r}\n(n = {(data['role'] == r).sum()})" for r in ROLES])
    ax.set_ylabel(xlabel)
    ax.set_ylim(0.8, 5.2)


def forest(ax, labels, est, lo, hi, ref=0.0, xlabel="", decimals=3):
    """Coefficient plot, point estimates with capped 95% intervals and the
    estimate printed beside each interval."""
    y = np.arange(len(labels))[::-1]
    ax.axvline(ref, color=GREY, linewidth=0.8, linestyle="--", zorder=1)
    ax.errorbar(est, y, xerr=[np.asarray(est) - np.asarray(lo), np.asarray(hi) - np.asarray(est)],
                fmt="o", color=NAVY, ecolor=NAVY, elinewidth=1.1, capsize=3, capthick=1.1,
                markersize=5, zorder=3)
    span = max(hi) - min(lo)
    for e, h, yy in zip(est, hi, y):
        txt = f"{e:+.{decimals}f}".replace("-", "−")
        ax.text(h + 0.015 * span, yy, txt, va="center", ha="left", fontsize=8.5, color=INK, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(min(lo) - 0.05 * span, max(hi) + 0.14 * span)
    ax.set_xlabel(xlabel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: Path, out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    tdir, fdir = out_dir / "tables", out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    R = {}  # key results

    df = pd.read_csv(data_dir / "stage1_clean.csv")
    elig = pd.read_csv(data_dir / "stage1_eligible.csv")
    flow = pd.read_excel(data_dir / "stage1_clean.xlsx", sheet_name="Sample_flow")
    flow.to_csv(tdir / "sample_flow.csv", index=False)
    N = len(df)
    R["N_analytic"] = N
    R["N_eligible"] = len(elig)

    # ---------------- 1. Sample characteristics ----------------------
    rows = []
    def add_block(var, levels, title):
        for lev in levels:
            r = {"block": title, "level": lev}
            for role in ROLES + ["Overall"]:
                sub = df if role == "Overall" else df[df["role"] == role]
                k = int((sub[var] == lev).sum())
                r[f"{role}_n"] = k
                r[f"{role}_pct"] = 100 * k / len(sub)
            rows.append(r)
    add_block("gender", ["Female", "Male", "Not disclosed"], "Gender")
    add_block("age_band", AGE6 + ["Not disclosed"], "Age")
    add_block("sector", ["Public", "Private"], "Sector")
    add_block("education", ["High school or lower", "Diploma", "Bachelor's", "Master's", "Doctorate"], "Education")
    add_block("college_field", ["Health sciences", "Science, engineering and computing", "Business and law",
                                "Humanities, social sciences and education", "Deanship or administrative unit"],
              "College or unit")
    add_block("ethnicity", ["Middle Eastern or North African", "Asian", "White", "Black or African",
                            "Not disclosed"], "Ethnicity")
    add_block("recruit_role", ["Applicant only", "Recruiter only", "Both"], "Recruitment role experienced")
    add_block("recruit_exp", ["High", "Low"], "Recruitment experience")
    add_block("ai_experience", ["Positive", "Neutral", "Negative"], "Experience of AI or digital technology")
    sample = pd.DataFrame(rows)
    # Role differences in composition (chi-square on disclosed categories)
    tests = []
    for var, title in [("gender", "Gender"), ("age_band", "Age"), ("sector", "Sector"),
                       ("education", "Education"), ("college_field", "College or unit"),
                       ("ethnicity", "Ethnicity"), ("recruit_exp", "Recruitment experience"),
                       ("ai_experience", "Experience of AI or digital technology")]:
        sub = df[df[var] != "Not disclosed"]
        ct = pd.crosstab(sub["role"], sub[var])
        chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        tests.append({"block": title, "chi2": chi2, "df": dof, "p": p, "cramers_v": v,
                      "min_expected": exp.min(), "cells_expected_lt5": int((exp < 5).sum())})
    tests = pd.DataFrame(tests)
    sample.to_csv(tdir / "sample_characteristics.csv", index=False)
    tests.to_csv(tdir / "role_composition_tests.csv", index=False)
    R["role_n"] = df["role"].value_counts().to_dict()
    R["university_n"] = df["university"].value_counts().to_dict()

    # Construct descriptives by role
    cons = []
    for role in ROLES + ["Overall"]:
        sub = df if role == "Overall" else df[df["role"] == role]
        r = {"role": role, "n": len(sub)}
        for v in ["climate_12", "climate_org", "climate_pers", "pj"]:
            r[f"{v}_mean"], r[f"{v}_sd"] = sub[v].mean(), sub[v].std(ddof=1)
        cons.append(r)
    cons = pd.DataFrame(cons)
    # role differences on constructs (one-way ANOVA and Welch)
    role_tests = []
    for v in ["climate_12", "pj"]:
        groups = [df.loc[df["role"] == r, v] for r in ROLES]
        f, p = stats.f_oneway(*groups)
        a = smf.ols(f"{v} ~ C(role)", df).fit()
        eta2 = a.ess / (a.ess + a.ssr)
        # Welch ANOVA
        k = len(groups); ns = np.array([len(g) for g in groups]); ms = np.array([g.mean() for g in groups])
        vs = np.array([g.var(ddof=1) for g in groups]); w = ns / vs; mw = (w * ms).sum() / w.sum()
        A = (w * (ms - mw) ** 2).sum() / (k - 1)
        tmp = ((1 - w / w.sum()) ** 2 / (ns - 1)).sum()
        B = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
        fw = A / B; df2w = (k ** 2 - 1) / (3 * tmp); pw = stats.f.sf(fw, k - 1, df2w)
        role_tests.append({"construct": v, "F": f, "df1": k - 1, "df2": N - k, "p": p, "eta2": eta2,
                           "welch_F": fw, "welch_df2": df2w, "welch_p": pw})
    role_tests = pd.DataFrame(role_tests)
    # Pairwise role contrasts for expected PJ (HC3), reference academic staff
    pw = smf.ols("pj ~ C(role, Treatment('Academic staff'))", df).fit(cov_type="HC3", use_t=True)
    R["role_pj_pairwise"] = {k.split("[T.")[1].rstrip("]"): {"diff": pw.params[k], "p": pw.pvalues[k]}
                             for k in pw.params.index if k.startswith("C(role")}
    lv = lincom(pw, {"C(role, Treatment('Academic staff'))[T.Professional staff]": 1,
                     "C(role, Treatment('Academic staff'))[T.Leaders]": -1})
    R["role_pj_pairwise"]["Professional staff vs Leaders"] = {"diff": lv[0], "p": lv[2]}
    # Does the role difference in expected PJ remain once AI experience is held constant?
    adj = smf.ols("pj ~ C(role, Treatment('Academic staff')) + C(ai_experience, Treatment('Negative'))", df).fit(
        cov_type="HC3", use_t=True)
    role_terms = [t_ for t_ in adj.params.index if t_.startswith("C(role")]
    R["role_pj_adjusted_for_ai"] = dict(zip(["F", "df1", "df2", "p"], wald(adj, role_terms)))
    cons.to_csv(tdir / "construct_descriptives_by_role.csv", index=False)
    role_tests.to_csv(tdir / "role_differences_constructs.csv", index=False)

    # Do excluded or non-disclosing respondents differ on the constructs?
    comp = []
    groups_ = {
        "Failed vs passed the attention check item": (
            elig[elig["attention_pass"] == 0], elig[elig["attention_pass"] == 1]),
        "Did not vs did disclose gender and age": (
            df[df["regression_sample"] == 0], df[df["regression_sample"] == 1]),
    }
    for nm, (g1, g0) in groups_.items():
        for v in ["climate_12", "pj"]:
            t, p = stats.ttest_ind(g1[v], g0[v], equal_var=False)
            sp = np.sqrt((g1[v].var(ddof=1) + g0[v].var(ddof=1)) / 2)
            comp.append({"comparison": nm, "construct": v, "n1": len(g1), "mean1": g1[v].mean(),
                         "n0": len(g0), "mean0": g0[v].mean(), "welch_t": t, "p": p,
                         "cohens_d": (g1[v].mean() - g0[v].mean()) / sp})
    pd.DataFrame(comp).to_csv(tdir / "exclusion_comparisons.csv", index=False)

    # Survey order check. The attitudinal scales followed the conjoint task.
    # Attribute levels were randomised, so if the task shaped later answers,
    # the mix of levels a respondent happened to see would predict them.
    exp_cols = ["exp_transparency", "exp_privacy", "exp_usefulness", "exp_ease"]
    order = {"n_balanced": int(df["conjoint_balanced"].sum()), "n": N,
             "share_balanced": float(df["conjoint_balanced"].mean())}
    for v in ["pj", "climate_12"]:
        mo = ols_hc3(f"{v} ~ " + " + ".join(exp_cols), df)
        order[v] = dict(zip(["F", "df1", "df2", "p"], wald(mo, exp_cols)))
    R["survey_order_check"] = order
    pd.DataFrame([{"outcome": k, **order[k]} for k in ["pj", "climate_12"]]).to_csv(
        tdir / "survey_order_check.csv", index=False)

    # ---------------- 2. Item descriptives ---------------------------
    items = []
    for c in CLIM + PJ:
        v = df[c]
        items.append({"item": c, "text": ITEM_TEXT[c], "mean": v.mean(), "sd": v.std(ddof=1),
                      "skew": stats.skew(v, bias=False), "kurtosis": stats.kurtosis(v, bias=False),
                      "pct_1": 100 * (v == 1).mean(), "pct_5": 100 * (v == 5).mean()})
    items = pd.DataFrame(items)

    # ---------------- 3 and 4. Reliability and dimensionality --------
    cfa = {}
    cfa["clim_1f"] = fit_cfa("F =~ " + " + ".join(CLIM), df, CLIM)
    cfa["clim_2f"] = fit_cfa("ORG =~ " + " + ".join(ORG) + "\nPERS =~ " + " + ".join(PERS), df, CLIM)
    cfa["pj_1f"] = fit_cfa("PJ =~ " + " + ".join(PJ), df, PJ)
    cfa["pj_2f"] = fit_cfa("PJA =~ " + " + ".join(PJ_A) + "\nPJB =~ " + " + ".join(PJ_B), df, PJ)
    cfa["meas"] = fit_cfa("CLIM =~ " + " + ".join(CLIM) + "\nPJ =~ " + " + ".join(PJ), df, CLIM + PJ)
    cfa["harman"] = fit_cfa("CMV =~ " + " + ".join(CLIM + PJ), df, CLIM + PJ)

    fits = []
    names = {"clim_1f": "Climate, one factor (12 items)",
             "clim_2f": "Climate, two correlated factors (organisational 8, personal 4)",
             "pj_1f": "Expected PJ, one factor (7 items)",
             "pj_2f": "Expected PJ, two correlated factors (items 1, 2, 4, 6 and 3, 5, 7)",
             "meas": "Measurement model, climate and expected PJ (19 items)",
             "harman": "Single common factor for all 19 items"}
    for k, (fit, _, fcor) in cfa.items():
        fits.append({"model": names[k], "key": k, **fit,
                     "factor_correlation": next(iter(fcor.values()), np.nan)})
    fits = pd.DataFrame(fits)
    # chi-square difference tests (nested)
    def chidiff(a, b):
        d = fits.set_index("key")
        dchi = d.loc[a, "chi2"] - d.loc[b, "chi2"]
        ddf = d.loc[a, "df"] - d.loc[b, "df"]
        return dchi, int(ddf), stats.chi2.sf(dchi, ddf)
    fits["delta_chi2"], fits["delta_df"], fits["delta_p"] = np.nan, np.nan, np.nan
    for a, b in [("clim_1f", "clim_2f"), ("pj_1f", "pj_2f"), ("harman", "meas")]:
        dchi, ddf, p = chidiff(a, b)
        i = fits.index[fits["key"] == b][0]
        fits.loc[i, ["delta_chi2", "delta_df", "delta_p"]] = [dchi, ddf, p]
    fits.to_csv(tdir / "cfa_fit.csv", index=False)

    # standardised loadings
    l1 = cfa["clim_1f"][1].set_index("item")["std_loading"]
    l2 = cfa["clim_2f"][1].set_index("item")
    lp1 = cfa["pj_1f"][1].set_index("item")["std_loading"]
    lp2 = cfa["pj_2f"][1].set_index("item")
    items["loading_1f"] = items["item"].map(pd.concat([l1, lp1]))
    items["loading_2f"] = items["item"].map(pd.concat([l2["std_loading"], lp2["std_loading"]]))
    items["factor_2f"] = items["item"].map(pd.concat([l2["factor"], lp2["factor"]]))
    ir = pd.concat([item_rest(df[CLIM]), item_rest(df[PJ])])
    aid = pd.concat([alpha_if_deleted(df[CLIM]), alpha_if_deleted(df[PJ])])
    items["item_rest_r"] = items["item"].map(ir)
    items["alpha_if_deleted"] = items["item"].map(aid)
    items.to_csv(tdir / "item_statistics.csv", index=False)

    rel = []
    sets = {"Diversity and inclusion climate (12 items)": (CLIM, l1.loc[CLIM].values),
            "Organisational component (8 items)": (ORG, l2.loc[ORG, "std_loading"].values),
            "Personal component (4 items)": (PERS, l2.loc[PERS, "std_loading"].values),
            "Expected procedural justice (7 items)": (PJ, lp1.loc[PJ].values)}
    for nm, (it, lam) in sets.items():
        a = cronbach_alpha(df[it]); lo, hi = feldt_ci(a, N, len(it))
        om, ave = omega_cr_ave(lam)
        Rm = df[it].corr().values
        rel.append({"scale": nm, "k": len(it), "alpha": a, "alpha_lo": lo, "alpha_hi": hi,
                    "omega": om, "AVE": ave,
                    "mean_inter_item_r": Rm[np.triu_indices(len(it), 1)].mean(),
                    "KMO": kmo(df[it]) if len(it) > 2 else np.nan,
                    "mean": df[it].mean(axis=1).mean(), "sd": df[it].mean(axis=1).std(ddof=1)})
    rel = pd.DataFrame(rel)
    rel.to_csv(tdir / "reliability.csv", index=False)

    bart = {"climate": bartlett(df[CLIM]), "pj": bartlett(df[PJ])}
    pa_c = parallel_analysis(df[CLIM], N_PARALLEL, rng)
    pa_p = parallel_analysis(df[PJ], N_PARALLEL, rng)
    pa_c.to_csv(tdir / "parallel_analysis_climate.csv", index=False)
    # Exploratory factor analysis (ML, oblimin) to locate the sources of CFA misfit
    efa_rows = []
    for items_, k, tag in [(CLIM, 3, "climate"), (PJ, 2, "pj")]:
        fa = FactorAnalyzer(n_factors=k, rotation="oblimin", method="ml")
        fa.fit(df[items_])
        L = pd.DataFrame(fa.loadings_, index=items_, columns=[f"F{i + 1}" for i in range(k)])
        L["communality"] = fa.get_communalities()
        L.insert(0, "scale", tag)
        efa_rows.append(L.reset_index().rename(columns={"index": "item"}))
        phi = pd.DataFrame(fa.phi_, index=L.columns[1:k + 1], columns=L.columns[1:k + 1])
        phi.to_csv(tdir / f"efa_factor_correlations_{tag}.csv")
    pd.concat(efa_rows).to_csv(tdir / "efa_loadings.csv", index=False)
    pa_p.to_csv(tdir / "parallel_analysis_pj.csv", index=False)
    R["kmo"] = {"climate": kmo(df[CLIM]), "pj": kmo(df[PJ])}
    R["bartlett"] = {k: {"chi2": v[0], "df": v[1], "p": v[2]} for k, v in bart.items()}
    R["parallel_retain"] = {
        "climate": int((pa_c["observed"] > pa_c["random_p95"]).cumprod().sum()),
        "pj": int((pa_p["observed"] > pa_p["random_p95"]).cumprod().sum())}

    # ---------------- 5. Discriminant validity, common method --------
    ev_all = np.sort(np.linalg.eigvalsh(df[CLIM + PJ].corr().values))[::-1]
    lam_c = l1.loc[CLIM].values; lam_p = lp1.loc[PJ].values
    latent_r = cfa["meas"][2]
    R["discriminant"] = {
        "latent_r_climate_pj": next(iter(latent_r.values())),
        "htmt_climate_pj": htmt(df, CLIM, PJ),
        "htmt_org_pers": htmt(df, ORG, PERS),
        "ave_climate": np.mean(lam_c ** 2), "ave_pj": np.mean(lam_p ** 2),
        "latent_r_org_pers": next(iter(cfa["clim_2f"][2].values())),
        "harman_first_component_share": ev_all[0] / ev_all.sum(),
    }

    # ---------------- 6. Correlations --------------------------------
    cvars = {"climate_12": "D&I climate (12 items)", "climate_org": "Organisational component",
             "climate_pers": "Personal component", "pj": "Expected procedural justice",
             "age_ordinal": "Age (ordinal band)"}
    cr = []
    keys = list(cvars)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sub = df[[a, b]].dropna()
            r, lo, hi, p = pearson(sub[a], sub[b])
            cr.append({"var1": cvars[a], "var2": cvars[b], "n": len(sub), "r": r,
                       "ci_lo": lo, "ci_hi": hi, "p": p})
    cr = pd.DataFrame(cr)
    cr.to_csv(tdir / "correlations.csv", index=False)

    # Journal-format table, means, SDs, reliabilities on the diagonal and
    # pairwise correlations among the study variables
    dv = df.copy()
    dv["male"] = dv["gender"].map({"Male": 1, "Female": 0})
    dv["low_recruit"] = (dv["recruit_exp"] == "Low").astype(int)
    dv["neg_ai"] = (dv["ai_experience"] == "Negative").astype(int)
    tvars = [("climate_12", "Diversity and inclusion climate", CLIM),
             ("climate_org", "Organisational component", ORG),
             ("climate_pers", "Personal component", PERS),
             ("pj", "Expected procedural justice", PJ),
             ("age_ordinal", "Age band", None),
             ("male", "Gender (1 = male)", None),
             ("low_recruit", "Low recruitment experience (1 = yes)", None),
             ("neg_ai", "Negative experience of AI (1 = yes)", None)]
    rows_ = []
    for i, (v, nm, items_) in enumerate(tvars):
        row = {"variable": f"{i + 1}. {nm}", "M": dv[v].mean(), "SD": dv[v].std(ddof=1)}
        for j, (w, _, _) in enumerate(tvars):
            if j < i:
                sub = dv[[v, w]].dropna()
                r_, p_ = stats.pearsonr(sub[v], sub[w])
                row[str(j + 1)] = f"{r_:.2f}".replace("0.", ".").replace("-", "−") + \
                    ("**" if p_ < .01 else "*" if p_ < .05 else "")
            elif j == i:
                row[str(j + 1)] = f"({cronbach_alpha(dv[items_]):.2f})".replace("0.", ".") if items_ else ""
            else:
                row[str(j + 1)] = ""
        rows_.append(row)
    pd.DataFrame(rows_).to_csv(tdir / "descriptives_correlations.csv", index=False)
    R["correlation_table_n"] = {"all": N, "age": int(dv["age_ordinal"].notna().sum()),
                                "gender": int(dv["male"].notna().sum())}
    r, lo, hi, p = pearson(df["climate_12"], df["pj"])
    R["r_climate_pj"] = {"r": r, "lo": lo, "hi": hi, "p": p, "n": N}

    # PJ by AI experience
    aimeans = df.groupby("ai_experience")["pj"].agg(["count", "mean", "std"]).reindex(
        ["Negative", "Neutral", "Positive"])
    aimeans.to_csv(tdir / "pj_by_ai_experience.csv")
    R["pj_by_ai"] = aimeans["mean"].to_dict()

    # ---------------- 7. Main-effects regression ---------------------
    reg = df[df["regression_sample"] == 1].copy()
    reg["age_band5"] = pd.Categorical(reg["age_band5"], categories=AGE5)
    clim_mean = reg["climate_12"].mean()
    reg["climate_c"] = reg["climate_12"] - clim_mean
    Nr = len(reg)
    R["N_regression"] = Nr
    R["climate_centring_mean"] = clim_mean

    m0 = ols_hc3("pj ~ climate_c", reg)
    m1 = ols_hc3(FORMULA_MAIN, reg)
    m1_noclim = ols_hc3(FORMULA_MAIN.replace("climate_c + ", ""), reg)
    t1 = coef_table(m1)
    t1.to_csv(tdir / "main_effects.csv", index=False)
    beta_std = m1.params["climate_c"] * reg["climate_12"].std(ddof=1) / reg["pj"].std(ddof=1)
    dR2 = m1.rsquared - m1_noclim.rsquared
    f2 = dR2 / (1 - m1.rsquared)
    ai_terms = [t for t in m1.params.index if "ai_experience" in t]
    age_terms = [t for t in m1.params.index if "age_band5" in t and ":" not in t]
    R["m1"] = {
        "N": int(m1.nobs), "R2": m1.rsquared, "adjR2": m1.rsquared_adj,
        "F_robust": wald(m1, [t for t in m1.params.index if t != "Intercept"]),
        "b_climate": m1.params["climate_c"], "se_climate": m1.bse["climate_c"],
        "p_climate": m1.pvalues["climate_c"], "beta_std_climate": beta_std,
        "deltaR2_climate": dR2, "f2_climate": f2,
        "wald_ai": wald(m1, ai_terms), "wald_age": wald(m1, age_terms),
        "ai_pos_vs_neu": lincom(m1, {ai_terms[1]: 1, ai_terms[0]: -1}),
        "m0_b": m0.params["climate_c"], "m0_se": m0.bse["climate_c"], "m0_R2": m0.rsquared,
    }
    # Semi-partial contributions for each block
    blocks = {"D&I climate": ["climate_c"], "Recruitment experience": [MODS["recruit"]],
              "AI experience": [MODS["ai"]], "Age": [MODS["age"]], "Gender": [MODS["gender"]]}
    contrib = []
    for nm, b in blocks.items():
        f = FORMULA_MAIN
        for term in b:
            f = f.replace(f" + {term}", "").replace(f"{term} + ", "")
        r2 = smf.ols(f, reg).fit().rsquared
        contrib.append({"block": nm, "delta_R2": m1.rsquared - r2})
    pd.DataFrame(contrib).to_csv(tdir / "unique_variance.csv", index=False)

    # ---------------- 8. Moderation ----------------------------------
    m2 = ols_hc3(FORMULA_INT, reg)
    t2 = coef_table(m2)
    t2.to_csv(tdir / "interaction_model_full.csv", index=False)
    int_terms = [t for t in m2.params.index if t.startswith("climate_c:")]
    ti = t2[t2["term"].isin(int_terms)].copy()
    ti["p_bh"] = multipletests(ti["p"], method="fdr_bh")[1]
    ti.to_csv(tdir / "interaction_terms.csv", index=False)
    omni = []
    for key, mod in MODS.items():
        terms = [t for t in int_terms if mod in t]
        F, d1, d2, p = wald(m2, terms)
        omni.append({"moderator": key, "F": F, "df1": d1, "df2": d2, "p": p})
    F, d1, d2, p = wald(m2, int_terms)
    omni.append({"moderator": "all", "F": F, "df1": d1, "df2": d2, "p": p})
    omni = pd.DataFrame(omni)
    omni["p_bh_moderators"] = np.nan
    omni.loc[omni["moderator"] != "all", "p_bh_moderators"] = multipletests(
        omni.loc[omni["moderator"] != "all", "p"], method="fdr_bh")[1]
    omni.to_csv(tdir / "moderation_omnibus.csv", index=False)
    # nested F test (classical) for the interaction block, as a cross-check
    m1_plain = smf.ols(FORMULA_MAIN, reg).fit()
    m2_plain = smf.ols(FORMULA_INT, reg).fit()
    Fn, pn, dn = m2_plain.compare_f_test(m1_plain)
    R["m2"] = {"R2": m2.rsquared, "adjR2": m2.rsquared_adj, "N": int(m2.nobs),
               "nested_F_classical": (Fn, pn, dn), "deltaR2": m2.rsquared - m1.rsquared}

    # Age as a linear trend in the interaction
    ra = reg.copy()
    m_age_lin = ols_hc3(FORMULA_MAIN.replace(f"C(age_band5, Treatment('18 to 24'))", "age_ordinal")
                        + " + climate_c:age_ordinal", ra)
    R["age_linear_interaction"] = {
        "b": m_age_lin.params["climate_c:age_ordinal"], "se": m_age_lin.bse["climate_c:age_ordinal"],
        "p": m_age_lin.pvalues["climate_c:age_ordinal"]}

    # One moderator at a time
    single = []
    for key, mod in MODS.items():
        m = ols_hc3(FORMULA_MAIN + f" + climate_c:{mod}", reg)
        terms = [t for t in m.params.index if t.startswith("climate_c:")]
        F, d1, d2, p = wald(m, terms)
        single.append({"moderator": key, "F": F, "df1": d1, "df2": d2, "p": p})
    single = pd.DataFrame(single)
    single["p_bh"] = multipletests(single["p"], method="fdr_bh")[1]
    single.to_csv(tdir / "moderation_one_at_a_time.csv", index=False)

    # Conditional climate slopes within each moderator level. Each slope is
    # computed from a model in which only that moderator is interacted with
    # climate, so the slope is the climate association within the group with
    # all other variables held at their main effects.
    slopes = []
    for key, mod in MODS.items():
        m = ols_hc3(FORMULA_MAIN + f" + climate_c:{mod}", reg)
        var = {"recruit": "recruit_exp", "ai": "ai_experience",
               "age": "age_band5", "gender": "gender"}[key]
        levels = {"recruit": ["High", "Low"], "ai": ["Negative", "Neutral", "Positive"],
                  "age": AGE5, "gender": ["Female", "Male"]}[key]
        for lev in levels:
            w = {"climate_c": 1.0}
            term = f"climate_c:{mod}[T.{lev}]"
            if term in m.params.index:
                w[term] = 1.0
            est, se, p, lo, hi = lincom(m, w)
            slopes.append({"moderator": key, "level": lev, "n": int((reg[var] == lev).sum()),
                           "slope": est, "se": se, "p": p, "ci_lo": lo, "ci_hi": hi})
    slopes = pd.DataFrame(slopes)
    slopes.to_csv(tdir / "conditional_slopes.csv", index=False)

    # Exploratory follow-up (post hoc). The neutral and positive slopes are
    # almost identical, so the pattern is summarised by one contrast, the
    # climate slope among respondents with negative AI experience against the
    # average slope of the other two groups. Labelled post hoc in reporting.
    mai = ols_hc3(FORMULA_MAIN + f" + climate_c:{MODS['ai']}", reg)
    tn = f"climate_c:{MODS['ai']}[T.Neutral]"
    tp = f"climate_c:{MODS['ai']}[T.Positive]"
    R["posthoc_ai_contrast"] = lincom(mai, {tn: 0.5, tp: 0.5})
    wg = []
    for lev in ["Negative", "Neutral", "Positive"]:
        s_ = reg[reg["ai_experience"] == lev]
        r_, lo_, hi_, p_ = pearson(s_["climate_12"], s_["pj"])
        wg.append({"ai_experience": lev, "n": len(s_), "r": r_, "ci_lo": lo_, "ci_hi": hi_, "p": p_,
                   "pj_mean": s_["pj"].mean(), "pj_sd": s_["pj"].std(ddof=1),
                   "climate_mean": s_["climate_12"].mean(), "climate_sd": s_["climate_12"].std(ddof=1)})
    pd.DataFrame(wg).to_csv(tdir / "within_group_correlations_ai.csv", index=False)

    k_int = len(m2.params) - 1
    R["power"] = {"mdes_f2_1df_interaction": min_detectable_f2(Nr, k_int, 1),
                  "mdes_f2_main_climate": min_detectable_f2(Nr, len(m1.params) - 1, 1)}

    # ---------------- 9. Diagnostics ---------------------------------
    resid = m1_plain.resid
    bp = het_breuschpagan(resid, m1_plain.model.exog)
    jb = jarque_bera(resid)
    sw = stats.shapiro(resid)
    reset = linear_reset(m1_plain, power=[2, 3], use_f=True)
    reset_ai = linear_reset(smf.ols(FORMULA_MAIN + f" + climate_c:{MODS['ai']}", reg).fit(),
                            power=[2, 3], use_f=True)
    reset_m2 = linear_reset(m2_plain, power=[2, 3], use_f=True)
    reg["climate_c2"] = reg["climate_c"] ** 2
    mq = ols_hc3(FORMULA_MAIN + " + climate_c2", reg)
    infl = m1_plain.get_influence()
    cooks = infl.cooks_distance[0]
    thr = 4 / Nr
    gv = gvif(reg, FORMULA_MAIN.split("~", 1)[1])
    gv["label"] = gv["term"].map(label)
    gv.to_csv(tdir / "gvif.csv", index=False)
    diag = {
        "breusch_pagan_LM": bp[0], "breusch_pagan_p": bp[1], "breusch_pagan_df": int(m1_plain.df_model),
        "jarque_bera": jb[0], "jarque_bera_p": jb[1], "residual_skew": jb[2], "residual_kurtosis": jb[3],
        "shapiro_W": sw.statistic, "shapiro_p": sw.pvalue,
        "reset_F": float(reset.fvalue), "reset_p": float(reset.pvalue),
        "reset_df": (int(reset.df_num), int(reset.df_denom)),
        "reset_with_ai_interaction_F": float(reset_ai.fvalue), "reset_with_ai_interaction_p": float(reset_ai.pvalue),
        "reset_full_interaction_F": float(reset_m2.fvalue), "reset_full_interaction_p": float(reset_m2.pvalue),
        "quad_b": mq.params["climate_c2"], "quad_se": mq.bse["climate_c2"], "quad_p": mq.pvalues["climate_c2"],
        "cooks_threshold": thr, "n_cooks_above": int((cooks > thr).sum()), "max_cooks": float(cooks.max()),
        "gvif_climate": float(gv.loc[gv["term"] == "climate_c", "GVIF"].iloc[0]),
        "gvif_max_adj": float(gv["GVIF_adj"].max()),
    }
    R["diagnostics"] = diag
    pd.DataFrame([diag]).T.to_csv(tdir / "diagnostics.csv", header=["value"])

    # ---------------- 10. Robustness ---------------------------------
    rob = []
    def add(spec, res, term="climate_c", note=""):
        ci = res.conf_int().loc[term]
        rob.append({"specification": spec, "b": res.params[term], "se": res.bse[term],
                    "p": res.pvalues[term], "ci_lo": ci[0], "ci_hi": ci[1],
                    "N": int(res.nobs), "note": note})
    add("Main-effects model", m1)
    keep = reg[cooks <= thr]
    add("Excluding influential cases (Cook's D > 4/N)", ols_hc3(FORMULA_MAIN, keep))
    add("Adding stakeholder role", ols_hc3(FORMULA_MAIN + " + C(role, Treatment('Academic staff'))", reg))
    add("Adding university sector", ols_hc3(FORMULA_MAIN + " + C(sector, Treatment('Public'))", reg))
    add("Adding role and sector", ols_hc3(FORMULA_MAIN + " + C(role, Treatment('Academic staff')) + C(sector, Treatment('Public'))", reg))
    add("University fixed effects", ols_hc3(FORMULA_MAIN + " + C(university)", reg))
    add("Recruitment experience in three categories",
        ols_hc3(FORMULA_MAIN.replace("C(recruit_exp, Treatment('High'))",
                                     "C(recruit_role, Treatment('Both'))"), reg))
    add("Age as an ordinal linear term",
        ols_hc3(FORMULA_MAIN.replace("C(age_band5, Treatment('18 to 24'))", "age_ordinal"), reg))
    sl = reg[reg["straightline_both"] == 0]
    add("Excluding respondents who straight-lined both scales", ols_hc3(FORMULA_MAIN, sl))
    fr = reg[reg["fast_responder"] == 0]
    R["careless_flags"] = {
        "median_duration_sec_eligible": float(elig["duration_sec"].median()),
        "fast_threshold_sec": float(0.5 * elig["duration_sec"].median()),
        "straightliners_regression_sample": int(reg["straightline_both"].sum()),
        "fast_responders_regression_sample": int(reg["fast_responder"].sum()),
    }
    add("Excluding fast responders (below half the median duration)", ols_hc3(FORMULA_MAIN, fr))
    # include attention-check failures
    e = elig[(elig["gender"] != "Not disclosed") & (elig["age_band"] != "Not disclosed")].copy()
    e["age_band5"] = pd.Categorical(e["age_band5"], categories=AGE5)
    e["climate_c"] = e["climate_12"] - clim_mean
    add("Including attention check failures", ols_hc3(FORMULA_MAIN, e))
    # retain non-disclosers with an explicit category
    full = df.copy()
    full["age_band5"] = pd.Categorical(full["age_band5"], categories=AGE5 + ["Not disclosed"])
    full["climate_c"] = full["climate_12"] - clim_mean
    add("Retaining respondents who did not disclose gender or age", ols_hc3(FORMULA_MAIN, full))
    # alternative operationalisations
    reg["climate_c_org"] = reg["climate_org"] - reg["climate_org"].mean()
    add("Organisational component as predictor (8 items)",
        ols_hc3(FORMULA_MAIN.replace("climate_c", "climate_c_org"), reg), term="climate_c_org")
    for nm, its in [("Outcome PJ items 1, 2, 4, and 6", PJ_A), ("Outcome PJ items 3, 5, and 7", PJ_B)]:
        d = reg.copy(); d["pj"] = d[its].mean(axis=1)
        add(nm, ols_hc3(FORMULA_MAIN, d))
    # wild cluster bootstrap by university
    wcb = wild_cluster_bootstrap(FORMULA_MAIN, reg, "university", "climate_c", N_BOOT, rng)
    rob.append({"specification": "Clustered by university, wild cluster bootstrap (8 clusters)",
                "b": wcb["b"], "se": wcb["se_cr1"], "p": wcb["p_wcr"], "ci_lo": np.nan,
                "ci_hi": np.nan, "N": Nr, "note": "SE is CR1; p from restricted wild bootstrap, Webb weights"})
    rob = pd.DataFrame(rob)
    rob.to_csv(tdir / "robustness.csv", index=False)
    R["wild_cluster_bootstrap"] = wcb

    # Robustness of the moderation null to the same sample variants
    modrob = []
    for nm, d in [("Main sample", reg), ("Including attention check failures", e),
                  ("Excluding straight-liners", sl), ("Excluding fast responders", fr)]:
        mm = ols_hc3(FORMULA_INT, d)
        it = [t for t in mm.params.index if t.startswith("climate_c:")]
        F, d1, d2, p = wald(mm, it)
        modrob.append({"sample": nm, "N": int(mm.nobs), "F": F, "df1": d1, "df2": d2, "p": p,
                       "min_p_single_term": float(mm.pvalues[it].min()),
                       "min_p_term": label(str(mm.pvalues[it].idxmin()))})
    pd.DataFrame(modrob).to_csv(tdir / "moderation_robustness.csv", index=False)

    # ---------------- 11. Decomposition -------------------------------
    reg["climate_c_pers"] = reg["climate_pers"] - reg["climate_pers"].mean()
    md = ols_hc3(FORMULA_MAIN.replace("climate_c", "climate_c_org + climate_c_pers"), reg)
    td = coef_table(md)
    td = td[td["term"].isin(["climate_c_org", "climate_c_pers"])]
    td["label"] = td["term"].map({"climate_c_org": "Organisational component (centred)",
                                  "climate_c_pers": "Personal component (centred)"})
    eq = lincom(md, {"climate_c_org": 1, "climate_c_pers": -1})
    td.to_csv(tdir / "decomposition.csv", index=False)
    R["decomposition"] = {"R2": md.rsquared, "N": int(md.nobs), "diff_org_minus_pers": eq,
                          "vif_r_org_pers": float(np.corrcoef(reg["climate_org"], reg["climate_pers"])[0, 1])}

    # ---------------- 12. Figures ------------------------------------
    style()
    # Figure 3.4 gender by role
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    tab = pd.crosstab(df["role"], df["gender"]).reindex(index=ROLES, columns=["Female", "Male", "Not disclosed"])
    stacked_pct(ax, tab, [NAVY, ORANGE, LIGHT], "Gender")
    save(fig, fdir / "Figure_3_4_gender_by_role.png")
    # Figure 3.5 age by role
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    tab = pd.crosstab(df["role"], df["age_band"]).reindex(index=ROLES, columns=AGE6 + ["Not disclosed"]).fillna(0)
    stacked_pct(ax, tab, RAMP + [LIGHT], "Age band")
    save(fig, fdir / "Figure_3_5_age_by_role.png")
    # Figure 3.6 sector and university
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    uc = df.groupby(["sector", "university"]).size().reset_index(name="n")
    uc["sector"] = pd.Categorical(uc["sector"], ["Public", "Private"])
    uc = uc.sort_values(["sector", "n"], ascending=[True, True])
    colors = [NAVY if s == "Public" else ORANGE for s in uc["sector"]]
    ax.barh(range(len(uc)), uc["n"], color=colors, height=0.6)
    for i, v in enumerate(uc["n"]):
        ax.text(v + 0.8, i, str(v), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(uc)))
    ax.set_yticklabels(uc["university"].str.replace(" University", ""))
    ax.set_xlabel("Number of respondents")
    from matplotlib.patches import Patch
    ax.set_xlim(0, uc["n"].max() * 1.12)
    ax.legend(handles=[Patch(color=NAVY, label=f"Public (n = {(df['sector']=='Public').sum()})"),
                       Patch(color=ORANGE, label=f"Private (n = {(df['sector']=='Private').sum()})")],
              loc="lower right", fontsize=8.5)
    save(fig, fdir / "Figure_3_6_sector_university.png")
    # Figure 3.7 roles within sector
    fig, ax = plt.subplots(figsize=(6.3, 2.2))
    tab = pd.crosstab(df["sector"], df["role"]).reindex(index=["Public", "Private"], columns=ROLES)
    stacked_pct(ax, tab, [ROLE_COL[r] for r in ROLES], "Stakeholder role")
    save(fig, fdir / "Figure_3_7_role_within_sector.png")
    # Figure 3.8 recruitment role experienced
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    tab = pd.crosstab(df["role"], df["recruit_role"]).reindex(
        index=ROLES, columns=["Applicant only", "Recruiter only", "Both"]).fillna(0)
    stacked_pct(ax, tab, [RAMP[1], RAMP[3], NAVY], "Role experienced in recruitment")
    save(fig, fdir / "Figure_3_8_recruitment_by_role.png")
    # Figure 3.9 experience of AI
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    tab = pd.crosstab(df["role"], df["ai_experience"]).reindex(
        index=ROLES, columns=["Negative", "Neutral", "Positive"]).fillna(0)
    stacked_pct(ax, tab, [MAROON, LIGHT, NAVY], "Overall experience of AI or digital technology")
    save(fig, fdir / "Figure_3_9_ai_experience_by_role.png")
    # Figure 3.10 distributions of the two composites by role
    jrng = np.random.default_rng(SEED + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    dist_by_role(axes[0], df, "climate_12", "Score (1 to 5)", jrng)
    dist_by_role(axes[1], df, "pj", "", jrng)
    axes[0].set_title("(a) Diversity and inclusion climate")
    axes[1].set_title("(b) Expected procedural justice")
    fig.tight_layout()
    save(fig, fdir / "Figure_3_10_construct_distributions_by_role.png")
    # Figure 3.12 scatter
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # Many respondents share identical score pairs, so each distinct pair is drawn
    # once at its exact position, with marker area proportional to the number of
    # respondents at that position.
    cnt = df.groupby(["climate_12", "pj"]).size().reset_index(name="n")
    area = lambda k: 14 * k
    ax.scatter(cnt["climate_12"], cnt["pj"], s=area(cnt["n"]), facecolors="none",
               edgecolors=NAVY, linewidths=0.8, alpha=0.85, zorder=2)
    bl = smf.ols("pj ~ climate_12", df).fit()
    xs = np.linspace(df["climate_12"].min(), df["climate_12"].max(), 100)
    pr = bl.get_prediction(pd.DataFrame({"climate_12": xs})).summary_frame()
    ax.fill_between(xs, pr["mean_ci_lower"], pr["mean_ci_upper"], color=LIGHT, alpha=0.6,
                    linewidth=0, zorder=1)
    ax.plot(xs, pr["mean"], color=NAVY, linewidth=1.4, zorder=3)
    ax.set_xlabel("Diversity and inclusion climate (1 to 5)")
    ax.set_ylabel("Expected procedural justice (1 to 5)")
    rr = R["r_climate_pj"]
    ax.text(0.5, -0.245,
            f"r = {no_zero(rr['r'])}, 95% CI [{no_zero(rr['lo'])}, {no_zero(rr['hi'])}], N = {N}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK)
    for k in [1, 5, 10]:
        ax.scatter([], [], s=area(k), facecolors="none", edgecolors=NAVY, linewidths=0.8,
                   label=f"{k} respondent" + ("s" if k > 1 else ""))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.5,
              handletextpad=0.4, columnspacing=1.6)
    ax.set_xlim(1.9, 5.1); ax.set_ylim(0.9, 5.1)
    save(fig, fdir / "Figure_3_12_scatter.png")
    # Figure 3.13 coefficient plot
    tt = pd.concat([t1[t1["term"] == "climate_c"], t1[~t1["term"].isin(["Intercept", "climate_c"])]])
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    forest(ax, tt["label"].tolist(), tt["b"].values, tt["ci_lo"].values, tt["ci_hi"].values,
           xlabel="Coefficient (b) with 95% confidence interval")
    save(fig, fdir / "Figure_3_13_coefficients.png")
    # Figure 3.14 conditional slopes
    lab = {"recruit": "Recruitment experience", "ai": "AI experience", "age": "Age", "gender": "Gender"}
    sl_lab = [f"{lab[r.moderator]}: {r.level} (n = {r.n})" for r in slopes.itertuples()]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    forest(ax, sl_lab, slopes["slope"].values, slopes["ci_lo"].values, slopes["ci_hi"].values,
           xlabel="Climate slope within group, 95% CI (HC3)")
    ax.axvline(m1.params["climate_c"], color=ORANGE, linewidth=1.2, zorder=1,
               label=f"Average slope ({m1.params['climate_c']:.3f})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    save(fig, fdir / "Figure_3_14_conditional_slopes.png")
    # Predicted expected PJ across climate by AI experience, from the model with
    # the climate by AI experience interaction, other covariates at their modes.
    ai_col = {"Negative": MAROON, "Neutral": GREY, "Positive": NAVY}
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    base = reg.iloc[[0]].copy()
    for lev in ["Negative", "Neutral", "Positive"]:
        s_ = reg[reg["ai_experience"] == lev]
        xs = np.linspace(s_["climate_c"].quantile(0.05), s_["climate_c"].quantile(0.95), 60)
        grid = pd.concat([base] * len(xs), ignore_index=True)
        grid["climate_c"] = xs
        grid["ai_experience"] = lev
        for var in ["recruit_exp", "gender"]:
            grid[var] = reg[var].mode()[0]
        grid["age_band5"] = pd.Categorical([reg["age_band5"].mode()[0]] * len(xs), categories=AGE5)
        pr = mai.get_prediction(grid).summary_frame(alpha=0.05)
        x_raw = xs + clim_mean
        ax.fill_between(x_raw, pr["mean_ci_lower"], pr["mean_ci_upper"], color=ai_col[lev],
                        alpha=0.15, linewidth=0)
        ax.plot(x_raw, pr["mean_ci_lower"], color=ai_col[lev], linewidth=0.6, linestyle=":")
        ax.plot(x_raw, pr["mean_ci_upper"], color=ai_col[lev], linewidth=0.6, linestyle=":")
        ax.plot(x_raw, pr["mean"], color=ai_col[lev], linewidth=1.8,
                linestyle="--" if lev == "Neutral" else "-", label=f"{lev} (n = {len(s_)})")
    ax.set_xlabel("Diversity and inclusion climate (1 to 5)")
    ax.set_ylabel("Predicted expected procedural justice")
    ax.legend(title="Experience of AI", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              fontsize=9, title_fontsize=9, handlelength=2.4, columnspacing=1.4)
    save(fig, fdir / "Figure_3_15_predicted_by_ai_experience.png")
    # Figure 3.11 parallel analysis
    # Observed eigenvalues as bars, the random-data benchmark as a dashed line.
    # Components whose eigenvalue exceeds the benchmark are retained (dark bars).
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, pa, ttl in [(axes[0], pa_c, "(a) Diversity and inclusion climate"),
                        (axes[1], pa_p, "(b) Expected procedural justice")]:
        keep = (pa["observed"] > pa["random_p95"]).cumprod().astype(bool)
        ax.bar(pa["component"], pa["observed"], width=0.6,
               color=np.where(keep, NAVY, LIGHT), zorder=2)
        ax.plot(pa["component"], pa["random_p95"], color=ORANGE, linewidth=1.5,
                linestyle="--", marker="o", markersize=3.5, zorder=3)
        top = pa["observed"].iloc[0]
        ax.annotate(f"Retained\n({top:.2f})", xy=(1, top), xytext=(2.3, top * 0.86),
                    fontsize=9, color=INK, va="center",
                    arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.7})
        last = pa["component"].iloc[-1]
        ax.text(last + 0.4, pa["random_p95"].iloc[-1] + 0.08 * top, "Random data\n(95th percentile)",
                fontsize=8.5, color=INK, ha="right", va="bottom")
        ax.set_title(ttl)
        ax.set_xlabel("Component")
        ax.set_xticks(pa["component"])
        ax.set_ylim(0, top * 1.08)
    axes[0].set_ylabel("Eigenvalue")
    fig.tight_layout()
    save(fig, fdir / "Figure_3_11_parallel_analysis.png")
    # Figure A.1 residual diagnostics
    # Both panels use internally studentised residuals so they share one scale.
    # The lowess band is a pairs bootstrap; the quantile-plot envelope simulates
    # residuals from the fitted design matrix under normal errors (Atkinson, 1985).
    rng_fig = np.random.default_rng(SEED + 2)
    fitted = m1_plain.fittedvalues.to_numpy()
    sres = infl.resid_studentized_internal
    n_obs = len(sres)
    far = np.abs(sres) > 3
    grid_x = np.linspace(fitted.min(), fitted.max(), 120)
    lw = sm.nonparametric.lowess(sres, fitted, frac=0.6, return_sorted=True)
    lw_fit = np.interp(grid_x, lw[:, 0], lw[:, 1])
    boot = np.empty((1000, grid_x.size))
    for b in range(boot.shape[0]):
        idx = rng_fig.integers(0, n_obs, n_obs)
        lb = sm.nonparametric.lowess(sres[idx], fitted[idx], frac=0.6, return_sorted=True)
        boot[b] = np.interp(grid_x, lb[:, 0], lb[:, 1])
    band_lo, band_hi = np.percentile(boot, [2.5, 97.5], axis=0)

    Xd = m1_plain.model.exog
    H = Xd @ np.linalg.pinv(Xd)
    M = np.eye(n_obs) - H
    h = np.diag(H)
    sims = np.empty((1000, n_obs))
    p_par = Xd.shape[1]
    for b in range(sims.shape[0]):
        e = M @ rng_fig.standard_normal(n_obs)
        s2 = (e @ e) / (n_obs - p_par)
        sims[b] = np.sort(e / np.sqrt(s2 * (1 - h)))
    env_lo, env_hi = np.percentile(sims, [2.5, 97.5], axis=0)
    env_mid = np.median(sims, axis=0)
    order = np.argsort(sres)
    q_obs = sres[order]
    q_theo = stats.norm.ppf((np.arange(1, n_obs + 1) - 0.375) / (n_obs + 0.25))

    PT, HI, LINE = NAVY, MAROON, NAVY
    lim = np.ceil(np.abs(sres).max() * 2) / 2 + 0.25
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    ax = axes[0]
    for yv in (-3, 3):
        ax.axhline(yv, color=INK2, linewidth=0.7, linestyle=":", zorder=1)
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
    ax.fill_between(grid_x, band_lo, band_hi, color=LIGHT, alpha=0.6, linewidth=0, zorder=2)
    ax.scatter(fitted[~far], sres[~far], s=12, facecolors="none", edgecolors=PT,
               linewidths=0.6, alpha=0.8, zorder=3)
    ax.scatter(fitted[far], sres[far], s=24, color=HI, zorder=4)
    ax.plot(grid_x, lw_fit, color=ORANGE, linewidth=1.6, zorder=5)
    ax.set_xlabel("Fitted values (expected procedural justice)")
    ax.set_ylabel("Studentised residuals")
    ax.set_title("(a) Residuals against fitted values")

    ax = axes[1]
    ax.fill_between(q_theo, env_lo, env_hi, color=LIGHT, alpha=0.6, linewidth=0, zorder=1)
    ax.plot(q_theo, env_mid, color=ORANGE, linewidth=1.2, zorder=2)
    outside = np.abs(q_obs) > 3
    n_out_env = int(((q_obs < env_lo) | (q_obs > env_hi)).sum())
    ax.scatter(q_theo[~outside], q_obs[~outside], s=12, facecolors="none", edgecolors=PT,
               linewidths=0.6, alpha=0.8, zorder=3)
    ax.scatter(q_theo[outside], q_obs[outside], s=24, color=HI, zorder=4)
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_title("(b) Normal quantile plot")
    for ax in axes:
        ax.set_ylim(-lim, lim)
    fig.tight_layout(w_pad=1.5)
    top = int(np.argmax(np.abs(sres)))
    R["residual_plot"] = {"n_beyond_3": int(far.sum()), "n_outside_envelope": n_out_env,
                          "lowess_band_reps": 1000, "envelope_reps": 1000,
                          "largest_studentised_residual": float(sres[top]),
                          "largest_case_fitted": float(fitted[top]),
                          "largest_case_observed_pj": float(reg["pj"].iloc[top]),
                          "largest_case_climate": float(reg["climate_12"].iloc[top]),
                          "largest_case_cooks_d": float(infl.cooks_distance[0][top])}
    save(fig, fdir / "Figure_A_1_residual_diagnostics.png")

    # ---------------- write key results -------------------------------
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    R["role_tests"] = role_tests.to_dict(orient="records")
    with open(out_dir / "key_results.json", "w") as fh:
        json.dump(clean(R), fh, indent=2)
    print(json.dumps(clean({k: R[k] for k in ["N_analytic", "N_regression", "r_climate_pj", "m1", "m2"]}), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=Path("data"), type=Path)
    ap.add_argument("--out", default=Path("outputs"), type=Path)
    args = ap.parse_args()
    main(args.data, args.out)
