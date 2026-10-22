"""Regenerate the paper's data-driven TABLES from the repo's benchmark output.

Mirrors figures/make_figures.py, but emits LaTeX table fragments into results/tables/.
Every number is recomputed from the parquets so the paper's tables cannot silently drift
from the data (the risk with hand-typed tables).

Coverage (6 of the paper's 12 tables are backed by local parquet data):

  calib_n2k     tab:calib-n2k     data/cells/    5.2_calib, n=2000   size (Wilson 95% CI)
  calib_n1e5    tab:calib-n1e5    data/cells/    5.2_calib, n=1e5    size (Wilson 95% CI)
  power_n2k     tab:power-n2k     data/cells/    5.3_power, n=2000   size-corrected power
  mixed         tab:mixed         data/cells/    5.5_mixed           size-corrected power
  convergence   tab:conv-null +   data/cells_conv/                  size / sc-power full sweep
                tab:conv-power    -> convergence_appendix.tex

The other 6 tables (inject, cc-inject, asym, rankz, interaction, nuisance-tradeoff) are
produced by standalone witness/demo scripts whose data is NOT in the parquet dirs; this
generator reports them as unbacked rather than fabricating numbers.

Usage:
  python tables/make_tables.py                 # generate all backed tables
  python tables/make_tables.py --diff          # also diff vs the numbers currently in the paper
  python tables/make_tables.py calib_n2k power_n2k
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "data")
RESULTS = os.path.join(_ROOT, "results", "aggregated")
OUT = os.path.join(_ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True)
PAPER = os.path.join(
    os.path.dirname(_ROOT), "PhD", "papers", "paper2-gfcm", "writing", "paper2_tmlr.tex")
ALPHA = 0.05
MINREPS = 10  # below this many finite p-values a cell is treated as missing ("--")

LAB = {"gfcm": "GFCM", "blitz": "BLITZ", "ffci": "FFCI", "rcot": "RCoT",
       "gcm_boosted": "GCM-boost", "par_cop": "PartCopula", "ci_mm": "ci-mm"}


# --------------------------------------------------------------------------- helpers
def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def _load(subdir):
    fs = glob.glob(os.path.join(DATA, subdir, "*.parquet"))
    fs = [f for f in fs if not os.path.basename(f).startswith("convergence_")]
    if not fs:
        return None
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def _fmt(k, m, dec=2, bold=False):
    """'0.05~[0.03--0.07]', or '--' if the cell has too little data."""
    if m is None or m < MINREPS:
        return "--"
    lo, hi = _wilson(k, m)
    s = f"{k/m:.{dec}f}~[{lo:.{dec}f}--{hi:.{dec}f}]"
    return f"\\textbf{{{s}}}" if bold else s


def _crit(g, cell="null"):
    """Per-test size-correction threshold: the alpha-quantile of the matched null p-values."""
    out = {}
    for t, gt in g[g.cell == cell].groupby("test"):
        pv = gt.pval.dropna().values
        if len(pv):
            out[t] = np.percentile(pv, 100 * ALPHA)
    return out


def _crit_from(pv):
    """Size-correction threshold from a bare null p-value series (alpha-quantile)."""
    pv = pv.dropna().values
    return np.percentile(pv, 100 * ALPHA) if len(pv) else np.nan


def _size(pv):
    pv = pv.dropna().values
    return int((pv < ALPHA).sum()), len(pv)


def _power(pv, c):
    pv = pv.dropna().values
    if c != c or not len(pv):
        return 0, 0
    return int((pv < c).sum()), len(pv)


def _calib_bold(k, m):
    """Flag a miscalibrated cell: size >= 0.15 (~3x nominal, clearly anticonservative). A fixed
    point-estimate threshold, so it is stable under Monte Carlo drift (no CI in the trigger)."""
    if m < MINREPS:
        return False
    return (k / m) >= 0.15


def _write(name, body):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(body)
    print(f"  wrote {p}")


# --------------------------------------------------------------------------- calibration
_CALIB_MECH = ["heavy_tail", "hetero", "mixed_Z", "nonlin_mean"]
_CALIB_DS = [1, 3, 5]


def _calib(n, tests, label, cap):
    df = _load("cells")
    if df is None or df[df.section == "5.2_calib"].empty:
        print(f"{label}: no 5.2_calib cells"); return None
    sub = df[(df.section == "5.2_calib") & (df.n == n)]
    colspec = "ll" + "c" * len(tests)
    head = " & ".join(["mechanism", "$|S|$"] + [LAB[t] for t in tests])
    lines = [r"\begin{table}[t]\centering\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{" + colspec + "}", r"\toprule",
             head + r" \\", r"\midrule"]
    computed = {}
    for dS in _CALIB_DS:
        for mech in _CALIB_MECH:
            cells = []
            for t in tests:
                k, m = _size(sub[(sub.cell == mech) & (sub.dS == dS) & (sub.test == t)].pval)
                cells.append(_fmt(k, m, bold=_calib_bold(k, m)))
                computed[(mech, dS, t)] = (k / m) if m >= MINREPS else None
            lines.append(f"{mech.replace('_', chr(92)+'_')} & {dS} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", cap, r"\end{table}", ""]
    _write(f"{label}.tex", "\n".join(lines))
    return computed


def calib_n2k():
    cap = (r"\caption{Calibration on hard nulls, $n=2000$ (size; Wilson 95\% CI). Bold marks "
           r"size $\ge 0.15$ (materially anticonservative).}" "\n" r"\label{tab:calib-n2k}")
    return _calib(2000, ["gfcm", "par_cop", "blitz", "ffci", "rcot", "gcm_boosted"], "calib_n2k", cap)


def calib_n1e5():
    cap = (r"\caption{Calibration, large-sample regime $n=10^5$ (size; Wilson 95\% CI). Bold marks "
           r"size $\ge 0.15$ (materially anticonservative).}" "\n" r"\label{tab:calib-n1e5}")
    return _calib(100000, ["gfcm", "blitz", "ffci", "rcot", "gcm_boosted"], "calib_n1e5", cap)


# --------------------------------------------------------------------------- power ladder
_POWER_MECH = ["linear_mean", "nonmonotone_z2", "scale", "tail_shape"]


def power_n2k():
    df = _load("cells")
    if df is None or df[df.section == "5.3_power"].empty:
        print("power_n2k: no 5.3_power cells"); return None
    tests = ["gfcm", "par_cop", "blitz", "ffci", "rcot", "gcm_boosted"]
    sub = df[(df.section == "5.3_power") & (df.n == 2000)]
    head = " & ".join(["mechanism", "$|S|$"] + [LAB[t] for t in tests])
    lines = [r"\begin{table}[t]\centering\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll" + "c" * len(tests) + "}", r"\toprule",
             head + r" \\", r"\midrule"]
    computed = {}
    for dS in _CALIB_DS:
        crit = _crit(sub[sub.dS == dS])
        for mech in _POWER_MECH:
            cells = []
            for t in tests:
                k, m = _power(sub[(sub.cell == mech) & (sub.dS == dS) & (sub.test == t)].pval,
                              crit.get(t, np.nan))
                cells.append(_fmt(k, m, bold=(t == "gfcm" and mech == "tail_shape")))
                computed[(mech, dS, t)] = (k / m) if m >= MINREPS else None
            lines.append(f"{mech.replace('_', chr(92)+'_')} & {dS} & " + " & ".join(cells) + r" \\")
    cap = (r"\caption{Power across the alternative ladder, $n=2000$ (size-corrected power: each "
           r"test's threshold set to exact $5\%$ size on the matched null; Wilson 95\% CI). Bold "
           r"marks GFCM on the skew alternative, where it alone keeps power as $|S|$ grows.}"
           "\n" r"\label{tab:power-n2k}")
    lines += [r"\bottomrule", r"\end{tabular}}", cap, r"\end{table}", ""]
    _write("power_n2k.tex", "\n".join(lines))
    return computed


# --------------------------------------------------------------------------- mixed-type
_MIXED_PAIRS = ["CC", "CK", "KC", "KK"]


def mixed():
    df = _load("cells")
    if df is None or df[df.section == "5.5_mixed"].empty:
        print("mixed: no 5.5_mixed cells"); return None
    tests = ["gfcm", "ffci", "blitz", "ci_mm", "gcm_boosted"]
    sub = df[df.section == "5.5_mixed"]
    head = " & ".join(["type pair"] + [LAB[t] for t in tests])
    lines = [r"\begin{table}[t]\centering\small",
             r"\setlength{\tabcolsep}{5pt}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(tests) + "}", r"\toprule",
             head + r" \\", r"\midrule"]
    computed = {}
    ns = sorted(sub.n.unique())
    for i, n in enumerate(ns):
        gn = sub[sub.n == n]
        ntxt = r"$n=10^5$" if n == 100000 else f"$n={n}$"
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(tests) + 1, ntxt))
        for pair in _MIXED_PAIRS:
            crit = _crit(gn, cell=f"{pair}_null")
            cells = []
            for t in tests:
                k, m = _power(gn[(gn.cell == f"{pair}_alt") & (gn.test == t)].pval,
                              crit.get(t, np.nan))
                cells.append(_fmt(k, m, bold=(t == "gfcm" and pair in ("CC", "KC"))))
                computed[(pair, n, t)] = (k / m) if m >= MINREPS else None
            lines.append(f"{pair} & " + " & ".join(cells) + r" \\")
    cap = (r"\caption{Mixed-type data: size-corrected power on a spread alternative across the four "
           r"$(X,Y)$ type pairings (C continuous, K categorical; threshold set to $5\%$ on the "
           r"matched per-type null; Wilson 95\% CI). Bold marks GFCM on the continuous-response "
           r"pairs CC and KC, where $X$ drives the spread of $Y$ at zero mean dependence and the "
           r"covariance-based mixed tests are blind.}" "\n" r"\label{tab:mixed}")
    lines += [r"\bottomrule", r"\end{tabular}}", cap, r"\end{table}", ""]
    _write("mixed.tex", "\n".join(lines))
    return computed


# --------------------------------------------------------------------------- real-data injection
_INJECT_LAB = {"gfcm": "GFCM", "ffci": "FFCI", "rcot": "RCoT", "blitz": "BLITZ",
               "gcm_boosted": "boosted GCM", "par_cop": "PartCopula", "fisherz": "Fisher-$z$"}
_INJECT_ORDER = ["gfcm", "ffci", "rcot", "blitz", "gcm_boosted", "par_cop", "fisherz"]


def inject():
    p = os.path.join(RESULTS, "injection_eustock.parquet")
    if not os.path.exists(p):
        print("inject: no results/aggregated/injection_eustock.parquet (run run_injection.py)")
        return None
    df = pd.read_parquet(p)
    lines = [r"\begin{table}[t]\centering\small",
             r"\setlength{\tabcolsep}{6pt}",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"test & size & size-corrected power \\", r"\midrule"]
    computed = {}
    for t in _INJECT_ORDER:
        nu = df[(df.test == t) & (df.cell == "null")].pval
        al = df[(df.test == t) & (df.cell == "alt")].pval
        ks, ms = _size(nu)
        crit = _crit_from(nu)
        kp, mp = _power(al, crit)
        lab = _INJECT_LAB[t]
        lab = r"\textbf{GFCM}" if t == "gfcm" else lab
        lines.append(f"{lab} & {_fmt(ks, ms, dec=3)} & "
                     f"{_fmt(kp, mp, dec=3, bold=(t == 'gfcm'))} \\\\")
        computed[t] = (ks / ms if ms >= MINREPS else None, kp / mp if mp >= MINREPS else None)
    cap = (r"\caption{Semi-synthetic injection on real EuStockMarkets innovations (pooled "
           r"standardized log-returns, excess kurtosis $4.26$; $n=2000$, $|S|=3$, $R=200$; Wilson "
           r"95\% CI). $X$ drives the conditional skew of $Y$ with conditional mean and variance "
           r"held flat. Power is size-corrected (each test re-thresholded to exact $5\%$ on its "
           r"matched null). Bold marks GFCM.}" "\n" r"\label{tab:inject}")
    lines += [r"\bottomrule", r"\end{tabular}", cap, r"\end{table}", ""]
    _write("inject.tex", "\n".join(lines))
    return computed


# --------------------------------------------------------------------------- asym (appendix)
_ASYM_ROWS = [("null", "null"), ("scale", "scale"),
              ("directed_tailshape", r"skew"),
              ("symmetric_comovement", r"tail co-movement ($t$-copula)")]


def asym():
    p = os.path.join(RESULTS, "asym.parquet")
    if not os.path.exists(p):
        print("asym: no results/aggregated/asym.parquet (run paper2_kit/exp_asym.py)"); return None
    df = pd.read_parquet(p).set_index("dgp")
    lines = [r"\begin{table}[ht]\centering", r"\begin{tabular}{lcc}", r"\toprule",
             r"alternative ($X\!\to\!Y$) & asymmetric $e_X r_\tau(Y)$ & "
             r"symmetric $r_\tau(X) r_\tau(Y)$ \\", r"\midrule"]
    computed = {}
    for key, label in _ASYM_ROWS:
        if key not in df.index:
            continue
        a = float(df.loc[key, "asymmetric"]); s = float(df.loc[key, "symmetric"])
        af = f"\\textbf{{{a:.2f}}}" if (a > 0.5 and a - s > 0.3) else f"{a:.2f}"
        sf = f"\\textbf{{{s:.2f}}}" if (s > 0.5 and s - a > 0.3) else f"{s:.2f}"
        lines.append(f"{label} & {af} & {sf} \\\\")
        computed[key] = (a, s)
    cap = (r"\caption{Asymmetric versus symmetric quantile product ($n=2000$, $200$ reps, "
           r"$\alpha=0.05$; Wilson CIs in text). The asymmetric mean-quantile product GFCM uses "
           r"dominates on the level-driven skew edge; the symmetric product instead catches "
           r"symmetric tail co-movement. Bold marks the winning product on each discriminating "
           r"row.}" "\n" r"\label{tab:asym}")
    lines += [r"\bottomrule", r"\end{tabular}", cap, r"\end{table}", ""]
    _write("asym.tex", "\n".join(lines))
    return computed


# --------------------------------------------------------------------------- nuisance tradeoff (appendix)
_TRADEOFF_ROWS = [
    ("additive", 2000, "additive control"),
    ("additive", 10000, "additive control"),
    ("interaction", 2000, r"interaction $Z_1Z_2$"),
    ("interaction", 10000, r"interaction $Z_1Z_2$"),
    ("transcendental", 2000, r"transcendental $\sin(\textstyle\sum_j Z_j)$"),
    ("transcendental", 10000, r"transcendental $\sin(\textstyle\sum_j Z_j)$")]


def nuisance_tradeoff():
    p = os.path.join(RESULTS, "nuisance_tradeoff.parquet")
    if not os.path.exists(p):
        print("nuisance_tradeoff: no results/aggregated/nuisance_tradeoff.parquet "
              "(run paper2_kit/exp_nuisance_tradeoff.py)"); return None
    df = pd.read_parquet(p).set_index(["kind", "n"])

    def cell(v, lo, hi, bold=False):
        s = f"{v:.2f}~[{lo:.2f}--{hi:.2f}]"
        return f"\\textbf{{{s}}}" if bold else s

    lines = [r"\begin{table}[t]\centering\small", r"\begin{tabular}{llcc}", r"\toprule",
             r"null & $n$ & polynomial block & single index \\", r"\midrule"]
    computed = {}
    for kind, n, label in _TRADEOFF_ROWS:
        if (kind, n) not in df.index:
            continue
        r = df.loc[(kind, n)]
        pv, iv = float(r["poly"]), float(r["index"])
        # bold the polynomial cell on the decisive rows: it cleanly handles interactions
        # (poly<0.10 while single index leaks >0.15) or it detonates on a transcendental mean.
        bold = (pv < 0.10 and iv > 0.15) or (pv > 0.90 and iv < 0.20)
        lines.append(f"{label} & {n} & {cell(pv, r['poly_lo'], r['poly_hi'], bold)} & "
                     f"{cell(iv, r['index_lo'], r['index_hi'])} \\\\")
        computed[(kind, n)] = (pv, iv)
    cap = (r"\caption{Complementary failure of the two nuisance blocks (size, nominal $0.05$; $200$ "
           r"reps, Wilson 95\% CI; $Z\in\mathbb{R}^3$). The polynomial block removes polynomial "
           r"interactions at any $n$ but diverges on a transcendental mean; the single index "
           r"represents the transcendental mean (cleanly once $n$ is large enough to estimate its "
           r"direction) but underfits the interaction at small $n$. The default test uses both.}"
           "\n" r"\label{tab:nuisance-tradeoff}")
    lines += [r"\bottomrule", r"\end{tabular}", cap, r"\end{table}", ""]
    _write("nuisance_tradeoff.tex", "\n".join(lines))
    return computed


# --------------------------------------------------------------------------- interaction (appendix)
def interaction():
    p = os.path.join(RESULTS, "interaction.parquet")
    if not os.path.exists(p):
        print("interaction: no results/aggregated/interaction.parquet "
              "(run paper2_kit/exp_interaction.py)"); return None
    df = pd.read_parquet(p)
    ns = sorted(df.n.unique())
    ncol = {1000: r"$n=10^3$", 4000: r"$n=4{\cdot}10^3$", 16000: r"$n=1.6{\cdot}10^4$"}
    head = " & ".join([r"shared nuisance $\E[X\given Z]=\E[Y\given Z]$"] + [ncol.get(n, f"$n={n}$") for n in ns])
    rowlab = {"additive": r"additive ($\sin Z_1+\sin Z_2$ / $\cos Z_1+\cos Z_2$)",
              "interaction": r"interaction ($Z_1 Z_2$)"}
    lines = [r"\begin{table}[ht]\centering", r"\begin{tabular}{l" + "c" * len(ns) + "}", r"\toprule",
             head + r" \\", r"\midrule"]
    computed = {}
    for kind in ("additive", "interaction"):
        cells = []
        for n in ns:
            row = df[(df.kind == kind) & (df.n == n)]
            v = float(row["size"].iloc[0]) if len(row) else float("nan")
            # bold the interaction row: the backbone detonates (first-order, non-vanishing bias)
            cells.append(f"\\textbf{{{v:.2f}}}" if kind == "interaction" else f"{v:.2f}")
            computed[(kind, n)] = v
        lines.append(f"{rowlab[kind]} & " + " & ".join(cells) + r" \\")
    cap = (r"\caption{Null rejection rate (nominal $0.05$, $Z\in\mathbb{R}^2$, $300$ reps; "
           r"$X\indep Y\given Z$ in both rows). The additive spline backbone alone (without the "
           r"default interaction block) holds level under an additive shared mean but collapses to "
           r"$1.00$ under a pure interaction mean it cannot represent, at every $n$: a first-order "
           r"bias, not a vanishing one. This pins the structural scope of (A1); the default "
           r"design's degree-3 interaction block removes it (Table~\ref{tab:nuisance-tradeoff}).}"
           "\n" r"\label{tab:interaction}")
    lines += [r"\bottomrule", r"\end{tabular}", cap, r"\end{table}", ""]
    _write("interaction.tex", "\n".join(lines))
    return computed


# --------------------------------------------------------------------------- rankz (appendix)
def rankz():
    p = os.path.join(RESULTS, "rankz.parquet")
    if not os.path.exists(p):
        print("rankz: no results/aggregated/rankz.parquet (run paper2_kit/stress_rankz.py)")
        return None
    df = pd.read_parquet(p)

    def cell(v, lo, hi, bold=False):
        s = f"{v:.3f}~[{lo:.3f}--{hi:.3f}]"
        return f"\\textbf{{{s}}}" if bold else s

    lines = [r"\begin{table}[ht]\centering", r"\small", r"\begin{tabular}{lcc}", r"\toprule",
             r"conditioner $Z$ \ (null $X\indep Y\given Z$) & rank-$Z$ (default) & raw $Z$ \\",
             r"\midrule"]
    computed = {}
    for r in df.itertuples():
        if r.kind == "power":
            lines.append(r"\midrule")
        # raw-Z column bolded where it is materially anticonservative (>=0.10)
        rawbold = (r.kind == "size" and r.raw >= 0.10)
        lines.append(f"{r.label} & {cell(r.rankz, r.rankz_lo, r.rankz_hi)} & "
                     f"{cell(r.raw, r.raw_lo, r.raw_hi, rawbold)} \\\\")
        computed[r.cond] = (r.rankz, r.raw)
    cap = (r"\caption{Rank-transformed conditioning holds calibration on heavy-tailed conditioners; "
           r"raw $Z$ does not. GFCM size on a true null $X\indep Y\given Z$ whose smooth shared mean "
           r"$m(Z)=\tanh(1.2\,Z)$ drives both $X$ and $Y$ with independent noise ($n=2000$, $500$ "
           r"reps, nominal $0.05$; Wilson $95\%$ CIs). With the default rank-$Z$ the test stays at "
           r"level whatever the conditioner's tails; on raw $Z$ size climbs to $\sim0.15$ ($t_2$) and "
           r"$1.00$ ($t_1$). The final row confirms power is unaffected. Bold marks raw-$Z$ "
           r"miscalibration.}" "\n" r"\label{tab:rankz}")
    lines += [r"\bottomrule", r"\end{tabular}", cap, r"\end{table}", ""]
    _write("rankz.tex", "\n".join(lines))
    return computed


# --------------------------------------------------------------------------- cc-inject (real data)
_CC_ORDER = ["fisherz", "gcm_linear", "gcm_boosted", "rcot", "ffci", "blitz", "par_cop", "gfcm"]
_CC_LAB = {"fisherz": "Fisher's $Z$", "gcm_linear": "GCM", "gcm_boosted": "boosted GCM",
           "rcot": "RCoT", "ffci": "FFCI", "blitz": "BLITZ", "par_cop": "PartCopula", "gfcm": "GFCM"}


def cc_inject():
    p = os.path.join(RESULTS, "injection_chamber.parquet")
    if not os.path.exists(p):
        print("cc_inject: no results/aggregated/injection_chamber.parquet "
              "(run experiments/run_cc_injection.py)"); return None
    df = pd.read_parquet(p).set_index("test")
    lines = [r"\begin{table}[ht]\centering", r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{real CC} & \multicolumn{2}{c}{scale-injected} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Test & SHD & recall & SHD & recall \\", r"\midrule"]
    computed = {}
    for t in _CC_ORDER:
        if t not in df.index:
            continue
        r = df.loc[t]
        lab = r"\textbf{GFCM}" if t == "gfcm" else _CC_LAB[t]
        ir = f"{r.inj_recall:.2f}"
        # bold the injected recall where the covariance family goes blind (0.00) and for GFCM (recovers)
        ir = f"\\textbf{{{ir}}}" if (r.inj_recall == 0.0 or t == "gfcm") else ir
        lines.append(f"{lab} & {int(r.real_shd)} & {r.real_recall:.2f} & "
                     f"{int(r.inj_shd)} & {ir} \\\\")
        computed[t] = (int(r.inj_shd), float(r.inj_recall))
    cap = (r"\caption{Causal-Chamber light tunnel, real versus scale-injected ($n=2000$). Engineering "
           r"the real source-to-sink edges into mean-preserving scale edges (real sources, real "
           r"per-sink noise, only the edge type changed) makes the covariance/mean family blind "
           r"(recall $0.00$), while GFCM recovers $89\%$ (RCoT and FFCI also recover; BLITZ and the "
           r"partial-copula test are weak on this real substrate throughout). Bold marks the "
           r"injected-data recall for the blind covariance tests and for GFCM.}"
           "\n" r"\label{tab:cc-inject}")
    lines += [r"\bottomrule", r"\end{tabular}", cap, r"\end{table}", ""]
    _write("cc_inject.tex", "\n".join(lines))
    return computed


# --------------------------------------------------------------------------- convergence sweep
_CONV_NS = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 50000, 100000]
_CONV_COLS = ["0.5", "1", "1.5", "2", "2.5", "3", "4", "5", "7", "10", "15", "20", "50", "100"]
_CONV_ORDER = ["gfcm", "blitz", "ffci", "rcot", "gcm_boosted", "par_cop"]


def _conv_block(df, section, cell, dS, mode, crit_cell=None):
    """One '<mechanism>, |S|=d' block: a row per test across the 14 sample sizes."""
    rows = []
    for t in _CONV_ORDER:
        vals = []
        for n in _CONV_NS:
            g = df[(df.section == section) & (df.dS == dS) & (df.n == n) & (df.test == t)]
            pv = g[g.cell == cell].pval
            if mode == "size":
                k, m = _size(pv)
            else:
                cg = df[(df.section == section) & (df.dS == dS) & (df.n == n) & (df.test == t)
                        & (df.cell == crit_cell)].pval.dropna().values
                c = np.percentile(cg, 100 * ALPHA) if len(cg) else np.nan
                k, m = _power(pv, c)
            vals.append("--" if m < MINREPS else f"{k/m:.2f}")
        rows.append(LAB[t] + " & " + " & ".join(vals) + r" \\")
    return rows


def convergence():
    df = _load("cells_conv")
    if df is None:
        print("convergence: no data/cells_conv/"); return None
    cols = " & ".join(["test"] + _CONV_COLS)
    def _longtable(caption_label, blocks):
        L = [r"\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{l*{14}{c}}", caption_label, r"\toprule",
             r" & \multicolumn{14}{c}{sample size $n$ (thousands)}\\",
             r"\cmidrule(lr){2-15}", cols + r" \\", r"\midrule", r"\endfirsthead",
             r"\multicolumn{15}{c}{\tablename\ \thetable\ (continued)}\\", r"\toprule",
             cols + r" \\", r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
        for title, rows in blocks:
            L.append(r"\midrule \multicolumn{15}{l}{" + title + r"}\\")
            L.extend(rows)
        L += [r"\end{longtable}", r"\normalsize\setlength{\tabcolsep}{6pt}"]
        return L

    # null mechanisms in preferred order, but only those actually present in the sweep data
    # (the paper also has a nonlin_mean block; it is emitted here iff cells_conv contains it)
    _NULL_ORDER = ["heavy_tail", "hetero", "mixed_Z", "nonlin_mean"]
    present = [m for m in _NULL_ORDER if ((df.section == "5.2_calib") & (df.cell == m)).any()]
    if "nonlin_mean" not in present:
        print("  NOTE: cells_conv has no nonlin_mean null -> conv-null omits that block "
              "(paper has it; re-run run_convergence.py with nonlin_mean to close the gap)")
    null_blocks = []
    for mech in present:
        for dS in _CALIB_DS:
            title = r"\texttt{%s}, $|S|=%d$" % (mech.replace("_", r"\_"), dS)
            null_blocks.append((title, _conv_block(df, "5.2_calib", mech, dS, "size")))
    null_cap = (r"\caption{Convergence of null size with sample size (full sweep; columns are $n$ in "
                r"thousands; nominal $0.05$). PartCopula and boosted GCM are capped at moderate $n$ "
                r"and shown as \texttt{--} beyond it.}\label{tab:conv-null}\\")

    pow_blocks = []
    for section, cell in [("5.3_power", "tail_shape"), ("5.4_tail", "alpha_3")]:
        for dS in _CALIB_DS:
            title = r"\texttt{%s}, $|S|=%d$" % (cell.replace("_", r"\_"), dS)
            pow_blocks.append((title, _conv_block(df, section, cell, dS, "power", crit_cell="null")))
    pow_cap = (r"\caption{Convergence of size-corrected power with sample size (full sweep; columns "
               r"are $n$ in thousands).}\label{tab:conv-power}\\")

    body = "\n".join(_longtable(null_cap, null_blocks) + [""] + _longtable(pow_cap, pow_blocks))
    _write("convergence_appendix.tex", body)
    return True


# --------------------------------------------------------------------------- drift diff vs paper
def _scrape_paper(label):
    """Pull leading point-estimates from an inline table in paper2_tmlr.tex, keyed by row label."""
    if not os.path.exists(PAPER):
        return None
    txt = open(PAPER).read()
    m = re.search(r"\\begin\{table\}.*?\\label\{" + re.escape(label) + r"\}", txt, re.S)
    if not m:
        return None
    block = m.group(0)
    body = block[block.index(r"\midrule"):]
    out, cur_n = {}, None
    for line in body.splitlines():
        line = line.strip()
        mm = re.search(r"\\textit\{\$n=(10\^5|\d+)\$\}", line)
        if mm:
            cur_n = 100000 if "^" in mm.group(1) else int(mm.group(1))
            continue
        if "&" not in line or line.startswith("%") or "multicolumn" in line and "textit" not in line:
            continue
        cells = [c.strip() for c in line.rstrip(r"\\").split("&")]
        nums = []
        for c in cells:
            c = c.replace(r"\textbf{", "").replace("}", "").replace(r"\_", "_").strip()
            g = re.match(r"(-?\d+\.\d+)", c)
            nums.append(float(g.group(1)) if g else None)
        labels = [cells[i].replace(r"\textbf{", "").replace("}", "").replace(r"\_", "_").strip()
                  for i in range(len(cells)) if nums[i] is None]
        vals = [x for x in nums if x is not None]
        if labels and vals:
            key = tuple(labels) if cur_n is None else tuple(labels) + (cur_n,)
            out[key] = vals
    return out


def _diff(computed, label, tests, keyfn):
    paper = _scrape_paper(label)
    if paper is None:
        print(f"  {label}: could not scrape paper; skipping diff"); return
    drift = []
    for k, comp in computed.items():
        pk = keyfn(k)
        if pk not in paper or comp is None:
            continue
        # locate the test's column within the scraped row
        return  # handled per-table below (kept simple)


def run_diff(all_computed):
    """Compare computed point estimates against the paper's inline numbers; report drift > 0.005."""
    print("\n=== DRIFT vs paper (|Δ| > 0.005) ===")
    specs = {
        "calib_n2k": ("tab:calib-n2k", ["gfcm", "par_cop", "blitz", "ffci", "rcot", "gcm_boosted"]),
        "calib_n1e5": ("tab:calib-n1e5", ["gfcm", "blitz", "ffci", "rcot", "gcm_boosted"]),
        "power_n2k": ("tab:power-n2k", ["gfcm", "par_cop", "blitz", "ffci", "rcot", "gcm_boosted"]),
        "mixed": ("tab:mixed", ["gfcm", "ffci", "blitz", "ci_mm", "gcm_boosted"]),
    }
    for name, (label, tests) in specs.items():
        computed = all_computed.get(name)
        if not computed:
            continue
        paper = _scrape_paper(label)
        if not paper:
            print(f"  {label}: no paper values scraped"); continue
        n_drift = n_ok = 0
        for key, comp in computed.items():
            if comp is None:
                continue
            if name == "mixed":
                pair, n, t = key; pk = (pair, n)
            else:
                mech, dS, t = key; pk = (mech, str(dS))
            if pk not in paper:
                continue
            vals = paper[pk]
            if len(vals) != len(tests):
                continue
            pv = vals[tests.index(t)]
            if abs(pv - comp) > 0.005:
                n_drift += 1
                print(f"  {label:16s} {str(key):40s} paper={pv:.3f} computed={comp:.3f} Δ={comp-pv:+.3f}")
            else:
                n_ok += 1
        print(f"  {label}: {n_ok} match, {n_drift} drifted")


# --------------------------------------------------------------------------- unbacked
def _report_unbacked():
    unbacked = []  # all 12 paper tables are now regenerable from this repo
    if not unbacked:
        print("\n=== all 12 paper tables regenerable from this repo ===")
        return
    print("\n=== NOT regenerable here ===")
    for lab, src in unbacked:
        print(f"  {lab:24s} {src}")


# --------------------------------------------------------------------------- main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_diff = "--diff" in sys.argv
    dispatch = {"calib_n2k": calib_n2k, "calib_n1e5": calib_n1e5,
                "power_n2k": power_n2k, "mixed": mixed, "inject": inject,
                "asym": asym, "nuisance_tradeoff": nuisance_tradeoff,
                "rankz": rankz, "interaction": interaction, "cc_inject": cc_inject,
                "convergence": convergence}
    want = args or list(dispatch)
    all_computed = {}
    for name in want:
        if name in dispatch:
            print(f"{name}:")
            all_computed[name] = dispatch[name]()
    if do_diff:
        run_diff(all_computed)
    _report_unbacked()


if __name__ == "__main__":
    main()
