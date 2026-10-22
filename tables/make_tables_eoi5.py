"""Tables added for the TMLR revision (Reviewer Eoi5, requested change 3), from data_reviewers/eoi5/.

  combrule   tab:combrule   Cauchy (ACAT) vs Bonferroni vs joint sign-flip combination of the three
                            block p-values: size on every null cell at |S| <= 5, alpha = 0.05
                            (Wilson 95% CI); the 0.01 / 0.10 maxima are printed for the caption.
  misspec    tab:misspec    Level when assumption (A5) fails: nulls whose conditional skewness varies
                            with Z_1, against location-scale controls.
  pcext      tab:pcext      PC discovery at p = 20 (degree 2 and 4) and p = 50, uncapped conditioning
                            (requested change 4): skeleton SHD / precision / recall, CPDAG SHD, arrowhead
                            precision, end-to-end wall time. Replaces tab:p20. FFCI and RCoT at p = 50 come
                            from the capped run (flagged with a dagger). Prints the oracle floor and the
                            capped-vs-uncapped GFCM deltas for the text.
  envelope   tab:envelope   Operating envelope of the published method (requested change 4): size at alpha =
             fig:envelope   0.05 on seven null families at |S| = 1..20 and n = 1e3..1e5, per family (table) and
                            pooled / worst family / design-dimension bound (figure). Reads envelope_all.parquet.

Writes results/tables/<name>.tex (full float) and <name>_data.tex (tabular only, for \\input with the
caption kept in the paper), mirroring make_tables.py.

  python tables/make_tables_eoi5.py
"""
import glob
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "data_reviewers", "eoi5")
OUT = os.path.join(_ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.05

NULLS = {("calib", "heavy_tail"): "heavy tail", ("calib", "hetero"): "heteroscedastic",
         ("calib", "nonlin_mean"): "nonlinear mean", ("calib", "mixed_Z"): r"mixed $Z$",
         ("ladder", "null"): "ladder null"}
MISSPEC = {"ls_gauss": "location-scale, Gaussian noise", "ls_skew_fixed": "location-scale, skewed noise",
           "notls_skew_z": r"skewness varies with $Z_1$", "notls_skew_z8": r"skewness varies with $Z_1$, strong"}
RULES = [("acat3", "Cauchy (default)"), ("bonf3", "Bonferroni"), ("joint3", "joint sign-flip")]
PC_TESTS = [("gfcm", r"\textbf{GFCM}"), ("ffci", "FFCI"), ("blitz", "BLITZ"), ("rcot", "RCoT")]
PC_CELLS = [(20, 2, 2000), (20, 2, 10000), (20, 4, 2000), (20, 4, 10000), (50, 2, 2000), (50, 2, 10000)]


def _wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _fmt(k, m, dec=3):
    lo, hi = _wilson(k, m)
    return f"{k/m:.{dec}f}~[{lo:.{dec}f}--{hi:.{dec}f}]"


def _load():
    d = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(DATA, "eoi5_ci__*.parquet")))
    return d[d.test == "gfcm"].copy()


def _write(name, float_lines, tab_lines):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write("\n".join(float_lines) + "\n")
    with open(os.path.join(OUT, name + "_data.tex"), "w") as f:
        f.write("\n".join(tab_lines) + "\n")
    print(f"  wrote {name}.tex and {name}_data.tex")


def combrule(d):
    sub = d[(d.dS <= 5) & d.apply(lambda r: (r.family, r.cell) in NULLS, axis=1)]
    tab = [r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{llr" + "c" * len(RULES) + "}", r"\toprule",
           "null & $|S|$ & $n$ & " + " & ".join(lab for _, lab in RULES) + r" \\", r"\midrule"]
    maxima = {a: {r: 0.0 for r, _ in RULES} for a in (0.01, 0.05, 0.10)}
    for (fam, cell), lab in NULLS.items():
        for dS in (1, 3, 5):
            for n in (2000, 10000):
                g = sub[(sub.family == fam) & (sub.cell == cell) & (sub.dS == dS) & (sub.n == n)]
                if g.empty:
                    continue
                cells = []
                for col, _ in RULES:
                    cells.append(_fmt(int((g[col] < ALPHA).sum()), len(g)))
                    for a in maxima:
                        maxima[a][col] = max(maxima[a][col], float((g[col] < a).mean()))
                nlab = "$10^4$" if n == 10000 else str(n)
                tab.append(f"{lab} & {dS} & {nlab} & " + " & ".join(cells) + r" \\")
    tab += [r"\bottomrule", r"\end{tabular}}"]
    cap = (r"\caption{Combination of the three block $p$-values: size at $\alpha=0.05$ on every null "
           r"cell with $|S|\le5$ (500 replications; Wilson 95\% CI). The Cauchy rule is the default; "
           r"Bonferroni controls the level at every $\alpha$; the joint sign-flip calibrates the Cauchy "
           r"statistic with one Rademacher draw shared across blocks. Worst size over these cells at "
           + r"$\alpha=0.01$: Cauchy " + f"{maxima[0.01]['acat3']:.3f}" + ", Bonferroni "
           + f"{maxima[0.01]['bonf3']:.3f}" + r"; at $\alpha=0.10$: Cauchy " + f"{maxima[0.10]['acat3']:.3f}"
           + ", Bonferroni " + f"{maxima[0.10]['bonf3']:.3f}" + ".}" + "\n" + r"\label{tab:combrule}")
    flt = [r"\begin{table}[t]\centering\small"] + tab + [cap, r"\end{table}"]
    _write("combrule", flt, tab)
    print("  maxima:", {a: {r: round(v, 3) for r, v in m.items()} for a, m in maxima.items()})


def misspec(d):
    sub = d[d.family == "misspec"]
    tab = [r"\begin{tabular}{lrrcc}", r"\toprule",
           r"null & $|S|$ & $n$ & Cauchy (default) & Bonferroni \\", r"\midrule"]
    for cell, lab in MISSPEC.items():
        for dS in (1, 3):
            for n in (2000, 10000):
                g = sub[(sub.cell == cell) & (sub.dS == dS) & (sub.n == n)]
                if g.empty:
                    continue
                nlab = "$10^4$" if n == 10000 else str(n)
                tab.append(f"{lab} & {dS} & {nlab} & "
                           + _fmt(int((g.acat3 < ALPHA).sum()), len(g)) + " & "
                           + _fmt(int((g.bonf3 < ALPHA).sum()), len(g)) + r" \\")
    tab += [r"\bottomrule", r"\end{tabular}"]
    cap = (r"\caption{Level when assumption (A5) fails. $X\indep Y\given Z$ holds in every cell; "
           r"$Y=\sum_j Z_j+\eta$ with $\eta$ Gaussian, skew-normal of fixed shape (both location-scale), "
           r"or skew-normal whose shape is $3Z_1$ or $8Z_1$ (not location-scale). Size at $\alpha=0.05$, "
           r"500 replications, Wilson 95\% CI.}" + "\n" + r"\label{tab:misspec}")
    flt = [r"\begin{table}[t]\centering\small"] + tab + [cap, r"\end{table}"]
    _write("misspec", flt, tab)


def _mci(x, dec=2):
    m = x.mean(); h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:.{dec}f}~[{m-h:.{dec}f}--{m+h:.{dec}f}]"


def _load_pc():
    files = glob.glob(os.path.join(DATA, "**", "eoi5_pc__*.parquet"), recursive=True)   # p50comp/ subfolder too
    d = pd.concat(pd.read_parquet(f).assign(uncapped="capnone" in os.path.basename(f)) for f in files)
    if "error" in d:
        d = d[d["error"].isna()]
    return d


def pcext():
    d = _load_pc()
    tab = [r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{lrrlccccccrr}", r"\toprule",
           r"Test & $p$ & deg & $n$ & SHD & precision & recall & FP edges & CPDAG SHD & arrow prec. & queries & min \\", r"\midrule"]
    deltas = []; any_flag = [False]
    for p, deg, n in PC_CELLS:
        for key, lab in PC_TESTS:
            u = d[(d.p == p) & (d.deg == deg) & (d.n == n) & (d.test == key) & d.uncapped]
            flag = ""
            if u.empty:
                u = d[(d.p == p) & (d.deg == deg) & (d.n == n) & (d.test == key) & ~d.uncapped]
                flag = r"$^\dagger$"; any_flag[0] = True
            if u.empty:
                continue
            c = d[(d.p == p) & (d.deg == deg) & (d.n == n) & (d.test == key) & ~d.uncapped]
            if key == "gfcm" and not c.empty and not flag:
                deltas.append((p, deg, n, round(u.shd.mean() - c.shd.mean(), 2), len(c), round(u.max_S.mean(), 1),
                               round(u.n_queries.mean() / c.n_queries.mean(), 2), round(u.wall_s.mean() / c.wall_s.mean(), 2)))
            nlab = "$10^4$" if n == 10000 else str(n)
            tab.append(f"{lab}{flag} & {p} & {deg} & {nlab} & {_mci(u.shd)} & {_mci(u.precision)} & {_mci(u.recall)} & {_mci(u.fp_edges)} & "
                       f"{_mci(u.cpdag_shd)} & {_mci(u.arrow_precision)} & {u.n_queries.median():.0f} & {u.wall_s.median()/60:.0f} \\\\")
        tab.append(r"\midrule")
    tab[-1] = r"\bottomrule"; tab += [r"\end{tabular}}"]
    cap = (r"\caption{PC discovery at $p=20$ and $p=50$ with no cap on the conditioning set. Random typed DAGs "
           r"of expected degree $2$ (and $4$ at $p=20$), $100$ replicates at $p=20$ and $50$ at $p=50$, normal "
           r"$95\%$ intervals in brackets. Skeleton SHD, precision, recall, and false positive edge count as in "
           r"Figure~\ref{fig:pc}; CPDAG SHD and arrowhead precision score the oriented output against the true "
           r"CPDAG; the last two columns are the median number of CI queries and the median end-to-end wall time "
           r"of one discovery run in minutes."
           + (r" $^\dagger$Run with the $|S|\le3$ cap only." if any_flag[0] else "") + "}" + "\n" + r"\label{tab:pcext}")
    flt = [r"\begin{table}[t]\centering\small", cap] + tab + [r"\end{table}"]
    _write("pcext", flt, tab)
    print("  GFCM uncapped minus capped SHD (p, deg, n, delta, capped reps, mean max|S|, query ratio, time ratio):")
    for t in deltas:
        print("   ", t)
    o = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(DATA, "eoi5_oracle__*.parquet")))
    print("  oracle floor (mean SHD / cpdag SHD / queries):")
    print(o.groupby(["p", "edge_prob", "cap"])[["shd", "cpdag_shd", "n_queries", "max_S"]].mean().round(2).to_string())
    print("  uncapped GFCM: max |S| reached, mean and max over runs:")
    g = d[d.uncapped & (d.test == "gfcm")]
    print(g.groupby(["p", "deg", "n"]).max_S.agg(["mean", "max"]).round(1).to_string())


# The four calibration nulls of Table 1 / Table 2, extended in |S| and n. The power ladder's null
# belongs to tab:deeppower (where it size-corrects the power rows), and the two interdepth families
# were built for an abandoned probe; neither is a calibration null of this paper.
ENV_CELLS = [("heavy_tail", "heavy tail"), ("hetero", "heteroscedastic"), ("nonlin_mean", "nonlinear mean"),
             ("mixed_Z", r"mixed $Z$")]
ENV_DS = [1, 2, 3, 5, 8, 10, 15, 20]
ENV_NS = [1000, 2000, 5000, 10000, 20000, 50000, 100000]


def _K(n):
    return int(min(150, max(25, round(10.0 * n ** 0.2))))


def _bound(n, dS, folds=5):
    from scipy.stats import norm
    P = dS * (_K(n) + 3) + 1.5 * dS * (dS - 1); r = P / (n * (folds - 1) / folds)
    return 2 * (1 - norm.cdf(1.96 * np.sqrt(1 - r))) if r < 1 else np.nan


def _nlab(n):
    return {1000: "$10^3$", 2000: "$2\\times10^3$", 5000: "$5\\times10^3$", 10000: "$10^4$",
            20000: "$2\\times10^4$", 50000: "$5\\times10^4$", 100000: "$10^5$"}[n]


def envelope():
    E = pd.read_parquet(os.path.join(DATA, "envelope_all.parquet"))
    E = E[E.cell.isin([k for k, _ in ENV_CELLS])]   # calibration nulls ONLY; this table is scoped to them
    E["rej"] = (E.pval < ALPHA).astype(float)
    S = E.groupby(["cell", "n", "dS"]).rej.mean()
    reps = E.groupby(["cell", "n", "dS"]).size()
    P = (E[E.cell.isin([k for k, _ in ENV_CELLS])].groupby(["n", "dS"]).rej
         .agg(["sum", "count"]).T.to_dict())                            # pooled over the four only
    # ---- tables: one block per null family, rows n, columns |S|, Wilson intervals in every cell;
    # split in two floats (|S| <= 5 and |S| >= 8) so the bracketed cells fit the page width
    def block(ds, label, part):
        tab = [r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%",
               r"\begin{tabular}{lr" + "c" * len(ds) + "}", r"\toprule",
               r"null family & $n$ & " + " & ".join(f"$|S|={d}$" for d in ds) + r" \\", r"\midrule"]
        for key, lab in (ENV_CELLS + [(None, r"\emph{all four pooled}")]):
            for i, n in enumerate(ENV_NS):
                cells = []
                for d in ds:
                    if key is None:
                        m = int(P[(n, d)]["count"]); k = int(P[(n, d)]["sum"])
                    elif (key, n, d) in S.index:
                        m = int(reps[(key, n, d)]); k = int(round(S[(key, n, d)] * m))
                    else:
                        cells.append("--"); continue
                    txt = _fmt(k, m, 3); lo, hi = _wilson(k, m)
                    cells.append((r"\textbf{" + txt + "}") if lo > ALPHA else txt)
                row = (lab if i == 0 else "") + " & " + _nlab(n) + " & " + " & ".join(cells) + r" \\"
                tab.append(row if key is not None else r"\emph{" + row.replace(r"\emph{all four pooled}", "all four pooled") + "}" if False else row)
            tab.append(r"\midrule")
        tab[-1] = r"\bottomrule"; tab += [r"\end{tabular}}"]
        cap = (r"\caption{Operating range of the test as published, part " + part + r": size at $\alpha=0.05$ on "
               r"the four calibration nulls of Table~\ref{tab:calib-n2k} ($X\indep Y\given Z$ in every cell) at conditioning sizes "
               + ", ".join(f"$|S|={d}$" for d in ds) + r" and $n=10^3$ to $10^5$, Wilson 95\% intervals in "
               r"brackets. Replications per cell: 2000 for $n\le5\times10^3$, 1000 for $n=10^4$ and "
               r"$2\times10^4$, 500 for $n\ge5\times10^4$ and four times that in the pooled block. Entries whose "
               r"interval lies above $0.05$ in bold, so an entry in roman type is a size not significantly "
               r"above $0.05$.}" + "\n"
               + r"\label{" + label + "}")
        return [r"\begin{table}[p]\centering\scriptsize", cap] + tab + [r"\end{table}"], tab
    fa, ta = block(ENV_DS[:4], "tab:envelope", "1 of 2")
    fb, tb = block(ENV_DS[4:], "tab:envelopeb", "2 of 2")
    _write("envelope", fa + [""] + fb, ta + [""] + tb)
    # ---- figure: pooled over the five calibration nulls / worst family / bound
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    five = [k for k, _ in ENV_CELLS]
    E = E[E.cell.isin(five)]                                   # figure: calibration nulls only
    pooled = E[E.cell.isin(five)].groupby(["n", "dS"]).rej.mean().unstack("dS").reindex(index=ENV_NS, columns=ENV_DS)
    worst = S.groupby(["n", "dS"]).max().unstack("dS").reindex(index=ENV_NS, columns=ENV_DS)
    bound = pd.DataFrame([[_bound(n, d) for d in ENV_DS] for n in ENV_NS], index=ENV_NS, columns=ENV_DS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, M, title in zip(axes, (pooled, worst, bound),
                            ("size, four calibration nulls pooled", "size, worst of the four",
                             "design-dimension bound $2(1-\\Phi(1.96\\sqrt{1-P/n_{\\rm train}}))$")):
        im = ax.imshow(M.values, cmap="Reds", vmin=0.05, vmax=0.35, aspect="auto", origin="lower")
        for i in range(len(ENV_NS)):
            for j in range(len(ENV_DS)):
                v = M.values[i, j]
                ax.text(j, i, "--" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (not np.isnan(v) and v > 0.2) else "black")
        ax.set_xticks(range(len(ENV_DS))); ax.set_xticklabels(ENV_DS); ax.set_xlabel("$|S|$")
        ax.set_title(title, fontsize=9)
    axes[0].set_yticks(range(len(ENV_NS))); axes[0].set_yticklabels(["1e3", "2e3", "5e3", "1e4", "2e4", "5e4", "1e5"])
    axes[0].set_ylabel("$n$")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="size at $\\alpha=0.05$")
    os.makedirs(os.path.join(_ROOT, "results", "figures"), exist_ok=True)
    q = os.path.join(_ROOT, "results", "figures", "fig_envelope.pdf"); fig.savefig(q, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  wrote fig_envelope.pdf")
    print("  pooled five nulls:\n" + pooled.round(3).to_string())
    print("  worst family:\n" + worst.round(3).to_string())
    print("  bound:\n" + bound.round(3).to_string())
    print("  deepest |S| with pooled size <= 0.07 per n:",
          {n: max([d for d in ENV_DS if pooled.loc[n, d] <= 0.07] or [0]) for n in ENV_NS})
    print("  deepest |S| with worst family <= 0.10 per n:",
          {n: max([d for d in ENV_DS if worst.loc[n, d] <= 0.10] or [0]) for n in ENV_NS})


DEEP_TESTS = [("acat3", r"\textbf{GFCM}"), ("blitz", "BLITZ"), ("rcot", "RCoT"), ("ffci", "FFCI")]
DEEP_ALTS = [("linear_mean", "linear mean"), ("nonmonotone_z2", "non-monotone mean"), ("scale", "scale"),
             ("tail_shape", "tail shape")]


def deeppower():
    """tab:deeppower: size-corrected power at |S| = 10 and 20 (ladder family, requested change 4), with the
    raw null size of each test in the first row of each block; threshold = alpha-quantile of the test's own
    p-values on the ladder null at the same (|S|, n)."""
    files = glob.glob(os.path.join(DATA, "eoi5_ci__*.parquet"))
    d = pd.concat(pd.read_parquet(f) for f in files); d = d[d.family == "ladder"]
    if "error" in d:
        d = d[d["error"].isna()]
    parts = [d[d.test == "gfcm"].assign(arm="acat3", p=lambda g: g.acat3)]
    o = d[d.test != "gfcm"]; parts.append(o.assign(arm=o.test, p=o.pval))
    L = pd.concat(parts)[["cell", "dS", "n", "arm", "p"]]
    tab = [r"\begin{tabular}{llr" + "c" * len(DEEP_TESTS) + "}", r"\toprule",
           r"alternative & $|S|$ & $n$ & " + " & ".join(lab for _, lab in DEEP_TESTS) + r" \\", r"\midrule"]
    for dS in (10, 20):
        for n in (2000, 10000):
            nlab = "$10^4$" if n == 10000 else str(n)
            row = []
            for arm, _ in DEEP_TESTS:
                nul = L[(L.cell == "null") & (L.dS == dS) & (L.n == n) & (L.arm == arm)].p
                if len(nul) == 0:
                    row.append("--"); continue
                k = int((nul < ALPHA).sum()); m = len(nul); lo, hi = _wilson(k, m); txt = _fmt(k, m, 2)
                row.append((r"\textbf{" + txt + "}") if lo > ALPHA else txt)
            tab.append(r"\emph{null size} & " + f"{dS} & {nlab} & " + " & ".join(row) + r" \\")
            for cell, lab in DEEP_ALTS:
                row = []
                for arm, _ in DEEP_TESTS:
                    nul = L[(L.cell == "null") & (L.dS == dS) & (L.n == n) & (L.arm == arm)].p
                    alt = L[(L.cell == cell) & (L.dS == dS) & (L.n == n) & (L.arm == arm)].p
                    if len(nul) == 0 or len(alt) == 0:
                        row.append("--"); continue
                    thr = np.quantile(nul, ALPHA); k = int((alt <= thr).sum()); m = len(alt)
                    row.append(_fmt(k, m, 2))
                tab.append(f"{lab} & {dS} & {nlab} & " + " & ".join(row) + r" \\")
            tab.append(r"\midrule")
    tab[-1] = r"\bottomrule"; tab.append(r"\end{tabular}")
    cap = (r"\caption{Power beyond $|S|=5$: size-corrected power on the alternative ladder of "
           r"Section~\ref{sec:ablation} at $|S|=10$ and $20$ ($500$ replications, Wilson 95\% CI), with each "
           r"test re-thresholded to exact $5\%$ on its own null at the same $(|S|,n)$; the italic rows give the "
           r"raw size on that null with its Wilson interval, bold where the interval lies above $0.05$. This null belongs to the power family, whose conditioners drive both $X$ and $Y$; it is not one of the calibration nulls of Tables~\ref{tab:envelope} and \ref{tab:envelopeb}. Size correction "
           r"removes the inflation but not its cost: where a test's null $p$-values are degenerate, the corrected "
           r"threshold leaves it no power.}" + "\n" + r"\label{tab:deeppower}")
    flt = [r"\begin{table}[t]\centering\small", cap] + tab + [r"\end{table}"]
    _write("deeppower", flt, tab)


if __name__ == "__main__":
    d = _load()
    combrule(d)
    misspec(d)
    pcext()
    envelope()
    deeppower()


CAP_CELLS = [(20, 2, 2000), (20, 2, 10000), (20, 4, 2000), (20, 4, 10000), (50, 2, 2000), (50, 2, 10000)]
CAP_EP = {(20, 2): 0.105263, (20, 4): 0.210526, (50, 2): 0.040816}


def captable():
    """tab:cap: what the |S| <= 3 cap of the submitted experiments costs. GFCM capped vs uncapped on the
    same cells, the depth the uncapped search actually reaches, and the cost of the same cap to a
    d-separation oracle (its excess edges are what a test with full power would pay). Wall time is
    deliberately excluded: the capped and uncapped batches ran under different machine load."""
    files = glob.glob(os.path.join(DATA, "**", "eoi5_pc__*.parquet"), recursive=True)
    d = pd.concat(pd.read_parquet(f).assign(uncapped="capnone" in os.path.basename(f)) for f in files)
    if "error" in d:
        d = d[d["error"].isna()]
    d = d[d.test == "gfcm"]
    o = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(DATA, "eoi5_oracle__*.parquet")))
    tab = [r"\begin{tabular}{rrlcccccc}", r"\toprule",
           r"& & & \multicolumn{2}{c}{skeleton SHD} & \multicolumn{2}{c}{CI queries} & "
           r"$|S|$ reached & oracle SHD \\",
           r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"$p$ & deg & $n$ & cap $3$ & uncapped & cap $3$ & uncapped & mean / max & at cap $3$ \\",
           r"\midrule"]
    for p, deg, n in CAP_CELLS:
        c = d[(d.p == p) & (d.deg == deg) & (d.n == n) & ~d.uncapped]
        u = d[(d.p == p) & (d.deg == deg) & (d.n == n) & d.uncapped]
        oc = o[(o.p == p) & (np.isclose(o.edge_prob, CAP_EP[(p, deg)])) & (o.cap == 3)]
        nlab = "$10^4$" if n == 10000 else str(n)
        tab.append(f"{p} & {deg} & {nlab} & {_mci(c.shd)} & {_mci(u.shd)} & {c.n_queries.mean():.0f} & "
                   f"{u.n_queries.mean():.0f} & {u.max_S.mean():.1f} / {u.max_S.max():.0f} & "
                   f"{oc.shd.mean():.2f}" + r" \\")
    tab += [r"\bottomrule", r"\end{tabular}"]
    cap = (r"\caption{What the $|S|\le3$ cap costs. The $8$-node panel of Figure~\ref{fig:pc} runs "
           r"capped; the $p=20$ and $p=50$ runs of Table~\ref{tab:pcext} do not. Here GFCM is run on "
           r"those larger graphs both ways ($100$ replicates at $p=20$, $50$ at $p=50$; normal $95\%$ "
           r"intervals), with the deepest separating set the uncapped search reaches and the excess "
           r"edges the same cap leaves in the output of a $d$-separation oracle, which is what a test "
           r"with full power would pay for it. The uncapped oracle recovers every graph exactly. Wall "
           r"time is not compared because the two batches ran under different machine load.}"
           + "\n" + r"\label{tab:cap}")
    flt = [r"\begin{table}[t]\centering\small", cap] + tab + [r"\end{table}"]
    _write("captable", flt, tab)


P2K_TESTS = [("gfcm", r"\textbf{GFCM}"), ("par_cop", "PartCopula"), ("blitz", "BLITZ"),
             ("ffci", "FFCI"), ("rcot", "RCoT"), ("gcm_boosted", "GCM-boost")]
P2K_MECH = [("linear_mean", r"linear\_mean"), ("nonmonotone_z2", r"nonmonotone\_z2"),
            ("scale", "scale"), ("tail_shape", r"tail\_shape")]


def power_n2k_rev():
    """tab:power-n2k, regenerated with the raw size of the matched null shown above each block, so the
    size correction each test needed is visible. Same data and same size-corrected power as the
    submitted table; only the italic null rows are new."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mt", os.path.join(_HERE, "make_tables.py"))
    mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
    df = mt._load("cells"); sub = df[(df.section == "5.3_power") & (df.n == 2000)]
    tab = [r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{ll" + "c" * len(P2K_TESTS) + "}", r"\toprule",
           " & ".join(["mechanism", "$|S|$"] + [lab for _, lab in P2K_TESTS]) + r" \\", r"\midrule"]
    for dS in (1, 3, 5):
        g = sub[sub.dS == dS]
        crit = mt._crit(g)
        cells = []
        for t, _ in P2K_TESTS:
            pv = g[(g.cell == "null") & (g.test == t)].pval.dropna().values
            if not len(pv):
                cells.append("--"); continue
            k, m = int((pv < ALPHA).sum()), len(pv); lo, _hi = _wilson(k, m); txt = _fmt(k, m, 3)
            cells.append((r"\textbf{" + txt + "}") if lo > ALPHA else txt)
        tab.append(r"\emph{null size} & " + str(dS) + " & " + " & ".join(cells) + r" \\")
        for mech, mlab in P2K_MECH:
            cells = []
            for t, _ in P2K_TESTS:
                k, m = mt._power(g[(g.cell == mech) & (g.test == t)].pval, crit.get(t, np.nan))
                cells.append(mt._fmt(k, m, bold=(t == "gfcm" and mech == "tail_shape")))
            tab.append(f"{mlab} & {dS} & " + " & ".join(cells) + r" \\")
        tab.append(r"\midrule")
    tab[-1] = r"\bottomrule"; tab += [r"\end{tabular}}"]
    cap = (r"\caption{Power across the alternative ladder, $n=2000$ (size-corrected power: each test's "
           r"threshold set to exact $5\%$ size on the matched null; Wilson 95\% CI). \rev{The italic rows "
           r"give the raw size of each test on that null at $\alpha=0.05$, bold where the interval lies "
           r"above $0.05$, so the correction each test needed is visible. GFCM's null rejection rate on "
           r"this family reaches $0.13$ and $0.15$ at $|S|=3$ and $5$, against $0.05$ to $0.09$ for the "
           r"others, because the family's $X$ has conditional mean $\sin(\sum_j Z_j)$, a ridge function "
           r"that no additive nuisance can represent, so assumption (A1) fails for it; on the four "
           r"calibration nulls of Table~\ref{tab:calib-n2k}, which satisfy (A1), GFCM is at $0.04$ to "
           r"$0.10$ at the same depths.} Bold marks GFCM on the skew alternative, where it alone keeps power as $|S|$ grows.}"
           + "\n" + r"\label{tab:power-n2k}")
    flt = [r"\begin{table}[t]\centering\small", cap] + tab + [r"\end{table}"]
    _write("power_n2k_rev", flt, tab)


MIX_TESTS = [("gfcm", r"\textbf{GFCM}"), ("ffci", "FFCI"), ("blitz", "BLITZ"),
             ("ci_mm", "ci-mm"), ("gcm_boosted", "GCM-boost")]
MIX_PAIRS = ["CC", "CK", "KC", "KK"]


def mixed_rev():
    """tab:mixed, regenerated with the raw size of each matched per-type null shown beneath its power
    row, so FFCI's failure is visible in the table rather than only asserted in the prose. Same data and
    same size-corrected power as the submitted table; only the italic null rows are new."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mt", os.path.join(_HERE, "make_tables.py"))
    mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
    sub = mt._load("cells"); sub = sub[sub.section == "5.5_mixed"]
    tab = [r"\setlength{\tabcolsep}{5pt}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{ll" + "c" * len(MIX_TESTS) + "}", r"\toprule",
           " & ".join(["type pair", "row"] + [lab for _, lab in MIX_TESTS]) + r" \\", r"\midrule"]
    for i, n in enumerate(sorted(sub.n.unique())):
        gn = sub[sub.n == n]
        if i:
            tab.append(r"\midrule")
        ntxt = r"$n=10^5$" if n == 100000 else f"$n={n}$"
        tab.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(MIX_TESTS) + 2, ntxt))
        for pair in MIX_PAIRS:
            crit = mt._crit(gn, cell=f"{pair}_null")
            cells, nulls = [], []
            for t, _ in MIX_TESTS:
                k, m = mt._power(gn[(gn.cell == f"{pair}_alt") & (gn.test == t)].pval, crit.get(t, np.nan))
                cells.append(mt._fmt(k, m))   # no bold: BLITZ ties GFCM in every power cell
                pv = gn[(gn.cell == f"{pair}_null") & (gn.test == t)].pval.dropna().values
                if len(pv):
                    kk, mm = int((pv < ALPHA).sum()), len(pv); lo, _ = _wilson(kk, mm)
                    txt = _fmt(kk, mm, 3)
                    nulls.append((r"\textbf{" + txt + "}") if lo > ALPHA else txt)
                else:
                    nulls.append("--")
            tab.append(f"{pair} & " + r"\emph{null size} & " + " & ".join(nulls) + r" \\")
            tab.append(f"{pair} & alternative & " + " & ".join(cells) + r" \\")
    tab += [r"\bottomrule", r"\end{tabular}}"]
    cap = (r"\caption{Mixed-type data: size-corrected power across the four $(X,Y)$ type pairings "
           r"(C continuous, K categorical). \rev{The alternative is matched to the response type, $X$ "
           r"driving $\Var(Y\given Z)$ at unchanged conditional mean for continuous $Y$ (CC, KC) and $X$ "
           r"shifting the class probabilities for categorical $Y$ (CK, KK).} Each test is re-thresholded "
           r"to exact $5\%$ on its own matched null, whose raw size at $\alpha=0.05$ is the italic row "
           r"above it, bold where its Wilson 95\% interval lies above $0.05$.}"
           + "\n" + r"\label{tab:mixed}")
    flt = [r"\begin{table}[t]\centering\small", cap] + tab + [r"\end{table}"]
    _write("mixed_rev", flt, tab)


LADD_TESTS = [("gfcm", r"\textbf{GFCM}"), ("blitz", "BLITZ"), ("rcot", "RCoT"), ("ffci", "FFCI")]
LADD_ALTS = [("linear_mean", "linear mean"), ("nonmonotone_z2", "non-monotone mean"), ("scale", "scale"),
             ("tail_shape", "tail shape")]
LADD_GRID = [(2000, (1, 3, 5, 8, 10, 15, 20)), (10000, (1, 3, 5, 8, 10, 15, 20)),
             (100000, (1, 3, 5, 8, 10, 15, 20))]   # n=5e4 was run at |S|=8,15 only; left out of the
                                                   # tables for a uniform grid, quoted in the caption
LADD_MIN_DS = 1   # show every cell the run covers; n=5e4 was run at |S|=8 and 15 only


def ladderadd():
    """tab:deeppower / tab:deeppowerb, on the additive-mean ladder (dgp.power_ladder_additive): the
    same four alternatives as Table 3 on a null whose X mean is additive, so (A1) holds. Raw null
    size (italic, bold if its interval lies above 0.05) above each block, size-corrected power
    beneath. Split in two floats on the same depth boundary as tab:envelope / tab:envelopeb, so a
    reader moving between them sees the same structure; part 1 is |S| <= 5, part 2 is |S| >= 8.
    Reads data_reviewers/eoi5/ladderadd_all.parquet (folded on the server)."""
    d = pd.read_parquet(os.path.join(DATA, "ladderadd_all.parquet"))
    if "error" in d:
        d = d[d["error"].isna()]
    NLAB = {2000: "2000", 10000: "$10^4$", 50000: r"$5\times10^4$", 100000: "$10^5$"}

    def block(dss, label, part, extra):
        tab = [r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%",
               r"\begin{tabular}{llr" + "c" * len(LADD_TESTS) + "}", r"\toprule",
               r"alternative & $|S|$ & $n$ & " + " & ".join(l for _, l in LADD_TESTS) + r" \\",
               r"\midrule"]
        first = True
        for n, grid in LADD_GRID:
            for dS in grid:
                if dS not in dss:
                    continue
                g = d[(d.n == n) & (d.dS == dS)]
                if g.empty:
                    continue
                if not first:
                    tab.append(r"\midrule")
                first = False
                row = []
                for arm, _ in LADD_TESTS:
                    nul = g[(g.cell == "null") & (g.test == arm)].pval.dropna()
                    if len(nul) == 0:
                        row.append("--"); continue
                    k = int((nul < ALPHA).sum()); m = len(nul); lo, _ = _wilson(k, m)
                    txt = _fmt(k, m, 3)
                    row.append((r"\textbf{" + txt + "}") if lo > ALPHA else txt)
                tab.append(r"\emph{null size} & " + f"{dS} & {NLAB[n]} & " + " & ".join(row) + r" \\")
                for cell, lab in LADD_ALTS:
                    row = []
                    for arm, _ in LADD_TESTS:
                        nul = g[(g.cell == "null") & (g.test == arm)].pval.dropna()
                        alt = g[(g.cell == cell) & (g.test == arm)].pval.dropna()
                        if len(nul) == 0 or len(alt) == 0:
                            row.append("--"); continue
                        thr = np.quantile(nul, ALPHA)
                        k = int((alt <= thr).sum()); m = len(alt)
                        _, hi = _wilson(k, m); txt = _fmt(k, m, 2)
                        row.append((r"\textbf{" + txt + "}") if hi < 0.80 else txt)
                    tab.append(f"{lab} & {dS} & {NLAB[n]} & " + " & ".join(row) + r" \\")
        tab += [r"\bottomrule", r"\end{tabular}}"]
        cap = (r"\caption{Calibration and power on an additive-mean null the validity theory covers, part "
               + part + r": the four alternatives of Table~\ref{tab:power-n2k} with the conditional mean of "
               r"$X$ made additive, $\sum_j\sin(1.5Z_j)/\sqrt{|S|}$ in place of $\sin(\sum_j Z_j)$, so that "
               r"assumption (A1) holds for the $X$ nuisance ($500$ replications, Wilson 95\% CI). The italic "
               r"rows give the raw size of each test on the matched null at $\alpha=0.05$; the rows "
               r"beneath give size-corrected power, each test re-thresholded to exact $5\%$ on that "
               r"null. Bold marks a cell that fails, a size whose interval lies above $0.05$ or a power "
               r"whose interval lies below $0.80$. "
               r"alternatives on the original non-additive mean, so no cell here repeats one there. " + extra + "}"
               + "\n" + r"\label{" + label + "}")
        return [r"\begin{table}[p]\centering\small", cap] + tab + [r"\end{table}"], tab

    extra_a = (r"GFCM's interval covers $0.05$ at $|S|\le3$ for every sample size, and at $|S|=5$ only "
               r"at $n=10^5$. Its tail power is $1.00$ from $n=10^4$ and, at $n=2000$, $0.98$, $0.83$ and "
               r"$0.64$ against $0.16$, $0.17$ and $0.14$ for BLITZ. RCoT and FFCI lose their level from "
               r"$|S|=3$ once $n\ge10^4$.")
    extra_b = (r"At $n=10^5$ GFCM's interval covers $0.05$ at $|S|=8$ but not at $10$, with tail power "
               r"$1.00$ at both; at $n=10^4$ it is inflated, at $0.088$ and $0.094$, but holds the tail edge "
               r"at $1.00$ against BLITZ's $0.55$ and $0.48$; at $n=2000$ it is outside the operating range "
               r"of Section~\ref{sec:nuisance} and size correction leaves it $0.26$ and $0.08$, where BLITZ "
               r"holds its level throughout. RCoT and FFCI are inflated in almost every cell here.")
    extra_c = (r"At $n=10^5$ GFCM's interval covers $0.05$ at both depths with tail power $1.00$, while "
               r"BLITZ reaches $0.210$ and $0.564$; GFCM is the only test calibrated at $|S|=20$. The "
               r"inflation at smaller $n$ is a finite-sample effect. At $|S|=15$ GFCM's size falls from "
               r"$0.720$ at $n=2000$ to $0.130$, then $0.072$ at $n=5\times10^4$ (run at $|S|=8$ and $15$ "
               r"only, not tabulated), then $0.044$ at $n=10^5$.")
    fa, ta = block((1, 3, 5), "tab:deeppower", "1 of 3", extra_a)
    fb, tb = block((8, 10), "tab:deeppowerb", "2 of 3", extra_b)
    fc, tc = block((15, 20), "tab:deeppowerc", "3 of 3", extra_c)
    _write("ladderadd", fa + [""] + fb + [""] + fc, ta + [""] + tb + [""] + tc)
