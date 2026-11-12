"""Aggregate the T2 (null) and T3 (power) grids written by run_grid.py into one parquet and the LaTeX tables.

Usage: python experiments/aggregate_grid.py DIR [--out results/aggregated] [--alpha 0.05]
Writes grid_all.parquet (one row per test x table x cell x |S| x n: reps, rejections, size or power, Wilson interval,
status counts, median wall time) and tables/grid_null.tex, tables/grid_power.tex. Power rows carry the raw size on the
matched null of the same (|S|, n) and the size-corrected power (threshold = the empirical 5 percent quantile of the
matched null's p-values, as in the submitted tables).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TESTS = ["gfcm", "blitz", "rcot", "ffci", "gcm_boosted", "par_cop"]
LABEL = {"gfcm": r"\textbf{GFCM}", "blitz": "BLITZ", "rcot": "RCoT", "ffci": "FFCI", "gcm_boosted": "GCM-boost", "par_cop": "PartCopula"}


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "grid_*.parquet")))
    if not fs: sys.exit(f"no grid parquets in {d}")
    return pd.concat((pd.read_parquet(f) for f in fs), ignore_index=True)


def summarise(D, alpha):
    rows = []
    nulls = {}                                                  # (dS, n, test) -> p-values of the matched additive null (power table)
    # workers that started before the status fix labelled every NaN "outside domain"; only GFCM refuses by design
    D.loc[(D.test != "gfcm") & (D.status == "outside domain"), "status"] = "nan"
    for (tb, cell, dS, n, test), g in D.groupby(["table", "cell", "dS", "n", "test"]):
        ok = g[g.status == "ok"]; reps = len(g); k = int((ok.pval <= alpha).sum()); m = len(ok)
        lo, hi = wilson(k, m)
        st = g.status.value_counts().to_dict()
        rows.append(dict(table=tb, cell=cell, dS=dS, n=n, test=test, reps=reps, ok=m, rej=k, rate=(k / m if m else np.nan), lo=lo, hi=hi,
                         outside=st.get("outside domain", 0), not_run=st.get("not run", 0), nan=st.get("nan", 0),
                         error=sum(v for s_, v in st.items() if str(s_).startswith("error")), wall_med=float(ok.wall_s.median()) if m else np.nan))
        if tb == "power" and cell == "null": nulls[(dS, n, test)] = ok.pval.to_numpy()
    S = pd.DataFrame(rows)
    # size-corrected power: threshold at the empirical alpha-quantile of the matched null
    sc = []
    for i, r in S.iterrows():
        if r.table != "power" or r.cell == "null": sc.append(np.nan); continue
        pn = nulls.get((r.dS, r.n, r.test))
        g = D[(D.table == "power") & (D.cell == r.cell) & (D.dS == r.dS) & (D.n == r.n) & (D.test == r.test) & (D.status == "ok")]
        if pn is None or len(pn) < 20 or len(g) == 0: sc.append(np.nan); continue
        thr = np.quantile(pn, alpha); sc.append(float((g.pval <= thr).mean()))
    S["power_sc"] = sc
    return S


def cellfmt(r, alpha):
    if r.ok == 0:
        if r.not_run == r.reps: return "not run"
        if r.outside == r.reps: return "outside domain"
        if r.nan == r.reps: return "no value"
        return "--"
    s = f"{r.rate:.3f}~[{r.lo:.3f}--{r.hi:.3f}]"
    if r.table == "null" or r.cell == "null":
        if r.lo > alpha: s = r"\textbf{" + s + "}"
    return s


def tex_null(S, alpha, path):
    S = S[S.table == "null"]
    lines = [r"\begin{tabular}{ll" + "c" * len(TESTS) + "}", r"\toprule", "null & $|S|$ / $n$ & " + " & ".join(LABEL[t] for t in TESTS) + r" \\", r"\midrule"]
    for cell in ["heavy_tail", "hetero", "nonlin_mean", "mixed_Z"]:
        for n in sorted(S.n.unique()):
            for dS in sorted(S.dS.unique()):
                g = S[(S.cell == cell) & (S.n == n) & (S.dS == dS)].set_index("test")
                if g.empty: continue
                lines.append(f"{cell.replace('_', chr(92) + '_')} & {dS} / {n} & " + " & ".join(cellfmt(g.loc[t], alpha) if t in g.index else "--" for t in TESTS) + r" \\")
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def tex_power(S, alpha, path):
    S = S[S.table == "power"]
    lines = [r"\begin{tabular}{ll" + "c" * len(TESTS) + "}", r"\toprule", "alternative & $|S|$ / $n$ & " + " & ".join(LABEL[t] for t in TESTS) + r" \\", r"\midrule"]
    for n in sorted(S.n.unique()):
        for dS in sorted(S.dS.unique()):
            g0 = S[(S.cell == "null") & (S.n == n) & (S.dS == dS)].set_index("test")
            if g0.empty: continue
            lines.append(r"\textit{size on the null} & " + f"{dS} / {n} & " + " & ".join(cellfmt(g0.loc[t], alpha) if t in g0.index else "--" for t in TESTS) + r" \\")
            for cell in ["linear_mean", "nonmonotone_z2", "scale", "tail_shape"]:
                g = S[(S.cell == cell) & (S.n == n) & (S.dS == dS)].set_index("test")
                if g.empty: continue
                def f(t):
                    if t not in g.index: return "--"
                    r = g.loc[t]
                    if r.ok == 0: return cellfmt(r, alpha)
                    return f"{r.rate:.2f} / {r.power_sc:.2f}" if not np.isnan(r.power_sc) else f"{r.rate:.2f}"
                lines.append(f"{cell.replace('_', ' ')} & {dS} / {n} & " + " & ".join(f(t) for t in TESTS) + r" \\")
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--out", default=None); ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "aggregated")
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    D = load(a.dir); S = summarise(D, a.alpha)
    S.to_parquet(os.path.join(out, "grid_all.parquet"), index=False)
    tex_null(S, a.alpha, os.path.join(out, "tables", "grid_null.tex")); tex_power(S, a.alpha, os.path.join(out, "tables", "grid_power.tex"))
    print(S.groupby(["table", "test"]).agg(cells=("rate", "size"), reps=("reps", "sum"), outside=("outside", "sum"), not_run=("not_run", "sum"), nan=("nan", "sum"), error=("error", "sum")).to_string())
    print(f"wrote {out}/grid_all.parquet and tables/grid_null.tex, grid_power.tex")


if __name__ == "__main__":
    main()
