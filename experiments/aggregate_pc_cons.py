"""Aggregate the conservative-PC runs written by run_pc_cons.py into one parquet and the T8 table.

Usage: python experiments/aggregate_pc_cons.py [DIR] [--out results/aggregated] [--collider majority]

DIR defaults to data/pc_cons. Writes pc_cons_all.parquet (one row per setting x test x collider
rule: replications, mean and 95 percent interval for every metric) and tables/pc_cons.tex.

Each graph setting is a block of the table; within a block one row per test. Both collider rules
were run on the same draws, so the sepset column isolates what the majority rule changes.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TESTS = ["gfcm", "blitz", "rcot", "ffci", "gcm_boosted"]
LABEL = {"gfcm": r"\textbf{GFCM}", "blitz": "BLITZ", "rcot": "RCoT", "ffci": "FFCI",
         "gcm_boosted": "GCM-boost"}
# column key, header, decimals, and whether larger is better (for the bold rule)
METRICS = [("shd", "skeleton SHD", 1, False), ("precision", "precision", 3, True),
           ("recall", "recall", 3, True), ("fp_edges", "FP edges", 1, False),
           ("cpdag_shd", "CPDAG SHD", 1, False), ("arrow_precision", "arrow prec.", 3, True),
           ("arrow_recall", "arrow rec.", 3, True)]


def setting_label(p, deg, edge_prob):
    if deg == deg and deg is not None:          # NaN-safe
        return f"$p={int(p)}$, degree {int(deg)}"
    return f"$p={int(p)}$, edge prob.\\ {edge_prob:.2g}"


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "pccons_*.parquet")))
    if not fs:
        sys.exit(f"no pccons_*.parquet in {d}")
    return pd.concat((pd.read_parquet(f) for f in fs), ignore_index=True)


def summarise(D):
    """Mean and 95 percent interval per setting x test x rule, plus the median cost."""
    if "error" in D.columns:
        D = D[D.error.isna()]
    keys = ["p", "deg", "edge_prob", "n", "test", "collider"]
    rows = []
    for k, g in D.groupby(keys, dropna=False):
        r = dict(zip(keys, k), reps=len(g))
        for col, *_ in METRICS:
            v = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
            if len(v):
                se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                r[col], r[col + "_lo"], r[col + "_hi"] = v.mean(), v.mean() - 1.96 * se, v.mean() + 1.96 * se
            else:
                r[col] = r[col + "_lo"] = r[col + "_hi"] = np.nan
        r["n_queries"] = float(g.n_queries.median()) if "n_queries" in g else np.nan
        r["wall_s"] = float(g.wall_s.median()) if "wall_s" in g else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def tex(S, path, collider="majority", ps=(20, 50)):
    S = S[(S.collider == collider) & (S.p.isin(ps))]
    if S.empty:
        print(f"pc_cons: nothing for collider={collider}, p in {ps}"); return
    cols = "l" + "c" * len(METRICS) + "rr"
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             "test & " + " & ".join(h for _, h, _, _ in METRICS) + r" & CI calls & s/graph \\"]
    for key, g in S.groupby(["p", "deg", "edge_prob"], dropna=False):
        lines += [r"\midrule", r"\multicolumn{" + str(len(METRICS) + 3) + "}{l}{\\textit{"
                  + setting_label(*key) + "}} \\\\"]
        best = {}
        for col, _, _, big in METRICS:
            v = g.set_index("test")[col].dropna()
            if len(v): best[col] = v.idxmax() if big else v.idxmin()
        for t in TESTS:
            r = g[g.test == t]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = []
            for col, _, dec, _ in METRICS:
                x = f"{r[col]:.{dec}f}" if r[col] == r[col] else "--"
                cells.append(r"\textbf{" + x + "}" if best.get(col) == t else x)
            lines.append(f"{LABEL.get(t, t)} & " + " & ".join(cells)
                         + f" & {r.n_queries:,.0f} & {r.wall_s:,.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("dir", nargs="?", default=os.path.join(root, "data", "pc_cons"))
    ap.add_argument("--out", default=os.path.join(root, "results", "aggregated"))
    ap.add_argument("--collider", default="majority", choices=["majority", "sepset"])
    ap.add_argument("--ps", default="20,50", help="graph sizes in the table (the figure uses all)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    S = summarise(load(a.dir))
    S.to_parquet(os.path.join(a.out, "pc_cons_all.parquet"), index=False)
    tex(S, os.path.join(a.out, "tables", "pc_cons.tex"), a.collider,
        tuple(int(x) for x in a.ps.split(",")))
    print(S.groupby(["p", "collider"], dropna=False)
           .agg(tests=("test", "nunique"), reps=("reps", "max")).to_string())
    print(f"wrote {a.out}/pc_cons_all.parquet")


if __name__ == "__main__":
    main()
