"""Aggregate the two mixed-type property runs written by run_cat_level_means.py into T4b and T4c.

Usage: python experiments/aggregate_mixed.py labelinv|cardinality [DIR] [--out results/aggregated]
                                             [--alpha 0.05]

DIR defaults to data/mixed.

labelinv (T4b) -- the same dataset with its levels renumbered. For a test that handles a nominal
variable natively the p-value cannot move at all, so the quantity reported is the spread of the
p-value across labellings within a replication (median and worst case), and the share of
replications in which the decision at alpha changes with the numbering. A test that reads the
levels as numbers has no reason to be invariant, and the table shows what that costs.

cardinality (T4c) -- rejection rate against the number of levels, at a signal whose spread over
levels is held fixed. GFCM pools the K-1 residual directions in one quadratic form; the other arms
are the best a scalar-input test can do, either the integer code or one test per level with a
Bonferroni correction. delta = 0 is the matched null, so the same table carries size and power.
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
ARM = {"pooled": "pooled", "code": "integer code", "perlevel_bonf": "per level, Bonferroni"}
TAG = {"Xcat": "categorical $X$", "Zcat": "categorical conditioner"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def load(d, prefix):
    fs = sorted(glob.glob(os.path.join(d, f"{prefix}*.parquet")))
    if not fs:
        sys.exit(f"no {prefix}*.parquet in {d}")
    return pd.concat((pd.read_parquet(f) for f in fs), ignore_index=True)


# ------------------------------------------------------------------ T4b
def labelinv(D, alpha):
    """Per cell and test: the spread of the p-value over labellings, and the decision flips."""
    ok = D[D.status == "ok"]
    rows = []
    for (tag, K, delta, n, test), g in ok.groupby(["tag", "K", "delta", "n", "test"]):
        per_rep = g.groupby("rep").pval
        spread = (per_rep.max() - per_rep.min()).to_numpy()
        flip = ((per_rep.min() <= alpha) & (per_rep.max() > alpha)).to_numpy()
        reps = int(g.rep.nunique())
        lo, hi = wilson(int(flip.sum()), reps)
        rows.append(dict(tag=tag, K=int(K), delta=float(delta), n=int(n), test=test, reps=reps,
                         labellings=int(g.groupby("rep").perm.nunique().median()),
                         spread_med=float(np.median(spread)), spread_max=float(spread.max()),
                         flip_rate=float(flip.mean()), flip_lo=lo, flip_hi=hi,
                         nan=int((D[(D.tag == tag) & (D.K == K) & (D.delta == delta)
                                    & (D.n == n) & (D.test == test)].status != "ok").sum())))
    return pd.DataFrame(rows)


def tex_labelinv(S, path, alpha):
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"case & test & $K$ & median spread & worst spread & decision flips \\"]
    for tag in ("Xcat", "Zcat"):
        g0 = S[S.tag == tag]
        if g0.empty:
            continue
        lines.append(r"\midrule")
        for i, test in enumerate([t for t in TESTS if t in set(g0.test)]):
            for K in sorted(g0[g0.test == test].K.unique()):
                r = g0[(g0.test == test) & (g0.K == K)]
                if r.empty:
                    continue
                r = r.iloc[0]
                first = TAG[tag] if (i == 0 and K == sorted(g0.K.unique())[0]) else ""
                flip = (r"$0$" if r.flip_rate == 0 else
                        f"{r.flip_rate:.3f}~[{r.flip_lo:.3f}--{r.flip_hi:.3f}]")
                lines.append(f"{first} & {LABEL.get(test, test)} & {int(r.K)} & "
                             f"{r.spread_med:.1e} & {r.spread_max:.1e} & {flip}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# ------------------------------------------------------------------ T4c
def cardinality(D, alpha):
    ok = D[D.status == "ok"]
    rows = []
    for (K, n, delta, test, arm), g in ok.groupby(["K", "n", "delta", "test", "arm"]):
        k = int((g.pval <= alpha).sum()); m = len(g)
        lo, hi = wilson(k, m)
        rows.append(dict(K=int(K), n=int(n), delta=float(delta), test=test, arm=arm,
                         reps=m, rej=k, rate=k / m if m else np.nan, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def tex_cardinality(S, path, alpha):
    Ks = sorted(S.K.unique()); ns = sorted(S.n.unique())
    lines = [r"\begin{tabular}{ll" + "c" * len(Ks) + "}", r"\toprule",
             "test & arm & " + " & ".join(f"$K={k}$" for k in Ks) + r" \\"]
    for n in ns:
        for delta, what in ((0.0, "size"), (None, "power")):
            sub = S[(S.n == n) & ((S.delta == 0) if delta == 0 else (S.delta > 0))]
            if sub.empty:
                continue
            lines += [r"\midrule", r"\multicolumn{" + str(len(Ks) + 2)
                      + "}{l}{\\textit{" + f"{what}, $n={n:,}$".replace(",", "{,}") + "}} \\\\"]
            for test in [t for t in TESTS if t in set(sub.test)]:
                for arm in ("pooled", "code", "perlevel_bonf"):
                    g = sub[(sub.test == test) & (sub.arm == arm)]
                    if g.empty:
                        continue
                    cells = []
                    for K in Ks:
                        r = g[g.K == K]
                        cells.append(f"{r.iloc[0].rate:.2f}" if len(r) else "--")
                    lines.append(f"{LABEL.get(test, test)} & {ARM[arm]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["labelinv", "cardinality"])
    ap.add_argument("dir", nargs="?", default=os.path.join(root, "data", "mixed"))
    ap.add_argument("--out", default=os.path.join(root, "results", "aggregated"))
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    if a.exp == "labelinv":
        S = labelinv(load(a.dir, "mix_labelinv"), a.alpha)
        S.to_parquet(os.path.join(a.out, "mixed_labelinv.parquet"), index=False)
        tex_labelinv(S, os.path.join(a.out, "tables", "mixed_labelinv.tex"), a.alpha)
        print(S[["tag", "test", "K", "reps", "spread_med", "spread_max", "flip_rate"]].to_string(index=False))
    else:
        S = cardinality(load(a.dir, "mix_card"), a.alpha)
        S.to_parquet(os.path.join(a.out, "mixed_cardinality.parquet"), index=False)
        tex_cardinality(S, os.path.join(a.out, "tables", "mixed_cardinality.tex"), a.alpha)
        print(S.pivot_table(index=["test", "arm"], columns=["delta", "K"], values="rate").to_string())


if __name__ == "__main__":
    main()
