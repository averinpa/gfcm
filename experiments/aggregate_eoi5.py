"""Aggregate run_eoi5.py outputs into the Review 2 tables, loud about partial cells."""
import glob
import os
import sys

import numpy as np
import pandas as pd

ALPHA = 0.05
GFCM_RULES = ["acat3", "bonf3", "joint3", "acat4", "bonf4", "joint4"]


def _dir():
    if len(sys.argv) > 1:
        return sys.argv[1]
    base = os.environ.get("REBUTTAL_OUT") or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "rebuttal")
    return os.path.join(base, "eoi5")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f},{hi:.3f}]" if n else "-"


def long_ci(d):
    """One row per (family, cell, dS, n, rep, arm, pval): GFCM expanded into its rules."""
    g = d[d.test == "gfcm"]
    parts = [g.assign(arm=r, p=g[r]) for r in GFCM_RULES if r in g]
    o = d[d.test != "gfcm"]
    if "error" in o:
        o = o[o["error"].isna()]
    parts.append(o.assign(arm=o.test, p=o.pval))
    return pd.concat(parts)[["family", "cell", "dS", "n", "rep", "arm", "p"]]


def ci_tables(D):
    files = glob.glob(os.path.join(D, "eoi5_ci__*.parquet"))
    if not files:
        print("CI: ABSENT"); return
    d = pd.concat(pd.read_parquet(f) for f in files)
    if "error" in d and d["error"].notna().any():
        print("CI errors:\n", d[d["error"].notna()].groupby(["test", "error"]).size())
    L = long_ci(d)
    reps = L.groupby(["family", "cell", "dS", "n", "arm"]).size()
    print(f"CI: {len(files)} datasets; reps per cell min {reps.min()} max {reps.max()}"
          + ("  <-- PARTIAL" if reps.min() != reps.max() else ""))
    L["rej"] = (L.p < ALPHA).astype(int)
    order = [a for a in GFCM_RULES + ["rcot", "ffci", "blitz"] if a in set(L.arm)]
    for fam in ["evenscale", "calib", "ladder", "interdepth", "misspec"]:
        F = L[L.family == fam]
        if F.empty:
            continue
        print(f"\n=== {fam}: rejection rate at alpha={ALPHA} (Wilson 95%) ===")
        t = F.groupby(["dS", "n", "cell", "arm"]).rej.agg(["sum", "count"])
        t = t.apply(lambda r: fmt(r["sum"], r["count"]), axis=1).unstack("arm")
        print(t[[a for a in order if a in t]].to_string())
        if fam in ("ladder", "evenscale", "interdepth"):
            print(f"\n--- {fam}: size-corrected power (threshold from own null cell) ---")
            rows = []
            for (dS, n, arm), G in F.groupby(["dS", "n", "arm"]):
                nul = G[G.cell == "null"].p
                if nul.empty:
                    continue
                thr = np.quantile(nul, ALPHA)
                for cell, H in G[G.cell != "null"].groupby("cell"):
                    rows.append(dict(dS=dS, n=n, cell=cell, arm=arm, power=float((H.p <= thr).mean())))
            if rows:
                P = pd.DataFrame(rows).pivot_table(index=["dS", "n", "cell"], columns="arm", values="power")
                print(P[[a for a in order if a in P]].round(3).to_string())
    for a in (0.01, 0.10):
        S = L[L.family == "calib"]
        if not S.empty:
            print(f"\n--- calib: size at alpha={a} ---")
            print(S.assign(r=(S.p < a).astype(float)).pivot_table(index=["dS", "n", "cell"], columns="arm",
                                                                    values="r")[[x for x in order if x in set(S.arm)]]
                  .round(3).to_string())


def pc_tables(D):
    files = glob.glob(os.path.join(D, "eoi5_pc__*.parquet"))
    if not files:
        print("\nPC: ABSENT"); return
    d = pd.concat(pd.read_parquet(f) for f in files)
    if "error" in d and d["error"].notna().any():
        print("PC errors:\n", d[d["error"].notna()].groupby(["test", "error"]).size())
        d = d[d["error"].isna()]
    cols = ["shd", "precision", "recall", "fp_edges", "cpdag_shd", "arrow_precision", "arrow_recall",
            "n_queries", "wall_s"]
    g = d.groupby(["p", "deg", "n", "test"])
    m = g[cols].mean(); s = g[cols].sem(); c = g.size().rename("reps")
    out = m.round(3).astype(str) + " (" + s.round(3).astype(str) + ")"
    print("\n=== PC: mean (sem) ===")
    print(pd.concat([out, c], axis=1).to_string())


if __name__ == "__main__":
    D = _dir()
    print("data:", D)
    ci_tables(D)
    pc_tables(D)
