"""PC discovery under the majority (conservative) collider rule, uncapped: skeleton and CPDAG panels.

Reads results/aggregated/pc_cons_all.parquet (written by experiments/aggregate_pc_cons.py), one row
per graph setting x test x collider rule with the mean and its 95 percent interval.

Top row is the skeleton, which the collider rule cannot touch; bottom row is the CPDAG, which is
where the rule acts. Each panel plots the four graph settings on the x axis, one line per test.

Usage:  python figures/fig_pc_cons.py [--collider majority] [--out DIR]
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(_ROOT, "results", "aggregated", "pc_cons_all.parquet")
OUT = os.path.join(_ROOT, "results", "figures")

ORDER = ["gfcm", "blitz", "rcot", "ffci", "gcm_boosted"]
LAB = {"gfcm": "GFCM", "blitz": "BLITZ", "rcot": "RCoT", "ffci": "FFCI", "gcm_boosted": "GCM-boost"}
COLORS = {"gfcm": "#d62728", "blitz": "#1f77b4", "rcot": "#9467bd", "ffci": "#2ca02c",
          "gcm_boosted": "#8c564b"}
PANELS = [[("shd", "skeleton SHD"), ("precision", "skeleton precision"), ("recall", "skeleton recall")],
          [("cpdag_shd", "CPDAG SHD"), ("arrow_precision", "arrowhead precision"),
           ("arrow_recall", "arrowhead recall")]]


def _settings(df):
    """Graph settings on the x axis, ordered by size then density."""
    keys = df[["p", "deg", "edge_prob"]].drop_duplicates()
    keys = keys.sort_values(["p", "deg"], na_position="first")
    out = []
    for _, r in keys.iterrows():
        lab = f"p={int(r.p)}\ndeg {int(r.deg)}" if r.deg == r.deg else f"p={int(r.p)}\nep {r.edge_prob:.2g}"
        out.append((r.p, r.deg, r.edge_prob, lab))
    return out


def pc_cons(agg=AGG, out=OUT, collider="majority"):
    if not os.path.exists(agg):
        print(f"pc_cons: no {agg} (run experiments/aggregate_pc_cons.py)"); return None
    df = pd.read_parquet(agg)
    df = df[df.collider == collider]
    if df.empty:
        print(f"pc_cons: nothing for collider={collider}"); return None
    setts = _settings(df)
    x = np.arange(len(setts))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), squeeze=False)
    for i, row in enumerate(PANELS):
        for j, (col, title) in enumerate(row):
            ax = axes[i][j]
            for k, t in enumerate(ORDER):
                xs, ys, lo, hi = [], [], [], []
                for s, (p, deg, ep, _) in enumerate(setts):
                    m = df[(df.test == t) & (df.p == p) & (df.edge_prob == ep)
                           & ((df.deg == deg) | (df.deg.isna() & (deg != deg)))]
                    if m.empty or m.iloc[0][col] != m.iloc[0][col]:
                        continue
                    r = m.iloc[0]
                    xs.append(s + (k - 2) * 0.06); ys.append(r[col])
                    lo.append(max(0.0, r[col] - r[col + "_lo"])); hi.append(max(0.0, r[col + "_hi"] - r[col]))
                if not xs:
                    continue
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", ms=4, lw=1.5, capsize=2.5,
                            color=COLORS.get(t), label=LAB.get(t, t))
            ax.set_xticks(x); ax.set_xticklabels([s[3] for s in setts], fontsize=8)
            ax.set_xlim(-0.5, len(setts) - 0.5)
            ax.set_title(title, fontsize=10)
            ax.set_axisbelow(True); ax.grid(True, axis="y", color="#dddddd", lw=0.7)
            if col in ("precision", "recall", "arrow_precision", "arrow_recall"):
                ax.set_ylim(-0.03, 1.03)
    axes[0][0].set_ylabel("skeleton", fontsize=10); axes[1][0].set_ylabel("CPDAG", fontsize=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(out, exist_ok=True)
    q = os.path.join(out, "fig_pc_cons.pdf"); fig.savefig(q, dpi=150); plt.close(fig)
    print(f"wrote {q}")
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=AGG)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--collider", default="majority", choices=["majority", "sepset"])
    a = ap.parse_args()
    pc_cons(a.agg, a.out, a.collider)
