"""The speed figure: per-call wall time against conditioning-set size, one panel per sample size.

Reads results/aggregated/speed_grid.parquet (written by experiments/run_speed.py), one row per
(test, n, |S|) with the median per-call time in milliseconds.

Usage:  python figures/fig_speed.py [--out DIR] [--max-S 20]
"""
import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(_ROOT, "results", "aggregated", "speed_grid.parquet")
OUT = os.path.join(_ROOT, "results", "figures")

ORDER = ["gfcm", "gfcm_exact", "gfcm_published", "blitz", "ffci", "rcot", "gcm_boosted", "par_cop"]
LAB = {"gfcm": "GFCM", "gfcm_exact": "GFCM (exact)", "gfcm_published": "GFCM (as submitted)",
       "blitz": "BLITZ", "ffci": "FFCI", "rcot": "RCoT", "gcm_boosted": "GCM-boost",
       "par_cop": "PartCopula"}
COLORS = {"gfcm": "#d62728", "gfcm_exact": "#ff9896", "gfcm_published": "#8c564b",
          "blitz": "#1f77b4", "ffci": "#2ca02c", "rcot": "#9467bd",
          "gcm_boosted": "#bcbd22", "par_cop": "#7f7f7f"}


def speed(grid=GRID, out=OUT, max_S=20):
    if not os.path.exists(grid):
        print(f"speed: no {grid} (run experiments/run_speed.py)"); return None
    df = pd.read_parquet(grid)
    df = df[df.S <= max_S]          # PC-relevant depths; the reduced design at |S|=50 is not comparable
    ns = sorted(df.n.unique()); ds = sorted(df.S.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.2 * len(ns), 3.0), squeeze=False, sharey=True)
    for j, n in enumerate(ns):
        ax = axes[0][j]; sub = df[df.n == n]
        for t in ORDER:
            ts = sub[sub.test == t].sort_values("S")
            if ts.empty or ts.ms.isna().all():
                continue
            ax.plot(ts.S, ts.ms, "-o", ms=4, lw=1.7, color=COLORS.get(t), label=LAB.get(t, t))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(ds); ax.set_xticklabels([str(v) for v in ds]); ax.minorticks_off()
        ax.set_axisbelow(True); ax.grid(True, which="major", axis="y", color="#dddddd", lw=0.7)
        title = r"$n = 10^5$" if n == 100000 else r"$n = " + f"{n:,}".replace(",", "{,}") + "$"
        ax.set_title(title, fontsize=11); ax.set_xlabel(r"$|S|$", fontsize=10)
        if j == 0:
            ax.set_ylabel("per-call time (ms, log)", fontsize=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(out, exist_ok=True)
    q = os.path.join(out, "fig_speed.pdf"); fig.savefig(q, dpi=150); plt.close(fig)
    print(f"wrote {q}")
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-S", type=int, default=20)
    a = ap.parse_args()
    speed(a.grid, a.out, a.max_S)
