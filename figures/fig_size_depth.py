"""Size against conditioning depth, the test as submitted against the test with the correction.

Reads results/aggregated/grid_all.parquet (written by experiments/aggregate_grid.py from the
`run_grid.py --table null` cells), takes the `gfcm_published` and `gfcm` arms, and pools the
calibration nulls at one sample size: rejections and usable replications are summed across cells,
so the point is the size of the whole null family rather than of one cell, with a Wilson interval.

The two arms run on the same draws, so the gap between the lines is the correction and nothing else.

Usage:  python figures/fig_size_depth.py [--n 2000] [--cells a,b] [--alpha 0.05] [--out DIR]
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(_ROOT, "results", "aggregated", "grid_all.parquet")
OUT = os.path.join(_ROOT, "results", "figures")

ARMS = [("gfcm_nocorr", "without the correction", "#7f7f7f"), ("gfcm", "GFCM", "#d62728")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def size_depth(agg=AGG, out=OUT, n=2000, cells=None, alpha=0.05):
    if not os.path.exists(agg):
        print(f"size_depth: no {agg} (run experiments/aggregate_grid.py)"); return None
    df = pd.read_parquet(agg)
    df = df[(df.table == "null") & (df.n == n)]
    if cells:
        df = df[df.cell.isin(cells)]
    if df.empty:
        print(f"size_depth: no null cells at n={n}"); return None
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ds = sorted(df.dS.unique())
    drawn = 0; top = alpha
    for test, lab, colour in ARMS:
        g = df[df.test == test]
        if g.empty:
            print(f"  (no {test} arm in the aggregate; the figure needs both)"); continue
        xs, ys, lo, hi = [], [], [], []
        for d in ds:
            gd = g[g.dS == d]
            k, m = int(gd.rej.sum()), int(gd.ok.sum())        # pool the null cells
            if m == 0:
                continue
            a, b = wilson(k, m)
            xs.append(d); ys.append(k / m); lo.append(k / m - a); hi.append(b - k / m)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", ms=4, lw=1.6, capsize=3,
                    color=colour, label=lab)
        top = max(top, max(y + h for y, h in zip(ys, hi)))      # both arms, error bars included
        drawn += 1
    if not drawn:
        print("size_depth: neither arm present"); plt.close(fig); return None
    ax.axhline(alpha, ls="--", lw=1.0, color="#555555", zorder=0)
    ax.set_xscale("log"); ax.set_xticks(ds); ax.set_xticklabels([str(d) for d in ds])
    ax.minorticks_off()
    ax.set_xlabel(r"conditioning depth $|Z|$", fontsize=10)
    ax.set_ylabel("size", fontsize=10)
    ax.set_ylim(0, top * 1.12)
    ax.set_axisbelow(True); ax.grid(True, axis="y", color="#dddddd", lw=0.7)
    ax.legend(fontsize=9, frameon=True, framealpha=1.0, edgecolor="#999999")
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    q = os.path.join(out, "fig_size_depth.pdf"); fig.savefig(q, dpi=150); plt.close(fig)
    print(f"wrote {q}")
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=AGG)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--cells", help="comma-separated subset of the null cells (default: all four)")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    size_depth(a.agg, a.out, a.n, a.cells.split(",") if a.cells else None, a.alpha)
