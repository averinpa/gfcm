"""Regenerate the paper's data-driven figures from the repo's benchmark output.

  convergence  : reads data/cells_conv/ -> conv_calibration_onset.pdf + conv_power_crossover.pdf
                 (size / size-corrected power vs n, per depth, with Wilson error bars).
  pc           : reads data/cells_pc/  -> fig_pc_panel.pdf
                 (2x3 forest: rows n=2000 / n=1e5, columns SHD / precision / recall, mean +/- 95% CI).

The other paper figures are either table-value plots (tail-strength/tail-depth/speed, drawn from
results/aggregated) or the detection-class illustration (from the archived quantile DGP); those are
not regenerated here.

Usage:  python figures/make_figures.py [convergence] [pc]     # default: all available
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (gfcm pkg)

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gfcm import GFCMConfig
from gfcm.core import (_design, _fold_solvers, _ls_quant_resid, _mvgcm_p, _resid_with, k_rule)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "data")
RESULTS = os.path.join(_ROOT, "results", "aggregated")
OUT = os.path.join(_ROOT, "results", "figures")   # figure PDFs land under results/ (mountable)
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.05
ORDER = ["gfcm", "blitz", "ffci", "rcot", "gcm_boosted", "par_cop", "fisherz"]
LAB = {"gfcm": "GFCM", "blitz": "BLITZ", "ffci": "FFCI", "rcot": "RCoT",
       "gcm_boosted": "GCM-boost", "par_cop": "PartCopula", "fisherz": "Fisher-Z"}
COLORS = {"gfcm": "#d62728", "blitz": "#1f77b4", "ffci": "#2ca02c", "rcot": "#9467bd",
          "gcm_boosted": "#8c564b", "par_cop": "#7f7f7f", "fisherz": "#e377c2"}


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def _load(subdir):
    fs = glob.glob(os.path.join(DATA, subdir, "*.parquet"))
    fs = [f for f in fs if not os.path.basename(f).startswith("convergence_")]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


# --------------------------------------------------------------------------- convergence
def _conv_aggregate(df):
    rows = []
    for (section, dS, n, test), g in df.groupby(["section", "dS", "n", "test"]):
        null_pv = g[g.cell == "null"].pval.dropna().values
        for cell in g.cell.unique():
            pv = g[g.cell == cell].pval.dropna().values
            if len(pv) < 10:
                continue
            k, m = int((pv < ALPHA).sum()), len(pv)
            lo, hi = _wilson(k, m)
            if cell == "null" or section == "5.2_calib":
                rows.append(dict(cell=cell, dS=dS, test=test, n=n, metric="size",
                                 value=k / m, lo=lo, hi=hi))
            elif len(null_pv) >= 20:
                c = np.percentile(null_pv, 100 * ALPHA)
                ka = int((pv < c).sum()); alo, ahi = _wilson(ka, m)
                rows.append(dict(cell=cell, dS=dS, test=test, n=n, metric="adj_power",
                                 value=ka / m, lo=alo, hi=ahi))
    return pd.DataFrame(rows)


def _conv_grid(res, metric, cells, title, fname, hline=None, label=""):
    cells = [c for c in cells if ((res.cell == c) & (res.metric == metric)).any()]
    dss = sorted(res.dS.unique())
    if not cells or not dss:
        print(f"  (no data for {title})"); return
    nv = sorted(res.n.unique())
    fig, axes = plt.subplots(len(cells), len(dss), figsize=(3.6 * len(dss), 2.8 * len(cells)),
                             squeeze=False, sharex=True, sharey=True)
    for i, cell in enumerate(cells):
        for j, dS in enumerate(dss):
            ax = axes[i][j]
            sub = res[(res.cell == cell) & (res.dS == dS) & (res.metric == metric)]
            for t in ORDER:
                ts = sub[sub.test == t].sort_values("n")
                if ts.empty:
                    continue
                ax.errorbar(ts.n, ts.value,
                            yerr=[(ts.value - ts.lo).clip(lower=0), (ts.hi - ts.value).clip(lower=0)],
                            fmt="-o", ms=4, lw=1.7, capsize=2, elinewidth=0.8,
                            color=COLORS.get(t), label=LAB.get(t, t))
            ax.set_xscale("log"); ax.set_xlim(nv[0] * 0.85, nv[-1] * 1.15)
            ax.set_axisbelow(True); ax.grid(True, axis="y", color="#dddddd", lw=0.7)
            if hline is not None:
                ax.axhline(hline, color="k", lw=0.9, ls="--", alpha=0.6)
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(f"|S|={dS}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{cell}\n{metric.replace('_', ' ')}", fontsize=9)
            if i == len(cells) - 1:
                ax.set_xlabel("n", fontsize=10)
    h, l = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=12)
    if label:
        fig.text(0.01, 0.99, label, fontsize=16, fontweight="bold", va="top", ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    p = os.path.join(OUT, fname); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


def convergence():
    df = _load("cells_conv")
    if df is None:
        print("convergence: no data/cells_conv/ yet (run run_convergence)"); return
    res = _conv_aggregate(df)
    # titles/labels dropped from the figure: they live in the LaTeX (\makebox A/B + caption)
    _conv_grid(res, "size", ["heavy_tail", "mixed_Z"],
               "", "conv_calibration_onset.pdf", hline=ALPHA)
    _conv_grid(res, "adj_power", ["alpha_3"],
               "", "conv_power_crossover.pdf")


# --------------------------------------------------------------------------- PC panel
def pc():
    df = _load("cells_pc")
    if df is None:
        print("pc: no data/cells_pc/ yet (run run_pc)"); return
    metrics = [("shd", "skeleton SHD"), ("precision", "precision"), ("recall", "recall")]
    ns = sorted(df.n.unique())
    fig, axes = plt.subplots(len(ns), len(metrics), figsize=(3.4 * len(metrics), 2.4 * len(ns)),
                             squeeze=False)
    for i, n in enumerate(ns):
        sub = df[df.n == n]
        # par_cop (infeasible) and gcm_boosted (uninformative) are excluded at scale (n>=1e5)
        excl = {"par_cop", "gcm_boosted"} if n >= 100000 else set()
        present = [t for t in ORDER if t in sub.test.unique() and t not in excl]
        # rows ordered by mean skeleton SHD (best at top), same order across all columns
        mshd = {t: pd.to_numeric(sub[sub.test == t]["shd"], errors="coerce").dropna().mean() for t in present}
        tests = sorted(present, key=lambda t: mshd[t])
        for j, (col, label) in enumerate(metrics):
            ax = axes[i][j]
            for yi, t in enumerate(tests):
                v = pd.to_numeric(sub[sub.test == t][col], errors="coerce").dropna().values
                if not len(v):
                    continue
                mean = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                ax.errorbar(mean, yi, xerr=1.96 * se, fmt="o", ms=6, capsize=3,
                            color=("#d62728" if t == "gfcm" else "#4c4c4c"))
            ax.set_yticks(range(len(tests)))
            ax.set_yticklabels([LAB.get(t, t) for t in tests] if j == 0 else [], fontsize=9)  # test names on the left column only
            ax.set_ylim(-0.6, len(tests) - 0.4); ax.invert_yaxis()
            ax.set_axisbelow(True); ax.grid(True, axis="x", color="#dddddd", lw=0.7)
            if i == 0:
                ax.set_title(label, fontsize=11)
            if i == len(ns) - 1:
                ax.set_xlabel(label, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"n={n:,}", fontsize=10)
    # title dropped: it lives in the LaTeX caption (no plot-top duplicate)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "fig_pc_panel.pdf"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


# --------------------------------------------------------------------------- speed
def speed():
    p = os.path.join(RESULTS, "speed_grid.parquet")
    if not os.path.exists(p):
        print("speed: no results/aggregated/speed_grid.parquet (run run_speed)"); return
    df = pd.read_parquet(p)
    df = df[df.S <= 20]   # cap |S| at PC-relevant depths (drop reduced-design |S|=50)
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
    axes[0][0].set_ylim(3, 3e5)   # shared y: floor between 10^0 and 10^1 (lifts 10^1 off the bottom); top between 10^5 and 10^6 lifts the 10^5 line off the top edge for headroom
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
    # title dropped: it lives in the LaTeX caption (no plot-top duplicate)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    q = os.path.join(OUT, "fig_speed.pdf"); fig.savefig(q, dpi=150); plt.close(fig)
    print(f"  wrote {q}")


# --------------------------------------------------------------------------- tail figures
def _sc_power(df, section, n=None):
    """Per (dS, cell, test): size-corrected power (null-calibrated) for a section's alt cells."""
    sub = df[df.section == section]
    if n is not None:
        sub = sub[sub.n == n]
    rows = []
    for dS, g in sub.groupby("dS"):
        crit = {t: np.percentile(gt.pval.dropna().values, 100 * ALPHA)
                for t, gt in g[g.cell == "null"].groupby("test") if len(gt.pval.dropna())}
        for (cell, test), gc in g[g.cell != "null"].groupby(["cell", "test"]):
            pv = gc.pval.dropna().values; c = crit.get(test, np.nan)
            if c == c and len(pv):
                k = int((pv < c).sum()); lo, hi = _wilson(k, len(pv))
                rows.append(dict(dS=dS, cell=cell, test=test, power=k / len(pv), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def tail_strength():
    """fig:tail-strength -- size-corrected power vs tail-edge strength (5.4_tail, |S|=1, n=2000)."""
    df = _load("cells")
    if df is None or df[df.section == "5.4_tail"].empty:
        print("tail_strength: no 5.4_tail cells (run run_cells)"); return
    res = _sc_power(df, "5.4_tail", n=2000)
    res = res[res.cell.str.startswith("alpha_")].copy()
    res["strength"] = res.cell.str.split("_").str[1].astype(int)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for t in ORDER:
        ts = res[res.test == t].sort_values("strength")
        if ts.empty:
            continue
        ax.errorbar(ts.strength, ts.power,
                    yerr=[(ts.power - ts.lo).clip(lower=0), (ts.hi - ts.power).clip(lower=0)],
                    fmt="-o", ms=5, lw=1.8, capsize=2.5, color=COLORS.get(t), label=LAB.get(t, t))
    ax.set_xlabel("effect strength (skew coefficient $c$)", fontsize=10)
    ax.set_ylabel("size-corrected power", fontsize=10); ax.set_ylim(-0.03, 1.03)
    ax.set_axisbelow(True); ax.grid(True, axis="y", color="#dddddd", lw=0.7)
    ax.legend(fontsize=9, frameon=False)   # title dropped: it lives in the LaTeX caption (no plot-top duplicate)
    fig.tight_layout(); q = os.path.join(OUT, "fig_tail_strength.pdf"); fig.savefig(q, dpi=150); plt.close(fig)
    print(f"  wrote {q}")


def tail_depth():
    """fig:tail-depth -- graceful degradation: tail-shape power vs conditioning depth (5.3_power tail_shape, n=2000)."""
    df = _load("cells")
    if df is None or df[df.section == "5.3_power"].empty:
        print("tail_depth: no 5.3_power cells (run run_cells)"); return
    res = _sc_power(df, "5.3_power", n=2000)
    res = res[res.cell == "tail_shape"]
    if res.empty:
        print("tail_depth: no tail_shape cells"); return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for t in ORDER:
        ts = res[res.test == t].sort_values("dS")
        if ts.empty:
            continue
        ax.errorbar(ts.dS, ts.power,
                    yerr=[(ts.power - ts.lo).clip(lower=0), (ts.hi - ts.power).clip(lower=0)],
                    fmt="-o", ms=5, lw=1.8, capsize=2.5, color=COLORS.get(t), label=LAB.get(t, t))
    ax.set_xlabel("conditioning depth |S|", fontsize=10)
    ax.set_ylabel("size-corrected power", fontsize=10); ax.set_ylim(-0.03, 1.03)
    ax.set_axisbelow(True); ax.grid(True, axis="y", color="#dddddd", lw=0.7)
    ax.legend(fontsize=9, frameon=False)
    # title dropped: it lives in the LaTeX caption (no plot-top duplicate)
    fig.tight_layout(); q = os.path.join(OUT, "fig_tail_depth.pdf"); fig.savefig(q, dpi=150); plt.close(fig)
    print(f"  wrote {q}")


# --------------------------------------------------------------------------- detection class
def _quantile_block_p(X, Y, Z, taus, cfg, seed=0, B=199):
    """Single-orientation quantile-block p-value beta(tau)=E[e_X r_tau(Y)] for a tau grid --
    the sub-bank the detection-class figure isolates (no covariance/scale blocks)."""
    X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    Zt = _design(Z, k_rule(len(X)), cfg)
    folds = list(KFold(cfg.folds, shuffle=True, random_state=42).split(Zt))
    solvers = _fold_solvers(Zt, folds, None)
    eX = _resid_with(solvers, Zt, X, None)
    eY = _resid_with(solvers, Zt, Y, None)
    cY = _resid_with(solvers, Zt, np.abs(eY), None)
    s2Y = (np.abs(eY) - cY) ** 2                      # |e|-scale -> variance proxy for standardization
    RY = np.column_stack([_ls_quant_resid(eY, s2Y, folds, taus, cfg)[t] for t in taus])
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(B, len(X)))
    return _mvgcm_p(eX[:, None] * RY, signs, cfg)


def _scale_alt(seed, n):
    """Median-blind scale alternative: X drives Y's spread, conditional median fixed.

    Hand-built (not dagsampler) by necessity. The quantile block isolated in this figure tests
    the directional orientation e_X * r_tau(Y), which detects a *monotone* (level-driven) scale
    edge; exp(0.9*Xc) is monotone in X's residual, so the tail levels catch it (power 1.00).
    dagsampler's heteroskedastic funcs are all |parent|-based (even in the driver), i.e. a
    magnitude-driven scale edge that is invisible to this orientation (verified: tail power ~0.05
    even at large X-noise). Migrating this would need a monotone/exp heteroskedastic mechanism
    that dagsampler does not yet provide."""
    r = np.random.default_rng(seed)
    Z = r.uniform(-1.5, 1.5, n); Xc = r.normal(size=n); X = Z + Xc
    Y = 0.5 * Z + np.exp(0.9 * Xc) * r.normal(size=n)
    return X, Y, Z[:, None]


def _localized_alt(seed, n):
    """Effect curve vanishes at 0.1/0.5/0.9, bump at 0.7."""
    r = np.random.default_rng(seed)
    Z = r.uniform(-1.5, 1.5, n); U = r.uniform(0.0, 1.0, n)
    h = np.where((U > 0.6) & (U < 0.8), np.sin(2 * np.pi * (U - 0.6) / 0.2), 0.0)
    X = Z + 2.0 * h + r.normal(size=n)
    Y = 0.5 * Z + norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
    return X, Y, Z[:, None]


def _det_rates(dgp, grids, reps, n, cfg):
    out = []
    for taus in grids:
        k = sum(_quantile_block_p(*dgp(2000 + it, n), taus, cfg, seed=it) < ALPHA for it in range(reps))
        lo, hi = _wilson(k, reps)
        out.append(dict(grid="{" + ",".join(str(t) for t in taus) + "}", power=k / reps, lo=lo, hi=hi))
    return pd.DataFrame(out)


def detection(reps=200, n=2000):
    """fig:detection -- which finite quantile grids catch which alternatives."""
    cfg = GFCMConfig()
    # Symmetric panels: each isolates the single decisive level (left {0.9}, right {0.7}).
    left = _det_rates(_scale_alt, [(0.5,), (0.1, 0.9), (0.9,)], reps, n, cfg)
    right = _det_rates(_localized_alt, [(0.1, 0.5, 0.9), (0.1, 0.5, 0.7, 0.9), (0.7,)], reps, n, cfg)
    DETECT, BLIND = "#1f77b4", "#d62728"                    # blue = caught, red = blind grid
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))
    for ax, res, title in ((axes[0], left, "scale alternative\n(median is blind)"),
                           (axes[1], right, "localized alternative\n(blind at 0.1/0.5/0.9)")):
        y = np.arange(len(res))
        for yi, row in zip(y, res.itertuples()):
            c = BLIND if row.power < 0.5 else DETECT
            ax.errorbar(row.power, yi,
                        xerr=[[max(row.power - row.lo, 0.0)], [max(row.hi - row.power, 0.0)]],
                        fmt="o", ms=7, capsize=3, color=c, zorder=3)
            ax.annotate(f"{row.power:.2f}", (row.power, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=c)
        ax.set_yticks(y); ax.set_yticklabels(res.grid, fontsize=10)
        ax.set_ylim(-0.6, len(res) - 0.4); ax.invert_yaxis()
        ax.axvline(ALPHA, color="0.45", ls="--", lw=0.9)
        ax.text(ALPHA + 0.012, len(res) - 1, r"$\alpha=0.05$", fontsize=8, color="0.45", va="center")
        ax.set_xlim(-0.03, 1.10); ax.set_xlabel("power", fontsize=10)
        ax.set_axisbelow(True); ax.grid(True, axis="x", color="#dddddd", lw=0.7)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    q = os.path.join(OUT, "fig_detection_class.pdf"); fig.savefig(q, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {q}")


def main():
    want = sys.argv[1:] or ["convergence", "pc", "speed", "tail_strength", "tail_depth", "detection"]
    dispatch = {"convergence": convergence, "pc": pc, "speed": speed,
                "tail_strength": tail_strength, "tail_depth": tail_depth, "detection": detection}
    for name in want:
        if name in dispatch:
            print(f"{name}:"); dispatch[name]()


if __name__ == "__main__":
    main()
