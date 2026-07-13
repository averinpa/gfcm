"""Size-corrected aggregation of the per-rep cell parquets into paper-shaped tables.

Reads the long-format cells written by run_cells (data/cells/*.parquet: one row per rep, with the
raw p-value kept) and produces, per (section, n, dS, test, cell):

  * raw          -- rejection rate at alpha (this is the SIZE on null cells, raw power on alts),
  * sc_power     -- SIZE-CORRECTED power on alt cells: a test's threshold is calibrated to its OWN
                    empirical null (c = alpha-quantile of the matched null cell's p-values), then
                    power = fraction of the alt p-values <= c. This is the methodological backbone:
                    no test benefits from being liberal, because its own null sets its threshold,
  * Wilson 95% intervals for both.

Output: a tidy table written to results/aggregated/cells_summary.parquet (small, committable) and
a readable per-section summary printed to stdout.

Usage:  python experiments/aggregate.py [--section 5.3_power ...] [--alpha 0.05]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
CELLDIR = os.environ.get("CELLS_OUT") or os.path.join(os.path.dirname(_HERE), "data", "cells")
OUTDIR = os.path.join(os.path.dirname(_HERE), "results", "aggregated")


def load(celldir=CELLDIR):
    fs = glob.glob(os.path.join(celldir, "*.parquet"))
    if not fs:
        sys.exit(f"no cells in {celldir} (run run_cells first)")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def _crit(null_pvals, alpha):
    """Size-correcting threshold: the alpha-quantile of a test's null p-values."""
    p = np.asarray(null_pvals, float); p = p[~np.isnan(p)]
    return float(np.quantile(p, alpha)) if len(p) else np.nan


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    c = (phat + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(df, alpha=0.05):
    rows = []
    for (section, n, dS), g in df.groupby(["section", "n", "dS"]):
        # per-test size-correcting threshold from the matched null cell
        crit = {t: _crit(gt.pval.values, alpha) for t, gt in g[g.label == "null"].groupby("test")}
        for (cell, label, test), gc in g.groupby(["cell", "label", "test"]):
            pv = gc.pval.dropna().values
            m = len(pv)
            k_raw = int((pv < alpha).sum())
            lo, hi = _wilson(k_raw, m)
            rec = dict(section=section, n=n, dS=dS, cell=cell, label=label, test=test,
                       n_reps=m, raw=(k_raw / m if m else np.nan), raw_lo=lo, raw_hi=hi,
                       sc_power=np.nan, sc_lo=np.nan, sc_hi=np.nan)
            if label == "alt":
                c = crit.get(test, np.nan)
                if c == c and m:                       # matched null present
                    k = int((pv <= c).sum())
                    slo, shi = _wilson(k, m)
                    rec.update(sc_power=k / m, sc_lo=slo, sc_hi=shi)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["section", "n", "dS", "label", "cell", "test"])


def _render(summary):
    for (section, n), g in summary.groupby(["section", "n"]):
        print(f"\n{'='*70}\n{section}   n={n}\n{'='*70}")
        nulls = g[g.label == "null"]
        if len(nulls):
            print("  size (null, raw rejection @alpha):")
            for _, r in nulls.iterrows():
                print(f"    {r.cell:16s} dS={r.dS} {r.test:8s} size={r.raw:.3f} "
                      f"[{r.raw_lo:.3f},{r.raw_hi:.3f}]  (n_reps={r.n_reps})")
        alts = g[g.label == "alt"]
        if len(alts):
            print("  size-corrected power (alt):")
            for _, r in alts.iterrows():
                sc = f"{r.sc_power:.3f} [{r.sc_lo:.3f},{r.sc_hi:.3f}]" if r.sc_power == r.sc_power else "--"
                print(f"    {r.cell:16s} dS={r.dS} {r.test:8s} sc_power={sc}  (raw={r.raw:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="*", help="restrict to these sections")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = load()
    if args.section:
        df = df[df.section.isin(args.section)]
    summary = summarize(df, alpha=args.alpha)
    os.makedirs(OUTDIR, exist_ok=True)
    outp = os.path.join(OUTDIR, "cells_summary.parquet")
    summary.to_parquet(outp, index=False)
    _render(summary)
    print(f"\nwrote {outp}  ({len(summary)} rows)")


if __name__ == "__main__":
    main()
