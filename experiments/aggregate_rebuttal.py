"""Aggregate the rebuttal per-rep parquets (data/rebuttal/*) into response-shaped tables.

Reads the three sub-experiments written by run_rebuttal.py and produces one tidy summary each,
matching aggregate.py's conventions (parquet to results/aggregated/ + a readable print to stdout):

  grid_pc   -> rebuttal_gridpc_summary.parquet
               per (grid, n): mean SHD / precision / recall + per-edge-type recall (item 3a).
  grid_null -> rebuttal_gridnull_summary.parquet
               per (grid, n): size = mean reject at alpha, with Wilson 95% interval (item 3b).
  p20       -> rebuttal_p20_summary.parquet
               per (test, n): mean SHD / precision / recall / fp_edges / wall time (items 4+5).

Any sub-experiment with no files yet is skipped (p20 is staged, so it may lag items 3a/3b).

Usage:  python experiments/aggregate_rebuttal.py [--alpha 0.05]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REBDIR = os.environ.get("REBUTTAL_OUT") or os.path.join(os.path.dirname(_HERE), "data", "rebuttal")
OUTDIR = os.path.join(os.path.dirname(_HERE), "results", "aggregated")

GRID_ORDER = ["median", "tails", "default", "dense5", "wide5"]
EXPECTED_P20_PANEL = ["gfcm", "blitz", "rcot", "ffci"]


def _load(prefix):
    fs = glob.glob(os.path.join(REBDIR, f"{prefix}*.parquet"))
    if not fs:
        return None
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    c = (phat + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def _grid_cat(df):
    df = df.copy()
    df["grid"] = pd.Categorical(df["grid"], categories=GRID_ORDER, ordered=True)
    return df


def summarize_grid_pc(df):
    """item 3a: quantile grid vs PC discovery. Mean of the score columns per (grid, n)."""
    df = _grid_cat(df)
    recall_cols = sorted(c for c in df.columns if c.startswith("recall_"))
    metrics = ["shd", "precision", "recall"] + recall_cols
    agg = (df.groupby(["grid", "n"], observed=True)[metrics]
             .agg(lambda s: np.nanmean(s.values)).round(3))
    agg["n_reps"] = df.groupby(["grid", "n"], observed=True).size()
    agg["shd_sem"] = (df.groupby(["grid", "n"], observed=True)["shd"]
                        .agg(lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)).round(3)
    return agg.reset_index()


def summarize_grid_null(df):
    """item 3b: quantile grid vs null size. reject rate + Wilson interval per (grid, n)."""
    df = _grid_cat(df)
    rows = []
    for (grid, n), g in df.groupby(["grid", "n"], observed=True):
        rej = g["reject"].dropna().astype(int).values
        m = len(rej); k = int(rej.sum())
        lo, hi = _wilson(k, m)
        rows.append(dict(grid=grid, n=n, n_reps=m,
                         size=round(k / m, 3) if m else np.nan,
                         size_lo=round(lo, 3), size_hi=round(hi, 3)))
    return pd.DataFrame(rows).sort_values(["n", "grid"]).reset_index(drop=True)


def summarize_p20(df):
    """items 4+5: p=20 discovery + FP + timing. Mean per (test, n)."""
    df = df[~df.get("error", pd.Series(index=df.index, dtype=object)).notna()] if "error" in df else df
    recall_cols = sorted(c for c in df.columns if c.startswith("recall_"))
    metrics = ["shd", "precision", "recall", "fp_edges", "wall_s"] + recall_cols
    metrics = [m for m in metrics if m in df.columns]
    agg = (df.groupby(["test", "n"])[metrics].agg(lambda s: np.nanmean(s.values)).round(3))
    agg["n_reps"] = df.groupby(["test", "n"]).size()
    agg["shd_sem"] = (df.groupby(["test", "n"])["shd"]
                        .agg(lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0)).round(3)
    return agg.reset_index()


def _completeness(df, kind, nfiles):
    """Return (note, is_partial). Loud so a half-finished run never masquerades as complete."""
    ns = sorted(int(x) for x in df["n"].unique())
    lines = [f"  files={nfiles}  n={ns}"]
    partial = False
    if kind in ("grid_pc", "grid_null"):
        have = set(df["grid"])
        missing = [g for g in GRID_ORDER if g not in have]
        rc = df.groupby(["grid", "n"], observed=True).size()
        lines.append(f"  grids present = {[g for g in GRID_ORDER if g in have]}")
        if missing:
            lines.append(f"  !! grids MISSING: {missing}"); partial = True
    else:  # p20
        have = sorted(set(df["test"]))
        missing = [t for t in EXPECTED_P20_PANEL if t not in have]
        rc = df.groupby(["test", "n"]).size()
        lines.append(f"  panel present = {have}")
        if missing:
            lines.append(f"  !! panel INCOMPLETE, missing: {missing} (staged run in progress?)"); partial = True
    lines.append(f"  reps/group: min={int(rc.min())} max={int(rc.max())}")
    if rc.min() != rc.max():
        lines.append("  !! uneven rep counts across groups -- a run is likely still in progress"); partial = True
    return "\n".join(lines), partial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05, help="(recorded for provenance; reject already uses it)")
    ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    jobs = [
        ("grid_pc",   "rebuttal_gridpc__",   "grid_pc  (item 3a) -- quantile grid vs PC discovery", summarize_grid_pc,   "rebuttal_gridpc_summary.parquet"),
        ("grid_null", "rebuttal_gridnull__", "grid_null (item 3b) -- quantile grid vs null size",   summarize_grid_null, "rebuttal_gridnull_summary.parquet"),
        ("p20",       "rebuttal_p20__",      "p20 (items 4+5) -- 20-node discovery + FP + timing",  summarize_p20,       "rebuttal_p20_summary.parquet"),
    ]
    present, absent, partial = [], [], []
    for kind, prefix, title, fn, outname in jobs:
        df = _load(prefix)
        if df is None:
            absent.append(kind)
            print(f"\n=== {title} ===\n  ABSENT: no files matching {prefix}* in {REBDIR}")
            continue
        present.append(kind)
        nfiles = len(glob.glob(os.path.join(REBDIR, f"{prefix}*.parquet")))
        note, is_partial = _completeness(df, kind, nfiles)
        if is_partial:
            partial.append(kind)
        summary = fn(df)
        outp = os.path.join(OUTDIR, outname)
        summary.to_parquet(outp, index=False)
        print(f"\n=== {title} ===")
        print(note)
        print(summary.to_string(index=False))
        print(f"  -> {outp}")

    print("\n" + "=" * 64)
    print(f"PRESENT: {present or '(none)'}")
    print(f"ABSENT:  {absent or '(none)'}")
    if partial:
        print("!!! PARTIAL -- do NOT treat these as complete tables:")
        for k in partial:
            print(f"      - {k}")
    print("=" * 64)


if __name__ == "__main__":
    main()
