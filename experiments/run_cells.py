"""Per-(cell, test) benchmark runner (resumable).

Each (cell, test) is an independent unit written to data/cells/<cellkey>__<test>.parquet in LONG
format -- one row per repetition, storing the seed, the raw p-value, the reject indicator, and a
config_id for provenance. This keeps p-values (so size-corrected power is pure post-processing),
keeps seeds (so any cell reproduces exactly), and skips finished tests on restart. Atomic .claim
files make parallel workers safe.

Configuration is explicit: GFCM runs at its canonical GFCMConfig by default (no environment
variables). Competitor tests are deferred to the Docker workstream -- run with `--only gfcm`.

Usage:
  python experiments/run_cells.py --only gfcm [--panel NAME] [--tier N] [--limit N]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # experiments/ : dgp, manifest, adapters, gfcm_citest
sys.path.insert(0, os.path.dirname(_HERE))      # repo root     : the gfcm package

import argparse

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters
import manifest as M

OUT = os.environ.get("CELLS_OUT") or os.path.join(os.path.dirname(_HERE), "data", "cells")
ALPHA = float(os.environ.get("CELLS_ALPHA", "0.05"))
SEED0 = int(os.environ.get("CELLS_SEED0", "1000"))


def _outp(cellkey, test):
    return os.path.join(OUT, f"{cellkey}__{test}.parquet")


def run_cell(cell, panel_tests, nrep):
    """Run the pending tests of one cell; data is generated once per rep and shared across tests."""
    key = cell["key"]; x, y, S = cell["query"]; build = cell["build"]
    pending = []
    for t in panel_tests:
        outp = _outp(key, t)
        if os.path.exists(outp) and os.path.getsize(outp) > 0:
            continue
        try:                                            # atomic claim so parallel workers don't duplicate
            fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
            pending.append(t)
        except FileExistsError:
            pass
    if not pending:
        return
    print(f"cell {key}: running {pending}", flush=True)
    acc = {t: {"seed": [], "pval": []} for t in pending}
    label = "unknown"
    for it in range(nrep):
        seed = SEED0 + it
        out = CausalDataGenerator(build(seed)).simulate(); df = out["data"]
        if it == 0:
            label = M.oracle_label(out, x, y, S)
        cols = list(df.columns); xi, yi = cols.index(x), cols.index(y); Si = [cols.index(s) for s in S]
        d = df.to_numpy(float)
        for t in pending:
            try:
                p = float(adapters.make_test(t, d)(xi, yi, Si))
            except NotImplementedError:
                raise
            except Exception:
                p = float("nan")
            acc[t]["seed"].append(seed); acc[t]["pval"].append(p)
    metric = "power" if label == "alt" else "size"
    for t in pending:
        pv = np.array(acc[t]["pval"], float)
        rej = np.where(np.isnan(pv), np.nan, (pv < ALPHA).astype(float))
        pd.DataFrame({
            "rep": range(nrep), "seed": acc[t]["seed"], "pval": pv, "reject": rej,
            "test": t, "label": label, "metric": metric,
            "section": cell["section"], "cell": cell["cell"], "n": cell["n"], "dS": cell["dS"],
            "config_id": adapters.config_id(t), "n_iter": nrep,
        }).to_parquet(_outp(key, t), index=False)
        k = int((pv < ALPHA).sum()); n = int(np.isfinite(pv).sum())
        print(f"  done {key}__{t}: {metric}={(k / n if n else float('nan')):.3f} (n={n})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated test names to run (e.g. gfcm)")
    ap.add_argument("--panel", help="override the panel for every cell")
    ap.add_argument("--tier", type=int, help="run only this tier")
    ap.add_argument("--section", nargs="*", help="run only these sections (e.g. 5.3_power)")
    ap.add_argument("--limit", type=int, help="run at most this many cells (smoke)")
    ap.add_argument("--reps", type=int, help="override n_iter (smoke)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    cells = [c for c in M.manifest() if c["kind"] == "ci"]
    if args.tier is not None:
        cells = [c for c in cells if c["tier"] == args.tier]
    if args.section:
        cells = [c for c in cells if c["section"] in set(args.section)]
    if args.limit:
        cells = cells[:args.limit]
    print(f"run_cells: {len(cells)} ci cells -> {OUT}", flush=True)
    for c in cells:
        panel = (args.panel or c.get("panel", "gfcm"))
        panel_tests = panel.split(",") if isinstance(panel, str) else list(panel)
        if only:
            panel_tests = [t for t in panel_tests if t in only]
        if not panel_tests:
            continue
        run_cell(c, panel_tests, args.reps or int(c["n_iter"]))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
