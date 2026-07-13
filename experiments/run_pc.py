"""PC discovery runner (resumable) with per-CI-decision recording.

Per (PC cell, rep): simulate the typed DAG, run cbcd's PC for each panel test, and write
  (a) per-test skeleton scores (SHD / recall / precision / F1 / FP + per-edge-type recall)
      -> data/cells_pc/<key>__rep<NNNN>.parquet, and
  (b) the full CI-decision log (every (x, y, S) query, its p-value, depth, indep decision,
      whether it is a true edge) -> data/decisions_pc/<key>__rep<NNNN>.parquet.
Resumable per rep via atomic .claim files.

GFCM runs at its canonical GFCMConfig (no environment variables); the dataset-level type mask is
resolved once inside GFCMCITest and threaded to every CI query by column index (so type handling
is correct and cheap inside PC). Competitor tests are deferred to the Docker workstream -- run
with `--only gfcm`.

Usage:
  python experiments/run_pc.py --only gfcm [--n 2000] [--reps 100]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # experiments/ : dgp, manifest, adapters, discovery, metrics, gfcm_citest
sys.path.insert(0, os.path.dirname(_HERE))      # repo root     : the gfcm package

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "XGB_NTHREAD"):
    os.environ.setdefault(_v, "1")

import argparse

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters
import discovery as _disc
import manifest as M
import metrics as _metrics

OUT = os.environ.get("CELLS_PC_OUT") or os.path.join(os.path.dirname(_HERE), "data", "cells_pc")
DEC = os.environ.get("DEC_PC_OUT") or os.path.join(os.path.dirname(_HERE), "data", "decisions_pc")
SEED0 = 10000
ALPHA = 0.05


def run_rep(cell, it, panel_tests):
    key = cell["key"]
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        return
    try:
        fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return
    cfg, etype = cell["build_typed"](SEED0 + it)
    cfg["simulation_params"]["n_samples"] = cell["n"]
    df = CausalDataGenerator(cfg).simulate()["data"]
    data = df.to_numpy(float); nodes = list(df.columns)
    true_sk = {frozenset((a, b)) for a, b in cfg["graph_params"]["edges"]}
    rows, dec_rows = [], []
    for nm in panel_tests:
        log = []
        try:
            ci = adapters.make_test(nm, data)
            cg = _disc.run_pc(data, nodes, ci, ALPHA, cell["max_cond"], log=log)
            learned = _disc.cpdag_skeleton(cg, nodes)
            sc = _metrics.skeleton_scores(true_sk, learned, nodes)
            sc.update({f"recall_{t}": v for t, v in _metrics.per_type_recall(etype, learned).items()})
            sc.update({"test": nm, "rep": it, "n": cell["n"], "cell": cell["cell"],
                       "section": cell["section"], "config_id": adapters.config_id(nm)})
            rows.append(sc)
        except NotImplementedError:
            raise
        except Exception as e:
            rows.append({"test": nm, "rep": it, "n": cell["n"], "cell": cell["cell"], "error": str(e)[:200]})
        for (x, y, S, p) in log:
            dec_rows.append({"test": nm, "rep": it, "n": cell["n"],
                             "x": x, "y": y, "xname": nodes[x], "yname": nodes[y],
                             "S": ",".join(str(s) for s in S),
                             "Snames": ",".join(nodes[s] for s in S),
                             "depth": len(S), "p_value": p,
                             "indep": bool(p >= ALPHA),
                             "true_edge": bool(frozenset((nodes[x], nodes[y])) in true_sk)})
    pd.DataFrame(rows).to_parquet(outp, index=False)
    if dec_rows:
        pd.DataFrame(dec_rows).to_parquet(os.path.join(DEC, f"{key}__rep{it:04d}.parquet"), index=False)
    print(f"done {key} rep {it} ({len(rows)} tests, {len(dec_rows)} CI decisions)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated test names (e.g. gfcm)")
    ap.add_argument("--panel", help="override the panel for every cell")
    ap.add_argument("--n", type=int, help="run only PC cells at this sample size")
    ap.add_argument("--reps", type=int, help="override the number of reps")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True); os.makedirs(DEC, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    cells = [c for c in M.manifest() if c["kind"] == "pc" and (args.n is None or c["n"] == args.n)]
    print(f"run_pc: {len(cells)} pc cells -> {OUT} | decisions -> {DEC}", flush=True)
    for c in cells:
        panel = args.panel or c["panel"]
        panel_tests = panel.split(",") if isinstance(panel, str) else list(panel)
        if only:
            panel_tests = [t for t in panel_tests if t in only]
        if not panel_tests:
            continue
        reps = args.reps or int(c.get("reps", 100))
        for it in range(reps):
            run_rep(c, it, panel_tests)
    print("PC DONE", flush=True)


if __name__ == "__main__":
    main()
