"""Rebuttal experiments for the TMLR review (resumable, claim-safe, same engine as run_pc.py).

Three sub-experiments, all reusing dgp / adapters / discovery / metrics so they match the paper:

  grid_pc   (Review item 3a) -- quantile-grid sensitivity at the PC level.
            8-node tailgraph (as in 5.8/5.10), GFCM only, one PC run per (grid, n, rep).
            Grids: median {0.5}, tails {0.1,0.9}, default {0.1,0.5,0.9},
                   dense5 {0.1,0.3,0.5,0.7,0.9}, wide5 {0.05,0.25,0.5,0.75,0.95}.
            Reports SHD / precision / recall / fp_edges + per-edge-type recall.

  grid_null (Review item 3b) -- per-test size under the heavy_tail null for each grid.
            The 5.2 heavy_tail calibration null at dS=3, GFCM only, one test per (grid, n, rep).
            Reports the raw p-value and the reject indicator (size = mean reject at alpha).

  p20       (Review items 4+5) -- larger-p PC discovery for the scalability + ACAT-accumulation ask.
            20-node sparse tailgraph (edge_prob=0.105 -> expected degree ~2), GFCM vs BLITZ / RCoT /
            FFCI. Reports SHD / precision / recall / fp_edges + per-query wall time.
            NOTE: run this mode SERIALLY (JOBS=1) if you want clean wall times; SHD/precision/FP are
            unaffected by contention.

GFCM's quantile grid is the only thing varied here, via GFCMConfig(taus=...). Everything else is the
canonical published configuration. Competitor tests (p20) need the Docker environment.

Usage:
  python experiments/run_rebuttal.py --mode grid_pc                 # GFCM only, no Docker backend
  python experiments/run_rebuttal.py --mode grid_null              # GFCM only, no Docker backend
  python experiments/run_rebuttal.py --mode p20                    # needs Docker (R/JRE/citests)
  python experiments/run_rebuttal.py --mode grid_pc --reps 2 --n 2000   # smoke
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # experiments/ : dgp, adapters, discovery, metrics, gfcm_citest
sys.path.insert(0, os.path.dirname(_HERE))      # repo root     : the gfcm package

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "XGB_NTHREAD"):
    os.environ.setdefault(_v, "1")

import argparse

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters
import dgp
import discovery as _disc
import metrics as _metrics
from gfcm import GFCMConfig

OUT = os.environ.get("REBUTTAL_OUT") or os.path.join(os.path.dirname(_HERE), "data", "rebuttal")
SEED0 = 20000        # distinct from run_pc's SEED0=10000 so the two never share a seed stream
ALPHA = 0.05
MAX_COND = 3

GRIDS = {
    "median":  (0.5,),
    "tails":   (0.1, 0.9),
    "default": (0.1, 0.5, 0.9),
    "dense5":  (0.1, 0.3, 0.5, 0.7, 0.9),
    "wide5":   (0.05, 0.25, 0.5, 0.75, 0.95),
}
NS = (2000, 10000)                 # n = 2e3 and 1e4 (skip 1e5 unless cheap; add it to NS if wanted)
GRID_PC_REPS   = 100
GRID_NULL_REPS = 500               # matches the calibration cells' rep count
GRID_NULL_DS   = 3
P20_P          = 20
P20_EDGE_PROB  = 0.105             # expected degree ~2 on a 20-node lower-triangular random DAG
P20_PANEL      = ("gfcm", "blitz", "rcot", "ffci")
P20_REPS       = 100               # heavy with competitors at n=1e4; lower with --reps if needed


def _claim(outp):
    """Atomic claim; return True if this worker owns the (already-nonexistent) output."""
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        return False
    try:
        fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return False
    return True


def _typed_dag(p, edge_prob, seed, n):
    """Simulate one typed DAG at sample size n; return (data, nodes, true_skeleton, etype)."""
    cfg, etype = dgp.random_typed_dag(p, edge_prob, seed_structure=seed, seed_data=seed, types=dgp.TYPES_TAIL)
    cfg["simulation_params"]["n_samples"] = n
    df = CausalDataGenerator(cfg).simulate()["data"]
    true_sk = {frozenset((a, b)) for a, b in cfg["graph_params"]["edges"]}
    return df.to_numpy(float), list(df.columns), true_sk, etype


def _pc_scores(nm, data, nodes, true_sk, etype, ci):
    """Run PC with a prebuilt CI test, return the skeleton score row (SHD/precision/recall/fp + types)."""
    cg = _disc.run_pc(data, nodes, ci, ALPHA, MAX_COND)
    learned = _disc.cpdag_skeleton(cg, nodes)
    sc = _metrics.skeleton_scores(true_sk, learned, nodes)
    sc.update({f"recall_{t}": v for t, v in _metrics.per_type_recall(etype, learned).items()})
    return sc


def rep_grid_pc(gridlabel, taus, n, it):
    key = f"rebuttal_gridpc__{gridlabel}__n{n}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    data, nodes, true_sk, etype = _typed_dag(8, 0.4, SEED0 + it, n)
    ci = adapters.make_test("gfcm", data, config=GFCMConfig(taus=tuple(taus)))
    sc = _pc_scores("gfcm", data, nodes, true_sk, etype, ci)
    sc.update(dict(test="gfcm", grid=gridlabel, taus=str(taus), n=n, rep=it, p=8, edge_prob=0.4))
    pd.DataFrame([sc]).to_parquet(outp, index=False)
    print(f"grid_pc {gridlabel} n{n} rep {it}: SHD={sc['shd']} prec={sc['precision']:.2f}", flush=True)


def rep_grid_null(gridlabel, taus, n, it):
    key = f"rebuttal_gridnull__{gridlabel}__n{n}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    cell = dgp.calib_nulls(n, seed=SEED0 + it, dS=GRID_NULL_DS)["heavy_tail"]
    df = CausalDataGenerator(cell).simulate()["data"]
    cols = list(df.columns)
    xi, yi = cols.index("X"), cols.index("Y")
    Si = [cols.index(f"Z{i}") for i in range(1, GRID_NULL_DS + 1)]
    ci = adapters.make_test("gfcm", df.to_numpy(float), config=GFCMConfig(taus=tuple(taus)))
    p = float(ci(xi, yi, Si))
    row = dict(test="gfcm", grid=gridlabel, taus=str(taus), n=n, rep=it,
               cell="heavy_tail", dS=GRID_NULL_DS, pval=p, reject=int(p < ALPHA))
    pd.DataFrame([row]).to_parquet(outp, index=False)
    print(f"grid_null {gridlabel} n{n} rep {it}: p={p:.3f}", flush=True)


def rep_p20(n, it, panel_tests):
    key = f"rebuttal_p20__n{n}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    data, nodes, true_sk, etype = _typed_dag(P20_P, P20_EDGE_PROB, SEED0 + it, n)
    rows = []
    for nm in panel_tests:
        t0 = time.perf_counter()
        try:
            ci = adapters.make_test(nm, data)
            sc = _pc_scores(nm, data, nodes, true_sk, etype, ci)
            sc.update(dict(test=nm, n=n, rep=it, p=P20_P, edge_prob=P20_EDGE_PROB,
                           wall_s=time.perf_counter() - t0, config_id=adapters.config_id(nm)))
            rows.append(sc)
        except NotImplementedError:
            raise
        except Exception as e:
            rows.append(dict(test=nm, n=n, rep=it, p=P20_P, error=str(e)[:200],
                             wall_s=time.perf_counter() - t0))
    pd.DataFrame(rows).to_parquet(outp, index=False)
    print(f"p20 n{n} rep {it}: {len(rows)} tests", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["grid_pc", "grid_null", "p20"])
    ap.add_argument("--n", type=int, help="restrict to one sample size")
    ap.add_argument("--reps", type=int, help="override rep count (smoke)")
    ap.add_argument("--only", help="p20 only: comma-separated subset of the panel (e.g. gfcm)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ns = (args.n,) if args.n else NS

    if args.mode == "grid_pc":
        reps = args.reps or GRID_PC_REPS
        for gl, taus in GRIDS.items():
            for n in ns:
                for it in range(reps):
                    rep_grid_pc(gl, taus, n, it)
    elif args.mode == "grid_null":
        reps = args.reps or GRID_NULL_REPS
        for gl, taus in GRIDS.items():
            for n in ns:
                for it in range(reps):
                    rep_grid_null(gl, taus, n, it)
    else:  # p20
        reps = args.reps or P20_REPS
        panel = list(P20_PANEL)
        if args.only:
            keep = set(args.only.split(","))
            panel = [t for t in panel if t in keep]
        for n in ns:
            for it in range(reps):
                rep_p20(n, it, panel)
    print(f"REBUTTAL {args.mode} DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
