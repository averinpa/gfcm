"""Convergence sweep: run the cells of interest over a FINE n-grid (and conditioning depth
dS in {1,3,5}) to see WHERE calibration and power converge -- n=2000 and n=1e5 are deliberate
endpoints that hide the transition. Same DGP family at each n (fixed structure seed, growing
sample), so it is a clean 'same problem, more data' sweep.

Reuses run_cells.run_cell, so every cell stores per-rep seed + p-value + config_id and is
test-by-test resumable; size-corrected power at each n is then pure post-processing (aggregate.py).
GFCM runs at its canonical GFCMConfig; competitors are deferred to the Docker workstream
(run with --only gfcm). Output goes to data/cells_conv/ (gitignored).

Usage:
  python experiments/run_convergence.py --only gfcm [--reps 500] [--ds 1,3,5] [--ngrid ...]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # experiments/ : run_cells, dgp, adapters, ...
sys.path.insert(0, os.path.dirname(_HERE))      # repo root     : the gfcm package

import argparse

import dgp
import run_cells as RC

# (section, builder, [cell names]) -- include the matched 'null' so size-corrected power is
# computable. Each family is swept over the depth grid and the n-grid.
INTEREST = [
    ("5.3_power", dgp.power_ladder,    ["tail_shape", "null"]),                 # centerpiece washout
    ("5.4_tail",  dgp.tail_edge_sweep, ["alpha_3", "null"]),                    # tail-edge washout
    ("5.2_calib", dgp.calib_nulls,     ["heavy_tail", "hetero", "mixed_Z"]),    # calibration onset
]
DEFAULT_NGRID = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000,
                 7000, 10000, 15000, 20000, 50000, 100000]


def _zq(dS):
    return ("X", "Y", [f"Z{i}" for i in range(1, dS + 1)])


def _make_cell(section, builder, dS, cellname, n, nrep):
    build = (lambda b, nn, ds, cl: (lambda seed: b(nn, seed, ds)[cl]))(builder, n, dS, cellname)
    return {"key": f"{section}__{cellname}__n{n}__dS{dS}", "kind": "ci",
            "build": build, "query": _zq(dS), "section": section, "cell": cellname,
            "n": n, "dS": dS, "n_iter": nrep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated test names (e.g. gfcm)")
    ap.add_argument("--panel", default="gfcm", help="comma-separated panel for every cell")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--ds", default="1,3,5", help="comma-separated conditioning depths")
    ap.add_argument("--ngrid", help="comma-separated sample sizes (defaults to the fine 14-point grid)")
    args = ap.parse_args()

    RC.OUT = os.environ.get("CONV_OUT") or os.path.join(os.path.dirname(_HERE), "data", "cells_conv")
    os.makedirs(RC.OUT, exist_ok=True)

    grid = [int(x) for x in args.ngrid.split(",")] if args.ngrid else DEFAULT_NGRID
    ds_grid = [int(x) for x in args.ds.split(",")]
    panel = args.panel.split(",")
    only = set(args.only.split(",")) if args.only else None
    if only:
        panel = [t for t in panel if t in only]

    # cheap cells first (small n), so the transition fills in early
    units = [_make_cell(sec, b, dS, nm, n, args.reps)
             for sec, b, names in INTEREST for dS in ds_grid for n in grid for nm in names]
    units.sort(key=lambda c: (c["n"], c["dS"], c["section"], c["cell"]))
    print(f"convergence: {len(INTEREST)} families x {len(ds_grid)} depths x {len(grid)} n-values, "
          f"panel {panel} = {len(units)} cell-runs -> {RC.OUT}", flush=True)
    for cell in units:
        RC.run_cell(cell, panel, args.reps)
    print("CONV DONE", flush=True)


if __name__ == "__main__":
    main()
