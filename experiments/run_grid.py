"""The two main grids of the resubmission on one footing: T2 (level on the four hard nulls) and T3 (power on the four
additive alternatives with their matched null), every test of the panel on the same data, one parquet per
(table, cell, |S|, n, replication). Claim files let any number of workers (processes or containers) share the work.

Panel: gfcm (the default test), blitz, rcot, ffci, gcm_boosted, par_cop.
Inclusion rule, one line: a test runs in a cell when one call at that n and |S| takes under a minute in the speed
grid (notes/speed_grid_v6_fullpanel_docker.parquet). That excludes only par_cop above n = 5000 and at n = 10000
beyond |S| = 5. A GFCM call outside its domain returns NaN, which the tables print as "outside domain".

Usage: python experiments/run_grid.py --table null|power [--ns 2000,10000,100000] [--ds 1,2,3,5,8,10,15,20]
                                      [--reps 500] [--only gfcm,blitz] [--out DIR]
Threads: GRID_THREADS (default 1) sets the BLAS / OpenMP pool of the worker; GFCM caps itself below that, never above.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

_T = os.environ.get("GRID_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "XGB_NTHREAD", "R_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters
import dgp

OUT = os.environ.get("GRID_OUT") or os.path.join(os.path.dirname(_HERE), "data", "grid")
SEED0 = 50000                                   # disjoint from run_pc (10000), run_rebuttal (20000), eoi5 ci (30000), envelope (40000)
NS = (1000, 2000, 5000, 10000, 20000, 50000, 100000)
DS = (1, 2, 3, 5, 8, 10, 15, 20)
REPS = 500
PANEL = ("gfcm", "blitz", "rcot", "ffci", "gcm_boosted", "par_cop")
# "gfcm_published" is the submitted configuration (variance correction off) on the SAME seeds, so the effect of the
# correction is read cell by cell from paired runs rather than from the published tables.
PANEL_EXTRA = ("gfcm_published",)
FAMILY = {"null": "calib", "power": "ladderadd"}


def runs_here(test, n, dS):
    """The one-minute rule read off the speed grid (64-thread competitors, GFCM capped at 16)."""
    if test == "par_cop":
        return n <= 5000 or (n == 10000 and dS <= 5)
    return True


def _claim(outp):
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        return False
    try:
        fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return False
    return True


def _cells(table, n, seed, dS):
    return dgp.calib_nulls(n, seed, dS) if table == "null" else dgp.power_ladder_additive(n, seed, dS)


def rep(table, cell, dS, n, it, panel):
    key = f"grid_{table}__{cell}__dS{dS}__n{n}" + ("__pub" if panel == ["gfcm_published"] else "")
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    cfg = _cells(table, n, SEED0 + it, dS)[cell]
    cfg["simulation_params"]["store_ci_oracle"] = False
    df = CausalDataGenerator(cfg).simulate()["data"]
    cols = list(df.columns); data = df.to_numpy(float)
    xi, yi = cols.index("X"), cols.index("Y"); Si = [cols.index(f"Z{i}") for i in range(1, dS + 1)]
    base = dict(table=table, cell=cell, dS=dS, n=n, rep=it, seed=SEED0 + it)
    rows = []
    for nm in panel:
        if not runs_here(nm, n, dS):
            rows.append({**base, "test": nm, "pval": float("nan"), "status": "not run", "wall_s": float("nan")}); continue
        t0 = time.perf_counter()
        try:
            p = float(adapters.make_test(nm, data)(xi, yi, Si))
            st = "ok" if not np.isnan(p) else ("outside domain" if nm == "gfcm" else "nan")   # only GFCM refuses by design
            rows.append({**base, "test": nm, "pval": p, "status": st, "wall_s": time.perf_counter() - t0})
        except NotImplementedError:
            raise                                    # backend absent: this worker cannot run the panel at all
        except Exception as e:
            rows.append({**base, "test": nm, "pval": float("nan"), "status": f"error: {str(e)[:160]}", "wall_s": time.perf_counter() - t0})
    pd.DataFrame(rows).to_parquet(outp, index=False)
    print(f"{key} rep {it}: " + " ".join(f"{r['test']}={r['pval']:.3f}" for r in rows if r["status"] == "ok"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, choices=["null", "power"])
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--ds", default=",".join(map(str, DS)))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--only", help="comma-separated subset of the panel")
    ap.add_argument("--cells", help="comma-separated subset of the cells (e.g. mixed_Z)")
    ap.add_argument("--out", help="output directory (default GRID_OUT or data/grid)")
    args = ap.parse_args()
    global OUT
    if args.out: OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    panel = [t for t in PANEL + PANEL_EXTRA if (not args.only or t in args.only.split(","))]
    ns = [int(x) for x in args.ns.split(",")]; ds = [int(x) for x in args.ds.split(",")]
    print(f"grid {args.table}: panel {panel} x n {ns} x |S| {ds} x {args.reps} reps -> {OUT}", flush=True)
    for n in ns:
        for dS in ds:
            keep = set(args.cells.split(",")) if args.cells else None
            for cell in _cells(args.table, 50, 1, dS).keys():
                if keep and cell not in keep: continue
                for it in range(args.reps):
                    rep(args.table, cell, dS, n, it, panel)
    print(f"GRID {args.table} DONE", flush=True)


if __name__ == "__main__":
    main()
