"""GFCM against the contingency-table tests on NATIVE DISCRETE data, where G^2 needs no discretisation and is
the efficient tool. X, Y and every Z are categorical; Z is a common cause of both; the alternative adds X -> Y.

The informative axis is the number of conditioning STRATA, KZ^|S|, because a contingency-table test splits the
sample across them while a regression-based test does not: at n = 2000 with three-level conditioners, |S| = 1 gives
667 rows per stratum and |S| = 4 gives 25. Both tests are at their best at |S| = 1, so the question the experiment
answers is what each gives up as the conditioning set grows.

Usage: python experiments/run_discrete.py [--ns 2000,10000] [--ds 1,2,3,4] [--reps 500] [--strength 0.15]
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "R_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("GRID_THREADS", "1"))

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters
import dgp

OUT = os.environ.get("DISC_OUT") or os.path.join(os.path.dirname(_HERE), "data", "discrete")
SEED0 = 980000
PANEL = ("gfcm", "gsq", "chisq", "blitz", "rcot", "ffci")


def _claim(outp):
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        return False
    try:
        fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return False
    return True


def rep(cell, dS, n, K, KZ, strength, it, panel):
    key = f"disc__{cell}__dS{dS}__n{n}__K{K}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    cfg = dgp.discrete_cells(n, SEED0 + it, dS=dS, K=K, KZ=KZ, strength=strength)[cell]
    cfg["simulation_params"]["store_ci_oracle"] = False
    df = CausalDataGenerator(cfg).simulate()["data"]
    c = list(df.columns); d = df.to_numpy(float)
    xi, yi = c.index("X"), c.index("Y"); Si = [c.index(f"Z{i}") for i in range(1, dS + 1)]
    rows = []
    for nm in panel:
        t0 = time.perf_counter()
        try:
            if nm == "gfcm":
                from gfcm import GFCMConfig, unified_test
                pv = float(unified_test(d[:, xi], d[:, yi], d[:, Si], config=GFCMConfig(),
                                        x_cat=True, y_cat=True, z_cat=[True] * dS))
            else:
                pv = float(adapters.make_test(nm, d)(xi, yi, Si))
            st = "ok" if not np.isnan(pv) else "nan"
        except NotImplementedError:
            raise
        except Exception as e:
            pv, st = float("nan"), f"error: {str(e)[:140]}"
        rows.append(dict(cell=cell, dS=dS, n=n, K=K, KZ=KZ, strata=KZ ** dS, rows_per_stratum=n // (KZ ** dS),
                         rep=it, test=nm, pval=pv, status=st, wall_s=time.perf_counter() - t0))
    pd.DataFrame(rows).to_parquet(outp, index=False)
    print(f"{key} rep {it}: " + " ".join(f"{r['test']}={r['pval']:.3f}" for r in rows if r["status"] == "ok"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2000,10000")
    ap.add_argument("--ds", default="1,2,3,4")
    ap.add_argument("--ks", default="3")
    ap.add_argument("--kz", type=int, default=3)
    ap.add_argument("--strength", type=float, default=0.15)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--only")
    ap.add_argument("--out")
    a = ap.parse_args()
    global OUT
    if a.out: OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    panel = [t for t in PANEL if (not a.only or t in a.only.split(","))]
    ns = [int(x) for x in a.ns.split(",")]; ds = [int(x) for x in a.ds.split(",")]; ks = [int(x) for x in a.ks.split(",")]
    print(f"native-discrete comparison: {panel} x n {ns} x |S| {ds} x K {ks} (KZ={a.kz}, strength {a.strength}) "
          f"x {a.reps} reps -> {OUT}", flush=True)
    for n in ns:
        for dS in ds:
            for K in ks:
                for cell in ("null", "alt"):
                    for it in range(a.reps):
                        rep(cell, dS, n, K, a.kz, a.strength, it, panel)
    print("DISCRETE DONE", flush=True)


if __name__ == "__main__":
    main()
