"""The deciding ablation for the co-movement block: is the extra block free on level and on power, and what does
it buy where it is supposed to help?

Three arms on the SAME draw (paired, so a difference is never a difference in data):
  default      the shipped three blocks
  scale        + a fourth block holding cX.cY alone               (one coordinate)
  full         + a fourth block holding cX.cY and the matched tail corners rX.rY   (three coordinates)

Three families:
  null    the four calibration nulls            -> what the extra block costs in level
  power   the four additive alternatives        -> what it costs in power (Cauchy dilution)
  covol   a shared latent scale driver with the means conditioned away, and its matched null
          -> what it buys, the case every default product cancels on

One parquet per cell x replication; claim files let any number of workers share the queue.
Usage: python experiments/run_comove.py --family null|power|covol [--ns ...] [--ds ...] [--reps N]
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("GRID_THREADS", "1"))

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import dgp
from gfcm import GFCMConfig, unified_test

OUT = os.environ.get("COMOVE_OUT") or os.path.join(os.path.dirname(_HERE), "data", "comove")
SEED0 = 900000
ARMS = {"default": GFCMConfig(varcorr_decide=None),
        "scale": GFCMConfig(varcorr_decide=None, comove=True, comove_parts=("scale",)),
        "full": GFCMConfig(varcorr_decide=None, comove=True, comove_parts=("scale", "tail"))}


def _claim(outp):
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        return False
    try:
        fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return False
    return True


def _cells(family, n, seed, dS):
    if family == "null":
        return dgp.calib_nulls(n, seed, dS)
    if family == "power":
        return dgp.power_ladder_additive(n, seed, dS)
    return dgp.comove_cells(n, seed, dS)


def rep(family, cell, dS, n, it):
    key = f"cmv_{family}__{cell}__dS{dS}__n{n}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    cfg = _cells(family, n, SEED0 + it, dS)[cell]
    cfg["simulation_params"]["store_ci_oracle"] = False
    df = CausalDataGenerator(cfg).simulate()["data"]
    c = list(df.columns); d = df.to_numpy(float)
    X, Y = d[:, c.index("X")], d[:, c.index("Y")]
    Z = d[:, [c.index(f"Z{i}") for i in range(1, dS + 1)]]          # latents U, V are never conditioned on
    rows = []
    for lab, conf in ARMS.items():
        t0 = time.perf_counter()
        try:
            pv = float(unified_test(X, Y, Z, config=conf)); st = "ok" if not np.isnan(pv) else "outside domain"
        except Exception as e:
            pv, st = float("nan"), f"error: {str(e)[:140]}"
        rows.append(dict(family=family, cell=cell, dS=dS, n=n, rep=it, arm=lab, pval=pv, status=st,
                         wall_s=time.perf_counter() - t0))
    pd.DataFrame(rows).to_parquet(outp, index=False)
    print(f"{key} rep {it}: " + " ".join(f"{r['arm']}={r['pval']:.3f}" for r in rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["null", "power", "covol"])
    ap.add_argument("--ns", default="2000,10000")
    ap.add_argument("--ds", default="1,3,5,10")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out")
    a = ap.parse_args()
    global OUT
    if a.out: OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    ns = [int(x) for x in a.ns.split(",")]; ds = [int(x) for x in a.ds.split(",")]
    print(f"co-movement ablation, family {a.family}: arms {list(ARMS)} x n {ns} x |S| {ds} x {a.reps} reps -> {OUT}", flush=True)
    for n in ns:
        for dS in ds:
            for cell in _cells(a.family, 50, 1, dS).keys():
                for it in range(a.reps):
                    rep(a.family, cell, dS, n, it)
    print(f"COMOVE {a.family} DONE", flush=True)


if __name__ == "__main__":
    main()
