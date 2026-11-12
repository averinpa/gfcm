"""PC discovery with the CONSERVATIVE (majority-rule) collider orientation, uncapped, for every graph size the
paper reports: p = 8, 20 and 50.

Why: the arrowhead trace showed that GFCM's wrong arrowheads are false colliders produced by a single recorded
separating set, not by miscalibration. The majority rule re-tests every conditioning subset of an unshielded
triple and orients a collider only when the middle node lies in fewer than half of the separating subsets,
leaving the rest unoriented (Colombo and Maathuis 2014). It is order-independent and costs many more CI calls.

Both orientations are run on the SAME data so the difference is attributable to the rule alone. Metrics are the
ones the paper reports: skeleton SHD, precision, recall and false-positive edges; CPDAG SHD and arrowhead
precision against the true CPDAG; per-edge-type recall; and the median CI-query count and wall time.

Usage: python experiments/run_pc_cons.py --p 8|20|50 [--n 2000,10000] [--reps N] [--only gfcm,blitz]
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "R_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("GRID_THREADS", "1"))

import networkx as nx
import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters
import dgp
import discovery as _disc
import metrics as _metrics
from discovery import true_cpdag as _true_cpdag
from metrics import cpdag_scores as _cpdag_scores

OUT = os.environ.get("PCC_OUT") or os.path.join(os.path.dirname(_HERE), "data", "pc_cons")
SEED0 = 20000                      # same stream as run_eoi5's PC cells, so the graphs match the paper
ALPHA = 0.05
PANEL = ("gfcm", "blitz", "rcot", "ffci", "gcm_boosted")
# the paper's graph settings: p=8 uses edge probability .4; p=20 and 50 use expected degree 2 (and 4 at p=20)
SETTINGS = {8: [dict(ep=0.4, deg=None)], 20: [dict(ep=None, deg=2), dict(ep=None, deg=4)], 50: [dict(ep=None, deg=2)]}
REPS = {8: 100, 20: 100, 50: 50}


def _claim(outp):
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        return False
    try:
        fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return False
    return True


def rep(p, deg, ep, n, it, panel, colliders):
    tag = f"p{p}__" + (f"ep{ep:g}" if ep is not None else f"deg{deg}") + f"__n{n}"
    outp = os.path.join(OUT, f"pccons__{tag}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    e = ep if ep is not None else deg / (p - 1)
    cfg, etype = dgp.random_typed_dag(p, e, seed_structure=SEED0 + it, seed_data=SEED0 + it, types=dgp.TYPES_TAIL)
    cfg["simulation_params"]["n_samples"] = n
    cfg["simulation_params"]["store_ci_oracle"] = False
    df = CausalDataGenerator(cfg).simulate()["data"]
    data, nodes = df.to_numpy(float), list(df.columns)
    true_sk = {frozenset((a, b)) for a, b in cfg["graph_params"]["edges"]}
    true_cg = _true_cpdag(cfg["graph_params"]["edges"], nodes)
    rows = []
    for nm in panel:
        for coll in colliders:
            t0 = time.perf_counter()
            try:
                ci = adapters.make_test(nm, data)
                log = []
                cg = _disc.run_pc(data, nodes, ci, ALPHA, None, log=log, collider=coll)   # None = NO cap
                wall = time.perf_counter() - t0
                learned = _disc.cpdag_skeleton(cg, nodes)
                sc = _metrics.skeleton_scores(true_sk, learned, nodes)
                sc.update({f"recall_{t}": v for t, v in _metrics.per_type_recall(etype, learned).items()})
                sc.update(_cpdag_scores(true_cg, cg))
                sc.update(dict(test=nm, collider=coll, n=n, rep=it, p=p, deg=deg, edge_prob=e,
                               n_queries=len(log), wall_s=wall, max_S=max((len(q[2]) for q in log), default=0)))
                rows.append(sc)
            except NotImplementedError:
                raise
            except Exception as ex:
                rows.append(dict(test=nm, collider=coll, n=n, rep=it, p=p, deg=deg, edge_prob=e,
                                 error=str(ex)[:200], wall_s=time.perf_counter() - t0))
    pd.DataFrame(rows).to_parquet(outp, index=False)
    print(f"pccons {tag} rep {it}: " + ", ".join(
        f"{r['test']}/{r['collider']} shd={r.get('shd')} cpdag={r.get('cpdag_shd')}" for r in rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True, choices=[8, 20, 50])
    ap.add_argument("--ns", default="2000")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--only")
    ap.add_argument("--colliders", default="sepset,majority")
    ap.add_argument("--out")
    a = ap.parse_args()
    global OUT
    if a.out: OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    panel = [t for t in PANEL if (not a.only or t in a.only.split(","))]
    colliders = a.colliders.split(",")
    reps = a.reps or REPS[a.p]
    ns = [int(x) for x in a.ns.split(",")]
    print(f"conservative PC, p={a.p}, UNCAPPED: {panel} x {colliders} x n {ns} x {reps} reps -> {OUT}", flush=True)
    for n in ns:
        for st in SETTINGS[a.p]:
            for it in range(reps):
                rep(a.p, st["deg"], st["ep"], n, it, panel, colliders)
    print(f"PCCONS p={a.p} DONE", flush=True)


if __name__ == "__main__":
    main()
