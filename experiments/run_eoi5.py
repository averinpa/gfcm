"""Review 2 (Eoi5) experiments for the TMLR revision: modes ci, pc, oracle, envelope, ladderadd."""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "XGB_NTHREAD"):
    os.environ.setdefault(_v, "1")

import argparse

import bnmetrics
import cbcd
import networkx as nx
import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters
import dgp
import discovery as _disc
import metrics as _metrics
from discovery import DSepOracle as _Oracle, true_cpdag as _true_cpdag
from metrics import cpdag_scores as _cpdag_scores
from gfcm import GFCMConfig, infer_types, unified_test

# one place for the eoi5 cells: aggregate_eoi5.py and tables/make_tables_eoi5.py both read it.
# REBUTTAL_OUT still moves the whole tree (the server runs set it); run_rebuttal.py keeps the parent.
OUT = os.path.join(os.environ.get("REBUTTAL_OUT")
                   or os.path.join(os.path.dirname(_HERE), "data", "rebuttal"), "eoi5")
SEED0_CI = 30000     # disjoint from run_pc (10000) and run_rebuttal (20000)
SEED0_ENV = 40000    # the operating-envelope grid
SEED0_PC = 20000     # SAME as run_rebuttal p20, so the p = 20 skeleton numbers reproduce
ALPHA = 0.05
MAX_COND = 3          # cap on |S| inside PC; --max-cond 0 removes it (uncapped)
NS = (2000, 10000)
CI_REPS = 500
PC_REPS = 100
COMPETITORS = ("rcot", "ffci", "blitz")
PC_PANEL = ("gfcm", "gfcm_comove", "blitz", "rcot", "ffci")
COMOVE = GFCMConfig(comove=True)

# (family, dS list, full panel?) in priority order
CI_PLAN = [
    ("evenscale", (0, 3), True),
    ("calib", (10, 20), True),
    ("ladder", (10, 20), True),
    ("calib", (1, 3, 5), False),
    ("ladder", (1, 3, 5), False),
    # truncated-basis stress test: a fixed Z1*Z2 interaction while the conditioning set grows.
    ("interdepth", (2, 5, 10, 20), True),
    # (A5) violated: skewness of Y|Z varies with Z1. GFCM only; it is about our own assumption.
    ("misspec", (1, 3), False),
]


def _claim(outp):
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        return False
    try:
        fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return False
    return True


def _cells(family, n, seed, dS):
    if family == "evenscale":
        return dgp.evenscale_cells(n, seed, dS)
    if family == "calib":
        return dgp.calib_nulls(n, seed, dS)
    if family == "ladder":
        return dgp.power_ladder(n, seed, dS)
    if family == "ladderadd":
        return dgp.power_ladder_additive(n, seed, dS)
    if family == "interdepth":
        return dgp.interaction_at_depth(n, seed, dS)
    if family == "misspec":
        return dgp.misspec_nulls(n, seed, dS)
    raise ValueError(family)


def _cell_names(family, dS):
    return list(_cells(family, 50, 1, dS).keys())


ENV_NS = (1000, 2000, 5000, 10000, 20000, 50000, 100000)
ENV_DS = (1, 2, 3, 5, 8, 10, 15, 20)
ENV_CELLS = [("calib", "heavy_tail"), ("calib", "hetero"), ("calib", "nonlin_mean"),
             ("calib", "mixed_Z"), ("ladder", "null"),
             ("interdepth", "additive"), ("interdepth", "interaction")]
# as submitted (the submitted CODE: no configuration of gfcm/ reproduces it) | exact corrected
ENV_ARMS = {"published": adapters.submitted_test, "corrected": GFCMConfig(varcorr_decide=None)}


def env_reps(n):
    """Fewer replications where a call is expensive; 500 gives a Wilson half-width of about 0.02 at 0.05."""
    return 2000 if n <= 5000 else 1000 if n <= 20000 else 300


def rep_envelope(family, cell, dS, n, it):
    key = f"eoi5_env__{family}__{cell}__dS{dS}__n{n}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    cfg = _cells(family, n, SEED0_ENV + it, dS)[cell]
    cfg["simulation_params"]["store_ci_oracle"] = False
    df = CausalDataGenerator(cfg).simulate()["data"]
    cols = list(df.columns)
    data = df.to_numpy(float)
    xi, yi = cols.index("X"), cols.index("Y")
    Si = [cols.index(f"Z{i}") for i in range(1, dS + 1)]
    is_cat = infer_types(data)
    rows = []
    for nm, conf in ENV_ARMS.items():
        t0 = time.perf_counter()
        fn, conf = (conf, adapters.PUBLISHED_CONFIG) if callable(conf) else (unified_test, conf)
        p = float(fn(data[:, xi], data[:, yi], data[:, Si], config=conf,
                     x_cat=bool(is_cat[xi]), y_cat=bool(is_cat[yi]),
                     z_cat=(is_cat[Si] if Si else None)))
        rows.append(dict(family=family, cell=cell, dS=dS, n=n, rep=it, arm=nm, pval=p,
                         wall_s=time.perf_counter() - t0))
    pd.DataFrame(rows).to_parquet(outp, index=False)
    print(f"env {key} rep {it}: " + " ".join(f"{r['arm']}={r['pval']:.3f}" for r in rows), flush=True)


def rep_ci(family, cell, dS, n, it, full):
    key = f"eoi5_ci__{family}__{cell}__dS{dS}__n{n}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    cfg = _cells(family, n, SEED0_CI + it, dS)[cell]
    # The oracle is deterministic and the truth is known by construction, so this changes no data.
    cfg["simulation_params"]["store_ci_oracle"] = False
    df = CausalDataGenerator(cfg).simulate()["data"]
    cols = list(df.columns)
    data = df.to_numpy(float)
    xi, yi = cols.index("X"), cols.index("Y")
    Si = [cols.index(f"Z{i}") for i in range(1, dS + 1)]
    base = dict(family=family, cell=cell, dS=dS, n=n, rep=it, seed=SEED0_CI + it)
    rows = []

    t0 = time.perf_counter()
    is_cat = infer_types(data)
    d = unified_test(data[:, xi], data[:, yi], data[:, Si], x_cat=bool(is_cat[xi]), y_cat=bool(is_cat[yi]),
                     z_cat=(is_cat[Si] if Si else None), detail=True)
    rows.append({**base, "test": "gfcm", "pval": d["p"], "wall_s": time.perf_counter() - t0, **d})
    t0 = time.perf_counter()                      # the corrected test (no detail path; one p-value)
    pv = float(unified_test(data[:, xi], data[:, yi], data[:, Si], config=adapters.EXACT, x_cat=bool(is_cat[xi]),
                            y_cat=bool(is_cat[yi]), z_cat=(is_cat[Si] if Si else None)))
    rows.append({**base, "test": "gfcm_vc", "pval": pv, "wall_s": time.perf_counter() - t0})

    if full:
        for nm in COMPETITORS:
            t0 = time.perf_counter()
            try:
                p = float(adapters.make_test(nm, data)(xi, yi, Si))
                rows.append({**base, "test": nm, "pval": p, "wall_s": time.perf_counter() - t0})
            except NotImplementedError:
                raise
            except Exception as e:
                rows.append({**base, "test": nm, "error": str(e)[:200], "wall_s": time.perf_counter() - t0})
    pd.DataFrame(rows).to_parquet(outp, index=False)
    print(f"ci {key} rep {it}: gfcm p={d['p']:.3f} acat4={d['acat4']:.3f}", flush=True)


# ---------------------------------------------------------------- PC with CPDAG scoring







def rep_oracle(p, deg, it, cap, ep=None):
    """PC with a d-separation oracle: the cap's cost when the CI test is perfect."""
    ep = deg / (p - 1) if ep is None else ep
    key = f"eoi5_oracle__p{p}__ep{ep:g}__cap{cap if cap else 'none'}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    cfg, etype = dgp.random_typed_dag(p, ep, seed_structure=SEED0_PC + it, seed_data=SEED0_PC + it,
                                      types=dgp.TYPES_TAIL)
    nodes = cfg["graph_params"]["nodes"]
    true_sk = {frozenset((a, b)) for a, b in cfg["graph_params"]["edges"]}
    idx = {v: i for i, v in enumerate(nodes)}
    G = nx.DiGraph(); G.add_nodes_from(range(len(nodes)))
    G.add_edges_from((idx[a], idx[b]) for a, b in cfg["graph_params"]["edges"])
    log = []
    cg = cbcd.pc(np.zeros((4, len(nodes))), ci_test=_Oracle(G, len(nodes), log), alpha=ALPHA,
                 max_cond_set=cap, var_names=list(nodes))
    sc = _metrics.skeleton_scores(true_sk, _disc.cpdag_skeleton(cg, nodes), nodes)
    sc.update({f"recall_{t}": v for t, v in _metrics.per_type_recall(etype, nodes and _disc.cpdag_skeleton(cg, nodes)).items()})
    sc.update(_cpdag_scores(_true_cpdag(cfg["graph_params"]["edges"], nodes), cg))
    sc.update(dict(test="oracle", p=p, deg=deg, edge_prob=ep, rep=it, cap=(cap or 0),
                   n_queries=len(log), max_S=max(log) if log else 0))
    pd.DataFrame([sc]).to_parquet(outp, index=False)
    print(f"oracle {key} rep {it}: shd={sc['shd']} fp={sc['fp_edges']} queries={len(log)}", flush=True)


def rep_pc(p, deg, n, it, panel, max_cond=MAX_COND, ep=None):
    ep = deg / (p - 1) if ep is None else ep
    key = f"eoi5_pc__p{p}__deg{deg}__n{n}" + ("" if max_cond == MAX_COND else f"__cap{max_cond or 'none'}")
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    cfg, etype = dgp.random_typed_dag(p, ep, seed_structure=SEED0_PC + it, seed_data=SEED0_PC + it,
                                      types=dgp.TYPES_TAIL)
    cfg["simulation_params"]["n_samples"] = n
    df = CausalDataGenerator(cfg).simulate()["data"]
    data, nodes = df.to_numpy(float), list(df.columns)
    true_sk = {frozenset((a, b)) for a, b in cfg["graph_params"]["edges"]}
    true_cg = _true_cpdag(cfg["graph_params"]["edges"], nodes)
    rows = []
    for nm in panel:
        t0 = time.perf_counter()
        try:
            ci = (adapters.make_test("gfcm", data, config=COMOVE) if nm == "gfcm_comove"
                  else adapters.make_test(nm, data))
            log = []
            cg = _disc.run_pc(data, nodes, ci, ALPHA, max_cond, log=log)
            wall = time.perf_counter() - t0
            learned = _disc.cpdag_skeleton(cg, nodes)
            sc = _metrics.skeleton_scores(true_sk, learned, nodes)
            sc.update({f"recall_{t}": v for t, v in _metrics.per_type_recall(etype, learned).items()})
            sc.update(_cpdag_scores(true_cg, cg))
            sc.update(dict(test=nm, n=n, rep=it, p=p, deg=deg, edge_prob=ep, n_queries=len(log), wall_s=wall,
                           cap=(max_cond or 0), max_S=max((len(q[2]) for q in log), default=0),
                           config_id=(adapters.config_id("gfcm", COMOVE)
                                      if nm == "gfcm_comove" else adapters.config_id(nm))))
            rows.append(sc)
        except NotImplementedError:
            raise
        except Exception as e:
            rows.append(dict(test=nm, n=n, rep=it, p=p, deg=deg, edge_prob=ep, cap=(max_cond or 0),
                             error=str(e)[:200],
                             wall_s=time.perf_counter() - t0))
    pd.DataFrame(rows).to_parquet(outp, index=False)
    print(f"pc {key} rep {it}: " + ", ".join(f"{r['test']} cpdag_shd={r.get('cpdag_shd')}" for r in rows),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["ci", "pc", "oracle", "envelope", "ladderadd"])
    ap.add_argument("--family", help="ci: restrict to one family (evenscale / calib / ladder)")
    ap.add_argument("--dS", type=int, help="ci: restrict to one conditioning size")
    ap.add_argument("--n", type=int, help="restrict to one sample size")
    ap.add_argument("--reps", type=int, help="override rep count")
    ap.add_argument("--p", type=int, default=20, help="pc: number of nodes")
    ap.add_argument("--deg", type=int, default=2, help="pc: expected degree (edge_prob = deg/(p-1))")
    ap.add_argument("--only", help="pc: comma-separated subset of the panel")
    ap.add_argument("--max-cond", type=int, default=MAX_COND,
                    help="pc/oracle: cap on |S| inside PC; 0 = uncapped")
    ap.add_argument("--edge-prob", type=float,
                    help="pc/oracle: edge probability, overriding deg/(p-1) (the p=8 cells use 0.4)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ns = (args.n,) if args.n else NS

    if args.mode == "ci":
        reps = args.reps or CI_REPS
        for family, dSs, full in CI_PLAN:
            if args.family and family != args.family:
                continue
            for dS in dSs:
                if args.dS is not None and dS != args.dS:
                    continue
                for n in ns:
                    for cell in _cell_names(family, dS):
                        for it in range(reps):
                            rep_ci(family, cell, dS, n, it, full)
    elif args.mode == "ladderadd":
        # additive-mean power ladder (satisfies (A1)); full panel; grid fixed here, --n/--dS/--reps narrow it
        LA_NS = (2000, 10000, 50000); LA_DS = {2000: (1, 3, 5, 8, 10), 10000: (1, 3, 5, 8, 10), 50000: (8, 15)}
        for n in (ns if args.n else LA_NS):
            for dS in (LA_DS[n] if args.dS is None else (args.dS,)):
                for cell in _cell_names("ladderadd", dS):
                    for it in range(args.reps or CI_REPS):
                        rep_ci("ladderadd", cell, dS, n, it, True)
    elif args.mode == "envelope":
        for n in (ns if args.n else ENV_NS):
            for dS in (ENV_DS if args.dS is None else (args.dS,)):
                for family, cell in ENV_CELLS:
                    if family == "interdepth" and dS < 2:
                        continue                      # the Z1*Z2 mean needs two conditioners
                    if args.family and family != args.family:
                        continue
                    for it in range(args.reps or env_reps(n)):
                        rep_envelope(family, cell, dS, n, it)
    elif args.mode == "oracle":
        reps = args.reps or PC_REPS
        for cap in (MAX_COND, None):
            for it in range(reps):
                rep_oracle(args.p, args.deg, it, cap, args.edge_prob)
    else:
        reps = args.reps or PC_REPS
        panel = list(PC_PANEL)
        if args.only:
            keep = set(args.only.split(","))
            panel = [t for t in panel if t in keep]
        cap = args.max_cond or None
        for n in ns:
            for it in range(reps):
                rep_pc(args.p, args.deg, n, it, panel, cap, args.edge_prob)
    print(f"EOI5 {args.mode} DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
