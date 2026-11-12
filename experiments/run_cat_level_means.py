"""The two mixed-type property experiments of the resubmission.

  --exp labelinv   M1. The p-value of a test that handles a nominal variable natively cannot depend on how its
                   levels happen to be numbered. One dataset per replication, LABELS re-drawn R_PERM times
                   (identity first), every test run on each labelling. Reports, per test, the spread of the
                   p-value over labellings and whether the decision at alpha can change with the numbering.

  --exp cardinality  M2. Categorical X with K levels whose level means alternate in sign with the between-level
                   variance held fixed, so the signal is constant in K and only its spread over levels changes.
                   Four arms: GFCM pooled (the K-1 residual directions in one Wald statistic), GFCM per level with
                   Bonferroni (same test, same data, only the combination differs -- this isolates pooling),
                   every competitor on the integer code as shipped, and every competitor per level with Bonferroni
                   over the dummies, which is the best a scalar-input test can do natively.

One parquet per cell x replication; claim files let any number of workers share the queue.
Usage: python experiments/run_cat_level_means.py --exp labelinv|cardinality [--reps N] [--only t1,t2] [--ns ...] [--ks ...]
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
_T = os.environ.get("GRID_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "XGB_NTHREAD", "R_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters
import dgp

OUT = os.environ.get("MIXED_OUT") or os.path.join(os.path.dirname(_HERE), "data", "mixed")
SEED0 = 700000
# par_cop is excluded from the mixed-type experiments: it cannot take a categorical variable. In the main grid it
# returns NaN on every draw with a categorical conditioner, and given a categorical X (integer code or 0/1 dummy)
# its R backend spins without returning -- measured at 99.9 percent CPU for nearly ten hours on a single call. That
# is reported as "cannot run" in the mixed tables rather than hidden.
PANEL = ("gfcm", "blitz", "rcot", "ffci", "gcm_boosted")
R_PERM = 20                      # label re-drawings per replication (the first is the identity)
# Alternative strength, calibrated per sample size so every cell has headroom (at n=2000, delta .08 gives GFCM
# pooled .79 at K=3 and .60 at K=12, with the matched null at .05; .12 already saturates). Power scales with sqrt(n),
# so the larger n carries the proportionally smaller effect.
DELTA_BY_N = {1000: 0.115, 2000: 0.08, 5000: 0.051, 10000: 0.036, 20000: 0.025}


def delta_for(n):
    return float(os.environ.get("MIXED_DELTA") or DELTA_BY_N.get(n, 0.08 * (2000.0 / n) ** 0.5))


def _claim(outp):
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        return False
    try:
        fd = os.open(outp + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        return False
    return True


def _draw(n, seed, K, dS, delta, cat_in_Z=False):
    cfg = dgp.cat_level_means(n, seed, K=K, dS=dS, delta=delta, cat_in_Z=cat_in_Z)
    cfg["simulation_params"]["store_ci_oracle"] = False
    df = CausalDataGenerator(cfg).simulate()["data"]
    c = list(df.columns); d = df.to_numpy(float)
    return d[:, c.index("X")], d[:, c.index("Y")], d[:, [c.index(f"Z{i}") for i in range(1, dS + 1)]], c


def _p(name, X, Y, Z, x_cat=False, z_cat=None):
    """One p-value from one test, with the categorical flag only GFCM can use."""
    data = np.column_stack([X, Y, Z]); S = list(range(2, 2 + Z.shape[1]))
    if name == "gfcm":
        from gfcm import GFCMConfig, unified_test
        return float(unified_test(X, Y, Z, config=GFCMConfig(), x_cat=x_cat,
                                  z_cat=(np.array(z_cat) if z_cat is not None else None)))
    return float(adapters.make_test(name, data)(0, 1, S))


def _dummies(X, K):
    """K-1 indicator columns (level 0 is the reference)."""
    return [(X == j).astype(float) for j in range(1, K)]


def rep_labelinv(K, cat_in_Z, delta, n, it, panel):
    tag = "Zcat" if cat_in_Z else "Xcat"
    key = f"mix_labelinv__{tag}__K{K}__d{delta}__n{n}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    X, Y, Z, _ = _draw(n, SEED0 + it, K, 1, delta, cat_in_Z)
    rng = np.random.default_rng(90000 + it)
    rows = []
    for g in range(R_PERM):
        perm = np.arange(K) if g == 0 else rng.permutation(K)
        if cat_in_Z:
            Xg, Zg = X, np.column_stack([perm[Z[:, 0].astype(int)].astype(float)] + [Z[:, j] for j in range(1, Z.shape[1])])
            kw = dict(x_cat=False, z_cat=[True] + [False] * (Z.shape[1] - 1))
        else:
            Xg, Zg = perm[X.astype(int)].astype(float), Z
            kw = dict(x_cat=True, z_cat=None)
        for nm in panel:
            t0 = time.perf_counter()
            try:
                pv = _p(nm, Xg, Y, Zg, **kw); st = "ok" if not np.isnan(pv) else "nan"
            except NotImplementedError:
                raise
            except Exception as e:
                pv, st = float("nan"), f"error: {str(e)[:120]}"
            rows.append(dict(exp="labelinv", tag=tag, K=K, delta=delta, n=n, rep=it, perm=g,
                             test=nm, pval=pv, status=st, wall_s=time.perf_counter() - t0))
    pd.DataFrame(rows).to_parquet(outp, index=False)
    d = pd.DataFrame(rows)
    sp = d[d.status == "ok"].groupby("test").pval.agg(lambda s: s.max() - s.min())
    print(f"{key} rep {it}: spread " + " ".join(f"{k}={v:.1e}" for k, v in sp.items()), flush=True)


def rep_cardinality(K, n, delta, dS, it, panel):
    key = f"mix_card__K{K}__n{n}__d{delta}__dS{dS}"
    outp = os.path.join(OUT, f"{key}__rep{it:04d}.parquet")
    if not _claim(outp):
        return
    X, Y, Z, _ = _draw(n, SEED0 + 200000 + it, K, dS, delta)
    base = dict(exp="cardinality", K=K, n=n, delta=delta, dS=dS, rep=it)
    rows = []
    dums = _dummies(X, K)
    for nm in panel:
        # arm 1/3: the test as shipped -- GFCM on the categorical, the others on the integer code
        t0 = time.perf_counter()
        try:
            pv = _p(nm, X, Y, Z, x_cat=(nm == "gfcm")); st = "ok" if not np.isnan(pv) else "nan"
        except NotImplementedError:
            raise
        except Exception as e:
            pv, st = float("nan"), f"error: {str(e)[:120]}"
        rows.append({**base, "test": nm, "arm": "pooled" if nm == "gfcm" else "code", "pval": pv, "status": st,
                     "wall_s": time.perf_counter() - t0})
        # arm 2/4: per level, Bonferroni over the K-1 dummies
        t0 = time.perf_counter(); ps = []
        try:
            for dcol in dums:
                ps.append(_p(nm, dcol, Y, Z, x_cat=(nm == "gfcm")))
            pv2 = float(min(1.0, (K - 1) * np.nanmin(ps))) if len(ps) else float("nan")
            st2 = "ok" if not np.isnan(pv2) else "nan"
        except NotImplementedError:
            raise
        except Exception as e:
            pv2, st2 = float("nan"), f"error: {str(e)[:120]}"
        rows.append({**base, "test": nm, "arm": "perlevel_bonf", "pval": pv2, "status": st2,
                     "wall_s": time.perf_counter() - t0})
    pd.DataFrame(rows).to_parquet(outp, index=False)
    d = pd.DataFrame(rows); ok = d[d.status == "ok"]
    print(f"{key} rep {it}: " + " ".join(f"{r.test}/{r.arm}={r.pval:.3f}" for r in ok.itertuples()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["labelinv", "cardinality"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--only", help="comma-separated subset of the panel")
    ap.add_argument("--ks", default="3,5,8,12")
    ap.add_argument("--ns", default=None)
    ap.add_argument("--ds", default="1")
    ap.add_argument("--out")
    a = ap.parse_args()
    global OUT
    if a.out: OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    panel = [t for t in PANEL if (not a.only or t in a.only.split(","))]
    ks = [int(x) for x in a.ks.split(",")]
    if a.exp == "labelinv":
        reps = a.reps or 100
        ns = [int(x) for x in (a.ns or "2000").split(",")]
        print(f"M1 label invariance: panel {panel} x K {ks} x n {ns} x {R_PERM} labellings x {reps} reps -> {OUT}", flush=True)
        for n in ns:
            for K in ks:
                for cat_in_Z in (False, True):
                    for delta in (0.0, delta_for(n)):
                        for it in range(reps):
                            rep_labelinv(K, cat_in_Z, delta, n, it, panel)
    else:
        reps = a.reps or 500
        ns = [int(x) for x in (a.ns or "2000,10000").split(",")]
        print(f"M2 cardinality: panel {panel} x K {ks} x n {ns} x delta 0/" + ",".join(f"{delta_for(n):.3f}" for n in ns) + f" x {reps} reps -> {OUT}", flush=True)
        for n in ns:
            for dS in [int(x) for x in a.ds.split(",")]:
                for K in ks:
                    for delta in (0.0, delta_for(n)):
                        for it in range(reps):
                            rep_cardinality(K, n, delta, dS, it, panel)
    print(f"MIXED {a.exp} DONE", flush=True)


if __name__ == "__main__":
    main()
