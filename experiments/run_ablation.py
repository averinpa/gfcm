"""Ablation-appendix tables, config-based (no environment variables):

  asym       (tab:asym)             -- asymmetric (eX*r_tau(Y), r_tau(X)*eY) vs symmetric
                                       (r_tau(X)*r_tau(Y)) quantile products, marginal |S|=0.
  interaction(tab:interaction)      -- GFCM size on an additive null vs an interaction (Z1*Z2) null.
  tradeoff   (tab:nuisance-tradeoff)-- complementary failure of the two nuisance blocks: poly-only
                                       (proj off) removes polynomial interactions but detonates on a
                                       transcendental mean; index-only (inter off, single-index
                                       ungated) does the reverse.
  rankz      (tab:rankz)            -- rank-Z robustness to heavy-tailed conditioners: GFCM null
                                       size with rankz on vs off across conditioner distributions.

The kit produced these by flipping GFCM_* env vars and a separate single-index module; here every
variant is a GFCMConfig. asym/interaction use the paper's dagsampler DGPs (dgp.tab_asym /
dgp.tab_interaction); tradeoff/rankz use the reconstructed diagnostic DGPs described in the paper.

Usage:  python experiments/run_ablation.py [asym interaction tradeoff rankz]   # default: all
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import argparse

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import dgp
from gfcm import GFCMConfig, unified_test
from gfcm.core import _cauchy, _mvgcm_p, _qind

CANON = GFCMConfig()
OUT = os.path.join(os.path.dirname(_HERE), "results", "aggregated")
ALPHA = 0.05
TAUS_ASYM = (0.1, 0.9)


def _save(df, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"ablation_{name}.parquet")
    df.to_parquet(p, index=False)
    print(df.to_string(index=False))
    print(f"wrote {p}\n")


# ---------------------------------------------------------------- asym
def _asym_sym_p(X, Y, cfg, B=199, seed=0):
    X = np.asarray(X, float); Y = np.asarray(Y, float); n = len(X)
    eX = X - X.mean(); eY = Y - Y.mean()
    RY = np.column_stack([_qind(Y, np.quantile(Y, t), t, cfg) for t in TAUS_ASYM])
    RX = np.column_stack([_qind(X, np.quantile(X, t), t, cfg) for t in TAUS_ASYM])
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(B, n))
    p_asym = _cauchy([_mvgcm_p(eX[:, None] * RY, signs, cfg), _mvgcm_p(RX * eY[:, None], signs, cfg)])
    p_sym = _mvgcm_p(RX * RY, signs, cfg)                    # symmetric quantile-quantile product
    return p_asym, p_sym


def asym(reps, n=2000):
    cells = {"null": lambda s: dgp.indep_null(n, s),
             "scale": lambda s: dgp.tab_asym(n=n, seed=s)["scale"],
             "directed_tailshape": lambda s: dgp.tab_asym(n=n, seed=s)["directed_tailshape"],
             "symmetric_comovement": lambda s: dgp.tab_asym(n=n, seed=s)["symmetric_comovement"]}
    rows = []
    for name, mk in cells.items():
        ca = cs = 0
        for it in range(reps):
            df = CausalDataGenerator(mk(2000 + it)).simulate()["data"]
            pa, ps = _asym_sym_p(df["X"].to_numpy(float), df["Y"].to_numpy(float), CANON, seed=it)
            ca += int(pa < ALPHA); cs += int(ps < ALPHA)
        rows.append(dict(dgp=name, asymmetric=ca / reps, symmetric=cs / reps))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- interaction
def interaction(reps, n=2000):
    rows = []
    for cell in ("additive", "interaction"):
        c = 0
        for it in range(reps):
            cfg = dgp.tab_interaction(n=n, seed=2000 + it)[cell]
            df = CausalDataGenerator(cfg).simulate()["data"]
            p = unified_test(df["X"].to_numpy(float), df["Y"].to_numpy(float),
                             df[["Z1", "Z2"]].to_numpy(float), config=CANON, seed=it)
            c += int(p < ALPHA)
        rows.append(dict(null=cell, size=c / reps))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- nuisance-tradeoff
def _gen_tradeoff(kind, n, seed):
    rng = np.random.RandomState(seed); Z = rng.randn(n, 3)
    if kind == "additive":
        X = Z[:, 0] + 0.5 * Z[:, 1] + rng.randn(n); Y = Z[:, 1] + 0.5 * Z[:, 2] + rng.randn(n)
    elif kind == "interaction":
        m = Z[:, 0] * Z[:, 1]; X = m + rng.randn(n); Y = m + rng.randn(n)
    else:  # transcendental
        s = np.sin(1.2 * Z.sum(1)); X = s + rng.randn(n); Y = s + rng.randn(n)
    return X, Y, Z


def tradeoff(reps):
    poly_only = GFCMConfig(proj=False)                          # polynomial block, no single index
    index_only = GFCMConfig(inter=False, proj=True, proj_nmin=1)  # single index, no poly, ungated
    rows = []
    for kind in ("additive", "interaction", "transcendental"):
        for n in (2000, 10000):
            kp = sum(unified_test(*_gen_tradeoff(kind, n, 1000 + s), config=poly_only, seed=s) < ALPHA
                     for s in range(reps)) / reps
            ki = sum(unified_test(*_gen_tradeoff(kind, n, 1000 + s), config=index_only, seed=s) < ALPHA
                     for s in range(reps)) / reps
            rows.append(dict(null=kind, n=n, poly_only=kp, index_only=ki))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- rankz
_CONDITIONERS = {
    "gaussian":  lambda r, n: r.normal(size=n),
    "t5":        lambda r, n: r.standard_t(5, size=n),
    "t2":        lambda r, n: r.standard_t(2, size=n),
    "cauchy":    lambda r, n: r.standard_cauchy(size=n),
    "lognormal": lambda r, n: r.lognormal(size=n),
    "power":     lambda r, n: r.pareto(3.0, size=n),
}


def _rankz_null(dist, n, seed):
    r = np.random.default_rng(seed)
    Z = _CONDITIONERS[dist](r, n)
    m = np.sin(Z)                                              # nonlinear: raw-range knots starve the bulk
    return m + r.normal(size=n), m + r.normal(size=n), Z[:, None]   # X _||_ Y | Z


def rankz(reps, n=2000):
    on, off = GFCMConfig(rankz=True), GFCMConfig(rankz=False)
    rows = []
    for dist in _CONDITIONERS:
        r_on = sum(unified_test(*_rankz_null(dist, n, 2000 + s), config=on, seed=s) < ALPHA
                   for s in range(reps)) / reps
        r_off = sum(unified_test(*_rankz_null(dist, n, 2000 + s), config=off, seed=s) < ALPHA
                    for s in range(reps)) / reps
        rows.append(dict(conditioner=dist, size_rankz_on=r_on, size_rankz_off=r_off))
    return pd.DataFrame(rows)


TABLES = {"asym": asym, "interaction": interaction, "tradeoff": tradeoff, "rankz": rankz}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="*", choices=list(TABLES), default=list(TABLES))
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    want = args.tables or list(TABLES)
    for t in want:
        print(f"=== tab:{t} (reps={args.reps}) ===")
        _save(TABLES[t](args.reps), t)


if __name__ == "__main__":
    main()
