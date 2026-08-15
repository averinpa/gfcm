"""Semi-synthetic real-data injection (tab:inject).

Noise / marginals are REAL heavy-tailed EU stock-index return innovations (EuStockMarkets,
excess kurtosis ~4.26); the X->Y edge is INJECTED with known ground truth as a mean- AND
covariance-preserving upper-tail (skew) kick, so mean/covariance/scale tests are blind by
construction and only a tail-sensitive test recovers it. Per method:
  level = reject rate with NO edge (X _||_ Y | Z true)  -> want ~0.05
  power = reject rate with the tail edge injected        -> the claim

The EuStock innovations are computed from the vendored raw prices (datasets/eustockmarkets.csv):
per-series daily log-returns, standardized, pooled. GFCM runs locally; competitor columns
(par_cop, blitz need R) come online in the Docker image.

Usage:  python experiments/run_injection.py [--panel gfcm,rcot,gcm_boosted,fisherz] [--reps 120]

Note: the second injection substrate in the paper (tab:cc-inject, the Causal-Chamber light tunnel)
is NOT yet ported -- its runner (injection_sc.py) and the chamber dataset download are outstanding.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import argparse

import numpy as np
import pandas as pd

import adapters

OUT = os.path.join(_ROOT, "results", "aggregated")
ALPHA = 0.05


def eu_innov():
    """Pooled standardized EuStockMarkets return innovations (real heavy-tailed marginals)."""
    m = np.loadtxt(os.path.join(_ROOT, "datasets", "eustockmarkets.csv"), delimiter=",", skiprows=1)
    lr = np.diff(np.log(m), axis=0)                 # daily log-returns per index
    z = (lr - lr.mean(0)) / lr.std(0)               # standardize each series
    return z.reshape(-1)                            # pool -> 1-D innovation bank


def make(rng, pool, n, dz, c, inject):
    """Semi-synthetic (X, Y, Z): real innovations + optional SKEW-NORMAL tail edge X->Y.

    X drives the conditional SKEW of Y via a skew-normal shape, with conditional mean AND variance
    held flat EXACTLY per row (Var(es)=1 for any marginal, because es mixes two independent
    standardized draws with coefficients sqrt(1-delta^2) and delta). So mean/covariance/scale tests
    are blind by construction and only a skew/tail-sensitive test recovers the edge -- this is the
    construction behind tab:inject. (The earlier Fleishman variant es=(eY+d(eY^2-1)/sqrt2)/sqrt(1+d^2)
    only preserves variance for Gaussian eY; on heavy-tailed innovations its eY^2 term leaks a
    detectable scale edge, so it does NOT match the paper.)"""
    a_mean = np.mean(np.abs(pool)); a_std = np.std(np.abs(pool))
    Z = rng.choice(pool, size=(n, dz), replace=True)
    w = rng.normal(size=dz) / np.sqrt(dz); v = rng.normal(size=dz) / np.sqrt(dz)
    eX = rng.choice(pool, size=n); X = Z @ w + eX
    Xc = eX                                          # Z-orthogonal part of X
    b = rng.choice(pool, size=n); u = rng.choice(pool, size=n)
    au = (np.abs(u) - a_mean) / a_std                # |u| centered/scaled to mean 0, var 1
    alpha = (c * Xc) if inject else np.zeros(n)
    delta = alpha / np.sqrt(1.0 + alpha ** 2)
    es = np.sqrt(1.0 - delta ** 2) * b + delta * au  # mean 0, var 1 exactly, skew ~ delta(Xc)
    Y = Z @ v + es
    return np.column_stack([X, Y, Z])                # cols: 0=X, 1=Y, 2..=Z


def run(panel, n=2000, dz=3, c=4.0, R=200, alpha=ALPHA, seed=0):
    """Emit one row per (test, cell, rep, pval). Retaining the raw p-values (not just reject
    counts) lets the table size-correct each test to exact 5% on its matched null -- the metric
    tab:inject reports -- rather than raw alpha=0.05 power."""
    pool = eu_innov()
    rng = np.random.default_rng(seed)
    rows = []
    S = list(range(2, 2 + dz))
    for r in range(R):
        for inject in (False, True):
            data = make(rng, pool, n, dz, c, inject)
            for nm in panel:
                try:
                    p = float(adapters.make_test(nm, data)(0, 1, S))
                except NotImplementedError:
                    raise
                except Exception:
                    p = float("nan")
                rows.append(dict(test=nm, cell=("alt" if inject else "null"), rep=r, pval=p))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="gfcm", help="comma-separated test names")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--c", type=float, default=4.0, help="skew-normal edge strength")
    args = ap.parse_args()
    panel = args.panel.split(",")
    print(f"tab:inject -- EuStock skew-normal tail-edge injection (n={args.n}, c={args.c}, reps={args.reps})")
    df = run(panel, n=args.n, c=args.c, R=args.reps)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "injection_eustock.parquet")
    df.to_parquet(p, index=False)
    # quick console summary: raw size + size-corrected power per test
    for nm in panel:
        nu = df[(df.test == nm) & (df.cell == "null")].pval.dropna().values
        al = df[(df.test == nm) & (df.cell == "alt")].pval.dropna().values
        if len(nu) and len(al):
            crit = np.percentile(nu, 100 * ALPHA)
            print(f"  {nm:12s} size={ (nu<ALPHA).mean():.3f}  sc-power={ (al<crit).mean():.3f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
