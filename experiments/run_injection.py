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


def make(rng, pool, n, dz, gamma, inject):
    """Semi-synthetic (X, Y, Z): real innovations + optional mean/cov-preserving tail edge X->Y."""
    Z = rng.choice(pool, size=(n, dz), replace=True)
    w = rng.normal(size=dz) / np.sqrt(dz); v = rng.normal(size=dz) / np.sqrt(dz)
    eX = rng.choice(pool, size=n); eY = rng.choice(pool, size=n)
    X = Z @ w + eX
    Xc = eX                                          # Z-orthogonal part of X
    # Fleishman-style skew injection: X drives Y's skewness, mean AND variance preserved.
    d = (gamma * np.tanh(Xc)) if inject else np.zeros(n)
    es = (eY + d * (eY ** 2 - 1) / np.sqrt(2)) / np.sqrt(1 + d ** 2)
    Y = Z @ v + es
    return np.column_stack([X, Y, Z])                # cols: 0=X, 1=Y, 2..=Z


def run(panel, n=2000, dz=3, gamma=1.5, R=120, alpha=ALPHA, seed=0):
    pool = eu_innov()
    rng = np.random.default_rng(seed)
    acc = {nm: {"null": 0, "alt": 0} for nm in panel}
    S = list(range(2, 2 + dz))
    for _ in range(R):
        for inject in (False, True):
            data = make(rng, pool, n, dz, gamma, inject)
            for nm in panel:
                try:
                    p = float(adapters.make_test(nm, data)(0, 1, S))
                    acc[nm]["alt" if inject else "null"] += int(p < alpha)
                except NotImplementedError:
                    raise
                except Exception:
                    pass
    return pd.DataFrame([dict(method=nm, level=acc[nm]["null"] / R, power=acc[nm]["alt"] / R)
                         for nm in panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="gfcm", help="comma-separated test names")
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=1.5)
    args = ap.parse_args()
    panel = args.panel.split(",")
    print(f"tab:inject -- EuStock tail-edge injection (n={args.n}, gamma={args.gamma}, reps={args.reps})")
    df = run(panel, n=args.n, gamma=args.gamma, R=args.reps)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "injection_eustock.parquet")
    df.to_parquet(p, index=False)
    print(df.to_string(index=False))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
