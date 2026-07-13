"""Runtime grid behind fig:speed -- per-CI-query wall time across conditioning depth |S| and
sample size n, for each test in the panel.

The data-bound test object is built once, then a single (x, y, S) query is timed (warm-up +
adaptive reps), which is the cost that matters inside PC. GFCM runs locally; the R (BLITZ, par_cop)
and Java (FFCI) competitors need the Docker image.

Usage:  python experiments/run_speed.py [--panel gfcm,rcot,...] [--ns 2000,10000] [--ds 2,5,20]
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import argparse

import numpy as np
import pandas as pd

import adapters

OUT = os.path.join(_ROOT, "results", "aggregated")


def _timeit(obj, x, y, S):
    """Per-query wall time in ms: warm up, then adaptively repeat so cheap calls average out."""
    def one():
        return float(obj(x, y, S))
    one()                                            # warm (JIT, JVM, first-fit caches)
    t0 = time.perf_counter(); one(); el = time.perf_counter() - t0
    reps = 1 if el > 5 else (3 if el > 0.5 else 8)
    if reps > 1:
        t0 = time.perf_counter()
        for _ in range(reps):
            one()
        el = (time.perf_counter() - t0) / reps
    return el * 1000.0


def run(panel, ns, ds):
    rows = []
    for n in ns:
        for d in ds:
            rng = np.random.RandomState(n * 100 + d)
            Z = rng.randn(n, d); X = Z[:, 0] + rng.randn(n); Y = Z[:, d - 1] + rng.randn(n)
            data = np.column_stack([X, Y, Z]); S = list(range(2, 2 + d))
            for nm in panel:
                try:
                    ms = _timeit(adapters.make_test(nm, data), 0, 1, S)
                except NotImplementedError:
                    raise
                except Exception:
                    ms = float("nan")
                rows.append(dict(test=nm, n=n, S=d, ms=ms))
                print(f"  {nm:12s} n={n:<7d} |S|={d:<3d} {ms:9.1f} ms", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="gfcm", help="comma-separated test names")
    ap.add_argument("--ns", default="2000,5000,10000,100000", help="sample sizes")
    ap.add_argument("--ds", default="2,5,20,50", help="conditioning depths |S|")
    args = ap.parse_args()
    panel = args.panel.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    ds = [int(x) for x in args.ds.split(",")]
    print(f"speed grid: {panel} x n{ns} x |S|{ds}")
    df = run(panel, ns, ds)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "speed_grid.parquet")
    df.to_parquet(p, index=False)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
