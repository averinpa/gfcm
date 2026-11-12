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
DRAWS = int(os.environ.get("SPEED_DRAWS", "5"))   # independent null draws averaged per cell
SLOW = set(os.environ.get("SPEED_SLOW", "kci,cmiknn_mixed,pcm,wgcm,gkcm,gcm_boosted,par_cop,ci_mm").split(","))   # one draw per cell


def _timeit(mk, x, y, S):
    """Per-query wall time in ms, UNCACHED: rebuild the (cheap, lazy) test object each rep so any
    result cache never hits -- inside PC each new query is a cache miss, so the uncached compute is
    the cost that matters -- then report the median over adaptive reps."""
    obj = mk(); t0 = time.perf_counter(); float(obj(x, y, S)); el = time.perf_counter() - t0
    reps = 1 if el > 5 else (3 if el > 0.5 else 8)
    if el * 1000 > BUDGET_MS: return el * 1000.0
    times = [el]
    for _ in range(reps - 1):
        obj = mk(); t0 = time.perf_counter(); float(obj(x, y, S)); times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0


BUDGET_MS = float(os.environ.get("SPEED_BUDGET_MS", "120000"))   # a test whose query exceeds this is not timed at larger n / deeper |S|


def run(panel, ns, ds):
    rows = []; exhausted = {}                                  # test -> (n, d) of the first cell over budget
    # global warm-up: trigger JIT / JVM / R-init once per test so the grid times compute only
    warm = np.random.RandomState(0).randn(600, 5)
    for nm in panel:
        try:
            adapters.make_test(nm, warm)(0, 1, [2, 3])
        except Exception:
            pass
    for n in ns:
        for d in ds:
            # several independent null draws per cell: a lazily evaluated test's cost depends on the draw (it computes
            # the correction only when the uncorrected p lands at or below its band), so one draw would be a coin flip
            draws = []
            for k in range(DRAWS):
                rng = np.random.RandomState(n * 100 + d + 7919 * k)
                Z = rng.randn(n, d); X = Z[:, 0] + rng.randn(n); Y = Z[:, d - 1] + rng.randn(n)
                draws.append((np.column_stack([X, Y, Z]), list(range(2, 2 + d))))
            for nm in panel:
                ex = exhausted.get(nm)
                if ex is not None and n >= ex[0] and (n > ex[0] or d >= ex[1]):
                    rows.append(dict(test=nm, n=n, S=d, ms=float("nan"))); print(f"  {nm:12s} n={n:<7d} |S|={d:<3d}   over budget (skipped)", flush=True); continue
                try:
                    ms = float(np.mean([_timeit(lambda nm=nm, data=data: adapters.make_test(nm, data), 0, 1, S) for data, S in draws[:(1 if nm in SLOW else DRAWS)]]))
                except NotImplementedError:
                    raise
                except Exception:
                    ms = float("nan")
                if ms > BUDGET_MS: exhausted[nm] = (n, d)
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
