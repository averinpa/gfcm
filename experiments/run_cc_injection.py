"""Second injection substrate (tab:cc-inject): scale injection on real Causal-Chamber structure.

The light tunnel exposes a real bipartite structure -- the three light sources {red, green, blue}
drive the sensor sinks {ir_1, ir_2, ir_3} (9 source->sink edges); each sink's other parents (its
own l_* light-position sensors) differ per sink, so given the RGB sources the sinks are
conditionally independent. We run PC over the 6-node subsystem in two variants:

  real           -- the real light-tunnel data (covariance tests recover structure via the mean),
  scale-injected -- each source->sink edge is turned into a MEAN-preserving SCALE edge (real
                    sources, real per-sink noise, only the edge type changed), so a covariance
                    test sees Cov(source, sink) ~ 0 and deletes the edge (recall -> 0), while a
                    tail-sensitive test recovers it.

Reports per test: skeleton SHD and recall vs the true 9-edge skeleton, for both variants.

Requires the `causalchamber` package (downloads the light-tunnel dataset on first run). GFCM +
citests-family competitors run locally; BLITZ/FFCI need the Docker image.

NOTE: a reconstruction -- the original injection script was not preserved. The subsystem and the
mean-preserving scale transform follow the paper's description; exact SHD/recall values differ.

Usage:  python experiments/run_cc_injection.py [--panel gfcm,rcot,gcm_boosted,fisherz] [--reps 20]
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
import discovery
import metrics

OUT = os.path.join(_ROOT, "results", "aggregated")
SOURCES = ["red", "green", "blue"]
SINKS = ["ir_1", "ir_2", "ir_3"]
NODES = SOURCES + SINKS
TRUE_EDGES = {frozenset((s, k)) for s in SOURCES for k in SINKS}   # 9-edge bipartite skeleton


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def load_lt(n=2000):
    """Standardized light-tunnel columns for the 6-node subsystem."""
    import causalchamber.datasets as dd
    os.makedirs("/tmp/cc_data", exist_ok=True)
    d = dd.Dataset("lt_walks_v1", root="/tmp/cc_data", download=True)
    df = d.get_experiment("actuators_white").as_pandas_dataframe()
    idx = np.random.default_rng(0).choice(len(df), min(n, len(df)), replace=False)
    return {v: _z(df[v].to_numpy()[idx]) for v in NODES}


def inject(base, gamma, rng):
    """Turn each RGB->sink mean edge into a mean-preserving scale edge (Cov(source, sink) ~ 0)."""
    P = np.column_stack([base[s] for s in SOURCES])
    out = {s: base[s] for s in SOURCES}                    # keep the real sources
    A = np.column_stack([np.ones(len(P)), P])
    for k in SINKS:
        Y = base[k]
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)       # remove the real (linear) mean edge
        eY = Y - A @ coef                                  # real per-sink noise
        u = _z(P @ rng.normal(size=len(SOURCES)))          # a source combination
        out[k] = _z(eY) * np.exp(gamma * u)                # source drives the SCALE, mean ~flat
    return out


def _score(dmap, nm, alpha, max_cond):
    data = np.column_stack([dmap[v] for v in NODES])
    ci = adapters.make_test(nm, data)
    cg = discovery.run_pc(data, NODES, ci, alpha, max_cond)
    learned = discovery.cpdag_skeleton(cg, NODES)
    return metrics.skeleton_scores(TRUE_EDGES, learned, NODES)


def run(panel, gamma=1.5, reps=20, alpha=0.05, max_cond=3, seed=0):
    base = load_lt()
    rows = []
    for nm in panel:
        real = _score(base, nm, alpha, max_cond)           # real data is fixed -> single run
        sh, rc = [], []
        for it in range(reps):
            d = inject(base, gamma, np.random.default_rng(seed + it))
            s = _score(d, nm, alpha, max_cond)
            sh.append(s["shd"]); rc.append(s["recall"])
        rows.append(dict(test=nm, real_shd=real["shd"], real_recall=real["recall"],
                         inj_shd=float(np.mean(sh)), inj_recall=float(np.mean(rc))))
        print(f"  {nm:12s} real SHD={real['shd']} recall={real['recall']:.2f} | "
              f"injected SHD={np.mean(sh):.1f} recall={np.mean(rc):.2f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="gfcm,rcot,gcm_boosted,fisherz")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=1.5)
    args = ap.parse_args()
    print(f"tab:cc-inject -- light-tunnel scale injection (reps={args.reps}, gamma={args.gamma})")
    df = run(args.panel.split(","), gamma=args.gamma, reps=args.reps)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "injection_chamber.parquet")
    df.to_parquet(p, index=False)
    print(df.to_string(index=False)); print(f"wrote {p}")


if __name__ == "__main__":
    main()
