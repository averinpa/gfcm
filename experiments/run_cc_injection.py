"""Second injection substrate (tab:cc-inject): scale injection on real Causal-Chamber structure.

The light tunnel exposes a real bipartite structure -- the three light sources {red, green, blue}
drive the sensor sinks {current, vis_1, ir_1} (9 source->sink edges) in the `color_mix` experiment
of lt_walks_v1; given the RGB sources the sinks are conditionally independent (the hard part: the
sinks are marginally highly correlated but conditionally independent given the sources). We run PC
over the 6-node subsystem in two variants:

  real           -- the real light-tunnel data (covariance tests recover structure via the mean),
  scale-injected -- each source->sink edge is turned into a MEAN-preserving SCALE edge: a
                    RandomForest removes the real conditional mean, and the sources drive the
                    residual's SPREAD via s=exp(gamma*mean(sources)). Cov(source,sink)~0, so a
                    covariance test deletes the edge (recall -> 0) while a tail/scale-sensitive
                    test recovers it. Real sources and real per-sink noise are kept; only the edge
                    TYPE changes, and the skeleton is unchanged.

Reports per test: skeleton SHD and recall vs the true 9-edge skeleton, for both variants (single
run, so SHD is an integer and recall is k/9 -- matching the paper table).

This is the faithful port of the original archive/quantile_gcm/cc_scale_inject.py construction
(color_mix subsystem, sinks current/vis_1/ir_1, gamma=0.7, RF-residual scale injection), driven by
the current GFCM/competitor adapters. Requires the `causalchamber` package (downloads lt_walks_v1
on first run); GFCM runs locally, BLITZ/FFCI/GCM/RCoT need the Docker image.

Usage:  python experiments/run_cc_injection.py [--panel fisherz,gcm_linear,rcot,gfcm] [--gamma 0.7]
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
SINKS = ["current", "vis_1", "ir_1"]
NODES = SOURCES + SINKS
TRUE_EDGES = {frozenset((s, k)) for s in SOURCES for k in SINKS}   # 9-edge bipartite skeleton


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def load_lt(n=2000):
    """Standardized light-tunnel columns for the 6-node color_mix subsystem."""
    import causalchamber.datasets as dd
    os.makedirs("/tmp/cc_data", exist_ok=True)
    d = dd.Dataset("lt_walks_v1", root="/tmp/cc_data", download=True)
    df = d.get_experiment("color_mix").as_pandas_dataframe()
    idx = np.random.default_rng(0).choice(len(df), min(n, len(df)), replace=False)
    return {v: _z(df[v].to_numpy()[idx]) for v in NODES}


def inject(base, gamma, rng):
    """Mean-preserving SCALE injection on the real data, as in cc_scale_inject.py: a RandomForest
    removes the real conditional mean of each sink, then the sources drive the residual's spread via
    s=exp(gamma*mean(sources)). E[sink|src]=0 (covariance tests blind); the spread carries the edge."""
    from sklearn.ensemble import RandomForestRegressor
    src = np.column_stack([base[s] for s in SOURCES])
    s = np.exp(gamma * src.mean(axis=1))                    # spread driven by all sources
    out = {so: base[so] for so in SOURCES}                  # keep the real sources
    for k in SINKS:
        m = RandomForestRegressor(200, min_samples_leaf=5, n_jobs=-1).fit(src, base[k])
        resid = _z(base[k] - m.predict(src))                # real per-sink noise, standardized
        out[k] = _z(resid * s)                              # E[.|src]=0, spread ~ s(src)
    return out


def _score(dmap, nm, alpha, max_cond):
    data = np.column_stack([dmap[v] for v in NODES])
    ci = adapters.make_test(nm, data)
    cg = discovery.run_pc(data, NODES, ci, alpha, max_cond)
    learned = discovery.cpdag_skeleton(cg, NODES)
    return metrics.skeleton_scores(TRUE_EDGES, learned, NODES)


def run(panel, gamma=0.7, alpha=0.05, max_cond=3, seed=0):
    base = load_lt()
    inj = inject(base, gamma, np.random.default_rng(seed))
    rows = []
    for nm in panel:
        real = _score(base, nm, alpha, max_cond)            # real data: single deterministic run
        injd = _score(inj, nm, alpha, max_cond)             # scale-injected: single run
        rows.append(dict(test=nm, real_shd=int(real["shd"]), real_recall=real["recall"],
                         inj_shd=int(injd["shd"]), inj_recall=injd["recall"]))
        print(f"  {nm:12s} real SHD={int(real['shd'])} recall={real['recall']:.2f} | "
              f"injected SHD={int(injd['shd'])} recall={injd['recall']:.2f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="fisherz,gcm_linear,rcot,gfcm")
    ap.add_argument("--gamma", type=float, default=0.7)
    args = ap.parse_args()
    print(f"tab:cc-inject -- color_mix light-tunnel scale injection (gamma={args.gamma})")
    df = run(args.panel.split(","), gamma=args.gamma)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "injection_chamber.parquet")
    df.to_parquet(p, index=False)
    print(df.to_string(index=False)); print(f"wrote {p}")


if __name__ == "__main__":
    main()
