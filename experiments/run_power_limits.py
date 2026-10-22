"""Power characterization behind the "Power and its limits" paragraph of the appendix."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import argparse

import numpy as np
import pandas as pd
from dagsampler import CausalDataGenerator

import adapters

OUT = os.path.join(_ROOT, "results", "aggregated")
ALPHA = 0.05


def _gamma_scale(gamma):
    """Parent to std map 0.3 + exp(gamma * clip(X,-4,4)), monotone in X; gamma=0 gives no edge."""
    def f(parent_data):
        x = np.clip(parent_data["X"].to_numpy(), -4.0, 4.0)   # by NAME: X is not the first parent of Y
        return 0.3 + np.exp(gamma * x)
    return f


def _cfg(dS, gamma, edge, n, seed):
    """Location-scale DGP: a fork under the null, a triangle with an X->Y scale edge under the alternative."""
    Z = [f"Z{i}" for i in range(1, dS + 1)]
    src = {z: {"type": "continuous", "functional_form": {"name": "linear", "weights": {}},
               "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}} for z in Z}
    edges = [(z, "X") for z in Z] + [(z, "Y") for z in Z] + ([("X", "Y")] if edge else [])
    Xnode = {"type": "continuous", "functional_form": {"name": "linear", "weights": {z: 1.0 for z in Z}},
             "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}}
    if edge:
        Ynode = {"type": "continuous",
                 "functional_form": {"name": "linear", "weights": {**{z: 0.7 for z in Z}, "X": 0.0}},
                 "noise_model": {"name": "heteroskedastic", "func": _gamma_scale(gamma)}}
    else:
        Ynode = {"type": "continuous",
                 "functional_form": {"name": "linear", "weights": {z: 0.7 for z in Z}},
                 "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.3}}
    return {"simulation_params": {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                                  "store_ci_oracle": False},
            "graph_params": {"type": "custom", "nodes": Z + ["X", "Y"], "edges": edges},
            "node_params": {**src, "X": Xnode, "Y": Ynode}}


def _pvals(dS, gamma, edge, n, reps, panel):
    """Return {test: [pvals]} over reps for one (dS, gamma, edge, n) cell."""
    S = list(range(2, 2 + dS))                       # X=0, Y=1, Z=2..
    out = {t: [] for t in panel}
    for r in range(reps):
        data = CausalDataGenerator(_cfg(dS, gamma, edge, n, 1000 + r)).simulate()["data"]
        arr = data[[f"Z{i}" for i in range(1, dS + 1)]].to_numpy()
        arr = np.column_stack([data["X"].to_numpy(), data["Y"].to_numpy(), arr])
        for t in panel:
            try:
                out[t].append(float(adapters.make_test(t, arr)(0, 1, S)))
            except Exception:
                out[t].append(float("nan"))
    return out


def _sc_power(null_pv, alt_pv):
    """Size-corrected power: threshold at the alpha-quantile of the matched null."""
    null_pv = np.asarray(null_pv, float); null_pv = null_pv[~np.isnan(null_pv)]
    alt_pv = np.asarray(alt_pv, float); alt_pv = alt_pv[~np.isnan(alt_pv)]
    if not len(null_pv) or not len(alt_pv):
        return float("nan")
    c = np.percentile(null_pv, 100 * ALPHA)
    return float((alt_pv < c).mean())


def run(reps, gamma_weak, gamma_mod, panel=("gfcm", "gcm_linear")):
    panel = list(panel)
    rows = []

    # (1) level on the location-scale null, and the n-sweep at the weak edge
    ns = [200, 400, 800, 2000]
    null_by_n = {n: _pvals(3, 0.0, False, n, reps, panel) for n in ns}
    weak_by_n = {n: _pvals(3, gamma_weak, True, n, reps, panel) for n in ns}
    for n in ns:
        for t in panel:
            nu = null_by_n[n][t]; al = weak_by_n[n][t]
            size = float(np.mean(np.asarray(nu, float)[~np.isnan(nu)] < ALPHA))
            rows.append(dict(probe="n_sweep", dS=3, n=n, gamma=gamma_weak, test=t,
                             size=size, sc_power=_sc_power(nu, al)))
            print(f"  [n-sweep] n={n:<5d} {t:11s} size={size:.3f} sc_power={rows[-1]['sc_power']:.3f}", flush=True)

    # (2) |S|-invariance at a moderate edge, n=2000
    for dS in [4, 8, 12, 16]:
        nu = _pvals(dS, 0.0, False, 2000, reps, panel)
        al = _pvals(dS, gamma_mod, True, 2000, reps, panel)
        for t in panel:
            rows.append(dict(probe="S_sweep", dS=dS, n=2000, gamma=gamma_mod, test=t,
                             size=float(np.mean(np.asarray(nu[t], float)[~np.isnan(nu[t])] < ALPHA)),
                             sc_power=_sc_power(nu[t], al[t])))
            print(f"  [S-sweep] |S|={dS:<3d} {t:11s} sc_power={rows[-1]['sc_power']:.3f}", flush=True)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--gamma_weak", type=float, default=0.15)
    ap.add_argument("--gamma_mod", type=float, default=0.35)
    ap.add_argument("--panel", default="gfcm,gcm_linear")
    args = ap.parse_args()
    print(f"power-limits probe (reps={args.reps}, weak gamma={args.gamma_weak}, moderate gamma={args.gamma_mod})")
    df = run(args.reps, args.gamma_weak, args.gamma_mod, panel=tuple(args.panel.split(",")))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "power_limits.parquet")
    df.to_parquet(p, index=False)
    print(df.to_string(index=False)); print(f"wrote {p}")


if __name__ == "__main__":
    main()
