"""dagsampler config builders for the GFCM benchmark simulation tables.

Each builder returns a dagsampler config dict (or a {label: config} map) for a
single dataset; the benchmark harness loops seeds for replications and runs the
CI test / PC on each. Every config sets ``store_ci_oracle`` so the ground-truth
null/alternative label comes from d-separation, not from the test.

Requires dagsampler >= 0.4.0 (the ``shape`` noise model).

Edge-type primitives (each isolated to its own child, since dagsampler applies
one mechanism + one noise model per node):

  linear-mean      X -> Y via linear mean             (Cov != 0;  every test)
  non-monotone z^2 X -> Y via Y = w X^2 + noise       (Cov  = 0;  PCM, GFCM)
  scale            X drives Var(Y), mean flat         (mean+Cov 0; GFCM/quantile)
  tail-shape       X drives skew(Y), mean+var flat    (2nd moment 0; quantile only)
  co-movement      shared latent U drives Var(X),Var(Y) (mean indep; symmetric q-q)

Run ``python configs.py`` to smoke-build and oracle-check every config.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.stats import norm
from dagsampler import CausalDataGenerator


def _monotone_scale(parent_data):
    """Conditional std monotone in the first parent (an *odd* component, so the
    location-scale edge is detectable by the moment/quantile bank). `abs`-based
    scales are even in X and only `c_X c_Y` (excluded by default) can catch them."""
    x = np.clip(parent_data.iloc[:, 0].to_numpy(), -4.0, 4.0)
    return 0.3 + np.exp(0.5 * x)

# --------------------------------------------------------------------------
# Node / edge primitives
# --------------------------------------------------------------------------

def gaussian_source(mean: float = 0.0, std: float = 1.0) -> dict:
    return {"type": "continuous",
            "distribution": {"name": "gaussian", "mean": mean, "std": std}}


def heavy_source(df: int = 3) -> dict:
    """Heavy-tailed continuous source (Student-t)."""
    return {"type": "continuous", "distribution": {"name": "student_t", "df": df}}


def categorical_source(cardinality: int) -> dict:
    """Uniform-ish categorical source of the given cardinality."""
    return {"type": "categorical", "cardinality": cardinality,
            "distribution": {"probs": [1.0 / cardinality] * cardinality}}


def linear_child(parent: str, w: float = 1.0, noise_std: float = 1.0) -> dict:
    """Linear-mean edge: detectable by every test (Cov != 0)."""
    return {"type": "continuous",
            "functional_form": {"name": "linear", "weights": {parent: w}},
            "noise_model": {"name": "additive", "dist": "gaussian", "std": noise_std}}


def z2_child(parent: str, w: float = 1.0, noise_std: float = 0.5) -> dict:
    """Non-monotone-mean edge Y = w*X^2 + noise. Cov(X,Y)=0 for symmetric X."""
    return {"type": "continuous",
            "functional_form": {"name": "polynomial",
                                 "weights": {parent: w}, "degrees": {parent: 2}},
            "noise_model": {"name": "additive", "dist": "gaussian", "std": noise_std}}


def scale_child(parent: str, dist: str = "gaussian", func=_monotone_scale) -> dict:
    """Location-scale edge: X drives Var(Y); conditional mean flat.

    `func` is the parent->std map (default monotone, so the edge is detectable);
    pass ``"abs_first_parent"`` for the hard symmetric (even-|X|) probe. `dist`
    selects the unit-variance base distribution (Gaussian or heavy-tailed/skewed)."""
    nm = {"name": "heteroskedastic", "func": func}
    if dist != "gaussian":
        nm["dist"] = dist
    return {"type": "continuous",
            "functional_form": {"name": "linear", "weights": {parent: 0.0}},
            "noise_model": nm}


def tailshape_child(parent: str, std: float = 1.0) -> dict:
    """Tail-shape edge: X drives skew(Y); conditional mean and variance flat."""
    return {"type": "continuous",
            "functional_form": {"name": "linear", "weights": {parent: 0.0}},
            "noise_model": {"name": "shape", "func": "skew_first_parent", "std": std}}


def sim_params(n: int, seed: int, max_cond: int = 1) -> dict:
    return {"n_samples": n, "seed_structure": 7, "seed_data": seed,
            "store_ci_oracle": True, "ci_oracle_max_cond_set": max_cond}


# --------------------------------------------------------------------------
# tab:beyond / fig:beyond  -- per-edge-type recall + skeleton SHD (p=6)
# --------------------------------------------------------------------------

def tab_beyond(n: int = 2000, seed: int = 1) -> dict:
    """One 6-node DAG with three isolated edge types (linear, z^2, scale)."""
    return {
        "simulation_params": sim_params(n, seed),
        "graph_params": {"type": "custom",
                         "nodes": ["X1", "Y1", "X2", "Y2", "X3", "Y3"],
                         "edges": [("X1", "Y1"), ("X2", "Y2"), ("X3", "Y3")]},
        "node_params": {
            "X1": gaussian_source(), "Y1": linear_child("X1"),
            "X2": gaussian_source(), "Y2": z2_child("X2"),
            "X3": gaussian_source(), "Y3": scale_child("X3"),
        },
    }


# --------------------------------------------------------------------------
# tab:competitors -- 3 edge types (alt) + heavy-tailed null (size), n=1000
# --------------------------------------------------------------------------

def tab_competitors(n: int = 1000, seed: int = 1) -> dict:
    cfgs = {}
    base = tab_beyond(n, seed)
    cfgs["alt_linear"] = _single_edge(linear_child, n, seed)
    cfgs["alt_z2"] = _single_edge(z2_child, n, seed)
    cfgs["alt_scale"] = _single_edge(scale_child, n, seed)
    # Heavy-tailed null: X _||_ Y | Z with Z Student-t driving both means.
    cfgs["null_heavy"] = {
        "simulation_params": sim_params(n, seed, max_cond=1),
        "graph_params": {"type": "custom", "nodes": ["Z", "X", "Y"],
                         "edges": [("Z", "X"), ("Z", "Y")]},
        "node_params": {
            "Z": heavy_source(df=3),
            "X": linear_child("Z"), "Y": linear_child("Z"),
        },
    }
    return cfgs


def _single_edge(child_fn, n, seed) -> dict:
    return {
        "simulation_params": sim_params(n, seed),
        "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": [("X", "Y")]},
        "node_params": {"X": gaussian_source(), "Y": child_fn("X")},
    }


# --------------------------------------------------------------------------
# tab:ablation -- which block catches what (4 alternatives), n=2000
# --------------------------------------------------------------------------

def tab_ablation(n: int = 2000, seed: int = 1) -> dict:
    return {
        "linear_mean": _single_edge(linear_child, n, seed),
        "nonmonotone_z2": _single_edge(z2_child, n, seed),
        "scale": _single_edge(scale_child, n, seed),
        "tail_shape": _single_edge(tailshape_child, n, seed),
    }


# --------------------------------------------------------------------------
# tab:ablation-conditional -- the same ladder under conditioning, X _||_ Y | Z
# --------------------------------------------------------------------------

def _scale_X(parent_data):
    """Conditional std driven by X only (selected by name, so it ignores the
    confounder Z): 0.3 + exp(0.5 * clip(X, -4, 4))."""
    x = np.clip(parent_data["X"].to_numpy(), -4.0, 4.0)
    return 0.3 + np.exp(0.5 * x)


def _skew_X(parent_data):
    """Skew-normal shape alpha driven by X only (mean and variance held fixed)."""
    return 4.0 * parent_data["X"].to_numpy()


def tab_ablation_conditional(n: int = 2000, seed: int = 1) -> dict:
    """Conditioned feature-bank ablation: query X _||_ Y | Z.

    Null is a common-cause fork (Z->X, Z->Y, X and Y conditionally independent);
    each alternative adds a typed direct edge X->Y so that ONLY the intended moment
    of Y depends on X, and X _||_ Y | Z is false. Z drives X (sin) and Y (Z^2)
    nonlinearly, so the spline nuisance is non-trivial and the conditional test must
    actually regress out Z. Same detection-class ladder as tab:ablation, now under
    conditioning. g(Z) = 0.7 Z^2 feeds Y's mean in every config."""
    FORK = [("Z", "X"), ("Z", "Y")]
    TRI = FORK + [("X", "Y")]
    x_node = {"type": "continuous",
              "functional_form": {"name": "sin", "weights": {"Z": 1.5}},
              "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.7}}

    def cfg(edges, y_node):
        return {"simulation_params": {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                                      "store_ci_oracle": True, "ci_oracle_max_cond_set": 1},
                "graph_params": {"type": "custom", "nodes": ["Z", "X", "Y"], "edges": edges},
                "node_params": {"Z": gaussian_source(), "X": x_node, "Y": y_node}}

    def poly_y(weights, degrees, noise):
        return {"type": "continuous",
                "functional_form": {"name": "polynomial", "weights": weights, "degrees": degrees},
                "noise_model": noise}

    gauss = {"name": "additive", "dist": "gaussian", "std": 1.0}
    return {
        # X _||_ Y | Z TRUE: Y depends on Z only
        "null": cfg(FORK, poly_y({"Z": 0.7}, {"Z": 2}, gauss)),
        # X drives Y's conditional mean (linear)
        "linear_mean": cfg(TRI, poly_y({"Z": 0.7, "X": 0.6}, {"Z": 2, "X": 1}, gauss)),
        # X drives Y's conditional mean non-monotonically (Cov(X,Y|Z)~0)
        "nonmonotone_z2": cfg(TRI, poly_y({"Z": 0.7, "X": 0.5}, {"Z": 2, "X": 2}, gauss)),
        # X drives Y's conditional variance, mean flat in X
        "scale": cfg(TRI, poly_y({"Z": 0.7, "X": 0.0}, {"Z": 2, "X": 1},
                                 {"name": "heteroskedastic", "func": _scale_X})),
        # X drives Y's conditional skew, mean and variance fixed
        "tail_shape": cfg(TRI, poly_y({"Z": 0.7, "X": 0.0}, {"Z": 2, "X": 1},
                                      {"name": "shape", "func": _skew_X, "std": 1.0})),
    }


# --------------------------------------------------------------------------
# tab:mixed-conditional -- mixed-type pairings UNDER a mixed conditioning set
# Z=(Zc continuous, Zk 3-level categorical). Null = fork (Zc,Zk -> X,Y; X _||_ Y | Z);
# alternative adds a typed X->Y edge. CC and KC alts are beyond-covariance SPREAD edges
# (X sets Var(Y)); CK and KK alts are probability shifts. Query X _||_ Y | {Zc,Zk}.
# --------------------------------------------------------------------------

def _scale_Xcont(parent_data):
    """Continuous X drives Y's conditional std (CC spread alt), selected by name."""
    x = np.clip(parent_data["X"].to_numpy(), -4.0, 4.0)
    return 0.3 + np.exp(0.5 * x)


def _scale_Xcat(parent_data):
    """Categorical X (codes 0..) sets Y's conditional std (KC cat->spread alt)."""
    return 0.5 + 1.0 * parent_data["X"].to_numpy().astype(float)


def tab_mixed_conditional(n: int = 1000, seed: int = 1, cardinality: int = 3) -> dict:
    K = cardinality

    def sp():
        return {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                "store_ci_oracle": True, "ci_oracle_max_cond_set": 2}

    def graph(edges):
        return {"type": "custom", "nodes": ["Zc", "Zk", "X", "Y"], "edges": edges}

    FORK = [("Zc", "X"), ("Zk", "X"), ("Zc", "Y"), ("Zk", "Y")]
    TRI = FORK + [("X", "Y")]

    # continuous node with mean = stratum(Zk) + 0.8*Zc
    def cont_meanZ():
        return {"type": "continuous",
                "functional_form": {"name": "stratum_means",
                                    "strata_means": {f"Zk={i}": float(i - 1) for i in range(K)},
                                    "metric_weights": {"Zc": 0.8}},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}}

    # categorical node, logistic on its continuous+categorical parents (auto weights)
    def cat_logit():
        return {"type": "categorical", "cardinality": K,
                "categorical_model": {"name": "logistic"}}

    # categorical Y with an explicit, strong X->Y effect (Zc,Zk auto); used in the
    # CK/KK alternatives so the probability shift is clearly detectable (full power).
    def cat_logit_X(xspec):
        return {"type": "categorical", "cardinality": K,
                "categorical_model": {"name": "logistic", "weights": {"X": xspec}}}

    diag = [[2.5 if i == j else 0.0 for j in range(K)] for i in range(K)]  # cat X->Y (K,K)
    ckvec = [float(c - (K - 1) / 2.0) * 2.0 for c in range(K)]             # cont X->Y (K,)

    # continuous Y whose MEAN is X-independent (Zk strata only) but whose VARIANCE
    # is driven by X (spread alt). X-independent strata supplied for both key orders.
    def cont_spread(x_scale, x_is_cat):
        strata = {}
        if x_is_cat:
            for i in range(K):
                for j in range(K):
                    strata[f"Zk={i}|X={j}"] = float(i - 1)
                    strata[f"X={j}|Zk={i}"] = float(i - 1)
            mw = {"Zc": 0.8}
        else:  # X continuous: only Zk is categorical, so plain Zk strata; X out of mean
            strata = {f"Zk={i}": float(i - 1) for i in range(K)}
            mw = {"Zc": 0.8, "X": 0.0}
        return {"type": "continuous",
                "functional_form": {"name": "stratum_means", "strata_means": strata,
                                    "metric_weights": mw},
                "noise_model": {"name": "heteroskedastic", "func": x_scale}}

    def cfg(edges, x_node, y_node):
        return {"simulation_params": sp(), "graph_params": graph(edges),
                "node_params": {"Zc": gaussian_source(), "Zk": categorical_source(K),
                                "X": x_node, "Y": y_node}}

    return {
        # CC: continuous X, continuous Y; alt = X drives Var(Y) (beyond-covariance spread)
        "CC_null": cfg(FORK, cont_meanZ(), cont_meanZ()),
        "CC_alt":  cfg(TRI,  cont_meanZ(), cont_spread(_scale_Xcont, x_is_cat=False)),
        # CK: continuous X, categorical Y; alt = X shifts Y's class probabilities
        "CK_null": cfg(FORK, cont_meanZ(), cat_logit()),
        "CK_alt":  cfg(TRI,  cont_meanZ(), cat_logit_X(ckvec)),
        # KC: categorical X, continuous Y; alt = categorical X sets Var(Y), not mean
        "KC_null": cfg(FORK, cat_logit(), cont_meanZ()),
        "KC_alt":  cfg(TRI,  cat_logit(), cont_spread(_scale_Xcat, x_is_cat=True)),
        # KK: categorical X, categorical Y; alt = X shifts Y's class probabilities
        "KK_null": cfg(FORK, cat_logit(), cat_logit()),
        "KK_alt":  cfg(TRI,  cat_logit(), cat_logit_X(diag)),
    }


# --------------------------------------------------------------------------
# tab:asym -- asymmetric mean-quantile vs symmetric q-q, n=2000
# --------------------------------------------------------------------------

def tab_asym(n: int = 2000, seed: int = 1) -> dict:
    cfgs = {
        # directed tail-shape edge (asymmetric construction wins)
        "directed_tailshape": _single_edge(tailshape_child, n, seed),
        "scale": _single_edge(scale_child, n, seed),
    }
    # symmetric tail co-movement: shared heavy-tailed latent drives Var(X),Var(Y),
    # means independent (the q-q product catches it; asymmetric is blind).
    cfgs["symmetric_comovement"] = {
        "simulation_params": sim_params(n, seed, max_cond=1),
        "graph_params": {"type": "custom", "nodes": ["U", "X", "Y"],
                         "edges": [("U", "X"), ("U", "Y")]},
        "node_params": {
            "U": heavy_source(df=3),
            "X": scale_child("U"), "Y": scale_child("U"),
        },
    }
    return cfgs


# --------------------------------------------------------------------------
# tab:pscaling -- scaling in p and |S| (random Erdos-Renyi DAGs, degree ~3)
# --------------------------------------------------------------------------

def tab_pscaling(p: int = 10, n: int = 2000, seed: int = 1) -> dict:
    edge_prob = min(0.9, 3.0 / max(1, p - 1))  # expected degree ~3
    return {
        "simulation_params": sim_params(n, seed, max_cond=3),
        "graph_params": {"type": "random", "n_nodes": p, "edge_prob": edge_prob},
        # all-continuous; default endogenous mechanism mixes linear/poly/etc.
        "node_params": {"default_endogenous": {"type": "continuous"},
                        "default_exogenous": {"type": "continuous"}},
    }


# --------------------------------------------------------------------------
# tab:f2-nuisance -- descendant-conditioning on a heavy-tailed scale-child
# --------------------------------------------------------------------------

def tab_f2_nuisance(n: int = 2000, seed: int = 1) -> dict:
    """X -> Y (scale edge); W is a heavy-tailed scale-child of Y placed in the
    conditioning set. Tests X _||_ Y | W where W is heavy-tailed (the F2 trap)."""
    return {
        "simulation_params": sim_params(n, seed, max_cond=1),
        "graph_params": {"type": "custom", "nodes": ["X", "Y", "W"],
                         "edges": [("X", "Y"), ("Y", "W")]},
        "node_params": {
            "X": gaussian_source(),
            "Y": scale_child("X"),
            # heavy-tailed scale-child of Y
            "W": {"type": "continuous",
                  "functional_form": {"name": "linear", "weights": {"Y": 0.0}},
                  "noise_model": {"name": "heteroskedastic", "func": "abs_first_parent"}},
        },
    }


# --------------------------------------------------------------------------
# tab:mixed -- four (X,Y) type pairings (CC, CK, KC, KK)
# --------------------------------------------------------------------------

def tab_mixed(n: int = 2000, seed: int = 1, cardinality: int = 3) -> dict:
    cfgs = {}
    # CC: continuous -> continuous (linear mean)
    cfgs["CC"] = _single_edge(linear_child, n, seed)
    # KC: categorical -> continuous (stratum means)
    cfgs["KC"] = {
        "simulation_params": sim_params(n, seed),
        "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": [("X", "Y")]},
        "node_params": {
            "X": categorical_source(cardinality),
            "Y": {"type": "continuous",
                  "functional_form": {"name": "stratum_means"},
                  "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}},
        },
    }
    # CK: continuous -> categorical (logistic)
    cfgs["CK"] = {
        "simulation_params": sim_params(n, seed),
        "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": [("X", "Y")]},
        "node_params": {
            "X": gaussian_source(),
            "Y": {"type": "categorical", "cardinality": cardinality,
                  "categorical_model": {"name": "logistic",
                                        "intercepts": [0.0] * cardinality,
                                        "weights": {"X": [0.0] + [1.0] * (cardinality - 1)}}},
        },
    }
    # KK: categorical -> categorical (logistic, categorical parent)
    cfgs["KK"] = {
        "simulation_params": sim_params(n, seed),
        "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": [("X", "Y")]},
        "node_params": {
            "X": categorical_source(cardinality),
            "Y": {"type": "categorical", "cardinality": cardinality,
                  "categorical_model": {"name": "logistic"}},
        },
    }
    return cfgs


# --------------------------------------------------------------------------
# tab:cardinality -- J=10 pooled vs per-level (categorical X, continuous Y)
# --------------------------------------------------------------------------

def tab_cardinality(seed: int = 1) -> dict:
    cfgs = {}
    for J in (3, 10):
        for n in (1000, 2000):
            # alternative: location shift driven by the categorical level
            cfgs[f"alt_J{J}_n{n}"] = {
                "simulation_params": sim_params(n, seed),
                "graph_params": {"type": "custom", "nodes": ["X", "Y"],
                                 "edges": [("X", "Y")]},
                "node_params": {
                    "X": categorical_source(J),
                    "Y": {"type": "continuous",
                          "functional_form": {"name": "stratum_means"},
                          "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}},
                },
            }
            # null: X _||_ Y (no edge), Y continuous
            cfgs[f"null_J{J}_n{n}"] = {
                "simulation_params": sim_params(n, seed, max_cond=0),
                "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": []},
                "node_params": {"X": categorical_source(J), "Y": gaussian_source()},
            }
    return cfgs


# --------------------------------------------------------------------------
# tab:calib-grid -- heteroscedastic-in-Z null, heavy tails via Z scale-mixture
# --------------------------------------------------------------------------

def _hetero_exp_z1(parent_data):
    """Heteroscedastic std exp(0.4*Z1) -- X's spread driven by Z1."""
    return np.exp(0.4 * parent_data["Z1"].to_numpy())


def _hetero_exp_z0(parent_data):
    """Heteroscedastic std exp(0.4*Z0) -- Y's spread driven by Z0."""
    return np.exp(0.4 * parent_data["Z0"].to_numpy())


def tab_calib_grid(seed: int = 1) -> dict:
    """GFCM-vs-RCoT calibration on the paper's heteroscedastic-heavy-tailed null
    (reproduced in dagsampler via the 0.4.0 heteroskedastic `dist`): X _||_ Y |
    {Z0,Z1} with a LINEAR mean (X<-Z0, Y<-Z1) and cross-heteroscedastic noise
    (X scale ~ exp(0.4*Z1), Y scale ~ exp(0.4*Z0)), Student-t(3) base. Base dist
    and n swept.

    NOTE: a properly-implemented RCoT (RCIT) HOLDS nominal level here (~0.04-0.07),
    as does GFCM -- this is a calibration TIE. The dramatic 'RCoT -> 0.89 collapse'
    reported in earlier drafts came from a weak hand-rolled RFF, not RCIT; do not
    claim GFCM beats a real RCoT on this null."""
    cfgs = {}

    def child(scale_fn, mean_z, dist):
        nm = {"name": "heteroskedastic", "func": scale_fn, "df": 3}
        if dist != "gaussian":
            nm["dist"] = dist
        return {"type": "continuous",
                "functional_form": {"name": "linear",
                                     "weights": {"Z0": 1.0 if mean_z == "Z0" else 0.0,
                                                 "Z1": 1.0 if mean_z == "Z1" else 0.0}},
                "noise_model": nm}

    for dist in ("student_t", "gaussian", "laplace"):
        for n in (1000, 4000, 16000):
            cfgs[f"{dist}_n{n}"] = {
                "simulation_params": sim_params(n, seed, max_cond=2),
                "graph_params": {"type": "custom", "nodes": ["Z0", "Z1", "X", "Y"],
                                 "edges": [("Z0", "X"), ("Z1", "X"), ("Z0", "Y"), ("Z1", "Y")]},
                "node_params": {"Z0": gaussian_source(), "Z1": gaussian_source(),
                                "X": child(_hetero_exp_z1, "Z0", dist),
                                "Y": child(_hetero_exp_z0, "Z1", dist)},
            }
    return cfgs


# --------------------------------------------------------------------------
# tab:growingK -- misspecified sin(3Z) null
# --------------------------------------------------------------------------

def tab_growingK(n: int = 2000, seed: int = 1) -> dict:
    """X _||_ Y | Z null with a wiggly sin(3Z) conditional mean (misspecification
    stress for a fixed-degree nuisance)."""
    sin_child = lambda parent: {
        "type": "continuous",
        "functional_form": {"name": "sin", "weights": {parent: 3.0}},
        "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5}}
    return {
        "simulation_params": sim_params(n, seed, max_cond=1),
        "graph_params": {"type": "custom", "nodes": ["Z", "X", "Y"],
                         "edges": [("Z", "X"), ("Z", "Y")]},
        "node_params": {"Z": gaussian_source(), "X": sin_child("Z"), "Y": sin_child("Z")},
    }


# --------------------------------------------------------------------------
# tab:interaction -- additive vs interaction null, Z in R^2
# --------------------------------------------------------------------------

def tab_interaction(n: int = 2000, seed: int = 1) -> dict:
    cfgs = {}
    # additive null: X,Y each depend additively on Z1,Z2; X _||_ Y | {Z1,Z2}
    cfgs["additive"] = {
        "simulation_params": sim_params(n, seed, max_cond=2),
        "graph_params": {"type": "custom", "nodes": ["Z1", "Z2", "X", "Y"],
                         "edges": [("Z1", "X"), ("Z2", "X"), ("Z1", "Y"), ("Z2", "Y")]},
        "node_params": {
            "Z1": gaussian_source(), "Z2": gaussian_source(),
            "X": {"type": "continuous",
                  "functional_form": {"name": "linear", "weights": {"Z1": 1.0, "Z2": 1.0}},
                  "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5}},
            "Y": {"type": "continuous",
                  "functional_form": {"name": "linear", "weights": {"Z1": 1.0, "Z2": 1.0}},
                  "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5}},
        },
    }
    # interaction null: X,Y depend on Z1*Z2 (the additivity boundary)
    inter = lambda: {"type": "continuous",
                     "functional_form": {"name": "interaction", "weights": {"interaction": 1.0}},
                     "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5}}
    cfgs["interaction"] = copy.deepcopy(cfgs["additive"])
    cfgs["interaction"]["node_params"]["X"] = inter()
    cfgs["interaction"]["node_params"]["Y"] = inter()
    return cfgs


# --------------------------------------------------------------------------
# tab:confound -- nonlinear (monotone) confounding; fair-GCM check
# --------------------------------------------------------------------------

def tab_confound(n: int = 2000, seed: int = 1) -> dict:
    """Confounder Z -> X, Z -> Y through a nonlinear (sigmoid) mean; X _||_ Y | Z."""
    sig = lambda parent: {
        "type": "continuous",
        "functional_form": {"name": "sigmoid", "weights": {parent: 2.0}},
        "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5}}
    return {
        "simulation_params": sim_params(n, seed, max_cond=1),
        "graph_params": {"type": "custom", "nodes": ["Z", "X", "Y"],
                         "edges": [("Z", "X"), ("Z", "Y")]},
        "node_params": {"Z": gaussian_source(), "X": sig("Z"), "Y": sig("Z")},
    }


# --------------------------------------------------------------------------
# fig:detection -- scale alternative (invisible to median, caught by tails)
# --------------------------------------------------------------------------

def tab_limit_heavytail(n: int = 2000, seed: int = 1) -> dict:
    """LIMITATION probe. X _||_ Y | Z null with an infinite-variance (Cauchy =
    Student-t df 1) conditioning variable, so X and Y are heavy-tailed. The
    moment-based tests (GFCM, GCM) need finite conditional moments (assumption
    A3) and inflate badly here; bounded-kernel tests (RCoT, KCI) hold level.
    This is the honest cost of GFCM's GCM product statistic."""
    return {
        "simulation_params": sim_params(n, seed, max_cond=1),
        "graph_params": {"type": "custom", "nodes": ["Z", "X", "Y"],
                         "edges": [("Z", "X"), ("Z", "Y")]},
        "node_params": {"Z": heavy_source(df=1),  # Cauchy: infinite variance
                        "X": linear_child("Z"), "Y": linear_child("Z")},
    }


def fig_detection(n: int = 2000, seed: int = 1) -> dict:
    return {"scale_alt": _single_edge(scale_child, n, seed),
            "tailshape_alt": _single_edge(tailshape_child, n, seed)}


# --------------------------------------------------------------------------
# fig:reisach -- varsortability robustness (varsortability computed post hoc)
# --------------------------------------------------------------------------

def fig_reisach(n: int = 2000, seed: int = 1) -> dict:
    """Linear chain X1->X2->X3 with growing variances (high varsortability)."""
    return {
        "simulation_params": sim_params(n, seed, max_cond=1),
        "graph_params": {"type": "custom", "nodes": ["X1", "X2", "X3"],
                         "edges": [("X1", "X2"), ("X2", "X3")]},
        "node_params": {
            "X1": gaussian_source(std=1.0),
            "X2": linear_child("X1", w=1.0, noise_std=1.0),
            "X3": linear_child("X2", w=1.0, noise_std=1.0),
        },
    }


# --------------------------------------------------------------------------
# Registry + smoke check
# --------------------------------------------------------------------------

BUILDERS = {
    "tab:beyond": tab_beyond,
    "tab:competitors": tab_competitors,
    "tab:ablation": tab_ablation,
    "tab:asym": tab_asym,
    "tab:pscaling": tab_pscaling,
    "tab:f2-nuisance": tab_f2_nuisance,
    "tab:mixed": tab_mixed,
    "tab:cardinality": tab_cardinality,
    "tab:calib-grid": tab_calib_grid,
    "tab:growingK": tab_growingK,
    "tab:interaction": tab_interaction,
    "tab:confound": tab_confound,
    "fig:detection": fig_detection,
    "fig:reisach": fig_reisach,
}


def _iter_configs():
    for table, builder in BUILDERS.items():
        out = builder()
        if isinstance(out, dict) and "simulation_params" in out:
            yield table, "(single)", out
        else:
            for label, cfg in out.items():
                yield table, label, cfg


if __name__ == "__main__":

    n_ok = 0
    for table, label, cfg in _iter_configs():
        small = copy.deepcopy(cfg)
        small["simulation_params"]["n_samples"] = 400
        res = CausalDataGenerator(small).simulate()
        data = res["data"]
        assert "ci_oracle" in res, f"{table}/{label}: no oracle"
        assert np.isfinite(data.select_dtypes("number").to_numpy()).all(), \
            f"{table}/{label}: non-finite data"
        n_ok += 1
        print(f"  ok  {table:18s} {label}")
    print(f"\n{n_ok} configs built + oracle-checked OK")


# ==========================================================================
# Connected random typed DAGs (discovery / p-scaling) -- mirrors Paper 1's
# two-seed protocol: topology+types from default_rng(seed_structure), weights
# from a golden-ratio-offset RandomState, data from seed_data.
# ==========================================================================

TYPES = ("linear", "z2", "scale")
TYPES_TAIL = ("linear", "z2", "scale", "tail")   # discovery graphs incl. skew edges (§5.8)
LINEAR_BAND = (0.28, 1.01)
POLY2_BAND = (0.20, 0.71)
GOLDEN = 0x9E3779B1


def _scale_all_parents(parent_data):
    """Monotone-positive heteroscedastic std driven by ALL parents (clipped mean)."""
    m = np.clip(parent_data.to_numpy().mean(axis=1), -5.0, 5.0)
    return 0.3 + np.exp(0.4 * m)


def random_typed_dag(p, edge_prob, seed_structure, seed_data=None, types=TYPES):
    """Connected random lower-triangular DAG with per-node edge types
    (linear / z^2 / scale / tail). Topology + types from default_rng(seed_structure);
    per-mechanism weights from RandomState((seed_structure+GOLDEN)%2^32) in a
    separate pass (decoupled from the type rng); data from seed_data."""
    if seed_data is None:
        seed_data = seed_structure
    rng_s = np.random.default_rng(seed_structure)
    nodes = [f"V{i}" for i in range(p)]
    parents = {j: [] for j in range(p)}
    for j in range(1, p):
        for i in range(j):
            if rng_s.random() < edge_prob:
                parents[j].append(i)
        if not parents[j]:
            parents[j].append(int(rng_s.integers(0, j)))

    node_params, edges, etype, ntype = {}, [], {}, {}
    for j in range(p):
        if not parents[j]:
            node_params[nodes[j]] = gaussian_source()
            continue
        t = types[int(rng_s.integers(0, len(types)))]
        ntype[j] = t
        for i in parents[j]:
            edges.append((nodes[i], nodes[j]))
            etype[frozenset((nodes[i], nodes[j]))] = t

    rng_w = np.random.RandomState((int(seed_structure) + GOLDEN) % (2 ** 32))
    for j in range(p):
        if not parents[j]:
            continue
        pj = [nodes[i] for i in parents[j]]
        t = ntype[j]
        if t == "linear":
            node_params[nodes[j]] = {
                "type": "continuous",
                "functional_form": {"name": "linear",
                                    "weights": {pn: float(rng_w.uniform(*LINEAR_BAND)) for pn in pj}},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}}
        elif t == "z2":
            node_params[nodes[j]] = {
                "type": "continuous",
                "functional_form": {"name": "polynomial",
                                    "weights": {pn: float(rng_w.uniform(*POLY2_BAND)) for pn in pj},
                                    "degrees": {pn: 2 for pn in pj}},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.7}}
        elif t == "scale":
            node_params[nodes[j]] = {
                "type": "continuous",
                "functional_form": {"name": "linear", "weights": {pn: 0.0 for pn in pj}},
                "noise_model": {"name": "heteroskedastic", "func": _scale_all_parents}}
        else:  # tail: parents drive the child's skewness, conditional mean+var fixed
            node_params[nodes[j]] = {
                "type": "continuous",
                "functional_form": {"name": "linear", "weights": {pn: 0.0 for pn in pj}},
                "noise_model": {"name": "shape", "func": "skew_mean_parents", "std": 1.0}}

    cfg = {
        "simulation_params": {"n_samples": None, "seed_structure": int(seed_structure),
                              "seed_data": int(seed_data), "store_ci_oracle": False},
        "graph_params": {"type": "custom", "nodes": nodes, "edges": edges},
        "node_params": node_params,
    }
    return cfg, etype


# --------------------------------------------------------------------------
# cardinality (pooled vs per-level) -- MI-calibrated stratum-means alternative
# (Paper 1 protocol: hold MI constant across J so the comparison is fair).
# --------------------------------------------------------------------------

def cardinality_mi(J, sigma, rng, n=40000, reps=6):
    """E[MI(X;Y)] for X~Unif(J), Y=mu_X+N(0,1), mu_k~N(0,sigma^2)."""
    hyx = 0.5 * np.log(2 * np.pi * np.e)
    mis = []
    for _ in range(reps):
        mu = rng.normal(0, sigma, J)
        X = rng.integers(0, J, n)
        Y = mu[X] + rng.normal(0, 1, n)
        prob = np.mean(norm.pdf(Y[:, None] - mu[None, :]), axis=1)
        mis.append(-np.mean(np.log(prob + 1e-300)) - hyx)
    return float(np.mean(mis))


def cardinality_calibrate_sigma(J, target=0.15):
    """Bisect the stratum-mean spread so E[MI(X;Y)] == target (constant across J)."""
    rng = np.random.default_rng(1234 + J)
    lo, hi = 0.02, 3.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if cardinality_mi(J, mid, rng) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cardinality_alt(J, n, sigma, seed):
    """Categorical X (J levels) -> continuous Y via stratum-means at spread sigma."""
    return {"simulation_params": {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                                  "store_ci_oracle": False, "strata_means_spread": [sigma, sigma]},
            "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": [("X", "Y")]},
            "node_params": {"X": categorical_source(J),
                            "Y": {"type": "continuous", "functional_form": {"name": "stratum_means"},
                                  "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}}}}


def cardinality_null(J, n, seed):
    return {"simulation_params": {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                                  "store_ci_oracle": False},
            "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": []},
            "node_params": {"X": categorical_source(J), "Y": gaussian_source()}}


# --------------------------------------------------------------------------
# Null / backend-probe builders for the §5.5/§5.8 nuisance experiments
# --------------------------------------------------------------------------

def indep_null(n=2000, seed=1):
    """X _||_ Y marginally (two independent gaussians)."""
    return {"simulation_params": {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                                  "store_ci_oracle": False},
            "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": []},
            "node_params": {"X": gaussian_source(), "Y": gaussian_source()}}


def f2_null(n=2000, seed=1):
    """X _||_ Y | W true: X,Y independent, W a heavy-tailed scale-child of Y."""
    return {"simulation_params": {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                                  "store_ci_oracle": False},
            "graph_params": {"type": "custom", "nodes": ["X", "Y", "W"], "edges": [("Y", "W")]},
            "node_params": {"X": gaussian_source(), "Y": gaussian_source(),
                            "W": {"type": "continuous",
                                  "functional_form": {"name": "linear", "weights": {"Y": 0.0}},
                                  "noise_model": {"name": "heteroskedastic", "func": _monotone_scale,
                                                  "dist": "student_t", "df": 3}}}}


def nuisance_null(kind="smooth", n=2000, seed=1):
    """X _||_ Y | Z nulls for the nuisance-backend comparison:
    'smooth' (mild sin mean, gaussian Z), 'sin3z' (wiggly sin(3Z), gaussian Z),
    'sin3z_heavy' (wiggly sin(3Z), heavy-tailed Student-t Z)."""
    src = heavy_source(df=3) if kind == "sin3z_heavy" else gaussian_source()
    w = 3.0 if kind.startswith("sin3z") else 1.0
    child = lambda: {"type": "continuous",
                     "functional_form": {"name": "sin", "weights": {"Z": w}},
                     "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5}}
    return {"simulation_params": {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                                  "store_ci_oracle": False},
            "graph_params": {"type": "custom", "nodes": ["Z", "X", "Y"],
                             "edges": [("Z", "X"), ("Z", "Y")]},
            "node_params": {"Z": src, "X": child(), "Y": child()}}


def single_edge_at_n(n, seed=1):
    """A single linear X->Y edge at sample size n (for runtime sweeps)."""
    return {"simulation_params": {"n_samples": n, "seed_structure": 7, "seed_data": seed,
                                  "store_ci_oracle": False},
            "graph_params": {"type": "custom", "nodes": ["X", "Y", "Z"],
                             "edges": [("Z", "X"), ("Z", "Y"), ("X", "Y")]},
            "node_params": {"Z": gaussian_source(), "X": linear_child("Z"),
                            "Y": {"type": "continuous",
                                  "functional_form": {"name": "linear", "weights": {"Z": 0.8, "X": 0.6}},
                                  "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}}}}


# ==========================================================================
# SIMULATION PLAN (§5)  --  the benchmark, organized by section.
#
# Default n = 100_000 (laptop-feasible scale for the fast four GFCM/BLITZ/RCoT/
# GCM). The par_cop/KCI companion panels pass a moderate n (par_cop ~17 min/call at
# 1e5 is off the table locally). Conditioning depth via dS (the |S| grid).
# Every config sets the d-separation oracle, so the null/alt label is ground
# truth, not the test. Build + oracle-check every PLAN config below with
# ``python -c "import dgp; dgp.plan_smoke()"``.
# ==========================================================================

N_SCALE = 100_000          # headline sample size (the fast four can reach it)
N_MODERATE = 2_000         # par_cop/KCI companion + where detection differences separate


def _het_exp_z1(parent_data):
    """Conditional std exp(0.4*Z1): heteroskedastic-in-Z null noise."""
    return np.exp(0.4 * np.clip(parent_data["Z1"].to_numpy(), -5.0, 5.0))


def _znodes(dS, kind):
    """Z1..ZdS sources for a conditioning set of size dS.
    kind: 'gauss' | 'heavy' (Student-t(3)) | 'mixed' (Z1 categorical, rest gaussian)."""
    Z = [f"Z{i}" for i in range(1, dS + 1)]
    src = {}
    for i, z in enumerate(Z):
        if kind == "heavy":
            src[z] = heavy_source(df=3)
        elif kind == "mixed" and i == 0:
            src[z] = categorical_source(3)
        else:
            src[z] = gaussian_source()
    return Z, src


def _fork(Z):
    return [(z, "X") for z in Z] + [(z, "Y") for z in Z]


def _mean_over_Z(Z, form="linear", w=1.0, noise=None):
    """Child whose conditional mean is form(sum w*Z); default gaussian noise.
    form='stratum' uses stratum_means on a categorical Z1 + linear metric on the
    continuous Z's (for the mixed-conditioning null)."""
    nm = noise or {"name": "additive", "dist": "gaussian", "std": 1.0}
    if form == "stratum":
        strata = {f"Z1={i}": float(i - 1) for i in range(3)}
        return {"type": "continuous",
                "functional_form": {"name": "stratum_means", "strata_means": strata,
                                    "metric_weights": {z: 0.8 for z in Z if z != "Z1"}},
                "noise_model": nm}
    return {"type": "continuous",
            "functional_form": {"name": form, "weights": {z: w for z in Z}},
            "noise_model": nm}


def _sp(n, seed, dS):
    return {"n_samples": n, "seed_structure": 7, "seed_data": seed,
            "store_ci_oracle": True, "ci_oracle_max_cond_set": dS}


# §5.2  Calibration under the null -----------------------------------------

def calib_nulls(n: int = N_SCALE, seed: int = 1, dS: int = 3) -> dict:
    """Hard nulls X _||_ Y | Z, |Z| = dS, that break naive tests:
      heavy_tail   -- Student-t(3) conditioners
      hetero       -- conditional variance driven by Z (std = exp(0.4*Z1))
      nonlin_mean  -- wiggly sin(1.5*Z1) conditional mean, additive in a single parent
                      (the spline must fit it; sin-of-SUM would be non-additive -> outside
                      every additive nuisance's model class, an unfair interaction stress test)
      mixed_Z      -- Z = (categorical, continuous...) conditioning set
    Query X _||_ Y | {Z1..ZdS} (true; the only X-Y paths run through Z)."""
    def cfg(kind, child):
        knd = "heavy" if kind == "heavy_tail" else "mixed" if kind == "mixed_Z" else "gauss"
        Z, src = _znodes(dS, knd)
        return {"simulation_params": _sp(n, seed, dS),
                "graph_params": {"type": "custom", "nodes": Z + ["X", "Y"], "edges": _fork(Z)},
                "node_params": {**src, "X": child(Z), "Y": child(Z)}}
    het = {"name": "heteroskedastic", "func": _het_exp_z1}
    return {
        "heavy_tail":  cfg("heavy_tail",  lambda Z: _mean_over_Z(Z)),
        "hetero":      cfg("hetero",      lambda Z: _mean_over_Z(Z, noise=het)),
        "nonlin_mean": cfg("nonlin_mean", lambda Z: {
            "type": "continuous",
            "functional_form": {"name": "sin", "weights": {Z[0]: 1.5}},  # single-parent -> additive
            "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0}}),
        "mixed_Z":     cfg("mixed_Z",     lambda Z: _mean_over_Z(Z, form="stratum")),
    }


# §5.3  Power across the alternative ladder --------------------------------

def power_ladder(n: int = N_SCALE, seed: int = 1, dS: int = 3) -> dict:
    """null + typed alternatives, query X _||_ Y | Z (|Z| = dS). Each alt adds an
    X->Y edge touching ONLY one moment of Y (linear mean / non-monotone z^2 /
    scale / tail-shape). Z drives X (sin) and Y (Z^2) nonlinearly, so the
    conditional test must actually regress out Z. Generalizes tab:ablation to
    arbitrary conditioning depth."""
    Z, src = _znodes(dS, "gauss")
    fork = _fork(Z); tri = fork + [("X", "Y")]
    xnode = {"type": "continuous",
             "functional_form": {"name": "sin", "weights": {z: 1.0 for z in Z}},
             "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.7}}
    base_w = {z: 0.7 for z in Z}; base_d = {z: 2 for z in Z}       # Y mean = 0.7 sum Z^2

    def poly_y(extra_w, extra_d, noise):
        return {"type": "continuous",
                "functional_form": {"name": "polynomial",
                                    "weights": {**base_w, **extra_w},
                                    "degrees": {**base_d, **extra_d}},
                "noise_model": noise}

    def cfg(edges, y):
        return {"simulation_params": _sp(n, seed, dS),
                "graph_params": {"type": "custom", "nodes": Z + ["X", "Y"], "edges": edges},
                "node_params": {**src, "X": xnode, "Y": y}}
    gauss = {"name": "additive", "dist": "gaussian", "std": 1.0}
    return {
        "null":           cfg(fork, poly_y({}, {}, gauss)),
        "linear_mean":    cfg(tri,  poly_y({"X": 0.6}, {"X": 1}, gauss)),
        "nonmonotone_z2": cfg(tri,  poly_y({"X": 0.5}, {"X": 2}, gauss)),
        "scale":          cfg(tri,  poly_y({"X": 0.0}, {"X": 1},
                                           {"name": "heteroskedastic", "func": _scale_X})),
        "tail_shape":     cfg(tri,  poly_y({"X": 0.0}, {"X": 1},
                                           {"name": "shape", "func": _skew_X, "std": 1.0})),
    }


# §5.4  The conditional-tail edge (centerpiece) ----------------------------

def tail_edge_sweep(n: int = N_SCALE, seed: int = 1, dS: int = 1,
                    strengths=(1.0, 2.0, 3.0, 4.0)) -> dict:
    """X drives the conditional SKEW of Y (mean and variance held fixed), query
    X _||_ Y | Z. Skew strength alpha = c*X; small c = weak effect, the
    decay-as-effect-weakens panel. Includes the matching null (no X->Y edge)."""
    Z, src = _znodes(dS, "gauss")
    fork = _fork(Z); tri = fork + [("X", "Y")]
    xnode = _mean_over_Z(Z)                                # X depends linearly on Z
    # Y's mean is linear in Z with X EXPLICITLY excluded (weight 0) -- dagsampler
    # auto-weights unlisted parents, so X must be pinned to 0 to keep the mean fixed;
    # X enters Y ONLY through the skew of the noise.
    ymean = {"name": "linear", "weights": {**{z: 1.0 for z in Z}, "X": 0.0}}

    def y_skew(skew_fn):
        return {"type": "continuous", "functional_form": ymean,
                "noise_model": {"name": "shape", "func": skew_fn, "std": 1.0}}

    cfgs = {"null": {"simulation_params": _sp(n, seed, dS),
                     "graph_params": {"type": "custom", "nodes": Z + ["X", "Y"], "edges": fork},
                     "node_params": {**src, "X": xnode, "Y": _mean_over_Z(Z)}}}
    for c in strengths:
        fn = (lambda cc: (lambda p: cc * p["X"].to_numpy()))(c)
        cfgs[f"alpha_{c:g}"] = {"simulation_params": _sp(n, seed, dS),
                                "graph_params": {"type": "custom",
                                                 "nodes": Z + ["X", "Y"], "edges": tri},
                                "node_params": {**src, "X": xnode, "Y": y_skew(fn)}}
    return cfgs


# §5.5  Mixed-type conditioning  -> tab_mixed_conditional (existing)
# §5.6  Computational cost       -> single_edge_at_n (existing; speed harness sweeps n, |S|)
# §5.7  Semi-synthetic real-data -> injection_skewnorm.py (NOT dagsampler; real marginals)


# §5.8  GFCM inside PC -----------------------------------------------------

def pc_discovery(n: int = N_SCALE, p: int = 8, edge_prob: float = 0.4,
                 seed_structure: int = 7, seed_data: int = 1, max_cond: int = 4) -> dict:
    """A connected random DAG whose mechanisms span the detection ladder INCLUDING
    tail-shape and scale edges (TYPES_TAIL), so discovery accuracy (SHD, edge P/R)
    rewards a test that sees beyond the conditional mean. Two-seed (structure
    fixed across reps, data varies) as in Paper 1. The companion edge-type map is
    available via ``random_typed_dag(..., types=TYPES_TAIL)`` for per-type recall."""
    cfg, _ = random_typed_dag(p, edge_prob, seed_structure, seed_data, types=TYPES_TAIL)
    cfg["simulation_params"]["n_samples"] = n
    cfg["simulation_params"]["store_ci_oracle"] = True
    cfg["simulation_params"]["ci_oracle_max_cond_set"] = max_cond
    return cfg


# Appendix: graceful degradation -------------------------------------------

def growing_S(n: int = N_SCALE, seed: int = 1, dS_list=(1, 2, 3, 5, 8)) -> dict:
    """The tail-shape alternative under a GROWING conditioning set: power should
    decay gracefully (more nuisance to fit), not collapse. One alt per |S|."""
    return {f"dS{dS}": tail_edge_sweep(n, seed, dS=dS, strengths=(4.0,))["alpha_4"]
            for dS in dS_list}


# Plan registry: §-section -> builder. Statistical panels default to N_SCALE;
# pass n=N_MODERATE for the par_cop/KCI companion and the detection-separation runs.
PLAN = {
    "5.2_calibration":  calib_nulls,           # hard nulls x |S| x n
    "5.3_power_ladder": power_ladder,          # typed alts x |S|
    "5.4_tail_edge":    tail_edge_sweep,       # skew strength sweep (centerpiece)
    "5.5_mixed":        tab_mixed_conditional,  # mixed-type conditioning (conjunction)
    "5.6_speed":        single_edge_at_n,      # speed vs n / |S|
    "5.8_pc":           pc_discovery,          # PC on tail-mechanism graphs (matched-FP)
    "app_ablation":     tab_ablation,          # which block catches which moment
    "app_nuisance":     nuisance_null,         # spline/poly/gb/rf backend robustness
    "app_cardinality":  cardinality_alt,       # categorical cardinality (MI-matched)
    "app_growing_S":    growing_S,             # graceful decay in |S|
}


def plan_smoke(n: int = 400):
    """Build + oracle-check every PLAN config at small n (skips the speed builder,
    which carries no oracle). Prints one line per config; raises on any failure."""
    calls = {
        "5.2_calibration":  lambda: calib_nulls(n, 1, dS=3),
        "5.3_power_ladder": lambda: power_ladder(n, 1, dS=3),
        "5.4_tail_edge":    lambda: tail_edge_sweep(n, 1, dS=1),
        "5.5_mixed":        lambda: tab_mixed_conditional(n, 1),
        "5.8_pc":           lambda: pc_discovery(n, p=6),
        "app_ablation":     lambda: tab_ablation(n, 1),
        "app_nuisance":     lambda: {"smooth": nuisance_null("smooth", n)},
        "app_cardinality":  lambda: {"J3": cardinality_alt(3, n, 0.6, 1)},
        "app_growing_S":    lambda: growing_S(n, 1, dS_list=(1, 3)),
    }
    ok = 0
    for sec, build in calls.items():
        out = build()
        items = out.items() if "simulation_params" not in out else [("(single)", out)]
        for label, cfg in items:
            res = CausalDataGenerator(cfg).simulate()
            data = res["data"]
            if cfg["simulation_params"].get("store_ci_oracle"):   # nuisance/cardinality are known-by-construction
                assert "ci_oracle" in res, f"{sec}/{label}: no oracle"
            assert np.isfinite(data.select_dtypes("number").to_numpy()).all(), \
                f"{sec}/{label}: non-finite data"
            ok += 1
            print(f"  ok  {sec:18s} {label}")
    print(f"\n{ok} PLAN configs built + oracle-checked OK")
