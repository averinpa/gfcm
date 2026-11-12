"""PC discovery via cbcd (the project's PC algorithm).

Runs cbcd's PC with any data-bound CI test that satisfies the callable ``test(x, y, S) -> p``
shape (the GFCM adapter `GFCMCITest` does). The `CbcdTest` wrapper adds the pieces cbcd's CITest
protocol needs on top of a bare callable -- `details()` returning a CITestResult -- and, when a
`log` list is supplied, records every CI query as (x, y, S, p) for the per-decision discovery log
(cbcd's built-in RunRecorder is a stub, so decisions are captured here at the adapter).
"""
from __future__ import annotations

import cbcd                                       # pinned dependency (see requirements.txt)
import networkx as nx
import numpy as np
from cbcd.citest import CITestResult


class CbcdTest:
    """Adapt a callable(x, y, S)->p to cbcd's CITest protocol (n_vars + __call__ + details).
    If a list `log` is given, every CI query is appended as (x, y, S, p)."""

    def __init__(self, fn, n_vars, log=None):
        self._fn = fn
        self.n_vars = int(n_vars)
        self._log = log

    def __call__(self, x, y, S):
        p = float(self._fn(int(x), int(y), [int(s) for s in S]))
        if self._log is not None:
            self._log.append((int(x), int(y), tuple(int(s) for s in S), p))
        return p

    def details(self, x, y, S):
        return CITestResult(p_value=self(x, y, S))


def run_pc(data, nodes, test_callable, alpha=0.05, max_cond_set=None, log=None, collider="sepset"):
    """Run cbcd PC with a uniform CI test; return the cbcd CPDAG. Pass log=[] to capture every
    CI decision as (x, y, S, p).

    collider: "sepset" reads the single recorded separating set (PC as published); "majority" re-tests every
    conditioning subset of an unshielded triple and orients a collider only when the middle node lies in fewer
    than half of the separating subsets, leaving the ambiguous triples unoriented (Colombo and Maathuis 2014).
    The majority rule costs many more CI calls but is order-independent."""
    ci = CbcdTest(test_callable, np.asarray(data).shape[1], log=log)
    orient = cbcd.collider.MajorityColliderOrienter() if collider == "majority" else None
    return cbcd.pc(data, ci_test=ci, alpha=alpha, max_cond_set=max_cond_set, var_names=list(nodes),
                   collider=orient)


def cpdag_skeleton(cg, nodes):
    """Undirected skeleton (node-name frozensets) from a cbcd CPDAG."""
    idx = {frozenset((a, b)) for (a, b) in cg.directed_edges()} | set(cg.undirected_edges())
    out = set()
    for fs in idx:
        i, j = tuple(fs)
        out.add(frozenset((nodes[i], nodes[j])))
    return out


class DSepOracle:
    """d-separation oracle on the true DAG, in cbcd's CITest shape (p = 1 separated, 0 connected)."""

    def __init__(self, G, p, log=None):
        self.G, self.n_vars, self._log = G, p, log
        self._dsep = nx.is_d_separator if hasattr(nx, "is_d_separator") else nx.d_separated

    def __call__(self, x, y, S):
        S = [int(s) for s in S]
        if self._log is not None:
            self._log.append(len(S))
        return 1.0 if self._dsep(self.G, {int(x)}, {int(y)}, set(S)) else 0.0

    def details(self, x, y, S):
        return cbcd.CITestResult(p_value=self(x, y, S))


def true_cpdag(edges_named, nodes, alpha=0.05):
    """The CPDAG of the true DAG: PC driven by the oracle, so the Markov equivalence class is exact."""
    idx = {v: i for i, v in enumerate(nodes)}
    G = nx.DiGraph(); G.add_nodes_from(range(len(nodes)))
    G.add_edges_from((idx[a], idx[b]) for a, b in edges_named)
    return cbcd.pc(np.zeros((4, len(nodes))), ci_test=DSepOracle(G, len(nodes)), alpha=alpha,
                   var_names=list(nodes))
