"""PC discovery via cbcd (the project's PC algorithm).

Runs cbcd's PC with any data-bound CI test that satisfies the callable ``test(x, y, S) -> p``
shape (the GFCM adapter `GFCMCITest` does). The `CbcdTest` wrapper adds the pieces cbcd's CITest
protocol needs on top of a bare callable -- `details()` returning a CITestResult -- and, when a
`log` list is supplied, records every CI query as (x, y, S, p) for the per-decision discovery log
(cbcd's built-in RunRecorder is a stub, so decisions are captured here at the adapter).
"""
from __future__ import annotations

import cbcd                                       # pinned dependency (see requirements.txt)
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


def run_pc(data, nodes, test_callable, alpha=0.05, max_cond_set=None, log=None):
    """Run cbcd PC with a uniform CI test; return the cbcd CPDAG. Pass log=[] to capture every
    CI decision as (x, y, S, p)."""
    ci = CbcdTest(test_callable, np.asarray(data).shape[1], log=log)
    return cbcd.pc(data, ci_test=ci, alpha=alpha, max_cond_set=max_cond_set, var_names=list(nodes))


def cpdag_skeleton(cg, nodes):
    """Undirected skeleton (node-name frozensets) from a cbcd CPDAG."""
    idx = {frozenset((a, b)) for (a, b) in cg.directed_edges()} | set(cg.undirected_edges())
    out = set()
    for fs in idx:
        i, j = tuple(fs)
        out.add(frozenset((nodes[i], nodes[j])))
    return out
