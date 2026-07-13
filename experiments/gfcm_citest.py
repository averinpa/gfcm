"""Reference adapter: GFCM as a cbcd-compatible CITest.

Satisfies cbcd's CITest protocol -- an ``n_vars`` attribute and ``__call__(x, y, S) -> float``
returning the p-value -- by duck typing, so no cbcd import is required. This is the layer that
bridges cbcd's index-based CI queries and GFCM's typed interface.

The point of this adapter: the variable-type mask is resolved ONCE at construction (dataset-level)
via ``gfcm.infer_types``, then looked up by column index on every CI test the PC/FCI algorithm
issues. A variable's type is therefore fixed dataset-wide -- never auto-detected per call on a
subsample -- which is what keeps type handling correct and cheap inside constraint-based discovery.

Usage
-----
    from cbcd import pc                       # cbcd's PC algorithm
    from experiments.gfcm_citest import GFCMCITest

    ci = GFCMCITest(data, cat=is_cat_mask)    # declare the type mask once (or cat=None to auto-detect once)
    cpdag = pc(data, ci_test=ci)              # PC threads column indices; the adapter looks up types
"""
from collections.abc import Sequence

import numpy as np

from gfcm import GFCMConfig, infer_types, unified_test


class GFCMCITest:
    """GFCM conditional-independence test bound to a dataset, for constraint-based discovery.

    Parameters
    ----------
    data : array (n, p)
        The full dataset. Columns are addressed by index in ``__call__``.
    cat : optional
        Type declaration passed to ``infer_types`` (length-p bool sequence = complete declaration;
        dict {col: bool} = partial override; None = auto-detect every column once).
    config : GFCMConfig, optional
        GFCM configuration; defaults to the canonical published configuration.
    taus, K, B, seed :
        Optional per-call overrides forwarded to ``unified_test``.
    """

    n_vars: int

    def __init__(self, data, cat=None, config=None, taus=None, K=None, B=None, seed=0):
        self.data = np.ascontiguousarray(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-D (n x p), got shape {self.data.shape}")
        self.n_samples, self.n_vars = self.data.shape
        self.config = config if config is not None else GFCMConfig()
        self.is_cat = infer_types(self.data, declared=cat)   # bool[n_vars], resolved ONCE
        self.taus, self.K, self.B, self.seed = taus, K, B, seed

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        """p-value for H0: variable x independent of variable y given the columns in S."""
        S = [int(s) for s in S]
        return unified_test(
            self.data[:, x], self.data[:, y], self.data[:, S],
            config=self.config, taus=self.taus, K=self.K, B=self.B, seed=self.seed,
            x_cat=bool(self.is_cat[x]),
            y_cat=bool(self.is_cat[y]),
            z_cat=(self.is_cat[S] if S else None),
        )
