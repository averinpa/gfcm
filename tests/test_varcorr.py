"""Regression tests for the finite-sample variance correction (gfcm/varcorr.py).

The correction is estimated by Rademacher projections, so its p-value carries probe noise. Two
earlier routes computed the same Delta without projections: "probes" (P x P training objects plus
Hutchinson probes) and "exact" (the full m x m pair matrices). They were removed once the fast route
was validated against them; REFERENCE below holds the exact route's p-values on these four draws, so
the check they provided survives their deletion. FROZEN holds the shipped route's own values at
seed 0, which pin it against any change.
"""
import warnings

import numpy as np
import pytest

from gfcm import GFCMConfig, unified_test
from gfcm.varcorr import _BAND, design_width, max_cond_set_for

FOLDS = 5


def draw(n, dS, seed, alt=0.0):
    """Heteroskedastic, heavy-tailed, nonlinear-mean draw; alt > 0 puts X into Y's mean."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, dS))
    m = np.tanh(Z[:, 0]) + 0.5 * Z[:, min(1, dS - 1)] ** 2
    s = np.exp(0.3 * Z[:, 0])
    X = m + s * rng.standard_t(5, n) / np.sqrt(5 / 3)
    Y = m + s * rng.standard_t(5, n) / np.sqrt(5 / 3) + alt * X
    return X, Y, Z

# (n, dS, seed, alt): p-value of the removed "exact" route, and the tolerance the probe spread over
# eight probe seeds supports on that cell (sd .0008 to .027 depending on n and P/n_train).
REFERENCE = {
    (2000, 5, 21, 0.0): (0.22462, 0.05),
    (2000, 8, 22, 0.0): (0.93867, 0.02),
    (1000, 5, 13, 0.0): (0.06569, 0.03),
    (800, 3, 14, 0.08): (0.00125, 0.00300),
}

# The shipped route at seed 0, exact mode. Any change here is a change in the test's output.
FROZEN = {
    (2000, 5, 21, 0.0): 0.2292850287,
    (2000, 8, 22, 0.0): 0.9368975276,
    (1000, 5, 13, 0.0): 0.07588897024,
    (800, 3, 14, 0.08): 0.0009779638047,
}


@pytest.mark.parametrize("cell", list(REFERENCE))
def test_matches_the_removed_exact_route(cell):
    """The projected Delta agrees with the exact one within the probe spread, and decides the same."""
    ref, tol = REFERENCE[cell]
    p = float(unified_test(*draw(*cell), config=GFCMConfig(varcorr_decide=None), seed=0))
    assert abs(p - ref) <= tol, f"{cell}: {p:.5f} vs exact route {ref:.5f}"
    assert (p <= 0.05) == (ref <= 0.05), f"{cell}: decision at .05 differs from the exact route"


@pytest.mark.parametrize("cell", list(FROZEN))
def test_frozen_values(cell):
    """Pin the shipped route: same draw, same seed, same p-value."""
    p = float(unified_test(*draw(*cell), config=GFCMConfig(varcorr_decide=None), seed=0))
    assert p == pytest.approx(FROZEN[cell], rel=1e-6)


@pytest.mark.parametrize("cell", list(FROZEN))
def test_correction_never_lowers_the_block_p_values(cell, monkeypatch):
    """Delta is positive semi-definite on the kept rows, so no block p-value can go down.

    Both numbers come from one call: correct_blocks returns the corrected block p-values and, as
    p_raw, the uncorrected ones on the same kept rows, which is the comparison the lemma is about.
    """
    import gfcm.varcorr as vc
    seen = {}
    original = vc.correct_blocks

    def spy(*args, **kwargs):
        out = original(*args, **kwargs); seen.update(out); return out

    monkeypatch.setattr(vc, "correct_blocks", spy)
    unified_test(*draw(*cell), config=GFCMConfig(varcorr_decide=None), seed=0)
    assert seen, "the correction did not run"
    for b, (pc, pr) in enumerate(zip(seen["p_blocks"], seen["p_raw"])):
        assert pc >= pr - 1e-12, f"{cell} block {b}: corrected {pc:.6f} below uncorrected {pr:.6f}"
    for b, e in enumerate(seen["etas"]):
        assert (np.asarray(e) >= -1e-10).all(), f"{cell} block {b}: negative Delta diagonal {e}"


@pytest.mark.parametrize("cell", list(FROZEN))
def test_lazy_equals_exact_inside_the_band(cell):
    """Lazy mode returns a lower bound, and it is the exact p whenever that p is inside the band."""
    X, Y, Z = draw(*cell)
    p_lazy = float(unified_test(X, Y, Z, config=GFCMConfig(), seed=0))
    p_exact = float(unified_test(X, Y, Z, config=GFCMConfig(varcorr_decide=None), seed=0))
    assert p_lazy <= p_exact + 1e-12
    if p_lazy <= _BAND:
        assert p_lazy == pytest.approx(p_exact, rel=1e-9)


def test_refuses_outside_the_domain():
    """More design columns than training rows: NaN with a RuntimeWarning, never a number."""
    rng = np.random.default_rng(5)
    n, dS = 1000, 20
    assert design_width(dS, n) > n * (1 - 1.0 / FOLDS)
    Z = rng.normal(size=(n, dS))
    X = Z[:, 0] + rng.normal(size=n); Y = Z[:, 0] + rng.normal(size=n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = unified_test(X, Y, Z, config=GFCMConfig())
    assert np.isnan(p)
    assert any(issubclass(c.category, RuntimeWarning) for c in caught)


def test_max_cond_set_for_is_the_boundary():
    for n in (1000, 2000, 10000):
        d = max_cond_set_for(n, FOLDS)
        n_tr = int(n * (1 - 1.0 / FOLDS))
        assert design_width(d, n) < n_tr <= design_width(d + 1, n)


@pytest.mark.parametrize("K", [3, 6])
def test_label_invariance_with_a_categorical_conditioner(K):
    """Renumbering a nominal conditioner's levels cannot move the p-value."""
    rng = np.random.default_rng(7)
    n = 1500
    Zc = rng.integers(0, K, n); Zn = rng.normal(size=n)
    X = np.sin(Zc) + 0.5 * Zn + rng.normal(size=n)
    Y = np.sin(Zc) - 0.3 * Zn + rng.normal(size=n)
    Z = np.column_stack([Zc.astype(float), Zn])
    kw = dict(config=GFCMConfig(), z_cat=[True, False], seed=0)
    base = float(unified_test(X, Y, Z, **kw))
    for s in range(1, 5):
        perm = np.random.default_rng(100 + s).permutation(K)
        Zp = np.column_stack([perm[Zc].astype(float), Zn])
        assert float(unified_test(X, Y, Zp, **kw)) == pytest.approx(base, abs=1e-8)


# (conditioners, levels): 2x3 and 3x3 take the saturated joint-cell design, 3x8 is 512 cells and
# falls back to the pairwise products. Both paths have to be invariant, and the pairwise one was
# not: it dropped the first level of each factor, which is the reference-level defect again.
@pytest.mark.parametrize("dS,K", [(2, 3), (3, 3), (3, 8)])
def test_label_invariance_across_several_categorical_conditioners(dS, K):
    """The interaction block across categorical conditioners must be invariant too."""
    rng = np.random.default_rng(1)
    n = 2000
    Zc = np.column_stack([rng.integers(0, K, n).astype(float) for _ in range(dS)])
    base_m = np.sin(Zc.sum(1))
    X = base_m + rng.normal(size=n); Y = base_m + rng.normal(size=n)
    kw = dict(config=GFCMConfig(), z_cat=[True] * dS, seed=0)
    base = float(unified_test(X, Y, Zc, **kw))
    for s in range(1, 5):
        g = np.random.default_rng(50 + s)
        Zp = np.column_stack([g.permutation(K)[Zc[:, j].astype(int)].astype(float) for j in range(dS)])
        assert float(unified_test(X, Y, Zp, **kw)) == pytest.approx(base, abs=1e-8)
