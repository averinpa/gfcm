"""Smoke test: GFCM calibration, power, and runtime on a small battery of DGPs.

Run directly (`python tests/test_smoke.py`) for a printed report, or under pytest
(`pytest tests/test_smoke.py`) for pass/fail assertions on loose calibration/power bounds.
The bounds are deliberately slack (this guards against gross regressions, not exact numbers).
"""
import time

import numpy as np

from gfcm.core import unified_test

_R = 150          # replications per rejection-rate estimate
_ALPHA = 0.05


# --- DGPs: (X, Y, Z) with a shared rng so runs are reproducible ---
def _nsin(rng, m, heavy):
    """Wiggly-mean null (misspecified additive mean); heavy => Student-t(2) conditioner."""
    Z = (rng.standard_t(2, size=m) / 2) if heavy else rng.normal(size=m)
    s = np.sin(3 * Z)
    return s + rng.normal(size=m), s + rng.normal(size=m), Z[:, None]


def _hvy_null(rng, m):
    """Heavy-tailed heteroscedastic null (X and Y independent given Z)."""
    Z = rng.normal(size=(m, 2))
    X = Z[:, 0] + np.exp(0.4 * Z[:, 1]) * rng.standard_t(3, size=m)
    return X, Z[:, 1] + np.exp(0.4 * Z[:, 0]) * rng.standard_t(3, size=m), Z


def _lin(rng, m):
    """Linear-mean alternative."""
    Z = rng.normal(size=(m, 2)); X = Z[:, 0] + rng.normal(size=m)
    return X, Z[:, 1] + 0.9 * (X - Z[:, 0]) + rng.normal(size=m), Z


def _z2(rng, m):
    """Non-monotone (z^2) mean alternative."""
    Z = rng.normal(size=(m, 2)); X = Z[:, 0] + rng.normal(size=m); xc = X - Z[:, 0]
    return X, Z[:, 1] + 1.5 * (xc ** 2 - 1) + rng.normal(size=m), Z


def _scale(rng, m):
    """Scale (tail) alternative: X drives the spread of Y, conditional mean flat."""
    Z = rng.normal(size=(m, 2)); X = Z[:, 0] + rng.normal(size=m)
    return X, Z[:, 1] + np.exp(0.6 * (X - Z[:, 0])) * rng.standard_t(4, size=m), Z


def _skew(rng, m, g=1.2):
    """Conditional-skew alternative (mean and variance flat in X)."""
    Z = rng.normal(size=(m, 2)); X = Z[:, 0] + rng.normal(size=m); xc = X - Z[:, 0]
    e = rng.normal(size=m); d = g * np.tanh(xc)
    es = (e + d * (e ** 2 - 1) / np.sqrt(2)) / np.sqrt(1 + d ** 2)
    return X, Z[:, 1] + es, Z


def _cat_null(rng, m):
    """Categorical X and continuous Y, each driven by Z only -> X independent of Y given Z."""
    Z = rng.normal(size=(m, 2))
    X = np.digitize(Z[:, 0] + rng.normal(size=m), [-0.6, 0.6]).astype(float)   # 3 integer levels
    return X, Z[:, 1] + rng.normal(size=m), Z


def _cat_alt(rng, m):
    """Categorical X drives continuous Y given Z (mixed-type alternative)."""
    Z = rng.normal(size=(m, 2))
    X = np.digitize(Z[:, 0] + rng.normal(size=m), [-0.6, 0.6]).astype(float)
    return X, Z[:, 1] + 1.2 * X + rng.normal(size=m), Z


def _reject_rate(dgp, m, reps=_R):
    rng = np.random.default_rng(0)
    return sum(unified_test(*dgp(rng, m)) < _ALPHA for _ in range(reps)) / reps


# --- pytest entry points: loose bounds that only catch gross regressions ---
def test_null_calibration():
    assert _reject_rate(lambda r, m: _nsin(r, m, False), 1000) <= 0.12
    assert _reject_rate(lambda r, m: _nsin(r, m, True), 1000) <= 0.12
    assert _reject_rate(_hvy_null, 1000) <= 0.12


def test_power():
    assert _reject_rate(_lin, 2000) >= 0.80
    assert _reject_rate(_z2, 2000) >= 0.80
    assert _reject_rate(_scale, 2000) >= 0.80
    assert _reject_rate(_skew, 2000) >= 0.60


def test_mixed_calibration():
    # categorical X auto-detected; null must stay near nominal
    assert _reject_rate(_cat_null, 1500) <= 0.12


def test_mixed_power():
    # categorical X driving continuous Y must be detected
    assert _reject_rate(_cat_alt, 2000) >= 0.75


def test_mixed_marginal_runs():
    # regression guard: marginal (|Z|=0) categorical-vs-continuous must return a valid p, not crash
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, 500).astype(float)
    p = unified_test(X, rng.normal(size=500), np.empty((500, 0)))
    assert 0.0 <= p <= 1.0


def _report():
    print("GFCM smoke report (rank-Z + spline + scale/quantile transforms + Cauchy)")
    print("  null sin gauss :", _reject_rate(lambda r, m: _nsin(r, m, False), 1000))
    print("  null sin heavy :", _reject_rate(lambda r, m: _nsin(r, m, True), 1000))
    print("  null heavy-het :", _reject_rate(_hvy_null, 1000))
    print("  lin power      :", _reject_rate(_lin, 2000))
    print("  z2 power       :", _reject_rate(_z2, 2000))
    print("  scale power    :", _reject_rate(_scale, 2000))
    print("  skew power     :", _reject_rate(_skew, 2000))
    print("  mixed cat null :", _reject_rate(_cat_null, 1500))
    print("  mixed cat power:", _reject_rate(_cat_alt, 2000))
    rng = np.random.default_rng(0)
    for m in (50000, 100000):
        X, Y, Z = _scale(rng, m)
        t0 = time.perf_counter(); unified_test(X, Y, Z)
        print(f"  timing {m:>6d} : {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    _report()
