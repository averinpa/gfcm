"""Test-name -> data-bound CI test factory for the benchmark runners.

GFCM is wired via the packaged `gfcm` primitive (config-based, no environment variables). The
competitor tests (FFCI, BLITZ, RCoT, boosted GCM, partial-copula, ci_mm, ...) live in
`competitors/competitors.py` and require the Docker environment (R, a JRE + the Tetrad jar, and
the `citests` package). They are imported lazily, so a GFCM-only run needs none of that; asking
for a competitor without the environment raises a clear error when the test is built.
"""
import os
import sys

import gfcm_submitted
from gfcm import GFCMConfig
from gfcm_citest import GFCMCITest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "competitors"))

CANONICAL = GFCMConfig()                                     # the test, correction on and lazily evaluated
EXACT = GFCMConfig(varcorr_decide=None)                      # the same test, exact corrected p on every call
_GFCM = {"gfcm": CANONICAL, "gfcm_exact": EXACT,
         "gfcm_vc": EXACT, "gfcm_lazy": CANONICAL}          # (gfcm_vc / gfcm_lazy: older names, kept for the 23-24 Oct runs)
# The "as submitted" arm is the submitted CODE, not a flag on the current one: the current test
# differs from it in the categorical design as well as the variance correction, so no configuration
# of gfcm/ reproduces it. gfcm_submitted.py is the verbatim release copy.
PUBLISHED_CONFIG = gfcm_submitted.GFCMConfig()
submitted_test = gfcm_submitted.unified_test


def make_test(name, data, config=None):
    """Return a data-bound CI test callable ``(x, y, S) -> p_value`` for `name`."""
    if name == "gfcm_published":
        return _SubmittedCITest(data)
    if name in _GFCM:
        return GFCMCITest(data, config=config or _GFCM[name])
    import competitors as C  # lazy: a GFCM-only run never pulls in the R / Java / citests stack
    if name not in C.FACTORIES:
        raise ValueError(f"unknown test: {name!r}")
    try:
        return C.FACTORIES[name](data)               # citests / FFCI backends import here
    except ImportError as e:                          # backend absent (no Docker env)
        raise NotImplementedError(
            f"competitor test '{name}' needs the Docker environment (R / JRE / citests); "
            f"backend import failed: {e}. Run with --only gfcm for a GFCM-only reproduction.") from e


def config_id(name, config=None):
    """Provenance string for a test's exact configuration."""
    if name == "gfcm_published":
        return "gfcm_published:TMLR-submission code (experiments/gfcm_submitted.py)"
    if name in _GFCM:
        c = config or _GFCM[name]
        return (f"{name}:gcv={c.gcv},proj={c.proj},"
                f"rankz={c.rankz},inter={c.inter},"
                f"comove={c.comove},across={c.across},lazy={c.varcorr_decide}")
    if name == "ffci":
        return "ffci:tetrad_default"
    return f"{name}:default"


class _SubmittedCITest(GFCMCITest):
    """The submitted test bound to a dataset, same interface as GFCMCITest."""

    def __init__(self, data, cat=None):
        super().__init__(data, cat=cat, config=PUBLISHED_CONFIG)

    def __call__(self, x, y, S):
        S = [int(s) for s in S]
        return float(submitted_test(
            self.data[:, x], self.data[:, y], self.data[:, S], config=self.config,
            x_cat=bool(self.is_cat[x]), y_cat=bool(self.is_cat[y]),
            z_cat=(self.is_cat[S] if S else None)))
