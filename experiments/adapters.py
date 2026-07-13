"""Test-name -> data-bound CI test factory for the benchmark runners.

GFCM is wired via the packaged `gfcm` primitive (config-based, no environment variables). The
competitor tests (FFCI, BLITZ, RCoT, boosted GCM, partial-copula, ci_mm, ...) live in
`competitors/competitors.py` and require the Docker environment (R, a JRE + the Tetrad jar, and
the `citests` package). They are imported lazily, so a GFCM-only run needs none of that; asking
for a competitor without the environment raises a clear error when the test is built.
"""
import os
import sys

from gfcm import GFCMConfig
from gfcm_citest import GFCMCITest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "competitors"))

CANONICAL = GFCMConfig()   # the published configuration


def make_test(name, data, config=None):
    """Return a data-bound CI test callable ``(x, y, S) -> p_value`` for `name`."""
    if name == "gfcm":
        return GFCMCITest(data, config=config or CANONICAL)
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
    if name == "gfcm":
        c = config or CANONICAL
        return (f"gfcm:scale={c.scale},calib={c.calib},gcv={c.gcv},proj={c.proj},"
                f"rankz={c.rankz},inter={c.inter},combine={c.combine}")
    if name == "ffci":
        return "ffci:tetrad_default"
    return f"{name}:default"
