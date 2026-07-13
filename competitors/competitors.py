"""Competitor CI tests for the benchmark panel.

These require the Docker environment (R + BLITZ/RCIT, a Temurin JRE + the Tetrad jar for FFCI, and
the `citests` package); they are NOT importable in a GFCM-only run. Each exposes the uniform
callable ``test(x, y, S) -> p`` shape, so they slot into run_cells / run_pc exactly like the GFCM
adapter. Imports of the heavy backends are deferred to first use so this module loads even when a
backend is absent -- calling the test is what raises.

Bridges:
  * BLITZ, ParCop (par_cop)  -- managed R subprocess (blitz_server.R / par_cop_server.R),
  * FFCI                  -- Tetrad via jpype (ffci_tetrad.py),
  * rcot / gcm_boosted / ci_mm / ...  -- the citests package.
"""
import atexit as _atexit
import os as _os
import subprocess as _sp
import tempfile as _tf

import numpy as np

_HERE = _os.path.dirname(_os.path.abspath(__file__))


def _qseed(X, Y, S):
    return hash((int(X), int(Y), tuple(sorted(int(s) for s in S)))) & 0x7FFFFFFF


def _kill_server(cls):
    """Terminate and reap a bridge class's managed R subprocess (close stdin -> kill -> wait), so a
    dropped handle never leaves an orphaned Rscript blocked on readLines. Safe on a None handle."""
    p = getattr(cls, "_proc", None)
    cls._proc = None
    if p is None:
        return
    for step in (lambda: p.stdin and p.stdin.close(), p.kill, lambda: p.wait(timeout=5)):
        try:
            step()
        except Exception:
            pass


class _RServerTest:
    """Base for a CI test backed by a persistent Rscript server that reads a CSV path on stdin and
    writes one p-value on stdout. Subclasses set SCRIPT and (optionally) _parse / _jitter."""
    SCRIPT = None
    _proc = None

    def __init__(self, data, **kw):
        self.data = np.asarray(data, dtype=float)

    @classmethod
    def _server(cls):
        if cls._proc is not None and cls._proc.poll() is not None:
            _kill_server(cls)
        if cls._proc is None:
            cls._proc = _sp.Popen(["Rscript", _os.path.join(_HERE, cls.SCRIPT)],
                                  stdin=_sp.PIPE, stdout=_sp.PIPE, text=True,
                                  stderr=_sp.DEVNULL, bufsize=1, cwd=_HERE)
        return cls._proc

    def _columns(self, X, Y, S, rng):
        x, y = self.data[:, int(X)], self.data[:, int(Y)]
        z = self.data[:, S] if S else np.zeros((len(self.data), 0))
        return np.column_stack([x, y] + ([z] if S else []))

    def _parse(self, line):
        return float(line.strip())

    def __call__(self, X, Y, condition_set=None):
        S = [int(s) for s in condition_set] if condition_set else []
        rng = np.random.default_rng(_qseed(X, Y, S))
        arr = self._columns(X, Y, S, rng)
        fd, path = _tf.mkstemp(suffix=".csv"); _os.close(fd)
        np.savetxt(path, arr, delimiter=",")
        try:
            p = self._server()
            p.stdin.write(path + "\n"); p.stdin.flush()
            line = p.stdout.readline()
            if not line:                             # server crashed/EOF -> kill+reap, restart next call
                _kill_server(type(self)); return float("nan")
            return self._parse(line)
        except (BrokenPipeError, OSError, ValueError, IndexError):
            _kill_server(type(self)); return float("nan")
        finally:
            try:
                _os.remove(path)
            except OSError:
                pass


class BLITZ(_RServerTest):
    """Strobl's BLITZ (ci100), in a managed R subprocess (its C++ can segfault on near-degenerate
    inputs; isolating it means a crash kills only the server). Discrete columns get tiny
    deterministic jitter (the standard way to run a continuous-only test on mixed data)."""
    SCRIPT = "blitz_server.R"

    @staticmethod
    def _jit(a, rng):
        a = np.asarray(a, float)
        if a.size == 0:
            return a
        sd = np.std(a, axis=0); sd = np.where(sd > 0, sd, 1.0)
        return a + rng.normal(scale=1e-3 * sd, size=a.shape)

    def _columns(self, X, Y, S, rng):
        x = self._jit(self.data[:, int(X)], rng)
        y = self._jit(self.data[:, int(Y)], rng)
        z = self._jit(self.data[:, S], rng) if S else np.zeros((len(self.data), 0))
        return np.column_stack([x, y] + ([z] if S else []))


class ParCop(_RServerTest):
    """Petersen-Hansen partial-copula CI test (par_cop, quantile-regression based), in a managed R
    subprocess; returns the q=2 (df=4) p-value."""
    SCRIPT = "par_cop_server.R"

    def _parse(self, line):
        return float(line.strip().split(",")[0])       # q=2 p-value


_atexit.register(_kill_server, BLITZ)
_atexit.register(_kill_server, ParCop)


def _ffci_factory(data):
    import ffci_tetrad as F  # lazy: Tetrad/JVM backend; this module must load without it
    return F.FFCItetrad(np.asarray(data, float))        # Tetrad FfCi at its published defaults


def _citests_factory(classname, **fixed):
    """Build a factory(data) -> callable(X, Y, S) for a citests test class."""
    def factory(data):
        import citests.tests as ct  # lazy: citests backend; this module must load without it
        cls = getattr(ct, classname)
        kw = dict(fixed)
        if classname == "CMIknnMixed":                  # tigramite needs a per-column type mask
            from citests.tests.base import _is_categorical_column  # lazy: only the CMIknnMixed path
            d = np.asarray(data, dtype=float)
            col = np.array([1 if _is_categorical_column(d[:, j]) else 0
                            for j in range(d.shape[1])], dtype=int)
            kw["data_type"] = np.tile(col, (len(d), 1))
        return cls(np.asarray(data, dtype=float), **kw)
    return factory


# name -> factory(data) -> callable(X, Y, condition_set); GFCM is handled in adapters.py
FACTORIES = {
    "ffci":         _ffci_factory,
    "blitz":        lambda data: BLITZ(data),
    "par_cop":         lambda data: ParCop(data),
    "gcm_linear":   _citests_factory("GCM", reg="linear"),
    "gcm_boosted":  _citests_factory("GCM", reg="xgb"),
    "gkcm":         _citests_factory("GKCM"),
    "wgcm":         _citests_factory("WGCM"),
    "pcm":          _citests_factory("PCM"),
    "fisherz":      _citests_factory("FisherZ"),
    "spearman":     _citests_factory("Spearman"),
    "kci":          _citests_factory("KCI"),
    "rcot":         _citests_factory("RCoT"),
    "cmiknn_mixed": _citests_factory("CMIknnMixed"),
    "ci_mm":        _citests_factory("CiMM"),
    "chisq":        _citests_factory("ChiSq"),
    "gsq":          _citests_factory("GSq"),
}
