"""Official FFCI (Ramsey 2026, ci105) via the released Tetrad jar + JPype.

This is the REAL implementation shipped in Tetrad (edu.cmu.tetrad.search.test.FfCi), not a
from-paper reimplementation. FfCi is the mixed-type variant; on continuous-only data it reduces to
the RCoT-like continuous test (FfCiContinuous). We use FfCi everywhere so a single "ffci" panel
entry handles both continuous and mixed cells, exactly as Tetrad would.

A from-paper reimplementation (ffci.py) was tried first and FAILED calibration validation
(heavy_tail dS3 n=2k = 1.000 vs official RCoT 0.096) -- median-bandwidth RFF distorts on Student-t.
Using the official jar avoids misrepresenting the competitor (see verify-competitor-claims).

Contract matches tests.py: factory(data) -> callable(X, Y, condition_set) -> p-value.
Data is a numpy float matrix; categorical columns are detected with citests' _is_categorical_column
so the type mask matches the rest of the benchmark.
"""
import os
import sys
import threading
import numpy as np

_JAR = os.environ.get(
    "TETRAD_JAR", "/tmp/py-tetrad/pytetrad/resources/tetrad-current.jar")
_PYTETRAD = os.environ.get("PYTETRAD_DIR", "/tmp/py-tetrad")  # parent: enables `import pytetrad.tools...`
_LOCK = threading.Lock()
_STARTED = False

try:
    from citests.tests.base import _is_categorical_column as _is_cat
except Exception:
    def _is_cat(col):
        u = np.unique(col)
        return len(u) <= max(2, int(0.05 * len(col))) and np.allclose(col, np.round(col))


def _ensure_jvm():
    global _STARTED
    if _STARTED:
        return
    with _LOCK:
        if _STARTED:
            return
        import jpype  # lazy: JVM lifecycle -- edu.cmu.* / java.* import only once the JVM is up
        import jpype.imports  # noqa: F401  (enables `import edu.cmu...`)
        if not jpype.isJVMStarted():
            xmx = os.environ.get("TETRAD_XMX", "4g")
            # cap JVM core use (GC + internal parallelism ignore BLAS env vars) for heat safety
            procs = os.environ.get("TETRAD_PROCS", "2")
            jvm_args = [f"-Xmx{xmx}", f"-XX:ActiveProcessorCount={procs}"]
            jpype.startJVM(jpype.getDefaultJVMPath(), *jvm_args,
                           classpath=[_JAR], convertStrings=False)
        if _PYTETRAD not in sys.path:
            sys.path.insert(0, _PYTETRAD)
        _STARTED = True


def _build_dataset(data):
    """numpy (n x p) -> Tetrad BoxDataSet. Fast DoubleDataBox for continuous-only;
    official translate converter for mixed cells."""
    _ensure_jvm()
    import jpype  # lazy: JVM-bound -- edu.cmu.* / java.* import only after _ensure_jvm()
    import edu.cmu.tetrad.data as td
    import java.util as ju

    n, p = data.shape
    cat = [bool(_is_cat(data[:, j])) for j in range(p)]
    names = [f"V{j}" for j in range(p)]

    if not any(cat):
        variables = ju.ArrayList()
        for j in range(p):
            variables.add(td.ContinuousVariable(names[j]))
        arr = jpype.JArray(jpype.JDouble, 2)(
            np.ascontiguousarray(data, dtype=np.float64))
        box = td.DoubleDataBox(arr)
        return td.BoxDataSet(box, variables)

    # mixed path: reuse the official converter (categorical cols as int dtype -> DiscreteVariable)
    import pandas as pd  # lazy: only the mixed-cell conversion path needs pandas + pytetrad
    from pytetrad.tools import translate
    cols = {}
    for j in range(p):
        if cat[j]:
            col = data[:, j].astype(np.int64)
        else:
            col = data[:, j].astype(np.float64)
        cols[names[j]] = col
    df = pd.DataFrame(cols)
    return translate.pandas_data_to_tetrad(df, int_as_cont=False)


class FFCItetrad:
    def __init__(self, data, continuous=False, **params):
        self.data = np.asarray(data, dtype=float)
        self.ds = _build_dataset(self.data)
        import edu.cmu.tetrad.search.test as tt  # lazy: JVM-bound Tetrad class (JVM up via _build_dataset)
        self.test = (tt.FfCiContinuous(self.ds) if continuous
                     else tt.FfCi(self.ds))
        # optional setters (defaults match Tetrad unless overridden via env/params)
        for k, v in params.items():
            setter = "set" + k[0].upper() + k[1:]
            if hasattr(self.test, setter):
                getattr(self.test, setter)(v)
        self.vars = list(self.ds.getVariables())

    def __call__(self, X, Y, condition_set=None):
        import java.util as ju  # lazy: JVM-bound
        S = list(condition_set) if condition_set is not None else []
        zset = ju.HashSet()
        for s in S:
            zset.add(self.vars[int(s)])
        res = self.test.checkIndependence(
            self.vars[int(X)], self.vars[int(Y)], zset)
        return float(res.getPValue())
