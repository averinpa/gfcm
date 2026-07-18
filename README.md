# GFCM — reproduction archive

Reproduction code for **"GFCM: A Tail-Sensitive Mixed-Type Conditional Independence Test
for Causal Discovery"** (Averin).

> **Scope.** This repository is a *reproduction archive*: it holds the GFCM test, the Docker
> environment, the experiment runners, the competitor adapters, and the aggregated results
> needed to regenerate every figure and table in the paper. It is **not** a packaged library —
> the reusable GFCM test will ship inside [`citests`](https://github.com/averinpa/citests).
> Do not `pip install` this repo expecting a stable API.

## Layout

```
gfcm/
├── gfcm/
│   ├── __init__.py       # exports: unified_test, GFCMConfig, infer_types, k_rule
│   └── core.py           # the GFCM test (single canonical implementation)
├── tests/test_smoke.py   # calibration / power / mixed-type smoke assertions
├── examples/quickstart.ipynb  # minimal end-to-end usage walkthrough
├── docker/Dockerfile     # fully-pinned, self-contained build of the exact environment
├── experiments/          # dgp, manifest, adapters, run_*.py (one runner per paper section),
│                         #   aggregate, discovery, metrics, gfcm_citest (cbcd CITest adapter)
├── competitors/          # FFCI / BLITZ / RCoT / partial-copula adapters
├── results/              # aggregated/ (result tables) + figures/ (output PDFs)
├── figures/make_figures.py   # regenerates the paper's figures from results/ + data/
├── datasets/eustockmarkets.csv  # checked-in raw data (EU stock markets)
└── data/                 # gitignored — loaders only (EuStock, Causal-Chamber)
```

## Using the test

```python
import numpy as np
from gfcm import unified_test, GFCMConfig

n = 2000
Z = np.random.default_rng(0).normal(size=(n, 2))
X = Z[:, 0] + np.random.default_rng(1).normal(size=n)
Y = Z[:, 1] + np.exp(0.6 * (X - Z[:, 0])) * np.random.default_rng(2).standard_t(4, size=n)

p = unified_test(X, Y, Z)                       # canonical config by default -> p-value
```

`unified_test(X, Y, Z)` returns the p-value for H0: *X independent of Y given Z*.

## Configuration

Configuration is **explicit**, via the frozen `GFCMConfig` dataclass — there are no environment
variables. The dataclass **defaults are the canonical published configuration**, so calling
`unified_test(X, Y, Z)` with no `config` reproduces the paper's test. Ablations construct a
modified config; the active configuration is logged once (via the `gfcm` logger).

```python
unified_test(X, Y, Z, config=GFCMConfig(scale="e2"))   # e.g. the squared-scale ablation
```

Key fields (defaults shown are canonical):

| field | default | meaning |
|-------|---------|---------|
| `scale`   | `"abs"`  | scale feature `\|e\|` (robust); **not** the non-robust `"e2"` |
| `calib`   | `"chi2"` | block calibration (analytic chi-squared) |
| `gcv`     | `2`      | GCV ridge selection on the scale fit |
| `proj`    | `True`   | single-index (SIR) block, n-gated to `\|S\|>=2, n>=5000` |
| `rankz`   | `True`   | rank-transform the conditioners before the spline |
| `inter`   | `True`   | degree-3 polynomial interaction block |
| `nuisance`| `"spline"`| nuisance backend (`"spline"` or `"poly"`) |
| `taus`    | `(0.1, 0.5, 0.9)` | quantile levels |

(Structural parameters — `combine`, `b`, `folds`, `ties`, `inter_maxd`, `rawz_lin`, `proj_k`,
`proj_nmin` — are also fields; see `GFCMConfig` in `gfcm/core.py`.)

## Mixed-type data

GFCM handles categorical variables natively (one-hot residual bank for a categorical X or Y;
one-hot dummies for categorical Z-columns). Declare types with the `x_cat` / `y_cat` / `z_cat`
flags; left as `None`, each is **auto-detected** by `infer_types` — an integer-coded column with
`<= 15` distinct levels is treated as categorical, else continuous.

```python
X = np.random.default_rng(0).integers(0, 3, n).astype(float)   # 3-level categorical
p = unified_test(X, Y, Z)                                       # X auto-detected as categorical
p = unified_test(X, Y, Z, x_cat=True)                          # or declared explicitly
```

**Inside constraint-based discovery, declare types once at the dataset level and pass them to
every call — do not rely on per-call auto-detection** (it re-infers per subsample, which is
wasteful and can be inconsistent). Use `infer_types` once, then thread the mask down:

```python
from gfcm import infer_types
is_cat = infer_types(data, declared=None)          # bool[p], resolved ONCE
# per CI test on columns (i, j, S):
unified_test(data[:, i], data[:, j], data[:, S],
             x_cat=bool(is_cat[i]), y_cat=bool(is_cat[j]), z_cat=is_cat[S])
```

`experiments/gfcm_citest.py` packages exactly this as a cbcd-compatible `CITest`:

```python
from cbcd import pc
from experiments.gfcm_citest import GFCMCITest
cpdag = pc(data, ci_test=GFCMCITest(data, cat=is_cat))   # types fixed once; PC threads indices
```

## Reproducing the paper

Two paths. **GFCM + citests-family competitors** run locally with the pinned Python deps; the
**full competitor panel** (adding BLITZ / FFCI, which need R / Java) runs in the Docker image.

```bash
pip install -r requirements.txt        # numpy/scipy/... + dagsampler, citests, cbcd, bnmetrics
```

Runs (all resumable; output to gitignored `data/`, aggregated tables to `results/aggregated/`):

```bash
# §5.2 calibration / §5.3 power / §5.4 tail / §5.5 mixed  (GFCM only, or add a panel)
python experiments/run_cells.py --only gfcm
python experiments/run_cells.py --panel gfcm,rcot,gcm_boosted,fisherz,ci_mm   # + citests family

python experiments/run_convergence.py --only gfcm     # §5.6 convergence sweep
python experiments/run_pc.py          --only gfcm     # §5.8/§5.10 PC discovery (+ decision logs)
python experiments/run_ablation.py                    # appendix: asym / interaction / tradeoff / rankz
python experiments/run_injection.py   --panel gfcm    # tab:inject (EuStock real-data injection)
python experiments/run_cc_injection.py                # tab:cc-inject (Causal-Chamber light tunnel)
python experiments/run_speed.py       --panel gfcm    # fig:speed timing grid

# size-corrected power tables + all figures
python experiments/aggregate.py                       # -> results/aggregated/cells_summary.parquet
python figures/make_figures.py                        # convergence, pc, speed, tail, detection
```

**Coverage.** Every table and figure now has a runner. Exact: the calibration / power / tail /
mixed / convergence / PC results, plus `tab:asym`, `tab:interaction`, `tab:nuisance-tradeoff`,
`tab:inject`, and the convergence / PC / speed / tail figures. **Reconstructions** (the original
generators were not preserved, so they show the qualitative effect but not the paper's exact
values): `tab:rankz`, the `fig:detection` localized panel, and `tab:cc-inject` (uses real
Causal-Chamber data + scale injection + PC; it shows GFCM/RCoT recovering the scale-injected edges
while the covariance family degrades, but the recall values differ from the paper).

## Reproduce with Docker (full competitor panel)

The Dockerfile is **fully pinned** (base image by digest, every package by version/SHA, the Tetrad
jar md5-verified), so building it from a clone gives the byte-identical environment. The image is
**self-contained** — it bakes in the reproduction code (`COPY . /work`), so once built nothing
depends on your host. **Build it, don't pull** — no registry, no hosting, no cost, and the code
stays inside your private clone:

```bash
git clone https://github.com/averinpa/gfcm && cd gfcm
docker build -f docker/Dockerfile -t gfcm-repro .        # ~5 GB, slow; turnkey (no placeholders)
IMG=gfcm-repro
```

Run the full pipeline in-container, mounting host dirs onto `data/` (raw per-rep output) and
`results/` (aggregated tables + figures) so everything lands on your machine (the code stays baked
into the image):

```bash
mkdir -p out/data out/results
run() { docker run --rm -v "$PWD/out/data":/work/data -v "$PWD/out/results":/work/results \
        "$IMG" "$@"; }
PANEL=gfcm,ffci,blitz,rcot,gcm_boosted,par_cop

# core benchmark
run python experiments/run_cells.py                            # §5.2-5.5 tables (each cell uses its
                                                               #   own manifest panel — do NOT pass
                                                               #   --panel here; that forces par_cop
                                                               #   onto the n=1e5 cells, where it
                                                               #   stalls for hours)
run python experiments/run_convergence.py  --panel $PANEL      # §5.6 convergence (gcm_boosted capped
                                                               #   at n=20000, par_cop at n=10000
                                                               #   inside run_convergence.py)
run python experiments/run_pc.py           --panel $PANEL      # §5.8/5.10 PC discovery
# appendix + real-data + speed
run python experiments/run_ablation.py                         # asym / interaction / tradeoff / rankz
run python experiments/run_injection.py    --panel $PANEL      # tab:inject (EuStock)
run python experiments/run_cc_injection.py --panel gfcm,rcot,gcm_boosted,fisherz   # tab:cc-inject
run python experiments/run_speed.py        --panel $PANEL      # fig:speed grid
# aggregate + figures
run python experiments/aggregate.py                            # -> out/results/aggregated/
run python figures/make_figures.py                             # -> out/results/figures/
```

Results end up under `out/data/` (raw) and `out/results/` (the paper's tables + figure PDFs).

**Scale and resumability.** A full run is 500 reps over many cells up to n=1e5 — **days** of
compute, not minutes. Every `(cell, test)` is written the moment it finishes and skipped on
restart (atomic `.claim` files), so you can stop/resume freely and launch **many workers in
parallel** on the same output dir — they coordinate via the claim files. Start small
(`--reps 20 --tier 0`) to sanity-check, then scale up.

**One-command driver.** `bash run_all.sh` runs the smoke check then the full pipeline above,
serially and deterministically on any machine. On a many-core box, set `JOBS` to fan the
`run_cells` phase across that many worker containers (each is thread-capped to one core, so pick
`JOBS` ≈ physical cores); they coordinate via the `.claim` files and the output is bit-identical
regardless of `JOBS`:

```bash
bash run_all.sh              # canonical serial reproduction
JOBS=48 bash run_all.sh      # same result, run_cells fanned across 48 workers
```

**No image is published.** Anyone you grant repo access to (a reviewer, a collaborator, you on a
server) builds it locally with the command above. Publishing a pre-built image to a registry is
optional and only worth doing *at release* (a public image is free; a private 5 GB image is not).

## Provenance / pinning

- Base image: `python:3.11.15-bookworm@sha256:091f0798…` (pinned by digest)
- `dagsampler==0.4.0`, `bnmetrics==0.2.2`, `causalchamber==0.2.8` — PyPI
- `citests`, `cbcd` — suite monorepo `constraint-based-causal-discovery-suite` @ `9797d5a` (`#subdirectory=citests` / `=cbcd`). citests is pinned from source, **not** PyPI: the benchmark's GCM uses a `reg=` nuisance selector (`linear`/`xgb`/…) that the same-numbered PyPI `citests==0.1.0` does not have.
- `py-tetrad` @ `acd876e8` — its `tetrad-current.jar` has md5 `03ae7e21f1fed311dc93e9782dd684ad` (build-time verified)
- R packages: CRAN bookworm snapshot + `ericstrobl/RCIT`, `ericstrobl/BLITZ`

(Full recipe and inline pinning notes are in `docker/Dockerfile`.)

## License

MIT (see `LICENSE`). Private for now; will be made public with the paper.
