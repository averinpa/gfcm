#!/usr/bin/env bash
# Reproduction driver for the GFCM paper: smoke -> full reproduction (all tables + figures).
#
# Resumable: every (cell, test) result is written the moment it finishes and skipped on
# restart via atomic `.claim` files, so you can stop/resume freely.
#
# Usage:
#   git clone https://github.com/averinpa/gfcm && cd gfcm
#   docker build -f docker/Dockerfile -t gfcm-repro .     # build once (fully pinned; turnkey)
#   bash run_all.sh                                       # canonical serial reproduction
#   JOBS=48 bash run_all.sh                               # faster: N parallel workers for run_cells
#
# JOBS controls how many worker containers run the `run_cells` phase concurrently. Workers
# coordinate through the `.claim` files (no duplicated work), and because every rep is seeded
# the output is BIT-IDENTICAL regardless of JOBS. Pick ~= physical cores (each worker is
# thread-capped to 1 core). Only the run_cells phase is parallelized (it is claim-safe); every
# other phase runs serially. Default JOBS=1 keeps the run deterministic on any machine.
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-gfcm-repro}"
JOBS="${JOBS:-1}"
LOG="$PWD/run_all.log"
exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[$(date)] driver start (pid $$)  IMAGE=$IMAGE  JOBS=$JOBS"

# ---- 0. require the image (build it first; see usage above) ----
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[$(date)] ERROR: image '$IMAGE' not found. Build it first:"
  echo "    docker build -f docker/Dockerfile -t $IMAGE ."
  exit 1
fi
echo "[$(date)] image: $(docker image inspect "$IMAGE" --format '{{.Id}} {{.Size}} bytes')"

# ---- 1. smoke test in an ISOLATED dir (its 20-rep cells never pollute the full run) ----
mkdir -p out_smoke/data out_smoke/results
smoke() { docker run --rm -v "$PWD/out_smoke/data":/work/data -v "$PWD/out_smoke/results":/work/results "$IMAGE" "$@"; }
echo "[$(date)] SMOKE: gfcm-only, reps=20, tier=0, section 5.3_power"
if smoke python experiments/run_cells.py --only gfcm --reps 20 --tier 0 --section 5.3_power; then
  echo "[$(date)] smoke run_cells OK"
else
  echo "[$(date)] SMOKE FAILED — aborting before the full run"; exit 1
fi
smoke pytest -q tests/test_smoke.py && echo "[$(date)] smoke pytest OK" || echo "[$(date)] WARN: pytest smoke non-zero (continuing)"

# ---- 2. FULL reproduction (resumable) ----
mkdir -p out/data out/results
run() { docker run --rm -v "$PWD/out/data":/work/data -v "$PWD/out/results":/work/results "$IMAGE" "$@"; }
PANEL=gfcm,ffci,blitz,rcot,gcm_boosted,par_cop

# run_cells: NO --panel here on purpose. Each cell carries its own panel in the manifest
# (par_cop is intentionally excluded at n=1e5, where it stalls for hours). Passing --panel
# would override every cell and force par_cop onto the large-n cells.
echo "[$(date)] === FULL: run_cells (5.2-5.5 tables)  JOBS=$JOBS ==="
if [ "$JOBS" -gt 1 ]; then
  for i in $(seq 1 "$JOBS"); do
    run python experiments/run_cells.py >>"out/run_cells_worker_$i.log" 2>&1 &
  done
  wait
else
  run python experiments/run_cells.py
fi

echo "[$(date)] === FULL: convergence (5.6) ==="
# gcm_boosted and par_cop are n-capped inside run_convergence.py (20000 / 10000) — they appear
# in the convergence table only up to their cap; the rest of the panel runs the full grid.
run python experiments/run_convergence.py --panel gfcm,ffci,blitz,rcot,gcm_boosted,par_cop
echo "[$(date)] === FULL: pc discovery (5.8/5.10) ==="
run python experiments/run_pc.py           --panel "$PANEL"
echo "[$(date)] === FULL: ablation (appendix) ==="
run python experiments/run_ablation.py
echo "[$(date)] === FULL: injection (tab:inject, EuStock) ==="
run python experiments/run_injection.py    --panel "$PANEL"
echo "[$(date)] === FULL: cc_injection (tab:cc-inject) ==="
run python experiments/run_cc_injection.py --panel gfcm,rcot,gcm_boosted,fisherz
echo "[$(date)] === FULL: speed (fig:speed) ==="
run python experiments/run_speed.py --panel gfcm,ffci,blitz,rcot,gcm_boosted

echo "[$(date)] === aggregate -> results tables ==="
run python experiments/aggregate.py
echo "[$(date)] === figures ==="
run python figures/make_figures.py

echo "[$(date)] ALL DONE  (JOBS=$JOBS)"
echo "  raw:     $PWD/out/data"
echo "  tables:  $PWD/out/results/aggregated"
echo "  figures: $PWD/out/results/figures"
