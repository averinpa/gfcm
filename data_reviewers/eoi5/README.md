
## Added 2026-09-23 (Reviewer Eoi5, requested change 4)
- `eoi5_pc__*__capnone__rep*.parquet`: PC with no conditioning cap, p=20 (deg 2, 4; 100 reps) and p=50 (deg 2;
  50 reps; gfcm, gfcm_comove, blitz only). Server `out_v2/data/rebuttal_v2`, script `fan_uncapped2.sh`.
- `eoi5_oracle__*.parquet`: PC with the d-separation oracle, cap 3 and uncapped (800 files); the floor for the cap's cost.
- `p50comp/` (pending): FFCI and RCoT at p=50 uncapped, 50 reps, from `fan_p50comp2.sh`; separate folder because the
  pc filename does not encode the test subset. `tables/make_tables_eoi5.py::pcext` globs recursively.
- `envelope_all.parquet` (464,400 rows, 7.5 MB): the operating-envelope run folded on the server from
  `out_v2/data/envelope/` (per-rep files stay there). Columns family, cell, dS, n, rep, arm, pval, wall_s.
  Published method only (`max_cols_frac=0`). Reps: 2000 (n<=5e3), 1000 (1e4, 2e4), 300 (5e4, 1e5). Fifteen
  OOM-orphaned reps at n=1e5 were rerun by `fan_env_sweep.sh` with a 22 GB cap. `tables/make_tables_eoi5.py::envelope`.
- `p50comp/`: FFCI and RCoT at p=50 uncapped (done 2026-09-25); tab:pcext no longer has dagger rows.
