# data_reviewers — experiments run for the TMLR reviewers (container 2)

Not part of the paper's original experiment set; produced to answer specific requested changes.

  rebuttal/   Reviewer njpw items 3, 4, 5: quantile-grid sensitivity at the PC level (grid_pc),
              per-test null size by grid (grid_null), 20-node discovery (p20).
              Runner: experiments/run_rebuttal.py.  Summary: papers/.../rebuttals/results.md

  eoi5/       Reviewer Eoi5 weaknesses 1, 3, 4, 5: even-scale cells, calibration and power at
              |S| = 1..20, combination rules (ACAT / Bonferroni / joint sign-flip), co-movement
              blocks, PC with CPDAG scoring at p = 20 and 50, and the dimension-budget
              confirmation (max_cols_frac 1/4, 1/8, 1/16).
              Runners: experiments/run_eoi5.py (modes ci, pc, budget).
              Aggregators: experiments/aggregate_eoi5.py, experiments/aggregate_budget.py.

Both trees are v1 (GFCM as submitted). The v2 versions land here after the rerun, with these
frozen as data_reviewers_v1/. `.claim` lock files are not copied.
