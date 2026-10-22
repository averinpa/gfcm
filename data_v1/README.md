# data_v1 — snapshot of the SUBMITTED (v1) main-experiment parquets

Frozen 2026-09-21, before the v2 rerun. v1 = GFCM as submitted to TMLR on 2026-08-15, i.e.
`GFCMConfig(max_cols_frac=0.0)`: the nuisance knot rule ignores |S|, no dimension budget.
Reproduce exactly with `GFCMConfig(max_cols_frac=0.0)` (verified byte-identical, 16/16 p-values,
continuous and categorical paths).

Kept because the revision reports v1 and v2 side by side for Reviewer Eoi5's weakness 5
(calibration beyond |S| = 5). `data/` holds the version the revised paper reports.
