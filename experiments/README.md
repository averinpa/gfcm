# Experiments

One row per table and figure of the resubmission.

## Tables

| artifact | description | configuration | experiment script | artifact script |
|---|---|---|---|---|
| T2 | calibration | DGP: calib_nulls.<br>cells: heavy_tail, hetero, nonlin_mean, mixed_Z.<br>sample_size: 1000, 2000, 5000, 10000, 20000, 50000, 100000.<br>S: 1, 2, 3, 5, 8, 10, 15, 20.<br>reps: 500.<br>tests: GFCM, BLITZ, RCoT, FFCI, GCM-boost, PartCopula.<br>seed: 50000.<br>output: data/grid/ | `run_grid.py --table null` | `aggregate_grid.py` |
| T3 | power | DGP: power_ladder_additive.<br>cells: linear_mean, nonmonotone_z2, scale, tail_shape, null.<br>sample_size: 1000, 2000, 5000, 10000, 20000, 50000, 100000.<br>S: 1, 2, 3, 5, 8, 10, 15, 20.<br>reps: 500.<br>tests: GFCM, BLITZ, RCoT, FFCI, GCM-boost, PartCopula.<br>seed: 50000.<br>output: data/grid/ | `run_grid.py --table power` | `aggregate_grid.py` |
| T4a | mixed-type | DGP: tab_mixed_conditional.<br>cells: CC, CK, KC, KK.<br>sample_size: 2000, 100000.<br>S: 2.<br>reps: 500.<br>tests: GFCM, BLITZ, FFCI, ci_mm, cmiknn_mixed, GCM-boost, RCoT; cmiknn_mixed dropped at 100000.<br>seed: 7.<br>output: data/mixed/ | `CELLS_OUT=data/mixed run_cells.py --section 5.5_mixed` | `tables/make_tables.py mixed` |
| T4b | label invariance | DGP: cat_level_means.<br>K: 3, 5, 8, 12.<br>sample_size: 2000.<br>S: 1.<br>reps: 500.<br>tests: GFCM, BLITZ, RCoT, FFCI, GCM-boost.<br>seed: 700000.<br>output: data/mixed/ | `run_cat_level_means.py --exp labelinv` | `aggregate_mixed.py labelinv` |
| T4c | power against cardinality, pooled quadratic form | DGP: cat_level_means.<br>K: 3, 5, 8, 12.<br>sample_size: 2000, 10000.<br>S: 1.<br>reps: 500.<br>tests: GFCM, BLITZ, RCoT, FFCI, GCM-boost.<br>seed: 900000.<br>output: data/mixed/ | `run_cat_level_means.py --exp cardinality` | `aggregate_mixed.py cardinality` |
| T5 | EuStockMarkets | DGP: EuStockMarkets with scale injection.<br>sample_size: 2000.<br>reps: 500.<br>strength: 4.0.<br>tests: GFCM.<br>output: results/aggregated/injection_eustock.parquet | `run_injection.py` | `tables/make_tables.py inject` |
| T6 | Causal Chamber light tunnel | DGP: real light-tunnel data; sources red, green, blue; sinks current, vis_1, ir_1.<br>gamma: 0.7.<br>tests: Fisher-Z, GCM-linear, RCoT, GFCM.<br>output: results/aggregated/ | `run_cc_injection.py` | `tables/make_tables.py cc_inject` |
| T7 | quantile-grid sensitivity inside PC | DGP: calib_nulls, and random_typed_dag at p = 20, edge probability 0.105.<br>cells: heavy_tail.<br>grids: median (0.5), tails (0.1, 0.9), default (0.1, 0.5, 0.9), dense5, wide5.<br>sample_size: 2000, 10000.<br>S: 3.<br>reps: 500 null, 100 PC.<br>tests: GFCM; the PC arm also BLITZ, RCoT, FFCI.<br>seed: 20000.<br>output: data/rebuttal/ | `run_rebuttal.py` | `aggregate_rebuttal.py` |
| T8 | PC discovery at p = 20 and p = 50 | DGP: random_typed_dag, TYPES_TAIL.<br>graphs: p = 20 at expected degree 2 and 4, p = 50 at degree 2.<br>sample_size: 2000.<br>reps: 100 at p = 20, 50 at p = 50.<br>tests: GFCM, BLITZ, RCoT, FFCI, GCM-boost.<br>orientations: sepset and majority on the same draw.<br>conditioning cap: none.<br>seed: 20000.<br>output: data/pc_cons/ | `run_pc_cons.py --p 20` and `--p 50` | `aggregate_pc_cons.py` |
| T9 | asymmetric versus symmetric quantile product | DGP: tab_asym, and indep_null for the null row.<br>cells: null, scale, directed_tailshape, symmetric_comovement.<br>sample_size: 2000.<br>S: 0, marginal.<br>reps: 500.<br>arms: asymmetric eX r(Y) and r(X) eY, against symmetric r(X) r(Y).<br>taus: 0.1, 0.9.<br>tests: GFCM.<br>seed: 2000.<br>output: results/aggregated/asym.parquet | `run_components.py asym` | `tables/make_tables.py asym` |
| T10 | rank-transformed conditioning holds calibration on heavy-tailed conditioners | DGP: shared nonlinear mean sin(Z), X and Y independent given Z.<br>cells: gaussian, t5, t2, cauchy, lognormal, power.<br>sample_size: 2000.<br>S: 1.<br>reps: 500.<br>arms: rankz on and off, same draws.<br>tests: GFCM.<br>seed: 2000.<br>output: results/aggregated/rankz.parquet | `run_components.py rankz` | `tables/make_tables.py rankz` |
| T11 | null rejection rate | DGP: tab_interaction.<br>cells: additive (sin Z1 + sin Z2 against cos Z1 + cos Z2), interaction (Z1 Z2).<br>sample_size: 1000, 4000, 16000.<br>S: 2.<br>reps: 500.<br>tests: GFCM.<br>output: results/aggregated/interaction.parquet | `run_components.py interaction` (**needs extending**: it runs one sample size and writes `null` where the table reads `kind` and `n`) | `tables/make_tables.py interaction` |
| T12 | complementary failure of the two nuisance blocks | DGP: shared mean in three gaussian conditioners.<br>cells: additive, interaction (Z1 Z2), transcendental (sin of the sum).<br>sample_size: 2000, 10000.<br>S: 3.<br>reps: 500.<br>arms: polynomial block only, single-index block only and ungated.<br>tests: GFCM.<br>seed: 1000.<br>output: results/aggregated/tradeoff.parquet | `run_components.py tradeoff` | `tables/make_tables.py nuisance_tradeoff` |
| T13 | level when assumption (A5) fails | DGP: misspec family.<br>cells: see CI_PLAN.<br>sample_size: 2000, 10000.<br>reps: 500.<br>tests: GFCM, RCoT, FFCI, BLITZ.<br>seed: 30000.<br>output: data/rebuttal/eoi5/ | `run_eoi5.py --mode ci --family misspec` | `tables/make_tables_eoi5.py misspec` |
| T14 | combination of the three block p-values, Bonferroni and Cauchy only | DGP: the CI_PLAN families.<br>sample_size: 2000, 10000.<br>reps: 500.<br>arms: per-block p-values via detail=True; the joint sign-flip arm no longer exists in the code.<br>seed: 30000.<br>output: data/rebuttal/eoi5/ | `run_eoi5.py --mode ci` | `tables/make_tables_eoi5.py combrule` |
| T15 | convergence of null size with sample size | DGP: calib_nulls, power_ladder, tail_edge_sweep.<br>cells: heavy_tail, hetero, mixed_Z, nonlin_mean; tail_shape, null; alpha_3, null.<br>sample_size: 500 to 100000, 14 values.<br>S: 1, 3, 5.<br>reps: 500.<br>tests: GFCM; GCM-boost capped at 20000, PartCopula at 10000.<br>output: data/cells_conv/ | `run_convergence.py` | `tables/make_tables.py convergence` |

## Figures

| artifact | description | configuration | experiment script | artifact script |
|---|---|---|---|---|
| F1 | detection class of the quantile subset | sample_size: 2000.<br>reps: 500.<br>data drawn inside the figure script.<br>output: results/figures/fig_detection_class.pdf | — | `figures/make_figures.py detection` |
| F2 | the conditional-tail edge | DGP: tail_edge_sweep, X drives the conditional skew of Y with mean and variance held fixed.<br>cells: alpha_1, alpha_2, alpha_3, alpha_4, and the matched null.<br>sample_size: 2000.<br>S: 1.<br>reps: 500.<br>tests: GFCM, BLITZ, FFCI, RCoT, GCM-boost, PartCopula.<br>metric: size-corrected power, each test thresholded at the 5 percent quantile of its own matched null; Wilson 95 percent intervals.<br>output: data/cells/ | `run_cells.py --section 5.4_tail` | `figures/make_figures.py tail_strength` |
| F3 | convergence with sample size | the T15 run.<br>output: data/cells_conv/, figures to results/figures/ | `run_convergence.py` | `figures/make_figures.py convergence` |
| F5 | speed, the full panel of eleven tests, one GFCM line | sample_size: 2000, 5000, 10000, 100000.<br>S: 2, 5, 20; 50 is run but dropped from the figure, the design is reduced there.<br>draws: slow tests a single draw under a per-cell time budget.<br>threads: GFCM capped at 16, the others at 64.<br>output: results/aggregated/speed_grid.parquet | `run_speed.py` | `figures/fig_speed.py` |
| F6 | PC under the majority rule, skeleton and CPDAG panels, no cap | the T8 run at all three graph sizes: p = 8 at edge probability 0.4, p = 20 at degree 2 and 4, p = 50 at degree 2.<br>output: data/pc_cons/ | `run_pc_cons.py --p 8`, `--p 20`, `--p 50` | `aggregate_pc_cons.py` then `figures/fig_pc_cons.py` |
| F7 | size against conditioning depth, with and without the variance correction | DGP: calib_nulls.<br>cells: heavy_tail, hetero, nonlin_mean, mixed_Z, pooled.<br>sample_size: 2000.<br>S: 1, 2, 3, 5, 8, 10, 15, 20.<br>reps: 500.<br>tests: GFCM, gfcm_nocorr.<br>seed: 50000.<br>output: data/grid/ | `run_grid.py --table null` | `aggregate_grid.py` then `figures/fig_size_depth.py` |

## Not in the paper

| | description | configuration | experiment script |
|---|---|---|---|
| discrete | GFCM against the contingency-table tests on natively discrete data | DGP: discrete_cells.<br>cells: null, alt.<br>sample_size: 2000, 10000.<br>S: 1, 2, 3, 4.<br>levels: K = 3, KZ = 3.<br>strength: 0.15.<br>reps: 500.<br>tests: GFCM, G-squared, chi-squared, BLITZ, RCoT, FFCI.<br>seed: 980000.<br>output: data/discrete/ | `run_discrete.py` |
| co-movement | the fourth block, on and off; out of this paper | DGP: calib_nulls, power_ladder_additive, comove_cells.<br>cells: the four nulls; the four alternatives; covol and its matched null.<br>arms: default, scale, full.<br>sample_size: 2000, 10000.<br>S: 1, 3, 5, 10.<br>reps: 500.<br>seed: 900000.<br>output: data/comove/ | `run_comove.py` |

## Shared modules

| file | what it is |
|---|---|
| `adapters.py` | test name to data-bound CI test; holds the GFCM arms and routes `gfcm_published` to the submitted code |
| `gfcm_submitted.py` | verbatim copy of the test at tag `tmlr-rev1`. Never edit it: a change here silently invalidates every paired table |
| `gfcm_citest.py` | GFCM in cbcd's CITest shape, type mask resolved once per dataset |
| `dgp.py` | every data-generating configuration, consumed by `dagsampler` |
| `discovery.py` | PC wrapper, the d-separation oracle, the true CPDAG |
| `metrics.py` | skeleton and CPDAG scores |
