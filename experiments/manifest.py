"""The benchmark cell manifest: one entry per (experiment cell), each a self-describing unit
the runners consume. Ported from the kit's run_benchmark.py, decoupled from the resume/test
infrastructure so it depends only on `dgp`.

Each CI cell is a dict with: key, section, n, dS, cell, build (seed -> dagsampler config),
query (X, Y, S), panel (comma-separated test names), n_iter, kind="ci". PC cells carry
build_typed / reps / max_cond instead.
"""
import dgp

# --- panels (which tests run in each cell; the GFCM-only runner filters with --only gfcm) ---
SCALE_PANEL  = "gfcm,blitz,ffci,rcot,gcm_boosted"                 # n=1e5: par_cop/KCI infeasible
MOD_PANEL    = "gfcm,par_cop,blitz,ffci,rcot,gcm_boosted"            # moderate n: prior-art + lean-5 + FFCI
MIXED_PANEL  = "gfcm,blitz,ffci,ci_mm,cmiknn_mixed,gcm_boosted,rcot"

N_SCALE = dgp.N_SCALE       # 100_000
N_MOD   = dgp.N_MODERATE    # 2_000
REPS = {N_MOD: 500, N_SCALE: 500}
DS_GRID = (1, 3, 5)


def _ci(tier, section, builder, cells, n, panel, dS, query_S, reps=None):
    S = [f"Z{i}" for i in range(1, dS + 1)] if query_S is None else query_S
    out = []
    for cell in cells:
        key = f"{section}__{cell}__n{n}__dS{dS}"
        build = (lambda b, nn, ds, cl: (lambda seed: b(nn, seed, ds)[cl]))(builder, n, dS, cell)
        out.append(dict(tier=tier, kind="ci", key=key, section=section, n=n, dS=dS, cell=cell,
                        build=build, query=("X", "Y", S), panel=panel,
                        n_iter=reps or REPS[n]))
    return out


def manifest():
    cells = []
    CAL = ["heavy_tail", "hetero", "nonlin_mean", "mixed_Z"]
    LAD = ["null", "linear_mean", "nonmonotone_z2", "scale", "tail_shape"]
    TAIL = ["null", "alpha_1", "alpha_2", "alpha_3", "alpha_4"]
    MIX = ["CC_null", "CC_alt", "CK_null", "CK_alt", "KC_null", "KC_alt", "KK_null", "KK_alt"]

    # Tier 0 -- headline + prior-art, moderate n
    for dS in DS_GRID:
        cells += _ci(0, "5.2_calib", dgp.calib_nulls, CAL, N_MOD, MOD_PANEL, dS, None)
        cells += _ci(0, "5.3_power", dgp.power_ladder, LAD, N_MOD, MOD_PANEL, dS, None)
    cells += _ci(0, "5.4_tail", dgp.tail_edge_sweep, TAIL, N_MOD, MOD_PANEL, 1, ["Z1"])

    # Tier 1 -- the scale story, n=1e5
    for dS in DS_GRID:
        cells += _ci(1, "5.2_calib", dgp.calib_nulls, CAL, N_SCALE, SCALE_PANEL, dS, None)
        cells += _ci(1, "5.3_power", dgp.power_ladder, LAD, N_SCALE, SCALE_PANEL, dS, None)
    cells += _ci(1, "5.4_tail", dgp.tail_edge_sweep, TAIL, N_SCALE, SCALE_PANEL, 1, ["Z1"])

    # Tier 2 -- mixed-type conditioning
    MIXED_SCALE = "gfcm,blitz,ffci,ci_mm,gcm_boosted,rcot"
    for n, panel in ((N_MOD, MIXED_PANEL), (N_SCALE, MIXED_SCALE)):
        for cell in MIX:
            key = f"5.5_mixed__{cell}__n{n}__dS2"
            build = (lambda nn, cl: (lambda seed: dgp.tab_mixed_conditional(nn, seed)[cl]))(n, cell)
            cells.append(dict(tier=2, kind="ci", key=key, section="5.5_mixed", n=n, dS=2, cell=cell,
                              build=build, query=("X", "Y", ["Zc", "Zk"]), panel=panel,
                              n_iter=REPS[n]))

    # Tier 3 -- GFCM inside PC (handled by run_pc, not run_cells; kept here for completeness)
    for n in (N_MOD, N_SCALE):
        cells.append(dict(tier=3, kind="pc", key=f"5.8_pc__tailgraph__n{n}", section="5.8_pc",
                          n=n, dS=None, cell="tailgraph",
                          build_typed=lambda seed: dgp.random_typed_dag(
                              8, 0.4, seed_structure=seed, seed_data=seed, types=dgp.TYPES_TAIL),
                          panel="gfcm,blitz,ffci,rcot,gcm_boosted" if n == N_SCALE
                                else "gfcm,par_cop,blitz,ffci,rcot,gcm_boosted",
                          reps=100, max_cond=3))

    # Tier 4 -- appendix: block ablation + graceful degradation in |S|
    cells += _ci(4, "app_ablation", lambda nn, sd, ds: dgp.tab_ablation(nn, sd),
                 ["linear_mean", "nonmonotone_z2", "scale", "tail_shape"], N_MOD,
                 "gfcm,par_cop,blitz,ffci,rcot,gcm_boosted", 0, [])
    for dS in (1, 2, 3, 5, 8):
        cells += _ci(4, "app_growingS",
                     lambda nn, sd, ds: dgp.tail_edge_sweep(nn, sd, dS=ds, strengths=(4.0,)),
                     ["alpha_4"], N_MOD, "gfcm,par_cop,blitz,ffci,rcot", dS, None)
    return cells


def oracle_label(res, x, y, S):
    """Null/alt label for X _||_ Y | S from dagsampler's d-separation oracle."""
    want = set(S)
    for e in res.get("ci_oracle", []):
        if {e["x"], e["y"]} == {x, y} and set(e["conditioning_set"]) == want:
            return "null" if e["is_independent"] else "alt"
    return "unknown"
