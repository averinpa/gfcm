"""GFCM: a tail-sensitive, mixed-type conditional-independence test.

GCM template on a bank of conditionally-mean-zero residual features of (X, Y) given Z:
mean residuals e (cross-fit Ridge on a spline basis), centered scale c = s(e) - Ehat[s(e)|Z]
(s = |e| by default), and location-scale quantile-indicator residuals r_tau. The features are
pooled in blocks and the block p-values are combined by the Cauchy (ACAT) rule.

Combination "p3" (3 blocks, Cauchy across, Wald within): [cov + both scale transforms pooled]
| [eX x rY_tau levels] | [rX_tau x eY]. Pooling coherent evidence inside a block removes ACAT
dilution; keeping the two quantile orientations in separate blocks preserves precision.

Fast: only Ridge fits (no quantile LP, no kernels, no per-fit CV). Valid: the spline basis is
nonparametric (orthogonality under misspecification holds); the GCM product bias is second-order
in the nuisance errors. Symmetric: both orientations are included. Mixed-type: the same template
takes one-hot residual columns in the multivariate blocks.

Configuration is explicit via `GFCMConfig`. The defaults are the canonical published
configuration; construct a modified `GFCMConfig(...)` for ablations.
"""
import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2, rankdata
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

__all__ = ["GFCMConfig", "unified_test", "k_rule", "infer_types"]

# --- internal solver constants (not user knobs) ---
_ALPHAS = np.logspace(-2, 3, 10)   # RidgeCV grid (only used if GFCMConfig disables the fast path)
_RIDGE_ALPHA = 1.0                 # default spline ridge penalty
_USE_CV = False                    # RidgeCV fallback; off (GCV selects the penalty instead)
_GCV_GRID = np.logspace(-2, 4, 13)
_CAT_MAX_LEVELS = 15               # auto-detection: integer column with <= this many levels -> categorical

_LOG = logging.getLogger("gfcm")
_LOGGED_CONFIGS = set()


@dataclass(frozen=True)
class GFCMConfig:
    """GFCM configuration. Defaults are the canonical published configuration.

    Ablated in the paper: scale, calib, gcv, proj, rankz, inter, nuisance, taus.
    Structural (fixed method parameters): combine, b, folds, ties, inter_maxd,
    rawz_lin, proj_k, proj_nmin.
    """
    # --- ablated ---
    scale: str = "abs"                 # scale feature: "abs" robust |e| | "e2" squared
    calib: str = "chi2"                # block calibration: "chi2" analytic | "signflip" resampled
    gcv: int = 2                       # GCV ridge selection: 0 off | 1 all fits | 2 scale-fit only
    proj: bool = True                  # single-index (SIR) nuisance block, n-gated
    rankz: bool = True                 # rank-transform Z before the spline
    inter: bool = True                 # degree-3 polynomial interaction block
    nuisance: str = "spline"           # nuisance backend: "spline" | "poly"
    taus: tuple = (0.1, 0.5, 0.9)      # quantile levels
    # --- structural ---
    combine: str = "p3"                # block combination: "p3" (default) | "p2" | "cauchy" (5-way)
    b: int = 999                       # sign-flip resamples (only used when calib="signflip")
    folds: int = 5                     # cross-fit folds
    ties: str = "mid"                  # quantile indicator: "mid" tie-aware | "le" 1{x<=q}
    inter_maxd: int = 20               # interaction block active for |S| <= inter_maxd (O(d^2) cost)
    rawz_lin: bool = True              # add a raw standardized linear-Z block (heavy-tail fix)
    proj_k: int = 8                    # spline knots on each estimated index direction
    proj_nmin: int = 5000              # n-gate: single-index block active at n >= proj_nmin, |S| >= 2


def _log_config(cfg):
    """Log the active configuration once per distinct config, so a run's settings are visible."""
    key = repr(cfg)
    if key not in _LOGGED_CONFIGS:
        _LOGGED_CONFIGS.add(key)
        _LOG.info(
            "active config: scale=%s calib=%s gcv=%s proj=%s rankz=%s inter=%s "
            "nuisance=%s taus=%s combine=%s",
            cfg.scale, cfg.calib, cfg.gcv, cfg.proj, cfg.rankz, cfg.inter,
            cfg.nuisance, cfg.taus, cfg.combine,
        )


def _qind(x, q, tau, cfg):
    """Quantile-indicator residual tau - 1{x <= q}; "mid" uses the tie-aware
    mid-distribution indicator 1{x<q} + 0.5*1{x=q} (Parzen), identical for continuous
    data but unbiased and non-degenerate for discrete/binary variables."""
    if cfg.ties == "mid":
        return tau - ((x < q) + 0.5 * (x == q))
    return tau - (x <= q).astype(float)


def _reg(alpha=None):
    """Nuisance regressor: spline+Ridge (the fast linear-on-basis backend)."""
    if _USE_CV:
        return RidgeCV(alphas=_ALPHAS)
    return Ridge(alpha=_RIDGE_ALPHA if alpha is None else alpha)


def _interaction_block(Z, cfg):
    """Degree-3 polynomial cross-monomials {Zi*Zj, Zi^2*Zj, Zi*Zj^2} on RAW standardized
    (signed) Z. The additive spline cannot represent an interaction in E[.|Z], so without these
    the residuals retain it, the GCM cross-covariance bias does not vanish, and Type-I inflates
    (-> 1 as n grows). The rank transform destroys the interaction sign, so these are on raw signed
    Z. GCV shrinks the columns to ~0 when no interaction is present, preserving the additive nulls
    and the runtime. Cost is O(d^2) columns (3*C(d,2)), capped at |S| <= cfg.inter_maxd so it does
    not dominate the solve at large conditioning sets. LIMITATIONS: polynomial interactions only;
    transcendental g(Zi)*Zj interactions are not represented (the single-index block targets the
    single-index subset of that regime)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1 or Z.shape[1] < 2 or Z.shape[1] > cfg.inter_maxd:
        return None
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    d = Z.shape[1]
    cols = []
    for i in range(d):
        for j in range(i + 1, d):
            a, b = Zs[:, i], Zs[:, j]
            cols.extend([a * b, a * a * b, a * b * b])
    return np.column_stack(cols)


def _proj_designs(Zb, Zs, dir_targets, folds, K, dir_ridge=1e-3):
    """CROSS-FIT single-index augmentation. For each fold, estimate the index direction(s) a and fit
    the 1-D index spline on the TRAIN rows only, then apply both to the TEST rows (no leakage). Returns
    per fold (tr, te, Xc, mu, Xte_c) for the augmented design [base | spline(a.Z) per direction], with
    Xc the train-centered design, mu the train column means, Xte_c the test design minus mu. The
    direction is the ridge-OLS coefficient (Stein/average-derivative: proportional to a for a single
    index, any smooth g). Zs is the standardized raw Z used to form the index."""
    out = []
    for tr, te in folds:
        Ztr = Zs[tr]; Zte = Zs[te]
        G = Ztr.T @ Ztr
        G[np.diag_indices_from(G)] += dir_ridge * np.trace(G) / Ztr.shape[1] + 1e-9
        cfd = cho_factor(G, lower=True)
        ctr = [Zb[tr]]; cte = [Zb[te]]
        dirs = []
        for t in dir_targets:
            tc = t[tr] - t[tr].mean()
            a = cho_solve(cfd, Ztr.T @ tc); na = np.linalg.norm(a)   # MEAN direction (SIR/Stein)
            if na >= 1e-8:
                dirs.append(a / na)
        for a in dirs:
            st = SplineTransformer(n_knots=K, degree=3).fit((Ztr @ a)[:, None])
            ctr.append(st.transform((Ztr @ a)[:, None]))
            cte.append(st.transform((Zte @ a)[:, None]))
        Xtr = np.column_stack(ctr); mu = Xtr.mean(0)
        out.append((tr, te, Xtr - mu, mu, np.column_stack(cte) - mu))
    return out


def _resid_proj(designs, T, alpha):
    """Cross-fit residuals through the per-fold augmented projection designs (see _proj_designs)."""
    a = float(_RIDGE_ALPHA if alpha is None else alpha)
    T = np.asarray(T, float); O = np.empty_like(T)
    for tr, te, Xc, mu, Xtec in designs:
        G = Xc.T @ Xc
        G[np.diag_indices_from(G)] += a
        cf = cho_factor(G, lower=True)
        tm = T[tr].mean(0)
        beta = cho_solve(cf, Xc.T @ (T[tr] - tm))
        O[te] = T[te] - (Xtec @ beta + tm)
    return O


def _design(Z, K, cfg, z_cat=None):
    """Build the conditioning design matrix for the nuisance regressions. Spline basis on
    rank-Z by default, augmented with (i) a RAW standardized linear-Z block (cfg.rawz_lin, on by
    default) and (ii) a degree-3 polynomial interaction block (cfg.inter, on by default). The raw-Z
    block fixes heavy-tail nulls: the rank transform destroys scale, so the rank-Z spline cannot
    represent a heavy-tailed Z's LINEAR effect (-> correlated residuals at the leverage points ->
    Type-I inflation); the raw linear column captures it exactly, and GCV shrinks it to ~0 when not
    needed. The interaction block does the same for interaction confounders. cfg.nuisance="poly"
    uses a degree-3 polynomial feature map instead of the spline (it already includes linear +
    interaction terms). When z_cat flags categorical Z-columns, those enter as one-hot dummies
    (J-1 levels, integer codes NOT splined) and the continuous columns go through the usual
    spline/linear/interaction machinery; with no categorical column flagged the result is
    byte-identical to the continuous design."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if z_cat is not None and np.ndim(z_cat) and np.any(z_cat):
        z_cat = np.asarray(z_cat, bool)
        blocks = []
        Zc = Z[:, ~z_cat]
        if Zc.shape[1] > 0:
            blocks.append(_design(Zc, K, cfg, None))          # continuous machinery, recursed
        for j in np.where(z_cat)[0]:
            col = Z[:, j]; levels = np.unique(col)
            if len(levels) > 1:                               # one-hot, drop first level (J-1 dummies)
                blocks.append(np.column_stack([(col == c).astype(float) for c in levels[1:]]))
        return np.column_stack(blocks) if len(blocks) > 1 else blocks[0]
    Zr = _rank_z(Z) if cfg.rankz else np.asarray(Z, dtype=float)
    if cfg.nuisance == "poly":
        return PolynomialFeatures(degree=3, include_bias=False).fit_transform(Zr)
    blocks = [SplineTransformer(n_knots=K, degree=3).fit_transform(Zr)]
    if cfg.rawz_lin:
        Zraw = np.asarray(Z, dtype=float)
        if Zraw.ndim == 1:
            Zraw = Zraw[:, None]
        blocks.append((Zraw - Zraw.mean(0)) / (Zraw.std(0) + 1e-12))
    if cfg.inter:
        inter = _interaction_block(Z, cfg)
        if inter is not None:
            blocks.append(inter)
    return np.column_stack(blocks) if len(blocks) > 1 else blocks[0]


def _gcv_alpha(Zt, targets, grid=_GCV_GRID):
    """ONE global ridge penalty per test call, chosen by closed-form GCV on the mean
    regressions (all alphas evaluated in O(d) each -- NOT per-fit CV). Data-adaptive:
    small alpha when E[.|Z] is wiggly (validity preserved), large alpha when smooth
    (power at deep |S|). Uses the d x d Gram eigendecomposition."""
    Xc = Zt - Zt.mean(0)
    G = Xc.T @ Xc
    w, V = np.linalg.eigh(G)
    s2 = np.clip(w, 0, None)
    s = np.sqrt(np.maximum(s2, 1e-12))
    n = len(Zt)
    yc = [np.asarray(t, float) - np.mean(t) for t in targets]
    uty = [(V.T @ (Xc.T @ y)) / s for y in yc]
    yss = [float(y @ y) for y in yc]
    best_a, best_g = grid[0], np.inf
    for a in grid:
        d = s2 / (s2 + a)
        df = d.sum() + 1.0
        g = 0.0
        for u, ss in zip(uty, yss):
            rss = max(ss - float(((2 * d - d * d) * (u * u)).sum()), 1e-12)
            g += rss / (n - df) ** 2
        if g < best_g:
            best_g, best_a = g, a
    return float(best_a)


def _cauchy(ps):
    ps = np.clip(np.asarray(ps, float), 1e-15, 1 - 1e-15)
    return float(0.5 - np.arctan(np.mean(np.tan((0.5 - ps) * np.pi))) / np.pi)


def _rank_z(Z):
    Z = np.asarray(Z, float)
    Z = Z[:, None] if Z.ndim == 1 else Z
    n = len(Z)
    return np.column_stack([rankdata(Z[:, j]) / (n + 1) for j in range(Z.shape[1])])


def _signflip_p(R, signs):
    """Two-sided sign-flip p-value for E[R]=0 (R conditionally mean-zero)."""
    T0 = abs(R.sum())
    Tb = np.abs(signs @ R)
    return (1 + int((Tb >= T0).sum())) / (len(Tb) + 1)


def _mvgcm_p(M, signs, cfg, ridge=1e-6, center=False):
    """Multivariate GCM: pool the columns of M (e.g. quantile levels) via a Wald
    quadratic form T'Sinv T, calibrated by analytic chi^2 (cfg.calib="chi2") or sign-flip
    resampling ("signflip"). Pooling beats per-column Cauchy. center=True removes the signal
    mean from the covariance (no self-normalization shrinkage under strong alternatives)."""
    n, k = M.shape
    Mc = M - M.mean(0) if center else M
    G = Mc.T @ Mc
    Sig = G + ridge * np.trace(G) / k * np.eye(k)
    Si = np.linalg.pinv(Sig)
    T = M.sum(0); S0 = float(T @ Si @ T)
    if cfg.calib == "chi2":
        return float(chi2.sf(S0, df=k))
    Tb = signs @ M
    Sb = np.einsum("bi,ij,bj->b", Tb, Si, Tb)
    return (1 + int((Sb >= S0).sum())) / (len(signs) + 1)


def _fold_solvers(Zt, folds, alpha):
    """Per-fold centered-Gram Cholesky, SHARED across the four nuisance targets
    (X, Y, s(e_X), s(e_Y)) so the same (Zt[tr]^T Zt[tr] + alpha I) is factored once per
    fold instead of refit from scratch by each sklearn Ridge.fit. Matches
    Ridge(fit_intercept=True, alpha) -- centering is REQUIRED. alpha=None means _RIDGE_ALPHA."""
    a = float(_RIDGE_ALPHA if alpha is None else alpha)
    out = []
    for tr, te in folds:
        Ztr = Zt[tr]; mu = Ztr.mean(0); Zc = Ztr - mu
        G = Zc.T @ Zc
        G[np.diag_indices_from(G)] += a
        try:
            cf = cho_factor(G, lower=True)
        except np.linalg.LinAlgError:
            cf = None                       # near-singular fold: fall back per fold
        out.append((tr, te, cf, Zc, mu))
    return out


def _resid_with(solvers, Zt, t, alpha):
    """Cross-fit mean residual using pre-factored per-fold solvers (see _fold_solvers)."""
    return _resid_withM(solvers, Zt, np.asarray(t, float)[:, None], alpha)[:, 0]


def _resid_withM(solvers, Zt, T, alpha):
    """Batched cross-fit mean residuals: all target columns of T (n x m) share one cho_solve
    per fold (matrix RHS) instead of re-traversing the folds once per target."""
    T = np.asarray(T, float)
    O = np.empty_like(T)
    for tr, te, cf, Zc, mu in solvers:
        tm = T[tr].mean(0)
        if cf is None:                       # PD fallback for that fold
            for j in range(T.shape[1]):
                O[te, j] = T[te, j] - _reg(alpha).fit(Zt[tr], T[tr, j]).predict(Zt[te])
        else:
            beta = cho_solve(cf, Zc.T @ (T[tr] - tm))
            O[te] = T[te] - ((Zt[te] - mu) @ beta + tm)
    return O


def _ls_quant_resid(e, s2hat, folds, taus, cfg):
    """Location-scale quantile residuals, REUSING the mean fit (e) and the conditional
    second-moment fit (s2hat = Ehat[e^2|Z]): u = e/s, q_tau = train-fold quantile of u,
    r_tau = tau - 1{u <= q_tau}. No extra regressions (fast, low-variance at deep |S|);
    exact E[r|Z]=0 under location-scale nulls, and residual bias is second-order in the
    GCM product, as for any nuisance error."""
    s = np.sqrt(np.maximum(s2hat, 1e-3 * np.mean(s2hat) + 1e-12))
    u = e / s
    out = {}
    for t in taus:
        r = np.empty(len(u))
        for tr, te in folds:
            q = np.quantile(u[tr], t)
            r[te] = _qind(u[te], q, t, cfg)
        out[t] = r
    return out


def k_rule(n):
    """Growing-knot rule that makes assumption (A1) bind: K -> infinity slowly with n so
    the spline approximation error in the GCM product vanishes faster than the sqrt(n)
    statistic amplifies it, while the basis stays cheap. Calibrated on a wiggly-mean null:
    40 @1e3, 63 @1e4, 100 @1e5; nominal level across n."""
    return int(min(150, max(25, round(10.0 * n ** 0.2))))


# ---------------- mixed-type (categorical) support ----------------
def _is_cat(col, max_levels=_CAT_MAX_LEVELS):
    """Fallback type detector (used only when the caller passes no flags): integer-coded
    column with few distinct values, matching the project's categorical convention."""
    col = np.asarray(col, float)
    col = col[~np.isnan(col)]
    if col.size == 0:
        return False
    u = np.unique(col)
    return len(u) <= max_levels and bool(np.allclose(u, np.round(u)))


def infer_types(data, declared=None):
    """Resolve a per-column categorical mask for a full n x p dataset, ONCE.

    Use this at the dataset / PC-FCI stage: compute the mask a single time and pass
    is_cat[i]/is_cat[j]/is_cat[S] to each `unified_test` call, so a variable's type is fixed
    dataset-wide rather than re-inferred per CI test on a subsample (which would be wasteful,
    inconsistent across subsamples, and could misclassify a continuous column that happens to
    show few distinct values in a sparse conditioning slice).

    declared: optional overrides. A length-p boolean sequence is taken as the COMPLETE type
    declaration (auto-detection ignored); a dict {column_index: bool} overrides only those
    columns and auto-detects the rest. None auto-detects every column.

    Returns a length-p boolean numpy array.
    """
    data = np.asarray(data, float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (n x p), got shape {data.shape}")
    p = data.shape[1]
    if isinstance(declared, dict):
        mask = np.array([_is_cat(data[:, j]) for j in range(p)], bool)
        for j, v in declared.items():
            mask[int(j)] = bool(v)
        return mask
    if declared is not None:
        mask = np.asarray(declared, bool).reshape(-1)
        if mask.size != p:
            raise ValueError(f"declared must have length p={p}, got {mask.size}")
        return mask
    return np.array([_is_cat(data[:, j]) for j in range(p)], bool)


def _report_types(X, Y, Z, x_cat, y_cat, z_cat, auto):
    """Make type resolution visible: DEBUG-log the resolved types, and warn on the gray zone
    (an auto-detected integer column with just over max_levels distinct values, which is treated
    as continuous but may be a categorical the caller forgot to flag)."""
    if _LOG.isEnabledFor(logging.DEBUG):
        _LOG.debug("types: x_cat=%s y_cat=%s z_cat=%s (auto=%s)",
                   bool(x_cat), bool(y_cat), [bool(v) for v in np.atleast_1d(z_cat)], auto)
    if not auto:                                  # explicit flags: caller owns the types, no warning
        return

    def _grayzone(col, name):
        c = np.asarray(col, float); c = c[~np.isnan(c)]
        if c.size == 0 or not np.all(c == np.round(c)):     # cheap integer pre-check (no sort)
            return
        k = len(np.unique(c))
        if _CAT_MAX_LEVELS < k <= 2 * _CAT_MAX_LEVELS:
            warnings.warn(
                f"{name} is integer-coded with {k} distinct levels and was auto-detected as "
                f"continuous (> max_levels={_CAT_MAX_LEVELS}). If it is categorical, pass an "
                f"explicit type flag (x_cat / y_cat / z_cat).", stacklevel=3)

    if not bool(x_cat):
        _grayzone(X, "X")
    if not bool(y_cat):
        _grayzone(Y, "Y")
    Za = np.asarray(Z, float)
    if Za.ndim == 2:
        for j in range(Za.shape[1]):
            if not bool(np.atleast_1d(z_cat)[j]):
                _grayzone(Za[:, j], f"Z[:,{j}]")


def _cat_resid(A, Zt, folds, solvers):
    """One-hot-minus-conditional-probability residual bank: r_c = 1{A=c} - Ehat[1{A=c}|Z], with
    Ehat a cross-fitted RIDGE linear-probability fit on the SAME spline design (shared per-fold
    Cholesky `solvers` as the continuous nuisances). GCM validity needs only a consistent estimate
    of the conditional mean E[1{A=c}|Z], NOT a logistic link, so the J-1 indicators residualize at
    the cost of J-1 extra continuous targets in one batched solve (no per-fold IRLS). Linear-
    probability fits may stray outside [0,1] -- irrelevant, we use the residual and the chi^2/sign-
    flip calibration handles it. Reference = most-frequent level dropped (the pooled quadratic form
    is invariant to which level is dropped, hence to relabeling)."""
    A = np.asarray(A, float)
    levels = np.unique(A); J = len(levels)
    Ind = np.column_stack([(A == c).astype(float) for c in levels])
    ref = int(np.argmax(Ind.sum(0)))                                       # drop most-frequent level
    M = Ind[:, [j for j in range(J) if j != ref]]                          # n x (J-1) indicators
    if solvers is None or M.shape[1] == 0:                                 # marginal (|Z|=0)
        return M - M.mean(0)
    return _resid_withM(solvers, Zt, M, None)                             # batched cross-fit ridge residuals


def _cont_feats(T, Zt, folds, solvers, taus, cfg):
    """Continuous-variable feature list a categorical partner's vector is tested against: mean
    residual e, centered scale c, and the location-scale quantile residuals r_tau. Reuses the
    shared per-fold Cholesky `solvers` for the mean fit. At |Z|=0 (marginal, solvers is None)
    the residuals are empirical (centering only, no regression), mirroring `_cat_resid`."""
    sq = (lambda e: np.abs(e)) if cfg.scale == "abs" else (lambda e: e ** 2)
    T = np.asarray(T, float)
    if solvers is None:                          # marginal (|Z|=0): empirical centering, no regression
        e = T - T.mean()
        c = sq(e) - sq(e).mean()
    else:
        e = _resid_with(solvers, Zt, T, None)
        if cfg.gcv == 2:
            aq = _gcv_alpha(Zt, (sq(e),)); solvers_q = _fold_solvers(Zt, folds, aq)
        else:
            solvers_q = solvers
        c = _resid_with(solvers_q, Zt, sq(e), None)
    s2 = sq(e) - c
    if cfg.scale == "abs":
        s2 = s2 ** 2
    r = _ls_quant_resid(e, s2, folds, taus, cfg)
    return [e, c] + [r[t] for t in taus]


def _mixed_combine(Xf, Yf, x_cat, y_cat, signs, cfg):
    """Mixed-type combination: pool the categorical side's (J-1) levels into one multivariate-
    GCM quadratic form per feature of the other variable, Cauchy-combine across features."""
    ps = []
    if x_cat and not y_cat:
        for f in Yf:
            ps.append(_mvgcm_p(Xf * f[:, None], signs, cfg))
    elif y_cat and not x_cat:
        for f in Xf:
            ps.append(_mvgcm_p(Yf * f[:, None], signs, cfg))
    else:                                                  # both categorical: pool X, iterate Y levels
        for j in range(Yf.shape[1]):
            ps.append(_mvgcm_p(Xf * Yf[:, j:j + 1], signs, cfg))
    return _cauchy(ps)


def unified_test(X, Y, Z, taus=None, K=None, B=None, seed=0, config=None,
                 x_cat=None, y_cat=None, z_cat=None):
    """GFCM conditional-independence p-value for H0: X independent of Y given Z.

    config: a GFCMConfig; defaults to the canonical published configuration.
    taus/K/B override the config's quantile levels / spline knots / sign-flip resamples.
    x_cat / y_cat / z_cat: declare variable types. x_cat/y_cat are booleans; z_cat is a
    per-column boolean array. Left as None, each is auto-detected via `_is_cat` (integer-coded
    with few distinct levels -> categorical, else continuous). A categorical X or Y switches to
    the one-hot residual bank; categorical Z-columns enter the design as one-hot dummies.
    """
    cfg = config if config is not None else GFCMConfig()
    _log_config(cfg)
    if K is None:
        K = k_rule(len(X))         # default: growing-knot rule (A1 binds); pass int to override
    if B is None:                  # adaptive resolution: fine p-value floor where cheap,
        B = cfg.b if len(X) <= 20000 else min(cfg.b, 199)   # lean at large n (keeps 1e5 fast)
    taus = cfg.taus if taus is None else taus
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    # --- type resolution: explicit flags, else auto-detect integer-coded few-level columns ---
    auto = x_cat is None and y_cat is None and z_cat is None
    if x_cat is None:
        x_cat = _is_cat(X)
    if y_cat is None:
        y_cat = _is_cat(Y)
    if z_cat is None:
        z_cat = np.array([_is_cat(Z[:, j]) for j in range(Z.shape[1])], bool)
    else:
        z_cat = np.asarray(z_cat, bool).reshape(-1)
    _report_types(X, Y, Z, x_cat, y_cat, z_cat, auto)
    if bool(x_cat) or bool(y_cat):            # mixed-type: one-hot residual bank (Section sec:mixed)
        nn = len(X)
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(B, nn))
        if Z.shape[1] == 0:
            Zt = None; solvers = None
            folds = list(KFold(cfg.folds, shuffle=True, random_state=42).split(np.zeros((nn, 1))))
        else:
            Zt = _design(Z, K, cfg, z_cat)
            folds = list(KFold(cfg.folds, shuffle=True, random_state=42).split(Zt))
            solvers = _fold_solvers(Zt, folds, None)          # one factorization shared by X and Y
        Xf = _cat_resid(X, Zt, folds, solvers) if x_cat else _cont_feats(X, Zt, folds, solvers, taus, cfg)
        Yf = _cat_resid(Y, Zt, folds, solvers) if y_cat else _cont_feats(Y, Zt, folds, solvers, taus, cfg)
        return _mixed_combine(Xf, Yf, bool(x_cat), bool(y_cat), signs, cfg)
    sq = (lambda e: np.abs(e)) if cfg.scale == "abs" else (lambda e: e ** 2)
    if Z.shape[1] == 0:                       # marginal (|S|=0): center / empirical
        eX = X - X.mean(); eY = Y - Y.mean()
        cX = sq(eX) - sq(eX).mean(); cY = sq(eY) - sq(eY).mean()
        rxX = {t: _qind(X, np.quantile(X, t), t, cfg) for t in taus}
        rxY = {t: _qind(Y, np.quantile(Y, t), t, cfg) for t in taus}
    else:
        Zt = _design(Z, K, cfg, z_cat)
        folds = list(KFold(cfg.folds, shuffle=True, random_state=42).split(Zt))
        # GCV penalty selection only applies to the ridge-on-spline default backend
        al = _gcv_alpha(Zt, (X, Y)) if (cfg.gcv == 1 and cfg.nuisance == "spline") else None
        _proj_on = cfg.proj and cfg.nuisance == "spline" and Z.shape[1] >= 2 and len(X) >= cfg.proj_nmin and not bool(np.any(z_cat))
        if _proj_on:
            # CROSS-FIT single-index augmentation: direction a and index-spline fit per TRAIN fold,
            # applied to TEST (no leakage). One augmented design reused for the mean and scale fits.
            Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
            pdes = _proj_designs(Zt, Zs, (X, Y), folds, cfg.proj_k)
            RXY = _resid_proj(pdes, np.column_stack([X, Y]), al)
            eX, eY = RXY[:, 0], RXY[:, 1]
            CXY = _resid_proj(pdes, np.column_stack([sq(eX), sq(eY)]), al)
            cX, cY = CXY[:, 0], CXY[:, 1]
        else:
            solvers = _fold_solvers(Zt, folds, al)
            RXY = _resid_withM(solvers, Zt, np.column_stack([X, Y]), al)  # batched: one solve/fold
            eX, eY = RXY[:, 0], RXY[:, 1]
            if cfg.gcv == 2:
                aq = _gcv_alpha(Zt, (sq(eX), sq(eY)))
                solvers_q = _fold_solvers(Zt, folds, aq)
            else:
                aq, solvers_q = al, solvers      # reuse factorization (default path)
            CXY = _resid_withM(solvers_q, Zt, np.column_stack([sq(eX), sq(eY)]), aq)  # s(e)-Ehat[s(e)|Z]
            cX, cY = CXY[:, 0], CXY[:, 1]
        s2X = sq(eX) - cX                  # cross-fitted Ehat[s(e)|Z] -> scale
        s2Y = sq(eY) - cY
        if cfg.scale == "abs":
            s2X, s2Y = s2X ** 2, s2Y ** 2  # |e|-scale -> variance proxy
        rxX = _ls_quant_resid(eX, s2X, folds, taus, cfg)
        rxY = _ls_quant_resid(eY, s2Y, folds, taus, cfg)
    n = len(X)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(B, n))

    RY = np.column_stack([rxY[t] for t in taus])
    RX = np.column_stack([rxX[t] for t in taus])
    if cfg.combine == "p3":               # DEFAULT: [cov + both scales] | [eX.rY] | [rX.eY]
        # pool the coherent moment evidence in one block; keep the two quantile ORIENTATIONS apart
        ps = [_mvgcm_p(np.column_stack([eX * eY, cX * eY, eX * cY]), signs, cfg),
              _mvgcm_p(eX[:, None] * RY, signs, cfg),
              _mvgcm_p(RX * eY[:, None], signs, cfg)]
        return _cauchy(ps)
    if cfg.combine == "p2":               # [cov + both scales] | [both quantile orientations pooled]
        ps = [_mvgcm_p(np.column_stack([eX * eY, cX * eY, eX * cY]), signs, cfg),
              _mvgcm_p(np.column_stack([eX[:, None] * RY, RX * eY[:, None]]), signs, cfg)]
        return _cauchy(ps)
    if cfg.combine == "cauchy":           # 5-way ACAT: every product a separate block (no pooling)
        ps = [_signflip_p(eX * eY, signs),                  # covariance
              _signflip_p(cX * eY, signs),                  # X-scale x Y-level
              _signflip_p(eX * cY, signs),                  # X-level x Y-scale
              _mvgcm_p(eX[:, None] * RY, signs, cfg),        # quantile block, one orientation
              _mvgcm_p(RX * eY[:, None], signs, cfg)]        # other orientation
        return _cauchy(ps)
    raise ValueError(f"unknown combine rule: {cfg.combine!r} (expected 'p3', 'p2', or 'cauchy')")
