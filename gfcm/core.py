"""GFCM: a tail-sensitive, mixed-type conditional-independence test.

GCM template on a bank of conditionally-mean-zero residual features of (X, Y) given Z: mean
residuals e (cross-fit Ridge on a spline basis), centered scale c = |e| - Ehat[|e| : Z], and
location-scale quantile-indicator residuals r_tau. Features are pooled in blocks and the block
p-values are combined by the Cauchy (ACAT) rule. Defaults reproduce the published configuration;
construct a modified `GFCMConfig(...)` to vary them.
"""
import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2, rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

__all__ = ["GFCMConfig", "unified_test", "k_rule", "infer_types", "max_cond_set_for"]

# internal constants (not user knobs)
_RIDGE_ALPHA = 1.0
_GCV_GRID = np.logspace(-2, 4, 13)
_CAT_MAX_LEVELS = 15               # auto-detection: integer column with <= this many levels -> categorical
_INTER_MAXD = 20                   # interaction block active for |S| <= this (O(d^2) columns)
_PROJ_K = 8                        # spline knots on each estimated index direction
_CAT_INTER_MAXCOLS = 400           # budget on the interaction columns across categorical conditioners
_BLAS_THREADS = 16                 # cap on the BLAS threads of one call (scoped); 0 = leave the pool alone

_LOG = logging.getLogger("gfcm")


def max_cond_set_for(n, folds=5, config=None):
    """Largest conditioning-set size inside the test's domain at sample size n."""
    from .varcorr import max_cond_set_for as _m
    return _m(n, folds, config)


@dataclass(frozen=True)
class GFCMConfig:
    """GFCM configuration. Defaults are the canonical published configuration."""
    # --- method choices ---
    gcv: int = 2                       # GCV ridge selection: 0 off | 1 all fits | 2 scale-fit only
    proj: bool = True                  # single-index (SIR) nuisance block, n-gated
    rankz: bool = True                 # rank-transform Z before the spline
    inter: bool = True                 # degree-3 polynomial interaction block
    taus: tuple = (0.1, 0.5, 0.9)      # quantile levels
    # --- structural ---
    folds: int = 5                     # cross-fit folds
    proj_nmin: int = 5000              # n-gate: single-index block active at n >= proj_nmin, |S| >= 2
    # --- optional; the defaults reproduce the published configuration exactly ---
    comove: bool = False                      # add the co-movement block
    comove_parts: tuple = ("scale", "tail")   # "scale" = cX.cY, "tail" = matched corners rX.rY
    across: str = "cauchy"                    # "cauchy" | "bonferroni"
    # --- finite-sample variance correction (gfcm/varcorr.py); continuous X and Y only ---
    # Lazily evaluated: the exact corrected p whenever it is at or below the band, the guarded
    # uncorrected p (a lower bound) above it. varcorr_decide=None computes the exact p on every call.
    # varcorr=False is the same test with R in place of R + Delta, the arm the correction is measured
    # against; it is NOT the submitted test, which also differs in the categorical design.
    varcorr: bool = True
    varcorr_decide: float = 0.05              # lazy evaluation at this level (None = exact p always)


def _qind(x, q, tau):
    """Quantile-indicator residual, tie-aware (Parzen mid-distribution): tau - 1{x < q} - 0.5 1{x = q}."""
    return tau - ((x < q) + 0.5 * (x == q))


def _interaction_block(Z, cfg):
    """Degree-3 cross-monomials {Zi*Zj, Zi^2*Zj, Zi*Zj^2} on raw standardized Z.

    The additive spline cannot represent an interaction in E[.|Z], so without these the GCM
    cross-covariance bias does not vanish. Raw signed Z, since the rank transform destroys the
    interaction sign. Capped at |S| <= _INTER_MAXD. Polynomial only;
    transcendental g(Zi)*Zj is left to the single-index block.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1 or Z.shape[1] < 2 or Z.shape[1] > _INTER_MAXD:
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
    """Cross-fit single-index augmentation: direction and index spline fitted on train rows only.

    Returns per fold (tr, te, Str, Ste, mu_Z, mu_S); the augmented design stays implicit and its
    centred Gram is assembled by blocks in _resid_proj. The direction is the ridge-OLS coefficient
    (Stein / average derivative). Zs is the standardized raw Z used to form the index.
    """
    out = []
    for tr, te in folds:
        Ztr = Zs[tr]; Zte = Zs[te]
        G = Ztr.T @ Ztr
        G[np.diag_indices_from(G)] += dir_ridge * np.trace(G) / Ztr.shape[1] + 1e-9
        cfd = cho_factor(G, lower=True)
        str_, ste_ = [], []
        dirs = []
        for t in dir_targets:
            tc = t[tr] - t[tr].mean()
            a = cho_solve(cfd, Ztr.T @ tc); na = np.linalg.norm(a)
            if na >= 1e-8:
                dirs.append(a / na)
        for a in dirs:
            st = SplineTransformer(n_knots=K, degree=3).fit((Ztr @ a)[:, None])
            str_.append(st.transform((Ztr @ a)[:, None]))
            ste_.append(st.transform((Zte @ a)[:, None]))
        Str = np.column_stack(str_) if str_ else np.zeros((len(tr), 0)); Ste = np.column_stack(ste_) if ste_ else np.zeros((len(te), 0))
        out.append((tr, te, Str, Ste, Zb[tr].mean(0), Str.mean(0)))
    return out


def _resid_proj(designs, T, alpha, cache=None, Zb=None, grams=None):
    """Cross-fit residuals through the augmented projection designs.

    The centred augmented Gram is assembled by blocks from the shared base fold Gram. cache holds
    the per-fold factors under the penalty, so a second call with the same penalty forms nothing.
    """
    a = float(_RIDGE_ALPHA if alpha is None else alpha)
    T = np.asarray(T, float); O = np.empty_like(T)
    key = ("cf", a); cfs = cache.get(key) if cache is not None else None
    if cfs is None:
        cfs = []
        n = len(Zb)
        for k, (tr, te, Str, Ste, muZ, muS) in enumerate(designs):
            Sc = Str - muS; Sf = np.zeros((n, Sc.shape[1])); Sf[tr] = Sc
            GZ = grams[1][k][0]
            ZS = Zb.T @ Sf; SS = Sc.T @ Sc                        # Zc'Sc = Zb[tr]'Sc, Sc has zero column sums
            G = np.block([[GZ, ZS], [ZS.T, SS]]); G[np.diag_indices_from(G)] += a
            cfs.append(cho_factor(G, lower=True))
        if cache is not None: cache[key] = cfs
    n = len(Zb); P = Zb.shape[1]
    for (tr, te, Str, Ste, muZ, muS), cf in zip(designs, cfs):
        tm = T[tr].mean(0); Tf = np.zeros_like(T); Tf[tr] = T[tr] - tm
        rhs = np.vstack([Zb.T @ Tf, Str.T @ Tf[tr]])
        beta = cho_solve(cf, rhs)
        O[te] = T[te] - ((Zb[te] - muZ) @ beta[:P] + (Ste - muS) @ beta[P:] + tm)
    return O


def _design(Z, K, cfg, z_cat=None):
    """Conditioning design matrix for the nuisance regressions.

    Spline basis on rank-Z, always augmented with a raw standardized linear-Z block (it fixes the
    heavy-tail nulls the rank transform cannot represent) and, under cfg.inter, a degree-3
    interaction block. Columns flagged in z_cat enter as one-hot dummies instead; with none flagged the
    result is byte-identical to the continuous design.
    """
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
            if len(levels) > 1:
                # All J dummies, centred: a relabelling then only permutes columns, which a ridge fit
                # is equivariant to, so the test is exactly label-invariant. Dropping a reference
                # level instead makes the p-value depend on how the levels are numbered.
                Dm = np.column_stack([(col == c).astype(float) for c in levels])
                blocks.append(Dm - Dm.mean(0))
        # Interaction across categorical conditioners, the counterpart of _interaction_block. Without
        # it the design spans only additive functions of Z and the level goes to 1 as n grows
        # (measured 1.000 against .044 saturated, four three-level conditioners, n=5e4). Saturated
        # joint cells when they fit the budget, else pairwise products. EVERY cell and level pair is
        # kept and centred, for the same reason as the main dummies: dropping one makes the ridge fit
        # depend on which one, so the p-value would move when the levels are renumbered.
        jc = np.where(z_cat)[0]
        if len(jc) >= 2:
            levs = [np.unique(Z[:, j]) for j in jc]
            budget = _CAT_INTER_MAXCOLS
            ncell = int(np.prod([len(l) for l in levs]))
            cols = []
            if ncell - 1 <= budget:                            # saturated: one indicator per joint cell
                code = np.zeros(len(Z), dtype=np.int64)
                for j, lv in zip(jc, levs):
                    idx = np.searchsorted(lv, Z[:, j]); code = code * len(lv) + idx
                cols = [(code == c).astype(float) for c in np.unique(code)]
            else:                                              # pairwise products of the dummies
                for a in range(len(jc)):
                    for b in range(a + 1, len(jc)):
                        la, lb = levs[a], levs[b]
                        if len(la) < 2 or len(lb) < 2 or len(cols) + len(la) * len(lb) > budget:
                            continue
                        for ca in la:
                            ia = (Z[:, jc[a]] == ca).astype(float)
                            for cb in lb:
                                cols.append(ia * (Z[:, jc[b]] == cb).astype(float))
            if cols:
                Im = np.column_stack(cols); blocks.append(Im - Im.mean(0))
        return np.column_stack(blocks) if len(blocks) > 1 else blocks[0]
    Zr = _rank_z(Z) if cfg.rankz else np.asarray(Z, dtype=float)
    Zraw = Z if Z.ndim == 2 else Z[:, None]
    blocks = [SplineTransformer(n_knots=K, degree=3).fit_transform(Zr),
              (Zraw - Zraw.mean(0)) / (Zraw.std(0) + 1e-12)]        # raw standardized linear-Z block
    if cfg.inter:
        inter = _interaction_block(Z, cfg)
        if inter is not None:
            blocks.append(inter)
    return np.column_stack(blocks) if len(blocks) > 1 else blocks[0]


def _gcv_alpha(Zt, targets, grid=_GCV_GRID, G_all=None):
    """One global ridge penalty per test call, by closed-form GCV on the mean regressions.

    All alphas evaluated in O(d) each from the d x d Gram eigendecomposition, not per-fit CV.
    """
    Xc = Zt - Zt.mean(0)
    G = Xc.T @ Xc if G_all is None else G_all - len(Zt) * np.outer(Zt.mean(0), Zt.mean(0))   # centred full Gram
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


def _mvgcm_p(M, ridge=1e-6):
    """Multivariate GCM: pool the columns of M in a Wald form T'Sinv T, referred to chi^2 on k coordinates."""
    n, k = M.shape
    G = M.T @ M
    Si = np.linalg.pinv(G + ridge * np.trace(G) / k * np.eye(k))
    T = M.sum(0)
    return float(chi2.sf(float(T @ Si @ T), df=k))


def _comove_block(cX, cY, RX, RY, taus, parts=("scale", "tail")):
    """Products of two higher-order features, cX.cY and the matched tail corners rX.rY."""
    lo, hi = int(np.argmin(taus)), int(np.argmax(taus))
    cols = []
    if "scale" in parts:
        cols.append(cX * cY)
    if "tail" in parts:
        cols.append(RX[:, lo] * RY[:, lo])
        if hi != lo:
            cols.append(RX[:, hi] * RY[:, hi])
    if not cols:
        raise ValueError(f"comove_parts must contain 'scale' and/or 'tail', got {parts!r}")
    return np.column_stack(cols)


def _combine_blocks(Ms, cfg):
    """Block p-values -> one p-value under cfg.across."""
    ps = [_mvgcm_p(M) for M in Ms]
    if cfg.across == "cauchy":
        return _cauchy(ps)
    if cfg.across == "bonferroni":
        return float(min(1.0, len(ps) * min(ps)))
    raise ValueError(f"unknown across rule: {cfg.across!r} (expected 'cauchy' or 'bonferroni')")


def _detail(Ms, cfg):
    """Every combination rule, with and without the co-movement block, from one set of features."""
    ps = [_mvgcm_p(M) for M in Ms]
    out = {f"p_block{i}": float(p) for i, p in enumerate(ps)}
    for tag, m in (("3", 3), ("4", len(Ms))):
        out[f"acat{tag}"] = _cauchy(ps[:m])
        out[f"bonf{tag}"] = float(min(1.0, m * min(ps[:m])))
    return out


def _fold_grams(Zt, folds):
    """Per-fold centred training Grams, formed once and by subtraction from the full Gram.

    Zc'Zc = G_all - G_te - n_tr mu mu'. Shared by every ridge penalty the call uses.
    """
    G_all = Zt.T @ Zt; s_all = Zt.sum(0); grams = []
    for tr, te in folds:
        Zte = Zt[te]; mu = (s_all - Zte.sum(0)) / len(tr); G = G_all - Zte.T @ Zte - len(tr) * np.outer(mu, mu)
        grams.append((G, mu))
    return G_all, grams


def _fold_solvers(Zt, folds, alpha, grams=None):
    """Per-fold centered-Gram Cholesky, shared across the four nuisance targets.

    Matches Ridge(fit_intercept=True, alpha); centering is required. alpha=None means _RIDGE_ALPHA.
    """
    a = float(_RIDGE_ALPHA if alpha is None else alpha)
    out = []
    for k, (tr, te) in enumerate(folds):
        if grams is not None:
            G, mu = grams[1][k]; G = G.copy(); Zc = Zt[tr] - mu
        else:
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
    """Cross-fit mean residual using pre-factored per-fold solvers."""
    return _resid_withM(solvers, Zt, np.asarray(t, float)[:, None], alpha)[:, 0]


def _resid_withM(solvers, Zt, T, alpha):
    """Batched cross-fit mean residuals: all columns of T share one cho_solve per fold."""
    T = np.asarray(T, float)
    O = np.empty_like(T)
    for tr, te, cf, Zc, mu in solvers:
        tm = T[tr].mean(0)
        if cf is None:                       # PD fallback for that fold
            for j in range(T.shape[1]):
                r = Ridge(alpha=_RIDGE_ALPHA if alpha is None else alpha)
                O[te, j] = T[te, j] - r.fit(Zt[tr], T[tr, j]).predict(Zt[te])
        else:
            beta = cho_solve(cf, Zc.T @ (T[tr] - tm))
            O[te] = T[te] - ((Zt[te] - mu) @ beta + tm)
    return O


def _ls_quant_resid(e, s2hat, folds, taus):
    """Location-scale quantile residuals, reusing the mean fit e and the second-moment fit s2hat.

    u = e/s, q_tau the train-fold quantile of u, r_tau = tau - 1{u <= q_tau}. No extra regressions;
    exact under location-scale nulls, and residual bias is second-order in the GCM product.
    """
    s = np.sqrt(np.maximum(s2hat, 1e-3 * np.mean(s2hat) + 1e-12))
    u = e / s
    out = {}
    for t in taus:
        r = np.empty(len(u))
        for tr, te in folds:
            q = np.quantile(u[tr], t)
            r[te] = _qind(u[te], q, t)
        out[t] = r
    return out


def k_rule(n):
    """Growing-knot rule that makes (A1) bind: 40 @1e3, 63 @1e4, 100 @1e5."""
    return int(min(150, max(25, round(10.0 * n ** 0.2))))


# ---------------- mixed-type (categorical) support ----------------
def _is_cat(col, max_levels=_CAT_MAX_LEVELS):
    """Fallback type detector, used only when the caller passes no flags."""
    col = np.asarray(col, float)
    col = col[~np.isnan(col)]
    if col.size == 0:
        return False
    u = np.unique(col)
    return len(u) <= max_levels and bool(np.allclose(u, np.round(u)))


def infer_types(data, declared=None):
    """Resolve a per-column categorical mask for a full n x p dataset, once.

    Use this at the dataset / PC-FCI stage so a variable's type is fixed dataset-wide rather than
    re-inferred per CI test on a subsample. declared: a length-p boolean sequence is taken as the
    complete declaration; a dict {column_index: bool} overrides only those columns; None
    auto-detects every column. Returns a length-p boolean array.
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
    """DEBUG-log the resolved types and warn on an auto-detected integer column just over max_levels."""
    if _LOG.isEnabledFor(logging.DEBUG):
        _LOG.debug("types: x_cat=%s y_cat=%s z_cat=%s (auto=%s)",
                   bool(x_cat), bool(y_cat), [bool(v) for v in np.atleast_1d(z_cat)], auto)
    if not auto:                                  # explicit flags: caller owns the types
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
    """One-hot-minus-conditional-probability residual bank, r_c = 1{A=c} - Ehat[1{A=c}|Z].

    Ehat is a cross-fitted ridge linear-probability fit on the same design (shared per-fold
    Cholesky). GCM validity needs only a consistent conditional mean, not a logistic link. The
    most-frequent level is dropped; the pooled quadratic form is invariant to which one.
    """
    A = np.asarray(A, float)
    levels = np.unique(A); J = len(levels)
    Ind = np.column_stack([(A == c).astype(float) for c in levels])
    ref = int(np.argmax(Ind.sum(0)))
    M = Ind[:, [j for j in range(J) if j != ref]]                          # n x (J-1) indicators
    if solvers is None or M.shape[1] == 0:                                 # marginal (|Z|=0)
        return M - M.mean(0)
    return _resid_withM(solvers, Zt, M, None)


def _cont_feats(T, Zt, folds, solvers, taus, cfg):
    """Features a categorical partner is tested against: mean residual e, scale c, quantiles r_tau."""
    T = np.asarray(T, float)
    if solvers is None:                          # marginal (|Z|=0): empirical centering
        e = T - T.mean()
        c = np.abs(e) - np.abs(e).mean()
    else:
        e = _resid_with(solvers, Zt, T, None)
        if cfg.gcv == 2:
            aq = _gcv_alpha(Zt, (np.abs(e),)); solvers_q = _fold_solvers(Zt, folds, aq, _fold_grams(Zt, folds))
        else:
            solvers_q = solvers
        c = _resid_with(solvers_q, Zt, np.abs(e), None)
    s2 = (np.abs(e) - c) ** 2                    # |e|-scale -> variance proxy
    r = _ls_quant_resid(e, s2, folds, taus)
    return [e, c] + [r[t] for t in taus]


def _mixed_combine(Xf, Yf, x_cat, y_cat):
    """Pool the categorical side's levels in one Wald form per feature of the other variable."""
    ps = []
    if x_cat and not y_cat:
        for f in Yf:
            ps.append(_mvgcm_p(Xf * f[:, None]))
    elif y_cat and not x_cat:
        for f in Xf:
            ps.append(_mvgcm_p(Yf * f[:, None]))
    else:                                                  # both categorical: pool X, iterate Y levels
        for j in range(Yf.shape[1]):
            ps.append(_mvgcm_p(Xf * Yf[:, j:j + 1]))
    return _cauchy(ps)


try:
    from threadpoolctl import ThreadpoolController as _TPC
    _TPC_ONE = _TPC()                                # one controller per process: the scan costs ~1 ms otherwise
    _threadpool_limits = _TPC_ONE.limit
    _POOL_NOW = max([i.get("num_threads", 1) for i in _TPC_ONE.info()] or [1])
except Exception:                                  # optional dependency: without it the cap is a no-op
    _threadpool_limits = None; _POOL_NOW = 1


def unified_test(X, Y, Z, taus=None, K=None, seed=0, config=None,
                 x_cat=None, y_cat=None, z_cat=None, detail=False):
    """GFCM conditional-independence p-value (see _unified_test).

    The BLAS thread pool is capped at _BLAS_THREADS for the duration of the call: the test is
    many small matrix products, and a default pool of 64 threads makes them several times slower
    than 16 on the same machine. 0 leaves the pool alone.
    """
    cfg = config if config is not None else GFCMConfig()
    if _BLAS_THREADS > 0 and _threadpool_limits is not None and _POOL_NOW > _BLAS_THREADS:   # a cap, never a raise
        with _threadpool_limits(limits=_BLAS_THREADS):
            return _unified_test(X, Y, Z, taus, K, seed, cfg, x_cat, y_cat, z_cat, detail)
    return _unified_test(X, Y, Z, taus, K, seed, cfg, x_cat, y_cat, z_cat, detail)


def _unified_test(X, Y, Z, taus=None, K=None, seed=0, config=None,
                  x_cat=None, y_cat=None, z_cat=None, detail=False):
    """GFCM conditional-independence p-value for H0: X independent of Y given Z.

    config: a GFCMConfig; defaults to the canonical published configuration.
    taus/K override the config's quantile levels and spline knots.
    x_cat / y_cat / z_cat declare variable types (z_cat is per column); left as None each is
    auto-detected. A categorical X or Y switches to the one-hot residual bank; categorical
    Z-columns enter the design as one-hot dummies.
    detail: a dict of the UNCORRECTED block p-values and every combination rule, for diagnostics
    (continuous X and Y only); the single p-value the test reports is the corrected one.
    """
    cfg = config if config is not None else GFCMConfig()
    if K is None:
        K = k_rule(len(X))
    taus = cfg.taus if taus is None else taus
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
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
    if bool(x_cat) or bool(y_cat):            # mixed-type: one-hot residual bank
        if detail or cfg.comove or cfg.across != "cauchy":
            raise NotImplementedError("comove / across / detail are implemented for continuous X, Y only")
        nn = len(X)
        if Z.shape[1] == 0:
            Zt = None; solvers = None
            folds = list(KFold(cfg.folds, shuffle=True, random_state=42).split(np.zeros((nn, 1))))
        else:
            Zt = _design(Z, K, cfg, z_cat)
            folds = list(KFold(cfg.folds, shuffle=True, random_state=42).split(Zt))
            solvers = _fold_solvers(Zt, folds, None)          # one factorization shared by X and Y
        Xf = _cat_resid(X, Zt, folds, solvers) if x_cat else _cont_feats(X, Zt, folds, solvers, taus, cfg)
        Yf = _cat_resid(Y, Zt, folds, solvers) if y_cat else _cont_feats(Y, Zt, folds, solvers, taus, cfg)
        return _mixed_combine(Xf, Yf, bool(x_cat), bool(y_cat))
    if Z.shape[1] == 0:                       # marginal (|S|=0): center / empirical
        eX = X - X.mean(); eY = Y - Y.mean()
        cX = np.abs(eX) - np.abs(eX).mean(); cY = np.abs(eY) - np.abs(eY).mean()
        rxX = {t: _qind(X, np.quantile(X, t), t) for t in taus}
        rxY = {t: _qind(Y, np.quantile(Y, t), t) for t in taus}
    else:
        Zt = _design(Z, K, cfg, z_cat)
        folds = list(KFold(cfg.folds, shuffle=True, random_state=42).split(Zt))
        al = _gcv_alpha(Zt, (X, Y)) if cfg.gcv == 1 else None
        _proj_on = cfg.proj and Z.shape[1] >= 2 and len(X) >= cfg.proj_nmin and not bool(np.any(z_cat))
        solvers = solvers_q = None; aq = al          # the correction reuses these when available
        if _proj_on:
            Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
            grams = _fold_grams(Zt, folds)                 # base fold Grams, shared with the augmented designs
            pdes = _proj_designs(Zt, Zs, (X, Y), folds, _PROJ_K); _pc = {}
            RXY = _resid_proj(pdes, np.column_stack([X, Y]), al, _pc, Zt, grams)
            eX, eY = RXY[:, 0], RXY[:, 1]
            CXY = _resid_proj(pdes, np.column_stack([np.abs(eX), np.abs(eY)]), al, _pc, Zt, grams)   # same penalty: factors reused
            cX, cY = CXY[:, 0], CXY[:, 1]
        else:
            grams = _fold_grams(Zt, folds)                 # one Gram per fold, shared by the two penalties
            solvers = _fold_solvers(Zt, folds, al, grams)
            RXY = _resid_withM(solvers, Zt, np.column_stack([X, Y]), al)
            eX, eY = RXY[:, 0], RXY[:, 1]
            if cfg.gcv == 2:
                aq = _gcv_alpha(Zt, (np.abs(eX), np.abs(eY)), G_all=grams[0])
                solvers_q = _fold_solvers(Zt, folds, aq, grams)
            else:
                aq, solvers_q = al, solvers      # reuse factorization (default path)
            CXY = _resid_withM(solvers_q, Zt, np.column_stack([np.abs(eX), np.abs(eY)]), aq)
            cX, cY = CXY[:, 0], CXY[:, 1]
        s2X = (np.abs(eX) - cX) ** 2           # |e|-scale -> variance proxy
        s2Y = (np.abs(eY) - cY) ** 2
        rxX = _ls_quant_resid(eX, s2X, folds, taus)
        rxY = _ls_quant_resid(eY, s2Y, folds, taus)
    n = len(X)
    RY = np.column_stack([rxY[t] for t in taus])
    RX = np.column_stack([rxX[t] for t in taus])
    # three blocks: [cov + both scales] | [eX.rY] | [rX.eY]
    Ms = [np.column_stack([eX * eY, cX * eY, eX * cY]),
          eX[:, None] * RY,
          RX * eY[:, None]]
    if detail:
        if cfg.comove:
            raise NotImplementedError("detail is defined without the co-movement block")
        out = _detail(Ms + [_comove_block(cX, cY, RX, RY, taus)], cfg)
        out["p"] = _combine_blocks(Ms, cfg)
        ps3 = [out[f"p_block{i}"] for i in range(3)]
        for tag, part in (("ss", ("scale",)), ("tt", ("tail",))):   # each co-movement part alone
            pb = _mvgcm_p(_comove_block(cX, cY, RX, RY, taus, part))
            out[f"p_{tag}"] = float(pb)
            out[f"acat4_{tag}"] = _cauchy(ps3 + [pb])
        return out
    if cfg.comove:
        Ms.append(_comove_block(cX, cY, RX, RY, taus, cfg.comove_parts))
    if cfg.varcorr and Z.shape[1] > 0 and cfg.across == "cauchy":
        # R + Delta per block. With the single-index block on, the base design's smoother
        # stands in for the augmented one.
        from .varcorr import correct_blocks
        vc = correct_blocks(Zt, folds, float(_RIDGE_ALPHA if al is None else al),
                            float(_RIDGE_ALPHA if aq is None else aq),
                            eX, eY, cX, cY, s2X, s2Y, taus, Ms, cfg, grams=grams,
                            seed=seed, decide_at=cfg.varcorr_decide)
        return vc["p"]
    return _combine_blocks(Ms, cfg)
