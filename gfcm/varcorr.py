"""Finite-sample variance correction for the GFCM block statistics.

WHY. Every score coordinate is a product of two cross-fitted residuals. The self-normaliser R-hat = M'M/n treats
rows as independent, but the fitted values borrow noise from the training rows, and rows in the same fold borrow
from the same rows, so their fit errors are correlated. Under the null the variance of the block score vector is
R + Delta with Delta the cross-observation covariance that R-hat misses; it grows with P/n_train (the design width
over the training size) and is the source of the depth inflation (Paper 2, operating range).

WHAT. For linear smoothers (ridge on the design) Delta has a closed form. Each side of a coordinate is
    a_i = eps_i + sum_j A_ij u_j + sum_j B_ij s_j
(own noise, borrowed mean-fit noise u through the mean smoother W, borrowed scale-fit noise s = |e| - E|e| through
the scale smoother W_q): e side A = -W; c side B = -W_q; r_tau side A = -diag(f/g) W, B = -diag(q f/g) W_q with f
the density of the standardised residual at the pooled quantile q and g the fitted scale. The covariance of a side
across rows i != i' is
    C_ii' = A_ii' c_u,i' + A_i'i c_u,i + B_ii' c_s,i' + B_i'i c_s,i
            + sum_j [A_ij A_i'j D_j + (A_ij B_i'j + B_ij A_i'j) kap_j + B_ij B_i'j Vs_j]
and Delta_ab = (1/m) sum_{i != i'} CX_ab,ii' CY_ab,ii' for every pair of coordinates (a, b) of a block. The corrected
block statistic is T'(G + m Delta)^-1 T on the kept rows (one k x k matrix per block, not a scalar).

HOW (estimation; Cattaneo, Jansson and Newey 2018 for the proof). D_j = Var(u_j | Z) is estimated by deconvolution,
D = (I + W∘W)^-1 e^2 (the cross-fit form of their HC_K; every per-row rescaling of e^2 is inconsistent when P/n does
not vanish, their Theorem 3). It is defined and consistent when the leverage v_i = sum_j W_ij^2 is below 1 on every
test row (diagonal dominance; their max leverage < 1/2), so rows with v_i > 1 are dropped from the TEST side (the
leverage guard; they stay in training). Nothing of size P x P or n x n is formed beyond the fold Choleskys the test
has anyway; every quantity is reached by Rademacher projections through the row maps (Kline, Saggio and Solvsten
2020, App. B.3.2).

WHEN. In LAZY mode (cfg.varcorr_decide = alpha) the correction can only raise p on the kept rows, so it is computed
only when it can matter: every p at or below the band (0.2) is exact; above the band the kept-row p is
returned (a lower bound). Rejection rates at every alpha <= band, and every reported p <= band, are identical to
exact mode by construction (one lemma: R + Delta dominates R on the kept rows). Continuous X, Y with combine "p3",
calib "chi2", across "cauchy" only; with the single-index block on, the base design's smoother stands in for the
per-fold augmented one (a few columns in ~1400, documented approximation).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2, gaussian_kde

from .core import _INTER_MAXD, _cauchy

# internal constants (not user knobs)
_MAXE = 4_000_000          # work arrays stay under this many doubles
_PROBES_MIN = 4            # pair-sum probes: max(_PROBES_MIN, P / _PROBES_DIV)
_PROBES_DIV = 10.0
_PROBES_LEV_MIN = 50       # projections for the leverages and the deconvolution
_LEV_EXACT = 0.5           # rows whose projected leverage exceeds this get the exact value
_F32 = True                # probe products in single precision (variance estimates; rounding 1e-7)
_LEV_GUARD = 1.0           # test rows whose leverage v_i exceeds this are dropped (they stay in training)
_PAIR_ROWS = 5000          # test-row subsample for the pair sums
_BAND = 0.2                # in lazy mode every p-value at or below this is the exact corrected one


def _block_stat(M, Dl=None, ridge=1e-6):
    """The test's Wald statistic T'(G + ridge)^-1 T, with G -> G + m Delta when Dl is given."""
    m, k = M.shape; G = M.T @ M
    if Dl is not None: G = G + m * Dl
    Si = np.linalg.pinv(G + ridge * np.trace(G) / k * np.eye(k)); T = M.sum(0)
    return float(T @ Si @ T)


def _wm(Ain, Aout, T, rows, n_tr, K):
    """sum_i' W_ii' T_i' with W_ii' = 1/n_tr + Aout_i'Ain_i' for i, i' in different folds (zero within a fold)."""
    m, c = T.shape; P = Ain.shape[1]; out = np.empty_like(T); step = max(1, int(_MAXE // max(K * P, 1)))
    for c0 in range(0, c, step):
        Tc = T[:, c0:c0 + step]; cc = Tc.shape[1]
        Tk = [Tc[r].sum(0) if len(r) else np.zeros(cc, dtype=T.dtype) for r in rows]; Hk = [Ain[r].T @ Tc[r] if len(r) else np.zeros((P, cc), dtype=T.dtype) for r in rows]
        Ts, Hs = sum(Tk), sum(Hk)
        for k in range(K):
            r = rows[k]
            if len(r): out[r, c0:c0 + step] = (Ts - Tk[k])[None, :] / n_tr + Aout[r] @ (Hs - Hk[k])
    return out


def _chol_folds(gr, alpha):
    out = []
    for G in gr:
        G = G.copy(); G[np.diag_indices_from(G)] += alpha; out.append(cho_factor(G, lower=True))
    return out


def _wproj(Zc, folds, cfs, V, n_tr, Yk=None):
    """(W V)_i for every row i, V (n x c) on the training rows: (Vs - V_k)/n_tr + z_i' Ginv_k (Ys - Y_k), Y_k = Z[k]'V[k].
    Yk may be passed when V is reused with another penalty."""
    n = len(Zc); out = np.empty((n, V.shape[1]))
    if Yk is None: Yk = [Zc[te].T @ V[te] for _, te in folds]
    Vk = [V[te].sum(0) for _, te in folds]; Ys, Vs = sum(Yk), sum(Vk)
    for k, (tr, te) in enumerate(folds):
        out[te] = (Vs - Vk[k])[None, :] / n_tr + Zc[te] @ cho_solve(cfs[k], Ys - Yk[k])
    return out, Yk


def _lev_exact(Zc, folds, cfs, alpha, rows, n_tr):
    """Exact v_i = 1/n_tr + 2 a_i'u1/n_tr + a_i'z_i - alpha ||a_i||^2 for the rows given (a boolean mask)."""
    v = np.full(len(Zc), np.nan)
    for k, (tr, te) in enumerate(folds):
        r = te[rows[te]]
        if not len(r): continue
        A = cho_solve(cfs[k], Zc[r].T).T; z = Zc[r]
        v[r] = 1.0 / n_tr + 2.0 * (A @ Zc[tr].sum(0)) / n_tr + (A * z).sum(1) - alpha * (A * A).sum(1)
    return v


def _rowmaps(Zc, folds, cfs, idx):
    """a_i = Ginv_k z_i for the rows idx (in idx order)."""
    fold_of = np.empty(len(Zc), int)
    for k, (_, te) in enumerate(folds): fold_of[te] = k
    A = np.empty((len(idx), Zc.shape[1])); fS = fold_of[idx]
    for k in range(len(folds)):
        ik = np.where(fS == k)[0]
        if len(ik): A[ik] = cho_solve(cfs[k], Zc[idx[ik]].T).T
    return A, fS


def _refuse(P, n_tr, Ms, n, dropped=None, vmax=float("nan")):
    """Beyond the test's domain: more design columns than training rows, or the guard keeps under 10 percent of the
    rows. No valid p-value exists here in either mode (the uncorrected one is far too small: .74 at n=1000, |S|=20),
    so the test REFUSES: p = NaN with a warning. Inside PC a NaN is "not tested" (the edge is kept), and
    `max_cond_set_for(n)` gives the largest |S| inside the domain to use as the cap."""
    import warnings
    warnings.warn(f"GFCM: {P} design columns on {n_tr} training rows (P/n_train = {P / n_tr:.2f}); the test is "
                  f"undefined here and returns NaN. Cap the conditioning set at max_cond_set_for(n).", RuntimeWarning)
    return dict(p=float("nan"), p_blocks=[float("nan")] * len(Ms), etas=None, dropped=n if dropped is None else dropped, vmax=vmax, skipped=False, gated="beyond-domain")


def _coords(taus, comove_parts=None):
    """Side pairs of every block, in the column order the test builds them (see core._comove_block).
    comove_parts (None = no co-movement block) appends a FOURTH block: "scale" is the c-c product and "tail" the
    matched corners of the lowest and highest quantile level."""
    cs = [[("e", "e"), ("c", "e"), ("e", "c")], [("e", ("r", t)) for t in taus], [(("r", t), "e") for t in taus]]
    if comove_parts:
        lo, hi = min(taus), max(taus); cm = []
        if "scale" in comove_parts:
            cm.append(("c", "c"))
        if "tail" in comove_parts:
            cm.append((("r", lo), ("r", lo)))
            if hi != lo:
                cm.append((("r", hi), ("r", hi)))
        cs.append(cm)
    return cs


def correct_blocks(Zt, folds, alpha_e, alpha_q, eX, eY, cX, cY, s2X, s2Y, taus, Ms, cfg,
                   grams=None, seed=0, decide_at=None):
    """Returns dict(p, p_blocks, p_raw, etas, dropped, vmax, skipped, gated). p is the Cauchy of the corrected
    block p-values. Ms = [B1 (e_X e_Y, c_X e_Y, e_X c_Y), B2 (e_X r_tau(Y)), B3 (r_tau(X) e_Y)] on all rows.

    Every quantity is reached by Rademacher projections through the row maps:
      * leverage v_i = sum_j W_ij^2 = ||W_i.||^2 by projection, (W R)_i = (Rs - R_k)/n_tr + z_i' Ginv_k (Z'R)_(-k),
        one solve per fold; rows whose projected leverage is above _LEV_EXACT get the exact value, so the guard is
        exact.
      * deconvolution ((W∘W) D0)_i = ||W_i. diag(sqrt D0)||^2 by the same device with fresh projections.
      * pair sums: C' G through W' (idx -> all training rows), the weights, and W (all rows -> idx); the base
        diagonal is estimated from the same probes, c_i = (1/p) sum_g g_i (C'g)_i, whose covariance with the probe
        total is exactly (1/p) times the off-diagonal sum, so Delta = [total - sum_i cX_i cY_i] / (1 - 1/p) is
        unbiased.
    """
    n, P = Zt.shape; n_tr = len(folds[0][0]); K = len(folds); rng = np.random.default_rng(seed)
    kdf = [M.shape[1] for M in Ms]        # every block is referred to chi^2 on ITS OWN number of coordinates
    if P >= n_tr: return _refuse(P, n_tr, Ms, n)                           # beyond the domain: decided from the sizes alone
    mz = Zt.mean(0); Zc = Zt - mz
    if grams is not None: gr = [G + len(tr) * np.outer(mu - mz, mu - mz) for (G, mu), (tr, _) in zip(grams[1], folds)]
    else: gr = [Zc[tr].T @ Zc[tr] for tr, _ in folds]
    cfe = _chol_folds(gr, alpha_e)
    p = max(_PROBES_MIN, int(round(P / _PROBES_DIV))); p_lev = max(_PROBES_LEV_MIN, p)
    # --- leverages by projection; exact where the projection says the guard may bind ---
    R = rng.choice([-1.0, 1.0], size=(n, p_lev))
    WR, Yk = _wproj(Zc, folds, cfe, R, n_tr); v = (WR * WR).sum(1) / p_lev
    cand = v > _LEV_EXACT
    if cand.any(): v = np.where(cand, _lev_exact(Zc, folds, cfe, alpha_e, cand, n_tr), v)
    kept = np.where(v <= _LEV_GUARD)[0]; mk = len(kept); dropped = n - mk
    if mk < max(50, 0.1 * n): return _refuse(P, n_tr, Ms, n, dropped, float(v.max()))
    p_kept = [float(chi2.sf(_block_stat(M[kept]), k)) for M, k in zip(Ms, kdf)]
    if decide_at is not None and _cauchy(p_kept) > max(_BAND, decide_at):
        return dict(p=_cauchy(p_kept), p_blocks=p_kept, etas=None, dropped=dropped, vmax=float(v.max()), skipped=False, gated="above-band")
    # Everything below is only reached when the corrected p can change a decision, so the scale
    # smoother is factored here and not above the gate: on a gated call it was pure waste.
    cfq = cfe if alpha_q == alpha_e else _chol_folds(gr, alpha_q)
    WRq, _ = _wproj(Zc, folds, cfq, R, n_tr, Yk); vq = (WRq * WRq).sum(1) / p_lev
    # --- deconvolution D = e2 - (W∘W) D0, D0 = e2/(1+v), by projection with fresh signs; rows with v > 1 frozen ---
    e2X, e2Y = eX ** 2, eY ** 2; D0X, D0Y = e2X / (1.0 + v), e2Y / (1.0 + v)
    if v.max() < 0.2: DX, DY = D0X, D0Y
    else:
        R2 = rng.choice([-1.0, 1.0], size=(n, p_lev)); V = np.hstack([np.sqrt(D0X)[:, None] * R2, np.sqrt(D0Y)[:, None] * R2])
        Q, _ = _wproj(Zc, folds, cfe, V, n_tr); qX = (Q[:, :p_lev] ** 2).sum(1) / p_lev; qY = (Q[:, p_lev:] ** 2).sum(1) / p_lev
        frozen = v > 1.0; DX = np.where(frozen, D0X, e2X - qX); DY = np.where(frozen, D0Y, e2Y - qY)
    VsX, VsY = cX ** 2 / (1.0 + vq), cY ** 2 / (1.0 + vq); kapX, kapY = eX * cX, eY * cY
    gX = np.sqrt(np.maximum(s2X, 1e-3 * np.mean(s2X) + 1e-12)); gY = np.sqrt(np.maximum(s2Y, 1e-3 * np.mean(s2Y) + 1e-12))
    uX, uY = eX / gX, eY / gY

    def quant(u):
        r, q, f = {}, {}, {}
        sub = u[rng.choice(n, min(n, 20000), replace=False)]; kde = gaussian_kde(sub)
        for t in taus:
            rr = np.empty(n); qq = 0.0
            for tr, te in folds:
                qt = np.quantile(u[tr], t)
                rr[te] = t - ((u[te] < qt) + 0.5 * (u[te] == qt))
                qq += qt / K
            r[t], q[t], f[t] = rr, qq, float(kde(qq)[0])
        return r, q, f
    rX, qX_, fX = quant(uX); rY, qY_, fY = quant(uY)
    # --- test-row subsample for the pair sums, its row maps ---
    m_pairs = min(mk, _PAIR_ROWS); idx = kept if mk <= m_pairs else np.sort(rng.choice(kept, m_pairs, replace=False)); m = len(idx)
    scale = mk * (mk - 1) / (m * (m - 1)) if m < mk else 1.0
    Ae, fS = _rowmaps(Zc, folds, cfe, idx); Aq, _ = _rowmaps(Zc, folds, cfq, idx); rows = [np.where(fS == k)[0] for k in range(K)]
    one_m = np.ones(m)

    def sides(e, c, g, D_, kap_, Vs_, r, q, f):
        S = {"e": dict(cu=D_[idx], cs=kap_[idx]), "c": dict(cu=kap_[idx], cs=Vs_[idx])}
        for t in taus: S[("r", t)] = dict(cu=(r[t] * e)[idx], cs=(r[t] * c)[idx])
        return S
    SX = sides(eX, cX, gX, DX, kapX, VsX, rX, qX_, fX); SY = sides(eY, cY, gY, DY, kapY, VsY, rY, qY_, fY)
    coords = _coords(taus, cfg.comove_parts if cfg.comove else None)
    dt = np.float32 if _F32 else np.float64
    Zc, Ae, Aq = Zc.astype(dt, copy=False), Ae.astype(dt, copy=False), Aq.astype(dt, copy=False); Zi = Zc[idx]
    G = rng.choice([-1.0, 1.0], size=(m, p)).astype(dt); S3 = np.hstack([G, G / gX[idx][:, None], G / gY[idx][:, None]]).astype(dt)

    def wT(A, T):
        """(W'T)_j for every training row j, T on the idx rows: (Ts - T_k)/n_tr + z_j'(Hs - H_k), H_k = A[k]'T[k]."""
        Hk = [A[r].T @ T[r] if len(r) else np.zeros((P, T.shape[1])) for r in rows]; Tk = [T[r].sum(0) if len(r) else np.zeros(T.shape[1]) for r in rows]
        Hs, Ts = sum(Hk), sum(Tk); out = np.empty((n, T.shape[1]))
        for k, (_, te) in enumerate(folds): out[te] = (Ts - Tk[k])[None, :] / n_tr + Zc[te] @ (Hs - Hk[k])
        return out

    def wU(A, U):
        """(W U)_i for the idx rows, U on every training row: (Us - U_k)/n_tr + a_i'(Ys - Y_k), Y_k = Z[k]'U[k]."""
        Yk = [Zc[te].T @ U[te] for _, te in folds]; Uk = [U[te].sum(0) for _, te in folds]; Ys, Us = sum(Yk), sum(Uk)
        out = np.empty((m, U.shape[1]))
        for k in range(K):
            r = rows[k]
            if len(r): out[r] = (Us - Uk[k])[None, :] / n_tr + A[r] @ (Ys - Yk[k])
        return out
    WeS, WqS = wT(Ae, S3), wT(Aq, S3)
    # S3 is [G | G/gX | G/gY], so X and Y share its first p columns: these two run once on all 3p
    # columns instead of once per variable on 2p, and each variable slices out the pair it needs.
    WetA, WqtA = _wm(Ae, Zi, S3, rows, n_tr, K), _wm(Aq, Zi, S3, rows, n_tr, K)
    cols = {"X": np.r_[0:p, p:2 * p], "Y": np.r_[0:p, 2 * p:3 * p]}
    off = {"X": p, "Y": 2 * p}
    sidesd = {"X": SX, "Y": SY}; shapes, ops = {}, {}
    for w, g, f, q, D_, kap_, Vs_ in (("X", gX, fX, qX_, DX, kapX, VsX), ("Y", gY, fY, qY_, DY, kapY, VsY)):
        D_, kap_, Vs_ = D_.astype(dt), kap_.astype(dt), Vs_.astype(dt)
        Te, Tq = WeS[:, cols[w]], WqS[:, cols[w]]
        Ue = wU(Ae, np.hstack([D_[:, None] * Te, kap_[:, None] * Tq]))                     # ee' T | eq T   (output on the e side)
        Uq = wU(Aq, np.hstack([kap_[:, None] * Te, Vs_[:, None] * Tq]))                    # eq' T | qq' T  (output on the q side)
        split = lambda M: {1: M[:, :p], "inv": M[:, p:2 * p]}
        V = {1: one_m.astype(dt), "inv": (1.0 / g[idx]).astype(dt)}
        shapes[w] = {"e": (1.0, 0.0, 1), "c": (0.0, 1.0, 1), **{a: (f[a[1]], q[a[1]] * f[a[1]], "inv") for a in sidesd[w] if a not in ("e", "c")}}
        keys = list(sidesd[w]); o = off[w]
        for a in keys: sidesd[w][a]["cu"] = sidesd[w][a]["cu"].astype(dt); sidesd[w][a]["cs"] = sidesd[w][a]["cs"].astype(dt)
        cuG = np.hstack([sidesd[w][a]["cu"][:, None] * G for a in keys]); csG = np.hstack([sidesd[w][a]["cs"][:, None] * G for a in keys])
        Wem, Wqm = _wm(Zi, Ae, cuG, rows, n_tr, K), _wm(Zi, Aq, csG, rows, n_tr, K)
        ops[w] = dict(EE=split(Ue[:, :2 * p]), EQm=split(Ue[:, 2 * p:]), EQt=split(Uq[:, :2 * p]), QQ=split(Uq[:, 2 * p:]),
                      Wet={1: WetA[:, :p], "inv": WetA[:, o:o + p]}, Wqt={1: WqtA[:, :p], "inv": WqtA[:, o:o + p]},
                      Wem={a: Wem[:, i * p:(i + 1) * p] for i, a in enumerate(keys)}, Wqm={a: Wqm[:, i * p:(i + 1) * p] for i, a in enumerate(keys)}, V=V)
    cache = {}

    def Ct(w, a, b):
        """C^w_ab' G (m x p) and its probe-estimated base diagonal."""
        key = (w, a, b)
        if key in cache: return cache[key]
        sa, sa2, ka = shapes[w][a]; sb, sb2, kb = shapes[w][b]; O = ops[w]; Sb = sidesd[w][b]
        ab, bb = (sb * O["V"][kb])[:, None], (sb2 * O["V"][kb])[:, None]; out = np.zeros((m, p), dtype=dt)
        if sa and sb: out += ab * (sa * O["EE"][ka])
        if sa and sb2: out += bb * (sa * O["EQt"][ka])
        if sa2 and sb: out += ab * (sa2 * O["EQm"][ka])
        if sa2 and sb2: out += bb * (sa2 * O["QQ"][ka])
        if sb: out -= ab * O["Wem"][a]
        if sb2: out -= bb * O["Wqm"][a]
        if sa: out -= Sb["cu"][:, None] * (sa * O["Wet"][ka])
        if sa2: out -= Sb["cs"][:, None] * (sa2 * O["Wqt"][ka])
        cache[key] = (out, (G * out).sum(1, dtype=np.float64) / p); return cache[key]
    p_blocks, etas = [], []
    for M, cs in zip(Ms, coords):
        Mk = M[kept]; Rm = Mk.T @ Mk / mk; kb = len(cs); Dl = np.zeros((kb, kb))
        for a in range(kb):
            for b in range(a, kb):
                CX, dX = Ct("X", cs[a][0], cs[b][0]); CY, dY = Ct("Y", cs[a][1], cs[b][1])
                Dl[a, b] = Dl[b, a] = scale * (float(np.einsum("ij,ij->", CX, CY, dtype=np.float64)) / p - float(dX @ dY)) / (1.0 - 1.0 / p) / mk
        p_blocks.append(float(chi2.sf(_block_stat(Mk, Dl), kb))); etas.append(np.diag(Dl) / np.diag(Rm))
    return dict(p=_cauchy(p_blocks), p_blocks=p_blocks, p_raw=p_kept, etas=[np.asarray(e) for e in etas], dropped=dropped, vmax=float(v.max()), skipped=False, gated=False, probes=p)


def design_width(dS, n, cfg=None):
    """Number of design columns the test builds for |S| = dS continuous conditioners at sample size n
    (spline knots K(n) + 3 per conditioner, the raw linear block, and 3 C(dS, 2) interaction columns)."""
    from .core import k_rule
    K = k_rule(n); inter = 3 * dS * (dS - 1) // 2 if (cfg is None or cfg.inter) and dS <= _INTER_MAXD else 0
    return dS * (K + 3) + dS + inter


def max_cond_set_for(n, folds=5, cfg=None):
    """Largest |S| inside the test's domain at sample size n: design_width(|S|, n) < n (1 - 1/folds). Use it as PC's
    conditioning-set cap; beyond it the test returns NaN."""
    n_tr = int(n * (1 - 1.0 / folds)); d = 0
    while design_width(d + 1, n, cfg) < n_tr: d += 1
    return d
