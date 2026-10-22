"""MinT with diagonal shrinkage, in three computational forms with the same result (Section 5.2).

W = (1 - lam) E^T E / T + lam diag(psi), psi_i = mean of e_i^2 (uncentred). With F = U E^T (K x T):
  * dense: W is formed (N x N);                          O(T N^2 + K^3) time, O(N^2) memory
  * K x K: U W U^T = (1 - lam)/T F F^T + lam U diag(psi) U^T, W x applied from E;   O(T N + K^2 T + K^3), O(N + K T + K^2)
  * T x T: Woodbury identity on the same matrix (sparse LU of the tree-structured part); O(T N + K T^2 + T^3), O(N + K T + T^2)
MinT-Shrink adds the Schafer-Strimmer intensity, O(T N min(T, N)) time and O(min(T, N)^2) memory.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


def ss_lambda(E: np.ndarray, block: int = 256) -> float:
    """Schafer-Strimmer shrinkage intensity (uncentred correlations, diagonal target), through the N x N
    Gram matrix if N < T and through the T x T one otherwise: O(T N min(T, N)) time, O(min(T, N)^2) memory."""
    n, N = E.shape
    d = np.einsum("ti,ti->i", E, E) / n
    isd = 1.0 / np.sqrt(np.maximum(d, 1e-300))
    if N < n:                                   # N x N route, row blocks
        P = np.zeros((N, N)); Q = np.zeros((N, N))
        for r0 in range(0, n, block):
            xb = E[r0:r0 + block] * isd
            P += xb.T @ xb
            x2 = xb * xb
            Q += x2.T @ x2
        v = (Q - P ** 2 / n) / (n * (n - 1))
        np.fill_diagonal(v, 0.0)
        corm = P / n
        np.fill_diagonal(corm, 0.0)
        return float(np.clip(v.sum() / max((corm ** 2).sum(), 1e-300), 0.0, 1.0))
    G = np.zeros((n, n)); rs = np.zeros(n); s_q_diag = 0.0; col2 = np.empty(N)
    for c0 in range(0, N, block):
        xb = E[:, c0:c0 + block] * isd[c0:c0 + block]
        G += xb @ xb.T
        x2 = xb * xb
        rs += x2.sum(1); s_q_diag += float(np.sum(x2 * x2)); col2[c0:c0 + block] = x2.sum(0)
    s_q_all = float(np.sum(rs ** 2)); s_p2_all = float(np.sum(G * G)); s_p2_diag = float(np.sum(col2 ** 2))
    num = ((s_q_all - s_q_diag) - (s_p2_all - s_p2_diag) / n) / (n * (n - 1))
    den = (s_p2_all - s_p2_diag) / n ** 2
    return float(np.clip(num / max(den, 1e-300), 0.0, 1.0))


def _UEt(U, E, block: int = 256) -> np.ndarray:
    """U E^T (K x T) without a transposed copy of E."""
    n = E.shape[0]
    out = np.empty((U.shape[0], n))
    for r0 in range(0, n, block):
        out[:, r0:r0 + block] = np.asarray(U @ np.ascontiguousarray(E[r0:r0 + block].T))
    return out


def _mint_lowrank(Yhat, U, E, lam="ss", solver="auto"):
    """K x K form (solver 'direct') or T x T form (solver 'woodbury'); 'auto': direct if K <= T."""
    import scipy.sparse.linalg as spla
    E = np.asarray(E, float)
    n, N = E.shape
    lam_v = ss_lambda(E) if lam == "ss" else float(lam)
    U = U.tocsr() if sp.issparse(U) else sp.csr_matrix(np.asarray(U))
    K = U.shape[0]
    Y = np.asarray(Yhat, float)
    vec = Y.ndim == 1
    if vec:
        Y = Y[:, None]
    d = np.einsum("ti,ti->i", E, E) / n
    UE = _UEt(U, E)                                                   # K x T
    r = np.asarray(U @ Y)
    if solver == "direct" or (solver == "auto" and K <= n):
        M = (1.0 - lam_v) / n * (UE @ UE.T) + lam_v * (U @ sp.diags(d) @ U.T).toarray()
        a = sla.cho_solve(sla.cho_factor(M, overwrite_a=True), r)
    else:
        A = (lam_v * (U @ sp.diags(d) @ U.T)).tocsc()
        lu = spla.splu(A)
        B = np.sqrt((1.0 - lam_v) / n) * UE
        AiB = lu.solve(B)
        C = np.eye(n) + B.T @ AiB
        Air = lu.solve(r)
        a = Air - AiB @ sla.cho_solve(sla.cho_factor(C, overwrite_a=True), B.T @ Air)
    V = np.asarray(U.T @ a)
    out = Y - ((1.0 - lam_v) / n * (E.T @ (E @ V)) + lam_v * d[:, None] * V)
    return (out[:, 0] if vec else out), lam_v


def _mint_dense(Yhat, U, E, lam="ss"):
    """Dense form: forms the N x N covariance and factorizes U W U^T (Cholesky)."""
    E = np.asarray(E, float)
    n = E.shape[0]
    covm = E.T @ E / n
    if lam == "ss":                                    # Schafer-Strimmer through the N x N matrices
        xs = E / np.sqrt(np.maximum(np.diag(covm), 1e-300))
        corm = xs.T @ xs / n
        v = ((xs ** 2).T @ (xs ** 2) - (xs.T @ xs) ** 2 / n) / (n * (n - 1))
        np.fill_diagonal(v, 0.0)
        np.fill_diagonal(corm, 0.0)
        lam_v = float(np.clip(v.sum() / max((corm ** 2).sum(), 1e-300), 0.0, 1.0))
    else:
        lam_v = float(lam)
    W = lam_v * np.diag(np.diag(covm)) + (1 - lam_v) * covm
    Ud = U.toarray() if sp.issparse(U) else np.asarray(U, float)
    Y = np.asarray(Yhat, float)
    WUt = W @ Ud.T
    c = sla.cho_factor(Ud @ WUt)
    return Y - WUt @ sla.cho_solve(c, Ud @ Y), lam_v


def mint(Yhat, U, E, lam="ss", form="auto"):
    """MinT reconciliation  y - W U^T (U W U^T)^{-1} U y  with  W = (1 - lam) E^T E / T + lam diag(E^T E / T).

    lam   'ss' (MinT-Shrink: Schafer-Strimmer intensity) or a number (MinT-Fix uses 0.1)
    form  'dense' (N x N matrix), 'kxk' (K x K system, no N x N matrix), 'txt' (Woodbury, T x T system)
          or 'auto' ('kxk' if K <= T, else 'txt'). All forms give the same result up to rounding.
    Returns (reconciled forecasts, lam)."""
    if form == "dense":
        return _mint_dense(Yhat, U, E, lam)
    solver = {"kxk": "direct", "txt": "woodbury", "auto": "auto"}[form]
    return _mint_lowrank(Yhat, U, E, lam, solver)
