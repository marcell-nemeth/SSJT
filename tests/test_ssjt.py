"""Correctness tests: coherence, matrix-free SSJT = explicit family matrices, MinT forms agree."""
import numpy as np
import pytest

from ssjt import SSJT, Hierarchy, mint, ss_lambda


def _problem(h, T, seed=0, H=3):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((T, h.N)) @ (np.eye(h.N) + 0.3 * rng.standard_normal((h.N, h.N)))
    Yhat = rng.uniform(10, 100, (h.N, H))
    return E, Yhat


def _reference(h, E, Yhat, lam=0.1, safeguard="convex"):
    """SSJT with explicit (m+1) x (m+1) family covariance matrices, one family at a time."""
    T = E.shape[0]
    depth = {h.root: 0}
    for n in h.pre_order:
        for c in h.children.get(n, []):
            depth[c] = depth[n] + 1
    fams = sorted([p for p in h.nodes if h.children.get(p)], key=lambda p: -depth[p])
    Y = Yhat.copy()
    Sig = {}
    for k in range(Yhat.shape[1]):
        for p in fams:                                            # upward pass
            ix = [h.idx[p]] + [h.idx[c] for c in h.children[p]]
            Ek = E[:, ix]; W = Ek.T @ Ek / (T - 1)
            s = np.sqrt(np.diag(W))
            yc = np.maximum(Yhat[ix[1:], k], 0); w = yc / yc.sum() if yc.sum() > 0 else np.full(len(yc), 1 / len(yc))
            a = np.r_[1.0, np.sqrt(w)]
            Tk = np.diag(s) @ (np.outer(a, a) + np.diag(np.r_[0.0, 1 - w])) @ np.diag(s)
            eps = max(1e-6 * np.trace(W), 1e-12)
            S = (1 - lam) * W + lam * Tk + eps * np.eye(len(ix))
            Sig[(p, k)] = (S, s)
            u = np.r_[1.0, -np.ones(len(ix) - 1)]
            g = S @ u; v = u @ g; eta = u @ Y[ix, k]
            kap = g[0] / v; q = -g[1:] / (-g[1:]).sum()
            if safeguard == "convex":
                kap = min(max(kap, 0.0), 1.0)
                qq = np.maximum(-g[1:], 0)
                q = qq / qq.sum() if qq.sum() > 0 else s[1:] ** 2 / (s[1:] ** 2).sum()
            Y[ix[0], k] -= kap * eta
            Y[ix[1:], k] += q * (1 - kap) * eta
        for p in reversed(fams):                                  # downward pass
            ix = [h.idx[p]] + [h.idx[c] for c in h.children[p]]
            S, s = Sig[(p, k)]; s1 = S[1:, 1:] @ np.ones(len(ix) - 1)
            om = s1 / s1.sum()
            if safeguard == "convex":
                pp = np.maximum(s1, 0)
                om = pp / pp.sum() if pp.sum() > 0 else s[1:] ** 2 / (s[1:] ** 2).sum()
            Y[ix[1:], k] += om * (Y[ix[0], k] - Y[ix[1:], k].sum())
    return Y


TREES = [Hierarchy.balanced(3, 3), Hierarchy.balanced(2, 5),
         Hierarchy.imbalanced(5, 4, np.random.default_rng(1))]


@pytest.mark.parametrize("h", TREES)
@pytest.mark.parametrize("safeguard", ["convex", "none"])
def test_matrix_free_equals_explicit(h, safeguard):
    E, Yhat = _problem(h, T=60)
    fast = SSJT(h, E, 0.1, safeguard=safeguard).reconcile(Yhat)
    ref = _reference(h, E, Yhat, 0.1, safeguard)
    assert np.max(np.abs(fast - ref)) < 1e-8


@pytest.mark.parametrize("h", TREES)
def test_coherent(h):
    E, Yhat = _problem(h, T=40)
    Yt = SSJT(h, E).reconcile(Yhat)
    assert np.max(np.abs(h.U @ Yt)) < 1e-8


@pytest.mark.parametrize("T", [30, 400])              # T < N and T > N
@pytest.mark.parametrize("lam", ["ss", 0.1])
def test_mint_forms_agree(T, lam):
    h = Hierarchy.balanced(3, 4)                          # N = 121, K = 40
    E, Yhat = _problem(h, T)
    d, ld = mint(Yhat, h.U, E, lam, "dense")
    k, lk = mint(Yhat, h.U, E, lam, "kxk")
    t, lt = mint(Yhat, h.U, E, lam, "txt")
    assert abs(ld - lk) < 1e-12 and abs(ld - lt) < 1e-12
    assert np.max(np.abs(d - k)) < 1e-8 and np.max(np.abs(d - t)) < 1e-8
    assert np.max(np.abs(h.U @ d)) < 1e-8


def test_safeguard_keeps_parent_between_values():
    h = Hierarchy.balanced(2, 1)                          # one family: root and two children
    rng = np.random.default_rng(3)
    e = rng.standard_normal((50, 2))
    E = np.c_[-2 * e.sum(1), e]                           # strongly negative parent-children correlation
    Yhat = np.array([[100.0], [30.0], [40.0]])
    yp = SSJT(h, E, 0.1).reconcile(Yhat)[0, 0]
    assert 70.0 - 1e-9 <= yp <= 100.0 + 1e-9
