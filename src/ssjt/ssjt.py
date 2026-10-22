"""SSJT: Structurally Shrunk Junction Tree reconciliation (matrix-free, with the convex safeguard).

The tree is split into parent-children families C_k = {p} + child(p). Each family covariance is
    Sigma_k = (1 - lam) W_k + lam T_k + eps_k I,
where W_k = E_k^T E_k / (T - 1) is the family sample covariance and T_k = D R D the structural target
(R = a a^T + diag(0, 1 - w), a = (1, sqrt(w)), w = shares of the children's base forecasts).
The two traversals use Sigma_k only through Sigma_k u (u = (1, -1, ..., -1)) and Sigma_cc 1, so:
  * Phase I   one pass over the rows of E accumulates the variances, W_k u and W_cc 1 of all families: O(T N);
  * Phase II  upward pass, deepest families first: local MinT step, split as kappa (parent) and q (children);
  * Phase III downward pass from the root: the children are fixed to their parent with weights omega.
No covariance matrix is formed. Runtime O(T N) per fit plus O(N) per forecast vector; memory O(N).
Families whose parents have the same depth share no node and are processed together (vectorized).
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .hierarchy import Hierarchy


def _segsum(x, starts):
    return np.add.reduceat(x, starts, axis=0)


class SSJT:
    def __init__(self, hier: Hierarchy, E: np.ndarray, lam: float = 0.1, ridge_rel: float = 1e-6,
                 block: int = 32, safeguard: str = "convex"):
        """Phase I: residual statistics of all families in one pass over E (row chunks of length `block`).

        hier      single-parent tree; the columns of E are ordered as hier.nodes
        E         (T, N) in-sample one-step residuals of the base models
        lam       shrinkage intensity towards the structural target (0.1 in the article)
        ridge_rel ridge eps_k = ridge_rel * tr(W_k) (at least 1e-12)
        safeguard 'convex' (the article's SSJT): kappa clipped to [0, 1], allocation weights of the children
                  clipped to be non-negative and renormalized (variance shares if all are zero);
                  'none': the unconstrained local MinT steps (used in the tests).
        """
        if safeguard not in ("convex", "none"):
            raise ValueError("safeguard must be 'convex' or 'none'")
        self.safeguard = safeguard
        t0 = time.perf_counter()
        self.h, self.lam = hier, float(lam)
        E = np.asarray(E, dtype=float)
        T = E.shape[0]
        depth = {hier.root: 0}
        for n in hier.pre_order:
            for c in hier.children.get(n, []):
                depth[c] = depth[n] + 1
        idx = hier.idx
        parents = [n for n in hier.nodes if hier.children.get(n)]
        N, F = E.shape[1], len(parents)
        fidx = {p: k for k, p in enumerate(parents)}
        pcol = np.array([idx[p] for p in parents])
        # child -> family incidence (N x F), one nonzero per non-root node
        rows, cols = [], []
        for p in parents:
            for c in hier.children[p]:
                rows.append(idx[c]); cols.append(fidx[p])
        C = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, F))
        CTc = C.T.tocsr()
        var = np.zeros(N); Su_p_all = np.zeros(F); Su_node = np.zeros(N); Scc1_node = np.zeros(N)
        for r0 in range(0, T, block):                     # row blocks: O(block * N) extra memory
            Eb = E[r0:r0 + block]
            ECb = np.asarray((CTc @ Eb.T).T)              # E_c 1 per family, block x F
            EUb = Eb[:, pcol] - ECb                       # E_k u per family
            var += np.einsum("ti,ti->i", Eb, Eb)
            Su_p_all += np.einsum("tf,tf->f", Eb[:, pcol], EUb)
            Su_node += np.einsum("ti,ti->i", Eb, np.asarray((C @ EUb.T).T))
            Scc1_node += np.einsum("ti,ti->i", Eb, np.asarray((C @ ECb.T).T))
        var /= T - 1; Su_p_all /= T - 1; Su_node /= T - 1; Scc1_node /= T - 1
        sig = np.sqrt(np.maximum(var, 0.0))
        self.levels = []
        for dl in sorted({depth[p] for p in parents}, reverse=True):
            fam = [p for p in parents if depth[p] == dl]
            fk = np.array([fidx[p] for p in fam])
            pi = pcol[fk]
            ch = [np.array([idx[c] for c in hier.children[p]]) for p in fam]
            m = np.array([len(c) for c in ch])
            ci = np.concatenate(ch)
            seg = np.repeat(np.arange(len(fam)), m)
            starts = np.concatenate([[0], np.cumsum(m)[:-1]])
            eps = np.maximum((var[pi] + _segsum(var[ci], starts)) * ridge_rel, 1e-12)
            self.levels.append(dict(pi=pi, ci=ci, seg=seg, starts=starts, m=m, Su_p=Su_p_all[fk],
                                    Su_c=Su_node[ci], Scc1=Scc1_node[ci], eps=eps, sp=sig[pi], sc=sig[ci]))
        self.runtimes = {"precompute": time.perf_counter() - t0, "upward": 0.0, "downward": 0.0}

    def reconcile(self, Yhat: np.ndarray) -> np.ndarray:
        Y = np.asarray(Yhat, float)
        one = Y.ndim == 1
        if one:
            Y = Y[:, None]
        lam = self.lam
        yt = Y.copy()
        t0 = time.perf_counter()
        s1 = []
        for L in self.levels:
            pi, ci, seg, st = L["pi"], L["ci"], L["seg"], L["starts"]
            sp, sc = L["sp"][:, None], L["sc"][:, None]
            eps = L["eps"][:, None]
            ypos = np.maximum(Y[ci], 0.0)
            tot = _segsum(ypos, st)
            ok = tot > 0
            w = np.where(ok[seg], ypos / np.where(ok, tot, 1.0)[seg], 1.0 / L["m"][seg][:, None])
            a = np.sqrt(np.clip(w, 0.0, 1.0))
            av = sp - _segsum(a * sc, st)                  # a^T D u
            Tu_p = sp * av
            Tu_c = sc * (a * av[seg] - (1.0 - w) * sc)
            Gp = lam * Tu_p + (1 - lam) * L["Su_p"][:, None] + eps
            Gc = lam * Tu_c + (1 - lam) * L["Su_c"][:, None] - eps[seg]
            vr = Gp - _segsum(Gc, st)
            vr = np.where(vr == 0, 1e-12, vr)
            disc = yt[pi] - _segsum(yt[ci], st)
            if self.safeguard == "none":
                r = disc / vr
                yt[pi] = yt[pi] - Gp * r
                yt[ci] = yt[ci] - Gc * r[seg]
            else:
                kap = np.clip(Gp / vr, 0.0, 1.0)                # parent moves by kap * disc
                q = np.maximum(-Gc, 0.0)                         # children receive (1 - kap) * disc
                qs = _segsum(q, st)
                vv = (sc ** 2) * np.ones_like(q)
                q = np.where((qs > 0)[seg], q / np.where(qs > 0, qs, 1.0)[seg], vv / _segsum(vv, st)[seg])
                yt[pi] = yt[pi] - kap * disc
                yt[ci] = yt[ci] + q * ((1.0 - kap) * disc)[seg]
            Tcc1 = sc * (a * _segsum(a * sc, st)[seg] + (1.0 - w) * sc)
            s1.append(lam * Tcc1 + (1 - lam) * L["Scc1"][:, None] + eps[seg])
        t1 = time.perf_counter()
        ys = np.zeros_like(yt)
        top = self.levels[-1]["pi"]
        ys[top] = yt[top]
        for L, S1 in zip(reversed(self.levels), reversed(s1)):
            pi, ci, seg, st = L["pi"], L["ci"], L["seg"], L["starts"]
            den = _segsum(S1, st)
            okd = np.abs(den) > 1e-12
            wc = np.where(okd[seg], S1 / np.where(okd, den, 1.0)[seg], 1.0 / L["m"][seg][:, None])
            if self.safeguard == "convex":
                wp = np.maximum(S1, 0.0); ws = _segsum(wp, st)
                vv = (L["sc"][:, None] ** 2) * np.ones_like(wp)
                wc = np.where((ws > 0)[seg], wp / np.where(ws > 0, ws, 1.0)[seg], vv / _segsum(vv, st)[seg])
            ys[ci] = yt[ci] + wc * (ys[pi] - _segsum(yt[ci], st))[seg]
        self.runtimes["upward"] += t1 - t0
        self.runtimes["downward"] += time.perf_counter() - t1
        return ys[:, 0] if one else ys
