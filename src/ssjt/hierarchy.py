"""Hierarchy representation, generators and grouped-to-tree conversion.

A :class:`Hierarchy` is a single-parent tree. Nodes are identified by string
names; every vector in the package is ordered as ``Hierarchy.nodes``.
Sparse matrices are used for S and U so that large synthetic trees fit in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class Hierarchy:
    nodes: List[str]
    parent: Dict[str, str]                 # child -> parent (root has no entry)
    children: Dict[str, List[str]]         # aggregate -> ordered children
    name: str = "hierarchy"
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ basics
    def __post_init__(self):
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        roots = [n for n in self.nodes if n not in self.parent]
        if len(roots) != 1:
            raise ValueError(f"hierarchy must have exactly one root, found {len(roots)}")
        self.root = roots[0]
        self.leaves = [n for n in self.nodes if n not in self.children]
        self.aggregates = [n for n in self.nodes if n in self.children]
        depth = {self.root: 0}
        order = [self.root]
        for n in order:                       # BFS
            for c in self.children.get(n, []):
                depth[c] = depth[n] + 1
                order.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("hierarchy is not connected")
        self.depth_of = depth
        self.pre_order = [n for n in order if n in self.children]      # root -> leaves
        self.post_order = self.pre_order[::-1]                         # leaves -> root

    @property
    def N(self) -> int:
        return len(self.nodes)

    @property
    def M(self) -> int:
        return len(self.leaves)

    @property
    def K(self) -> int:
        return len(self.aggregates)

    @property
    def depth(self) -> int:
        return max(self.depth_of.values())

    @property
    def branching(self) -> np.ndarray:
        return np.array([len(self.children[p]) for p in self.aggregates])

    def family(self, p: str) -> List[int]:
        return [self.idx[p]] + [self.idx[c] for c in self.children[p]]

    def level_indices(self) -> Dict[str, np.ndarray]:
        """Top / Interim / Leaf / All index sets (used for per-level metrics)."""
        leaf = np.array([self.idx[l] for l in self.leaves])
        top = np.array([self.idx[self.root]])
        interim = np.array([self.idx[a] for a in self.aggregates if a != self.root], dtype=int)
        return {"Top": top, "Interim": interim, "Leaf": leaf, "All": np.arange(self.N)}

    # --------------------------------------------------------------- matrices
    @property
    def S(self) -> sp.csr_matrix:
        """Summing matrix (N x M), sparse."""
        if "S" not in self._cache:
            lidx = {l: j for j, l in enumerate(self.leaves)}
            rows, cols = [], []
            for l in self.leaves:
                c = l
                while True:
                    rows.append(self.idx[c]); cols.append(lidx[l])
                    if c not in self.parent:
                        break
                    c = self.parent[c]
            self._cache["S"] = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.N, self.M))
        return self._cache["S"]

    @property
    def U(self) -> sp.csr_matrix:
        """Constraint matrix (K x N): row p is e_p - sum_{c in ch(p)} e_c, so U y = 0 iff coherent."""
        if "U" not in self._cache:
            rows, cols, vals = [], [], []
            for r, p in enumerate(self.aggregates):
                rows.append(r); cols.append(self.idx[p]); vals.append(1.0)
                for c in self.children[p]:
                    rows.append(r); cols.append(self.idx[c]); vals.append(-1.0)
            self._cache["U"] = sp.csr_matrix((vals, (rows, cols)), shape=(self.K, self.N))
        return self._cache["U"]

    @property
    def n_leaves_below(self) -> np.ndarray:
        return np.asarray(self.S.sum(axis=1)).ravel()

    def leaf_index(self) -> np.ndarray:
        return np.array([self.idx[l] for l in self.leaves])

    def aggregate_bottom_up(self, Y_leaves: np.ndarray) -> np.ndarray:
        """Y_leaves: (M,) or (M, H) -> full coherent (N,) or (N, H)."""
        return self.S @ Y_leaves

    def coherence_residual(self, y: np.ndarray) -> float:
        return float(np.max(np.abs(self.U @ y))) if self.K else 0.0

    def summary(self) -> dict:
        b = self.branching
        return {"name": self.name, "N": self.N, "M": self.M, "K": self.K, "depth": self.depth,
                "levels": self.depth + 1, "b_min": int(b.min()), "b_mean": float(b.mean()),
                "b_max": int(b.max())}

    # ------------------------------------------------------------- generators
    @classmethod
    def from_parent_map(cls, parent: Dict[str, str], nodes: Optional[List[str]] = None, name="hierarchy"):
        children: Dict[str, List[str]] = {}
        if nodes is None:
            s = set(parent) | set(parent.values())
            nodes = sorted(s)
        for n in nodes:
            if n in parent:
                children.setdefault(parent[n], []).append(n)
        return cls(list(nodes), dict(parent), children, name=name)

    @classmethod
    def balanced(cls, b: int, depth: int) -> "Hierarchy":
        """b-ary tree with `depth` edges on every root-to-leaf path (depth+1 levels)."""
        nodes, parent, cur = ["T"], {}, ["T"]
        for _ in range(depth):
            nxt = []
            for p in cur:
                for i in range(b):
                    c = f"{p}.{i}"
                    parent[c] = p; nodes.append(c); nxt.append(c)
            cur = nxt
        return cls.from_parent_map(parent, nodes, name=f"balanced(b={b},depth={depth})")

    @classmethod
    def star(cls, m: int) -> "Hierarchy":
        nodes = ["T"] + [f"T.{i}" for i in range(m)]
        return cls.from_parent_map({n: "T" for n in nodes[1:]}, nodes, name=f"star(m={m})")

    @classmethod
    def imbalanced(cls, b_max: int, d_max: int, rng: np.random.Generator, p_leaf: float = 0.3) -> "Hierarchy":
        """Random tree: each node below the root becomes a leaf with prob. p_leaf (always at d_max);
        otherwise it gets U{2..b_max} children. The root always has >= 2 children."""
        nodes, parent = ["T"], {}
        frontier = [("T", 0)]
        while frontier:
            p, d = frontier.pop(0)
            if d >= d_max or (p != "T" and rng.random() < p_leaf):
                continue
            k = int(rng.integers(2, b_max + 1))
            for i in range(k):
                c = f"{p}.{i}"
                parent[c] = p; nodes.append(c); frontier.append((c, d + 1))
        return cls.from_parent_map(parent, nodes, name=f"imbalanced(b_max={b_max},d_max={d_max})")


# ---------------------------------------------------------------------------
# Grouped structure -> single-parent tree (Section 6.3 of the article)
# ---------------------------------------------------------------------------
def tree_from_summing_matrix(S_df, name: str = "tree") -> "Hierarchy":
    """Convert a (possibly grouped) summing matrix, given as a pandas DataFrame with node
    names as index and bottom-series names as columns, into a single-parent tree (Section 6.3 of the article).

    Aggregates are ordered by the number of bottom series they aggregate (ties keep the input order);
    an aggregate is kept iff its footprint is disjoint from or nested in the footprint of every
    aggregate kept before it. Each kept aggregate is attached to the closest earlier kept aggregate
    whose footprint contains it, and each bottom series to the closest kept aggregate that contains it.
    Aggregates with identical footprints (and aggregates equal to a single bottom series) therefore
    form one-child chains instead of being dropped, and the result always has a single root.
    """
    A = S_df.values
    names = list(S_df.index)
    bottoms = list(S_df.columns)
    foot = {n: frozenset(np.flatnonzero(A[i] == 1.0)) for i, n in enumerate(names)}
    bottom_set = set(bottoms)
    order = {n: i for i, n in enumerate(names)}
    aggs = sorted([n for n in names if n not in bottom_set], key=lambda n: (-len(foot[n]), order[n]))
    kept: List[str] = []
    for n in aggs:
        f = foot[n]
        if all((not (f & foot[u])) or f <= foot[u] or foot[u] <= f for u in kept):
            kept.append(n)
    parent: Dict[str, str] = {}
    for i, n in enumerate(kept[1:], start=1):
        cands = [u for u in kept[:i] if foot[n] <= foot[u]]
        if not cands:
            raise ValueError(f"no parent for {n}: the structure has no common root")
        best = min(len(foot[u]) for u in cands)
        parent[n] = [u for u in cands if len(foot[u]) == best][-1]
    for j, bnode in enumerate(bottoms):
        cands = [u for u in kept if j in foot[u]]
        best = min(len(foot[u]) for u in cands)
        parent[bnode] = [u for u in cands if len(foot[u]) == best][-1]
    keep = set(kept) | bottom_set
    nodes = [n for n in names if n in keep]
    return Hierarchy.from_parent_map(parent, nodes, name=name)


def constraint_matrix_from_S(S: np.ndarray, is_bottom: np.ndarray) -> np.ndarray:
    """U for a general (grouped) structure: one row per aggregate node,
    e_i - sum_{j bottom in footprint(i)} e_j. U @ y = 0 iff y = S b for some b."""
    N = S.shape[0]
    bidx = np.flatnonzero(is_bottom)
    aidx = np.flatnonzero(~is_bottom)
    assert np.allclose(S[bidx], np.eye(len(bidx))), "bottom rows of S must be the identity in column order"
    U = np.zeros((len(aidx), N))
    for r, i in enumerate(aidx):
        U[r, i] = 1.0
        U[r, bidx] = -S[i, :]
    return U
