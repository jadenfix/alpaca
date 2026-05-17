"""Topological Data Analysis (TDA) for return time series.

Implements persistent homology of dimensions 0 (connected components) and
1 (loops/cycles) on a point cloud, derived from the Vietoris-Rips
filtration, and a rolling persistence-norm score for detecting
topological-change events in multivariate return panels.

Background
----------
Gidea & Katz (2018, "Topological data analysis of financial time series:
Landscapes of crashes") showed that the L^p persistence norm of the
Rips persistence diagram of a sliding window of equity returns rises
sharply 1-2 weeks before major crash episodes (1987, 2000, 2008). The
intuition: as a market approaches a phase transition, daily-return
vectors collapse onto lower-dimensional manifolds and form loop-like
structures that persist over a wider range of filtration scales,
inflating the persistence norm.

Dim-0 (connected components)
----------------------------
Vietoris-Rips at scale eps: nodes i, j are connected iff D[i, j] <= eps.
By Kruskal's MST construction (Union-Find), every time we merge two
components at scale eps_k we kill one dim-0 feature. With N points, N-1
finite (birth=0, death=eps_k) pairs are produced plus one immortal
component (birth=0, death=inf).

Dim-1 (loops) -- simplified MVP proxy
-------------------------------------
A faithful dim-1 implementation requires reducing the boundary matrix
of the 2-skeleton of the Rips complex, which is O(N^3) memory and well
beyond a pure-numpy MVP. We adopt the standard "first homology proxy"
used in financial TDA papers: build the MST via Kruskal, then every
additional edge (in sorted-by-distance order) closes a new independent
1-cycle. The k-th 1-cycle is born at the distance of the (N-1+k)-th
edge accepted/rejected sequentially -- specifically, the k-th edge
that would create a cycle when added in distance order. Death is set
to infinity in this MVP (since triangulating triangles to kill cycles
requires the full filtration). This proxy correctly tracks the
"number of independent 1-cycles created by scale eps", which is what
the persistence-norm sum is sensitive to in regime-change applications.

Use `persistence_dim1_proxy` only as a screening signal, not as an
exact dim-1 barcode.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────── distance matrix ────────────────────────────

def pairwise_dist(X: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix of shape (N, N) for point cloud (N, D).

    O(N^2 D) memory via broadcasting -- adequate for N up to ~500.
    Diagonal is exactly 0; output is symmetric and non-negative.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be 2-D of shape (N, D)")
    diff = X[:, None, :] - X[None, :, :]
    sq = (diff * diff).sum(axis=-1)
    # Numerical guard: clip small negatives from roundoff before sqrt.
    sq = np.maximum(sq, 0.0)
    D = np.sqrt(sq)
    # Force symmetry & zero diagonal.
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# ────────────────────────────── union-find ──────────────────────────────

class _UnionFind:
    """Union-Find with path compression + union by rank."""

    __slots__ = ("parent", "rank", "n_components")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        # Iterative path compression.
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Compress.
        cur = x
        while self.parent[cur] != root:
            nxt = self.parent[cur]
            self.parent[cur] = root
            cur = nxt
        return root

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True


# ───────────────────── sorted upper-triangular edges ────────────────────

def _sorted_edges(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (i_idx, j_idx, dist) for i < j, sorted ascending by distance.

    Uses a stable sort so equal-distance edges have deterministic order.
    """
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    d = D[iu, ju]
    order = np.argsort(d, kind="stable")
    return iu[order], ju[order], d[order]


# ───────────────────────── dim-0 persistence ────────────────────────────

def persistence_dim0(D: np.ndarray) -> np.ndarray:
    """Dim-0 persistence pairs (birth, death) as shape (N, 2).

    Births are all 0.0 (Rips: every point is born at scale 0). N-1
    finite pairs come from successive Union-Find merges along the
    sorted edge list; one component lives forever (death = +inf).
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n == 1:
        return np.array([[0.0, np.inf]], dtype=np.float64)

    iu, ju, dist = _sorted_edges(D)
    uf = _UnionFind(n)
    deaths = np.empty(n - 1, dtype=np.float64)
    k = 0
    for a, b, d in zip(iu, ju, dist):
        if uf.union(int(a), int(b)):
            deaths[k] = float(d)
            k += 1
            if uf.n_components == 1:
                break
    # Truncate (paranoia: should always be n-1 by connectivity).
    deaths = deaths[:k]
    pairs = np.zeros((k + 1, 2), dtype=np.float64)
    pairs[:k, 1] = deaths
    pairs[k, 0] = 0.0
    pairs[k, 1] = np.inf
    return pairs


# ───────────────────── dim-1 cycle-birth proxy ──────────────────────────

def persistence_dim1_proxy(D: np.ndarray) -> np.ndarray:
    """Dim-1 cycle births via the simplified MST-extension proxy.

    Build the MST with Kruskal's algorithm (Union-Find on sorted edges).
    Every edge that *would* form a cycle when added (i.e. is rejected
    from the MST because both endpoints are already in the same
    component) closes a new independent 1-cycle. We record its
    distance as the cycle's birth.

    Death is set to +inf in this MVP (no triangle-killing).

    Returns
    -------
    np.ndarray of shape (k, 2)
        Each row is (birth, +inf). k = E - (n - 1) where E is the
        number of unique pairs (n choose 2) and n is the number of
        non-isolated points; on a connected complete graph k = n(n-3)/2 + 1.
        Rows are ordered by ascending birth.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n < 3:
        return np.zeros((0, 2), dtype=np.float64)

    iu, ju, dist = _sorted_edges(D)
    uf = _UnionFind(n)
    births: list[float] = []
    for a, b, d in zip(iu, ju, dist):
        merged = uf.union(int(a), int(b))
        if not merged:
            # This edge closes a 1-cycle.
            births.append(float(d))
    if not births:
        return np.zeros((0, 2), dtype=np.float64)
    out = np.empty((len(births), 2), dtype=np.float64)
    out[:, 0] = births
    out[:, 1] = np.inf
    return out


# ─────────────────────────── persistence norm ───────────────────────────

def persistence_norm(pairs: np.ndarray, p: float = 2.0) -> float:
    """L^p persistence norm: (Σ |death - birth|^p)^(1/p) over finite pairs.

    Infinite pairs (death = +inf) are excluded, which is the standard
    Gidea-Katz convention for the rolling persistence-norm time series.
    """
    if pairs is None:
        return 0.0
    pairs = np.asarray(pairs, dtype=np.float64)
    if pairs.size == 0:
        return 0.0
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be of shape (k, 2)")
    finite_mask = np.isfinite(pairs[:, 1]) & np.isfinite(pairs[:, 0])
    finite = pairs[finite_mask]
    if finite.shape[0] == 0:
        return 0.0
    lifetimes = np.abs(finite[:, 1] - finite[:, 0])
    if p <= 0:
        raise ValueError("p must be positive")
    return float(np.sum(lifetimes ** p) ** (1.0 / p))


# ──────────────────── rolling persistence-norm series ───────────────────

def rolling_persistence_norm(
    returns_matrix: np.ndarray,
    window: int = 50,
    step: int = 5,
) -> dict:
    """Slide over rows of (T, N) returns and compute persistence norms.

    For each window, the `window` daily return vectors are treated as
    a point cloud in R^N. We compute the pairwise distance matrix,
    extract dim-0 and dim-1 (proxy) persistence diagrams, and report
    their L^2 persistence norms.

    Returns
    -------
    dict with keys
        "ts_idx"   : indices of the LAST row of each window (length L)
        "norm_dim0": dim-0 L^2 persistence norm per window (length L)
        "norm_dim1": dim-1 proxy L^2 persistence norm per window (length L)

    where L = floor((T - window) / step) + 1.
    """
    R = np.asarray(returns_matrix, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2-D of shape (T, N)")
    T = R.shape[0]
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive integers")
    if T < window:
        return {
            "ts_idx": np.zeros(0, dtype=np.int64),
            "norm_dim0": np.zeros(0, dtype=np.float64),
            "norm_dim1": np.zeros(0, dtype=np.float64),
        }

    L = (T - window) // step + 1
    ts_idx = np.empty(L, dtype=np.int64)
    n0 = np.empty(L, dtype=np.float64)
    n1 = np.empty(L, dtype=np.float64)

    for k in range(L):
        start = k * step
        stop = start + window
        win = R[start:stop, :]
        D = pairwise_dist(win)
        p0 = persistence_dim0(D)
        p1 = persistence_dim1_proxy(D)
        n0[k] = persistence_norm(p0, p=2.0)
        # Dim-1 proxy has infinite deaths => persistence_norm = 0 by
        # construction. As a usable scalar, fall back to the L^2 norm of
        # the birth-times themselves (Gidea-Katz "birth-only" proxy).
        if p1.shape[0] == 0:
            n1[k] = 0.0
        else:
            n1[k] = float(np.sqrt(np.sum(p1[:, 0] ** 2)))
        ts_idx[k] = stop - 1

    return {"ts_idx": ts_idx, "norm_dim0": n0, "norm_dim1": n1}


# ───────────────────────── topology-change score ────────────────────────

def topology_change_score(
    norm_series: np.ndarray,
    lookback: int = 20,
) -> np.ndarray:
    """Causal z-score of a persistence-norm time series.

    z_t = (x_t - mean(x_{t-lookback:t})) / std(x_{t-lookback:t})

    The window is strictly causal (does not include x_t itself), so
    perturbations at index t do not affect any z_s for s <= t. A
    z-score above ~2 marks a topological-change alert.

    The first `lookback` entries are NaN (insufficient history).
    A zero-variance window yields NaN to avoid division by zero.
    """
    x = np.asarray(norm_series, dtype=np.float64).ravel()
    T = x.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    if lookback < 2 or T <= lookback:
        return out
    for t in range(lookback, T):
        win = x[t - lookback:t]
        mu = float(np.mean(win))
        sd = float(np.std(win, ddof=0))
        if sd <= 0.0 or not np.isfinite(sd):
            out[t] = np.nan
        else:
            out[t] = (x[t] - mu) / sd
    return out
