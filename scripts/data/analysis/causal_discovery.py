"""PC algorithm (Spirtes-Glymour-Scheines 2000) for constraint-based
causal-graph discovery on continuous data.

Pipeline:
  1. Skeleton phase. Start from the complete undirected graph. For every
     edge (i, j), search for a subset S of currently-adjacent nodes such
     that i and j become conditionally independent given S. If found,
     remove the edge and store `sepset(i, j) = S`.
  2. V-structure orientation. For every unshielded triple (i — k — j)
     whose missing edge (i, j) does NOT have k in its sepset, orient
     i → k ← j (collider).
  3. Meek propagation. Apply rules R1-R4 to extend orientations without
     creating new v-structures or directed cycles.

Conditional-independence test (continuous data):
  ρ(X, Y | Z) = -Ω[X, Y] / sqrt(Ω[X, X] · Ω[Y, Y])
  where Ω = inv(Σ) is the precision matrix of [X, Y, *Z]. Fisher's
  z-transform:  z = ½ ln((1+ρ)/(1-ρ))  satisfies
  z · sqrt(n - |Z| - 3)  →  N(0, 1)  under H0: ρ = 0.

Pure numpy + stdlib only.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


# ───────────────────────── conditional-independence ─────────────────────


def partial_corr(data: np.ndarray, i: int, j: int, condset: list[int]) -> float:
    """Partial correlation of columns ``i`` and ``j`` given ``condset``.

    Uses ``np.linalg.inv`` on the sub-covariance over the index list
    ``[i, j, *condset]``. Returns 0.0 on a singular covariance (treat the
    pair as independent — the safest fallback in CI testing).
    """
    cols = [i, j] + list(condset)
    sub = data[:, cols]
    # Sample covariance, unbiased (ddof=1).
    cov = np.cov(sub, rowvar=False, ddof=1)
    # When condset is empty, cov is 2×2; np.cov returns a 0-d array only
    # when there is a single variable, which never happens here.
    cov = np.atleast_2d(cov)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0
    denom = prec[0, 0] * prec[1, 1]
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    rho = -prec[0, 1] / math.sqrt(denom)
    # Clamp tiny numerical excursions outside [-1, 1].
    if rho > 1.0:
        rho = 1.0
    elif rho < -1.0:
        rho = -1.0
    return float(rho)


def fisher_z_pvalue(rho: float, n: int, k: int) -> float:
    """Two-sided p-value for H0: rho == 0.

    ``n`` is the sample size, ``k`` is the size of the conditioning set.
    For ``|rho| >= 1`` (degenerate) returns 1.0 — we can never reject on
    a perfect correlation that's actually a numerical artifact.
    """
    if not math.isfinite(rho) or abs(rho) >= 1.0:
        return 1.0
    df = n - k - 3
    if df <= 0:
        return 1.0
    z = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
    stat = abs(z) * math.sqrt(df)
    # Two-sided tail of standard normal: erfc(|stat| / sqrt(2)).
    return math.erfc(stat / math.sqrt(2.0))


# ────────────────────────────── skeleton ────────────────────────────────


def _adj_neighbors(adj: np.ndarray, node: int, exclude: int) -> list[int]:
    """Indices currently adjacent to ``node`` excluding ``exclude``."""
    n = adj.shape[0]
    return [k for k in range(n) if k != node and k != exclude and adj[node, k]]


def pc_skeleton(data: np.ndarray, alpha: float = 0.05,
                max_condset_size: int = 3) -> tuple[np.ndarray, dict]:
    """Return ``(adj, sepsets)``.

    ``adj`` is a symmetric boolean adjacency matrix (no self-loops).
    Starts as the complete graph. For ``level = 0, 1, …, max_condset_size``
    iterate over currently-adjacent pairs and, for each, try every subset
    of size ``level`` drawn from the current neighbours of either endpoint.
    Remove the edge and record the separating set (under both ``(i, j)``
    and ``(j, i)`` keys) on the first subset that yields ``p ≥ alpha``.

    Removals take effect immediately within a level (PC's standard
    ordered-by-pair processing). The outer loop exits early once no
    endpoint has enough remaining neighbours to form a subset of the
    next size.
    """
    data = np.asarray(data, dtype=np.float64)
    n_samples, n_vars = data.shape
    adj = np.ones((n_vars, n_vars), dtype=bool)
    np.fill_diagonal(adj, False)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}

    level = 0
    while level <= max_condset_size:
        progressed = False  # did any endpoint still have enough neighbours?
        # Snapshot current edge list so we iterate over a stable ordering.
        edges = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)
                 if adj[i, j]]
        for i, j in edges:
            if not adj[i, j]:
                continue  # already removed earlier in this level
            # Candidate conditioning pools (from either endpoint).
            for endpoint, other in ((i, j), (j, i)):
                neigh = _adj_neighbors(adj, endpoint, other)
                if len(neigh) < level:
                    continue
                progressed = True
                # Enumerate all subsets of exactly size ``level``.
                found = False
                for cond in itertools.combinations(neigh, level):
                    rho = partial_corr(data, i, j, list(cond))
                    p = fisher_z_pvalue(rho, n_samples, len(cond))
                    if p >= alpha:
                        adj[i, j] = False
                        adj[j, i] = False
                        sepsets[(i, j)] = tuple(cond)
                        sepsets[(j, i)] = tuple(cond)
                        found = True
                        break
                if found:
                    break
        if not progressed:
            break
        level += 1
    return adj, sepsets


# ─────────────────────────── orientation ────────────────────────────────


def orient_v_structures(adj: np.ndarray, sepsets: dict) -> np.ndarray:
    """Orient unshielded colliders.

    For every triple (i, k, j) with ``adj[i, k] = adj[k, j] = True`` and
    ``adj[i, j] = False`` — i.e., (i, j) is missing but both share k —
    if ``k not in sepsets[(i, j)]`` then orient ``i → k ← j``.

    Returns ``dirs`` where ``dirs[a, b] = 1`` means ``a → b`` is oriented.
    """
    n = adj.shape[0]
    dirs = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                continue  # need an unshielded pair
            sep = sepsets.get((i, j), sepsets.get((j, i)))
            if sep is None:
                continue
            sep_set = set(sep)
            for k in range(n):
                if k == i or k == j:
                    continue
                if adj[i, k] and adj[k, j] and k not in sep_set:
                    dirs[i, k] = 1
                    dirs[j, k] = 1
    return dirs


def _is_undirected(adj: np.ndarray, dirs: np.ndarray, a: int, b: int) -> bool:
    """True iff edge ``a — b`` exists in the skeleton with no orientation."""
    return bool(adj[a, b]) and dirs[a, b] == 0 and dirs[b, a] == 0


def _has_directed(dirs: np.ndarray, a: int, b: int) -> bool:
    return dirs[a, b] == 1


def _adjacent(adj: np.ndarray, dirs: np.ndarray, a: int, b: int) -> bool:
    """Any kind of edge (directed either way OR undirected) between a, b."""
    return bool(adj[a, b]) or dirs[a, b] == 1 or dirs[b, a] == 1


def meek_rules(adj: np.ndarray, dirs: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Propagate orientations under Meek's rules R1-R4.

    R1: i → j and j — k with i and k NOT adjacent  ⇒  orient j → k
        (avoid creating a new v-structure).
    R2: i → k → j and i — j                        ⇒  orient i → j
        (avoid a directed cycle once we close the triangle).
    R3: i — j and there exist k, l such that k → j, l → j,
        i — k, i — l, and k not adjacent to l       ⇒  orient i → j.
    R4: i — j and a chain i — k → l → j with i and l adjacent and
        k and j NOT adjacent                        ⇒  orient i → j.

    Mutates ``dirs`` in place and returns it.
    """
    n = adj.shape[0]
    for _ in range(max_iter):
        changed = False
        # R1
        for i in range(n):
            for j in range(n):
                if not _has_directed(dirs, i, j):
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if _is_undirected(adj, dirs, j, k) and not _adjacent(adj, dirs, i, k):
                        dirs[j, k] = 1
                        changed = True
        # R2
        for i in range(n):
            for j in range(n):
                if i == j or not _is_undirected(adj, dirs, i, j):
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if _has_directed(dirs, i, k) and _has_directed(dirs, k, j):
                        dirs[i, j] = 1
                        changed = True
                        break
        # R3
        for i in range(n):
            for j in range(n):
                if i == j or not _is_undirected(adj, dirs, i, j):
                    continue
                # find two parents k, l of j that are both undirected-adjacent
                # to i and non-adjacent to each other.
                parents = [p for p in range(n)
                           if p != i and p != j and _has_directed(dirs, p, j)
                           and _is_undirected(adj, dirs, i, p)]
                done = False
                for a in range(len(parents)):
                    for b in range(a + 1, len(parents)):
                        k, l = parents[a], parents[b]
                        if not _adjacent(adj, dirs, k, l):
                            dirs[i, j] = 1
                            changed = True
                            done = True
                            break
                    if done:
                        break
        # R4
        for i in range(n):
            for j in range(n):
                if i == j or not _is_undirected(adj, dirs, i, j):
                    continue
                done = False
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if not _is_undirected(adj, dirs, i, k):
                        continue
                    for l in range(n):
                        if l in (i, j, k):
                            continue
                        if (_has_directed(dirs, k, l) and _has_directed(dirs, l, j)
                                and _adjacent(adj, dirs, i, l)
                                and not _adjacent(adj, dirs, k, j)):
                            dirs[i, j] = 1
                            changed = True
                            done = True
                            break
                    if done:
                        break
        if not changed:
            break
    return dirs


# ──────────────────────────── full pipeline ─────────────────────────────


def pc_algorithm(data: np.ndarray, names: list[str],
                 alpha: float = 0.05, max_condset_size: int = 3) -> dict:
    """Run the full PC pipeline and return a structured result.

    Output keys:
      - ``skeleton``:        symmetric bool adjacency matrix
      - ``sepsets``:         dict[(i, j) -> tuple[int, ...]]
      - ``v_structures``:    list of (i_name, k_name, j_name) colliders
      - ``directed_edges``:  list of (src_name, dst_name)
      - ``undirected_edges``: list of (name_a, name_b) with a < b
    """
    data = np.asarray(data, dtype=np.float64)
    n_vars = data.shape[1]
    if len(names) != n_vars:
        raise ValueError("len(names) must match data.shape[1]")

    adj, sepsets = pc_skeleton(data, alpha=alpha,
                               max_condset_size=max_condset_size)
    dirs = orient_v_structures(adj, sepsets)

    # Record the v-structures BEFORE Meek expands the orientation set.
    v_structures: list[tuple[str, str, str]] = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if adj[i, j]:
                continue
            sep = sepsets.get((i, j), sepsets.get((j, i)))
            if sep is None:
                continue
            sep_set = set(sep)
            for k in range(n_vars):
                if k in (i, j):
                    continue
                if adj[i, k] and adj[k, j] and k not in sep_set:
                    v_structures.append((names[i], names[k], names[j]))

    meek_rules(adj, dirs)

    directed_edges: list[tuple[str, str]] = []
    undirected_edges: list[tuple[str, str]] = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            if dirs[i, j] == 1 and dirs[j, i] == 0:
                directed_edges.append((names[i], names[j]))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if adj[i, j] and dirs[i, j] == 0 and dirs[j, i] == 0:
                undirected_edges.append((names[i], names[j]))

    return {
        "skeleton": adj,
        "sepsets": sepsets,
        "v_structures": v_structures,
        "directed_edges": directed_edges,
        "undirected_edges": undirected_edges,
    }
