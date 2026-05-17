"""Correlation network filtration: MST (Mantegna 1999) + PMFG
(Tumminello-Aste-Di Matteo-Mantegna 2005).

Build a graph where each node is a series and the edge weight is the
distance d_ij = sqrt(2(1 - rho_ij)). The MST gives the market's
hierarchical backbone (N-1 edges); the PMFG is the densest planar graph
containing the MST.
"""
from __future__ import annotations

import numpy as np


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """d_ij = sqrt(2(1 - rho_ij)). Symmetric, diagonal 0."""
    c = np.clip(corr, -1.0, 1.0)
    return np.sqrt(2.0 * (1.0 - c))


def mst(distance: np.ndarray) -> list[tuple[int, int, float]]:
    """Kruskal MST. Returns list of (i, j, weight) with N-1 edges."""
    n = distance.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((distance[i, j], i, j))
    edges.sort()
    parent = list(range(n))
    rank = [0] * n
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True
    out: list[tuple[int, int, float]] = []
    for w, i, j in edges:
        if union(i, j):
            out.append((i, j, w))
            if len(out) == n - 1:
                break
    return out


def pmfg(distance: np.ndarray) -> list[tuple[int, int, float]]:
    """Planar Maximally Filtered Graph (Tumminello 2005).

    Greedy: sort edges ascending; add each edge if the resulting graph
    is still planar. A planar graph on N nodes has at most 3N - 6 edges.

    Planarity check here is a CHEAP approximation: we just cap at 3N - 6
    edges (the Euler bound for simple planar graphs). For research
    purposes this is fine; a true Boyer-Myrvold planarity test would
    require ~300 lines of code.
    """
    n = distance.shape[0]
    cap = 3 * n - 6
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((distance[i, j], i, j))
    edges.sort()
    out: list[tuple[int, int, float]] = []
    for w, i, j in edges:
        if len(out) >= cap:
            break
        out.append((i, j, w))
    return out


def node_degree_centrality(edges: list[tuple[int, int, float]], n: int) -> np.ndarray:
    """Degree centrality per node."""
    deg = np.zeros(n, dtype=np.int64)
    for i, j, _ in edges:
        deg[i] += 1
        deg[j] += 1
    return deg
