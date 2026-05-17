"""Diffusion maps — Coifman & Lafon (2006).

Build a random-walk kernel K_ε(i, j) = exp(-||x_i - x_j||² / ε), row-
normalize to a transition matrix P, and the top non-trivial eigenvectors
of P^t form a Euclidean embedding such that diffusion distance in the
embedding equals geodesic distance on the underlying manifold.

PCA is the linear special case; diffusion maps generalize to nonlinear
manifolds (e.g., regime curves that bend through feature space).
"""
from __future__ import annotations

import numpy as np


def diffusion_map(data: np.ndarray, n_components: int = 2,
                  epsilon: float | None = None, t_diffusion: int = 1) -> dict:
    """Compute a diffusion-map embedding of the COLUMNS of `data` (T, N).

    Args:
        data:         (T, N) array. Each column is one feature/series.
        n_components: number of embedding dimensions (default 2).
        epsilon:      kernel bandwidth. If None, set to median pairwise distance.
        t_diffusion:  diffusion time (P^t). Larger = coarser embedding.

    Returns dict with:
        embedding:    (N, n_components) — the manifold coordinates of each column.
        eigenvalues:  the leading eigenvalues (including the trivial 1).
    """
    # Treat each COLUMN of `data` as one point in T-dimensional space
    # (one "feature vector" per series).
    points = data.T  # shape (N, T)
    n = points.shape[0]
    # Pairwise squared Euclidean distances.
    sq_dists = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    if epsilon is None:
        # Median of non-zero distances as the kernel bandwidth.
        offdiag = sq_dists[~np.eye(n, dtype=bool)]
        epsilon = float(np.median(offdiag)) if offdiag.size > 0 else 1.0
        if epsilon <= 0:
            epsilon = 1.0
    K = np.exp(-sq_dists / max(epsilon, 1e-12))
    # Row-normalize: P_ij = K_ij / sum_k K_ik
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    P = K / row_sums
    # Eigendecompose P (non-symmetric). Use symmetric trick: P = D^-1 K,
    # related to symmetric K~ = D^-1/2 K D^-1/2 whose eigenvalues match P's.
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(row_sums.flatten(), 1e-12))
    K_sym = K * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    # Symmetric eigendecomposition.
    eigvals, eigvecs_sym = np.linalg.eigh(K_sym)
    # Sort descending.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs_sym = eigvecs_sym[:, order]
    # Back to non-symmetric eigenvectors of P.
    eigvecs = eigvecs_sym * d_inv_sqrt[:, None]
    # Drop the trivial eigenvector (eigval ≈ 1, constant direction).
    nontrivial = eigvals[1:n_components + 1]
    nontriv_vecs = eigvecs[:, 1:n_components + 1]
    # Diffusion-time scaling: multiply each coordinate by lambda^t.
    embedding = nontriv_vecs * (nontrivial ** t_diffusion)[None, :]
    return {"embedding": embedding, "eigenvalues": eigvals}
