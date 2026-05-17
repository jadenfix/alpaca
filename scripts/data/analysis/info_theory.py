"""Information-theoretic measures: MI, KL, Jensen-Shannon, Rényi, Transfer
Entropy. All quantities in NATS (natural log).

Each catches a different invariance class:
  - MI(X;Y) = H(X) + H(Y) - H(X,Y): invariant under any invertible
    transformation of X or Y. Zero iff independent.
  - KL(P||Q): non-symmetric divergence; +inf where Q=0 but P>0.
  - JS = 0.5 KL(P||M) + 0.5 KL(Q||M), M=0.5(P+Q): symmetric, bounded by ln 2.
  - Rényi H_alpha = (1/(1-alpha)) log sum(p^alpha): generalizes Shannon (alpha=1).
  - Transfer Entropy: directed information flow Y_{t+1} given Y's past plus
    X's past.
"""
from __future__ import annotations

import numpy as np


def _freedman_diaconis_bins(x: np.ndarray) -> int:
    """Freedman-Diaconis rule for histogram bin count."""
    n = x.size
    if n < 4:
        return max(2, n // 2)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return max(2, int(np.sqrt(n)))
    bin_width = 2.0 * iqr / n ** (1.0 / 3.0)
    if bin_width <= 0:
        return max(2, int(np.sqrt(n)))
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return max(2, min(bins, n // 3))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int | None = None) -> float:
    """MI(X; Y) in nats via 2-D histogram (plug-in estimator)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape, "MI: shape mismatch"
    if bins is None:
        bins = _freedman_diaconis_bins(x)
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist / hist.sum() if hist.sum() > 0 else hist
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_xy / (p_x * p_y)
        log_ratio = np.where(p_xy > 0, np.log(np.where(ratio > 0, ratio, 1.0)), 0.0)
    return float(np.sum(p_xy * log_ratio))


def entropy(x: np.ndarray, bins: int | None = None) -> float:
    """Shannon entropy H(X) in nats via histogram."""
    if bins is None:
        bins = _freedman_diaconis_bins(x)
    hist, _ = np.histogram(x, bins=bins)
    p = hist / hist.sum() if hist.sum() > 0 else hist
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) in nats. p, q must be probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    assert p.shape == q.shape
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Jensen-Shannon divergence (nats). Bounded above by ln 2."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def renyi_entropy(p: np.ndarray, alpha: float) -> float:
    """Rényi entropy H_alpha in nats. alpha = 1 reduces to Shannon."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-9:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p ** alpha)) / (1.0 - alpha))


def transfer_entropy(src: np.ndarray, dst: np.ndarray, k: int = 1, l: int = 1,
                     bins: int = 4) -> float:
    """Transfer Entropy T_{X->Y} in nats via binned plug-in.

    See crates/features/src/transfer_entropy.rs for derivation. Mirrors that
    implementation for cross-verification.
    """
    assert src.shape == dst.shape
    n = src.size
    history = max(k, l)
    if n < history + 2:
        return 0.0
    # Quantile-bin each series.
    src_bins = _quantile_bin(src, bins)
    dst_bins = _quantile_bin(dst, bins)
    base = bins
    stride = base ** (history + 1)
    joint: dict[int, float] = {}
    p_yk_xl: dict[int, float] = {}
    p_y1_yk: dict[int, float] = {}
    p_yk: dict[int, float] = {}
    total = 0.0
    for t in range(history, n - 1):
        y_next = int(dst_bins[t + 1])
        y_hist = _encode(dst_bins, t, k, base)
        x_hist = _encode(src_bins, t, l, base)
        joint_key = y_next * stride * stride + y_hist * stride + x_hist
        yk_xl_key = y_hist * stride + x_hist
        y1_yk_key = y_next * stride + y_hist
        yk_key = y_hist
        joint[joint_key] = joint.get(joint_key, 0.0) + 1.0
        p_yk_xl[yk_xl_key] = p_yk_xl.get(yk_xl_key, 0.0) + 1.0
        p_y1_yk[y1_yk_key] = p_y1_yk.get(y1_yk_key, 0.0) + 1.0
        p_yk[yk_key] = p_yk.get(yk_key, 0.0) + 1.0
        total += 1.0
    if total <= 0:
        return 0.0
    te = 0.0
    for jk, cnt in joint.items():
        p_joint = cnt / total
        y_next = jk // (stride * stride)
        y_hist = (jk % (stride * stride)) // stride
        x_hist = jk % stride
        a = p_yk_xl.get(y_hist * stride + x_hist, 0.0) / total
        b = p_y1_yk.get(y_next * stride + y_hist, 0.0) / total
        c = p_yk.get(y_hist, 0.0) / total
        if a <= 0 or b <= 0 or c <= 0:
            continue
        ratio = (p_joint * c) / (a * b)
        if ratio > 0:
            te += p_joint * np.log(ratio)
    return max(0.0, te)


def _quantile_bin(xs: np.ndarray, bins: int) -> np.ndarray:
    n = xs.size
    idx = np.argsort(xs, kind="stable")
    out = np.zeros(n, dtype=np.int64)
    for rank, i in enumerate(idx):
        out[i] = min(bins - 1, (rank * bins) // n)
    return out


def _encode(arr: np.ndarray, t: int, k: int, base: int) -> int:
    acc = 0
    for off in range(k):
        acc = acc * base + int(arr[t - off])
    return acc
