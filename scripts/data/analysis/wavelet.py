"""Continuous Wavelet Transform (CWT) with Morlet wavelet + cross-wavelet
coherence.

CWT decomposes a signal into time-frequency space; wavelet coherence
measures local correlation between two signals at each (time, scale).
Catches dependencies that exist at specific horizons (e.g., correlated
on weekly cycle but independent at daily resolution).

We implement the FFT-based CWT (Torrence & Compo 1998).
"""
from __future__ import annotations

import numpy as np


def morlet_cwt(x: np.ndarray, scales: np.ndarray, w0: float = 6.0) -> np.ndarray:
    """Continuous Morlet Wavelet Transform via FFT.

    Returns complex CWT coefficients of shape (n_scales, n_samples).
    """
    n = x.size
    x_mean = x - x.mean()
    # Pad to next power of 2 for FFT speed.
    n_pad = 1 << int(np.ceil(np.log2(n)))
    x_pad = np.zeros(n_pad)
    x_pad[:n] = x_mean
    fft_x = np.fft.fft(x_pad)
    omega = 2.0 * np.pi * np.fft.fftfreq(n_pad)
    out = np.zeros((scales.size, n), dtype=np.complex128)
    for i, s in enumerate(scales):
        # Fourier transform of the Morlet wavelet at scale s.
        norm = np.sqrt(2.0 * np.pi * s)
        arg = s * omega - w0
        psi_hat = norm * np.pi ** -0.25 * np.exp(-0.5 * arg ** 2) * (omega > 0)
        wave = np.fft.ifft(fft_x * psi_hat)
        out[i, :] = wave[:n]
    return out


def wavelet_coherence(x: np.ndarray, y: np.ndarray, scales: np.ndarray | None = None,
                      smoothing: int = 5) -> dict:
    """Wavelet coherence R² (t, s) between x and y at each (time, scale).

    R²(t, s) = |smooth(W_xy)|² / (smooth(|W_x|²) · smooth(|W_y|²))

    Returns dict {scales, coherence (S, T), mean_coherence_per_scale}.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if scales is None:
        # Log-spaced scales from ~2 (Nyquist) to ~n/4.
        s_max = max(8.0, n / 4.0)
        scales = np.geomspace(2.0, s_max, num=12)
    wx = morlet_cwt(x, scales)
    wy = morlet_cwt(y, scales)
    wxy = wx * np.conj(wy)
    # Smooth along time with a scale-proportional boxcar (Grinsted et al.
    # 2004). The intrinsic Morlet decorrelation length is ~s, so a fixed
    # window severely under-smooths large scales and inflates coherence on
    # independent inputs.
    def smooth_time(arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr)
        for i, s in enumerate(scales):
            k = max(int(round(s)), smoothing, 1)
            k = min(k, arr.shape[1])
            if k <= 1:
                out[i, :] = arr[i, :]
                continue
            kernel = np.ones(k) / k
            out[i, :] = np.convolve(arr[i, :], kernel, mode="same")
        return out
    s_wxy = smooth_time(wxy)
    s_wxx = smooth_time(np.abs(wx) ** 2)
    s_wyy = smooth_time(np.abs(wy) ** 2)
    denom = s_wxx * s_wyy
    # Cauchy-Schwarz says |s_wxy|² ≤ s_wxx · s_wyy. Where the energy density
    # is degenerate (no power at this scale), the ratio is undefined; we
    # leave such regions out via NaN so the mean does not get a spurious
    # boost from clamped denominators.
    safe = np.where(denom > 0, denom, 1.0)
    coh = np.where(denom > 0, np.abs(s_wxy) ** 2 / safe, np.nan)
    coh = np.clip(coh, 0.0, 1.0)
    return {
        "scales": scales,
        "coherence": coh,
        "mean_coherence_per_scale": np.nanmean(coh, axis=1),
    }
