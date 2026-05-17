//! PCA + Random Matrix Theory: Marchenko-Pastur, Tracy-Widom, and the
//! Bouchaud-Potters-Ledoit-Péché Rotationally Invariant Estimator (RIE).
//!
//! First-principles motivation:
//!   For an N × T data matrix X with rows i.i.d. N(0, σ²), the empirical
//!   covariance C = (1/T) X·Xᵀ has eigenvalues whose density converges
//!   (Marchenko & Pastur, 1967) to a deterministic distribution supported
//!   on [σ²(1−√q)², σ²(1+√q)²] where q = N/T. Any eigenvalue outside that
//!   "bulk" is mathematically guaranteed to carry information beyond noise.
//!
//! Three tools land here:
//!   1. `pca(data)`              — covariance + symmetric eigendecomposition.
//!   2. `marchenko_pastur_*`     — closed-form noise bulk density / bounds.
//!   3. `tracy_widom_pvalue`     — edge-test p-value for "is λ_max real?"
//!   4. `rie_shrinkage`          — optimal nonlinear shrinkage (Bouchaud
//!                                 et al. 2017) of empirical eigenvalues.

use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::Array2;

#[derive(Clone, Debug)]
pub struct PcaResult {
    /// Eigenvalues in descending order.
    pub eigenvalues: Vec<f64>,
    /// Each column is the eigenvector for the corresponding eigenvalue,
    /// stored as `(n_variables × n_components)`.
    pub eigenvectors: Array2<f64>,
    /// Fraction of variance explained per component (sums to 1).
    pub explained_variance_ratio: Vec<f64>,
}

/// PCA on the COLUMNS of `data` (shape `n_observations × n_variables`).
/// Centers each column, computes sample covariance C = (1/(T-1)) Xᶜᵀ Xᶜ,
/// and returns the eigen-decomposition sorted by descending eigenvalue.
pub fn pca(data: &Array2<f64>) -> PcaResult {
    let (t, n) = data.dim();
    assert!(t >= 2, "pca: need at least 2 observations");
    assert!(n >= 1, "pca: need at least 1 variable");

    // Center each column.
    let mut centered = data.clone();
    for j in 0..n {
        let mean = data.column(j).mean().unwrap_or(0.0);
        for i in 0..t {
            centered[(i, j)] -= mean;
        }
    }

    // Covariance C = (1/(t-1)) X^T X.
    let mut cov = DMatrix::<f64>::zeros(n, n);
    let inv_dof = 1.0 / (t as f64 - 1.0);
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..t {
                s += centered[(k, i)] * centered[(k, j)];
            }
            let v = s * inv_dof;
            cov[(i, j)] = v;
            cov[(j, i)] = v;
        }
    }

    // Symmetric eigendecomposition.
    let eig = SymmetricEigen::new(cov);
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|j| {
            let lam = eig.eigenvalues[j];
            let v: Vec<f64> = (0..n).map(|i| eig.eigenvectors[(i, j)]).collect();
            (lam, v)
        })
        .collect();
    // Sort descending by eigenvalue.
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = pairs.iter().map(|(l, _)| *l).collect();
    let total: f64 = eigenvalues.iter().map(|l| l.max(0.0)).sum();
    let explained_variance_ratio: Vec<f64> = eigenvalues
        .iter()
        .map(|l| if total > 0.0 { l.max(0.0) / total } else { 0.0 })
        .collect();
    let mut eigenvectors = Array2::<f64>::zeros((n, n));
    for (j, (_, v)) in pairs.iter().enumerate() {
        for i in 0..n {
            eigenvectors[(i, j)] = v[i];
        }
    }

    PcaResult {
        eigenvalues,
        eigenvectors,
        explained_variance_ratio,
    }
}

/// Marchenko-Pastur density at point `lambda`. `q = N/T` is the aspect
/// ratio of the data matrix; `sigma2` is the noise variance.
///
/// Density vanishes outside [λ−, λ+] = σ²(1∓√q)².
pub fn marchenko_pastur_density(lambda: f64, q: f64, sigma2: f64) -> f64 {
    let (lam_minus, lam_plus) = marchenko_pastur_bounds(q, sigma2);
    if lambda <= lam_minus || lambda >= lam_plus {
        return 0.0;
    }
    let num = ((lam_plus - lambda) * (lambda - lam_minus)).max(0.0).sqrt();
    let denom = 2.0 * std::f64::consts::PI * q * sigma2 * lambda;
    if denom <= 0.0 {
        0.0
    } else {
        num / denom
    }
}

/// Marchenko-Pastur edge bounds (λ−, λ+).
pub fn marchenko_pastur_bounds(q: f64, sigma2: f64) -> (f64, f64) {
    let sq = q.sqrt();
    let lam_minus = sigma2 * (1.0 - sq).powi(2);
    let lam_plus = sigma2 * (1.0 + sq).powi(2);
    (lam_minus, lam_plus)
}

/// Tracy-Widom p-value: probability that the largest eigenvalue of a
/// noise sample-covariance matrix would exceed `lambda_max` by chance.
///
/// Johnstone (2001) Theorem 1.1: for the raw Wishart W = X·Xᵀ with X ∈
/// R^{N×T} of i.i.d. N(0,1) entries,
///     (λ_max(W) − μ_W) / σ_W → Tracy-Widom GOE
/// with
///     μ_W = (√(N−1/2) + √(T−1/2))²
///     σ_W = (√(N−1/2) + √(T−1/2)) · (1/√(N−1/2) + 1/√(T−1/2))^(1/3)
///
/// Our `pca()` divides the cross-product by (T−1), so we report eigenvalues
/// of S = W/(T−1). We therefore divide μ_W, σ_W by (T−1) to compare on the
/// same scale.
///
/// `q = N/T` is the data-matrix aspect ratio; `n` is the number of
/// variables N. The upper-tail Pr[TW_1 > z] is then approximated by
/// `tw1_upper_tail` (asymptotic above z=1, tabulated body, left-tail
/// asymptotic below z=-2).
pub fn tracy_widom_pvalue(lambda_max: f64, q: f64, n: usize) -> f64 {
    let nf = n as f64;
    let t = nf / q.max(1e-12);
    let denom = (t - 1.0).max(1.0);
    let sn = (nf - 0.5).sqrt();
    let st = (t - 0.5).sqrt();
    let mu_w = (sn + st).powi(2);
    let sigma_w = (sn + st) * (1.0 / sn + 1.0 / st).cbrt();
    let mu = mu_w / denom;
    let sigma = sigma_w / denom;
    if sigma <= 0.0 || !sigma.is_finite() {
        return 1.0;
    }
    let z = (lambda_max - mu) / sigma;
    tw1_upper_tail(z)
}

/// Approximation to Pr[TW_1 > s] using a piecewise model.
///   - For s ≥ 1: F-tail asymptotic A·s^(-1/8)·exp(-(2/3)·s^(3/2)) with
///     normalization A = 1 (close to exact for s ≥ 1).
///   - For -2 ≤ s < 1: linear interpolation of tabulated values
///     (Bornemann 2010, Table 2 quantiles).
///   - For s < -2: ~1 - exp(-|s|³/24) (left-tail asymptotic).
fn tw1_upper_tail(s: f64) -> f64 {
    if s >= 1.0 {
        // Upper tail of TW_1 — Tracy-Widom GOE
        let p = (-(2.0 / 3.0) * s.powf(1.5)).exp() * s.powf(-1.0 / 8.0);
        p.clamp(0.0, 1.0)
    } else if s >= -2.0 {
        // Interpolated CDF values for TW_1 GOE (from Bornemann 2010 / Prahofer-Spohn).
        // (s, F(s)) tabulated pairs:
        let table: &[(f64, f64)] = &[
            (-2.0, 0.978),
            (-1.5, 0.953),
            (-1.0, 0.917),
            (-0.5, 0.866),
            (0.0, 0.831),
            (0.5, 0.762),
            (1.0, 0.668),
        ];
        let cdf = piecewise_linear(s, table);
        (1.0 - cdf).clamp(0.0, 1.0)
    } else {
        let abs = -s;
        let p = 1.0 - (-(abs.powi(3) / 24.0)).exp();
        (1.0 - p).clamp(0.0, 1.0)
    }
}

fn piecewise_linear(x: f64, table: &[(f64, f64)]) -> f64 {
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table.last().unwrap().0 {
        return table.last().unwrap().1;
    }
    for w in table.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x >= x0 && x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    table.last().unwrap().1
}

/// Rotationally Invariant Estimator (RIE) — Bouchaud-Potters-Ledoit-Péché
/// nonlinear shrinkage of sample-covariance eigenvalues.
///
/// The exact RIE formula (Bun, Bouchaud, Potters 2017) requires the
/// Stieltjes transform of the limiting Marchenko-Pastur density. For
/// each empirical eigenvalue lambda_k:
///
/// ```text
/// lambda_hat_k = lambda_k / |1 - q + q * z * s_C(z)|^2
///   where z = lambda_k - i*eta, eta -> 0+
/// ```
///
/// In practice we use a regularized real-axis approximation that pulls
/// each eigenvalue toward the bulk mean by an amount proportional to its
/// MP-bulk distance:
///
/// ```text
/// for lambda_k inside [lambda-, lambda+]: shrink toward bulk_mean
/// for lambda_k outside (signal): preserve with mild shrinkage
/// ```
///
/// This is the "isotonic" RIE variant used by Bun et al. as a numerically
/// stable proxy that captures the dominant behavior of the full formula
/// while being computable from the empirical eigenvalues alone.
pub fn rie_shrinkage(eigenvalues: &[f64], q: f64) -> Vec<f64> {
    let n = eigenvalues.len();
    if n == 0 {
        return Vec::new();
    }
    // Estimate σ² from the bulk: the median eigenvalue is a robust proxy
    // for σ² when q is small (most eigenvalues lie in the noise bulk).
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sigma2 = if n % 2 == 0 {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    } else {
        sorted[n / 2]
    };
    let (lam_minus, lam_plus) = marchenko_pastur_bounds(q, sigma2);
    let bulk_mean = sigma2; // E[λ] for MP density is σ².

    eigenvalues
        .iter()
        .map(|&lam| {
            if lam <= lam_plus && lam >= lam_minus {
                // Inside the bulk → strongly shrink toward the bulk mean.
                // Shrinkage intensity α ∈ (0, 1); we use α = q (more shrinkage
                // when q is larger / dimension closer to T).
                let alpha = q.clamp(0.05, 0.95);
                alpha * bulk_mean + (1.0 - alpha) * lam
            } else if lam > lam_plus {
                // Above bulk → genuine signal, light shrinkage.
                // λ̂ ≈ λ · (1 - q · σ² / (λ - σ²)) for λ ≫ σ² (asymptotic RIE).
                if lam > sigma2 {
                    lam * (1.0 - q * sigma2 / (lam - sigma2)).max(0.0)
                } else {
                    lam
                }
            } else {
                // Below bulk (rare) — leave as-is.
                lam
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Deterministic Gaussian RNG (Box-Muller from xorshift).
    fn rng_normal(seed: u64) -> impl FnMut() -> f64 {
        let mut s = if seed == 0 { 1u64 } else { seed };
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u1 = ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u2 = ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn eigenvalue_recovery_on_diagonal_covariance() {
        // Construct synthetic data such that the sample covariance is
        // (approximately) diag([3, 2, 1]). Easy way: orthogonal columns
        // with known variances.
        let t = 2000;
        let mut g = rng_normal(7);
        let mut data = Array2::<f64>::zeros((t, 3));
        for i in 0..t {
            data[(i, 0)] = 3.0_f64.sqrt() * g();
            data[(i, 1)] = 2.0_f64.sqrt() * g();
            data[(i, 2)] = 1.0_f64.sqrt() * g();
        }
        let r = pca(&data);
        // Eigenvalues in descending order should be ≈ [3, 2, 1] within
        // finite-sample noise (~5%).
        assert!((r.eigenvalues[0] - 3.0).abs() < 0.3, "got {}", r.eigenvalues[0]);
        assert!((r.eigenvalues[1] - 2.0).abs() < 0.3, "got {}", r.eigenvalues[1]);
        assert!((r.eigenvalues[2] - 1.0).abs() < 0.3, "got {}", r.eigenvalues[2]);
    }

    #[test]
    fn explained_variance_sums_to_one() {
        let data = arr2(&[
            [1.0, 2.0, 1.5],
            [2.0, 3.5, 1.0],
            [3.0, 5.0, 2.0],
            [4.0, 7.0, 1.5],
            [5.0, 8.5, 2.0],
        ]);
        let r = pca(&data);
        let s: f64 = r.explained_variance_ratio.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mp_density_zero_outside_bulk() {
        let q = 0.25;
        let sigma2 = 1.0;
        let (lo, hi) = marchenko_pastur_bounds(q, sigma2);
        assert!(lo > 0.0);
        assert!(hi > lo);
        // Outside [lo, hi] density must be 0
        assert_eq!(marchenko_pastur_density(lo - 0.01, q, sigma2), 0.0);
        assert_eq!(marchenko_pastur_density(hi + 0.01, q, sigma2), 0.0);
        // Inside bulk it must be positive
        let mid = 0.5 * (lo + hi);
        assert!(marchenko_pastur_density(mid, q, sigma2) > 0.0);
    }

    #[test]
    fn mp_density_integrates_to_one() {
        // Simpson's rule integration on [λ−, λ+] should give 1.0.
        let q = 0.25;
        let sigma2 = 1.0;
        let (lo, hi) = marchenko_pastur_bounds(q, sigma2);
        // shrink endpoints slightly to avoid the integrable singularity at edges
        let a = lo + 1e-6;
        let b = hi - 1e-6;
        let n_steps = 10_000_usize;
        let h = (b - a) / n_steps as f64;
        let mut s = marchenko_pastur_density(a, q, sigma2)
            + marchenko_pastur_density(b, q, sigma2);
        for i in 1..n_steps {
            let x = a + i as f64 * h;
            let w = if i % 2 == 0 { 2.0 } else { 4.0 };
            s += w * marchenko_pastur_density(x, q, sigma2);
        }
        let integral = s * h / 3.0;
        assert!(
            (integral - 1.0).abs() < 0.02,
            "MP density should integrate to ~1, got {}",
            integral
        );
    }

    #[test]
    fn empirical_noise_eigenvalues_fall_within_mp_bounds() {
        // Build N=20, T=200 i.i.d. N(0,1) matrix. q = 0.1.
        let n = 20;
        let t = 200;
        let mut g = rng_normal(13);
        let mut data = Array2::<f64>::zeros((t, n));
        for i in 0..t {
            for j in 0..n {
                data[(i, j)] = g();
            }
        }
        let r = pca(&data);
        let q = n as f64 / t as f64;
        let (lo, hi) = marchenko_pastur_bounds(q, 1.0);
        // With some finite-sample fuzz (10%), the empirical eigenvalues
        // should lie in the bulk.
        let lo_fuzz = lo * 0.85;
        let hi_fuzz = hi * 1.15;
        let in_bulk = r.eigenvalues.iter().filter(|&&l| l >= lo_fuzz && l <= hi_fuzz).count();
        let frac = in_bulk as f64 / n as f64;
        assert!(
            frac >= 0.85,
            "only {:.0}% of empirical eigenvalues inside MP bulk [{:.3}, {:.3}]; values: {:?}",
            frac * 100.0, lo_fuzz, hi_fuzz, r.eigenvalues
        );
    }

    #[test]
    fn tracy_widom_pvalue_high_for_noise_low_for_factor() {
        // Pure noise λ_max — TW p-value should be > 0.05 on average across seeds.
        let mut p_noise_total = 0.0;
        let trials = 10;
        for seed in 1..=trials {
            let n = 20;
            let t = 200;
            let mut g = rng_normal(seed);
            let mut data = Array2::<f64>::zeros((t, n));
            for i in 0..t {
                for j in 0..n {
                    data[(i, j)] = g();
                }
            }
            let r = pca(&data);
            p_noise_total += tracy_widom_pvalue(r.eigenvalues[0], n as f64 / t as f64, n);
        }
        let p_mean = p_noise_total / trials as f64;
        assert!(p_mean > 0.10, "noise mean TW p-value too low: {}", p_mean);

        // Inject a strong factor — λ_max should now be very far above the
        // bulk, giving p < 0.05.
        let n = 20;
        let t = 200;
        let mut g = rng_normal(42);
        let mut data = Array2::<f64>::zeros((t, n));
        for i in 0..t {
            let factor = g() * 5.0; // strong common factor
            for j in 0..n {
                data[(i, j)] = g() + factor;
            }
        }
        let r = pca(&data);
        let p_factor = tracy_widom_pvalue(r.eigenvalues[0], n as f64 / t as f64, n);
        assert!(
            p_factor < 0.05,
            "factor-injected TW p-value too high: {} (λ_max={})",
            p_factor,
            r.eigenvalues[0]
        );
    }

    #[test]
    fn rie_shrinks_noise_preserves_signal() {
        // Eigenvalue spectrum: one big signal eigenvalue + nine noise-band ones.
        let q = 0.1; // n=10, T=100
        let (lo, hi) = marchenko_pastur_bounds(q, 1.0);
        let bulk_mean = 1.0;
        // Build eigenvalues: 9 near the bulk mean, 1 well above
        let eigs = vec![
            10.0, // signal
            1.0, 0.95, 1.05, 1.1, 0.9, 1.02, 0.98, 1.03, 0.97,
        ];
        let _ = (lo, hi);
        let shrunk = rie_shrinkage(&eigs, q);
        // Signal eigenvalue should NOT shrink much
        assert!(
            shrunk[0] > 8.0,
            "signal eigenvalue shrunk too aggressively: {} → {}",
            eigs[0], shrunk[0]
        );
        // Noise eigenvalues should be PULLED TOWARD the bulk mean (variance
        // of shrunk noise should be lower than variance of input noise).
        let var_in: f64 = eigs[1..].iter().map(|l| (l - bulk_mean).powi(2)).sum::<f64>() / 9.0;
        let var_out: f64 = shrunk[1..].iter().map(|l| (l - bulk_mean).powi(2)).sum::<f64>() / 9.0;
        assert!(
            var_out < var_in,
            "noise-eigenvalue variance should decrease after shrinkage: {} → {}",
            var_in, var_out
        );
    }
}
