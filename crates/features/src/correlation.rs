//! Correlation primitives: Pearson, Spearman, Kendall, plus a full
//! correlation matrix over columns of an `Array2<f64>`, plus a streaming
//! rolling correlation built on top of `Welford`.
//!
//! First-principles framing:
//!   - **Pearson** is affine-invariant (captures *linear* dependence only).
//!     ρ = E[(X-μ_X)(Y-μ_Y)] / (σ_X σ_Y).
//!   - **Spearman** = Pearson on ranks; invariant under any monotone
//!     transformation (captures monotone dependence).
//!   - **Kendall τ** counts concordant minus discordant pairs / total pairs;
//!     distribution-free, robust to outliers.
//!
//! All three reduce to ±1 for perfectly (anti-)correlated input and to 0
//! in expectation under independence — but they catch *different*
//! dependence structures. Spearman picks up `y = x^3`; Pearson misses it
//! after centering.
//!
//! Reference: Spearman (1904); Kendall (1938).

use ndarray::Array2;

/// Pearson product-moment correlation. Returns 0.0 (with `is_nan` -> 0
/// coercion) when either input has zero variance, so callers don't have
/// to guard.
pub fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "pearson_corr: input lengths differ");
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = x.iter().sum::<f64>() / nf;
    let mean_y = y.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom <= 0.0 || !denom.is_finite() {
        0.0
    } else {
        (sxy / denom).clamp(-1.0, 1.0)
    }
}

/// Average ranks of `x` (with mid-rank tie-breaking).
fn ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        // average rank for ties
        let avg = ((i + j - 1) as f64) * 0.5 + 1.0; // 1-based ranks
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

/// Spearman rank correlation = Pearson on ranks. Invariant under monotone
/// transformations of either input.
pub fn spearman_corr(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "spearman_corr: input lengths differ");
    let rx = ranks(x);
    let ry = ranks(y);
    pearson_corr(&rx, &ry)
}

/// Kendall's τ (tau-b, with tie correction). O(n²) — fine for n up to a few
/// thousand. For the n we work with (universes of ≤100 series, windows of
/// ≤500 bars) this is ample.
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "kendall_tau: input lengths differ");
    if n < 2 {
        return 0.0;
    }
    let mut concordant = 0_i64;
    let mut discordant = 0_i64;
    let mut tied_x = 0_i64;
    let mut tied_y = 0_i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            let dy = y[j] - y[i];
            if dx == 0.0 && dy == 0.0 {
                // tied on both — counted in neither tie correction by convention
                continue;
            } else if dx == 0.0 {
                tied_x += 1;
            } else if dy == 0.0 {
                tied_y += 1;
            } else if dx.signum() == dy.signum() {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total = (n as i64) * (n as i64 - 1) / 2;
    let denom_x = ((total - tied_x) as f64).max(0.0);
    let denom_y = ((total - tied_y) as f64).max(0.0);
    let denom = (denom_x * denom_y).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        ((concordant - discordant) as f64 / denom).clamp(-1.0, 1.0)
    }
}

/// Full pairwise Pearson correlation matrix over the COLUMNS of `data`.
/// `data` is `(n_observations, n_variables)`. Returns an `n_variables ×
/// n_variables` symmetric matrix with 1.0 on the diagonal.
pub fn correlation_matrix(data: &Array2<f64>) -> Array2<f64> {
    let (_t, n) = data.dim();
    let mut out = Array2::<f64>::zeros((n, n));
    // Cache column slices once; we access them many times.
    let cols: Vec<Vec<f64>> = (0..n)
        .map(|j| data.column(j).iter().copied().collect())
        .collect();
    for i in 0..n {
        out[(i, i)] = 1.0;
        for j in (i + 1)..n {
            let r = pearson_corr(&cols[i], &cols[j]);
            out[(i, j)] = r;
            out[(j, i)] = r;
        }
    }
    out
}

/// Streaming rolling Pearson correlation via Welford-style cross-product.
///
/// Maintains running mean(x), mean(y), and the centered cross-product
/// Σ(x-μ_x)(y-μ_y). Exactly numerically equivalent to the batch formula
/// after `n` updates. Use for hot-path correlation tracking in live
/// strategies (e.g., realtime pair-trading z-score).
#[derive(Clone, Debug, Default)]
pub struct RollingCorr {
    n: u64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    cov: f64,
}

impl RollingCorr {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, x: f64, y: f64) {
        self.n += 1;
        let nf = self.n as f64;
        let dx = x - self.mean_x;
        let dy = y - self.mean_y;
        self.mean_x += dx / nf;
        self.mean_y += dy / nf;
        let dx2 = x - self.mean_x;
        let dy2 = y - self.mean_y;
        self.m2_x += dx * dx2;
        self.m2_y += dy * dy2;
        self.cov += dx * dy2;
    }
    pub fn n(&self) -> u64 {
        self.n
    }
    pub fn corr(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let denom = (self.m2_x * self.m2_y).sqrt();
        if denom <= 0.0 {
            0.0
        } else {
            (self.cov / denom).clamp(-1.0, 1.0)
        }
    }
    pub fn cov(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.cov / (self.n - 1) as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn pearson_perfect_positive_is_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((pearson_corr(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect_negative_is_minus_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|v| -v).collect();
        assert!((pearson_corr(&x, &y) - -1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_zero_variance_returns_zero_not_nan() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![5.0, 5.0, 5.0];
        let r = pearson_corr(&x, &y);
        assert_eq!(r, 0.0);
        assert!(r.is_finite());
    }

    #[test]
    fn spearman_invariant_under_monotone_transform() {
        // y = exp(x) is strictly monotone → Spearman(x, y) should be exactly 1.
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v.exp()).collect();
        let rho = spearman_corr(&x, &y);
        assert!((rho - 1.0).abs() < 1e-12, "got rho={}", rho);
        // Pearson on the same gives < 1 because exp is curved.
        let pear = pearson_corr(&x, &y);
        assert!(pear < 0.999, "expected Pearson < 1, got {}", pear);
    }

    #[test]
    fn kendall_hand_checked_n5() {
        // x = (1,2,3,4,5), y = (1,2,3,4,5) → all C(5,2)=10 pairs concordant → τ=1
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((kendall_tau(&x, &y) - 1.0).abs() < 1e-12);
        // Reverse y → all 10 pairs discordant → τ = -1
        let y_rev: Vec<f64> = y.iter().rev().copied().collect();
        assert!((kendall_tau(&x, &y_rev) - -1.0).abs() < 1e-12);
        // One swap (5 ↔ 4): 9 concordant, 1 discordant → (9-1)/10 = 0.8
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 5.0, 4.0];
        let tau = kendall_tau(&x, &y);
        assert!((tau - 0.8).abs() < 1e-12, "expected 0.8, got {}", tau);
    }

    #[test]
    fn correlation_matrix_symmetric_with_unit_diagonal() {
        // 3 cols, 5 rows
        let m = arr2(&[
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 3.0],
            [3.0, 6.0, 1.5],
            [4.0, 8.0, 4.0],
            [5.0, 10.0, 2.5],
        ]);
        let c = correlation_matrix(&m);
        assert_eq!(c.dim(), (3, 3));
        // Diagonal exactly 1
        for i in 0..3 {
            assert!((c[(i, i)] - 1.0).abs() < 1e-12);
        }
        // Symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((c[(i, j)] - c[(j, i)]).abs() < 1e-12);
            }
        }
        // Col 0 and col 1 are perfectly linearly related (y = 2x) → ρ = 1
        assert!((c[(0, 1)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rolling_corr_matches_batch() {
        let x: Vec<f64> = (1..=50).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (1..=50).map(|i| (i as f64 * 0.1).cos()).collect();
        let mut rc = RollingCorr::new();
        for i in 0..x.len() {
            rc.push(x[i], y[i]);
        }
        let batch = pearson_corr(&x, &y);
        assert!(
            (rc.corr() - batch).abs() < 1e-10,
            "streaming {} vs batch {}",
            rc.corr(),
            batch
        );
    }
}
