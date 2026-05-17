//! Distance correlation — Szekely, Rizzo, Bakirov (2007).
//!
//! For real series X, Y of length n:
//!   1. a_{ij} = |x_i - x_j|, b_{ij} = |y_i - y_j|.
//!   2. Double-center: A_{ij} = a_{ij} - ā_{i.} - ā_{.j} + ā_{..} (and B).
//!   3. dCov²(X,Y) = (1/n²) Σ A_{ij} B_{ij};
//!      dVar(X) = (1/n²) Σ A_{ij}², dVar(Y) similarly.
//!   4. dCor = √(dCov² / √(dVar_X · dVar_Y)).
//!
//! Key property: dCor(X, Y) = 0 ⇔ X ⫫ Y. **Catches `y = x²`** where
//! Pearson misses it entirely.
//!
//! O(n²) memory and time — fine for the universes we work with.

pub fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "distance_correlation: lengths differ");
    if n < 4 {
        return 0.0;
    }
    let nf = n as f64;

    let mut a = vec![0.0_f64; n * n];
    let mut b = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (x[i] - x[j]).abs();
            b[i * n + j] = (y[i] - y[j]).abs();
        }
    }

    let mut a_row = vec![0.0_f64; n];
    let mut a_col = vec![0.0_f64; n];
    let mut a_grand = 0.0;
    let mut b_row = vec![0.0_f64; n];
    let mut b_col = vec![0.0_f64; n];
    let mut b_grand = 0.0;
    for i in 0..n {
        for j in 0..n {
            a_row[i] += a[i * n + j];
            a_col[j] += a[i * n + j];
            a_grand += a[i * n + j];
            b_row[i] += b[i * n + j];
            b_col[j] += b[i * n + j];
            b_grand += b[i * n + j];
        }
    }
    for i in 0..n {
        a_row[i] /= nf;
        a_col[i] /= nf;
        b_row[i] /= nf;
        b_col[i] /= nf;
    }
    a_grand /= nf * nf;
    b_grand /= nf * nf;

    let mut d_cov_sq = 0.0;
    let mut d_var_x = 0.0;
    let mut d_var_y = 0.0;
    for i in 0..n {
        for j in 0..n {
            let ac = a[i * n + j] - a_row[i] - a_col[j] + a_grand;
            let bc = b[i * n + j] - b_row[i] - b_col[j] + b_grand;
            d_cov_sq += ac * bc;
            d_var_x += ac * ac;
            d_var_y += bc * bc;
        }
    }
    let nf2 = nf * nf;
    d_cov_sq /= nf2;
    d_var_x /= nf2;
    d_var_y /= nf2;

    let denom = (d_var_x * d_var_y).max(0.0).sqrt();
    if denom <= 0.0 {
        return 0.0;
    }
    (d_cov_sq / denom).max(0.0).sqrt().min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_uniform(seed: u64) -> impl FnMut() -> f64 {
        let mut s = if seed == 0 { 1u64 } else { seed };
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0)
                / ((1u64 << 53) as f64 + 1.0)
        }
    }

    #[test]
    fn dcor_perfect_dependence_is_one() {
        let x: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let y = x.clone();
        let d = distance_correlation(&x, &y);
        assert!((d - 1.0).abs() < 1e-9, "dCor(x,x) should be 1, got {d}");
    }

    #[test]
    fn dcor_independent_samples_is_small() {
        let mut g1 = rng_uniform(7);
        let mut g2 = rng_uniform(99);
        let n = 500;
        let x: Vec<f64> = (0..n).map(|_| g1()).collect();
        let y: Vec<f64> = (0..n).map(|_| g2()).collect();
        let d = distance_correlation(&x, &y);
        assert!(d < 0.15, "dCor on independent samples too large: {d}");
    }

    /// THE killer test: dCor catches y = x² where Pearson misses it.
    #[test]
    fn dcor_catches_y_equals_x_squared() {
        let mut g = rng_uniform(13);
        let n = 400;
        let x: Vec<f64> = (0..n).map(|_| 2.0 * g() - 1.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();
        let d = distance_correlation(&x, &y);
        assert!(
            d > 0.3,
            "dCor should be >> 0 for y=x² with x symmetric, got {d}"
        );
        // Sanity: confirm Pearson IS indeed near 0 here, so the test isn't vacuous.
        let mean_x = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        let pearson = sxy / (sxx * syy).sqrt();
        assert!(
            pearson.abs() < 0.15,
            "for the test to be meaningful, Pearson should be ~0; got {pearson}"
        );
    }
}
