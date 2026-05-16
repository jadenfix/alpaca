//! Vectorized cross-sectional z-scores. Used heavily by cross-sectional
//! momentum: rank symbols by recent return / vol, then z-score the ranks.

use ndarray::{Array1, ArrayView1};

/// Z-score a vector cross-sectionally: `(x - mean) / std`. NaNs in input are
/// excluded from mean/std and produce NaN outputs at their positions.
pub fn zscore_vec(xs: ArrayView1<f64>) -> Array1<f64> {
    let mut out = xs.to_owned();
    zscore_vec_into(&mut out);
    out
}

/// In-place vectorized z-score.
pub fn zscore_vec_into(xs: &mut Array1<f64>) {
    let (sum, sq_sum, n) = xs
        .iter()
        .filter(|v| v.is_finite())
        .fold((0.0_f64, 0.0_f64, 0_usize), |(s, sq, c), &v| {
            (s + v, sq + v * v, c + 1)
        });
    if n < 2 {
        xs.fill(f64::NAN);
        return;
    }
    let mean = sum / n as f64;
    let var = (sq_sum - mean * mean * n as f64) / (n as f64 - 1.0);
    let std = var.max(0.0).sqrt();
    if std == 0.0 {
        xs.fill(0.0);
        return;
    }
    // Vectorized assignment via mapv-inplace
    xs.mapv_inplace(|v| if v.is_finite() { (v - mean) / std } else { f64::NAN });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn zscore_known() {
        let xs = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = zscore_vec(xs.view());
        // mean=3, sample-std=sqrt(2.5)
        let std = 2.5_f64.sqrt();
        assert!((z[0] - (-2.0 / std)).abs() < 1e-9);
        assert!((z[2] - 0.0).abs() < 1e-9);
        assert!((z[4] - (2.0 / std)).abs() < 1e-9);
    }

    #[test]
    fn zero_variance_zeros_out() {
        let xs = array![5.0, 5.0, 5.0];
        let z = zscore_vec(xs.view());
        assert!(z.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn nan_passthrough_for_input_nan() {
        let xs = array![1.0, f64::NAN, 3.0, 5.0];
        let z = zscore_vec(xs.view());
        assert!(z[1].is_nan());
        assert!(z[0].is_finite());
    }
}
