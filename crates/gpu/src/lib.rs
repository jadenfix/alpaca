//! GPU / accelerated inference layer (feature-gated).
//!
//! Build with `--features gpu` to enable `candle` (CUDA on Linux, Metal on
//! macOS). Default build is CPU-only — `Inferencer` falls back to a tiny
//! ndarray path so the rest of the engine doesn't need conditional code.

use ndarray::{Array1, Array2};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("GPU feature not enabled — rebuild with --features gpu")]
    NotEnabled,
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
}

pub trait Inferencer: Send + Sync {
    /// Predict scores: input rows × feature dim → output row scores.
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>, GpuError>;
}

/// CPU baseline: linear model y = X @ w + b. Vectorized via BLAS through ndarray.
pub struct LinearCpu {
    pub w: Array1<f64>,
    pub b: f64,
}

impl Inferencer for LinearCpu {
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>, GpuError> {
        if features.ncols() != self.w.len() {
            return Err(GpuError::ShapeMismatch(format!(
                "features cols={} weights len={}",
                features.ncols(),
                self.w.len()
            )));
        }
        let mut y = features.dot(&self.w);
        y += self.b;
        Ok(y)
    }
}

#[cfg(feature = "gpu")]
pub mod candle_inferencer {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Skeleton: holds a candle device + (TODO) weights. Falls back to a
    /// shape check so the build compiles.
    pub struct CandleLinear {
        pub device: Device,
        pub w: Vec<f32>,
        pub b: f32,
        pub in_dim: usize,
    }

    impl Inferencer for CandleLinear {
        fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>, GpuError> {
            if features.ncols() != self.in_dim {
                return Err(GpuError::ShapeMismatch(format!(
                    "expected {} features",
                    self.in_dim
                )));
            }
            let n = features.nrows();
            let xs: Vec<f32> = features.iter().map(|v| *v as f32).collect();
            let x = Tensor::from_vec(xs, (n, self.in_dim), &self.device)
                .map_err(|e| GpuError::ShapeMismatch(e.to_string()))?;
            let w = Tensor::from_vec(self.w.clone(), (self.in_dim,), &self.device)
                .map_err(|e| GpuError::ShapeMismatch(e.to_string()))?;
            let y = x
                .matmul(&w.unsqueeze(1).map_err(|e| GpuError::ShapeMismatch(e.to_string()))?)
                .map_err(|e| GpuError::ShapeMismatch(e.to_string()))?;
            let y = y.squeeze(1).map_err(|e| GpuError::ShapeMismatch(e.to_string()))?;
            let v: Vec<f32> = y
                .to_vec1()
                .map_err(|e| GpuError::ShapeMismatch(e.to_string()))?;
            Ok(Array1::from(v.into_iter().map(|v| v as f64 + self.b as f64).collect::<Vec<_>>()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn linear_cpu_predicts_correctly() {
        let m = LinearCpu {
            w: ndarray::arr1(&[0.5, -0.25]),
            b: 1.0,
        };
        let x = arr2(&[[2.0, 4.0], [1.0, 1.0]]);
        let y = m.predict(&x).unwrap();
        // row 0: 2*0.5 + 4*(-0.25) + 1 = 1 - 1 + 1 = 1.0
        // row 1: 1*0.5 + 1*(-0.25) + 1 = 0.5 - 0.25 + 1 = 1.25
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 1.25).abs() < 1e-12);
    }
}
