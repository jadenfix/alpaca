//! Observability: HDR histograms for per-stage latency + tracing setup.

pub mod latency;
pub mod logging;

pub use latency::{LatencyStage, LatencyTracker};
pub use logging::init_tracing;
