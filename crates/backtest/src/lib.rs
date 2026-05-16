//! Event-driven backtest simulator.

pub mod analytics;
pub mod portfolio_state;
pub mod simulator;
pub mod synthetic;
pub mod walkforward;

pub use analytics::{compute as compute_metrics, Metrics};
pub use portfolio_state::{PortfolioState, Position};
pub use simulator::{BacktestConfig, BacktestReport, Simulator};
pub use synthetic::{
    generate_cointegrated_pair, generate_gbm_universe, generate_momentum_universe,
    SyntheticConfig, XorShift64,
};
pub use walkforward::{run_walkforward, FoldResult, WalkForwardConfig, WalkForwardReport};
