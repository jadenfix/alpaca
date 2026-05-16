//! Broker abstraction + Alpaca REST client (skeleton).
//!
//! The trait is `Broker`. Two implementations:
//!   - `AlpacaRest` — real HTTP client wired to paper or live endpoint.
//!   - `PaperBroker` (in `paper` module) — in-process simulator used in
//!     shadow-live and tests.
//!
//! The live binary MUST refuse unknown endpoints. The allowlist below is the
//! source of truth.

pub mod alpaca;
pub mod broker_trait;
pub mod endpoint;
pub mod paper;

pub use alpaca::AlpacaRest;
pub use broker_trait::{Broker, BrokerError, OrderAck};
pub use endpoint::{is_allowed_endpoint, Endpoint, ALLOWED_PAPER, ALLOWED_LIVE};
pub use paper::PaperBroker;
