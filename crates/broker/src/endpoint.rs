//! Endpoint allowlist. Live binary refuses to start with an unknown URL.
//!
//! These are the only URLs the engine will ever talk to. Strings are
//! compared verbatim — no clever normalization.

pub const ALLOWED_PAPER: &[&str] = &["https://paper-api.alpaca.markets/v2"];
pub const ALLOWED_LIVE: &[&str] = &["https://api.alpaca.markets/v2"];

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Endpoint {
    Paper,
    Live,
}

impl Endpoint {
    pub fn base_url(self) -> &'static str {
        match self {
            Endpoint::Paper => ALLOWED_PAPER[0],
            Endpoint::Live => ALLOWED_LIVE[0],
        }
    }
}

pub fn is_allowed_endpoint(url: &str, env: Endpoint) -> bool {
    let allow = match env {
        Endpoint::Paper => ALLOWED_PAPER,
        Endpoint::Live => ALLOWED_LIVE,
    };
    allow.contains(&url)
}
