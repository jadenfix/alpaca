//! Live trading entrypoint.
//!
//! Per the plan: this binary refuses to start without explicit confirmation
//! and only against an allowlisted endpoint. Today the WS + REST wiring is a
//! skeleton — calls return `NotImplemented`, which is precisely what we want
//! before the shadow-live milestone (no quiet half-trading).

use algo_broker::{AlpacaRest, Endpoint};
use algo_obs::init_tracing;
use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Mode {
    Paper,
    Live,
}

#[derive(Parser, Debug)]
#[command(name = "algo-live", version)]
struct Cli {
    #[arg(long, value_enum, default_value_t = Mode::Paper)]
    mode: Mode,
    #[arg(long, default_value_t = false)]
    json_logs: bool,
    /// Required to start in Live mode. Belt-and-braces with the env var.
    #[arg(long, default_value_t = false)]
    i_understand_this_is_real_money: bool,
}

/// Decide which endpoint to use based on mode + safety inputs.
/// Returns `Ok(Endpoint)` on accept; `Err(message)` if any safety guard fails.
///
/// This is the SINGLE source of truth for the live-mode safety policy.
/// All five guards in the plan flow through here so they can be regression-tested:
///   - Live mode without the explicit flag → reject
///   - Live mode without the env var       → reject
///   - Live mode with wrong env var value  → reject
///   - Live mode with both                  → accept
///   - Paper mode (no requirements)         → accept
fn validate_live_invocation(
    mode: Mode,
    real_money_flag: bool,
    live_env_var: Option<&str>,
) -> Result<Endpoint, String> {
    match mode {
        Mode::Paper => Ok(Endpoint::Paper),
        Mode::Live => {
            if !real_money_flag {
                return Err("Live mode requires --i-understand-this-is-real-money".into());
            }
            match live_env_var {
                Some("yes") => Ok(Endpoint::Live),
                Some(other) => Err(format!(
                    "Live mode requires LIVE_TRADING_CONFIRMED=yes; got {other:?}"
                )),
                None => Err("Live mode requires LIVE_TRADING_CONFIRMED=yes in environment".into()),
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);

    let live_env = std::env::var("LIVE_TRADING_CONFIRMED").ok();
    let endpoint = validate_live_invocation(
        cli.mode,
        cli.i_understand_this_is_real_money,
        live_env.as_deref(),
    )
    .map_err(|e| anyhow!(e))?;

    tracing::info!(?cli.mode, base_url = endpoint.base_url(), "starting live binary");

    let broker = AlpacaRest::from_env(endpoint)?;
    use algo_broker::Broker;
    match broker.account_snapshot().await {
        Ok(snap) => tracing::info!(?snap, "account snapshot ok"),
        Err(e) => {
            tracing::warn!(?e, "account snapshot not yet implemented — exiting cleanly");
            return Ok(());
        }
    }
    Ok(())
}

// ── safety-guard regression tests ──
//
// These tests would fail if a future refactor removed any of the live-mode
// guards. The validate_live_invocation function is the single chokepoint;
// keeping these green keeps the binary safe.

#[cfg(test)]
mod tests {
    use super::*;
    use algo_broker::Endpoint;

    #[test]
    fn paper_mode_accepts_without_any_guard() {
        let result = validate_live_invocation(Mode::Paper, false, None);
        assert_eq!(result.unwrap(), Endpoint::Paper);
    }

    #[test]
    fn paper_mode_ignores_live_env_var() {
        // Paper mode doesn't care what the env var says.
        let result = validate_live_invocation(Mode::Paper, false, Some("yes"));
        assert_eq!(result.unwrap(), Endpoint::Paper);
    }

    #[test]
    fn live_without_flag_is_rejected() {
        let result = validate_live_invocation(Mode::Live, false, Some("yes"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("--i-understand-this-is-real-money"));
    }

    #[test]
    fn live_without_env_var_is_rejected() {
        let result = validate_live_invocation(Mode::Live, true, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("LIVE_TRADING_CONFIRMED"));
    }

    #[test]
    fn live_with_wrong_env_var_value_is_rejected() {
        for bad_value in ["no", "true", "1", "", "YES"] {
            let result = validate_live_invocation(Mode::Live, true, Some(bad_value));
            assert!(
                result.is_err(),
                "env var '{bad_value}' must be rejected (only 'yes' accepted)"
            );
        }
    }

    #[test]
    fn live_with_flag_and_env_var_yes_is_accepted() {
        let result = validate_live_invocation(Mode::Live, true, Some("yes"));
        assert_eq!(result.unwrap(), Endpoint::Live);
    }

    /// Regression guard: even if a future refactor accidentally inverts the
    /// boolean check, Paper should NEVER be promoted to Live by the function.
    #[test]
    fn paper_mode_never_returns_live_endpoint() {
        for flag in [true, false] {
            for env in [Some("yes"), Some("no"), None] {
                let result = validate_live_invocation(Mode::Paper, flag, env);
                assert_eq!(
                    result.unwrap(),
                    Endpoint::Paper,
                    "paper must always → paper (flag={flag}, env={env:?})"
                );
            }
        }
    }
}
