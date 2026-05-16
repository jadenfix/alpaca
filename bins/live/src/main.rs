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

#[derive(Copy, Clone, Debug, ValueEnum)]
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);

    let endpoint = match cli.mode {
        Mode::Paper => Endpoint::Paper,
        Mode::Live => {
            if !cli.i_understand_this_is_real_money {
                return Err(anyhow!(
                    "Live mode requires --i-understand-this-is-real-money"
                ));
            }
            if std::env::var("LIVE_TRADING_CONFIRMED").as_deref() != Ok("yes") {
                return Err(anyhow!(
                    "Live mode requires LIVE_TRADING_CONFIRMED=yes in environment"
                ));
            }
            Endpoint::Live
        }
    };

    tracing::info!(?cli.mode, base_url = endpoint.base_url(), "starting live binary");

    let broker = AlpacaRest::from_env(endpoint)?;
    // Skeleton path: account snapshot returns NotImplemented today.
    // This proves the auth + endpoint guard at startup; full wiring lands
    // with the shadow-live milestone.
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
