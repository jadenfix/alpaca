//! Shadow-live binary.
//!
//! Connects to live data, runs strategies, but **does not send orders** —
//! every `OrderIntent` is logged to the journal as if it were sent. Used in
//! the 2-week shadow-live phase to catch eligibility / sizing / auth surprises
//! before any capital is at risk.

use algo_obs::init_tracing;
use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "algo-shadow", version)]
struct Cli {
    #[arg(long, default_value_t = false)]
    json_logs: bool,
    #[arg(long, default_value_t = String::from("./journal/shadow.log"))]
    journal: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);
    tracing::warn!(
        path = %cli.journal,
        "shadow-live binary: skeleton. Will subscribe to live WS data and log \
         intended orders only — never sends. Wire-up lands with mdata WS connect."
    );
    Ok(())
}
