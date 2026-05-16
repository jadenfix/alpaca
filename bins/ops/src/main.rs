//! Operator interventions: flatten-all, cancel-all, set-kill, reconcile.

use algo_obs::init_tracing;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "algo-ops", version)]
struct Cli {
    #[arg(long, default_value_t = false)]
    json_logs: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Create a KILL file. Live binary flattens on next loop iteration.
    SetKill {
        #[arg(long, default_value = "./KILL")]
        path: PathBuf,
    },
    /// Remove the KILL file.
    ClearKill {
        #[arg(long, default_value = "./KILL")]
        path: PathBuf,
    },
    /// Stub: flatten all open positions via Alpaca REST (NotImplemented for now).
    FlattenAll,
    /// Stub: cancel all open orders via Alpaca REST (NotImplemented for now).
    CancelAll,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);

    match cli.cmd {
        Cmd::SetKill { path } => {
            std::fs::write(&path, b"kill\n")?;
            tracing::warn!(?path, "KILL file created — live binary will halt on next tick");
        }
        Cmd::ClearKill { path } => {
            if path.exists() {
                std::fs::remove_file(&path)?;
                tracing::info!(?path, "KILL file removed");
            } else {
                tracing::info!(?path, "no KILL file present");
            }
        }
        Cmd::FlattenAll => {
            tracing::warn!("flatten-all: not yet wired to broker");
        }
        Cmd::CancelAll => {
            tracing::warn!("cancel-all: not yet wired to broker");
        }
    }
    Ok(())
}
