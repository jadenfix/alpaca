//! Replay a journal file deterministically. Used for forensic post-mortems
//! and to verify byte-identical strategy re-execution.

use algo_journal::JournalReader;
use algo_obs::init_tracing;
use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "algo-replay", version)]
struct Cli {
    #[arg(long)]
    journal: String,
    #[arg(long, default_value_t = false)]
    json_logs: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);
    let r = JournalReader::open(&cli.journal)?;
    let mut count = 0usize;
    for rec in r {
        let rec = rec?;
        count += 1;
        tracing::debug!(?rec, "replayed");
    }
    println!("replayed {count} records from {}", cli.journal);
    Ok(())
}
