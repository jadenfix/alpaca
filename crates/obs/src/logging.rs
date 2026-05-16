//! Tracing initialization.
//!
//! Default format: compact, ANSI only when stderr is a TTY, no per-event
//! target spam, no thread ids. JSON layout opt-in via `init_tracing(true)`.
//!
//! Filter precedence: `RUST_LOG` env var if set, else `info` for our crates
//! and `warn` for everything else (silences chatty hyper/reqwest/tungstenite
//! at info+).

use std::io::IsTerminal;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub fn init_tracing(json: bool) {
    let default_filter = "info,\
        hyper=warn,hyper_util=warn,reqwest=warn,tungstenite=warn,\
        tokio_tungstenite=warn,rustls=warn,h2=warn,want=warn,mio=warn";
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_filter));

    let use_ansi = std::io::stderr().is_terminal();

    if json {
        let layer = fmt::layer()
            .json()
            .with_current_span(false)
            .with_span_list(false)
            .with_target(true);
        let _ = tracing_subscriber::registry()
            .with(env_filter)
            .with(layer)
            .try_init();
    } else {
        let layer = fmt::layer()
            .with_ansi(use_ansi)
            .with_target(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .with_file(false)
            .with_line_number(false)
            .with_level(true)
            .compact();
        let _ = tracing_subscriber::registry()
            .with(env_filter)
            .with(layer)
            .try_init();
    }
}
