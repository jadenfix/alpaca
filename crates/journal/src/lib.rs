//! Append-only forensic event log + structured trade audit.
//!
//! Two complementary log streams:
//!   - `JournalRecord` — full event/intent/fill stream for deterministic replay.
//!     Every input event, decision, order intent, and fill is appended as a
//!     single line of JSON. Live: written from a tokio task with `fdatasync`;
//!     backtest: written synchronously.
//!   - `AuditEntry`    — structured, regime-tagged trade record for downstream
//!     attribution (Sharpe per regime, hit rate per strategy, etc.).

pub mod audit;
pub use audit::{AuditAction, AuditEntry, AuditLog};

use algo_core::{Fill, MarketEvent, OrderIntent, Ts};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum JournalRecord {
    Market(MarketEvent),
    Intent(OrderIntent),
    Fill(Fill),
    Risk { ts: Ts, reason: String, intent_id: String },
    Kill { ts: Ts, level: String, reason: String },
    Note { ts: Ts, msg: String },
}

#[derive(thiserror::Error, Debug)]
pub enum JournalError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct JournalWriter {
    inner: Mutex<BufWriter<File>>,
}

impl JournalWriter {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, JournalError> {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            inner: Mutex::new(BufWriter::new(f)),
        })
    }

    pub fn append(&self, rec: &JournalRecord) -> Result<(), JournalError> {
        let mut g = self.inner.lock();
        let line = serde_json::to_string(rec)?;
        g.write_all(line.as_bytes())?;
        g.write_all(b"\n")?;
        Ok(())
    }

    pub fn flush(&self) -> Result<(), JournalError> {
        self.inner.lock().flush()?;
        Ok(())
    }

    pub fn fdatasync(&self) -> Result<(), JournalError> {
        let g = self.inner.lock();
        g.get_ref().sync_data()?;
        Ok(())
    }
}

pub struct JournalReader {
    reader: BufReader<File>,
}

impl JournalReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, JournalError> {
        let f = File::open(path)?;
        Ok(Self {
            reader: BufReader::new(f),
        })
    }
}

impl Iterator for JournalReader {
    type Item = Result<JournalRecord, JournalError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    self.next()
                } else {
                    Some(serde_json::from_str(trimmed).map_err(Into::into))
                }
            }
            Err(e) => Some(Err(e.into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algo_core::{MarketEvent, Ts};

    #[test]
    fn roundtrip_journal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.log");
        let w = JournalWriter::open(&path).unwrap();
        let rec = JournalRecord::Market(MarketEvent::SessionOpen {
            ts: Ts::from_nanos(123),
        });
        w.append(&rec).unwrap();
        w.append(&JournalRecord::Note {
            ts: Ts::from_nanos(124),
            msg: "hello".into(),
        })
        .unwrap();
        w.flush().unwrap();
        drop(w);
        let r = JournalReader::open(&path).unwrap();
        let recs: Vec<_> = r.collect::<Result<_, _>>().unwrap();
        assert_eq!(recs.len(), 2);
    }
}
