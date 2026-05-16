//! Trade audit log: a structured record of every order intent + fill +
//! risk decision, with the regime and feature snapshot that justified it.
//!
//! Goal: complete explainability. Every trade can be traced back to:
//!   - the market events that triggered it
//!   - the strategy + parameters that generated it
//!   - the regime label at decision time
//!   - the feature values (z-score, vol, signal magnitude, etc.) used
//!   - the risk decision (Accept / Reject + reason)
//!   - the fill price + fees vs the assumed reference price (slippage attribution)
//!
//! Stored alongside the regular `JournalRecord` stream. Replay reproduces
//! the full audit; the strategy recommender reads it for per-regime PnL
//! attribution.

use algo_core::Ts;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    pub ts: Ts,
    pub strategy: String,
    pub symbol: String,
    pub action: AuditAction,
    /// Regime label at decision time, e.g. "bull", "bear", "sideways", "calm", "turbulent".
    pub regime: Option<String>,
    /// Free-form key-value features that motivated the decision.
    /// Stored as a BTreeMap for deterministic serialization order.
    pub features: BTreeMap<String, f64>,
    pub notes: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AuditAction {
    IntentEmitted {
        side: String,
        qty: f64,
        order_type: String,
    },
    RiskRejected {
        side: String,
        qty: f64,
        reason: String,
    },
    Filled {
        side: String,
        qty: f64,
        fill_price: f64,
        reference_price: f64,
        fees: f64,
    },
    RegimeShift {
        from: String,
        to: String,
        confidence: f64,
    },
}

#[derive(Default, Clone, Debug)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn record(&mut self, entry: AuditEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Group entries by `(strategy, regime)` for downstream analysis.
    pub fn group_by_strategy_regime(&self) -> BTreeMap<(String, String), Vec<&AuditEntry>> {
        let mut out: BTreeMap<(String, String), Vec<&AuditEntry>> = BTreeMap::new();
        for e in &self.entries {
            let key = (
                e.strategy.clone(),
                e.regime.clone().unwrap_or_else(|| "none".to_string()),
            );
            out.entry(key).or_default().push(e);
        }
        out
    }

    /// Write to a JSON-lines file. Each line is one AuditEntry.
    pub fn write_jsonl(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        use std::io::Write;
        let f = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(f);
        for e in &self.entries {
            let line = serde_json::to_string(e).map_err(std::io::Error::other)?;
            writeln!(w, "{line}")?;
        }
        Ok(())
    }

    /// Read from a JSON-lines file.
    pub fn read_jsonl(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        use std::io::BufRead;
        let f = std::fs::File::open(path)?;
        let r = std::io::BufReader::new(f);
        let mut log = Self::new();
        for line in r.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: AuditEntry = serde_json::from_str(&line).map_err(std::io::Error::other)?;
            log.record(entry);
        }
        Ok(log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_entry(strategy: &str, regime: &str, action: AuditAction) -> AuditEntry {
        let mut features = BTreeMap::new();
        features.insert("zscore".to_string(), 1.5);
        features.insert("vol".to_string(), 0.02);
        AuditEntry {
            ts: Ts::from_nanos(0),
            strategy: strategy.to_string(),
            symbol: "AAPL".to_string(),
            action,
            regime: Some(regime.to_string()),
            features,
            notes: None,
        }
    }

    #[test]
    fn round_trips_via_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let mut log = AuditLog::new();
        log.record(mk_entry("xs_momentum", "bull", AuditAction::IntentEmitted {
            side: "buy".into(),
            qty: 100.0,
            order_type: "market".into(),
        }));
        log.record(mk_entry("xs_momentum", "bear", AuditAction::RiskRejected {
            side: "buy".into(),
            qty: 100.0,
            reason: "name cap exceeded".into(),
        }));
        log.write_jsonl(&path).unwrap();
        let back = AuditLog::read_jsonl(&path).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.entries()[0].strategy, "xs_momentum");
    }

    #[test]
    fn group_by_strategy_regime_partitions_correctly() {
        let mut log = AuditLog::new();
        log.record(mk_entry("xs", "bull", AuditAction::IntentEmitted {
            side: "buy".into(), qty: 1.0, order_type: "market".into(),
        }));
        log.record(mk_entry("xs", "bull", AuditAction::IntentEmitted {
            side: "buy".into(), qty: 1.0, order_type: "market".into(),
        }));
        log.record(mk_entry("xs", "bear", AuditAction::IntentEmitted {
            side: "buy".into(), qty: 1.0, order_type: "market".into(),
        }));
        log.record(mk_entry("mr", "bull", AuditAction::IntentEmitted {
            side: "sell".into(), qty: 1.0, order_type: "market".into(),
        }));
        let groups = log.group_by_strategy_regime();
        assert_eq!(groups.get(&("xs".into(), "bull".into())).unwrap().len(), 2);
        assert_eq!(groups.get(&("xs".into(), "bear".into())).unwrap().len(), 1);
        assert_eq!(groups.get(&("mr".into(), "bull".into())).unwrap().len(), 1);
    }
}
