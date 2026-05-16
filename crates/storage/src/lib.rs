//! Bar storage. Skeleton: typed CSV roundtrip now; Parquet (polars) goes
//! behind a feature flag once we actually backfill history.

use algo_core::{Bar, Price, Qty, Symbol, Ts};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BarRow {
    pub ts_nanos: i64,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub span_secs: u32,
}

impl From<&Bar> for BarRow {
    fn from(b: &Bar) -> Self {
        Self {
            ts_nanos: b.ts.nanos,
            symbol: b.symbol.to_string(),
            open: b.open.to_f64(),
            high: b.high.to_f64(),
            low: b.low.to_f64(),
            close: b.close.to_f64(),
            volume: b.volume.to_f64(),
            span_secs: b.span_secs,
        }
    }
}

impl TryFrom<&BarRow> for Bar {
    type Error = StorageError;
    fn try_from(r: &BarRow) -> Result<Self, Self::Error> {
        Ok(Bar {
            ts: Ts::from_nanos(r.ts_nanos),
            symbol: Symbol::new(&r.symbol).ok_or_else(|| StorageError::BadSymbol(r.symbol.clone()))?,
            open: Price::from_f64(r.open).ok_or(StorageError::BadNumber)?,
            high: Price::from_f64(r.high).ok_or(StorageError::BadNumber)?,
            low: Price::from_f64(r.low).ok_or(StorageError::BadNumber)?,
            close: Price::from_f64(r.close).ok_or(StorageError::BadNumber)?,
            volume: Qty::from_i64(r.volume as i64),
            span_secs: r.span_secs,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad symbol: {0}")]
    BadSymbol(String),
    #[error("bad number")]
    BadNumber,
    #[error("parse: {0}")]
    Parse(String),
}

pub fn write_bars_csv(path: impl AsRef<Path>, bars: &[Bar]) -> Result<(), StorageError> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "ts_nanos,symbol,open,high,low,close,volume,span_secs")?;
    for b in bars {
        writeln!(
            w,
            "{},{},{},{},{},{},{},{}",
            b.ts.nanos,
            b.symbol,
            b.open.to_f64(),
            b.high.to_f64(),
            b.low.to_f64(),
            b.close.to_f64(),
            b.volume.to_f64(),
            b.span_secs
        )?;
    }
    Ok(())
}

pub fn read_bars_csv(path: impl AsRef<Path>) -> Result<Vec<Bar>, StorageError> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 8 {
            return Err(StorageError::Parse(format!("bad columns at line {i}")));
        }
        let row = BarRow {
            ts_nanos: cols[0]
                .parse()
                .map_err(|e: std::num::ParseIntError| StorageError::Parse(e.to_string()))?,
            symbol: cols[1].to_string(),
            open: cols[2]
                .parse()
                .map_err(|e: std::num::ParseFloatError| StorageError::Parse(e.to_string()))?,
            high: cols[3]
                .parse()
                .map_err(|e: std::num::ParseFloatError| StorageError::Parse(e.to_string()))?,
            low: cols[4]
                .parse()
                .map_err(|e: std::num::ParseFloatError| StorageError::Parse(e.to_string()))?,
            close: cols[5]
                .parse()
                .map_err(|e: std::num::ParseFloatError| StorageError::Parse(e.to_string()))?,
            volume: cols[6]
                .parse()
                .map_err(|e: std::num::ParseFloatError| StorageError::Parse(e.to_string()))?,
            span_secs: cols[7]
                .parse()
                .map_err(|e: std::num::ParseIntError| StorageError::Parse(e.to_string()))?,
        };
        out.push(Bar::try_from(&row)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bars.csv");
        let bars = vec![Bar {
            ts: Ts::from_nanos(1),
            symbol: Symbol::new("AAPL").unwrap(),
            open: Price::from_f64(100.0).unwrap(),
            high: Price::from_f64(101.0).unwrap(),
            low: Price::from_f64(99.5).unwrap(),
            close: Price::from_f64(100.5).unwrap(),
            volume: Qty::from_i64(12345),
            span_secs: 60,
        }];
        write_bars_csv(&path, &bars).unwrap();
        let back = read_bars_csv(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].close.to_f64(), 100.5);
    }
}
