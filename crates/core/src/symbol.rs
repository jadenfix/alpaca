//! Compact symbol type.
//!
//! US equity tickers are short (≤ 8 chars). Storing them inline avoids
//! heap allocs in hot-path event dispatch.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Symbol {
    bytes: [u8; 12],
    len: u8,
}

impl Symbol {
    /// Construct from a string. Returns `None` if too long or non-ASCII upper/digit/dot/dash.
    pub fn new(s: &str) -> Option<Self> {
        let bs = s.as_bytes();
        if bs.is_empty() || bs.len() > 12 {
            return None;
        }
        for &b in bs {
            if !(b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'.' || b == b'-') {
                return None;
            }
        }
        let mut bytes = [0u8; 12];
        bytes[..bs.len()].copy_from_slice(bs);
        Some(Self {
            bytes,
            len: bs.len() as u8,
        })
    }

    pub fn as_str(&self) -> &str {
        // SAFETY: validated ASCII in constructor.
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len as usize]) }
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol({})", self.as_str())
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_ascii_tickers() {
        for s in ["SPY", "AAPL", "BRK.B", "QQQ"] {
            let sym = Symbol::new(s).unwrap();
            assert_eq!(sym.as_str(), s);
        }
    }

    #[test]
    fn rejects_lowercase_and_long() {
        assert!(Symbol::new("aapl").is_none());
        assert!(Symbol::new("TOOLONGTICKER!").is_none());
        assert!(Symbol::new("").is_none());
    }
}
