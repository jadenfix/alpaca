//! Time primitives.
//!
//! `Ts` is a nanosecond timestamp anchored to TAI (via hifitime) and convertible
//! to UTC for display. All event timestamps in the engine are `Ts`.

use hifitime::Epoch;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Ts {
    /// Nanoseconds since the UTC epoch (1970-01-01T00:00:00Z).
    pub nanos: i64,
}

impl Ts {
    pub const fn from_nanos(nanos: i64) -> Self {
        Self { nanos }
    }

    pub fn from_epoch(e: Epoch) -> Self {
        // hifitime uses TAI internally; we coerce to UTC nanos for serde stability.
        let dur = e.to_utc_duration();
        let nanos = dur.to_parts();
        // (centuries, nanoseconds_within_century) — flatten:
        let total = nanos.0 as i64 * 100 * 365_2425i64 * 86_400 * 1_000_000_000 / 10_000
            + nanos.1 as i64;
        Self { nanos: total }
    }

    pub fn to_epoch(&self) -> Epoch {
        Epoch::from_unix_duration(hifitime::Duration::from_truncated_nanoseconds(self.nanos))
    }

    pub const fn millis(self) -> i64 {
        self.nanos / 1_000_000
    }

    pub const fn micros(self) -> i64 {
        self.nanos / 1_000
    }

    pub const fn zero() -> Self {
        Self { nanos: 0 }
    }

    pub fn checked_add_nanos(self, n: i64) -> Option<Self> {
        self.nanos.checked_add(n).map(|nanos| Self { nanos })
    }

    /// Non-negative elapsed nanoseconds from `earlier` to `self`. Returns 0 if `self < earlier`.
    pub fn elapsed_nanos_since(self, earlier: Self) -> i64 {
        self.nanos.saturating_sub(earlier.nanos).max(0)
    }
}

impl Ord for Ts {
    fn cmp(&self, other: &Self) -> Ordering {
        self.nanos.cmp(&other.nanos)
    }
}

impl PartialOrd for Ts {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Ts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sec = self.nanos / 1_000_000_000;
        let nsec = (self.nanos % 1_000_000_000).unsigned_abs();
        write!(f, "Ts({sec}.{nsec:09})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_is_nonneg_when_after() {
        let a = Ts::from_nanos(1_000);
        let b = Ts::from_nanos(5_000);
        assert_eq!(b.elapsed_nanos_since(a), 4_000);
    }

    #[test]
    fn elapsed_saturates_to_zero_when_before() {
        let a = Ts::from_nanos(5_000);
        let b = Ts::from_nanos(1_000);
        assert_eq!(b.elapsed_nanos_since(a), 0);
    }

    #[test]
    fn ord_works() {
        assert!(Ts::from_nanos(1) < Ts::from_nanos(2));
    }
}
