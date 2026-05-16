//! Money primitives.
//!
//! `Price` and `Notional` use `rust_decimal` at the API boundary for safe
//! display and serde. Inside hot-path math the engine converts to f64 for
//! BLAS-friendly vectorization; PnL accounting goes back through Decimal.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub struct Price(pub Decimal);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub struct Qty(pub Decimal);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub struct Notional(pub Decimal);

impl Price {
    pub const ZERO: Self = Self(dec!(0));
    pub fn from_f64(f: f64) -> Option<Self> {
        Decimal::try_from(f).ok().map(Self)
    }
    pub fn to_f64(self) -> f64 {
        f64::try_from(self.0).unwrap_or(f64::NAN)
    }
    pub fn is_positive(self) -> bool {
        self.0 > Decimal::ZERO
    }
}

impl Qty {
    pub const ZERO: Self = Self(dec!(0));
    pub fn from_i64(n: i64) -> Self {
        Self(Decimal::from(n))
    }
    pub fn to_f64(self) -> f64 {
        f64::try_from(self.0).unwrap_or(f64::NAN)
    }
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
    pub fn is_zero(self) -> bool {
        self.0.is_zero()
    }
    pub fn signum_i8(self) -> i8 {
        if self.0 > Decimal::ZERO {
            1
        } else if self.0 < Decimal::ZERO {
            -1
        } else {
            0
        }
    }
}

impl Notional {
    pub const ZERO: Self = Self(dec!(0));
    pub fn from_f64(f: f64) -> Option<Self> {
        Decimal::try_from(f).ok().map(Self)
    }
    pub fn to_f64(self) -> f64 {
        f64::try_from(self.0).unwrap_or(f64::NAN)
    }
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl Mul<Qty> for Price {
    type Output = Notional;
    fn mul(self, q: Qty) -> Notional {
        Notional(self.0 * q.0)
    }
}

impl Add for Notional {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Notional {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl AddAssign for Notional {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Notional {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for Notional {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Add for Qty {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Qty {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl AddAssign for Qty {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Qty {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for Qty {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.0)
    }
}

impl fmt::Display for Qty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Notional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn price_times_qty_is_notional() {
        let p = Price::from_f64(123.45).unwrap();
        let q = Qty::from_i64(10);
        let n = p * q;
        assert_eq!(n.0, dec!(1234.50));
    }

    #[test]
    fn notional_arith() {
        let a = Notional::from_f64(100.0).unwrap();
        let b = Notional::from_f64(50.0).unwrap();
        assert_eq!((a + b).to_f64(), 150.0);
        assert_eq!((a - b).to_f64(), 50.0);
        assert_eq!((-a).to_f64(), -100.0);
    }
}
