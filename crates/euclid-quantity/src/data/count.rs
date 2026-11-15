//! Dimensionless count newtypes.

use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[serde(transparent)]
pub struct Count(u64);

impl Count {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }

    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }

    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }

    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for Count {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Count {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Count {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Count {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl AddAssign<u64> for Count {
    fn add_assign(&mut self, rhs: u64) {
        self.0 += rhs;
    }
}

impl From<u64> for Count {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl std::str::FromStr for Count {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u64>().map(Self)
    }
}

impl From<Count> for u64 {
    fn from(value: Count) -> Self {
        value.0
    }
}

impl PartialEq<u64> for Count {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Count> for u64 {
    fn eq(&self, other: &Count) -> bool {
        *self == other.0
    }
}

impl PartialOrd<u64> for Count {
    fn partial_cmp(&self, other: &u64) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[serde(transparent)]
pub struct ThreadCount(u32);

impl ThreadCount {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ThreadCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 1 {
            write!(f, "1 thread")
        } else {
            write!(f, "{} threads", self.0)
        }
    }
}

impl From<u32> for ThreadCount {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<ThreadCount> for u32 {
    fn from(value: ThreadCount) -> Self {
        value.0
    }
}

impl euclid_units::Measure for Count {
    type Repr = u64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Dimensionless;
    const UNIT_NAME: &'static str = "";
    fn raw(&self) -> u64 {
        self.0
    }
    fn from_raw(v: u64) -> Self {
        Self(v)
    }
}

impl euclid_units::DiscreteQuantity for Count {
    fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

impl euclid_units::Measure for ThreadCount {
    type Repr = u32;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Dimensionless;
    const UNIT_NAME: &'static str = "thread";
    fn raw(&self) -> u32 {
        self.0
    }
    fn from_raw(v: u32) -> Self {
        Self(v)
    }
}

impl euclid_units::DiscreteQuantity for ThreadCount {
    fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::{Count, ThreadCount};

    #[test]
    fn count_is_dimensionless_discrete_measure() {
        use euclid_units::{DiscreteQuantity, Measure};
        assert_eq!(
            <Count as Measure>::DIMENSION,
            euclid_units::Dimension::Dimensionless
        );
        assert_eq!(
            DiscreteQuantity::add(Count::new(5), Count::new(3)),
            Count::new(8)
        );
        assert_eq!(
            DiscreteQuantity::saturating_sub(Count::new(3), Count::new(5)),
            Count::ZERO
        );
    }

    #[test]
    fn count_display_is_bare_number() {
        assert_eq!(Count::new(42).to_string(), "42");
    }

    #[test]
    fn count_serde_is_bare_u64() {
        let c = Count::new(42);
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, "42");
        let back: Count = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn count_arithmetic_round_trips() {
        let mut c = Count::ZERO;
        c += Count::new(5);
        c += Count::new(3);
        assert_eq!(c, Count::new(8));
        c -= Count::new(1);
        assert_eq!(c, Count::new(7));
    }

    #[test]
    fn thread_count_display_pluralizes() {
        assert_eq!(ThreadCount::new(1).to_string(), "1 thread");
        assert_eq!(ThreadCount::new(8).to_string(), "8 threads");
    }

    #[test]
    fn thread_count_serde_is_bare_u32() {
        let t = ThreadCount::new(8);
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "8");
        let back: ThreadCount = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }
}
