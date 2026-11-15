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
pub struct Bytes(u64);

impl Bytes {
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

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} B", self.0)
    }
}

impl Add for Bytes {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Bytes {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Bytes {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Bytes {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl std::ops::AddAssign<u64> for Bytes {
    fn add_assign(&mut self, rhs: u64) {
        self.0 += rhs;
    }
}

impl From<u64> for Bytes {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<Bytes> for u64 {
    fn from(value: Bytes) -> Self {
        value.0
    }
}

impl PartialEq<u64> for Bytes {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Bytes> for u64 {
    fn eq(&self, other: &Bytes) -> bool {
        *self == other.0
    }
}

impl PartialOrd<u64> for Bytes {
    fn partial_cmp(&self, other: &u64) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

#[repr(C)]
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
pub struct ByteRange {
    pub offset: Bytes,
    pub len: Bytes,
}

impl ByteRange {
    pub const fn new(offset: Bytes, len: Bytes) -> Self {
        Self { offset, len }
    }

    pub const fn empty(offset: Bytes) -> Self {
        Self {
            offset,
            len: Bytes::ZERO,
        }
    }

    pub fn end(&self) -> Bytes {
        self.offset + self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len.0 == 0
    }
}

impl fmt::Display for ByteRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}..{}) ({})", self.offset.0, self.end().0, self.len)
    }
}

impl euclid_units::Measure for Bytes {
    type Repr = u64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Information;
    const UNIT_NAME: &'static str = "B";
    fn raw(&self) -> u64 {
        self.0
    }
    fn from_raw(v: u64) -> Self {
        Self(v)
    }
}

impl euclid_units::DiscreteQuantity for Bytes {
    fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn bytes_is_information_measure() {
        use euclid_units::{DiscreteQuantity, Measure};
        assert_eq!(
            <Bytes as Measure>::DIMENSION,
            euclid_units::Dimension::Information
        );
        assert_eq!(<Bytes as Measure>::UNIT_NAME, "B");
        assert_eq!(
            DiscreteQuantity::add(Bytes::new(100), Bytes::new(40)),
            Bytes::new(140)
        );
        assert_eq!(
            DiscreteQuantity::saturating_sub(Bytes::new(40), Bytes::new(100)),
            Bytes::ZERO
        );
    }

    #[test]
    fn bytes_display_carries_unit() {
        assert_eq!(Bytes::new(1024).to_string(), "1024 B");
    }

    #[test]
    fn bytes_serde_is_bare_u64() {
        let b = Bytes::new(4096);
        let json = serde_json::to_string(&b).unwrap();
        assert_eq!(json, "4096");
        let back: Bytes = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn bytes_add_sub_round_trip() {
        let a = Bytes::new(100);
        let b = Bytes::new(40);
        assert_eq!((a + b).as_u64(), 140);
        assert_eq!((a - b).as_u64(), 60);
        let mut c = a;
        c += b;
        assert_eq!(c.as_u64(), 140);
        c -= b;
        assert_eq!(c.as_u64(), 100);
    }

    #[test]
    fn byte_range_end_and_emptiness() {
        let r = ByteRange::new(Bytes::new(16), Bytes::new(32));
        assert_eq!(r.end(), Bytes::new(48));
        assert!(!r.is_empty());
        assert!(ByteRange::empty(Bytes::new(7)).is_empty());
    }

    #[test]
    fn byte_range_display_uses_half_open_form() {
        let r = ByteRange::new(Bytes::new(16), Bytes::new(32));
        assert_eq!(r.to_string(), "[16..48) (32 B)");
    }

    #[test]
    fn byte_range_serde_round_trip() {
        let r = ByteRange::new(Bytes::new(16), Bytes::new(32));
        let json = serde_json::to_string(&r).unwrap();
        let back: ByteRange = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }
}
