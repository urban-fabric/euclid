use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[repr(transparent)]
/// A count of **bytes**, backed by `u64`.
///
/// Unsigned and integral because a size is neither negative nor fractional. Conversions to
/// KiB/MiB/GiB are binary (1024-based), not decimal — the units a filesystem and an
/// allocator report in.
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
    /// Zero.
    pub const ZERO: Self = Self(0);

    /// Wrap a raw `u64` already expressed in this unit. Asserted rather than checked.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The raw count, discarding the unit.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// The count as `f32`.
    ///
    /// Exact only up to 2^24 — about 16.7 million — so this is for display and interchange
    /// rather than arithmetic.
    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }

    /// The count as `f64`.
    ///
    /// Exact up to 2^53; above that, consecutive counts stop being distinguishable.
    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }

    /// Subtract, clamping at zero instead of underflowing.
    ///
    /// The clamp is why this type can be unsigned at all. Note it is **lossy by design**:
    /// `a.saturating_sub(b)` is zero whenever `b >= a`, so it cannot be used to discover which
    /// of two values was larger.
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
/// A half-open byte range `[start, end)`, as used for a range request or a slice of an
/// object.
///
/// Half-open so that an empty range is representable and adjacent ranges tile without
/// overlapping, which the inclusive spelling cannot do.
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
    /// Offset of the first byte in the range.
    pub offset: Bytes,
    /// Number of bytes in the range.
    pub len: Bytes,
}

impl ByteRange {
    /// A range of `len` bytes starting at `offset`.
    pub const fn new(offset: Bytes, len: Bytes) -> Self {
        Self { offset, len }
    }

    /// A zero-length range at `offset`.
    ///
    /// Representable because the range is half-open, which is the reason for that choice: an
    /// empty range still has a position, and an inclusive spelling cannot express one.
    pub const fn empty(offset: Bytes) -> Self {
        Self {
            offset,
            len: Bytes::ZERO,
        }
    }

    /// One past the last byte — `offset + len`, the exclusive upper bound.
    pub fn end(&self) -> Bytes {
        self.offset + self.len
    }

    /// Whether the range covers no bytes.
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

    /// The conversion and mixed-type comparison surface, which nothing was exercising.
    ///
    /// The tests above check the `DiscreteQuantity` trait methods — `DiscreteQuantity::add`,
    /// `DiscreteQuantity::saturating_sub` — and those are shared default bodies. The inherent
    /// `Bytes::saturating_sub`, the `From` conversions and the mixed `PartialEq`/`PartialOrd`
    /// against a bare `u64` are separate implementations, and mutation testing found all nineteen
    /// of them replaceable: `as_f64` could return `0.0`, `PartialEq<u64>` could return `true` for
    /// every pair, `PartialOrd<u64>` could answer `None`.
    ///
    /// The mixed comparisons are the ones worth caring about. `Bytes` exists so a byte count
    /// cannot be confused with any other number, and these impls are the deliberate escape hatch
    /// back to `u64`. An escape hatch that compares wrongly is worse than none.
    #[test]
    fn the_raw_accessors_agree_with_the_value() {
        for raw in [0_u64, 1, 1024, 4096, u64::MAX / 2] {
            let b = Bytes::new(raw);
            assert_eq!(b.as_u64(), raw, "as_u64 must be the raw count");
            #[allow(clippy::cast_precision_loss)]
            {
                assert_eq!(b.as_f64(), raw as f64, "as_f64 must widen the raw count");
                assert_eq!(b.as_f32(), raw as f32, "as_f32 must widen the raw count");
            }
        }
    }

    #[test]
    fn saturating_sub_clamps_at_zero_rather_than_underflowing() {
        assert_eq!(
            Bytes::new(100).saturating_sub(Bytes::new(40)),
            Bytes::new(60)
        );
        assert_eq!(Bytes::new(40).saturating_sub(Bytes::new(100)), Bytes::ZERO);
        assert_eq!(Bytes::new(0).saturating_sub(Bytes::new(1)), Bytes::ZERO);
        // Equal operands are the boundary, and must give zero rather than wrapping.
        assert_eq!(Bytes::new(7).saturating_sub(Bytes::new(7)), Bytes::ZERO);

        // The inherent method and the trait default must agree, since both are published.
        for (a, b) in [(100_u64, 40_u64), (40, 100), (0, 0), (7, 7)] {
            assert_eq!(
                Bytes::new(a).saturating_sub(Bytes::new(b)),
                euclid_units::DiscreteQuantity::saturating_sub(Bytes::new(a), Bytes::new(b)),
                "the inherent saturating_sub must agree with the trait one at ({a}, {b})"
            );
        }
    }

    #[test]
    fn the_u64_conversions_round_trip() {
        for raw in [0_u64, 1, 4096, u64::MAX] {
            assert_eq!(u64::from(Bytes::from(raw)), raw, "u64 -> Bytes -> u64");
            let b = Bytes::new(raw);
            assert_eq!(Bytes::from(u64::from(b)), b, "Bytes -> u64 -> Bytes");
        }
    }

    #[test]
    fn adding_a_bare_u64_matches_adding_the_wrapped_one() {
        for (start, add) in [(0_u64, 0_u64), (100, 24), (4096, 1)] {
            let mut raw_added = Bytes::new(start);
            raw_added += add;
            assert_eq!(
                raw_added,
                Bytes::new(start) + Bytes::new(add),
                "+= u64 must agree with + Bytes"
            );
        }
    }

    #[test]
    fn mixed_comparisons_agree_with_comparing_the_raw_values() {
        for a in [0_u64, 1, 7, 1024] {
            for b in [0_u64, 1, 7, 1024] {
                let wrapped = Bytes::new(a);
                assert_eq!(wrapped == b, a == b, "Bytes == u64 at ({a}, {b})");
                assert_eq!(b == wrapped, b == a, "u64 == Bytes at ({a}, {b})");
                assert_eq!(
                    wrapped.partial_cmp(&b),
                    a.partial_cmp(&b),
                    "Bytes partial_cmp u64 at ({a}, {b})"
                );
                // Never incomparable: these are integers.
                assert!(wrapped.partial_cmp(&b).is_some());
            }
        }
    }
}
