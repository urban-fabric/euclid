//! Dimensionless count newtypes.

use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[repr(transparent)]
/// A cardinal **count** of things, backed by `u64`.
///
/// Dimensionless but deliberately not a [`crate::Factor`]: a count is integral and
/// unsigned, and arithmetic that would make it fractional is a mistake worth a type error.
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
/// A count of **threads**, backed by `u32`.
///
/// Distinct from [`crate::Count`] because it is a configuration value with its own
/// sane range, and because a thread count accidentally used as an item count is the kind
/// of substitution a newtype exists to stop.
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
    /// Wrap a raw `u32` already expressed in this unit. Asserted rather than checked.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// The raw count, discarding the unit.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// The raw count as `usize`, for indexing a host collection.
    ///
    /// Truncates on a 32-bit target, where `u64` does not fit.
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

    /// The conversion, arithmetic and mixed-comparison surface for `Count` and `ThreadCount`.
    ///
    /// Thirty-four survivors lived here, and the cause is the same split the tests above show:
    /// `DiscreteQuantity::add` is a shared default and is checked, while `impl Add for Count`,
    /// `impl Sub for Count`, the `From` conversions, `FromStr` and the mixed comparisons are
    /// separate implementations that nothing reached.
    #[test]
    fn the_raw_accessors_agree_with_the_value() {
        for raw in [0_u64, 1, 42, u64::MAX / 2] {
            let c = Count::new(raw);
            assert_eq!(c.as_u64(), raw);
            #[allow(clippy::cast_precision_loss)]
            {
                assert_eq!(c.as_f64(), raw as f64);
                assert_eq!(c.as_f32(), raw as f32);
            }
        }
    }

    #[test]
    fn the_operators_agree_with_the_trait_defaults() {
        for (a, b) in [(0_u64, 0_u64), (5, 3), (3, 5), (42, 42)] {
            assert_eq!(
                Count::new(a) + Count::new(b),
                euclid_units::DiscreteQuantity::add(Count::new(a), Count::new(b)),
                "+ must agree with DiscreteQuantity::add at ({a}, {b})"
            );
            assert_eq!(
                Count::new(a) - Count::new(b.min(a)),
                Count::new(a - b.min(a)),
                "- must subtract at ({a}, {b})"
            );
        }

        // `+=` against a bare u64 is the same addition.
        let mut c = Count::new(5);
        c += 3_u64;
        assert_eq!(c, Count::new(8));
    }

    #[test]
    fn saturating_sub_clamps_at_zero() {
        assert_eq!(Count::new(3).saturating_sub(Count::new(5)), Count::new(0));
        assert_eq!(Count::new(5).saturating_sub(Count::new(3)), Count::new(2));
        assert_eq!(Count::new(4).saturating_sub(Count::new(4)), Count::new(0));
    }

    #[test]
    fn the_integer_conversions_round_trip() {
        for raw in [0_u64, 1, 42, u64::MAX] {
            assert_eq!(u64::from(Count::from(raw)), raw, "u64 -> Count -> u64");
        }
        for raw in [0_u32, 1, 8, u32::MAX] {
            assert_eq!(
                u32::from(ThreadCount::from(raw)),
                raw,
                "u32 -> ThreadCount -> u32"
            );
        }
    }

    #[test]
    fn parsing_a_count_reads_the_number() {
        use core::str::FromStr;
        assert_eq!(Count::from_str("0"), Ok(Count::new(0)));
        assert_eq!(Count::from_str("42"), Ok(Count::new(42)));
        // And a non-number is an error rather than a default.
        assert!(Count::from_str("not a number").is_err());
        assert!(Count::from_str("-1").is_err(), "a count is unsigned");
    }

    #[test]
    fn mixed_comparisons_agree_with_comparing_the_raw_values() {
        for a in [0_u64, 1, 7, 42] {
            for b in [0_u64, 1, 7, 42] {
                let wrapped = Count::new(a);
                assert_eq!(wrapped == b, a == b, "Count == u64 at ({a}, {b})");
                assert_eq!(b == wrapped, b == a, "u64 == Count at ({a}, {b})");
                assert_eq!(wrapped.partial_cmp(&b), a.partial_cmp(&b));
                assert!(wrapped.partial_cmp(&b).is_some());
            }
        }
    }

    #[test]
    fn the_thread_count_accessors_agree_with_the_value() {
        for raw in [0_u32, 1, 8, u32::MAX] {
            let t = ThreadCount::new(raw);
            assert_eq!(t.as_u32(), raw, "as_u32 must be the raw count");
            assert_eq!(
                t.as_usize(),
                raw as usize,
                "as_usize must widen the raw count"
            );
        }
    }
}
