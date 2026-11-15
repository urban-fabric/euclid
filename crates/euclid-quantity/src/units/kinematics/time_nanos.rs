use super::{NS_PER_S, NS_PER_US};
use crate::{TimeMicros, TimeMillis, TimeSeconds};
use bytemuck::{Pod, Zeroable};

impl core::fmt::Display for TimeNanos {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {}",
            <TimeNanos as euclid_units::Measure>::raw(self),
            <TimeNanos as euclid_units::Measure>::UNIT_NAME,
        )
    }
}

impl core::fmt::Display for TimeNanosDelta {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {}",
            <TimeNanosDelta as euclid_units::Measure>::raw(self),
            <TimeNanosDelta as euclid_units::Measure>::UNIT_NAME,
        )
    }
}

#[repr(transparent)]
/// A duration in **whole nanoseconds**, backed by `u64`. See
/// [`crate::TimeMicros`] for why the integer family is unsigned.
///
/// `u64` nanoseconds spans about 584 years, which is the practical bound on this type.
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct TimeNanos(pub(crate) u64);

#[repr(transparent)]
/// A signed difference between two [`TimeNanos`], backed by `i64`.
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct TimeNanosDelta(pub(crate) i64);

impl euclid_units::Measure for TimeNanos {
    type Repr = u64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "ns";
    fn raw(&self) -> u64 {
        self.0
    }
    fn from_raw(v: u64) -> Self {
        Self(v)
    }
}

impl TimeNanos {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0);

    /// Wrap a raw whole-nanosecond count. Asserted rather than checked.
    pub const fn new(ns: u64) -> Self {
        Self(ns)
    }

    /// The raw count in whole nanoseconds.
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Add, clamping at the type's maximum instead of overflowing.
    ///
    /// Saturating rather than wrapping or panicking: for a size or a count, a total that is
    /// implausibly large is a less dangerous failure than one that has silently wrapped to near
    /// zero and reads as plausible.
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Subtract, clamping at zero instead of underflowing.
    ///
    /// The clamp is why this type can be unsigned at all. Note it is **lossy by design**:
    /// `a.saturating_sub(b)` is zero whenever `b >= a`, so it cannot be used to discover which
    /// of two values was larger.
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Convert from a [`std::time::Duration`].
    ///
    /// Saturates at this type's maximum rather than wrapping, so a `Duration` beyond its range
    /// clamps. `Duration` is unsigned, so this is always a valid duration.
    pub fn from_duration(d: std::time::Duration) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self(d.as_nanos() as u64)
    }

    /// Truncating conversion to whole microseconds.
    pub const fn to_micros(self) -> TimeMicros {
        TimeMicros::new(self.0 / NS_PER_US)
    }

    /// Convert to [`TimeMillis`].
    pub fn to_millis(self) -> TimeMillis {
        self.to_seconds().to_millis()
    }

    /// Convert to [`TimeSeconds`].
    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds::new(self.0 as f64 / NS_PER_S)
    }
}

impl euclid_units::Measure for TimeNanosDelta {
    type Repr = i64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "ns (signed)";
    fn raw(&self) -> i64 {
        self.0
    }
    fn from_raw(v: i64) -> Self {
        Self(v)
    }
}

impl TimeNanosDelta {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0);

    /// Wrap a raw signed nanosecond difference.
    pub const fn new(ns: i64) -> Self {
        Self(ns)
    }

    /// The signed difference in whole nanoseconds.
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    /// The signed interval from `before` to `after`; negative when they are out of order.
    #[allow(clippy::cast_possible_wrap)]
    pub fn between(before: TimeNanos, after: TimeNanos) -> Self {
        Self(after.as_nanos() as i64 - before.as_nanos() as i64)
    }
}

impl std::ops::Add for TimeNanos {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for TimeNanos {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::iter::Sum for TimeNanos {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}
