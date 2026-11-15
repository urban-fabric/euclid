use super::MS_PER_S;
use crate::{TimeMillis, TimeSeconds};
use bytemuck::{Pod, Zeroable};

impl core::fmt::Display for TimeMicros {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {}",
            <TimeMicros as euclid_units::Measure>::raw(self),
            <TimeMicros as euclid_units::Measure>::UNIT_NAME,
        )
    }
}

impl core::fmt::Display for TimeMicrosDelta {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {}",
            <TimeMicrosDelta as euclid_units::Measure>::raw(self),
            <TimeMicrosDelta as euclid_units::Measure>::UNIT_NAME,
        )
    }
}

#[repr(transparent)]
/// A duration in **whole microseconds**, backed by `u64`.
///
/// Integer-backed and unsigned, which is the point: it cannot represent a negative
/// interval, so subtraction yields a [`TimeMicrosDelta`] rather than
/// wrapping. Exact at every magnitude, where a float duration loses sub-microsecond
/// resolution past about 2^53 µs.
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
pub struct TimeMicros(pub(crate) u64);

#[repr(transparent)]
/// A signed difference between two [`TimeMicros`], backed by `i64`.
///
/// Exists so that subtracting durations has a type: the unsigned duration cannot hold a
/// negative result, and silently saturating or wrapping would be worse than a distinct
/// type that says what it is.
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
pub struct TimeMicrosDelta(pub(crate) i64);

impl euclid_units::Measure for TimeMicros {
    type Repr = u64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "µs";
    fn raw(&self) -> u64 {
        self.0
    }
    fn from_raw(v: u64) -> Self {
        Self(v)
    }
}

impl TimeMicros {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0);

    /// Wrap a raw whole-microsecond count. Asserted rather than checked.
    pub const fn new(us: u64) -> Self {
        Self(us)
    }

    /// The raw count in whole microseconds.
    pub const fn as_micros(self) -> u64 {
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

    /// Convert from a [`std::time::Duration`].
    ///
    /// Saturates at this type's maximum rather than wrapping, so a `Duration` beyond its range
    /// clamps. `Duration` is unsigned, so this is always a valid duration.
    pub fn from_duration(d: std::time::Duration) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self(d.as_micros() as u64)
    }

    /// Convert to [`TimeSeconds`].
    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds::new(self.0 as f64 / 1_000_000.0)
    }

    /// Convert to [`TimeMillis`].
    pub fn to_millis(self) -> TimeMillis {
        TimeMillis::new(self.0 as f64 / MS_PER_S)
    }
}

impl euclid_units::Measure for TimeMicrosDelta {
    type Repr = i64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "µs (signed)";
    fn raw(&self) -> i64 {
        self.0
    }
    fn from_raw(v: i64) -> Self {
        Self(v)
    }
}

impl TimeMicrosDelta {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0);

    /// Wrap a raw signed microsecond difference.
    pub const fn new(us: i64) -> Self {
        Self(us)
    }

    /// The signed difference in whole microseconds.
    pub const fn as_micros(self) -> i64 {
        self.0
    }

    /// The signed interval from `before` to `after`.
    ///
    /// Negative when the arguments are out of order, which is the reason this returns a delta
    /// type: the unsigned duration could not say so.
    #[allow(clippy::cast_possible_wrap)]
    pub fn between(before: TimeMicros, after: TimeMicros) -> Self {
        Self(after.as_micros() as i64 - before.as_micros() as i64)
    }
}

impl std::ops::Add for TimeMicros {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for TimeMicros {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::iter::Sum for TimeMicros {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}
