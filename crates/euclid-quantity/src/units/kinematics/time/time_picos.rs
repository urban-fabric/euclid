use super::{PS_PER_NS, PS_PER_S};
use crate::{TimeMicros, TimeMillis, TimeNanos, TimeSeconds};
use bytemuck::{Pod, Zeroable};

impl core::fmt::Display for TimePicos {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {}",
            <TimePicos as euclid_units::Measure>::raw(self),
            <TimePicos as euclid_units::Measure>::UNIT_NAME,
        )
    }
}

impl core::fmt::Display for TimePicosDelta {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {}",
            <TimePicosDelta as euclid_units::Measure>::raw(self),
            <TimePicosDelta as euclid_units::Measure>::UNIT_NAME,
        )
    }
}

#[repr(transparent)]
/// A duration in **whole picoseconds**, backed by `u64`.
///
/// `u64` picoseconds spans only about 213 days — the shortest range in the time family,
/// and the reason to reach for it only where picosecond resolution is genuinely needed.
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
pub struct TimePicos(pub(crate) u64);

#[repr(transparent)]
/// A signed difference between two [`TimePicos`], backed by `i64`.
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
pub struct TimePicosDelta(pub(crate) i64);

impl euclid_units::Measure for TimePicos {
    type Repr = u64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "ps";
    fn raw(&self) -> u64 {
        self.0
    }
    fn from_raw(v: u64) -> Self {
        Self(v)
    }
}

impl TimePicos {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0);

    /// Wrap a raw whole-picosecond count. Asserted rather than checked.
    pub const fn new(ps: u64) -> Self {
        Self(ps)
    }

    /// The raw count in whole picoseconds.
    pub const fn as_picos(self) -> u64 {
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
        Self(d.as_nanos().saturating_mul(PS_PER_NS as u128) as u64)
    }

    /// Truncating conversion to whole nanoseconds.
    pub const fn to_nanos(self) -> TimeNanos {
        TimeNanos::new(self.0 / PS_PER_NS)
    }

    /// Truncating conversion to whole microseconds.
    pub const fn to_micros(self) -> TimeMicros {
        self.to_nanos().to_micros()
    }

    /// Convert to [`TimeMillis`].
    pub fn to_millis(self) -> TimeMillis {
        self.to_seconds().to_millis()
    }

    /// Convert to [`TimeSeconds`].
    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds::new(self.0 as f64 / PS_PER_S)
    }
}

impl euclid_units::Measure for TimePicosDelta {
    type Repr = i64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "ps (signed)";
    fn raw(&self) -> i64 {
        self.0
    }
    fn from_raw(v: i64) -> Self {
        Self(v)
    }
}

impl TimePicosDelta {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0);

    /// Wrap a raw signed picosecond difference.
    pub const fn new(ps: i64) -> Self {
        Self(ps)
    }

    /// The signed difference in whole picoseconds.
    pub const fn as_picos(self) -> i64 {
        self.0
    }

    /// The signed interval from `before` to `after`; negative when they are out of order.
    #[allow(clippy::cast_possible_wrap)]
    pub fn between(before: TimePicos, after: TimePicos) -> Self {
        Self(after.as_picos() as i64 - before.as_picos() as i64)
    }
}

impl std::ops::Add for TimePicos {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for TimePicos {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::iter::Sum for TimePicos {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}
