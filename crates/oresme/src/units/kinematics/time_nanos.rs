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
    pub const ZERO: Self = Self(0);

    pub const fn new(ns: u64) -> Self {
        Self(ns)
    }

    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    pub fn from_duration(d: std::time::Duration) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self(d.as_nanos() as u64)
    }

    /// Truncating conversion to whole microseconds.
    pub const fn to_micros(self) -> TimeMicros {
        TimeMicros::new(self.0 / NS_PER_US)
    }

    pub fn to_millis(self) -> TimeMillis {
        self.to_seconds().to_millis()
    }

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
    pub const ZERO: Self = Self(0);

    pub const fn new(ns: i64) -> Self {
        Self(ns)
    }

    pub const fn as_nanos(self) -> i64 {
        self.0
    }

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
