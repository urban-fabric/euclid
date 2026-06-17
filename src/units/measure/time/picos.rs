use bytemuck::{Pod, Zeroable};

use crate::{Dimension, Quantity};

use super::{PS_PER_NS, PS_PER_S, TimeMicros, TimeMillis, TimeNanos, TimeSeconds};

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
pub struct TimePicos(u64);

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
pub struct TimePicosDelta(i64);

impl TimePicos {
    pub const ZERO: Self = Self(0);

    pub const fn new(ps: u64) -> Self {
        Self(ps)
    }

    pub const fn as_picos(self) -> u64 {
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

    pub fn to_millis(self) -> TimeMillis {
        self.to_seconds().to_millis()
    }

    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds::new(self.0 as f64 / PS_PER_S)
    }
}

impl TimePicosDelta {
    pub const ZERO: Self = Self(0);

    pub const fn new(ps: i64) -> Self {
        Self(ps)
    }

    pub const fn as_picos(self) -> i64 {
        self.0
    }

    #[allow(clippy::cast_possible_wrap)]
    pub fn between(before: TimePicos, after: TimePicos) -> Self {
        Self(after.as_picos() as i64 - before.as_picos() as i64)
    }
}

impl Quantity for TimePicos {
    type Repr = u64;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "ps";
    fn raw(self) -> u64 {
        self.0
    }
    fn from_raw(v: u64) -> Self {
        Self(v)
    }
}

impl Quantity for TimePicosDelta {
    type Repr = i64;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "ps (signed)";
    fn raw(self) -> i64 {
        self.0
    }
    fn from_raw(v: i64) -> Self {
        Self(v)
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
