use bytemuck::{Pod, Zeroable};

use crate::{Dimension, Quantity};

use super::{MS_PER_S, TimeMillis, TimeSeconds};

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
pub struct TimeMicros(u64);

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
pub struct TimeMicrosDelta(i64);

impl TimeMicros {
    pub const ZERO: Self = Self(0);

    pub const fn new(us: u64) -> Self {
        Self(us)
    }

    pub const fn as_micros(self) -> u64 {
        self.0
    }

    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    pub fn from_duration(d: std::time::Duration) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self(d.as_micros() as u64)
    }

    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds::new(self.0 as f64 / 1_000_000.0)
    }

    pub fn to_millis(self) -> TimeMillis {
        TimeMillis::new(self.0 as f64 / MS_PER_S)
    }
}

impl TimeMicrosDelta {
    pub const ZERO: Self = Self(0);

    pub const fn new(us: i64) -> Self {
        Self(us)
    }

    pub const fn as_micros(self) -> i64 {
        self.0
    }

    #[allow(clippy::cast_possible_wrap)]
    pub fn between(before: TimeMicros, after: TimeMicros) -> Self {
        Self(after.as_micros() as i64 - before.as_micros() as i64)
    }
}

impl Quantity for TimeMicros {
    type Repr = u64;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "µs";
    fn raw(self) -> u64 {
        self.0
    }
    fn from_raw(v: u64) -> Self {
        Self(v)
    }
}

impl Quantity for TimeMicrosDelta {
    type Repr = i64;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "µs (signed)";
    fn raw(self) -> i64 {
        self.0
    }
    fn from_raw(v: i64) -> Self {
        Self(v)
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
