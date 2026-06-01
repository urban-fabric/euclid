use bytemuck::{Pod, Zeroable};

use super::{Dimension, Quantity};

const S_PER_MIN: f64 = 60.0;
const S_PER_HOUR: f64 = 3_600.0;
const MS_PER_S: f64 = 1_000.0;

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct TimeSeconds(f64);

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct TimeMinutes(f64);

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct TimeHours(f64);

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct TimeMillis(f64);

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

impl TimeSeconds {
    pub const ZERO: Self = Self(0.0);

    pub const INFINITY: Self = Self(f64::INFINITY);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_seconds(self) -> f64 {
        self.0
    }

    pub fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    pub fn is_greater_than_epsilon(self, epsilon: f64) -> bool {
        self.0 > epsilon
    }

    pub fn approx_eq(self, other: Self, tolerance: f64) -> bool {
        (self.0 - other.0).abs() <= tolerance
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    pub fn assert_positive(self) -> Self {
        debug_assert!(self.0 > 0.0, "expected positive TimeSeconds, got {self:?}");
        self
    }

    pub fn to_minutes(self) -> TimeMinutes {
        TimeMinutes(self.0 / S_PER_MIN)
    }

    pub fn to_hours(self) -> TimeHours {
        TimeHours(self.0 / S_PER_HOUR)
    }
}

impl TimeMinutes {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_minutes(self) -> f64 {
        self.0
    }

    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds(self.0 * S_PER_MIN)
    }
}

impl TimeHours {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_hours(self) -> f64 {
        self.0
    }

    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds(self.0 * S_PER_HOUR)
    }
}

impl TimeMillis {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_millis(self) -> f64 {
        self.0
    }

    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds(self.0 / MS_PER_S)
    }

    pub fn from_duration(d: std::time::Duration) -> Self {
        Self(d.as_secs_f64() * MS_PER_S)
    }
}

impl TimeSeconds {
    pub fn to_millis(self) -> TimeMillis {
        TimeMillis(self.0 * MS_PER_S)
    }
}

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
        TimeSeconds(self.0 as f64 / 1_000_000.0)
    }

    pub fn to_millis(self) -> TimeMillis {
        TimeMillis(self.0 as f64 / MS_PER_S)
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

impl Quantity for TimeSeconds {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "s";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl Quantity for TimeMinutes {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "min";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl Quantity for TimeMillis {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "ms";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
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

impl Quantity for TimeHours {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "h";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}
#[cfg(test)]
mod tests;
