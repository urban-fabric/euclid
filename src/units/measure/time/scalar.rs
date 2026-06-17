use bytemuck::{Pod, Zeroable};

use crate::{Dimension, LinearQuantity, Quantity};

use super::{MS_PER_S, S_PER_HOUR, S_PER_MIN};

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

    pub fn to_millis(self) -> TimeMillis {
        TimeMillis(self.0 * MS_PER_S)
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

impl LinearQuantity for TimeSeconds {}

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

impl LinearQuantity for TimeMinutes {}

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

impl LinearQuantity for TimeHours {}

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

impl LinearQuantity for TimeMillis {}
