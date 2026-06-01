use bytemuck::{Pod, Zeroable};

use super::{Dimension, Quantity};

const MS_PER_KPH: f64 = 1_000.0 / 3_600.0;
const MS_PER_MPH: f64 = 0.447_04;
const KPH_PER_MPH: f64 = 1.609_344;
const MPH_PER_KPH: f64 = 1.0 / KPH_PER_MPH;

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
pub struct SpeedMs(f64);

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
pub struct SpeedKph(f64);

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
pub struct SpeedMph(f64);

impl SpeedMs {
    pub const ZERO: Self = Self(0.0);

    pub const INFINITY: Self = Self(f64::INFINITY);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_ms(self) -> f64 {
        self.0
    }

    pub fn to_kph(self) -> SpeedKph {
        SpeedKph(self.0 / MS_PER_KPH)
    }

    pub fn to_mph(self) -> SpeedMph {
        SpeedMph(self.0 / MS_PER_MPH)
    }
}

impl SpeedKph {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    #[inline]
    #[must_use]
    pub const fn from_kph_u16(kph: u16) -> Self {
        Self(kph as f64)
    }

    pub const fn as_kph(self) -> f64 {
        self.0
    }

    pub fn to_f32_kph(self) -> f32 {
        self.0 as f32
    }

    pub fn is_finite(self) -> bool {
        self.0.is_finite()
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
        debug_assert!(self.0 > 0.0, "expected positive SpeedKph, got {self:?}");
        self
    }

    pub fn to_ms(self) -> SpeedMs {
        SpeedMs(self.0 * MS_PER_KPH)
    }

    pub fn to_mph(self) -> SpeedMph {
        SpeedMph(self.0 * MPH_PER_KPH)
    }
}

impl SpeedMph {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_mph(self) -> f64 {
        self.0
    }

    pub fn to_ms(self) -> SpeedMs {
        SpeedMs(self.0 * MS_PER_MPH)
    }

    pub fn to_kph(self) -> SpeedKph {
        SpeedKph(self.0 * KPH_PER_MPH)
    }
}

impl Quantity for SpeedMs {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Speed;
    const UNIT_NAME: &'static str = "m/s";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl Quantity for SpeedKph {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Speed;
    const UNIT_NAME: &'static str = "km/h";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl Quantity for SpeedMph {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Speed;
    const UNIT_NAME: &'static str = "mph";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}
#[cfg(test)]
mod tests;
