use bytemuck::{Pod, Zeroable};

use super::{Dimension, Quantity};

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
pub struct LengthMeters(f64);

impl LengthMeters {
    pub const ZERO: Self = Self(0.0);

    pub const INFINITY: Self = Self(f64::INFINITY);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_meters(self) -> f64 {
        self.0
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn to_f32_meters(self) -> f32 {
        self.0 as f32
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

    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    pub fn approx_eq(self, other: Self, tolerance: f64) -> bool {
        (self.0 - other.0).abs() <= tolerance
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    pub fn assert_positive(self) -> Self {
        debug_assert!(self.0 > 0.0, "expected positive LengthMeters, got {self:?}");
        self
    }
}

impl Quantity for LengthMeters {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Length;
    const UNIT_NAME: &'static str = "m";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

const M_PER_KM: f64 = 1_000.0;
const M_PER_FOOT: f64 = 0.304_8;

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
pub struct LengthKm(f64);

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
pub struct LengthFeet(f64);

impl LengthKm {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_km(self) -> f64 {
        self.0
    }

    pub fn to_meters(self) -> LengthMeters {
        LengthMeters(self.0 * M_PER_KM)
    }
}

impl LengthFeet {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_feet(self) -> f64 {
        self.0
    }

    pub fn to_meters(self) -> LengthMeters {
        LengthMeters(self.0 * M_PER_FOOT)
    }
}

impl LengthMeters {
    pub fn to_km(self) -> LengthKm {
        LengthKm(self.0 / M_PER_KM)
    }

    pub fn to_feet(self) -> LengthFeet {
        LengthFeet(self.0 / M_PER_FOOT)
    }
}

impl Quantity for LengthKm {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Length;
    const UNIT_NAME: &'static str = "km";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl Quantity for LengthFeet {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Length;
    const UNIT_NAME: &'static str = "ft";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}
#[cfg(test)]
mod tests;
