use bytemuck::{Pod, Zeroable};

use super::{Dimension, LinearQuantity, Quantity};

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
pub struct AngleRadians(f64);

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
pub struct AngleDegrees(f64);

impl AngleRadians {
    pub const ZERO: Self = Self(0.0);

    pub const INFINITY: Self = Self(f64::INFINITY);

    pub const PI: Self = Self(std::f64::consts::PI);

    pub const TAU: Self = Self(std::f64::consts::TAU);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub fn to_degrees(self) -> AngleDegrees {
        AngleDegrees(self.0.to_degrees())
    }

    pub const fn as_radians(self) -> f64 {
        self.0
    }

    pub fn to_f32_radians(self) -> f32 {
        self.0 as f32
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl AngleDegrees {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub fn to_radians(self) -> AngleRadians {
        AngleRadians(self.0.to_radians())
    }

    pub const fn as_degrees(self) -> f64 {
        self.0
    }
}

impl Quantity for AngleRadians {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Angle;
    const UNIT_NAME: &'static str = "rad";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for AngleRadians {}

impl Quantity for AngleDegrees {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Angle;
    const UNIT_NAME: &'static str = "deg";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for AngleDegrees {}
#[cfg(test)]
mod tests;
