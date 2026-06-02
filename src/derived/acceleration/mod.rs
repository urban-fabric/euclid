use bytemuck::{Pod, Zeroable};

use super::{Dimension, LinearQuantity, Quantity};

pub(crate) const MPS2_PER_CMPS2: f64 = 0.01;

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
pub struct AccelerationMetersPerSecondSquared(f64);

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
pub struct AccelerationCentimetersPerSecondSquared(f64);

impl AccelerationMetersPerSecondSquared {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_meters_per_second_squared(self) -> f64 {
        self.0
    }

    pub fn to_centimeters_per_second_squared(self) -> AccelerationCentimetersPerSecondSquared {
        AccelerationCentimetersPerSecondSquared(self.0 / MPS2_PER_CMPS2)
    }
}

impl AccelerationCentimetersPerSecondSquared {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_centimeters_per_second_squared(self) -> f64 {
        self.0
    }

    pub fn to_meters_per_second_squared(self) -> AccelerationMetersPerSecondSquared {
        AccelerationMetersPerSecondSquared(self.0 * MPS2_PER_CMPS2)
    }
}

impl Quantity for AccelerationMetersPerSecondSquared {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Acceleration;
    const UNIT_NAME: &'static str = "m/s^2";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for AccelerationMetersPerSecondSquared {}

impl Quantity for AccelerationCentimetersPerSecondSquared {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Acceleration;
    const UNIT_NAME: &'static str = "cm/s^2";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for AccelerationCentimetersPerSecondSquared {}

#[cfg(test)]
mod tests;
