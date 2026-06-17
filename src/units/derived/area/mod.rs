use bytemuck::{Pod, Zeroable};

use super::{Dimension, LinearQuantity, Quantity};

pub(crate) const SQUARE_METERS_PER_SQUARE_KILOMETER: f64 = 1_000_000.0;

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
pub struct AreaSquareMeters(f64);

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
pub struct AreaSquareKilometers(f64);

impl AreaSquareMeters {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_square_meters(self) -> f64 {
        self.0
    }

    pub fn to_square_kilometers(self) -> AreaSquareKilometers {
        AreaSquareKilometers(self.0 / SQUARE_METERS_PER_SQUARE_KILOMETER)
    }
}

impl AreaSquareKilometers {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_square_kilometers(self) -> f64 {
        self.0
    }

    pub fn to_square_meters(self) -> AreaSquareMeters {
        AreaSquareMeters(self.0 * SQUARE_METERS_PER_SQUARE_KILOMETER)
    }
}

impl Quantity for AreaSquareMeters {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Area;
    const UNIT_NAME: &'static str = "m^2";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for AreaSquareMeters {}

impl Quantity for AreaSquareKilometers {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Area;
    const UNIT_NAME: &'static str = "km^2";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for AreaSquareKilometers {}

#[cfg(test)]
mod tests;
