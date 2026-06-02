use bytemuck::{Pod, Zeroable};

use super::{Dimension, Factor, LinearQuantity, Quantity};

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
pub struct Percentage(f64);

impl Percentage {
    pub const ZERO: Self = Self(0.0);

    pub const FULL: Self = Self(100.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub fn from_fraction(f: Factor) -> Self {
        Self(f.as_f64() * 100.0)
    }

    pub fn to_fraction(self) -> Factor {
        Factor::new(self.0 / 100.0)
    }

    pub const fn as_percent(self) -> f64 {
        self.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    pub fn is_in_range(self) -> bool {
        self.0.is_finite() && (0.0..=100.0).contains(&self.0)
    }
}

impl Quantity for Percentage {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Dimensionless;
    const UNIT_NAME: &'static str = "%";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for Percentage {}
#[cfg(test)]
mod tests;
