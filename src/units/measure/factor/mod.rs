use std::ops::{Div, Mul};

use bytemuck::{Pod, Zeroable};

use super::{Dimension, LinearQuantity, Quantity};

#[repr(transparent)]
#[derive(
    Copy, Clone, Debug, PartialEq, PartialOrd, Pod, Zeroable, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct Factor(f64);

impl Default for Factor {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Factor {
    pub const IDENTITY: Self = Self(1.0);

    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    #[inline]
    #[must_use]
    pub const fn as_f64(self) -> f64 {
        self.0
    }

    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    #[inline]
    #[must_use]
    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    #[inline]
    #[must_use]
    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    #[inline]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    #[inline]
    #[must_use]
    pub fn is_greater_than_epsilon(self, epsilon: f64) -> bool {
        self.0 > epsilon
    }

    #[inline]
    #[must_use]
    pub fn clamped(self, min: f64, max: f64) -> Self {
        Self(self.0.clamp(min, max))
    }

    #[inline]
    #[must_use]
    pub fn non_negative(self) -> Self {
        Self(self.0.max(0.0))
    }
}

impl Quantity for Factor {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Dimensionless;
    const UNIT_NAME: &'static str = "";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for Factor {}

impl Mul for Factor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}
impl Div for Factor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}
#[cfg(test)]
mod tests;
