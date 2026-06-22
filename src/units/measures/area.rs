use core::ops::{Add, Mul, Neg, Sub};

use crate::units::{Dimension, LinearQuantity, Quantity};
use crate::Real;

/// Area in square meters.
#[derive(Clone, Debug, PartialEq)]
pub struct SquareMeters<T = f64>(pub(crate) T);

impl<T: Real> SquareMeters<T> {
    /// Wrap a carrier value.
    pub fn new(value: T) -> Self {
        Self(value)
    }
    /// Unwrap to the carrier.
    pub fn value(self) -> T {
        self.0
    }
}

impl<T: Real> Quantity for SquareMeters<T> {
    type Repr = T;
    const DIMENSION: Dimension = Dimension::Area;
    const UNIT_NAME: &'static str = "m²";
    fn raw(&self) -> T {
        self.0.clone()
    }
    fn from_raw(v: T) -> Self {
        Self(v)
    }
}

impl<T: Real> LinearQuantity for SquareMeters<T> {}

impl<T: Real> Add for SquareMeters<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<T: Real> Sub for SquareMeters<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<T: Real> Neg for SquareMeters<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for SquareMeters<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}
