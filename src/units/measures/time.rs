use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::units::{Dimension, LinearQuantity, Quantity};
use crate::Real;

/// Time in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct Seconds<T = f64>(pub(crate) T);

impl<T: Real> Seconds<T> {
    /// Wrap a carrier value.
    pub fn new(value: T) -> Self {
        Self(value)
    }
    /// Unwrap to the carrier.
    pub fn value(self) -> T {
        self.0
    }
}

impl<T: Real> Quantity for Seconds<T> {
    type Repr = T;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "s";
    fn raw(&self) -> T {
        self.0.clone()
    }
    fn from_raw(v: T) -> Self {
        Self(v)
    }
}

impl<T: Real> LinearQuantity for Seconds<T> {}

impl<T: Real> Add for Seconds<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<T: Real> Sub for Seconds<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<T: Real> Neg for Seconds<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for Seconds<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}
