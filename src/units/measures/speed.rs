use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::units::{Dimension, LinearQuantity, Quantity};
use crate::Real;

/// Speed in meters per second.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeedMs<T = f64>(pub(crate) T);

impl<T: Real> SpeedMs<T> {
    /// Wrap a carrier value.
    pub fn new(value: T) -> Self {
        Self(value)
    }
    /// Unwrap to the carrier.
    pub fn value(self) -> T {
        self.0
    }
}

impl<T: Real> Quantity for SpeedMs<T> {
    type Repr = T;
    const DIMENSION: Dimension = Dimension::Speed;
    const UNIT_NAME: &'static str = "m/s";
    fn raw(&self) -> T {
        self.0.clone()
    }
    fn from_raw(v: T) -> Self {
        Self(v)
    }
}

impl<T: Real> LinearQuantity for SpeedMs<T> {}

impl<T: Real> Add for SpeedMs<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<T: Real> Sub for SpeedMs<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<T: Real> Neg for SpeedMs<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for SpeedMs<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}
