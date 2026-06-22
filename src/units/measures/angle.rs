use core::ops::{Add, Mul, Neg, Sub};

use crate::units::{Dimension, LinearQuantity, Quantity};
use crate::Real;

/// Angle in radians.
#[derive(Clone, Debug, PartialEq)]
pub struct Angle<T = f64>(pub(crate) T);

impl<T: Real> Angle<T> {
    /// Wrap a carrier value in radians.
    pub fn from_radians(radians: T) -> Self {
        Self(radians)
    }

    /// Wrap a carrier value in degrees, converting to radians.
    pub fn from_degrees(degrees: T) -> Self {
        Self(degrees.to_radians())
    }

    /// Zero radians.
    pub fn zero() -> Self {
        Self(T::zero())
    }

    /// `π` radians.
    pub fn pi() -> Self {
        Self(T::pi())
    }

    /// `2π` radians.
    pub fn tau() -> Self {
        Self(T::tau())
    }

    /// Positive infinity, useful as an unbounded angular-distance sentinel.
    pub fn infinity() -> Self {
        Self(T::infinity())
    }

    /// Unwrap to the carrier value in radians.
    pub fn value(self) -> T {
        self.0
    }

    /// The angle in radians.
    pub fn radians(&self) -> T {
        self.0.clone()
    }

    /// The angle in degrees.
    pub fn degrees(&self) -> T {
        self.0.clone().to_degrees()
    }

    /// The absolute value.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Sine of the angle.
    pub fn sin(self) -> T {
        self.0.sin()
    }

    /// Cosine of the angle.
    pub fn cos(self) -> T {
        self.0.cos()
    }

    /// Tangent of the angle.
    pub fn tan(self) -> T {
        self.0.tan()
    }

    /// Normalize into `[0, 2π)`.
    pub fn normalized_unsigned(self) -> Self {
        Self(self.0.rem_euclid(T::tau()))
    }

    /// Normalize into `(-π, π]`.
    pub fn normalized_signed(self) -> Self {
        let tau = T::tau();
        let pi = T::pi();
        let r = self.0.rem_euclid(tau.clone());
        let shifted = r.clone() - tau;
        let over_pi = r.gt(&pi);
        Self(T::select(&over_pi, shifted, r))
    }
}

impl<T: Real> Quantity for Angle<T> {
    type Repr = T;
    const DIMENSION: Dimension = Dimension::Angle;
    const UNIT_NAME: &'static str = "rad";
    fn raw(&self) -> T {
        self.0.clone()
    }
    fn from_raw(v: T) -> Self {
        Self(v)
    }
}

impl<T: Real> LinearQuantity for Angle<T> {}

impl<T: Real> Add for Angle<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<T: Real> Sub for Angle<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<T: Real> Neg for Angle<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for Angle<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}
