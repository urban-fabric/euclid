use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::units::{Dimension, LinearQuantity, Quantity};
use crate::Real;

/// Length in meters.
#[derive(Clone, Debug, PartialEq)]
pub struct Meters<T = f64>(pub(crate) T);

impl<T: Real> Meters<T> {
    /// Wrap a carrier value.
    pub fn new(value: T) -> Self {
        Self(value)
    }
    /// Unwrap to the carrier value.
    pub fn value(self) -> T {
        self.0
    }
}

impl<T: Real> Quantity for Meters<T> {
    type Repr = T;
    const DIMENSION: Dimension = Dimension::Length;
    const UNIT_NAME: &'static str = "m";
    fn raw(&self) -> T {
        self.0.clone()
    }
    fn from_raw(v: T) -> Self {
        Self(v)
    }
}

impl<T: Real> LinearQuantity for Meters<T> {}

impl<T: Real> Add for Meters<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<T: Real> Sub for Meters<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<T: Real> Neg for Meters<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<T: Real> AddAssign for Meters<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() + rhs.0;
    }
}

impl<T: Real> SubAssign for Meters<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() - rhs.0;
    }
}

/// Scale by a bare carrier factor (`length * gain`).
impl<T: Real> Mul<T> for Meters<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}

/// Length in kilometers.
#[derive(Clone, Debug, PartialEq)]
pub struct Km<T = f64>(pub(crate) T);

impl<T: Real> Km<T> {
    /// Wrap a carrier value.
    pub fn new(value: T) -> Self {
        Self(value)
    }
    /// Unwrap to the carrier.
    pub fn value(self) -> T {
        self.0
    }
}

impl<T: Real> Quantity for Km<T> {
    type Repr = T;
    const DIMENSION: Dimension = Dimension::Length;
    const UNIT_NAME: &'static str = "km";
    fn raw(&self) -> T {
        self.0.clone()
    }
    fn from_raw(v: T) -> Self {
        Self(v)
    }
}

impl<T: Real> LinearQuantity for Km<T> {}

impl<T: Real> Add for Km<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<T: Real> Sub for Km<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<T: Real> Neg for Km<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<T: Real> AddAssign for Km<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() + rhs.0;
    }
}

impl<T: Real> SubAssign for Km<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() - rhs.0;
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for Km<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}

/// Meters per kilometer.
const M_PER_KM: f64 = 1_000.0;

impl<T: Real> Km<T> {
    /// Convert to meters, lifting the `1000` factor into the carrier.
    pub fn to_meters(self) -> Meters<T> {
        let factor = self.0.scalar_like(M_PER_KM);
        Meters(self.0 * factor)
    }
}
