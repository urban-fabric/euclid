//! Unit newtypes over any [`Real`] carrier.
//!
//! Each unit is a thin newtype over a [`Real`] carrier `T` with the same shape: the
//! struct, its `Quantity`/`LinearQuantity` impls, and the additive operator set
//! (`Add`/`Sub`/`Neg`/`AddAssign`/`SubAssign`, plus carrier scaling `Mul<T>`). The operator
//! impls are written out per unit because Rust's orphan rule forbids a blanket
//! `impl<Q: LinearQuantity> Add for Q`. Cross-unit products (length × length = area, …)
//! live in [`dimensional`](super::dimensional).

use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

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

impl<T: Real> AddAssign for Angle<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() + rhs.0;
    }
}

impl<T: Real> SubAssign for Angle<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() - rhs.0;
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for Angle<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}

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

impl<T: Real> AddAssign for Seconds<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() + rhs.0;
    }
}

impl<T: Real> SubAssign for Seconds<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() - rhs.0;
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for Seconds<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}

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

impl<T: Real> AddAssign for SpeedMs<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() + rhs.0;
    }
}

impl<T: Real> SubAssign for SpeedMs<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() - rhs.0;
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for SpeedMs<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}

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

impl<T: Real> AddAssign for SquareMeters<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() + rhs.0;
    }
}

impl<T: Real> SubAssign for SquareMeters<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() - rhs.0;
    }
}

/// Scale by a bare carrier factor.
impl<T: Real> Mul<T> for SquareMeters<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self(self.0 * k)
    }
}

/// Meters per kilometer.
const M_PER_KM: f64 = 1_000.0;

impl<T: crate::Real> Km<T> {
    /// Convert to meters, lifting the `1000` factor into the carrier.
    pub fn to_meters(self) -> Meters<T> {
        let factor = self.0.scalar_like(M_PER_KM);
        Meters(self.0 * factor)
    }
}
