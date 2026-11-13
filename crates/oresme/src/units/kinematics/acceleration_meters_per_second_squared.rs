use super::MPS2_PER_CMPS2;
use crate::conversion::dim;
use crate::{
    AccelerationCentimetersPerSecondSquared, ForceNewtons, JerkMetersPerSecondCubed, MassKilograms,
    SpeedMs, TimeSeconds,
};
use crate::{Dimension, LinearQuantity, Quantity};
use bytemuck::{Pod, Zeroable};
use euclid_units::Acceleration;

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
pub struct AccelerationMetersPerSecondSquared(pub(crate) f64);

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

impl crate::conversion::UnitOfMeasure for AccelerationMetersPerSecondSquared {
    type Dim = dim::Acceleration;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <AccelerationMetersPerSecondSquared as crate::Quantity>::DIMENSION
        .same(<dim::Acceleration as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(AccelerationMetersPerSecondSquared)
    ),
);

impl core::fmt::Display for AccelerationMetersPerSecondSquared {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for AccelerationMetersPerSecondSquared {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension =
        <AccelerationMetersPerSecondSquared as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str =
        <AccelerationMetersPerSecondSquared as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for AccelerationMetersPerSecondSquared {
    fn min(self, other: Self) -> Self {
        <Self as crate::Quantity>::from_raw(
            <Self as crate::Quantity>::raw(self).min(<Self as crate::Quantity>::raw(other)),
        )
    }
    fn max(self, other: Self) -> Self {
        <Self as crate::Quantity>::from_raw(
            <Self as crate::Quantity>::raw(self).max(<Self as crate::Quantity>::raw(other)),
        )
    }
    fn clamp(self, lo: Self, hi: Self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).clamp(
            <Self as crate::Quantity>::raw(lo),
            <Self as crate::Quantity>::raw(hi),
        ))
    }
    fn floor(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).floor())
    }
    fn ceil(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).ceil())
    }
    fn round(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).round())
    }
    fn trunc(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).trunc())
    }
    fn is_finite(&self) -> bool {
        <Self as crate::Quantity>::raw(*self).is_finite()
    }
    fn is_nan(&self) -> bool {
        <Self as crate::Quantity>::raw(*self).is_nan()
    }
    fn is_infinite(&self) -> bool {
        <Self as crate::Quantity>::raw(*self).is_infinite()
    }
    fn approx_eq(&self, other: &Self, tolerance: &Self) -> bool {
        (<Self as crate::Quantity>::raw(*self) - <Self as crate::Quantity>::raw(*other)).abs()
            <= <Self as crate::Quantity>::raw(*tolerance).abs()
    }
}

impl euclid_units::LinearQuantity for AccelerationMetersPerSecondSquared {
    fn add(self, other: Self) -> Self {
        <Self as crate::Quantity>::from_raw(
            <Self as crate::Quantity>::raw(self) + <Self as crate::Quantity>::raw(other),
        )
    }
    fn sub(self, other: Self) -> Self {
        <Self as crate::Quantity>::from_raw(
            <Self as crate::Quantity>::raw(self) - <Self as crate::Quantity>::raw(other),
        )
    }
    fn neg(self) -> Self {
        <Self as crate::Quantity>::from_raw(-<Self as crate::Quantity>::raw(self))
    }
    fn abs(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).abs())
    }
    fn signum(self) -> f64 {
        <Self as crate::Quantity>::raw(self).signum()
    }
    fn ratio(self, other: Self) -> f64 {
        <Self as crate::Quantity>::raw(self) / <Self as crate::Quantity>::raw(other)
    }
    fn lerp(self, other: Self, t: f64) -> Self {
        let lhs = <Self as crate::Quantity>::raw(self);
        let rhs = <Self as crate::Quantity>::raw(other);
        <Self as crate::Quantity>::from_raw(lhs * (1.0 - t) + rhs * t)
    }
    fn scale(self, factor: f64) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self) * factor)
    }
    fn scaled(self, k: f64) -> Self {
        euclid_units::LinearQuantity::scale(self, k)
    }
}

impl Acceleration for AccelerationMetersPerSecondSquared {
    type Speed = SpeedMs;
    type Time = TimeSeconds;
    type Force = ForceNewtons;
    type Mass = MassKilograms;
    type Jerk = JerkMetersPerSecondCubed;

    fn times_time(self, elapsed: Self::Time) -> Self::Speed {
        SpeedMs::from_raw(self.raw() * elapsed.raw())
    }
    fn times_mass(self, mass: Self::Mass) -> Self::Force {
        ForceNewtons::from_raw(self.raw() * mass.raw())
    }
    fn per_time(self, elapsed: Self::Time) -> Self::Jerk {
        JerkMetersPerSecondCubed::from_raw(self.raw() / elapsed.raw())
    }
}

impl ::core::ops::Add for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for AccelerationMetersPerSecondSquared {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for AccelerationMetersPerSecondSquared {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<AccelerationMetersPerSecondSquared> for f64 {
    type Output = AccelerationMetersPerSecondSquared;
    fn mul(self, x: AccelerationMetersPerSecondSquared) -> AccelerationMetersPerSecondSquared {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for AccelerationMetersPerSecondSquared {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
