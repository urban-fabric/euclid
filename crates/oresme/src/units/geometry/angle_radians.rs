use crate::conversion::dim;
use crate::{AngleDegrees, AngularVelocityRadiansPerSecond, TimeSeconds};
use crate::{Dimension, LinearQuantity, Quantity};
use bytemuck::{Pod, Zeroable};
use euclid_units::Angle;

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
pub struct AngleRadians(pub(crate) f64);

impl AngleRadians {
    pub const ZERO: Self = Self(0.0);

    pub const INFINITY: Self = Self(f64::INFINITY);

    pub const PI: Self = Self(std::f64::consts::PI);

    pub const TAU: Self = Self(std::f64::consts::TAU);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub fn to_degrees(self) -> AngleDegrees {
        AngleDegrees(self.0.to_degrees())
    }

    pub const fn as_radians(self) -> f64 {
        self.0
    }

    pub fn to_f32_radians(self) -> f32 {
        self.0 as f32
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl Quantity for AngleRadians {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Angle;
    const UNIT_NAME: &'static str = "rad";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for AngleRadians {}

impl crate::conversion::UnitOfMeasure for AngleRadians {
    type Dim = dim::Angle;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <AngleRadians as crate::Quantity>::DIMENSION
        .same(<dim::Angle as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(AngleRadians)),
);

impl core::fmt::Display for AngleRadians {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for AngleRadians {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = <AngleRadians as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str = <AngleRadians as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for AngleRadians {
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

impl euclid_units::LinearQuantity for AngleRadians {
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

impl Angle for AngleRadians {
    type AngularVelocity = AngularVelocityRadiansPerSecond;
    type Time = TimeSeconds;

    fn from_radians(radians: f64) -> Self {
        AngleRadians::from_raw(radians)
    }
    fn from_degrees(degrees: f64) -> Self {
        AngleRadians::from_raw(degrees.to_radians())
    }
    fn half_turn() -> Self {
        AngleRadians::from_raw(core::f64::consts::PI)
    }
    fn full_turn() -> Self {
        AngleRadians::from_raw(core::f64::consts::TAU)
    }
    fn radians(&self) -> f64 {
        self.raw()
    }
    fn degrees(&self) -> f64 {
        self.raw().to_degrees()
    }
    fn sin(&self) -> f64 {
        self.raw().sin()
    }
    fn cos(&self) -> f64 {
        self.raw().cos()
    }
    fn tan(&self) -> f64 {
        self.raw().tan()
    }
    fn normalized_unsigned(self) -> Self {
        AngleRadians::from_raw(self.raw().rem_euclid(core::f64::consts::TAU))
    }
    fn normalized_signed(self) -> Self {
        let tau = core::f64::consts::TAU;
        let mut wrapped = self.raw().rem_euclid(tau);
        if wrapped > core::f64::consts::PI {
            wrapped -= tau;
        }
        AngleRadians::from_raw(wrapped)
    }
    fn per_time(self, elapsed: Self::Time) -> Self::AngularVelocity {
        AngularVelocityRadiansPerSecond::from_raw(self.raw() / elapsed.raw())
    }
}

impl ::core::ops::Add for AngleRadians {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for AngleRadians {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for AngleRadians {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for AngleRadians {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for AngleRadians {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for AngleRadians {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<AngleRadians> for f64 {
    type Output = AngleRadians;
    fn mul(self, x: AngleRadians) -> AngleRadians {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for AngleRadians {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for AngleRadians {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
