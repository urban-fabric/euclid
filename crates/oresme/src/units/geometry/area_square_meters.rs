use super::SQUARE_METERS_PER_SQUARE_KILOMETER;
use crate::conversion::dim;
use crate::{
    AreaSquareKilometers, LengthMeters, SpeedMs, VolumeCubicMeters,
    VolumetricFlowRateCubicMetersPerSecond,
};
use crate::{Dimension, LinearQuantity, Quantity};
use bytemuck::{Pod, Zeroable};
use euclid_units::Area;

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
pub struct AreaSquareMeters(pub(crate) f64);

impl AreaSquareMeters {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_square_meters(self) -> f64 {
        self.0
    }

    pub fn to_square_kilometers(self) -> AreaSquareKilometers {
        AreaSquareKilometers(self.0 / SQUARE_METERS_PER_SQUARE_KILOMETER)
    }
}

impl Quantity for AreaSquareMeters {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Area;
    const UNIT_NAME: &'static str = "m^2";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for AreaSquareMeters {}

impl crate::conversion::UnitOfMeasure for AreaSquareMeters {
    type Dim = dim::Area;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <AreaSquareMeters as crate::Quantity>::DIMENSION
        .same(<dim::Area as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(AreaSquareMeters)),
);

impl core::fmt::Display for AreaSquareMeters {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for AreaSquareMeters {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = <AreaSquareMeters as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str = <AreaSquareMeters as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for AreaSquareMeters {
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

impl euclid_units::LinearQuantity for AreaSquareMeters {
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

impl Area for AreaSquareMeters {
    type Length = LengthMeters;
    type Volume = VolumeCubicMeters;
    type VolumetricFlowRate = VolumetricFlowRateCubicMetersPerSecond;
    type Speed = SpeedMs;

    fn per_length(self, side: Self::Length) -> Self::Length {
        LengthMeters::from_raw(self.raw() / side.raw())
    }
    fn times_length(self, height: Self::Length) -> Self::Volume {
        VolumeCubicMeters::from_raw(self.raw() * height.raw())
    }
    fn times_speed(self, speed: Self::Speed) -> Self::VolumetricFlowRate {
        VolumetricFlowRateCubicMetersPerSecond::from_raw(self.raw() * speed.raw())
    }
    fn side(self) -> Self::Length {
        LengthMeters::from_raw(self.raw().sqrt())
    }
}

impl ::core::ops::Add for AreaSquareMeters {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for AreaSquareMeters {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for AreaSquareMeters {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for AreaSquareMeters {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for AreaSquareMeters {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for AreaSquareMeters {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<AreaSquareMeters> for f64 {
    type Output = AreaSquareMeters;
    fn mul(self, x: AreaSquareMeters) -> AreaSquareMeters {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for AreaSquareMeters {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for AreaSquareMeters {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
