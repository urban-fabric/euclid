use super::{M_PER_FOOT, M_PER_KM};
use crate::conversion::dim;
use crate::{AreaSquareMeters, LengthFeet, LengthKm, SpeedMs, TimeSeconds, VolumeCubicMeters};
use crate::{Dimension, LinearQuantity, Quantity};
use bytemuck::{Pod, Zeroable};
use euclid_units::Length;

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
pub struct LengthMeters(pub(crate) f64);

impl LengthMeters {
    pub const ZERO: Self = Self(0.0);

    pub const INFINITY: Self = Self(f64::INFINITY);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_meters(self) -> f64 {
        self.0
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn to_f32_meters(self) -> f32 {
        self.0 as f32
    }

    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    pub fn is_greater_than_epsilon(self, epsilon: f64) -> bool {
        self.0 > epsilon
    }

    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    pub fn assert_positive(self) -> Self {
        debug_assert!(self.0 > 0.0, "expected positive LengthMeters, got {self:?}");
        self
    }
}

impl Quantity for LengthMeters {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Length;
    const UNIT_NAME: &'static str = "m";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for LengthMeters {}

impl LengthMeters {
    pub fn to_km(self) -> LengthKm {
        LengthKm(self.0 / M_PER_KM)
    }

    pub fn to_feet(self) -> LengthFeet {
        LengthFeet(self.0 / M_PER_FOOT)
    }
}

impl crate::conversion::UnitOfMeasure for LengthMeters {
    type Dim = dim::Length;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <LengthMeters as crate::Quantity>::DIMENSION
        .same(<dim::Length as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(LengthMeters)),
);

impl core::fmt::Display for LengthMeters {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for LengthMeters {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = <LengthMeters as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str = <LengthMeters as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for LengthMeters {
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

impl euclid_units::LinearQuantity for LengthMeters {
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

impl Length for LengthMeters {
    type Area = AreaSquareMeters;
    type Volume = VolumeCubicMeters;
    type Time = TimeSeconds;
    type Speed = SpeedMs;

    fn times_length(self, width: Self) -> Self::Area {
        AreaSquareMeters::from_raw(self.raw() * width.raw())
    }
    fn times_area(self, base: Self::Area) -> Self::Volume {
        VolumeCubicMeters::from_raw(self.raw() * base.raw())
    }
    fn per_time(self, elapsed: Self::Time) -> Self::Speed {
        SpeedMs::from_raw(self.raw() / elapsed.raw())
    }
    fn per_speed(self, speed: Self::Speed) -> Self::Time {
        TimeSeconds::from_raw(self.raw() / speed.raw())
    }
    fn squared(self) -> Self::Area {
        AreaSquareMeters::from_raw(self.raw() * self.raw())
    }
    fn cubed(self) -> Self::Volume {
        VolumeCubicMeters::from_raw(self.raw() * self.raw() * self.raw())
    }
}

impl ::core::ops::Add for LengthMeters {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for LengthMeters {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for LengthMeters {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for LengthMeters {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for LengthMeters {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for LengthMeters {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<LengthMeters> for f64 {
    type Output = LengthMeters;
    fn mul(self, x: LengthMeters) -> LengthMeters {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for LengthMeters {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for LengthMeters {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
