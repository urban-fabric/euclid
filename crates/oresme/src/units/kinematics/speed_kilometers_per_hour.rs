use super::{MPH_PER_KPH, MS_PER_KPH};
use crate::conversion::dim;
use crate::{Dimension, LinearQuantity, Quantity};
use crate::{SpeedMph, SpeedMs};
use bytemuck::{Pod, Zeroable};

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
pub struct SpeedKph(pub(crate) f64);

impl SpeedKph {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    #[inline]
    #[must_use]
    pub const fn from_kph_u16(kph: u16) -> Self {
        Self(kph as f64)
    }

    pub const fn as_kph(self) -> f64 {
        self.0
    }

    pub fn to_f32_kph(self) -> f32 {
        self.0 as f32
    }

    pub fn is_finite(self) -> bool {
        self.0.is_finite()
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

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    pub fn assert_positive(self) -> Self {
        debug_assert!(self.0 > 0.0, "expected positive SpeedKph, got {self:?}");
        self
    }

    pub fn to_ms(self) -> SpeedMs {
        SpeedMs(self.0 * MS_PER_KPH)
    }

    pub fn to_mph(self) -> SpeedMph {
        SpeedMph(self.0 * MPH_PER_KPH)
    }
}

impl Quantity for SpeedKph {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Speed;
    const UNIT_NAME: &'static str = "km/h";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for SpeedKph {}

impl crate::conversion::UnitOfMeasure for SpeedKph {
    type Dim = dim::Speed;
    const TO_PRIMARY: f64 = MS_PER_KPH;
}

const _: () = assert!(
    <SpeedKph as crate::Quantity>::DIMENSION
        .same(<dim::Speed as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(SpeedKph)),
);

impl core::fmt::Display for SpeedKph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for SpeedKph {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = <SpeedKph as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str = <SpeedKph as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for SpeedKph {
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

impl euclid_units::LinearQuantity for SpeedKph {
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

impl ::core::ops::Add for SpeedKph {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for SpeedKph {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for SpeedKph {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for SpeedKph {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for SpeedKph {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for SpeedKph {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<SpeedKph> for f64 {
    type Output = SpeedKph;
    fn mul(self, x: SpeedKph) -> SpeedKph {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for SpeedKph {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for SpeedKph {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
