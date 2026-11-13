use crate::Quantity;
use crate::conversion::dim;
use crate::{
    AngularAccelerationRadiansPerSecondSquared, AngularMomentumKilogramMetersSquaredPerSecond,
    AngularVelocityRadiansPerSecond, ForceNewtons, LengthMeters,
    MomentOfInertiaKilogramMetersSquared, PowerWatts, TimeSeconds,
};
use euclid_units::Torque;

/// Torque in newton-meters (N.m), the coherent SI unit.
#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct TorqueNewtonMeters(pub(crate) f64);

impl TorqueNewtonMeters {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_newton_meters(self) -> f64 {
        self.0
    }

    /// Whether the value is finite (neither infinite nor `NaN`).
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Whether the value is strictly greater than zero.
    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    /// Whether the value is greater than or equal to zero.
    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    /// Whether the value is exactly zero.
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// The absolute value, in the same unit.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl crate::Quantity for TorqueNewtonMeters {
    type Repr = f64;
    const DIMENSION: crate::Dimension = crate::Dimension::Torque;
    const UNIT_NAME: &'static str = "N.m";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl crate::LinearQuantity for TorqueNewtonMeters {}

impl crate::conversion::UnitOfMeasure for TorqueNewtonMeters {
    type Dim = dim::Torque;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <TorqueNewtonMeters as crate::Quantity>::DIMENSION
        .same(<dim::Torque as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(TorqueNewtonMeters)),
);

impl core::fmt::Display for TorqueNewtonMeters {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for TorqueNewtonMeters {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = <TorqueNewtonMeters as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str = <TorqueNewtonMeters as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for TorqueNewtonMeters {
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

impl euclid_units::LinearQuantity for TorqueNewtonMeters {
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

impl Torque for TorqueNewtonMeters {
    type Length = LengthMeters;
    type Force = ForceNewtons;
    type AngularVelocity = AngularVelocityRadiansPerSecond;
    type Power = PowerWatts;
    type Time = TimeSeconds;
    type AngularMomentum = AngularMomentumKilogramMetersSquaredPerSecond;
    type MomentOfInertia = MomentOfInertiaKilogramMetersSquared;
    type AngularAcceleration = AngularAccelerationRadiansPerSecondSquared;

    fn per_length(self, lever: Self::Length) -> Self::Force {
        ForceNewtons::from_raw(self.raw() / lever.raw())
    }
    fn times_angular_velocity(self, omega: Self::AngularVelocity) -> Self::Power {
        PowerWatts::from_raw(self.raw() * omega.raw())
    }
    fn times_time(self, elapsed: Self::Time) -> Self::AngularMomentum {
        AngularMomentumKilogramMetersSquaredPerSecond::from_raw(self.raw() * elapsed.raw())
    }
    fn per_moment_of_inertia(self, inertia: Self::MomentOfInertia) -> Self::AngularAcceleration {
        AngularAccelerationRadiansPerSecondSquared::from_raw(self.raw() / inertia.raw())
    }
}

impl ::core::ops::Add for TorqueNewtonMeters {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for TorqueNewtonMeters {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for TorqueNewtonMeters {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for TorqueNewtonMeters {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for TorqueNewtonMeters {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for TorqueNewtonMeters {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<TorqueNewtonMeters> for f64 {
    type Output = TorqueNewtonMeters;
    fn mul(self, x: TorqueNewtonMeters) -> TorqueNewtonMeters {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for TorqueNewtonMeters {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for TorqueNewtonMeters {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
