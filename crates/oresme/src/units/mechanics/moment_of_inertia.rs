use crate::Quantity;
use crate::conversion::dim;
use crate::{
    AngularAccelerationRadiansPerSecondSquared, AngularMomentumKilogramMetersSquaredPerSecond,
    AngularVelocityRadiansPerSecond, AreaSquareMeters, MassKilograms, TorqueNewtonMeters,
};
use euclid_units::MomentOfInertia;

/// Moment of inertia in kilogram-meters squared (kg.m^2), the coherent SI unit.
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
pub struct MomentOfInertiaKilogramMetersSquared(pub(crate) f64);

impl MomentOfInertiaKilogramMetersSquared {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_kilogram_meters_squared(self) -> f64 {
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

impl crate::Quantity for MomentOfInertiaKilogramMetersSquared {
    type Repr = f64;
    const DIMENSION: crate::Dimension = crate::Dimension::MomentOfInertia;
    const UNIT_NAME: &'static str = "kg.m^2";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl crate::LinearQuantity for MomentOfInertiaKilogramMetersSquared {}

impl crate::conversion::UnitOfMeasure for MomentOfInertiaKilogramMetersSquared {
    type Dim = dim::MomentOfInertia;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <MomentOfInertiaKilogramMetersSquared as crate::Quantity>::DIMENSION
        .same(<dim::MomentOfInertia as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(MomentOfInertiaKilogramMetersSquared)
    ),
);

impl core::fmt::Display for MomentOfInertiaKilogramMetersSquared {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for MomentOfInertiaKilogramMetersSquared {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension =
        <MomentOfInertiaKilogramMetersSquared as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str =
        <MomentOfInertiaKilogramMetersSquared as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for MomentOfInertiaKilogramMetersSquared {
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

impl euclid_units::LinearQuantity for MomentOfInertiaKilogramMetersSquared {
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

impl MomentOfInertia for MomentOfInertiaKilogramMetersSquared {
    type Mass = MassKilograms;
    type Area = AreaSquareMeters;
    type AngularVelocity = AngularVelocityRadiansPerSecond;
    type AngularMomentum = AngularMomentumKilogramMetersSquaredPerSecond;
    type AngularAcceleration = AngularAccelerationRadiansPerSecondSquared;
    type Torque = TorqueNewtonMeters;

    fn per_mass(self, mass: Self::Mass) -> Self::Area {
        AreaSquareMeters::from_raw(self.raw() / mass.raw())
    }
    fn times_angular_velocity(self, omega: Self::AngularVelocity) -> Self::AngularMomentum {
        AngularMomentumKilogramMetersSquaredPerSecond::from_raw(self.raw() * omega.raw())
    }
    fn times_angular_acceleration(self, alpha: Self::AngularAcceleration) -> Self::Torque {
        TorqueNewtonMeters::from_raw(self.raw() * alpha.raw())
    }
}

impl ::core::ops::Add for MomentOfInertiaKilogramMetersSquared {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for MomentOfInertiaKilogramMetersSquared {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for MomentOfInertiaKilogramMetersSquared {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for MomentOfInertiaKilogramMetersSquared {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for MomentOfInertiaKilogramMetersSquared {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for MomentOfInertiaKilogramMetersSquared {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<MomentOfInertiaKilogramMetersSquared> for f64 {
    type Output = MomentOfInertiaKilogramMetersSquared;
    fn mul(self, x: MomentOfInertiaKilogramMetersSquared) -> MomentOfInertiaKilogramMetersSquared {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for MomentOfInertiaKilogramMetersSquared {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for MomentOfInertiaKilogramMetersSquared {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
