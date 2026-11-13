use super::{KG_PER_GRAM, KG_PER_MILLIGRAM};
use crate::conversion::dim;
use crate::{
    AccelerationMetersPerSecondSquared, AmountMoles, AreaSquareMeters,
    DensityKilogramsPerCubicMeter, ForceNewtons, MassFlowRateKilogramsPerSecond, MassGrams,
    MassMilligrams, MolarMassKilogramsPerMole, MomentOfInertiaKilogramMetersSquared,
    MomentumKilogramMetersPerSecond, SpeedMs, TimeSeconds, VolumeCubicMeters,
};
use crate::{Dimension, LinearQuantity, Quantity};
use bytemuck::{Pod, Zeroable};
use euclid_units::Mass;

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
pub struct MassKilograms(pub(crate) f64);

impl MassKilograms {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_kilograms(self) -> f64 {
        self.0
    }

    pub fn to_grams(self) -> MassGrams {
        MassGrams(self.0 / KG_PER_GRAM)
    }

    pub fn to_milligrams(self) -> MassMilligrams {
        MassMilligrams(self.0 / KG_PER_MILLIGRAM)
    }
}

impl Quantity for MassKilograms {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Mass;
    const UNIT_NAME: &'static str = "kg";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for MassKilograms {}

impl crate::conversion::UnitOfMeasure for MassKilograms {
    type Dim = dim::Mass;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <MassKilograms as crate::Quantity>::DIMENSION
        .same(<dim::Mass as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(MassKilograms)),
);

impl core::fmt::Display for MassKilograms {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for MassKilograms {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = <MassKilograms as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str = <MassKilograms as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for MassKilograms {
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

impl euclid_units::LinearQuantity for MassKilograms {
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

impl Mass for MassKilograms {
    type Force = ForceNewtons;
    type Acceleration = AccelerationMetersPerSecondSquared;
    type Momentum = MomentumKilogramMetersPerSecond;
    type Speed = SpeedMs;
    type Density = DensityKilogramsPerCubicMeter;
    type Volume = VolumeCubicMeters;
    type MomentOfInertia = MomentOfInertiaKilogramMetersSquared;
    type Area = AreaSquareMeters;
    type MolarMass = MolarMassKilogramsPerMole;
    type AmountOfSubstance = AmountMoles;
    type Time = TimeSeconds;
    type MassFlowRate = MassFlowRateKilogramsPerSecond;

    fn times_acceleration(self, accel: Self::Acceleration) -> Self::Force {
        ForceNewtons::from_raw(self.raw() * accel.raw())
    }
    fn times_speed(self, speed: Self::Speed) -> Self::Momentum {
        MomentumKilogramMetersPerSecond::from_raw(self.raw() * speed.raw())
    }
    fn per_volume(self, volume: Self::Volume) -> Self::Density {
        DensityKilogramsPerCubicMeter::from_raw(self.raw() / volume.raw())
    }
    fn times_area(self, area: Self::Area) -> Self::MomentOfInertia {
        MomentOfInertiaKilogramMetersSquared::from_raw(self.raw() * area.raw())
    }
    fn per_amount(self, amount: Self::AmountOfSubstance) -> Self::MolarMass {
        MolarMassKilogramsPerMole::from_raw(self.raw() / amount.raw())
    }
    fn per_time(self, elapsed: Self::Time) -> Self::MassFlowRate {
        MassFlowRateKilogramsPerSecond::from_raw(self.raw() / elapsed.raw())
    }
}

impl ::core::ops::Add for MassKilograms {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for MassKilograms {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for MassKilograms {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for MassKilograms {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for MassKilograms {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for MassKilograms {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<MassKilograms> for f64 {
    type Output = MassKilograms;
    fn mul(self, x: MassKilograms) -> MassKilograms {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for MassKilograms {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for MassKilograms {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
