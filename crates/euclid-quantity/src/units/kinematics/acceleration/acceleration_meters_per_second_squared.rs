use super::MPS2_PER_CMPS2;
use crate::Measure;
use crate::conversion::dim;
use crate::{
    AccelerationCentimetersPerSecondSquared, ForceNewtons, JerkMetersPerSecondCubed, MassKilograms,
    SpeedMs, TimeSeconds,
};
use bytemuck::{Pod, Zeroable};
use euclid_units::Acceleration;

#[repr(transparent)]
/// An acceleration in **metres per second squared**, the anchor for the acceleration
/// dimension.
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
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0.0);

    /// Wrap a raw `f64` already expressed in this unit.
    ///
    /// Asserted rather than checked: the caller is promising `v` is in this unit, and nothing
    /// here can verify that. Passing a value in the wrong unit is the one mistake the type
    /// system cannot catch at this boundary, which is why conversions exist.
    ///
    /// # Panics
    /// On `NaN`, in a **debug build only** — the check is a `debug_assert!`, so a release build
    /// accepts a `NaN` and propagates it silently through every later operation.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// The raw value in meters per second squared, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_meters_per_second_squared(self) -> f64 {
        self.0
    }

    /// Convert to [`AccelerationCentimetersPerSecondSquared`].
    pub fn to_centimeters_per_second_squared(self) -> AccelerationCentimetersPerSecondSquared {
        AccelerationCentimetersPerSecondSquared(self.0 / MPS2_PER_CMPS2)
    }
}

impl crate::conversion::UnitOfMeasure for AccelerationMetersPerSecondSquared {
    type Dim = dim::Acceleration;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <AccelerationMetersPerSecondSquared as euclid_units::Measure>::DIMENSION
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
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Acceleration;
    const UNIT_NAME: &'static str = "m/s^2";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for AccelerationMetersPerSecondSquared {}

impl euclid_units::LinearQuantity for AccelerationMetersPerSecondSquared {}

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
