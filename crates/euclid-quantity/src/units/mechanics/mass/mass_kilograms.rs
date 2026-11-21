use super::{KG_PER_GRAM, KG_PER_MILLIGRAM};
use crate::{
    AccelerationMetersPerSecondSquared, AmountMoles, AreaSquareMeters,
    DensityKilogramsPerCubicMeter, ForceNewtons, MassFlowRateKilogramsPerSecond, MassGrams,
    MassMilligrams, MolarMassKilogramsPerMole, MomentOfInertiaKilogramMetersSquared,
    MomentumKilogramMetersPerSecond, SpeedMs, TimeSeconds, VolumeCubicMeters,
};
use bytemuck::{Pod, Zeroable};
use euclid_units::Mass;
use euclid_units::Measure;
use euclid_units::dim;

#[repr(transparent)]
/// A mass in **kilograms**, the SI base unit and the anchor for the mass dimension.
///
/// Note the SI oddity that the base unit carries a prefix, so grams and milligrams are
/// the derived ones here rather than the other way round.
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

    /// The raw value in kilograms, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_kilograms(self) -> f64 {
        self.0
    }

    /// Convert to [`MassGrams`].
    pub fn to_grams(self) -> MassGrams {
        MassGrams(self.0 / KG_PER_GRAM)
    }

    /// Convert to [`MassMilligrams`].
    pub fn to_milligrams(self) -> MassMilligrams {
        MassMilligrams(self.0 / KG_PER_MILLIGRAM)
    }
}

impl crate::conversion::UnitOfMeasure for MassKilograms {
    type Dim = dim::Mass;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <MassKilograms as euclid_units::Measure>::DIMENSION
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
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Mass;
    const UNIT_NAME: &'static str = "kg";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for MassKilograms {}

impl euclid_units::LinearQuantity for MassKilograms {}

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
