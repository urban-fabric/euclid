use crate::Measure;
use crate::conversion::dim;
use crate::{
    AmountMoles, ChargeCoulombs, ForceNewtons, HeatCapacityJoulesPerKelvin, LengthMeters,
    MassKilograms, MolarEnergyJoulesPerMole, PowerWatts, PressurePascals,
    SpecificEnergyJoulesPerKilogram, TemperatureKelvin, TimeSeconds, VoltageVolts,
    VolumeCubicMeters,
};
use euclid_units::Energy;

/// Energy in joules (J = N.m), the coherent SI unit.
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
pub struct EnergyJoules(pub(crate) f64);

impl EnergyJoules {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_joules(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for EnergyJoules {
    type Dim = dim::Energy;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <EnergyJoules as euclid_units::Measure>::DIMENSION
        .same(<dim::Energy as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(EnergyJoules)),
);

impl core::fmt::Display for EnergyJoules {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for EnergyJoules {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Energy;
    const UNIT_NAME: &'static str = "J";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for EnergyJoules {}

impl euclid_units::LinearQuantity for EnergyJoules {}

impl Energy for EnergyJoules {
    type Force = ForceNewtons;
    type Length = LengthMeters;
    type Time = TimeSeconds;
    type Power = PowerWatts;
    type Volume = VolumeCubicMeters;
    type Pressure = PressurePascals;
    type ElectricCharge = ChargeCoulombs;
    type Voltage = VoltageVolts;
    type Temperature = TemperatureKelvin;
    type HeatCapacity = HeatCapacityJoulesPerKelvin;
    type Mass = MassKilograms;
    type SpecificEnergy = SpecificEnergyJoulesPerKilogram;
    type AmountOfSubstance = AmountMoles;
    type MolarEnergy = MolarEnergyJoulesPerMole;

    fn per_length(self, distance: Self::Length) -> Self::Force {
        ForceNewtons::from_raw(self.raw() / distance.raw())
    }
    fn per_force(self, force: Self::Force) -> Self::Length {
        LengthMeters::from_raw(self.raw() / force.raw())
    }
    fn per_time(self, elapsed: Self::Time) -> Self::Power {
        PowerWatts::from_raw(self.raw() / elapsed.raw())
    }
    fn per_power(self, power: Self::Power) -> Self::Time {
        TimeSeconds::from_raw(self.raw() / power.raw())
    }
    fn per_volume(self, volume: Self::Volume) -> Self::Pressure {
        PressurePascals::from_raw(self.raw() / volume.raw())
    }
    fn per_charge(self, charge: Self::ElectricCharge) -> Self::Voltage {
        VoltageVolts::from_raw(self.raw() / charge.raw())
    }
    fn per_temperature(self, temperature: Self::Temperature) -> Self::HeatCapacity {
        HeatCapacityJoulesPerKelvin::from_raw(self.raw() / temperature.raw())
    }
    fn per_mass(self, mass: Self::Mass) -> Self::SpecificEnergy {
        SpecificEnergyJoulesPerKilogram::from_raw(self.raw() / mass.raw())
    }
    fn per_amount(self, amount: Self::AmountOfSubstance) -> Self::MolarEnergy {
        MolarEnergyJoulesPerMole::from_raw(self.raw() / amount.raw())
    }
}

impl ::core::ops::Add for EnergyJoules {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for EnergyJoules {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for EnergyJoules {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for EnergyJoules {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for EnergyJoules {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for EnergyJoules {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<EnergyJoules> for f64 {
    type Output = EnergyJoules;
    fn mul(self, x: EnergyJoules) -> EnergyJoules {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for EnergyJoules {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for EnergyJoules {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
