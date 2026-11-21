use crate::{ChargeCoulombs, ResistanceOhms, TimeSeconds, VoltageVolts};
use euclid_units::Capacitance;
use euclid_units::Measure;
use euclid_units::dim;

/// Capacitance in farads (F = C/V), the coherent SI unit.
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
pub struct CapacitanceFarads(pub(crate) f64);

impl CapacitanceFarads {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_farads(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for CapacitanceFarads {
    type Dim = dim::Capacitance;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <CapacitanceFarads as euclid_units::Measure>::DIMENSION
        .same(<dim::Capacitance as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(CapacitanceFarads)),
);

impl core::fmt::Display for CapacitanceFarads {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for CapacitanceFarads {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Capacitance;
    const UNIT_NAME: &'static str = "F";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for CapacitanceFarads {}

impl euclid_units::LinearQuantity for CapacitanceFarads {}

impl Capacitance for CapacitanceFarads {
    type Voltage = VoltageVolts;
    type ElectricCharge = ChargeCoulombs;
    type Resistance = ResistanceOhms;
    type Time = TimeSeconds;

    fn times_voltage(self, voltage: Self::Voltage) -> Self::ElectricCharge {
        ChargeCoulombs::from_raw(self.raw() * voltage.raw())
    }
    fn times_resistance(self, resistance: Self::Resistance) -> Self::Time {
        TimeSeconds::from_raw(self.raw() * resistance.raw())
    }
}

impl ::core::ops::Add for CapacitanceFarads {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for CapacitanceFarads {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for CapacitanceFarads {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for CapacitanceFarads {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for CapacitanceFarads {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for CapacitanceFarads {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<CapacitanceFarads> for f64 {
    type Output = CapacitanceFarads;
    fn mul(self, x: CapacitanceFarads) -> CapacitanceFarads {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for CapacitanceFarads {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for CapacitanceFarads {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
