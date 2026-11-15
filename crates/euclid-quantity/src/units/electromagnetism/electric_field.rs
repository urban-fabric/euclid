use crate::Measure;
use crate::conversion::dim;
use crate::{ChargeCoulombs, ForceNewtons, LengthMeters, VoltageVolts};
use euclid_units::ElectricField;

/// Electric field strength in volts per meter (V/m), the coherent SI unit.
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
pub struct ElectricFieldVoltsPerMeter(pub(crate) f64);

impl ElectricFieldVoltsPerMeter {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_volts_per_meter(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for ElectricFieldVoltsPerMeter {
    type Dim = dim::ElectricField;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <ElectricFieldVoltsPerMeter as euclid_units::Measure>::DIMENSION
        .same(<dim::ElectricField as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(ElectricFieldVoltsPerMeter)
    ),
);

impl core::fmt::Display for ElectricFieldVoltsPerMeter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for ElectricFieldVoltsPerMeter {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::ElectricField;
    const UNIT_NAME: &'static str = "V/m";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for ElectricFieldVoltsPerMeter {}

impl euclid_units::LinearQuantity for ElectricFieldVoltsPerMeter {}

impl ElectricField for ElectricFieldVoltsPerMeter {
    type Length = LengthMeters;
    type Voltage = VoltageVolts;
    type ElectricCharge = ChargeCoulombs;
    type Force = ForceNewtons;

    fn times_length(self, length: Self::Length) -> Self::Voltage {
        VoltageVolts::from_raw(self.raw() * length.raw())
    }
    fn times_charge(self, charge: Self::ElectricCharge) -> Self::Force {
        ForceNewtons::from_raw(self.raw() * charge.raw())
    }
}

impl ::core::ops::Add for ElectricFieldVoltsPerMeter {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for ElectricFieldVoltsPerMeter {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for ElectricFieldVoltsPerMeter {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for ElectricFieldVoltsPerMeter {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for ElectricFieldVoltsPerMeter {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for ElectricFieldVoltsPerMeter {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<ElectricFieldVoltsPerMeter> for f64 {
    type Output = ElectricFieldVoltsPerMeter;
    fn mul(self, x: ElectricFieldVoltsPerMeter) -> ElectricFieldVoltsPerMeter {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for ElectricFieldVoltsPerMeter {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for ElectricFieldVoltsPerMeter {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
