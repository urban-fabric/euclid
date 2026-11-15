use crate::Measure;
use crate::conversion::dim;
use crate::{
    AreaSquareMeters, CurrentAmperes, InductanceHenries, MagneticFluxDensityTeslas, TimeSeconds,
    VoltageVolts,
};
use euclid_units::MagneticFlux;

/// Magnetic flux in webers (Wb = V.s), the coherent SI unit.
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
pub struct MagneticFluxWebers(pub(crate) f64);

impl MagneticFluxWebers {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_webers(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for MagneticFluxWebers {
    type Dim = dim::MagneticFlux;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <MagneticFluxWebers as euclid_units::Measure>::DIMENSION
        .same(<dim::MagneticFlux as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(MagneticFluxWebers)),
);

impl core::fmt::Display for MagneticFluxWebers {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for MagneticFluxWebers {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::MagneticFlux;
    const UNIT_NAME: &'static str = "Wb";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for MagneticFluxWebers {}

impl euclid_units::LinearQuantity for MagneticFluxWebers {}

impl MagneticFlux for MagneticFluxWebers {
    type ElectricCurrent = CurrentAmperes;
    type Inductance = InductanceHenries;
    type Time = TimeSeconds;
    type Voltage = VoltageVolts;
    type Area = AreaSquareMeters;
    type MagneticFluxDensity = MagneticFluxDensityTeslas;

    fn per_current(self, current: Self::ElectricCurrent) -> Self::Inductance {
        InductanceHenries::from_raw(self.raw() / current.raw())
    }
    fn per_time(self, elapsed: Self::Time) -> Self::Voltage {
        VoltageVolts::from_raw(self.raw() / elapsed.raw())
    }
    fn per_area(self, area: Self::Area) -> Self::MagneticFluxDensity {
        MagneticFluxDensityTeslas::from_raw(self.raw() / area.raw())
    }
}

impl ::core::ops::Add for MagneticFluxWebers {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for MagneticFluxWebers {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for MagneticFluxWebers {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for MagneticFluxWebers {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for MagneticFluxWebers {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for MagneticFluxWebers {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<MagneticFluxWebers> for f64 {
    type Output = MagneticFluxWebers;
    fn mul(self, x: MagneticFluxWebers) -> MagneticFluxWebers {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for MagneticFluxWebers {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for MagneticFluxWebers {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
