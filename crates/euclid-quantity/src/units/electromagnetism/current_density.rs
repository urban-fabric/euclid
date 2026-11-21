use crate::{AreaSquareMeters, CurrentAmperes};
use euclid_units::CurrentDensity;
use euclid_units::Measure;
use euclid_units::dim;

/// Current density in amperes per square meter (A/m^2), the coherent SI unit.
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
pub struct CurrentDensityAmperesPerSquareMeter(pub(crate) f64);

impl CurrentDensityAmperesPerSquareMeter {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_amperes_per_square_meter(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for CurrentDensityAmperesPerSquareMeter {
    type Dim = dim::CurrentDensity;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <CurrentDensityAmperesPerSquareMeter as euclid_units::Measure>::DIMENSION
        .same(<dim::CurrentDensity as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(CurrentDensityAmperesPerSquareMeter)
    ),
);

impl core::fmt::Display for CurrentDensityAmperesPerSquareMeter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for CurrentDensityAmperesPerSquareMeter {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::CurrentDensity;
    const UNIT_NAME: &'static str = "A/m^2";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for CurrentDensityAmperesPerSquareMeter {}

impl euclid_units::LinearQuantity for CurrentDensityAmperesPerSquareMeter {}

impl CurrentDensity for CurrentDensityAmperesPerSquareMeter {
    type Area = AreaSquareMeters;
    type ElectricCurrent = CurrentAmperes;

    fn times_area(self, area: Self::Area) -> Self::ElectricCurrent {
        CurrentAmperes::from_raw(self.raw() * area.raw())
    }
}

impl ::core::ops::Add for CurrentDensityAmperesPerSquareMeter {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for CurrentDensityAmperesPerSquareMeter {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for CurrentDensityAmperesPerSquareMeter {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for CurrentDensityAmperesPerSquareMeter {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for CurrentDensityAmperesPerSquareMeter {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for CurrentDensityAmperesPerSquareMeter {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<CurrentDensityAmperesPerSquareMeter> for f64 {
    type Output = CurrentDensityAmperesPerSquareMeter;
    fn mul(self, x: CurrentDensityAmperesPerSquareMeter) -> CurrentDensityAmperesPerSquareMeter {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for CurrentDensityAmperesPerSquareMeter {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for CurrentDensityAmperesPerSquareMeter {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
