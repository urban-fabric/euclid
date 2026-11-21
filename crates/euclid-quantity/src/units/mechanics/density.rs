use crate::{MassKilograms, VolumeCubicMeters};
use euclid_units::Density;
use euclid_units::Measure;
use euclid_units::dim;

/// Density in kilograms per cubic meter (kg/m^3), the coherent SI unit.
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
pub struct DensityKilogramsPerCubicMeter(pub(crate) f64);

impl DensityKilogramsPerCubicMeter {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_kilograms_per_cubic_meter(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for DensityKilogramsPerCubicMeter {
    type Dim = dim::Density;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <DensityKilogramsPerCubicMeter as euclid_units::Measure>::DIMENSION
        .same(<dim::Density as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(DensityKilogramsPerCubicMeter)
    ),
);

impl core::fmt::Display for DensityKilogramsPerCubicMeter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for DensityKilogramsPerCubicMeter {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Density;
    const UNIT_NAME: &'static str = "kg/m^3";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for DensityKilogramsPerCubicMeter {}

impl euclid_units::LinearQuantity for DensityKilogramsPerCubicMeter {}

impl Density for DensityKilogramsPerCubicMeter {
    type Mass = MassKilograms;
    type Volume = VolumeCubicMeters;

    fn times_volume(self, volume: Self::Volume) -> Self::Mass {
        MassKilograms::from_raw(self.raw() * volume.raw())
    }
}

impl ::core::ops::Add for DensityKilogramsPerCubicMeter {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for DensityKilogramsPerCubicMeter {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for DensityKilogramsPerCubicMeter {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for DensityKilogramsPerCubicMeter {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for DensityKilogramsPerCubicMeter {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for DensityKilogramsPerCubicMeter {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<DensityKilogramsPerCubicMeter> for f64 {
    type Output = DensityKilogramsPerCubicMeter;
    fn mul(self, x: DensityKilogramsPerCubicMeter) -> DensityKilogramsPerCubicMeter {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for DensityKilogramsPerCubicMeter {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for DensityKilogramsPerCubicMeter {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
