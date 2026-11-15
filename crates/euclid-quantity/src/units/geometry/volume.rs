use crate::Measure;
use crate::conversion::dim;
use crate::{AmountMoles, AreaSquareMeters, LengthMeters, MolarVolumeCubicMetersPerMole};
use euclid_units::Volume;

/// Volume in cubic meters (m^3), the coherent SI unit.
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
pub struct VolumeCubicMeters(pub(crate) f64);

impl VolumeCubicMeters {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_cubic_meters(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for VolumeCubicMeters {
    type Dim = dim::Volume;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <VolumeCubicMeters as euclid_units::Measure>::DIMENSION
        .same(<dim::Volume as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(VolumeCubicMeters)),
);

impl core::fmt::Display for VolumeCubicMeters {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for VolumeCubicMeters {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Volume;
    const UNIT_NAME: &'static str = "m^3";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for VolumeCubicMeters {}

impl euclid_units::LinearQuantity for VolumeCubicMeters {}

impl Volume for VolumeCubicMeters {
    type Length = LengthMeters;
    type Area = AreaSquareMeters;
    type AmountOfSubstance = AmountMoles;
    type MolarVolume = MolarVolumeCubicMetersPerMole;

    fn per_length(self, height: Self::Length) -> Self::Area {
        AreaSquareMeters::from_raw(self.raw() / height.raw())
    }
    fn per_area(self, base: Self::Area) -> Self::Length {
        LengthMeters::from_raw(self.raw() / base.raw())
    }
    fn edge(self) -> Self::Length {
        LengthMeters::from_raw(self.raw().cbrt())
    }
    fn per_amount(self, amount: Self::AmountOfSubstance) -> Self::MolarVolume {
        MolarVolumeCubicMetersPerMole::from_raw(self.raw() / amount.raw())
    }
}

impl ::core::ops::Add for VolumeCubicMeters {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for VolumeCubicMeters {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for VolumeCubicMeters {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for VolumeCubicMeters {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for VolumeCubicMeters {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for VolumeCubicMeters {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<VolumeCubicMeters> for f64 {
    type Output = VolumeCubicMeters;
    fn mul(self, x: VolumeCubicMeters) -> VolumeCubicMeters {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for VolumeCubicMeters {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for VolumeCubicMeters {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
