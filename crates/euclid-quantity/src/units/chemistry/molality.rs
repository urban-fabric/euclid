use crate::{AmountMoles, MassKilograms};
use euclid_units::Measure;
use euclid_units::Molality;
use euclid_units::dim;

/// Molality in moles per kilogram (mol/kg), the coherent SI unit.
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
pub struct MolalityMolesPerKilogram(pub(crate) f64);

impl MolalityMolesPerKilogram {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_moles_per_kilogram(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for MolalityMolesPerKilogram {
    type Dim = dim::Molality;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <MolalityMolesPerKilogram as euclid_units::Measure>::DIMENSION
        .same(<dim::Molality as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(MolalityMolesPerKilogram)
    ),
);

impl core::fmt::Display for MolalityMolesPerKilogram {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for MolalityMolesPerKilogram {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Molality;
    const UNIT_NAME: &'static str = "mol/kg";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for MolalityMolesPerKilogram {}

impl euclid_units::LinearQuantity for MolalityMolesPerKilogram {}

impl Molality for MolalityMolesPerKilogram {
    type Mass = MassKilograms;
    type AmountOfSubstance = AmountMoles;

    fn times_mass(self, mass: Self::Mass) -> Self::AmountOfSubstance {
        AmountMoles::from_raw(self.raw() * mass.raw())
    }
}

impl ::core::ops::Add for MolalityMolesPerKilogram {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for MolalityMolesPerKilogram {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for MolalityMolesPerKilogram {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for MolalityMolesPerKilogram {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for MolalityMolesPerKilogram {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for MolalityMolesPerKilogram {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<MolalityMolesPerKilogram> for f64 {
    type Output = MolalityMolesPerKilogram;
    fn mul(self, x: MolalityMolesPerKilogram) -> MolalityMolesPerKilogram {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for MolalityMolesPerKilogram {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for MolalityMolesPerKilogram {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
