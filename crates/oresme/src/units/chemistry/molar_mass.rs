use crate::Measure;
use crate::conversion::dim;
use crate::{AmountMoles, MassKilograms};
use euclid_units::MolarMass;

/// Molar mass in kilograms per mole (kg/mol), the coherent SI unit.
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
pub struct MolarMassKilogramsPerMole(pub(crate) f64);

impl MolarMassKilogramsPerMole {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_kilograms_per_mole(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for MolarMassKilogramsPerMole {
    type Dim = dim::MolarMass;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <MolarMassKilogramsPerMole as euclid_units::Measure>::DIMENSION
        .same(<dim::MolarMass as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(MolarMassKilogramsPerMole)
    ),
);

impl core::fmt::Display for MolarMassKilogramsPerMole {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for MolarMassKilogramsPerMole {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::MolarMass;
    const UNIT_NAME: &'static str = "kg/mol";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for MolarMassKilogramsPerMole {}

impl euclid_units::LinearQuantity for MolarMassKilogramsPerMole {}

impl MolarMass for MolarMassKilogramsPerMole {
    type AmountOfSubstance = AmountMoles;
    type Mass = MassKilograms;

    fn times_amount(self, amount: Self::AmountOfSubstance) -> Self::Mass {
        MassKilograms::from_raw(self.raw() * amount.raw())
    }
}

impl ::core::ops::Add for MolarMassKilogramsPerMole {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for MolarMassKilogramsPerMole {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for MolarMassKilogramsPerMole {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for MolarMassKilogramsPerMole {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for MolarMassKilogramsPerMole {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for MolarMassKilogramsPerMole {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<MolarMassKilogramsPerMole> for f64 {
    type Output = MolarMassKilogramsPerMole;
    fn mul(self, x: MolarMassKilogramsPerMole) -> MolarMassKilogramsPerMole {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for MolarMassKilogramsPerMole {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for MolarMassKilogramsPerMole {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
