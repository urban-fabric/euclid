use crate::{
    CatalyticActivityKatals, ConcentrationMolesPerCubicMeter, MassKilograms,
    MolalityMolesPerKilogram, MolarMassKilogramsPerMole, TimeSeconds, VolumeCubicMeters,
};
use euclid_units::AmountOfSubstance;
use euclid_units::Measure;
use euclid_units::dim;

/// Amount of substance in moles (mol), the SI base unit.
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
pub struct AmountMoles(pub(crate) f64);

impl AmountMoles {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_moles(self) -> f64 {
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

impl crate::conversion::UnitOfMeasure for AmountMoles {
    type Dim = dim::AmountOfSubstance;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <AmountMoles as euclid_units::Measure>::DIMENSION
        .same(<dim::AmountOfSubstance as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(AmountMoles)),
);

impl core::fmt::Display for AmountMoles {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for AmountMoles {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::AmountOfSubstance;
    const UNIT_NAME: &'static str = "mol";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for AmountMoles {}

impl euclid_units::LinearQuantity for AmountMoles {}

impl AmountOfSubstance for AmountMoles {
    type Time = TimeSeconds;
    type CatalyticActivity = CatalyticActivityKatals;
    type Volume = VolumeCubicMeters;
    type Concentration = ConcentrationMolesPerCubicMeter;
    type MolarMass = MolarMassKilogramsPerMole;
    type Mass = MassKilograms;
    type Molality = MolalityMolesPerKilogram;

    fn per_time(self, elapsed: Self::Time) -> Self::CatalyticActivity {
        CatalyticActivityKatals::from_raw(self.raw() / elapsed.raw())
    }
    fn per_volume(self, volume: Self::Volume) -> Self::Concentration {
        ConcentrationMolesPerCubicMeter::from_raw(self.raw() / volume.raw())
    }
    fn times_molar_mass(self, molar_mass: Self::MolarMass) -> Self::Mass {
        MassKilograms::from_raw(self.raw() * molar_mass.raw())
    }
    fn per_mass(self, mass: Self::Mass) -> Self::Molality {
        MolalityMolesPerKilogram::from_raw(self.raw() / mass.raw())
    }
}

impl ::core::ops::Add for AmountMoles {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for AmountMoles {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for AmountMoles {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for AmountMoles {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for AmountMoles {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for AmountMoles {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<AmountMoles> for f64 {
    type Output = AmountMoles;
    fn mul(self, x: AmountMoles) -> AmountMoles {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for AmountMoles {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for AmountMoles {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
