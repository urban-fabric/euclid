use super::KG_PER_GRAM;
use crate::{MassKilograms, MassMilligrams};
use bytemuck::{Pod, Zeroable};
use euclid_units::dim;

#[repr(transparent)]
/// A mass in **grams**. Exactly 1/1000 kg.
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct MassGrams(pub(crate) f64);

impl MassGrams {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0.0);

    /// Wrap a raw `f64` already expressed in this unit.
    ///
    /// Asserted rather than checked: the caller is promising `v` is in this unit, and nothing
    /// here can verify that. Passing a value in the wrong unit is the one mistake the type
    /// system cannot catch at this boundary, which is why conversions exist.
    ///
    /// # Panics
    /// On `NaN`, in a **debug build only** — the check is a `debug_assert!`, so a release build
    /// accepts a `NaN` and propagates it silently through every later operation.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// The raw value in grams, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_grams(self) -> f64 {
        self.0
    }

    /// Convert to [`MassKilograms`].
    pub fn to_kilograms(self) -> MassKilograms {
        MassKilograms(self.0 * KG_PER_GRAM)
    }

    /// Convert to [`MassMilligrams`].
    pub fn to_milligrams(self) -> MassMilligrams {
        MassMilligrams(self.0 * 1_000.0)
    }
}

impl crate::conversion::UnitOfMeasure for MassGrams {
    type Dim = dim::Mass;
    const TO_PRIMARY: f64 = KG_PER_GRAM;
}

const _: () = assert!(
    <MassGrams as euclid_units::Measure>::DIMENSION
        .same(<dim::Mass as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(MassGrams)),
);

impl core::fmt::Display for MassGrams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for MassGrams {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Mass;
    const UNIT_NAME: &'static str = "g";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for MassGrams {}

impl euclid_units::LinearQuantity for MassGrams {}

impl ::core::ops::Add for MassGrams {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for MassGrams {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for MassGrams {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for MassGrams {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for MassGrams {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for MassGrams {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<MassGrams> for f64 {
    type Output = MassGrams;
    fn mul(self, x: MassGrams) -> MassGrams {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for MassGrams {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for MassGrams {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
