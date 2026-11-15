use super::M_PER_KM;
use crate::LengthMeters;
use crate::conversion::dim;
use bytemuck::{Pod, Zeroable};

#[repr(transparent)]
/// A length in **kilometres**. Exactly 1000 m, so conversion to metres is exact and the
/// return trip is not.
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
pub struct LengthKm(pub(crate) f64);

impl LengthKm {
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

    /// The raw value in km, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_km(self) -> f64 {
        self.0
    }

    /// Convert to [`LengthMeters`].
    pub fn to_meters(self) -> LengthMeters {
        LengthMeters(self.0 * M_PER_KM)
    }
}

impl crate::conversion::UnitOfMeasure for LengthKm {
    type Dim = dim::Length;
    const TO_PRIMARY: f64 = M_PER_KM;
}

const _: () = assert!(
    <LengthKm as euclid_units::Measure>::DIMENSION
        .same(<dim::Length as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(LengthKm)),
);

impl core::fmt::Display for LengthKm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for LengthKm {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Length;
    const UNIT_NAME: &'static str = "km";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for LengthKm {}

impl euclid_units::LinearQuantity for LengthKm {}

impl ::core::ops::Add for LengthKm {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for LengthKm {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for LengthKm {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for LengthKm {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for LengthKm {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for LengthKm {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<LengthKm> for f64 {
    type Output = LengthKm;
    fn mul(self, x: LengthKm) -> LengthKm {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for LengthKm {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for LengthKm {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
