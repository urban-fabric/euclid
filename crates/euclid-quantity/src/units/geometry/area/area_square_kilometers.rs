use super::SQUARE_METERS_PER_SQUARE_KILOMETER;
use crate::AreaSquareMeters;
use crate::conversion::dim;
use bytemuck::{Pod, Zeroable};

#[repr(transparent)]
/// An area in **square kilometres**. One km² is 10⁶ m², not 10³ — the factor is squared
/// along with the unit, which is the usual place this goes wrong by hand.
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
pub struct AreaSquareKilometers(pub(crate) f64);

impl AreaSquareKilometers {
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

    /// The raw value in square kilometers, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_square_kilometers(self) -> f64 {
        self.0
    }

    /// Convert to [`AreaSquareMeters`].
    pub fn to_square_meters(self) -> AreaSquareMeters {
        AreaSquareMeters(self.0 * SQUARE_METERS_PER_SQUARE_KILOMETER)
    }
}

impl crate::conversion::UnitOfMeasure for AreaSquareKilometers {
    type Dim = dim::Area;
    const TO_PRIMARY: f64 = SQUARE_METERS_PER_SQUARE_KILOMETER;
}

const _: () = assert!(
    <AreaSquareKilometers as euclid_units::Measure>::DIMENSION
        .same(<dim::Area as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(AreaSquareKilometers)),
);

impl core::fmt::Display for AreaSquareKilometers {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for AreaSquareKilometers {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Area;
    const UNIT_NAME: &'static str = "km^2";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for AreaSquareKilometers {}

impl euclid_units::LinearQuantity for AreaSquareKilometers {}

impl ::core::ops::Add for AreaSquareKilometers {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for AreaSquareKilometers {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for AreaSquareKilometers {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for AreaSquareKilometers {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for AreaSquareKilometers {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for AreaSquareKilometers {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<AreaSquareKilometers> for f64 {
    type Output = AreaSquareKilometers;
    fn mul(self, x: AreaSquareKilometers) -> AreaSquareKilometers {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for AreaSquareKilometers {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for AreaSquareKilometers {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
