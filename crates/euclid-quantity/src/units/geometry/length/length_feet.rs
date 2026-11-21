use super::M_PER_FOOT;
use crate::LengthMeters;
use bytemuck::{Pod, Zeroable};
use euclid_units::dim;

#[repr(transparent)]
/// A length in **international feet**, exactly 0.3048 m by definition.
///
/// Not the US survey foot, which differs by 2 ppm — enough to matter over a state plane
/// coordinate and nothing else.
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
pub struct LengthFeet(pub(crate) f64);

impl LengthFeet {
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

    /// The raw value in feet, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_feet(self) -> f64 {
        self.0
    }

    /// Convert to [`LengthMeters`].
    pub fn to_meters(self) -> LengthMeters {
        LengthMeters(self.0 * M_PER_FOOT)
    }
}

impl crate::conversion::UnitOfMeasure for LengthFeet {
    type Dim = dim::Length;
    const TO_PRIMARY: f64 = M_PER_FOOT;
}

const _: () = assert!(
    <LengthFeet as euclid_units::Measure>::DIMENSION
        .same(<dim::Length as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(LengthFeet)),
);

impl core::fmt::Display for LengthFeet {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for LengthFeet {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Length;
    const UNIT_NAME: &'static str = "ft";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for LengthFeet {}

impl euclid_units::LinearQuantity for LengthFeet {}

impl ::core::ops::Add for LengthFeet {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for LengthFeet {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for LengthFeet {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for LengthFeet {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for LengthFeet {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for LengthFeet {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<LengthFeet> for f64 {
    type Output = LengthFeet;
    fn mul(self, x: LengthFeet) -> LengthFeet {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for LengthFeet {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for LengthFeet {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
