use super::{MPH_PER_KPH, MS_PER_KPH};
use crate::{SpeedMph, SpeedMs};
use bytemuck::{Pod, Zeroable};
use euclid_units::dim;

#[repr(transparent)]
/// A speed in **kilometres per hour**. Exactly 1/3.6 m/s.
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
pub struct SpeedKph(pub(crate) f64);

impl SpeedKph {
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

    /// Build from an integer speed limit in km/h, the form a road network stores.
    ///
    /// `u16` because that is the width such data arrives in, and every value of it is exactly
    /// representable — so unlike the `f64` constructors this one cannot be handed a `NaN`.
    #[inline]
    #[must_use]
    pub const fn from_kph_u16(kph: u16) -> Self {
        Self(kph as f64)
    }

    /// The raw value in kph, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_kph(self) -> f64 {
        self.0
    }

    /// The raw value in kph, narrowed to `f32`.
    ///
    /// Lossy: about seven significant digits rather than sixteen, and values past `f32`'s range
    /// become infinite. For storage and interchange, not for further arithmetic.
    pub fn to_f32_kph(self) -> f32 {
        self.0 as f32
    }

    /// Neither infinite nor `NaN`.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Strictly greater than zero. `false` for zero and for `NaN`.
    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    /// Zero or greater. `false` for `NaN`, so this is **not** the negation of
    /// [`is_negative`-style checks](Self::is_positive) — both are false for `NaN`.
    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    /// Exactly zero. Both signed zeroes qualify; a value merely close to zero does not.
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// Strictly greater than a **dimensionless** threshold.
    ///
    /// `epsilon` is a bare `f64` compared against the raw value, so it is read in this type's
    /// own unit — a threshold chosen for metres means something else against kilometres. That
    /// is a seam in the units discipline, kept because a tolerance is usually a magnitude
    /// rather than a quantity.
    pub fn is_greater_than_epsilon(self, epsilon: f64) -> bool {
        self.0 > epsilon
    }

    /// Magnitude, dropping the sign. `NaN` stays `NaN`.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Return `self` unchanged, having checked it is positive.
    ///
    /// # Panics
    /// If the value is not strictly positive — in a **debug build only**. The check is a
    /// `debug_assert!`, so in release this is the identity function and the name promises a
    /// guarantee it does not deliver. Use it to catch a bug in testing, never to enforce an
    /// invariant a release build depends on.
    pub fn assert_positive(self) -> Self {
        debug_assert!(self.0 > 0.0, "expected positive SpeedKph, got {self:?}");
        self
    }

    /// Convert to [`SpeedMs`].
    pub fn to_ms(self) -> SpeedMs {
        SpeedMs(self.0 * MS_PER_KPH)
    }

    /// Convert to [`SpeedMph`].
    pub fn to_mph(self) -> SpeedMph {
        SpeedMph(self.0 * MPH_PER_KPH)
    }
}

impl crate::conversion::UnitOfMeasure for SpeedKph {
    type Dim = dim::Speed;
    const TO_PRIMARY: f64 = MS_PER_KPH;
}

const _: () = assert!(
    <SpeedKph as euclid_units::Measure>::DIMENSION
        .same(<dim::Speed as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(SpeedKph)),
);

impl core::fmt::Display for SpeedKph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for SpeedKph {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Speed;
    const UNIT_NAME: &'static str = "km/h";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for SpeedKph {}

impl euclid_units::LinearQuantity for SpeedKph {}

impl ::core::ops::Add for SpeedKph {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for SpeedKph {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for SpeedKph {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for SpeedKph {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for SpeedKph {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for SpeedKph {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<SpeedKph> for f64 {
    type Output = SpeedKph;
    fn mul(self, x: SpeedKph) -> SpeedKph {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for SpeedKph {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for SpeedKph {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
