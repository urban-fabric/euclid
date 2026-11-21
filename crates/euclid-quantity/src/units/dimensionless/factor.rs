use bytemuck::{Pod, Zeroable};
use euclid_units::dim;
use std::ops::{Div, Mul};

#[repr(transparent)]
/// A **dimensionless** multiplier.
///
/// The unit a ratio of two like quantities collapses to, so it is what
/// `length / length` produces and what scales any quantity without changing its
/// dimension.
#[derive(
    Copy, Clone, Debug, PartialEq, PartialOrd, Pod, Zeroable, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct Factor(pub(crate) f64);

impl Default for Factor {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Factor {
    /// The multiplicative identity, `1`. Scaling by it changes nothing.
    ///
    /// Named `IDENTITY` rather than `ONE` because that is what it is *for* here: a `Factor` is
    /// applied, and the neutral element of application is an identity.
    pub const IDENTITY: Self = Self(1.0);

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

    /// The raw value in f64, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    #[inline]
    #[must_use]
    pub const fn as_f64(self) -> f64 {
        self.0
    }

    /// Neither infinite nor `NaN`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Strictly greater than zero. `false` for zero and for `NaN`.
    #[inline]
    #[must_use]
    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    /// Zero or greater. `false` for `NaN`, so this is **not** the negation of
    /// [`is_negative`-style checks](Self::is_positive) — both are false for `NaN`.
    #[inline]
    #[must_use]
    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    /// Exactly zero. Both signed zeroes qualify; a value merely close to zero does not.
    #[inline]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// Strictly greater than a **dimensionless** threshold.
    ///
    /// `epsilon` is a bare `f64` compared against the raw value, so it is read in this type's
    /// own unit — a threshold chosen for metres means something else against kilometres. That
    /// is a seam in the units discipline, kept because a tolerance is usually a magnitude
    /// rather than a quantity.
    #[inline]
    #[must_use]
    pub fn is_greater_than_epsilon(self, epsilon: f64) -> bool {
        self.0 > epsilon
    }

    /// Clamp into `[min, max]`, both given as bare `f64` in this type's own (dimensionless)
    /// scale.
    ///
    /// # Panics
    /// Delegates to `f64::clamp`, which panics if `min > max` or either bound is `NaN`.
    #[inline]
    #[must_use]
    pub fn clamped(self, min: f64, max: f64) -> Self {
        Self(self.0.clamp(min, max))
    }

    /// Clamp to zero at the bottom, returning the value unchanged if already non-negative.
    ///
    /// For a quantity that is physically non-negative but arrived from arithmetic that could
    /// have gone slightly below zero — a subtraction of two nearly-equal measurements.
    #[inline]
    #[must_use]
    pub fn non_negative(self) -> Self {
        Self(self.0.max(0.0))
    }
}

impl Mul for Factor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for Factor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl crate::conversion::UnitOfMeasure for Factor {
    type Dim = dim::Dimensionless;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <Factor as euclid_units::Measure>::DIMENSION
        .same(<dim::Dimensionless as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(Factor)),
);

impl core::fmt::Display for Factor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for Factor {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Dimensionless;
    const UNIT_NAME: &'static str = "";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for Factor {}

impl euclid_units::LinearQuantity for Factor {}

impl euclid_units::Dimensionless for Factor {
    fn unit() -> Self {
        Default::default()
    }

    fn times(self, other: Self) -> Self {
        Self::new(self.as_f64() * other.as_f64())
    }

    fn recip(self) -> Self {
        Self::new(self.as_f64().recip())
    }
}

impl ::core::ops::Add for Factor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for Factor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for Factor {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for Factor {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for Factor {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for Factor {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<Factor> for f64 {
    type Output = Factor;
    fn mul(self, x: Factor) -> Factor {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for Factor {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for Factor {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
