use crate::Factor;
use bytemuck::{Pod, Zeroable};
use euclid_units::dim;

#[repr(transparent)]
/// A **dimensionless** proportion expressed in percent, so 50% is stored as `50.0` and
/// not `0.5`.
///
/// The storage convention is the whole reason this is a distinct type from
/// [`crate::Factor`]: both are dimensionless, and mixing them silently is a
/// factor-of-100 error that no dimension check can catch.
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
pub struct Percentage(pub(crate) f64);

impl Percentage {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0.0);

    /// One hundred percent, stored as `100.0` — the whole, not a fraction.
    ///
    /// The storage convention this type exists to keep straight: `FULL` is numerically `100`,
    /// where the equivalent [`crate::Factor`] is numerically `1`.
    pub const FULL: Self = Self(100.0);

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

    /// Build from a [`crate::Factor`], multiplying by 100.
    ///
    /// The only safe way across that boundary, since both types are dimensionless `f64` and
    /// nothing in the type system catches a raw value crossing it unscaled.
    pub fn from_fraction(f: Factor) -> Self {
        Self(f.as_f64() * 100.0)
    }

    /// Convert to [`Factor`].
    pub fn to_fraction(self) -> Factor {
        Factor::new(self.0 / 100.0)
    }

    /// The raw value in percent, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_percent(self) -> f64 {
        self.0
    }

    /// Exactly zero. Both signed zeroes qualify; a value merely close to zero does not.
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// Zero or greater. `false` for `NaN`, so this is **not** the negation of
    /// [`is_negative`-style checks](Self::is_positive) — both are false for `NaN`.
    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    /// Strictly greater than zero. `false` for zero and for `NaN`.
    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    /// Whether the value lies in `[0, 100]` and is finite.
    ///
    /// Not enforced on construction — a percentage outside the range is representable, because
    /// an over-100% utilisation or a negative change is often exactly what is being measured.
    pub fn is_in_range(self) -> bool {
        self.0.is_finite() && (0.0..=100.0).contains(&self.0)
    }
}

impl crate::conversion::UnitOfMeasure for Percentage {
    type Dim = dim::Dimensionless;
    const TO_PRIMARY: f64 = 1.0 / 100.0;
}

const _: () = assert!(
    <Percentage as euclid_units::Measure>::DIMENSION
        .same(<dim::Dimensionless as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(Percentage)),
);

impl core::fmt::Display for Percentage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for Percentage {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Dimensionless;
    const UNIT_NAME: &'static str = "%";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for Percentage {}

impl euclid_units::LinearQuantity for Percentage {}

impl euclid_units::Dimensionless for Percentage {
    fn unit() -> Self {
        Self::FULL
    }

    fn times(self, other: Self) -> Self {
        Self::from_fraction(self.to_fraction() * other.to_fraction())
    }

    fn recip(self) -> Self {
        Self::from_fraction(Factor::new(self.to_fraction().as_f64().recip()))
    }
}

impl ::core::ops::Add for Percentage {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for Percentage {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for Percentage {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for Percentage {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for Percentage {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for Percentage {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<Percentage> for f64 {
    type Output = Percentage;
    fn mul(self, x: Percentage) -> Percentage {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for Percentage {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for Percentage {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
