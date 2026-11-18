use super::{M_PER_FOOT, M_PER_KM};
use crate::Measure;
use crate::conversion::dim;
use crate::{AreaSquareMeters, LengthFeet, LengthKm, SpeedMs, TimeSeconds, VolumeCubicMeters};
use bytemuck::{Pod, Zeroable};
use euclid_units::Length;

#[repr(transparent)]
/// A length in **metres**, the SI base unit and this crate's anchor for the length
/// dimension — every other length converts through it.
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
pub struct LengthMeters(pub(crate) f64);

impl LengthMeters {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0.0);

    /// Positive infinity, for use as an unreachable sentinel in a minimisation.
    ///
    /// Arithmetic on it follows IEEE-754 rather than saturating, so `INFINITY - INFINITY` is
    /// `NaN` and not zero.
    pub const INFINITY: Self = Self(f64::INFINITY);

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

    /// The raw value in meters, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_meters(self) -> f64 {
        self.0
    }

    /// The raw value in meters, narrowed to `f32`.
    ///
    /// Lossy: about seven significant digits rather than sixteen, and values past `f32`'s range
    /// become infinite. For storage and interchange, not for further arithmetic.
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_f32_meters(self) -> f32 {
        self.0 as f32
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

    /// Neither infinite nor `NaN`.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
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
        debug_assert!(self.0 > 0.0, "expected positive LengthMeters, got {self:?}");
        self
    }
}

impl LengthMeters {
    /// Convert to [`LengthKm`].
    pub fn to_km(self) -> LengthKm {
        LengthKm(self.0 / M_PER_KM)
    }

    /// Convert to [`LengthFeet`].
    pub fn to_feet(self) -> LengthFeet {
        LengthFeet(self.0 / M_PER_FOOT)
    }
}

impl crate::conversion::UnitOfMeasure for LengthMeters {
    type Dim = dim::Length;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <LengthMeters as euclid_units::Measure>::DIMENSION
        .same(<dim::Length as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(LengthMeters)),
);

impl core::fmt::Display for LengthMeters {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for LengthMeters {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Length;
    const UNIT_NAME: &'static str = "m";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for LengthMeters {}

impl euclid_units::LinearQuantity for LengthMeters {}

impl Length for LengthMeters {
    type Area = AreaSquareMeters;
    type Volume = VolumeCubicMeters;
    type Time = TimeSeconds;
    type Speed = SpeedMs;

    fn times_length(self, width: Self) -> Self::Area {
        AreaSquareMeters::from_raw(self.raw() * width.raw())
    }
    fn times_area(self, base: Self::Area) -> Self::Volume {
        VolumeCubicMeters::from_raw(self.raw() * base.raw())
    }
    fn per_time(self, elapsed: Self::Time) -> Self::Speed {
        SpeedMs::from_raw(self.raw() / elapsed.raw())
    }
    fn per_speed(self, speed: Self::Speed) -> Self::Time {
        TimeSeconds::from_raw(self.raw() / speed.raw())
    }
    fn squared(self) -> Self::Area {
        AreaSquareMeters::from_raw(self.raw() * self.raw())
    }
    fn cubed(self) -> Self::Volume {
        VolumeCubicMeters::from_raw(self.raw() * self.raw() * self.raw())
    }
}

impl ::core::ops::Add for LengthMeters {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for LengthMeters {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for LengthMeters {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for LengthMeters {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for LengthMeters {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for LengthMeters {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<LengthMeters> for f64 {
    type Output = LengthMeters;
    fn mul(self, x: LengthMeters) -> LengthMeters {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for LengthMeters {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for LengthMeters {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
