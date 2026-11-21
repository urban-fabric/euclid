use super::MS_PER_S;
use crate::TimeSeconds;
use bytemuck::{Pod, Zeroable};
use euclid_units::dim;

#[repr(transparent)]
/// A duration in **milliseconds**, as a float — so it may be fractional, unlike the
/// integer-backed [`crate::TimeMicros`] family below it.
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
pub struct TimeMillis(pub(crate) f64);

impl TimeMillis {
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

    /// The raw value in millis, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_millis(self) -> f64 {
        self.0
    }

    /// Convert to [`TimeSeconds`].
    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds(self.0 / MS_PER_S)
    }

    /// Convert from a [`std::time::Duration`].
    ///
    /// Saturates at this type's maximum rather than wrapping, so a `Duration` beyond its range
    /// clamps. `Duration` is unsigned, so this is always a valid duration.
    pub fn from_duration(d: std::time::Duration) -> Self {
        Self(d.as_secs_f64() * MS_PER_S)
    }
}

impl crate::conversion::UnitOfMeasure for TimeMillis {
    type Dim = dim::Time;
    const TO_PRIMARY: f64 = 1.0 / MS_PER_S;
}

const _: () = assert!(
    <TimeMillis as euclid_units::Measure>::DIMENSION
        .same(<dim::Time as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(TimeMillis)),
);

impl core::fmt::Display for TimeMillis {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for TimeMillis {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "ms";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for TimeMillis {}

impl euclid_units::LinearQuantity for TimeMillis {}

impl ::core::ops::Add for TimeMillis {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for TimeMillis {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for TimeMillis {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for TimeMillis {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for TimeMillis {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for TimeMillis {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<TimeMillis> for f64 {
    type Output = TimeMillis;
    fn mul(self, x: TimeMillis) -> TimeMillis {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for TimeMillis {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for TimeMillis {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
