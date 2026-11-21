use super::S_PER_HOUR;
use crate::TimeSeconds;
use bytemuck::{Pod, Zeroable};
use euclid_units::dim;

#[repr(transparent)]
/// A duration in **hours**. Exactly 3600 s.
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
pub struct TimeHours(pub(crate) f64);

impl TimeHours {
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

    /// The raw value in hours, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_hours(self) -> f64 {
        self.0
    }

    /// Convert to [`TimeSeconds`].
    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds(self.0 * S_PER_HOUR)
    }
}

impl crate::conversion::UnitOfMeasure for TimeHours {
    type Dim = dim::Time;
    const TO_PRIMARY: f64 = S_PER_HOUR;
}

const _: () = assert!(
    <TimeHours as euclid_units::Measure>::DIMENSION
        .same(<dim::Time as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(TimeHours)),
);

impl core::fmt::Display for TimeHours {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for TimeHours {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "h";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for TimeHours {}

impl euclid_units::LinearQuantity for TimeHours {}

impl ::core::ops::Add for TimeHours {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for TimeHours {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for TimeHours {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for TimeHours {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for TimeHours {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for TimeHours {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<TimeHours> for f64 {
    type Output = TimeHours;
    fn mul(self, x: TimeHours) -> TimeHours {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for TimeHours {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for TimeHours {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
