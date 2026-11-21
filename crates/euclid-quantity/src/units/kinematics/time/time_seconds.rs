use super::{MS_PER_S, S_PER_HOUR, S_PER_MIN};
use crate::{
    AccelerationMetersPerSecondSquared, FrequencyHertz, LengthMeters, SpeedMs, TimeHours,
    TimeMillis, TimeMinutes,
};
use bytemuck::{Pod, Zeroable};
use euclid_units::Measure;
use euclid_units::Time;
use euclid_units::dim;

#[repr(transparent)]
/// A duration in **seconds**, the SI base unit and this crate's anchor for the time
/// dimension.
///
/// A duration rather than an instant: for a point on the timeline see
/// [`crate::UnixSeconds`], which is affine and does not add to itself meaningfully.
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
pub struct TimeSeconds(pub(crate) f64);

impl TimeSeconds {
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

    /// The raw value in seconds, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_seconds(self) -> f64 {
        self.0
    }

    /// Whether the duration is infinite — an unreachable sentinel rather than a real span.
    pub fn is_infinite(self) -> bool {
        self.0.is_infinite()
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
        debug_assert!(self.0 > 0.0, "expected positive TimeSeconds, got {self:?}");
        self
    }

    /// Convert to [`TimeMinutes`].
    pub fn to_minutes(self) -> TimeMinutes {
        TimeMinutes(self.0 / S_PER_MIN)
    }

    /// Convert to [`TimeHours`].
    pub fn to_hours(self) -> TimeHours {
        TimeHours(self.0 / S_PER_HOUR)
    }

    /// Convert to [`TimeMillis`].
    pub fn to_millis(self) -> TimeMillis {
        TimeMillis(self.0 * MS_PER_S)
    }
}

impl crate::conversion::UnitOfMeasure for TimeSeconds {
    type Dim = dim::Time;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <TimeSeconds as euclid_units::Measure>::DIMENSION
        .same(<dim::Time as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(TimeSeconds)),
);

impl core::fmt::Display for TimeSeconds {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for TimeSeconds {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "s";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for TimeSeconds {}

impl euclid_units::LinearQuantity for TimeSeconds {}

impl Time for TimeSeconds {
    type Length = LengthMeters;
    type Speed = SpeedMs;
    type Acceleration = AccelerationMetersPerSecondSquared;
    type Frequency = FrequencyHertz;

    fn times_speed(self, speed: Self::Speed) -> Self::Length {
        LengthMeters::from_raw(self.raw() * speed.raw())
    }
    fn times_acceleration(self, accel: Self::Acceleration) -> Self::Speed {
        SpeedMs::from_raw(self.raw() * accel.raw())
    }
    fn frequency(self) -> Self::Frequency {
        FrequencyHertz::from_raw(1.0 / self.raw())
    }
}

impl ::core::ops::Add for TimeSeconds {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for TimeSeconds {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for TimeSeconds {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for TimeSeconds {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for TimeSeconds {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for TimeSeconds {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<TimeSeconds> for f64 {
    type Output = TimeSeconds;
    fn mul(self, x: TimeSeconds) -> TimeSeconds {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for TimeSeconds {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for TimeSeconds {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
