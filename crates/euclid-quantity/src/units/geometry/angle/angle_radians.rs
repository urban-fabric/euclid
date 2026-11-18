use crate::Measure;
use crate::conversion::dim;
use crate::{AngleDegrees, AngularVelocityRadiansPerSecond, TimeSeconds};
use bytemuck::{Pod, Zeroable};
use euclid_units::Angle;

#[repr(transparent)]
/// An angle in **radians**, the unit every trigonometric function here takes.
///
/// Not normalised on construction: `AngleRadians::new(7.0)` stays 7 rather than wrapping
/// to `7 − 2π`, because a winding count is sometimes the point.
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
pub struct AngleRadians(pub(crate) f64);

impl AngleRadians {
    /// The additive identity in this unit.
    pub const ZERO: Self = Self(0.0);

    /// Positive infinity, for use as an unreachable sentinel in a minimisation.
    ///
    /// Arithmetic on it follows IEEE-754 rather than saturating, so `INFINITY - INFINITY` is
    /// `NaN` and not zero.
    pub const INFINITY: Self = Self(f64::INFINITY);

    /// A half turn, `π` radians — 180°.
    pub const PI: Self = Self(std::f64::consts::PI);

    /// A full turn, `τ = 2π` radians — 360°.
    pub const TAU: Self = Self(std::f64::consts::TAU);

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

    /// Convert to [`AngleDegrees`].
    pub fn to_degrees(self) -> AngleDegrees {
        AngleDegrees(self.0.to_degrees())
    }

    /// The raw value in radians, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_radians(self) -> f64 {
        self.0
    }

    /// The raw value in radians, narrowed to `f32`.
    ///
    /// Lossy: about seven significant digits rather than sixteen, and values past `f32`'s range
    /// become infinite. For storage and interchange, not for further arithmetic.
    pub fn to_f32_radians(self) -> f32 {
        self.0 as f32
    }

    /// Magnitude, dropping the sign. `NaN` stays `NaN`.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl crate::conversion::UnitOfMeasure for AngleRadians {
    type Dim = dim::Angle;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <AngleRadians as euclid_units::Measure>::DIMENSION
        .same(<dim::Angle as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(AngleRadians)),
);

impl core::fmt::Display for AngleRadians {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for AngleRadians {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Angle;
    const UNIT_NAME: &'static str = "rad";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for AngleRadians {}

impl euclid_units::LinearQuantity for AngleRadians {}

impl Angle for AngleRadians {
    type AngularVelocity = AngularVelocityRadiansPerSecond;
    type Time = TimeSeconds;

    fn from_radians(radians: f64) -> Self {
        AngleRadians::from_raw(radians)
    }
    fn from_degrees(degrees: f64) -> Self {
        AngleRadians::from_raw(degrees.to_radians())
    }
    fn half_turn() -> Self {
        AngleRadians::from_raw(core::f64::consts::PI)
    }
    fn full_turn() -> Self {
        AngleRadians::from_raw(core::f64::consts::TAU)
    }
    fn radians(&self) -> f64 {
        self.raw()
    }
    fn degrees(&self) -> f64 {
        self.raw().to_degrees()
    }
    fn sin(&self) -> f64 {
        self.raw().sin()
    }
    fn cos(&self) -> f64 {
        self.raw().cos()
    }
    fn tan(&self) -> f64 {
        self.raw().tan()
    }
    fn normalized_unsigned(self) -> Self {
        AngleRadians::from_raw(self.raw().rem_euclid(core::f64::consts::TAU))
    }
    fn normalized_signed(self) -> Self {
        let tau = core::f64::consts::TAU;
        let mut wrapped = self.raw().rem_euclid(tau);
        if wrapped > core::f64::consts::PI {
            wrapped -= tau;
        }
        AngleRadians::from_raw(wrapped)
    }
    fn per_time(self, elapsed: Self::Time) -> Self::AngularVelocity {
        AngularVelocityRadiansPerSecond::from_raw(self.raw() / elapsed.raw())
    }
}

impl ::core::ops::Add for AngleRadians {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for AngleRadians {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for AngleRadians {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for AngleRadians {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for AngleRadians {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for AngleRadians {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<AngleRadians> for f64 {
    type Output = AngleRadians;
    fn mul(self, x: AngleRadians) -> AngleRadians {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for AngleRadians {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for AngleRadians {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
