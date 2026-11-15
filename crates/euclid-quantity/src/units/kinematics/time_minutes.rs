use super::S_PER_MIN;
use crate::TimeSeconds;
use crate::conversion::dim;
use bytemuck::{Pod, Zeroable};

#[repr(transparent)]
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
pub struct TimeMinutes(pub(crate) f64);

impl TimeMinutes {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_minutes(self) -> f64 {
        self.0
    }

    pub fn to_seconds(self) -> TimeSeconds {
        TimeSeconds(self.0 * S_PER_MIN)
    }
}

impl crate::conversion::UnitOfMeasure for TimeMinutes {
    type Dim = dim::Time;
    const TO_PRIMARY: f64 = S_PER_MIN;
}

const _: () = assert!(
    <TimeMinutes as euclid_units::Measure>::DIMENSION
        .same(<dim::Time as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(TimeMinutes)),
);

impl core::fmt::Display for TimeMinutes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for TimeMinutes {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "min";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for TimeMinutes {}

impl euclid_units::LinearQuantity for TimeMinutes {}

impl ::core::ops::Add for TimeMinutes {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for TimeMinutes {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for TimeMinutes {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for TimeMinutes {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for TimeMinutes {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for TimeMinutes {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<TimeMinutes> for f64 {
    type Output = TimeMinutes;
    fn mul(self, x: TimeMinutes) -> TimeMinutes {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for TimeMinutes {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for TimeMinutes {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
