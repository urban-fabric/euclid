use super::{KPH_PER_MPH, MS_PER_MPH};
use crate::conversion::dim;
use crate::{SpeedKph, SpeedMs};
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
pub struct SpeedMph(pub(crate) f64);

impl SpeedMph {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_mph(self) -> f64 {
        self.0
    }

    pub fn to_ms(self) -> SpeedMs {
        SpeedMs(self.0 * MS_PER_MPH)
    }

    pub fn to_kph(self) -> SpeedKph {
        SpeedKph(self.0 * KPH_PER_MPH)
    }
}

impl crate::conversion::UnitOfMeasure for SpeedMph {
    type Dim = dim::Speed;
    const TO_PRIMARY: f64 = MS_PER_MPH;
}

const _: () = assert!(
    <SpeedMph as euclid_units::Measure>::DIMENSION
        .same(<dim::Speed as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(SpeedMph)),
);

impl core::fmt::Display for SpeedMph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for SpeedMph {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Speed;
    const UNIT_NAME: &'static str = "mph";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for SpeedMph {}

impl euclid_units::LinearQuantity for SpeedMph {}

impl ::core::ops::Add for SpeedMph {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for SpeedMph {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for SpeedMph {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for SpeedMph {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for SpeedMph {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for SpeedMph {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<SpeedMph> for f64 {
    type Output = SpeedMph;
    fn mul(self, x: SpeedMph) -> SpeedMph {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for SpeedMph {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for SpeedMph {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
