use super::{MS_PER_KPH, MS_PER_MPH};
use crate::Measure;
use crate::conversion::dim;
use crate::{AccelerationMetersPerSecondSquared, LengthMeters, SpeedKph, SpeedMph, TimeSeconds};
use bytemuck::{Pod, Zeroable};
use euclid_units::Speed;

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
pub struct SpeedMs(pub(crate) f64);

impl SpeedMs {
    pub const ZERO: Self = Self(0.0);

    pub const INFINITY: Self = Self(f64::INFINITY);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_ms(self) -> f64 {
        self.0
    }

    pub fn to_kph(self) -> SpeedKph {
        SpeedKph(self.0 / MS_PER_KPH)
    }

    pub fn to_mph(self) -> SpeedMph {
        SpeedMph(self.0 / MS_PER_MPH)
    }
}

impl crate::conversion::UnitOfMeasure for SpeedMs {
    type Dim = dim::Speed;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <SpeedMs as euclid_units::Measure>::DIMENSION
        .same(<dim::Speed as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(SpeedMs)),
);

impl core::fmt::Display for SpeedMs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for SpeedMs {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Speed;
    const UNIT_NAME: &'static str = "m/s";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for SpeedMs {}

impl euclid_units::LinearQuantity for SpeedMs {}

impl Speed for SpeedMs {
    type Length = LengthMeters;
    type Time = TimeSeconds;
    type Acceleration = AccelerationMetersPerSecondSquared;

    fn times_time(self, elapsed: Self::Time) -> Self::Length {
        LengthMeters::from_raw(self.raw() * elapsed.raw())
    }
    fn per_time(self, elapsed: Self::Time) -> Self::Acceleration {
        AccelerationMetersPerSecondSquared::from_raw(self.raw() / elapsed.raw())
    }
    fn per_acceleration(self, accel: Self::Acceleration) -> Self::Time {
        TimeSeconds::from_raw(self.raw() / accel.raw())
    }
}

impl ::core::ops::Add for SpeedMs {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for SpeedMs {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for SpeedMs {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for SpeedMs {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for SpeedMs {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for SpeedMs {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<SpeedMs> for f64 {
    type Output = SpeedMs;
    fn mul(self, x: SpeedMs) -> SpeedMs {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for SpeedMs {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for SpeedMs {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
