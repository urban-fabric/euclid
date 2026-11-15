use super::MPS2_PER_CMPS2;
use crate::AccelerationMetersPerSecondSquared;
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
pub struct AccelerationCentimetersPerSecondSquared(pub(crate) f64);

impl AccelerationCentimetersPerSecondSquared {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_centimeters_per_second_squared(self) -> f64 {
        self.0
    }

    pub fn to_meters_per_second_squared(self) -> AccelerationMetersPerSecondSquared {
        AccelerationMetersPerSecondSquared(self.0 * MPS2_PER_CMPS2)
    }
}

impl crate::conversion::UnitOfMeasure for AccelerationCentimetersPerSecondSquared {
    type Dim = dim::Acceleration;
    const TO_PRIMARY: f64 = MPS2_PER_CMPS2;
}

const _: () = assert!(
    <AccelerationCentimetersPerSecondSquared as euclid_units::Measure>::DIMENSION
        .same(<dim::Acceleration as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(AccelerationCentimetersPerSecondSquared)
    ),
);

impl core::fmt::Display for AccelerationCentimetersPerSecondSquared {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for AccelerationCentimetersPerSecondSquared {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Acceleration;
    const UNIT_NAME: &'static str = "cm/s^2";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for AccelerationCentimetersPerSecondSquared {}

impl euclid_units::LinearQuantity for AccelerationCentimetersPerSecondSquared {}

impl ::core::ops::Add for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for AccelerationCentimetersPerSecondSquared {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for AccelerationCentimetersPerSecondSquared {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<AccelerationCentimetersPerSecondSquared> for f64 {
    type Output = AccelerationCentimetersPerSecondSquared;
    fn mul(
        self,
        x: AccelerationCentimetersPerSecondSquared,
    ) -> AccelerationCentimetersPerSecondSquared {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for AccelerationCentimetersPerSecondSquared {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
