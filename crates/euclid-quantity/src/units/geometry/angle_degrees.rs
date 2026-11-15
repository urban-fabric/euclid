use crate::AngleRadians;
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
pub struct AngleDegrees(pub(crate) f64);

impl AngleDegrees {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub fn to_radians(self) -> AngleRadians {
        AngleRadians(self.0.to_radians())
    }

    pub const fn as_degrees(self) -> f64 {
        self.0
    }
}

impl crate::conversion::UnitOfMeasure for AngleDegrees {
    type Dim = dim::Angle;
    const TO_PRIMARY: f64 = std::f64::consts::PI / 180.0;
}

const _: () = assert!(
    <AngleDegrees as euclid_units::Measure>::DIMENSION
        .same(<dim::Angle as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(AngleDegrees)),
);

impl core::fmt::Display for AngleDegrees {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for AngleDegrees {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Angle;
    const UNIT_NAME: &'static str = "deg";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for AngleDegrees {}

impl euclid_units::LinearQuantity for AngleDegrees {}

impl ::core::ops::Add for AngleDegrees {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for AngleDegrees {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for AngleDegrees {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for AngleDegrees {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for AngleDegrees {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for AngleDegrees {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<AngleDegrees> for f64 {
    type Output = AngleDegrees;
    fn mul(self, x: AngleDegrees) -> AngleDegrees {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for AngleDegrees {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for AngleDegrees {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
