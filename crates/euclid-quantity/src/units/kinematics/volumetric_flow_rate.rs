use crate::Measure;
use crate::conversion::dim;
use crate::{AreaSquareMeters, SpeedMs, TimeSeconds, VolumeCubicMeters};
use euclid_units::VolumetricFlowRate;

/// Volumetric flow rate in cubic meters per second (m^3/s), the coherent SI unit.
#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct VolumetricFlowRateCubicMetersPerSecond(pub(crate) f64);

impl VolumetricFlowRateCubicMetersPerSecond {
    /// The zero value of this unit.
    pub const ZERO: Self = Self(0.0);

    /// Construct a value in this unit. Debug-asserts the input is not `NaN`.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// This quantity's raw numeric value in its SI unit.
    pub const fn as_cubic_meters_per_second(self) -> f64 {
        self.0
    }

    /// Whether the value is finite (neither infinite nor `NaN`).
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Whether the value is strictly greater than zero.
    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    /// Whether the value is greater than or equal to zero.
    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    /// Whether the value is exactly zero.
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    /// The absolute value, in the same unit.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl crate::conversion::UnitOfMeasure for VolumetricFlowRateCubicMetersPerSecond {
    type Dim = dim::VolumetricFlowRate;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <VolumetricFlowRateCubicMetersPerSecond as euclid_units::Measure>::DIMENSION
        .same(<dim::VolumetricFlowRate as euclid_units::Dimensioned>::DIMENSION),
    concat!(
        "dimension mismatch for ",
        stringify!(VolumetricFlowRateCubicMetersPerSecond)
    ),
);

impl core::fmt::Display for VolumetricFlowRateCubicMetersPerSecond {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for VolumetricFlowRateCubicMetersPerSecond {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::VolumetricFlowRate;
    const UNIT_NAME: &'static str = "m^3/s";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for VolumetricFlowRateCubicMetersPerSecond {}

impl euclid_units::LinearQuantity for VolumetricFlowRateCubicMetersPerSecond {}

impl VolumetricFlowRate for VolumetricFlowRateCubicMetersPerSecond {
    type Volume = VolumeCubicMeters;
    type Time = TimeSeconds;
    type Area = AreaSquareMeters;
    type Speed = SpeedMs;

    fn times_time(self, elapsed: Self::Time) -> Self::Volume {
        VolumeCubicMeters::from_raw(self.raw() * elapsed.raw())
    }
    fn per_area(self, section: Self::Area) -> Self::Speed {
        SpeedMs::from_raw(self.raw() / section.raw())
    }
    fn per_speed(self, speed: Self::Speed) -> Self::Area {
        AreaSquareMeters::from_raw(self.raw() / speed.raw())
    }
}

impl ::core::ops::Add for VolumetricFlowRateCubicMetersPerSecond {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for VolumetricFlowRateCubicMetersPerSecond {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for VolumetricFlowRateCubicMetersPerSecond {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for VolumetricFlowRateCubicMetersPerSecond {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for VolumetricFlowRateCubicMetersPerSecond {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for VolumetricFlowRateCubicMetersPerSecond {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<VolumetricFlowRateCubicMetersPerSecond> for f64 {
    type Output = VolumetricFlowRateCubicMetersPerSecond;
    fn mul(
        self,
        x: VolumetricFlowRateCubicMetersPerSecond,
    ) -> VolumetricFlowRateCubicMetersPerSecond {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for VolumetricFlowRateCubicMetersPerSecond {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for VolumetricFlowRateCubicMetersPerSecond {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
