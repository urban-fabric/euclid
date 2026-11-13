use super::{KELVIN_OFFSET_CELSIUS, KELVIN_OFFSET_FAHRENHEIT, KELVIN_PER_FAHRENHEIT_DEGREE};
use crate::conversion::dim;
use crate::{Dimension, Quantity};
use crate::{TemperatureCelsius, TemperatureFahrenheit};
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
pub struct TemperatureKelvin(pub(crate) f64);

impl TemperatureKelvin {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_kelvin(self) -> f64 {
        self.0
    }

    pub fn to_celsius(self) -> TemperatureCelsius {
        TemperatureCelsius(self.0 - KELVIN_OFFSET_CELSIUS)
    }

    pub fn to_fahrenheit(self) -> TemperatureFahrenheit {
        TemperatureFahrenheit((self.0 - KELVIN_OFFSET_FAHRENHEIT) / KELVIN_PER_FAHRENHEIT_DEGREE)
    }
}

impl Quantity for TemperatureKelvin {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Temperature;
    const UNIT_NAME: &'static str = "K";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl crate::conversion::UnitOfMeasure for TemperatureKelvin {
    type Dim = dim::Temperature;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <TemperatureKelvin as crate::Quantity>::DIMENSION
        .same(<dim::Temperature as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(TemperatureKelvin)),
);

impl core::fmt::Display for TemperatureKelvin {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for TemperatureKelvin {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = <TemperatureKelvin as crate::Quantity>::DIMENSION;
    const UNIT_NAME: &'static str = <TemperatureKelvin as crate::Quantity>::UNIT_NAME;
    fn raw(&self) -> f64 {
        <Self as crate::Quantity>::raw(*self)
    }
    fn from_raw(value: f64) -> Self {
        <Self as crate::Quantity>::from_raw(value)
    }
}

impl euclid_units::ScalarQuantity for TemperatureKelvin {
    fn min(self, other: Self) -> Self {
        <Self as crate::Quantity>::from_raw(
            <Self as crate::Quantity>::raw(self).min(<Self as crate::Quantity>::raw(other)),
        )
    }
    fn max(self, other: Self) -> Self {
        <Self as crate::Quantity>::from_raw(
            <Self as crate::Quantity>::raw(self).max(<Self as crate::Quantity>::raw(other)),
        )
    }
    fn clamp(self, lo: Self, hi: Self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).clamp(
            <Self as crate::Quantity>::raw(lo),
            <Self as crate::Quantity>::raw(hi),
        ))
    }
    fn floor(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).floor())
    }
    fn ceil(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).ceil())
    }
    fn round(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).round())
    }
    fn trunc(self) -> Self {
        <Self as crate::Quantity>::from_raw(<Self as crate::Quantity>::raw(self).trunc())
    }
    fn is_finite(&self) -> bool {
        <Self as crate::Quantity>::raw(*self).is_finite()
    }
    fn is_nan(&self) -> bool {
        <Self as crate::Quantity>::raw(*self).is_nan()
    }
    fn is_infinite(&self) -> bool {
        <Self as crate::Quantity>::raw(*self).is_infinite()
    }
    fn approx_eq(&self, other: &Self, tolerance: &Self) -> bool {
        (<Self as crate::Quantity>::raw(*self) - <Self as crate::Quantity>::raw(*other)).abs()
            <= <Self as crate::Quantity>::raw(*tolerance).abs()
    }
}

impl euclid_units::Temperature for TemperatureKelvin {
    fn from_kelvin(kelvin: f64) -> Self {
        Self::new(kelvin)
    }

    fn from_celsius(celsius: f64) -> Self {
        TemperatureCelsius::new(celsius).to_kelvin()
    }

    fn from_fahrenheit(fahrenheit: f64) -> Self {
        TemperatureFahrenheit::new(fahrenheit).to_kelvin()
    }

    fn kelvin(&self) -> f64 {
        self.as_kelvin()
    }

    fn celsius(&self) -> f64 {
        self.to_celsius().as_celsius()
    }

    fn fahrenheit(&self) -> f64 {
        self.to_fahrenheit().as_fahrenheit()
    }

    fn absolute_zero() -> Self {
        Self::ZERO
    }
}
