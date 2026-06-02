use bytemuck::{Pod, Zeroable};

use super::{Dimension, Quantity};

pub(crate) const KELVIN_OFFSET_CELSIUS: f64 = 273.15;
pub(crate) const KELVIN_PER_FAHRENHEIT_DEGREE: f64 = 5.0 / 9.0;
pub(crate) const KELVIN_OFFSET_FAHRENHEIT: f64 = 255.372_222_222_222_2;

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
pub struct TemperatureKelvin(f64);

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
pub struct TemperatureCelsius(f64);

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
pub struct TemperatureFahrenheit(f64);

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

impl TemperatureCelsius {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_celsius(self) -> f64 {
        self.0
    }

    pub fn to_kelvin(self) -> TemperatureKelvin {
        TemperatureKelvin(self.0 + KELVIN_OFFSET_CELSIUS)
    }

    pub fn to_fahrenheit(self) -> TemperatureFahrenheit {
        self.to_kelvin().to_fahrenheit()
    }
}

impl TemperatureFahrenheit {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_fahrenheit(self) -> f64 {
        self.0
    }

    pub fn to_kelvin(self) -> TemperatureKelvin {
        TemperatureKelvin(self.0 * KELVIN_PER_FAHRENHEIT_DEGREE + KELVIN_OFFSET_FAHRENHEIT)
    }

    pub fn to_celsius(self) -> TemperatureCelsius {
        self.to_kelvin().to_celsius()
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

impl Quantity for TemperatureCelsius {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Temperature;
    const UNIT_NAME: &'static str = "C";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl Quantity for TemperatureFahrenheit {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Temperature;
    const UNIT_NAME: &'static str = "F";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

#[cfg(test)]
mod tests;
