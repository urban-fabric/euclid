use super::KELVIN_OFFSET_CELSIUS;
use crate::conversion::dim;
use crate::{TemperatureFahrenheit, TemperatureKelvin};
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
pub struct TemperatureCelsius(pub(crate) f64);

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

impl crate::conversion::UnitOfMeasure for TemperatureCelsius {
    type Dim = dim::Temperature;
    const TO_PRIMARY: f64 = 1.0;
    const PRIMARY_OFFSET: f64 = KELVIN_OFFSET_CELSIUS;
}

const _: () = assert!(
    <TemperatureCelsius as euclid_units::Measure>::DIMENSION
        .same(<dim::Temperature as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(TemperatureCelsius)),
);

impl core::fmt::Display for TemperatureCelsius {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for TemperatureCelsius {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Temperature;
    const UNIT_NAME: &'static str = "°C";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for TemperatureCelsius {}

impl euclid_units::Temperature for TemperatureCelsius {
    fn from_kelvin(kelvin: f64) -> Self {
        TemperatureKelvin::new(kelvin).to_celsius()
    }

    fn from_celsius(celsius: f64) -> Self {
        Self::new(celsius)
    }

    fn from_fahrenheit(fahrenheit: f64) -> Self {
        TemperatureFahrenheit::new(fahrenheit).to_celsius()
    }

    fn kelvin(&self) -> f64 {
        self.to_kelvin().as_kelvin()
    }

    fn celsius(&self) -> f64 {
        self.as_celsius()
    }

    fn fahrenheit(&self) -> f64 {
        self.to_fahrenheit().as_fahrenheit()
    }

    fn absolute_zero() -> Self {
        TemperatureKelvin::ZERO.to_celsius()
    }
}
