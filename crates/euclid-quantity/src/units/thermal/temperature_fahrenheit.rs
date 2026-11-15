use super::{KELVIN_OFFSET_FAHRENHEIT, KELVIN_PER_FAHRENHEIT_DEGREE};
use crate::conversion::dim;
use crate::{TemperatureCelsius, TemperatureKelvin};
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
pub struct TemperatureFahrenheit(pub(crate) f64);

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

impl crate::conversion::UnitOfMeasure for TemperatureFahrenheit {
    type Dim = dim::Temperature;
    const TO_PRIMARY: f64 = KELVIN_PER_FAHRENHEIT_DEGREE;
    const PRIMARY_OFFSET: f64 = KELVIN_OFFSET_FAHRENHEIT;
}

const _: () = assert!(
    <TemperatureFahrenheit as euclid_units::Measure>::DIMENSION
        .same(<dim::Temperature as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(TemperatureFahrenheit)),
);

impl core::fmt::Display for TemperatureFahrenheit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for TemperatureFahrenheit {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Temperature;
    const UNIT_NAME: &'static str = "°F";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for TemperatureFahrenheit {}

impl euclid_units::Temperature for TemperatureFahrenheit {
    fn from_kelvin(kelvin: f64) -> Self {
        TemperatureKelvin::new(kelvin).to_fahrenheit()
    }

    fn from_celsius(celsius: f64) -> Self {
        TemperatureCelsius::new(celsius).to_fahrenheit()
    }

    fn from_fahrenheit(fahrenheit: f64) -> Self {
        Self::new(fahrenheit)
    }

    fn kelvin(&self) -> f64 {
        self.to_kelvin().as_kelvin()
    }

    fn celsius(&self) -> f64 {
        self.to_celsius().as_celsius()
    }

    fn fahrenheit(&self) -> f64 {
        self.as_fahrenheit()
    }

    fn absolute_zero() -> Self {
        TemperatureKelvin::ZERO.to_fahrenheit()
    }
}
