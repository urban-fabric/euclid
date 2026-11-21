use super::KELVIN_OFFSET_CELSIUS;
use crate::{TemperatureFahrenheit, TemperatureKelvin};
use bytemuck::{Pod, Zeroable};
use euclid_units::dim;

#[repr(transparent)]
/// A temperature in **degrees Celsius** — an **affine** scale, offset 273.15 from kelvin.
///
/// Affine means addition and scaling are not defined on it: `20 °C + 20 °C` has no
/// physical reading, and a *difference* of two Celsius temperatures is a kelvin
/// interval rather than a Celsius one. That is why this type does not implement the
/// linear quantity contract and [`crate::TemperatureKelvin`] does.
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
    /// Zero degrees Celsius — a **position on an affine scale**, not an additive identity.
    ///
    /// Adding it to a temperature does not leave that temperature unchanged, because
    /// addition is not defined on this scale at all. It is a named point (water's freezing
    /// point, or Fahrenheit's brine reference), which is a different kind of thing from
    /// [`crate::TemperatureKelvin::ZERO`] — absolute zero, where the scale genuinely begins.
    pub const ZERO: Self = Self(0.0);

    /// Wrap a raw `f64` already expressed in this unit.
    ///
    /// Asserted rather than checked: the caller is promising `v` is in this unit, and nothing
    /// here can verify that. Passing a value in the wrong unit is the one mistake the type
    /// system cannot catch at this boundary, which is why conversions exist.
    ///
    /// # Panics
    /// On `NaN`, in a **debug build only** — the check is a `debug_assert!`, so a release build
    /// accepts a `NaN` and propagates it silently through every later operation.
    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    /// The raw value in celsius, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_celsius(self) -> f64 {
        self.0
    }

    /// Convert to [`TemperatureKelvin`].
    ///
    /// An **affine** conversion: an offset as well as a scale, so it does not distribute
    /// over addition and a *difference* of two temperatures does not convert this way.
    pub fn to_kelvin(self) -> TemperatureKelvin {
        TemperatureKelvin(self.0 + KELVIN_OFFSET_CELSIUS)
    }

    /// Convert to [`TemperatureFahrenheit`].
    ///
    /// An **affine** conversion: an offset as well as a scale, so it does not distribute
    /// over addition and a *difference* of two temperatures does not convert this way.
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
