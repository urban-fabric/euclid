use super::{KELVIN_OFFSET_FAHRENHEIT, KELVIN_PER_FAHRENHEIT_DEGREE};
use crate::{TemperatureCelsius, TemperatureKelvin};
use bytemuck::{Pod, Zeroable};
use euclid_units::dim;

#[repr(transparent)]
/// A temperature in **degrees Fahrenheit** — affine, like
/// [`crate::TemperatureCelsius`], with both an offset and a scale
/// against kelvin.
///
/// The round trip through kelvin is ill-conditioned near the offset: converting 1 °F
/// passes through −17.2 °C and 255.9 K, so the absolute error of the large
/// intermediate lands on a small result. `tests/generated_conversions.rs` bounds it
/// against the largest magnitude the trip reaches rather than against the result.
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
    /// Zero degrees Fahrenheit — a **position on an affine scale**, not an additive identity.
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

    /// The raw value in fahrenheit, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_fahrenheit(self) -> f64 {
        self.0
    }

    /// Convert to [`TemperatureKelvin`].
    ///
    /// An **affine** conversion: an offset as well as a scale, so it does not distribute
    /// over addition and a *difference* of two temperatures does not convert this way.
    pub fn to_kelvin(self) -> TemperatureKelvin {
        TemperatureKelvin(self.0 * KELVIN_PER_FAHRENHEIT_DEGREE + KELVIN_OFFSET_FAHRENHEIT)
    }

    /// Convert to [`TemperatureCelsius`].
    ///
    /// An **affine** conversion: an offset as well as a scale, so it does not distribute
    /// over addition and a *difference* of two temperatures does not convert this way.
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
