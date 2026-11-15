use super::{KELVIN_OFFSET_CELSIUS, KELVIN_OFFSET_FAHRENHEIT, KELVIN_PER_FAHRENHEIT_DEGREE};
use crate::conversion::dim;
use crate::{TemperatureCelsius, TemperatureFahrenheit};
use bytemuck::{Pod, Zeroable};

#[repr(transparent)]
/// A temperature in **kelvin**, the SI base unit and the anchor for the temperature
/// dimension.
///
/// The only member of the temperature family whose zero is the dimension's zero, which
/// is what makes it the one that can be scaled: twice 100 K is 200 K, where twice
/// 100 °C is meaningless.
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
    /// Absolute zero, which is both the additive identity and the physical floor of the
    /// scale.
    ///
    /// That coincidence is what makes kelvin the linear member of the temperature family:
    /// scaling is meaningful because the zero is the dimension's zero. The affine siblings'
    /// `ZERO` constants are named points instead — see
    /// [`crate::TemperatureCelsius::ZERO`].
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

    /// The raw value in kelvin, discarding the dimension.
    ///
    /// The escape hatch out of the units system; prefer the dimensional operations and reach
    /// for this at an edge that does not speak units.
    pub const fn as_kelvin(self) -> f64 {
        self.0
    }

    /// Convert to [`TemperatureCelsius`].
    ///
    /// An **affine** conversion: an offset as well as a scale, so it does not distribute
    /// over addition and a *difference* of two temperatures does not convert this way.
    pub fn to_celsius(self) -> TemperatureCelsius {
        TemperatureCelsius(self.0 - KELVIN_OFFSET_CELSIUS)
    }

    /// Convert to [`TemperatureFahrenheit`].
    ///
    /// An **affine** conversion: an offset as well as a scale, so it does not distribute
    /// over addition and a *difference* of two temperatures does not convert this way.
    pub fn to_fahrenheit(self) -> TemperatureFahrenheit {
        TemperatureFahrenheit((self.0 - KELVIN_OFFSET_FAHRENHEIT) / KELVIN_PER_FAHRENHEIT_DEGREE)
    }
}

impl crate::conversion::UnitOfMeasure for TemperatureKelvin {
    type Dim = dim::Temperature;
    const TO_PRIMARY: f64 = 1.0;
}

const _: () = assert!(
    <TemperatureKelvin as euclid_units::Measure>::DIMENSION
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
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Temperature;
    const UNIT_NAME: &'static str = "K";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for TemperatureKelvin {}

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
