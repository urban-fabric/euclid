//! Thermal dimensions: [`Temperature`], [`HeatCapacity`], [`ThermalConductivity`].
//!
//! [`Temperature`] is the SI base dimension `[Θ]`, and — unlike every other dimension here — it
//! is an **absolute** scalar: it implements [`ScalarQuantity`] but not
//! [`LinearQuantity`], because adding two absolute temperatures or scaling
//! one is not meaningful (only differences are). The `energy = heat capacity × temperature`
//! relation therefore lives on [`HeatCapacity`] / [`Energy`], which accept the
//! temperature operand as an *interval* in the carrier's units.

use super::{Energy, LinearQuantity, ScalarQuantity};

/// A temperature-dimensioned quantity (`[Θ]`, the SI base dimension, the kelvin). Absolute, so
/// it is a [`ScalarQuantity`] but not a [`LinearQuantity`]: only temperature *differences*
/// support addition and scaling, and those are carried as the carrier's raw value.
pub trait Temperature: ScalarQuantity {
    /// Wrap a carrier value already in kelvin.
    fn from_kelvin(kelvin: Self::Repr) -> Self;
    /// Wrap a carrier value in degrees Celsius, converting to kelvin.
    fn from_celsius(celsius: Self::Repr) -> Self;
    /// Wrap a carrier value in degrees Fahrenheit, converting to kelvin.
    fn from_fahrenheit(fahrenheit: Self::Repr) -> Self;

    /// The temperature in kelvin.
    fn kelvin(&self) -> Self::Repr;
    /// The temperature in degrees Celsius.
    fn celsius(&self) -> Self::Repr;
    /// The temperature in degrees Fahrenheit.
    fn fahrenheit(&self) -> Self::Repr;

    /// Absolute zero, `0 K`.
    fn absolute_zero() -> Self;
}

/// A heat-capacity / entropy dimensioned quantity (`[L² M T⁻² Θ⁻¹]`, joules per kelvin).
/// `heat capacity × temperature = energy`, where the temperature is read as an interval.
pub trait HeatCapacity: LinearQuantity {
    /// The temperature interval that pairs with this heat capacity.
    type Temperature: Temperature<Repr = Self::Repr>;
    /// The energy produced by `heat capacity × temperature`.
    type Energy: Energy<Repr = Self::Repr>;

    /// `heat capacity × temperature = energy`. The temperature operand is interpreted as an
    /// interval (a `ΔT`) in the carrier's units.
    fn times_temperature(self, delta: Self::Temperature) -> Self::Energy;
}

/// A thermal-conductivity dimensioned quantity (`[L M T⁻³ Θ⁻¹]`, watts per metre-kelvin).
///
/// Tracked as a marker — it refines [`LinearQuantity`] with its dimension constant but carries no
/// single-operand cross-dimensional product into another *named* dimension in this set (Fourier's
/// law combines it with an area, a length, and a temperature gradient at once). A consumer that
/// needs that algebra adds it in its own impl.
pub trait ThermalConductivity: LinearQuantity {}
