//! Thermal dimensions: [`Temperature`], [`HeatCapacity`], [`ThermalConductivity`],
//! [`SpecificHeatCapacity`], [`ThermalResistance`], [`HeatFlux`].
//!
//! [`Temperature`] is the SI base dimension `[Θ]`, and — unlike every other dimension here — it
//! is an **absolute** scalar: it implements [`ScalarQuantity`] but not [`LinearQuantity`], because
//! adding two absolute temperatures or scaling one is not meaningful (only differences are). The
//! `energy = heat capacity × temperature` relation therefore lives on [`HeatCapacity`] /
//! [`Energy`], which accept the temperature operand as an *interval* in the carrier's units; the
//! same convention applies to [`SpecificHeatCapacity`] and [`ThermalResistance`] below.

use super::{Area, Energy, LinearQuantity, Mass, Power, ScalarQuantity, SpecificEnergy};

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
/// `heat capacity × temperature = energy`, where the temperature is read as an interval;
/// `heat capacity / mass = specific heat capacity`.
pub trait HeatCapacity: LinearQuantity {
    /// The temperature interval that pairs with this heat capacity.
    type Temperature: Temperature<Repr = Self::Repr>;
    /// The energy produced by `heat capacity × temperature`.
    type Energy: Energy<Repr = Self::Repr>;
    /// The mass that pairs with this heat capacity.
    type Mass: Mass<Repr = Self::Repr>;
    /// The specific heat capacity produced by `heat capacity / mass`.
    type SpecificHeatCapacity: SpecificHeatCapacity<Repr = Self::Repr>;

    /// `heat capacity × temperature = energy`. The temperature operand is interpreted as an
    /// interval (a `ΔT`) in the carrier's units.
    fn times_temperature(self, delta: Self::Temperature) -> Self::Energy;
    /// `heat capacity / mass = specific heat capacity`.
    fn per_mass(self, mass: Self::Mass) -> Self::SpecificHeatCapacity;
}

/// A thermal-conductivity dimensioned quantity (`[L M T⁻³ Θ⁻¹]`, watts per metre-kelvin).
///
/// Tracked as a marker — it refines [`LinearQuantity`] with its dimension constant but carries no
/// single-operand cross-dimensional product into another *named* dimension in this set (Fourier's
/// law combines it with an area, a length, and a temperature gradient at once). A consumer that
/// needs that algebra adds it in its own impl.
pub trait ThermalConductivity: LinearQuantity {}

/// A specific-heat-capacity dimensioned quantity (`[L² T⁻² Θ⁻¹]`, joules per kilogram-kelvin). Heat
/// capacity per unit mass — a material's intensive thermal property. `specific heat × mass = heat
/// capacity`, `specific heat × temperature = specific energy` (the temperature read as an interval).
pub trait SpecificHeatCapacity: LinearQuantity {
    /// The mass that pairs with this specific heat capacity.
    type Mass: Mass<Repr = Self::Repr>;
    /// The heat capacity produced by `specific heat × mass`.
    type HeatCapacity: HeatCapacity<Repr = Self::Repr>;
    /// The temperature interval that pairs with this specific heat capacity.
    type Temperature: Temperature<Repr = Self::Repr>;
    /// The specific energy produced by `specific heat × temperature`.
    type SpecificEnergy: SpecificEnergy<Repr = Self::Repr>;

    /// `specific heat × mass = heat capacity`.
    fn times_mass(self, mass: Self::Mass) -> Self::HeatCapacity;
    /// `specific heat × temperature = specific energy`. The temperature operand is interpreted as
    /// an interval (a `ΔT`) in the carrier's units.
    fn times_temperature(self, delta: Self::Temperature) -> Self::SpecificEnergy;
}

/// A thermal-resistance dimensioned quantity (`[L⁻² M⁻¹ T³ Θ]`, kelvin per watt). The thermal
/// analogue of electrical resistance: `thermal resistance × power = temperature` (the temperature
/// drop `ΔT = R·Q̇` across the resistance, read as an interval).
pub trait ThermalResistance: LinearQuantity {
    /// The power (heat-flow rate) that pairs with this thermal resistance.
    type Power: Power<Repr = Self::Repr>;
    /// The temperature interval produced by `thermal resistance × power`.
    type Temperature: Temperature<Repr = Self::Repr>;

    /// `thermal resistance × power = temperature` (`ΔT = R·Q̇`), the result read as an interval
    /// in the carrier's units.
    fn times_power(self, power: Self::Power) -> Self::Temperature;
}

/// A heat-flux (heat-flux density) dimensioned quantity (`[M T⁻³]`, watts per square metre). Power
/// per unit area through a surface. `heat flux × area = power`.
pub trait HeatFlux: LinearQuantity {
    /// The area that pairs with this heat flux.
    type Area: Area<Repr = Self::Repr>;
    /// The power produced by `heat flux × area`.
    type Power: Power<Repr = Self::Repr>;

    /// `heat flux × area = power`.
    fn times_area(self, area: Self::Area) -> Self::Power;
}
