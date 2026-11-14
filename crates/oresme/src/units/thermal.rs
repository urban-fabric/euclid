//! Thermal units: temperature, heat capacity, specific heat capacity, and heat flux.

mod heat_capacity;
mod heat_flux;
mod specific_heat_capacity;
mod temperature_celsius;
mod temperature_fahrenheit;
mod temperature_kelvin;

pub use heat_capacity::*;
pub use heat_flux::*;
pub use specific_heat_capacity::*;
pub use temperature_celsius::*;
pub use temperature_fahrenheit::*;
pub use temperature_kelvin::*;

pub(crate) const KELVIN_OFFSET_CELSIUS: f64 = 273.15;
pub(crate) const KELVIN_PER_FAHRENHEIT_DEGREE: f64 = 5.0 / 9.0;
pub(crate) const KELVIN_OFFSET_FAHRENHEIT: f64 = 255.372_222_222_222_2;

#[cfg(test)]
mod temperature_tests {
    use crate::{
        Quantity, ScalarQuantity, TemperatureCelsius, TemperatureFahrenheit, TemperatureKelvin,
    };

    #[test]
    fn celsius_and_kelvin_round_trip() {
        let c = TemperatureCelsius::new(25.0);
        let k = c.to_kelvin();
        assert_eq!(k, TemperatureKelvin::new(298.15));
        assert_eq!(k.to_celsius(), c);
    }

    #[test]
    fn fahrenheit_conversions_use_offset_and_scale() {
        let freezing = TemperatureCelsius::new(0.0);
        assert!(freezing.to_fahrenheit().approx_eq(
            TemperatureFahrenheit::new(32.0),
            TemperatureFahrenheit::new(1e-12)
        ));
        assert!(TemperatureFahrenheit::new(212.0).to_celsius().approx_eq(
            TemperatureCelsius::new(100.0),
            TemperatureCelsius::new(1e-12)
        ));
    }

    #[test]
    fn temperature_quantity_impls() {
        assert_eq!(TemperatureKelvin::DIMENSION, crate::Dimension::Temperature);
        assert_eq!(TemperatureKelvin::UNIT_NAME, "K");
        assert_eq!(TemperatureCelsius::UNIT_NAME, "°C");
        assert_eq!(TemperatureFahrenheit::UNIT_NAME, "°F");
    }
}
