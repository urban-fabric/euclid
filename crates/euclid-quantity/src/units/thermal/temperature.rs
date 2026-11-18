mod temperature_celsius;
mod temperature_fahrenheit;
mod temperature_kelvin;

pub use temperature_celsius::TemperatureCelsius;
pub use temperature_fahrenheit::TemperatureFahrenheit;
pub use temperature_kelvin::TemperatureKelvin;

const KELVIN_OFFSET_CELSIUS: f64 = 273.15;
const KELVIN_PER_FAHRENHEIT_DEGREE: f64 = 5.0 / 9.0;
const KELVIN_OFFSET_FAHRENHEIT: f64 = 255.372_222_222_222_2;
