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
