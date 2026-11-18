full_surface!(
    heat_capacity_joules_per_kelvin,
    crate::HeatCapacityJoulesPerKelvin,
    as_joules_per_kelvin
);
full_surface!(
    heat_flux_watts_per_square_meter,
    crate::HeatFluxWattsPerSquareMeter,
    as_watts_per_square_meter
);
full_surface!(
    specific_heat_capacity_joules_per_kilogram_kelvin,
    crate::SpecificHeatCapacityJoulesPerKilogramKelvin,
    as_joules_per_kilogram_kelvin
);
#[test]
fn temperature_celsius() {
    accessor_agrees!(crate::TemperatureCelsius, as_celsius);
    display_distinguishes!(crate::TemperatureCelsius);
}

#[test]
fn temperature_fahrenheit() {
    accessor_agrees!(crate::TemperatureFahrenheit, as_fahrenheit);
    display_distinguishes!(crate::TemperatureFahrenheit);
}

#[test]
fn temperature_kelvin() {
    accessor_agrees!(crate::TemperatureKelvin, as_kelvin);
    display_distinguishes!(crate::TemperatureKelvin);
}
