#[test]
fn heat_capacity_joules_per_kelvin_dimensional_algebra() {
    use euclid_units::HeatCapacity;
    times_method!(
        crate::HeatCapacityJoulesPerKelvin,
        HeatCapacity,
        times_temperature,
        Temperature,
        Energy
    );
    per_method!(
        crate::HeatCapacityJoulesPerKelvin,
        HeatCapacity,
        per_mass,
        Mass,
        SpecificHeatCapacity
    );
}

#[test]
fn heat_flux_watts_per_square_meter_dimensional_algebra() {
    use euclid_units::HeatFlux;
    times_method!(
        crate::HeatFluxWattsPerSquareMeter,
        HeatFlux,
        times_area,
        Area,
        Power
    );
}

#[test]
fn specific_heat_capacity_joules_per_kilogram_kelvin_dimensional_algebra() {
    use euclid_units::SpecificHeatCapacity;
    times_method!(
        crate::SpecificHeatCapacityJoulesPerKilogramKelvin,
        SpecificHeatCapacity,
        times_mass,
        Mass,
        HeatCapacity
    );
    times_method!(
        crate::SpecificHeatCapacityJoulesPerKilogramKelvin,
        SpecificHeatCapacity,
        times_temperature,
        Temperature,
        SpecificEnergy
    );
}
