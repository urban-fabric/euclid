#[test]
fn capacitance_farads_dimensional_algebra() {
    use euclid_units::Capacitance;
    times_method!(
        crate::CapacitanceFarads,
        Capacitance,
        times_voltage,
        Voltage,
        ElectricCharge
    );
    times_method!(
        crate::CapacitanceFarads,
        Capacitance,
        times_resistance,
        Resistance,
        Time
    );
}

#[test]
fn current_density_amperes_per_square_meter_dimensional_algebra() {
    use euclid_units::CurrentDensity;
    times_method!(
        crate::CurrentDensityAmperesPerSquareMeter,
        CurrentDensity,
        times_area,
        Area,
        ElectricCurrent
    );
}

#[test]
fn charge_coulombs_dimensional_algebra() {
    use euclid_units::ElectricCharge;
    per_method!(
        crate::ChargeCoulombs,
        ElectricCharge,
        per_time,
        Time,
        ElectricCurrent
    );
    per_method!(
        crate::ChargeCoulombs,
        ElectricCharge,
        per_current,
        ElectricCurrent,
        Time
    );
    times_method!(
        crate::ChargeCoulombs,
        ElectricCharge,
        times_voltage,
        Voltage,
        Energy
    );
    per_method!(
        crate::ChargeCoulombs,
        ElectricCharge,
        per_capacitance,
        Capacitance,
        Voltage
    );
    per_method!(
        crate::ChargeCoulombs,
        ElectricCharge,
        per_voltage,
        Voltage,
        Capacitance
    );
    times_method!(
        crate::ChargeCoulombs,
        ElectricCharge,
        times_electric_field,
        ElectricField,
        Force
    );
}

#[test]
fn current_amperes_dimensional_algebra() {
    use euclid_units::ElectricCurrent;
    times_method!(
        crate::CurrentAmperes,
        ElectricCurrent,
        times_time,
        Time,
        ElectricCharge
    );
    times_method!(
        crate::CurrentAmperes,
        ElectricCurrent,
        times_resistance,
        Resistance,
        Voltage
    );
    times_method!(
        crate::CurrentAmperes,
        ElectricCurrent,
        times_voltage,
        Voltage,
        Power
    );
    times_method!(
        crate::CurrentAmperes,
        ElectricCurrent,
        times_inductance,
        Inductance,
        MagneticFlux
    );
    per_method!(
        crate::CurrentAmperes,
        ElectricCurrent,
        per_area,
        Area,
        CurrentDensity
    );
}

#[test]
fn electric_field_volts_per_meter_dimensional_algebra() {
    use euclid_units::ElectricField;
    times_method!(
        crate::ElectricFieldVoltsPerMeter,
        ElectricField,
        times_length,
        Length,
        Voltage
    );
    times_method!(
        crate::ElectricFieldVoltsPerMeter,
        ElectricField,
        times_charge,
        ElectricCharge,
        Force
    );
}

#[test]
fn inductance_henries_dimensional_algebra() {
    use euclid_units::Inductance;
    times_method!(
        crate::InductanceHenries,
        Inductance,
        times_current,
        ElectricCurrent,
        MagneticFlux
    );
    per_method!(
        crate::InductanceHenries,
        Inductance,
        per_resistance,
        Resistance,
        Time
    );
    per_method!(
        crate::InductanceHenries,
        Inductance,
        per_time,
        Time,
        Resistance
    );
}

#[test]
fn magnetic_flux_webers_dimensional_algebra() {
    use euclid_units::MagneticFlux;
    per_method!(
        crate::MagneticFluxWebers,
        MagneticFlux,
        per_current,
        ElectricCurrent,
        Inductance
    );
    per_method!(
        crate::MagneticFluxWebers,
        MagneticFlux,
        per_time,
        Time,
        Voltage
    );
    per_method!(
        crate::MagneticFluxWebers,
        MagneticFlux,
        per_area,
        Area,
        MagneticFluxDensity
    );
}

#[test]
fn magnetic_flux_density_teslas_dimensional_algebra() {
    use euclid_units::MagneticFluxDensity;
    times_method!(
        crate::MagneticFluxDensityTeslas,
        MagneticFluxDensity,
        times_area,
        Area,
        MagneticFlux
    );
}

#[test]
fn resistance_ohms_dimensional_algebra() {
    use euclid_units::Resistance;
    times_method!(
        crate::ResistanceOhms,
        Resistance,
        times_current,
        ElectricCurrent,
        Voltage
    );
    times_method!(
        crate::ResistanceOhms,
        Resistance,
        times_time,
        Time,
        Inductance
    );
}

#[test]
fn voltage_volts_dimensional_algebra() {
    use euclid_units::Voltage;
    per_method!(
        crate::VoltageVolts,
        Voltage,
        per_current,
        ElectricCurrent,
        Resistance
    );
    per_method!(
        crate::VoltageVolts,
        Voltage,
        per_resistance,
        Resistance,
        ElectricCurrent
    );
    times_method!(
        crate::VoltageVolts,
        Voltage,
        times_current,
        ElectricCurrent,
        Power
    );
    times_method!(
        crate::VoltageVolts,
        Voltage,
        times_charge,
        ElectricCharge,
        Energy
    );
    per_method!(
        crate::VoltageVolts,
        Voltage,
        per_length,
        Length,
        ElectricField
    );
    times_method!(crate::VoltageVolts, Voltage, times_time, Time, MagneticFlux);
}
