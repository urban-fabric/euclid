#[test]
fn angular_momentum_kilogram_meters_squared_per_second_dimensional_algebra() {
    use euclid_units::AngularMomentum;
    per_method!(
        crate::AngularMomentumKilogramMetersSquaredPerSecond,
        AngularMomentum,
        per_moment_of_inertia,
        MomentOfInertia,
        AngularVelocity
    );
    per_method!(
        crate::AngularMomentumKilogramMetersSquaredPerSecond,
        AngularMomentum,
        per_time,
        Time,
        Torque
    );
    per_method!(
        crate::AngularMomentumKilogramMetersSquaredPerSecond,
        AngularMomentum,
        per_length,
        Length,
        Momentum
    );
}

#[test]
fn density_kilograms_per_cubic_meter_dimensional_algebra() {
    use euclid_units::Density;
    times_method!(
        crate::DensityKilogramsPerCubicMeter,
        Density,
        times_volume,
        Volume,
        Mass
    );
}

#[test]
fn dynamic_viscosity_pascal_seconds_dimensional_algebra() {
    use euclid_units::DynamicViscosity;
    per_method!(
        crate::DynamicViscosityPascalSeconds,
        DynamicViscosity,
        per_time,
        Time,
        Pressure
    );
}

#[test]
fn energy_joules_dimensional_algebra() {
    use euclid_units::Energy;
    per_method!(crate::EnergyJoules, Energy, per_length, Length, Force);
    per_method!(crate::EnergyJoules, Energy, per_force, Force, Length);
    per_method!(crate::EnergyJoules, Energy, per_time, Time, Power);
    per_method!(crate::EnergyJoules, Energy, per_power, Power, Time);
    per_method!(crate::EnergyJoules, Energy, per_volume, Volume, Pressure);
    per_method!(
        crate::EnergyJoules,
        Energy,
        per_charge,
        ElectricCharge,
        Voltage
    );
    per_method!(
        crate::EnergyJoules,
        Energy,
        per_temperature,
        Temperature,
        HeatCapacity
    );
    per_method!(crate::EnergyJoules, Energy, per_mass, Mass, SpecificEnergy);
    per_method!(
        crate::EnergyJoules,
        Energy,
        per_amount,
        AmountOfSubstance,
        MolarEnergy
    );
}

#[test]
fn force_newtons_dimensional_algebra() {
    use euclid_units::Force;
    per_method!(crate::ForceNewtons, Force, per_mass, Mass, Acceleration);
    per_method!(
        crate::ForceNewtons,
        Force,
        per_acceleration,
        Acceleration,
        Mass
    );
    times_method!(crate::ForceNewtons, Force, times_length, Length, Energy);
    per_method!(crate::ForceNewtons, Force, per_area, Area, Pressure);
    times_method!(crate::ForceNewtons, Force, times_time, Time, Momentum);
    times_method!(crate::ForceNewtons, Force, times_speed, Speed, Power);
    per_method!(
        crate::ForceNewtons,
        Force,
        per_length,
        Length,
        SurfaceTension
    );
    per_method!(
        crate::ForceNewtons,
        Force,
        per_charge,
        ElectricCharge,
        ElectricField
    );
}

#[test]
fn mass_kilograms_dimensional_algebra() {
    use euclid_units::Mass;
    times_method!(
        crate::MassKilograms,
        Mass,
        times_acceleration,
        Acceleration,
        Force
    );
    times_method!(crate::MassKilograms, Mass, times_speed, Speed, Momentum);
    per_method!(crate::MassKilograms, Mass, per_volume, Volume, Density);
    times_method!(
        crate::MassKilograms,
        Mass,
        times_area,
        Area,
        MomentOfInertia
    );
    per_method!(
        crate::MassKilograms,
        Mass,
        per_amount,
        AmountOfSubstance,
        MolarMass
    );
    per_method!(crate::MassKilograms, Mass, per_time, Time, MassFlowRate);
}

#[test]
fn mass_flow_rate_kilograms_per_second_dimensional_algebra() {
    use euclid_units::MassFlowRate;
    times_method!(
        crate::MassFlowRateKilogramsPerSecond,
        MassFlowRate,
        times_time,
        Time,
        Mass
    );
    times_method!(
        crate::MassFlowRateKilogramsPerSecond,
        MassFlowRate,
        times_speed,
        Speed,
        Force
    );
    per_method!(
        crate::MassFlowRateKilogramsPerSecond,
        MassFlowRate,
        per_density,
        Density,
        VolumetricFlowRate
    );
}

#[test]
fn moment_of_inertia_kilogram_meters_squared_dimensional_algebra() {
    use euclid_units::MomentOfInertia;
    per_method!(
        crate::MomentOfInertiaKilogramMetersSquared,
        MomentOfInertia,
        per_mass,
        Mass,
        Area
    );
    times_method!(
        crate::MomentOfInertiaKilogramMetersSquared,
        MomentOfInertia,
        times_angular_velocity,
        AngularVelocity,
        AngularMomentum
    );
    times_method!(
        crate::MomentOfInertiaKilogramMetersSquared,
        MomentOfInertia,
        times_angular_acceleration,
        AngularAcceleration,
        Torque
    );
}

#[test]
fn momentum_kilogram_meters_per_second_dimensional_algebra() {
    use euclid_units::Momentum;
    per_method!(
        crate::MomentumKilogramMetersPerSecond,
        Momentum,
        per_mass,
        Mass,
        Speed
    );
    per_method!(
        crate::MomentumKilogramMetersPerSecond,
        Momentum,
        per_speed,
        Speed,
        Mass
    );
    per_method!(
        crate::MomentumKilogramMetersPerSecond,
        Momentum,
        per_time,
        Time,
        Force
    );
    times_method!(
        crate::MomentumKilogramMetersPerSecond,
        Momentum,
        times_length,
        Length,
        AngularMomentum
    );
}

#[test]
fn power_watts_dimensional_algebra() {
    use euclid_units::Power;
    times_method!(crate::PowerWatts, Power, times_time, Time, Energy);
    per_method!(crate::PowerWatts, Power, per_area, Area, HeatFlux);
    per_method!(crate::PowerWatts, Power, per_force, Force, Speed);
    per_method!(crate::PowerWatts, Power, per_speed, Speed, Force);
    per_method!(
        crate::PowerWatts,
        Power,
        per_current,
        ElectricCurrent,
        Voltage
    );
    per_method!(
        crate::PowerWatts,
        Power,
        per_voltage,
        Voltage,
        ElectricCurrent
    );
}

#[test]
fn pressure_pascals_dimensional_algebra() {
    use euclid_units::Pressure;
    times_method!(crate::PressurePascals, Pressure, times_area, Area, Force);
    times_method!(
        crate::PressurePascals,
        Pressure,
        times_volume,
        Volume,
        Energy
    );
    times_method!(
        crate::PressurePascals,
        Pressure,
        times_time,
        Time,
        DynamicViscosity
    );
}

#[test]
fn specific_energy_joules_per_kilogram_dimensional_algebra() {
    use euclid_units::SpecificEnergy;
    times_method!(
        crate::SpecificEnergyJoulesPerKilogram,
        SpecificEnergy,
        times_mass,
        Mass,
        Energy
    );
}

#[test]
fn surface_tension_newtons_per_meter_dimensional_algebra() {
    use euclid_units::SurfaceTension;
    times_method!(
        crate::SurfaceTensionNewtonsPerMeter,
        SurfaceTension,
        times_length,
        Length,
        Force
    );
    times_method!(
        crate::SurfaceTensionNewtonsPerMeter,
        SurfaceTension,
        times_area,
        Area,
        Energy
    );
}

#[test]
fn torque_newton_meters_dimensional_algebra() {
    use euclid_units::Torque;
    per_method!(crate::TorqueNewtonMeters, Torque, per_length, Length, Force);
    times_method!(
        crate::TorqueNewtonMeters,
        Torque,
        times_angular_velocity,
        AngularVelocity,
        Power
    );
    times_method!(
        crate::TorqueNewtonMeters,
        Torque,
        times_time,
        Time,
        AngularMomentum
    );
    per_method!(
        crate::TorqueNewtonMeters,
        Torque,
        per_moment_of_inertia,
        MomentOfInertia,
        AngularAcceleration
    );
}
