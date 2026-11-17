//! The **dimensional algebra**: that `per_x` divides and `times_x` multiplies.
//!
//! # What this covers that nothing else does
//!
//! The suites in [`conformance_tests`](super::conformance_tests) are generic over one carrier at a
//! time -- `quantity_laws`, `operator_bridge_laws`, `scaling_laws` all take a `Q` and stay inside
//! it. The cross-dimensional methods leave it: `Length::per_time` consumes a `Time` and returns a
//! `Speed`, so no single-type law can reach them. They were the last large cluster of surviving
//! mutants in this crate, and they are also the crate's entire reason for existing -- a units
//! library whose dimensional algebra is unchecked is a units library with the interesting part
//! unchecked.
//!
//! # Why this is not a tautology
//!
//! Each of these methods has a one-line body of the form `Ret::from_raw(self.raw() OP arg.raw())`,
//! so a test generated *from the body* would assert that the body is the body. The expectation
//! has to come from somewhere else, and it does: **the method name**.
//!
//! The convention is `per_x` for division and `times_x` for multiplication, applied uniformly
//! across sixty-two traits. That is an independent statement of intent, and it is falsifiable --
//! a `per_time` that multiplies is a bug whatever its body says. So each assertion below reads
//! the operation off the name and requires the value to match.
//!
//! The *dimensions* are checked separately and independently, by
//! `every_derived_dimension_matches_its_physical_definition` in `euclid-units`, which asserts the
//! exponent table against physics definitions. Between the two, a method is pinned on both axes:
//! this file says `per_time` divides the magnitudes, that one says `Speed` is `Length / Time`.
//!
//! # Exact arithmetic, deliberately
//!
//! The operands are `3.0` and `2.0`, so every product and quotient here -- `6.0`, `1.5` -- is
//! exactly representable and the assertions need no tolerance. A dimensional method is a single
//! multiply or divide, so there is no accumulation to accommodate.

use euclid_units::Measure;

/// `times_x` must multiply the magnitudes.
macro_rules! times_method {
    ($unit:ty, $tr:path, $method:ident, $arg:ident, $ret:ident) => {{
        let a = <$unit as Measure>::from_raw(3.0);
        let b = <<$unit as $tr>::$arg as Measure>::from_raw(2.0);
        assert_eq!(
            <$unit as $tr>::$method(a, b),
            <<$unit as $tr>::$ret as Measure>::from_raw(6.0),
            concat!(
                stringify!($unit),
                "::",
                stringify!($method),
                " must multiply the magnitudes (3 x 2 = 6)"
            )
        );
    }};
}

/// `per_x` must divide the magnitudes.
macro_rules! per_method {
    ($unit:ty, $tr:path, $method:ident, $arg:ident, $ret:ident) => {{
        let a = <$unit as Measure>::from_raw(3.0);
        let b = <<$unit as $tr>::$arg as Measure>::from_raw(2.0);
        assert_eq!(
            <$unit as $tr>::$method(a, b),
            <<$unit as $tr>::$ret as Measure>::from_raw(1.5),
            concat!(
                stringify!($unit),
                "::",
                stringify!($method),
                " must divide the magnitudes (3 / 2 = 1.5)"
            )
        );
    }};
}

/// `squared` and `cubed` must be the repeated product.
macro_rules! power_method {
    ($unit:ty, $tr:path, $method:ident, $ret:ident, $want:expr) => {{
        let a = <$unit as Measure>::from_raw(3.0);
        assert_eq!(
            <$unit as $tr>::$method(a),
            <<$unit as $tr>::$ret as Measure>::from_raw($want),
            concat!(
                stringify!($unit),
                "::",
                stringify!($method),
                " must be the repeated product"
            )
        );
    }};
}

// ===== Chemistry ===========================================================

#[test]
fn amount_moles_dimensional_algebra() {
    use euclid_units::AmountOfSubstance;
    per_method!(
        crate::AmountMoles,
        AmountOfSubstance,
        per_time,
        Time,
        CatalyticActivity
    );
    per_method!(
        crate::AmountMoles,
        AmountOfSubstance,
        per_volume,
        Volume,
        Concentration
    );
    times_method!(
        crate::AmountMoles,
        AmountOfSubstance,
        times_molar_mass,
        MolarMass,
        Mass
    );
    per_method!(
        crate::AmountMoles,
        AmountOfSubstance,
        per_mass,
        Mass,
        Molality
    );
}

#[test]
fn catalytic_activity_katals_dimensional_algebra() {
    use euclid_units::CatalyticActivity;
    times_method!(
        crate::CatalyticActivityKatals,
        CatalyticActivity,
        times_time,
        Time,
        AmountOfSubstance
    );
}

#[test]
fn concentration_moles_per_cubic_meter_dimensional_algebra() {
    use euclid_units::Concentration;
    times_method!(
        crate::ConcentrationMolesPerCubicMeter,
        Concentration,
        times_volume,
        Volume,
        AmountOfSubstance
    );
}

#[test]
fn molality_moles_per_kilogram_dimensional_algebra() {
    use euclid_units::Molality;
    times_method!(
        crate::MolalityMolesPerKilogram,
        Molality,
        times_mass,
        Mass,
        AmountOfSubstance
    );
}

#[test]
fn molar_energy_joules_per_mole_dimensional_algebra() {
    use euclid_units::MolarEnergy;
    times_method!(
        crate::MolarEnergyJoulesPerMole,
        MolarEnergy,
        times_amount,
        AmountOfSubstance,
        Energy
    );
}

#[test]
fn molar_mass_kilograms_per_mole_dimensional_algebra() {
    use euclid_units::MolarMass;
    times_method!(
        crate::MolarMassKilogramsPerMole,
        MolarMass,
        times_amount,
        AmountOfSubstance,
        Mass
    );
}

#[test]
fn molar_volume_cubic_meters_per_mole_dimensional_algebra() {
    use euclid_units::MolarVolume;
    times_method!(
        crate::MolarVolumeCubicMetersPerMole,
        MolarVolume,
        times_amount,
        AmountOfSubstance,
        Volume
    );
}

// ===== Electromagnetism ====================================================

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

// ===== Geometry ============================================================

#[test]
fn angle_radians_dimensional_algebra() {
    use euclid_units::Angle;
    per_method!(crate::AngleRadians, Angle, per_time, Time, AngularVelocity);
}

#[test]
fn area_square_meters_dimensional_algebra() {
    use euclid_units::Area;
    per_method!(crate::AreaSquareMeters, Area, per_length, Length, Length);
    times_method!(crate::AreaSquareMeters, Area, times_length, Length, Volume);
    times_method!(
        crate::AreaSquareMeters,
        Area,
        times_speed,
        Speed,
        VolumetricFlowRate
    );
}

#[test]
fn length_meters_dimensional_algebra() {
    use euclid_units::Length;
    times_method!(crate::LengthMeters, Length, times_area, Area, Volume);
    per_method!(crate::LengthMeters, Length, per_time, Time, Speed);
    per_method!(crate::LengthMeters, Length, per_speed, Speed, Time);
    power_method!(crate::LengthMeters, Length, squared, Area, 9.0);
    power_method!(crate::LengthMeters, Length, cubed, Volume, 27.0);
}

#[test]
fn volume_cubic_meters_dimensional_algebra() {
    use euclid_units::Volume;
    per_method!(crate::VolumeCubicMeters, Volume, per_length, Length, Area);
    per_method!(crate::VolumeCubicMeters, Volume, per_area, Area, Length);
    per_method!(
        crate::VolumeCubicMeters,
        Volume,
        per_amount,
        AmountOfSubstance,
        MolarVolume
    );
}

// ===== Kinematics ==========================================================

#[test]
fn acceleration_meters_per_second_squared_dimensional_algebra() {
    use euclid_units::Acceleration;
    times_method!(
        crate::AccelerationMetersPerSecondSquared,
        Acceleration,
        times_time,
        Time,
        Speed
    );
    times_method!(
        crate::AccelerationMetersPerSecondSquared,
        Acceleration,
        times_mass,
        Mass,
        Force
    );
    per_method!(
        crate::AccelerationMetersPerSecondSquared,
        Acceleration,
        per_time,
        Time,
        Jerk
    );
}

#[test]
fn angular_acceleration_radians_per_second_squared_dimensional_algebra() {
    use euclid_units::AngularAcceleration;
    times_method!(
        crate::AngularAccelerationRadiansPerSecondSquared,
        AngularAcceleration,
        times_time,
        Time,
        AngularVelocity
    );
}

#[test]
fn angular_velocity_radians_per_second_dimensional_algebra() {
    use euclid_units::AngularVelocity;
    times_method!(
        crate::AngularVelocityRadiansPerSecond,
        AngularVelocity,
        times_time,
        Time,
        Angle
    );
    per_method!(
        crate::AngularVelocityRadiansPerSecond,
        AngularVelocity,
        per_time,
        Time,
        AngularAcceleration
    );
}

#[test]
fn jerk_meters_per_second_cubed_dimensional_algebra() {
    use euclid_units::Jerk;
    times_method!(
        crate::JerkMetersPerSecondCubed,
        Jerk,
        times_time,
        Time,
        Acceleration
    );
}

#[test]
fn speed_ms_dimensional_algebra() {
    use euclid_units::Speed;
    times_method!(crate::SpeedMs, Speed, times_time, Time, Length);
    per_method!(crate::SpeedMs, Speed, per_time, Time, Acceleration);
    per_method!(crate::SpeedMs, Speed, per_acceleration, Acceleration, Time);
}

#[test]
fn time_seconds_dimensional_algebra() {
    use euclid_units::Time;
    times_method!(crate::TimeSeconds, Time, times_speed, Speed, Length);
    times_method!(
        crate::TimeSeconds,
        Time,
        times_acceleration,
        Acceleration,
        Speed
    );
}

#[test]
fn volumetric_flow_rate_cubic_meters_per_second_dimensional_algebra() {
    use euclid_units::VolumetricFlowRate;
    times_method!(
        crate::VolumetricFlowRateCubicMetersPerSecond,
        VolumetricFlowRate,
        times_time,
        Time,
        Volume
    );
    per_method!(
        crate::VolumetricFlowRateCubicMetersPerSecond,
        VolumetricFlowRate,
        per_area,
        Area,
        Speed
    );
    per_method!(
        crate::VolumetricFlowRateCubicMetersPerSecond,
        VolumetricFlowRate,
        per_speed,
        Speed,
        Area
    );
}

// ===== Mechanics ===========================================================

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

// ===== Thermal =============================================================

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
