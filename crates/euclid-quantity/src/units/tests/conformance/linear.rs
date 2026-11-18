//! Generic quantity, operator and scaling laws for each linear carrier.

use euclid_units::conformance::{self as laws, ConformanceSample};

use crate::units::*;

use super::TOL;

/// Declare a unit's sample and tolerance once, then run every applicable suite from it.
///
/// This is the external-consumer side of `ConformanceSample`: the trait is euclid-units', the
/// impls are ours, and the suites are called with no sample argument at all. Adding a suite to
/// euclid-units now covers every unit declared here without touching this file.
macro_rules! linear_unit_laws {
    ($name:ident, $ty:ty, [$($v:expr),+ $(,)?]) => {
        impl ConformanceSample for $ty {
            fn sample() -> Vec<Self> {
                vec![$(<$ty>::new($v)),+]
            }
            fn tolerance() -> Self {
                <$ty>::new(TOL)
            }
        }

        #[test]
        fn $name() {
            laws::quantity_laws_for::<$ty>(stringify!($ty));
            // The operator bridge: this unit's hand-written `+`, `-`, `+=`, `-=`, unary `-` and
            // `sum` against the shared trait defaults they abbreviate. See
            // `euclid_units::conformance::operator_bridge_laws` for why the two halves are
            // checked separately -- the default bodies are one implementation, the operators are
            // sixty-one.
            laws::operator_bridge_laws_for::<$ty>(stringify!($ty));
            // And scalar scaling: `a * k`, `k * a` and `a / k` against the trait operations
            // they abbreviate. Note `a / k` is deliberately not `scale(1/k)` -- see
            // `euclid_units::conformance::scaling_laws`.
            laws::scaling_laws_for::<$ty>(stringify!($ty));
        }
    };
}

// A primary unit of a base dimension.
linear_unit_laws!(length_meters, LengthMeters, [0.0, 1.0, -2.5, 4.0, 0.5]);
// A non-primary unit of the same dimension: the conversion factor must not disturb the laws.
linear_unit_laws!(length_feet, LengthFeet, [0.0, 1.0, -3.0, 10.0, 0.25]);
// A primary unit of a derived dimension, with a compound symbol.
linear_unit_laws!(speed_ms, SpeedMs, [0.0, 1.0, -9.8, 100.0, 0.125]);
// A squared dimension, where a sign error in `abs` or `neg` would be easy to miss.
linear_unit_laws!(area_square_meters, AreaSquareMeters, [0.0, 1.0, 2.5, 16.0]);
// Time, the dimension most used in cross-dimensional algebra.
linear_unit_laws!(time_seconds, TimeSeconds, [0.0, 1.0, -0.5, 60.0, 0.25]);
// Dimensionless, where `UNIT_NAME` is empty and the ratio laws are most exercised.
linear_unit_laws!(factor, Factor, [0.0, 1.0, -1.0, 2.5, 0.5]);
// A dimensionless unit carrying a scale, so `in_primary` is not the identity.
linear_unit_laws!(percentage, Percentage, [0.0, 50.0, -25.0, 100.0]);

// --- Chemistry ---
linear_unit_laws!(amount_moles_laws, AmountMoles, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(
    catalytic_activity_katals_laws,
    CatalyticActivityKatals,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    concentration_moles_per_cubic_meter_laws,
    ConcentrationMolesPerCubicMeter,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    molality_moles_per_kilogram_laws,
    MolalityMolesPerKilogram,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    molar_energy_joules_per_mole_laws,
    MolarEnergyJoulesPerMole,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    molar_mass_kilograms_per_mole_laws,
    MolarMassKilogramsPerMole,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    molar_volume_cubic_meters_per_mole_laws,
    MolarVolumeCubicMetersPerMole,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);

// --- Electromagnetism ---
linear_unit_laws!(
    capacitance_farads_laws,
    CapacitanceFarads,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    charge_coulombs_laws,
    ChargeCoulombs,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    conductance_siemens_laws,
    ConductanceSiemens,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    current_amperes_laws,
    CurrentAmperes,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    current_density_amperes_per_square_meter_laws,
    CurrentDensityAmperesPerSquareMeter,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    electric_field_volts_per_meter_laws,
    ElectricFieldVoltsPerMeter,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    inductance_henries_laws,
    InductanceHenries,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    magnetic_flux_density_teslas_laws,
    MagneticFluxDensityTeslas,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    magnetic_flux_webers_laws,
    MagneticFluxWebers,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    resistance_ohms_laws,
    ResistanceOhms,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(voltage_volts_laws, VoltageVolts, [0.0, 1.0, -2.5, 4.0, 0.5]);

// --- Geometry ---
linear_unit_laws!(angle_degrees_laws, AngleDegrees, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(angle_radians_laws, AngleRadians, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(
    area_square_kilometers_laws,
    AreaSquareKilometers,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(length_km_laws, LengthKm, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(
    volume_cubic_meters_laws,
    VolumeCubicMeters,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);

// --- Kinematics ---
linear_unit_laws!(
    acceleration_centimeters_per_second_squared_laws,
    AccelerationCentimetersPerSecondSquared,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    acceleration_meters_per_second_squared_laws,
    AccelerationMetersPerSecondSquared,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    angular_acceleration_radians_per_second_squared_laws,
    AngularAccelerationRadiansPerSecondSquared,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    angular_velocity_radians_per_second_laws,
    AngularVelocityRadiansPerSecond,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    frequency_hertz_laws,
    FrequencyHertz,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    jerk_meters_per_second_cubed_laws,
    JerkMetersPerSecondCubed,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(speed_kph_laws, SpeedKph, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(speed_mph_laws, SpeedMph, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(time_hours_laws, TimeHours, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(time_millis_laws, TimeMillis, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(time_minutes_laws, TimeMinutes, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(
    volumetric_flow_rate_cubic_meters_per_second_laws,
    VolumetricFlowRateCubicMetersPerSecond,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);

// --- Mechanics ---
linear_unit_laws!(
    angular_momentum_kilogram_meters_squared_per_second_laws,
    AngularMomentumKilogramMetersSquaredPerSecond,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    density_kilograms_per_cubic_meter_laws,
    DensityKilogramsPerCubicMeter,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    dynamic_viscosity_pascal_seconds_laws,
    DynamicViscosityPascalSeconds,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(energy_joules_laws, EnergyJoules, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(force_newtons_laws, ForceNewtons, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(
    mass_flow_rate_kilograms_per_second_laws,
    MassFlowRateKilogramsPerSecond,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(mass_grams_laws, MassGrams, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(
    mass_kilograms_laws,
    MassKilograms,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    mass_milligrams_laws,
    MassMilligrams,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    moment_of_inertia_kilogram_meters_squared_laws,
    MomentOfInertiaKilogramMetersSquared,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    momentum_kilogram_meters_per_second_laws,
    MomentumKilogramMetersPerSecond,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(power_watts_laws, PowerWatts, [0.0, 1.0, -2.5, 4.0, 0.5]);
linear_unit_laws!(
    pressure_pascals_laws,
    PressurePascals,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    specific_energy_joules_per_kilogram_laws,
    SpecificEnergyJoulesPerKilogram,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    surface_tension_newtons_per_meter_laws,
    SurfaceTensionNewtonsPerMeter,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    torque_newton_meters_laws,
    TorqueNewtonMeters,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);

// --- Thermal ---
linear_unit_laws!(
    heat_capacity_joules_per_kelvin_laws,
    HeatCapacityJoulesPerKelvin,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    heat_flux_watts_per_square_meter_laws,
    HeatFluxWattsPerSquareMeter,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
linear_unit_laws!(
    specific_heat_capacity_joules_per_kilogram_kelvin_laws,
    SpecificHeatCapacityJoulesPerKilogramKelvin,
    [0.0, 1.0, -2.5, 4.0, 0.5]
);
