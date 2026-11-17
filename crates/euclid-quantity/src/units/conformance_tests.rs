//! The real `f64` units, checked against euclid's own contracts.
//!
//! Every suite here is [`euclid_units::conformance`] — public API that a carrier author in any
//! repository calls. This crate is the first such author, so these tests are also the proof
//! that the suites are usable from outside the crate that defines them.
//!
//! The spread is deliberate: one unit per structural shape rather than all 72. A conversion
//! factor cannot break a law that holds for `LengthMeters`, so the coverage that matters is
//! across *shapes* — a primary unit, a non-primary one, a dimensionless factor, a derived unit
//! with a compound symbol, and an absolute scalar that withholds `LinearQuantity`.

use euclid_units::conformance::{self as laws, ConformanceSample};

use crate::Measure;
use crate::*;
use crate::{TemperatureCelsius, TemperatureKelvin};

/// A tolerance small enough that `approx_eq` reflexivity is not trivially satisfied by a
/// permissive bound.
const TOL: f64 = 1e-12;

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

// ===== The remaining linear units ==========================================
//
// The spread above is one unit per structural *shape*, and the note on it is right that a
// conversion factor cannot break a law which holds for `LengthMeters`. What that argument does
// not cover is that each unit's `Measure` impl — `raw`, `from_raw` — and its arithmetic are
// written out per unit rather than derived, so there are sixty-one independent chances to get
// one wrong.
//
// Mutation testing measured it: around 480 survivors across `raw`, `add`, `sub`, `mul`, `div`,
// `neg` and `abs` in units this file did not name. Those are all trait methods, so unlike the
// inherent surface in `inherent_tests` they need no new laws — only the existing suites, pointed
// at the rest of the crate.
//
// The sample is uniform because the laws are: `quantity_laws` is algebraic, so what it needs is
// zero, one, a negative, and a pair whose ratio is not round. A unit whose own range is narrower
// than this still has to satisfy the laws over it — none of these carriers clamp.

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

impl ConformanceSample for TemperatureCelsius {
    fn sample() -> Vec<Self> {
        vec![
            Self::new(0.0),
            Self::new(20.0),
            Self::new(-40.0),
            Self::new(100.0),
        ]
    }
    fn tolerance() -> Self {
        Self::new(TOL)
    }
}

impl ConformanceSample for TemperatureKelvin {
    fn sample() -> Vec<Self> {
        vec![Self::new(0.0), Self::new(273.15), Self::new(1.0)]
    }
    fn tolerance() -> Self {
        Self::new(TOL)
    }
}

#[test]
fn absolute_scalars_get_the_measure_and_scalar_laws_only() {
    // *Every* temperature withholds `LinearQuantity`, Kelvin included — adding two absolute
    // temperatures is meaningless whatever the scale's zero happens to be. So they take the two
    // suites that apply and not the third, and `quantity_laws` would not compile for them.
    // That asymmetry is the contract working as intended.
    let celsius = [
        TemperatureCelsius::new(0.0),
        TemperatureCelsius::new(20.0),
        TemperatureCelsius::new(-40.0),
        TemperatureCelsius::new(100.0),
    ];
    laws::measure_laws(&celsius, "TemperatureCelsius");
    laws::scalar_laws(
        &celsius,
        &TemperatureCelsius::new(TOL),
        "TemperatureCelsius",
    );

    let kelvin = [
        TemperatureKelvin::new(0.0),
        TemperatureKelvin::new(273.15),
        TemperatureKelvin::new(1.0),
    ];
    laws::measure_laws(&kelvin, "TemperatureKelvin");
    laws::scalar_laws(&kelvin, &TemperatureKelvin::new(TOL), "TemperatureKelvin");
}

#[test]
fn integer_backed_storage_satisfies_the_discrete_laws() {
    use crate::{Bytes, Count, ThreadCount};

    laws::discrete_laws(&[Bytes::new(0), Bytes::new(1), Bytes::new(4096)], "Bytes");
    laws::discrete_laws(&[Count::new(0), Count::new(1), Count::new(999)], "Count");
    laws::discrete_laws(
        &[
            ThreadCount::new(0),
            ThreadCount::new(1),
            ThreadCount::new(8),
        ],
        "ThreadCount",
    );
}

#[test]
fn raw_round_trips_across_every_structural_shape() {
    // `measure_laws` on its own, over the widest spread — the one law every unit must satisfy
    // regardless of which tiers it reaches.
    laws::measure_laws(&[LengthMeters::new(1.5)], "LengthMeters");
    laws::measure_laws(&[LengthFeet::new(1.5)], "LengthFeet");
    laws::measure_laws(&[SpeedMs::new(1.5)], "SpeedMs");
    laws::measure_laws(&[Factor::new(1.5)], "Factor");
    laws::measure_laws(&[TemperatureCelsius::new(1.5)], "TemperatureCelsius");
    assert_eq!(LengthMeters::from_raw(2.0).raw(), 2.0);
}
