//! Every unit's **inherent** surface, checked against the trait surface it duplicates.
//!
//! # Why this is separate from `conformance_tests`
//!
//! [`conformance_tests`](super::conformance_tests) runs euclid's law suites over one unit per
//! structural shape, and says so: "a conversion factor cannot break a law that holds for
//! `LengthMeters`, so the coverage that matters is across *shapes*." That reasoning is sound,
//! and it is about the **trait** surface — `Measure`, `ScalarQuantity`, `LinearQuantity` — where
//! the implementation is generic and shared.
//!
//! It is silent about everything each unit declares *itself*. `as_<unit>()`, `is_finite`,
//! `is_positive`, `is_non_negative`, `is_zero`, `new` and `ZERO` are written out by hand in
//! every unit file, so there are sixty-four independent chances to transpose a comparison — and
//! a law proved for `LengthMeters` says nothing about
//! `AngularMomentumKilogramMetersSquaredPerSecond::is_positive`.
//!
//! Mutation testing measured the consequence: roughly 730 surviving mutants across those five
//! methods alone, including every unit's value accessor being replaceable with `0.0` and every
//! `is_finite` returning `true` for an infinity.
//!
//! # What is asserted
//!
//! Each unit's inherent methods are checked against the `f64` they wrap, which is the only
//! definition they have:
//!
//! - `new(v).as_x() == v` — the accessor inverts the constructor
//! - `as_x()` agrees with `Measure::raw`, the trait accessor it duplicates
//! - the predicates agree with the same predicate applied to the raw value
//! - `ZERO` really is zero
//!
//! Stated as agreement with `f64` rather than as a table of expected values per unit, so this is
//! one macro rather than sixty-four hand-written tables — which would carry the same
//! transcription risk it is meant to catch.
//!
//! `NaN` is absent from the probe set deliberately: `new` debug-asserts against it, so it is not
//! a value a caller may construct.
//!
//! # The surface is not uniform, and that is why there are three macros
//!
//! Writing this turned up something worth recording. Of the sixty-four f64 units, **43 carry all
//! four predicates, 19 carry none, and two sit in between** — `Percentage` and `TimeSeconds`
//! have the sign predicates but not `is_finite`, and `AngleRadians` has only `abs`.
//!
//! The split falls along age rather than meaning: the long-standing core units (`LengthMeters`,
//! `MassKilograms`, `TimeSeconds`, `SpeedMs`, `AreaSquareMeters`, the temperatures) are the ones
//! missing predicates that every later unit has. There is no reason a caller can see why
//! `ForceNewtons::is_positive` exists and `MassKilograms::is_positive` does not. Worth
//! levelling up, but that is an API change rather than a test, so this file records the shape
//! instead of assuming it.

use crate::Measure;

/// Values every unit must handle, including both zeros and both infinities.
const PROBES: [f64; 9] = [
    0.0,
    -0.0,
    1.0,
    -2.5,
    7.5,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::MAX,
    f64::MIN_POSITIVE,
];

/// The accessor half, which every f64 unit has: `as_x` inverts `new` and agrees with
/// `Measure::raw`.
macro_rules! accessor_agrees {
    ($ty:ty, $as:ident) => {{
        for v in PROBES {
            let q = <$ty>::new(v);
            let label = concat!(stringify!($ty), "::", stringify!($as));
            assert_eq!(q.$as(), v, "{label}({v:e}) must invert new");
            assert_eq!(
                q.$as(),
                Measure::raw(&q),
                "{label}({v:e}) must agree with Measure::raw"
            );
        }
        assert_eq!(<$ty>::ZERO.$as(), 0.0, "ZERO must be zero");
    }};
}

/// The sign predicates, which must mean what their names say.
macro_rules! sign_predicates_agree {
    ($ty:ty, $as:ident) => {{
        for v in PROBES {
            let q = <$ty>::new(v);
            let label = stringify!($ty);
            assert_eq!(
                q.is_positive(),
                v > 0.0,
                "{label}: is_positive({v:e}) must mean strictly greater than zero"
            );
            assert_eq!(
                q.is_non_negative(),
                v >= 0.0,
                "{label}: is_non_negative({v:e}) must include zero"
            );
            assert_eq!(
                q.is_zero(),
                v == 0.0,
                "{label}: is_zero({v:e}) must mean exactly zero"
            );
        }
        assert!(<$ty>::ZERO.is_zero());
        assert!(!<$ty>::ZERO.is_positive(), "zero is not positive");
        assert!(<$ty>::ZERO.is_non_negative(), "zero is non-negative");
    }};
}

/// `is_finite`, which must reject both infinities.
macro_rules! finiteness_agrees {
    ($ty:ty) => {{
        for v in PROBES {
            let q = <$ty>::new(v);
            assert_eq!(
                q.is_finite(),
                v.is_finite(),
                concat!(stringify!($ty), ": is_finite must agree with the raw value"),
            );
        }
        assert!(
            !<$ty>::new(f64::INFINITY).is_finite(),
            "an infinity is not finite"
        );
    }};
}

/// `abs`, where the unit offers it.
macro_rules! abs_agrees {
    ($ty:ty, $as:ident) => {{
        for v in PROBES {
            assert_eq!(
                <$ty>::new(v).abs().$as(),
                v.abs(),
                concat!(stringify!($ty), "::abs must agree with f64::abs"),
            );
        }
    }};
}

/// `Display` must distinguish values and name its unit.
///
/// Every unit's `Display` is a one-line delegation to the shared
/// `crate::conversion::fmt_quantity`, and mutation testing found all sixty-four replaceable with
/// a body that writes nothing and returns `Ok(())`. The shared helper is well covered; its
/// sixty-four delegating callers were not — the same split as the operator bridge.
///
/// Asserted format-agnostically, because the point is the delegation rather than the layout: two
/// different values must render differently, and a unit with a symbol must show it. A body that
/// writes nothing fails the first; one that writes only the symbol fails it too.
macro_rules! display_distinguishes {
    ($ty:ty) => {{
        let one = <$ty>::new(1.0).to_string();
        let two = <$ty>::new(2.0).to_string();
        assert!(
            !one.is_empty(),
            concat!(stringify!($ty), " rendered to nothing")
        );
        assert_ne!(
            one, two,
            concat!(stringify!($ty), ": Display must distinguish values")
        );
        if !<$ty as euclid_units::Measure>::UNIT_NAME.is_empty() {
            assert!(
                one.contains(<$ty as euclid_units::Measure>::UNIT_NAME),
                concat!(stringify!($ty), ": Display must name its unit"),
            );
        }
    }};
}

/// `is_greater_than_epsilon` is a strict comparison against its argument, not against zero.
macro_rules! epsilon_predicate_agrees {
    ($ty:ty) => {{
        for v in PROBES {
            for eps in [0.0_f64, 1e-9, 1.0, -1.0] {
                assert_eq!(
                    <$ty>::new(v).is_greater_than_epsilon(eps),
                    v > eps,
                    concat!(
                        stringify!($ty),
                        ": is_greater_than_epsilon must be a strict comparison against epsilon"
                    ),
                );
            }
        }
    }};
}

/// A unit carrying the whole inherent surface.
macro_rules! full_surface {
    ($name:ident, $ty:ty, $as:ident) => {
        #[test]
        fn $name() {
            accessor_agrees!($ty, $as);
            sign_predicates_agree!($ty, $as);
            finiteness_agrees!($ty);
            abs_agrees!($ty, $as);
            display_distinguishes!($ty);
        }
    };
}

// ===== Chemistry =============================================================

full_surface!(amount_moles, crate::AmountMoles, as_moles);
full_surface!(
    catalytic_activity_katals,
    crate::CatalyticActivityKatals,
    as_katals
);
full_surface!(
    concentration_moles_per_cubic_meter,
    crate::ConcentrationMolesPerCubicMeter,
    as_moles_per_cubic_meter
);
full_surface!(
    molality_moles_per_kilogram,
    crate::MolalityMolesPerKilogram,
    as_moles_per_kilogram
);
full_surface!(
    molar_energy_joules_per_mole,
    crate::MolarEnergyJoulesPerMole,
    as_joules_per_mole
);
full_surface!(
    molar_mass_kilograms_per_mole,
    crate::MolarMassKilogramsPerMole,
    as_kilograms_per_mole
);
full_surface!(
    molar_volume_cubic_meters_per_mole,
    crate::MolarVolumeCubicMetersPerMole,
    as_cubic_meters_per_mole
);

// ===== Dimensionless =========================================================

#[test]
fn factor() {
    accessor_agrees!(crate::Factor, as_f64);
    sign_predicates_agree!(crate::Factor, as_f64);
    finiteness_agrees!(crate::Factor);
    display_distinguishes!(crate::Factor);
    epsilon_predicate_agrees!(crate::Factor);
}

#[test]
fn percentage() {
    accessor_agrees!(crate::Percentage, as_percent);
    sign_predicates_agree!(crate::Percentage, as_percent);
    display_distinguishes!(crate::Percentage);
}

// ===== Electromagnetism ======================================================

full_surface!(capacitance_farads, crate::CapacitanceFarads, as_farads);
full_surface!(charge_coulombs, crate::ChargeCoulombs, as_coulombs);
full_surface!(conductance_siemens, crate::ConductanceSiemens, as_siemens);
full_surface!(current_amperes, crate::CurrentAmperes, as_amperes);
full_surface!(
    current_density_amperes_per_square_meter,
    crate::CurrentDensityAmperesPerSquareMeter,
    as_amperes_per_square_meter
);
full_surface!(
    electric_field_volts_per_meter,
    crate::ElectricFieldVoltsPerMeter,
    as_volts_per_meter
);
full_surface!(inductance_henries, crate::InductanceHenries, as_henries);
full_surface!(
    magnetic_flux_density_teslas,
    crate::MagneticFluxDensityTeslas,
    as_teslas
);
full_surface!(magnetic_flux_webers, crate::MagneticFluxWebers, as_webers);
full_surface!(resistance_ohms, crate::ResistanceOhms, as_ohms);
full_surface!(voltage_volts, crate::VoltageVolts, as_volts);

// ===== Geometry ==============================================================

#[test]
fn angle_degrees() {
    accessor_agrees!(crate::AngleDegrees, as_degrees);
    display_distinguishes!(crate::AngleDegrees);
}

#[test]
fn angle_radians() {
    accessor_agrees!(crate::AngleRadians, as_radians);
    abs_agrees!(crate::AngleRadians, as_radians);
    display_distinguishes!(crate::AngleRadians);
}

#[test]
fn area_square_kilometers() {
    accessor_agrees!(crate::AreaSquareKilometers, as_square_kilometers);
    display_distinguishes!(crate::AreaSquareKilometers);
}

#[test]
fn area_square_meters() {
    accessor_agrees!(crate::AreaSquareMeters, as_square_meters);
    display_distinguishes!(crate::AreaSquareMeters);
}

#[test]
fn length_feet() {
    accessor_agrees!(crate::LengthFeet, as_feet);
    display_distinguishes!(crate::LengthFeet);
}

#[test]
fn length_km() {
    accessor_agrees!(crate::LengthKm, as_km);
    display_distinguishes!(crate::LengthKm);
}

full_surface!(length_meters, crate::LengthMeters, as_meters);
full_surface!(
    volume_cubic_meters,
    crate::VolumeCubicMeters,
    as_cubic_meters
);

// ===== Kinematics ============================================================

#[test]
fn acceleration_centimeters_per_second_squared() {
    accessor_agrees!(
        crate::AccelerationCentimetersPerSecondSquared,
        as_centimeters_per_second_squared
    );
    display_distinguishes!(crate::AccelerationCentimetersPerSecondSquared);
}

#[test]
fn acceleration_meters_per_second_squared() {
    accessor_agrees!(
        crate::AccelerationMetersPerSecondSquared,
        as_meters_per_second_squared
    );
    display_distinguishes!(crate::AccelerationMetersPerSecondSquared);
}

full_surface!(
    angular_acceleration_radians_per_second_squared,
    crate::AngularAccelerationRadiansPerSecondSquared,
    as_radians_per_second_squared
);
full_surface!(
    angular_velocity_radians_per_second,
    crate::AngularVelocityRadiansPerSecond,
    as_radians_per_second
);
full_surface!(frequency_hertz, crate::FrequencyHertz, as_hertz);
full_surface!(
    jerk_meters_per_second_cubed,
    crate::JerkMetersPerSecondCubed,
    as_meters_per_second_cubed
);
full_surface!(speed_kph, crate::SpeedKph, as_kph);
#[test]
fn speed_mph() {
    accessor_agrees!(crate::SpeedMph, as_mph);
    display_distinguishes!(crate::SpeedMph);
}

#[test]
fn speed_ms() {
    accessor_agrees!(crate::SpeedMs, as_ms);
    display_distinguishes!(crate::SpeedMs);
}

#[test]
fn time_hours() {
    accessor_agrees!(crate::TimeHours, as_hours);
    display_distinguishes!(crate::TimeHours);
}

#[test]
fn time_millis() {
    accessor_agrees!(crate::TimeMillis, as_millis);
    display_distinguishes!(crate::TimeMillis);
}

#[test]
fn time_minutes() {
    accessor_agrees!(crate::TimeMinutes, as_minutes);
    display_distinguishes!(crate::TimeMinutes);
}

#[test]
fn time_seconds() {
    accessor_agrees!(crate::TimeSeconds, as_seconds);
    sign_predicates_agree!(crate::TimeSeconds, as_seconds);
    abs_agrees!(crate::TimeSeconds, as_seconds);
    display_distinguishes!(crate::TimeSeconds);
    epsilon_predicate_agrees!(crate::TimeSeconds);
}

full_surface!(
    volumetric_flow_rate_cubic_meters_per_second,
    crate::VolumetricFlowRateCubicMetersPerSecond,
    as_cubic_meters_per_second
);

// ===== Mechanics =============================================================

full_surface!(
    angular_momentum_kilogram_meters_squared_per_second,
    crate::AngularMomentumKilogramMetersSquaredPerSecond,
    as_kilogram_meters_squared_per_second
);
full_surface!(
    density_kilograms_per_cubic_meter,
    crate::DensityKilogramsPerCubicMeter,
    as_kilograms_per_cubic_meter
);
full_surface!(
    dynamic_viscosity_pascal_seconds,
    crate::DynamicViscosityPascalSeconds,
    as_pascal_seconds
);
full_surface!(energy_joules, crate::EnergyJoules, as_joules);
full_surface!(force_newtons, crate::ForceNewtons, as_newtons);
full_surface!(
    mass_flow_rate_kilograms_per_second,
    crate::MassFlowRateKilogramsPerSecond,
    as_kilograms_per_second
);
#[test]
fn mass_grams() {
    accessor_agrees!(crate::MassGrams, as_grams);
    display_distinguishes!(crate::MassGrams);
}

#[test]
fn mass_kilograms() {
    accessor_agrees!(crate::MassKilograms, as_kilograms);
    display_distinguishes!(crate::MassKilograms);
}

#[test]
fn mass_milligrams() {
    accessor_agrees!(crate::MassMilligrams, as_milligrams);
    display_distinguishes!(crate::MassMilligrams);
}

full_surface!(
    moment_of_inertia_kilogram_meters_squared,
    crate::MomentOfInertiaKilogramMetersSquared,
    as_kilogram_meters_squared
);
full_surface!(
    momentum_kilogram_meters_per_second,
    crate::MomentumKilogramMetersPerSecond,
    as_kilogram_meters_per_second
);
full_surface!(power_watts, crate::PowerWatts, as_watts);
full_surface!(pressure_pascals, crate::PressurePascals, as_pascals);
full_surface!(
    specific_energy_joules_per_kilogram,
    crate::SpecificEnergyJoulesPerKilogram,
    as_joules_per_kilogram
);
full_surface!(
    surface_tension_newtons_per_meter,
    crate::SurfaceTensionNewtonsPerMeter,
    as_newtons_per_meter
);
full_surface!(
    torque_newton_meters,
    crate::TorqueNewtonMeters,
    as_newton_meters
);

// ===== Thermal ===============================================================

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

// ===== The epsilon predicate, where a unit offers it ======================

/// Only three units carry `is_greater_than_epsilon`, so it gets one test rather than a
/// macro arm per unit.
#[test]
fn the_epsilon_predicate_is_a_strict_comparison() {
    epsilon_predicate_agrees!(crate::Factor);
    epsilon_predicate_agrees!(crate::SpeedKph);
    epsilon_predicate_agrees!(crate::TimeSeconds);
}
