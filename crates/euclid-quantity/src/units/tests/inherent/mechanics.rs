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
