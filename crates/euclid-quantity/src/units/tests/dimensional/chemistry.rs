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

/// `MolarVolume::concentration` is a reciprocal, which is the one dimensional method whose name
/// does not say `per_` or `times_` — so the generated dimensional suite does not cover it.
#[test]
fn molar_volume_inverts_to_concentration() {
    use euclid_units::MolarVolume;

    use crate::MolarVolumeCubicMetersPerMole;

    const TOL: f64 = 1e-12;
    for v in [0.5_f64, 1.0, 2.0, 4.0] {
        let got = MolarVolumeCubicMetersPerMole::new(v)
            .concentration()
            .as_moles_per_cubic_meter();
        assert!(
            (got - 1.0 / v).abs() <= TOL,
            "concentration({v}) = {got}, want {}",
            1.0 / v
        );
    }
}
