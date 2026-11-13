#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Value tests over the dimension-contract web.
//!
//! The contract impls in the sibling modules compiling at all is the structural proof that
//! oresme's primary units form a coherent implementor set for euclid's mutually recursive
//! contracts. These tests pin the *arithmetic*: that each `times_*`/`per_*` lands in the right
//! unit with the right magnitude, that the bespoke relations (roots, reciprocals, trig) are
//! correct, and that the new carriers carry their serde/`UnitOfMeasure` contracts.

use euclid_units::{
    Acceleration, AmountOfSubstance, Angle, Area, Conductance, Density, ElectricCurrent, Energy,
    Force, Frequency, Length, Mass, Power, Resistance, Speed, Time, Voltage, Volume,
};

use crate::{
    AccelerationMetersPerSecondSquared, AmountMoles, AngleRadians, AreaSquareMeters,
    ChargeCoulombs, ConductanceSiemens, CurrentAmperes, DensityKilogramsPerCubicMeter,
    EnergyJoules, ForceNewtons, FrequencyHertz, LengthMeters, MassKilograms,
    MolarMassKilogramsPerMole, PowerWatts, Quantity, ResistanceOhms, SpeedMs, TimeSeconds,
    UnitOfMeasure, VoltageVolts, VolumeCubicMeters,
};

const EPS: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS * (1.0 + a.abs() + b.abs())
}

#[test]
fn kinematics_chain_distance_speed_acceleration() {
    let distance = LengthMeters::new(100.0);
    let elapsed = TimeSeconds::new(10.0);

    let speed: SpeedMs = distance.per_time(elapsed);
    assert_eq!(speed, SpeedMs::new(10.0));

    // Round-trips back through the inverse relations.
    assert_eq!(speed.times_time(elapsed), distance);
    assert_eq!(distance.per_speed(speed), elapsed);

    let accel: AccelerationMetersPerSecondSquared = speed.per_time(TimeSeconds::new(2.0));
    assert_eq!(accel, AccelerationMetersPerSecondSquared::new(5.0));
    assert_eq!(accel.times_time(TimeSeconds::new(2.0)), speed);
}

#[test]
fn geometry_powers_and_roots() {
    assert_eq!(LengthMeters::new(3.0).squared(), AreaSquareMeters::new(9.0));
    assert_eq!(LengthMeters::new(2.0).cubed(), VolumeCubicMeters::new(8.0));
    assert_eq!(AreaSquareMeters::new(9.0).side(), LengthMeters::new(3.0));
    assert_eq!(VolumeCubicMeters::new(27.0).edge(), LengthMeters::new(3.0));
    assert_eq!(
        LengthMeters::new(2.0).times_length(LengthMeters::new(3.0)),
        AreaSquareMeters::new(6.0)
    );
}

#[test]
fn newtons_second_law_and_work() {
    // F = m * a
    let force: ForceNewtons =
        MassKilograms::new(2.0).times_acceleration(AccelerationMetersPerSecondSquared::new(3.0));
    assert_eq!(force, ForceNewtons::new(6.0));
    assert_eq!(
        force.per_mass(MassKilograms::new(2.0)),
        AccelerationMetersPerSecondSquared::new(3.0)
    );

    // E = F * d, then P = E / t
    let energy: EnergyJoules = force.times_length(LengthMeters::new(4.0));
    assert_eq!(energy, EnergyJoules::new(24.0));
    let power: PowerWatts = energy.per_time(TimeSeconds::new(2.0));
    assert_eq!(power, PowerWatts::new(12.0));
    assert_eq!(power.times_time(TimeSeconds::new(2.0)), energy);
}

#[test]
fn density_is_mass_over_volume() {
    let density: DensityKilogramsPerCubicMeter =
        MassKilograms::new(10.0).per_volume(VolumeCubicMeters::new(2.0));
    assert_eq!(density, DensityKilogramsPerCubicMeter::new(5.0));
    assert_eq!(
        density.times_volume(VolumeCubicMeters::new(2.0)),
        MassKilograms::new(10.0)
    );
}

#[test]
fn ohms_law_and_reciprocal_conductance() {
    // V = I * R
    let voltage: VoltageVolts = CurrentAmperes::new(2.0).times_resistance(ResistanceOhms::new(3.0));
    assert_eq!(voltage, VoltageVolts::new(6.0));
    assert_eq!(
        voltage.per_current(CurrentAmperes::new(2.0)),
        ResistanceOhms::new(3.0)
    );

    // Power and charge cross-checks.
    let power: PowerWatts = CurrentAmperes::new(2.0).times_voltage(VoltageVolts::new(6.0));
    assert_eq!(power, PowerWatts::new(12.0));
    let charge: ChargeCoulombs = CurrentAmperes::new(2.0).times_time(TimeSeconds::new(5.0));
    assert_eq!(charge, ChargeCoulombs::new(10.0));

    // G = 1 / R, round-tripping.
    let conductance: ConductanceSiemens = ResistanceOhms::new(4.0).conductance();
    assert_eq!(conductance, ConductanceSiemens::new(0.25));
    assert_eq!(conductance.resistance(), ResistanceOhms::new(4.0));
}

#[test]
fn frequency_is_reciprocal_time() {
    let f: FrequencyHertz = TimeSeconds::new(4.0).frequency();
    assert_eq!(f, FrequencyHertz::new(0.25));
    assert_eq!(f.period(), TimeSeconds::new(4.0));
}

#[test]
fn chemistry_amount_times_molar_mass_is_mass() {
    let mass = AmountMoles::new(2.0).times_molar_mass(MolarMassKilogramsPerMole::new(3.0));
    assert_eq!(mass, MassKilograms::new(6.0));
}

#[test]
fn angle_conversions_and_trig() {
    let half = AngleRadians::from_degrees(180.0);
    assert!(close(half.radians(), core::f64::consts::PI));
    assert!(close(half.degrees(), 180.0));
    assert!(close(half.sin(), 0.0));
    assert!(close(AngleRadians::from_radians(0.0).cos(), 1.0));

    let wrapped = AngleRadians::new(3.0 * core::f64::consts::TAU + 1.0).normalized_unsigned();
    assert!(close(wrapped.radians(), 1.0));
}

#[test]
fn new_carriers_carry_unit_of_measure_and_serde() {
    // Each new carrier is a primary unit: `in_primary` is the identity and the symbol prints.
    assert_eq!(ForceNewtons::new(5.0).in_primary(), 5.0);
    assert_eq!(ForceNewtons::new(5.0).to_string(), "5 N");
    assert_eq!(EnergyJoules::new(2.5).to_string(), "2.5 J");

    let json = serde_json::to_string(&EnergyJoules::new(2.5)).expect("serialize");
    assert_eq!(json, "2.5");
    let back: EnergyJoules = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, EnergyJoules::new(2.5));

    // Runtime dimension tags line up with euclid's.
    assert_eq!(
        <ForceNewtons as Quantity>::DIMENSION,
        euclid_units::Dimension::Force
    );
}

#[test]
fn euclid_contracts_are_reachable_from_the_crate_root() {
    // The dimension traits are re-exported from oresme, so the algebra reads ergonomically
    // without importing euclid directly.
    use crate::{Length, Mass};

    let speed = crate::LengthMeters::new(100.0).per_time(crate::TimeSeconds::new(10.0));
    assert_eq!(speed, crate::SpeedMs::new(10.0));

    let force = crate::MassKilograms::new(2.0)
        .times_acceleration(crate::AccelerationMetersPerSecondSquared::new(3.0));
    assert_eq!(force, crate::ForceNewtons::new(6.0));
}

#[test]
fn carriers_have_convenience_predicates_and_abs() {
    assert!(ForceNewtons::new(6.0).is_positive());
    assert!(!ForceNewtons::new(-1.0).is_non_negative());
    assert!(PowerWatts::ZERO.is_zero());
    assert!(EnergyJoules::new(1.0).is_finite());
    assert_eq!(EnergyJoules::new(-4.0).abs(), EnergyJoules::new(4.0));
}
