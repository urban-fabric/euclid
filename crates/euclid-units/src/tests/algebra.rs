//! Value-level tests of the cross-dimensional algebra, over the toy carriers in
//! [`toy_units`](super::toy_units).
//!
//! `f64: Real`, so these exercise the same generic code paths any other real carrier would
//! take, and `<f64 as Compare>::Mask = bool`, so the mask-returning predicates collapse to
//! plain `bool` here.

#![allow(clippy::float_cmp)]

// Glob imports: the carrier set is large and each test touches a different slice of it, so
// naming them individually would mean a long, churn-prone list per file.
use super::toy_units::*;
use crate::*;

// --- tests ---------------------------------------------------------------------------------

#[test]
fn linear_ops_over_f64() {
    let a = Len(3.0);
    let b = Len(4.0);
    assert!(approx(a.clone().add(b.clone()).raw(), 7.0));
    assert!(approx(b.clone().sub(a.clone()).raw(), 1.0));
    assert!(approx(a.clone().neg().raw(), -3.0));
    assert!(approx(a.clone().scaled(2.5).raw(), 7.5));
    assert!(approx(a.clone().scale(2.0).raw(), 6.0));
    assert!(approx(a.clone().lerp(b.clone(), 0.5).raw(), 3.5));
    assert!(approx(b.ratio(a), 4.0 / 3.0));
}

#[test]
fn selectors_and_masks_collapse_to_bool() {
    let a = Len(2.0);
    let b = Len(5.0);
    assert!(approx(a.clone().min(b.clone()).raw(), 2.0));
    assert!(approx(a.clone().max(b.clone()).raw(), 5.0));
    assert!(approx(Len(9.0).clamp(Len(0.0), Len(5.0)).raw(), 5.0));
    let finite: bool = a.is_finite();
    assert!(finite);
    assert!(!a.is_nan());
    assert!(a.approx_eq(&Len(2.0 + 1e-12), &Len(EPS)));
}

#[test]
fn geometry_algebra() {
    assert!(approx(Len(3.0).times_length(Len(4.0)).raw(), 12.0));
    assert!(approx(Len(3.0).squared().raw(), 9.0));
    assert!(approx(Len(2.0).cubed().raw(), 8.0));
    assert!(approx(Ar(12.0).per_length(Len(4.0)).raw(), 3.0));
    assert!(approx(Ar(16.0).side().raw(), 4.0));
    assert!(approx(Vol(27.0).edge().raw(), 3.0));
    assert!(approx(Len(2.0).times_area(Ar(6.0)).raw(), 12.0));
    assert!(approx(Vol(12.0).per_area(Ar(6.0)).raw(), 2.0));
    assert!(approx(Vol(12.0).per_length(Len(3.0)).raw(), 4.0));
}

#[test]
fn kinematic_ladder() {
    let v: Spd = Len(100.0).per_time(Tim(4.0));
    assert!(approx(v.clone().raw(), 25.0));
    assert!(approx(v.clone().times_time(Tim(4.0)).raw(), 100.0));
    assert!(approx(Len(100.0).per_speed(v.clone()).raw(), 4.0));

    let a: Acc = v.per_time(Tim(5.0));
    assert!(approx(a.clone().raw(), 5.0));
    assert!(approx(a.clone().times_time(Tim(2.0)).raw(), 10.0));
    assert!(approx(a.per_time(Tim(2.0)).raw(), 2.5)); // jerk

    assert!(approx(Tim(2.0).frequency().raw(), 0.5));
    assert!(approx(Frq(0.5).period().raw(), 2.0));
    assert!(approx(Ar(2.0).times_speed(Spd(3.0)).raw(), 6.0)); // flow rate
    assert!(approx(Flow(6.0).per_area(Ar(2.0)).raw(), 3.0));
}

#[test]
fn newtonian_dynamics() {
    let f: Frc = Mas(5.0).times_acceleration(Acc(2.0)); // F = ma
    assert!(approx(f.clone().raw(), 10.0));
    assert!(approx(f.clone().per_mass(Mas(5.0)).raw(), 2.0));
    assert!(approx(f.clone().per_acceleration(Acc(2.0)).raw(), 5.0));

    let work: Eng = f.clone().times_length(Len(3.0)); // W = F·d
    assert!(approx(work.clone().raw(), 30.0));
    assert!(approx(work.clone().per_time(Tim(5.0)).raw(), 6.0)); // P = W/t
    assert!(approx(work.per_force(Frc(10.0)).raw(), 3.0));

    assert!(approx(f.clone().per_area(Ar(2.0)).raw(), 5.0)); // pressure
    assert!(approx(f.clone().times_time(Tim(3.0)).raw(), 30.0)); // impulse → momentum
    assert!(approx(f.times_speed(Spd(4.0)).raw(), 40.0)); // power

    let p: Mom = Mas(5.0).times_speed(Spd(4.0));
    assert!(approx(p.clone().raw(), 20.0));
    assert!(approx(p.clone().per_mass(Mas(5.0)).raw(), 4.0));
    assert!(approx(p.per_time(Tim(4.0)).raw(), 5.0)); // F = dp/dt

    assert!(approx(Mas(12.0).per_volume(Vol(4.0)).raw(), 3.0)); // density
    assert!(approx(Den(3.0).times_volume(Vol(4.0)).raw(), 12.0));
    assert!(approx(Prs(5.0).times_volume(Vol(2.0)).raw(), 10.0)); // P·V = energy
}

#[test]
fn rotational_dynamics() {
    assert!(approx(AngVel(2.0).times_time(Tim(3.0)).raw(), 6.0)); // angle
    assert!(approx(Ang(6.0).per_time(Tim(3.0)).raw(), 2.0));

    let torque: Trq = Inr(4.0).times_angular_acceleration(AngAcc(2.0)); // τ = Iα
    assert!(approx(torque.clone().raw(), 8.0));
    assert!(approx(torque.clone().per_length(Len(2.0)).raw(), 4.0)); // → force
    assert!(approx(
        torque.times_angular_velocity(AngVel(3.0)).raw(),
        24.0
    )); // → power

    let l: AngMom = Inr(4.0).times_angular_velocity(AngVel(3.0)); // L = Iω
    assert!(approx(l.clone().raw(), 12.0));
    assert!(approx(l.clone().per_moment_of_inertia(Inr(4.0)).raw(), 3.0));
    assert!(approx(l.per_time(Tim(4.0)).raw(), 3.0)); // → torque
    assert!(approx(Mom(20.0).times_length(Len(2.0)).raw(), 40.0)); // r × p
}

#[test]
fn electromagnetic_circuits() {
    let v: Vlt = Cur(2.0).times_resistance(Res(3.0)); // Ohm: V = IR
    assert!(approx(v.clone().raw(), 6.0));
    assert!(approx(v.clone().per_current(Cur(2.0)).raw(), 3.0));
    assert!(approx(v.clone().per_resistance(Res(3.0)).raw(), 2.0));
    assert!(approx(v.clone().times_current(Cur(2.0)).raw(), 12.0)); // P = VI

    let q: Chg = Cur(3.0).times_time(Tim(4.0)); // Q = It
    assert!(approx(q.clone().raw(), 12.0));
    assert!(approx(q.clone().per_time(Tim(4.0)).raw(), 3.0));
    assert!(approx(q.times_voltage(Vlt(2.0)).raw(), 24.0)); // E = QV

    assert!(approx(Cap(5.0).times_voltage(Vlt(2.0)).raw(), 10.0)); // Q = CV
    assert!(approx(Res(4.0).conductance().raw(), 0.25));
    assert!(approx(Cnd(0.25).resistance().raw(), 4.0));
    assert!(approx(Cap(2.0).times_resistance(Res(3.0)).raw(), 6.0)); // RC time constant

    let flux: Flx = Ind(2.0).times_current(Cur(3.0)); // Φ = LI
    assert!(approx(flux.clone().raw(), 6.0));
    assert!(approx(flux.clone().per_time(Tim(2.0)).raw(), 3.0)); // Faraday: V = dΦ/dt
    assert!(approx(flux.per_area(Ar(3.0)).raw(), 2.0)); // B = Φ/A
    assert!(approx(Fld(2.0).times_area(Ar(3.0)).raw(), 6.0));
    assert!(approx(EFd(5.0).times_length(Len(2.0)).raw(), 10.0)); // V = E·d
    assert!(approx(EFd(5.0).times_charge(Chg(2.0)).raw(), 10.0)); // F = qE
}

#[test]
fn thermal_photometric_chemical() {
    // thermal — temperature is absolute, so the energy relation reads ΔT as a carrier interval.
    assert!(approx(Hcp(5.0).times_temperature(Tmp(2.0)).raw(), 10.0));
    assert!(approx(Eng(10.0).per_temperature(Tmp(2.0)).raw(), 5.0));
    assert!(approx(Tmp::from_celsius(0.0).kelvin(), 273.15));
    assert!(approx(Tmp::from_fahrenheit(32.0).celsius(), 0.0));
    assert!(approx(Tmp::absolute_zero().kelvin(), 0.0));

    // photometry
    let flux: LFl = Lum(4.0).times_solid_angle(Sld(2.0));
    assert!(approx(flux.clone().raw(), 8.0));
    assert!(approx(flux.clone().per_solid_angle(Sld(2.0)).raw(), 4.0));
    assert!(approx(flux.per_area(Ar(2.0)).raw(), 4.0)); // illuminance
    assert!(approx(Ilm(4.0).times_area(Ar(2.0)).raw(), 8.0));
    assert!(approx(
        Sld::full_sphere().steradians(),
        4.0 * std::f64::consts::PI
    ));

    // chemistry
    assert!(approx(Amt(6.0).per_time(Tim(2.0)).raw(), 3.0)); // catalytic activity
    assert!(approx(Cat(3.0).times_time(Tim(2.0)).raw(), 6.0));
    assert!(approx(Amt(6.0).per_volume(Vol(2.0)).raw(), 3.0)); // concentration
    assert!(approx(Cnc(3.0).times_volume(Vol(2.0)).raw(), 6.0));
    assert!(approx(Mlr(2.0).times_amount(Amt(3.0)).raw(), 6.0)); // mass
    assert!(approx(Mas(6.0).per_amount(Amt(3.0)).raw(), 2.0)); // molar mass

    // thermal conductivity is a tracked marker with no cross-dimensional product.
    assert_eq!(Tcd(0.5).raw(), 0.5);
    assert_eq!(Tcd::DIMENSION, Dimension::ThermalConductivity);
}

#[test]
fn derived_mechanics() {
    // mass flow rate
    let m_dot: Mfr = Mas(10.0).per_time(Tim(2.0));
    assert!(approx(m_dot.clone().raw(), 5.0));
    assert!(approx(m_dot.clone().times_time(Tim(2.0)).raw(), 10.0));
    assert!(approx(m_dot.clone().times_speed(Spd(3.0)).raw(), 15.0)); // thrust F = ṁv
    assert!(approx(m_dot.per_density(Den(2.0)).raw(), 2.5)); // → volumetric flow

    // kinematic viscosity ν = μ/ρ
    assert!(approx(Kvs(2.0).times_density(Den(3.0)).raw(), 6.0)); // → dynamic viscosity
    assert!(approx(Kvs(6.0).per_length(Len(2.0)).raw(), 3.0)); // → speed

    // specific energy and energy density
    let e_spec: SpE = Eng(20.0).per_mass(Mas(4.0));
    assert!(approx(e_spec.clone().raw(), 5.0));
    assert!(approx(e_spec.times_mass(Mas(4.0)).raw(), 20.0));
    assert!(approx(EnD(5.0).times_volume(Vol(2.0)).raw(), 10.0));

    // action — the Planck relation E = hf, and action / time = energy
    assert!(approx(Act(6.0).times_frequency(Frq(2.0)).raw(), 12.0));
    assert!(approx(Act(12.0).per_time(Tim(3.0)).raw(), 4.0));
}

#[test]
fn fields_and_material_constants() {
    // current density J = I/A
    let j: CrD = Cur(12.0).per_area(Ar(3.0));
    assert!(approx(j.clone().raw(), 4.0));
    assert!(approx(j.times_area(Ar(3.0)).raw(), 12.0));

    // magnetizing field H: Ampère's law and B = μH
    assert!(approx(Hfs(5.0).times_length(Len(2.0)).raw(), 10.0)); // → current
    assert!(approx(Hfs(2.0).times_permeability(Pmb(3.0)).raw(), 6.0)); // → flux density

    // permittivity and permeability scale with length to capacitance / inductance
    assert!(approx(Pmt(2.0).times_length(Len(3.0)).raw(), 6.0));
    assert!(approx(Pmb(2.0).times_length(Len(3.0)).raw(), 6.0));

    // resistivity ⇄ conductivity reciprocal pair, with E = ρJ and J = σE
    assert!(approx(Rsv(4.0).conductivity().raw(), 0.25));
    assert!(approx(Cdv(0.25).resistivity().raw(), 4.0));
    assert!(approx(Rsv(3.0).times_current_density(CrD(2.0)).raw(), 6.0)); // → electric field
    assert!(approx(Cdv(2.0).times_electric_field(EFd(3.0)).raw(), 6.0)); // → current density
}

#[test]
fn intensive_thermal_chemical_photometric() {
    // specific heat capacity
    let c: Shc = Hcp(12.0).per_mass(Mas(3.0));
    assert!(approx(c.clone().raw(), 4.0));
    assert!(approx(c.clone().times_mass(Mas(3.0)).raw(), 12.0)); // → heat capacity
    assert!(approx(c.times_temperature(Tmp(2.0)).raw(), 8.0)); // → specific energy (ΔT interval)

    // thermal resistance ΔT = R·Q̇, and heat flux q = P/A
    assert!(approx(Thr(2.0).times_power(Pwr(3.0)).raw(), 6.0));
    let q: Hfx = Pwr(12.0).per_area(Ar(4.0));
    assert!(approx(q.clone().raw(), 3.0));
    assert!(approx(q.times_area(Ar(4.0)).raw(), 12.0));

    // molar energy / volume / molality
    assert!(approx(Eng(20.0).per_amount(Amt(4.0)).raw(), 5.0)); // molar energy
    assert!(approx(MlE(5.0).times_amount(Amt(4.0)).raw(), 20.0));
    let vm: MlV = Vol(12.0).per_amount(Amt(3.0));
    assert!(approx(vm.clone().raw(), 4.0));
    assert!(approx(vm.clone().times_amount(Amt(3.0)).raw(), 12.0));
    assert!(approx(vm.concentration().raw(), 0.25)); // 1 / molar volume
    let b: Mlt = Amt(6.0).per_mass(Mas(2.0)); // molality
    assert!(approx(b.clone().raw(), 3.0));
    assert!(approx(b.times_mass(Mas(2.0)).raw(), 6.0));

    // luminance L = I/A
    let lum: Lmn = Lum(8.0).per_area(Ar(2.0));
    assert!(approx(lum.clone().raw(), 4.0));
    assert!(approx(lum.times_area(Ar(2.0)).raw(), 8.0));
}

#[test]
fn radiological_doses() {
    // becquerel: activity × time = expected decay count (dimensionless)
    assert!(approx(Rdy(100.0).times_time(Tim(3.0)).raw(), 300.0));

    // gray: absorbed dose × mass = deposited energy
    assert!(approx(AbD(2.0).times_mass(Mas(5.0)).raw(), 10.0));

    // sievert: weight a gray by a dimensionless radiation factor, then read energy back
    let h: DsE = AbD(2.0).weighted_by(20.0); // e.g. wᵣ = 20 for alpha particles
    assert!(approx(h.clone().raw(), 40.0));
    assert!(approx(h.times_mass(Mas(0.5)).raw(), 20.0));

    // gray and sievert share a base signature but are distinct names, as is specific energy.
    assert!(Dimension::AbsorbedDose.commensurable(Dimension::DoseEquivalent));
    assert!(Dimension::AbsorbedDose.commensurable(Dimension::SpecificEnergy));
    assert!(!Dimension::AbsorbedDose.same(Dimension::DoseEquivalent));
}

#[test]
fn dimensionless_ratios_compose() {
    let half = Rat(0.5);
    let third = Rat(3.0).recip();
    assert!(approx(third.clone().raw(), 1.0 / 3.0));
    assert!(approx(half.clone().times(Rat(4.0)).raw(), 2.0));
    assert!(approx(Rat(6.0).per(Rat(2.0)).raw(), 3.0));
    assert!(approx(Rat::unit().raw(), 1.0));
    // dimensionless still scales a dimensioned quantity through LinearQuantity::scale.
    assert!(approx(Len(10.0).scale(half.raw()).raw(), 5.0));
}

#[test]
fn type_level_markers_bridge_to_runtime() {
    // each marker maps to its runtime dimension and inherits the matching exponents.
    assert_eq!(<dim::Force as Dimensioned>::DIMENSION, Dimension::Force);
    assert_eq!(
        <dim::Force as Dimensioned>::EXPONENTS,
        Dimension::Force.exponents()
    );
    assert_eq!(<dim::Length as Dimensioned>::DIMENSION, Dimension::Length);
    assert_eq!(
        <dim::Voltage as Dimensioned>::EXPONENTS,
        [2, 1, -3, -1, 0, 0, 0]
    );
    assert_eq!(
        <dim::Dimensionless as Dimensioned>::DIMENSION,
        Dimension::Dimensionless
    );
    // a marker is a zero-sized tag.
    assert_eq!(core::mem::size_of::<dim::Energy>(), 0);
}

#[test]
fn angle_conversion_and_normalization() {
    let half_turn = Ang::from_degrees(180.0);
    assert!(approx(half_turn.radians(), std::f64::consts::PI));
    assert!(approx(half_turn.degrees(), 180.0));
    assert!(approx(
        Ang::from_degrees(450.0).normalized_unsigned().degrees(),
        90.0
    ));
    assert!(approx(
        Ang::from_degrees(270.0).normalized_signed().degrees(),
        -90.0
    ));
}

#[test]
fn unit_metadata() {
    assert_eq!(Len::UNIT_NAME, "m");
    assert_eq!(Frc::UNIT_NAME, "N");
    assert_eq!(Spd::DIMENSION, Dimension::Speed);
    assert_eq!(Cur::DIMENSION, Dimension::ElectricCurrent);
    assert!(Dimension::Length.same(Dimension::Length));
    assert!(!Dimension::Length.same(Dimension::Time));
}

#[test]
fn base_dimension_signatures() {
    // [L, M, T, I, Θ, N, J]
    let len: BaseExponents = Dimension::Length.exponents();
    assert_eq!(len, [1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(Dimension::Force.exponents(), [1, 1, -2, 0, 0, 0, 0]);
    assert_eq!(Dimension::Voltage.exponents(), [2, 1, -3, -1, 0, 0, 0]);

    // energy and torque share a base signature but are distinct names.
    assert!(Dimension::Energy.commensurable(Dimension::Torque));
    assert!(!Dimension::Energy.same(Dimension::Torque));

    // the same pattern recurs among the new derived names.
    assert!(Dimension::EnergyDensity.commensurable(Dimension::Pressure));
    assert!(!Dimension::EnergyDensity.same(Dimension::Pressure));
    assert!(Dimension::Action.commensurable(Dimension::AngularMomentum));
    assert!(!Dimension::Action.same(Dimension::AngularMomentum));
    assert!(Dimension::Luminance.commensurable(Dimension::Illuminance));
    assert!(!Dimension::Luminance.same(Dimension::Illuminance));

    // a couple of the new derived signatures spelled out explicitly.
    assert_eq!(Dimension::Permittivity.exponents(), [-3, -1, 4, 2, 0, 0, 0]);
    assert_eq!(Dimension::MolarEnergy.exponents(), [2, 1, -2, 0, 0, -1, 0]);
    assert_eq!(Dimension::AbsorbedDose.exponents(), [2, 0, -2, 0, 0, 0, 0]);

    // frequency, angular velocity, and radioactivity are all commensurable (`[T⁻¹]`).
    assert!(Dimension::Frequency.commensurable(Dimension::AngularVelocity));
    assert!(Dimension::Frequency.commensurable(Dimension::Radioactivity));

    // angles read as dimensionless in SI even though they are tracked names.
    assert!(Dimension::Angle.is_dimensionless());
    assert!(Dimension::SolidAngle.is_dimensionless());
    assert!(!Dimension::Length.is_dimensionless());
    assert!(!Dimension::Length.commensurable(Dimension::Time));
}
